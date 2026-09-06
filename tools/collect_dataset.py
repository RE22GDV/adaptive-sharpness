"""Record a run: full per-frame evaluation to CSV, plus sampled frames.

Thin driver around :class:`adaptive_sharpness.RunRecorder`, which owns the file
format so that this tool and the UI's record button cannot drift apart.

Writes into one directory:

``frames.csv``     every frame's complete evaluation
``frames/``        every ``--save-every``-th frame as PNG (lossless on purpose)
``manifest.json``  environment, configuration and run summary
``collect.log``    the run log

Examples::

    python3 tools/collect_dataset.py --backend gphoto2 --duration 60 \\
        --save-every 10 --scene --out data/run1

    python3 tools/collect_dataset.py --backend gphoto2 --duration 60 \\
        --out data/run1 --motor-file /run/focus_position
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import (  # noqa: E402
    ROI,
    RunRecorder,
    SceneEvaluator,
    SharpnessEvaluator,
    load_config,
    load_default_config,
)
from adaptive_sharpness.capture import CaptureError, open_source  # noqa: E402

logger = logging.getLogger("collect")

_stop_requested = False


def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ANN401, ARG001
    global _stop_requested
    _stop_requested = True
    logger.warning("interrupt received; finishing the current frame and closing files")


def read_motor_position(path: Path | None) -> float | None:
    """Read the focus actuator position an external controller publishes.

    The value is read *after* the frame was captured, so the pairing is
    approximate - good enough to label a slow stepped sweep, not good enough to
    time a moving actuator. See docs/EXPERIMENTS.md.
    """
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        logger.debug("could not read %s: %s", path, exc)
        return None
    if not text:
        return None
    try:
        return float(text.split()[0])
    except ValueError:
        logger.debug("motor file %s does not contain a number: %r", path, text[:40])
        return None


def scene_extra(scene_result: Any) -> dict[str, Any]:  # noqa: ANN401
    """Scene-level columns appended to each row in --scene mode."""
    subject = scene_result.subject
    return {
        "subject_label": scene_result.subject_label,
        "subject_x": "" if subject is None else subject.center[0],
        "subject_y": "" if subject is None else subject.center[1],
        "subject_map_score": "" if subject is None else subject.map_score,
        "subject_score": (
            "" if scene_result.subject_result is None
            else scene_result.subject_result.filtered_score
        ),
        "subject_confidence": (
            "" if scene_result.subject_result is None
            else scene_result.subject_result.confidence
        ),
        "decision_margin": scene_result.separation(),
        "region_count": len(scene_result.regions),
        "map_valid_fraction": scene_result.focus_map.valid_fraction,
        "map_relative_spread": scene_result.focus_map.relative_spread,
        "scene_time_s": scene_result.scene_time_s,
        "total_time_s": scene_result.total_time_s,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML config; defaults to the packaged one")
    parser.add_argument(
        "--backend", default=None,
        choices=["auto", "v4l2", "gphoto2", "file", "synthetic"],
    )
    parser.add_argument("--path", default=None, help="path for the file backend")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="output directory (default: data/run_<timestamp>)",
    )
    parser.add_argument(
        "--duration", type=float, default=0.0,
        help="record for this many seconds (0 = use --frames)",
    )
    parser.add_argument("--frames", type=int, default=0, help="0 = until interrupted")
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="save every Nth frame as PNG; 0 disables frame saving",
    )
    parser.add_argument("--roi", default=None, help="x,y,w,h in source pixels")
    parser.add_argument(
        "--scene", action="store_true",
        help="also locate the sharpest region and log the subject",
    )
    parser.add_argument(
        "--motor-file", type=Path, default=None,
        help="file an external controller keeps updated with the focus position",
    )
    parser.add_argument("--note", default="", help="free-text note stored in the manifest")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    out_dir = args.out or Path("data") / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(out_dir / "collect.log", encoding="utf-8"),
        ],
        force=True,
    )
    signal.signal(signal.SIGINT, _handle_sigint)

    config = load_config(args.config) if args.config else load_default_config()
    overrides: dict[str, Any] = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.path:
        overrides["path"] = args.path
    if overrides:
        config = config.with_overrides(capture=overrides)

    roi = None
    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.split(","))
        except ValueError:
            parser.error("--roi must be four integers: x,y,w,h")
        roi = ROI(x, y, w, h)

    evaluator: SceneEvaluator | SharpnessEvaluator
    evaluator = SceneEvaluator(config) if args.scene else SharpnessEvaluator(config)

    try:
        source = open_source(config.capture)
    except CaptureError as exc:
        logger.error("capture error: %s", exc)
        return 2

    recorder = RunRecorder(out_dir, config, save_every=args.save_every, note=args.note)
    logger.info("source: %s", source.description)
    if args.duration > 0:
        logger.info("recording for %.0f s", args.duration)

    started = time.perf_counter()
    try:
        recorder.start(source.description)
        while not _stop_requested:
            if args.duration > 0 and time.perf_counter() - started >= args.duration:
                break
            if args.frames > 0 and recorder.frames >= args.frames:
                break

            frame = source.read()
            if frame is None:
                logger.info("source exhausted")
                break

            motor = read_motor_position(args.motor_file)
            if motor is None:
                value = frame.meta.get("motor_position")
                motor = float(value) if value is not None else None

            if args.scene:
                scene_result = evaluator.evaluate(frame, motor_position=motor)
                result = scene_result.frame_result
                assert result is not None
                extra = scene_extra(scene_result)
            else:
                result = evaluator.evaluate(frame, roi=roi, motor_position=motor)
                extra = {}
            extra["source"] = frame.source

            recorder.add(result, frame.data, frame.index, extra=extra)
            if recorder.frames % 100 == 0:
                logger.info("%d frames, %.1f s elapsed", recorder.frames, recorder.elapsed)
    finally:
        source.close()
        summary = recorder.stop()
        summary["capture_dropped_stale"] = int(getattr(source, "dropped", 0))

    logger.info("summary: %s", json.dumps(summary))
    print(f"\nrecording written to {out_dir}")
    print(f"  frames.csv     {summary.get('frames', 0)} rows")
    print(f"  frames/        {summary.get('images_saved', 0)} PNG files")
    print("  manifest.json  environment, config and summary")
    print("  collect.log    run log")
    print(f"\nanalyse it with:\n  python3 tools/analyze_recording.py {out_dir}")
    return 0 if summary.get("frames") else 1


if __name__ == "__main__":
    raise SystemExit(main())
