"""Record a run: full per-frame evaluation to CSV, plus sampled frames.

Writes three things into one directory:

``frames.csv``
    Every frame's complete evaluation - both scores, the confidence, every
    metric's raw/normalised/weight/reliability/agreement/timing, all measured
    image statistics, the readiness fields, and the focus actuator position when
    an external system supplies one.
``frames/``
    Every ``--save-every``-th frame as **PNG**.  Lossless on purpose: JPEG
    compression removes exactly the high-frequency content the metrics measure,
    so a re-analysis of JPEG frames would not reproduce the recorded numbers.
``manifest.json``
    Environment, configuration, and the run summary - so a recording can be
    interpreted months later without guessing what produced it.

Image writing happens on a background thread.  Encoding a PNG takes long enough
to disturb the frame timing if done inline, which would corrupt the very latency
figures the recording is meant to capture.

Examples::

    python3 tools/collect_dataset.py --backend gphoto2 --duration 60 \\
        --save-every 10 --out data/run1

    python3 tools/collect_dataset.py --backend gphoto2 --duration 60 --scene \\
        --out data/run1 --motor-file /run/focus_position
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
import queue
import signal
import sys
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import (  # noqa: E402
    ROI,
    SceneEvaluator,
    SharpnessEvaluator,
    __version__,
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


class FrameWriter:
    """Writes sampled frames to disk on a background thread.

    Inline PNG encoding costs milliseconds per frame, which would land inside
    the measured inter-frame interval and make the recorded timing describe the
    recorder rather than the pipeline.
    """

    def __init__(self, directory: Path, max_pending: int = 64) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(max_pending)
        self._thread = threading.Thread(target=self._run, name="frame-writer", daemon=True)
        self.written = 0
        self.dropped = 0
        self._thread.start()

    def _run(self) -> None:
        import cv2

        while True:
            item = self._queue.get()
            if item is None:
                return
            name, image = item
            try:
                cv2.imwrite(str(self.directory / name), image)
                self.written += 1
            except Exception as exc:  # noqa: BLE001 - a bad write must not stop the run
                logger.error("could not write %s: %s", name, exc)

    def submit(self, name: str, image: np.ndarray) -> bool:
        """Queue a frame. Returns False if the queue is full (frame skipped)."""
        try:
            # A copy is required: the capture backend may reuse its buffer.
            self._queue.put_nowait((name, image.copy()))
            return True
        except queue.Full:
            self.dropped += 1
            logger.warning("frame-writer queue full; skipped %s", name)
            return False

    def close(self, timeout: float = 30.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("frame writer did not finish within %.0fs", timeout)


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


def environment() -> dict[str, Any]:
    import cv2

    info: dict[str, Any] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    model = Path("/proc/device-tree/model")
    if model.exists():
        try:
            info["device"] = model.read_text(errors="ignore").strip("\x00 \n")
        except OSError:
            pass
    return info


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

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    handlers.append(logging.FileHandler(out_dir / "collect.log", encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
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

    writer = FrameWriter(out_dir / "frames") if args.save_every > 0 else None
    csv_path = out_dir / "frames.csv"
    logger.info("source: %s", source.description)
    logger.info("writing %s", csv_path)
    if args.duration > 0:
        logger.info("recording for %.0f s", args.duration)

    written = 0
    csv_writer: csv.DictWriter | None = None
    intervals: list[float] = []
    started = time.perf_counter()
    previous = started
    try:
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            while not _stop_requested:
                if args.duration > 0 and time.perf_counter() - started >= args.duration:
                    break
                if args.frames > 0 and written >= args.frames:
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
                    row = result.as_row()
                    row["subject_label"] = scene_result.subject_label
                    subject = scene_result.subject
                    row["subject_x"] = "" if subject is None else subject.center[0]
                    row["subject_y"] = "" if subject is None else subject.center[1]
                    row["subject_map_score"] = "" if subject is None else subject.map_score
                    row["subject_score"] = (
                        "" if scene_result.subject_result is None
                        else scene_result.subject_result.filtered_score
                    )
                    row["subject_confidence"] = (
                        "" if scene_result.subject_result is None
                        else scene_result.subject_result.confidence
                    )
                    row["decision_margin"] = scene_result.separation()
                    row["region_count"] = len(scene_result.regions)
                    row["map_valid_fraction"] = scene_result.focus_map.valid_fraction
                    row["map_relative_spread"] = scene_result.focus_map.relative_spread
                    row["scene_time_s"] = scene_result.scene_time_s
                    row["total_time_s"] = scene_result.total_time_s
                else:
                    result = evaluator.evaluate(frame, roi=roi, motor_position=motor)
                    row = result.as_row()

                now = time.perf_counter()
                row["wall_time"] = round(now - started, 6)
                row["interval_s"] = round(now - previous, 6)
                intervals.append(now - previous)
                previous = now
                row["source"] = frame.source
                row["image_file"] = ""

                if writer is not None and written % args.save_every == 0:
                    name = f"frame_{frame.index:06d}.png"
                    if writer.submit(name, frame.data):
                        row["image_file"] = name

                if csv_writer is None:
                    csv_writer = csv.DictWriter(handle, fieldnames=list(row))
                    csv_writer.writeheader()
                csv_writer.writerow(row)
                written += 1
                if written % 100 == 0:
                    handle.flush()
                    logger.info("%d frames, %.1f s elapsed", written, now - started)
    finally:
        source.close()
        if writer is not None:
            writer.close()

    elapsed = time.perf_counter() - started
    summary: dict[str, Any] = {
        "frames": written,
        "duration_s": round(elapsed, 3),
        "mean_fps": round(written / elapsed, 3) if elapsed > 0 else 0.0,
        "images_saved": 0 if writer is None else writer.written,
        "images_skipped": 0 if writer is None else writer.dropped,
        "capture_dropped_stale": int(getattr(source, "dropped", 0)),
    }
    if intervals:
        arr = np.array(intervals[1:]) if len(intervals) > 1 else np.array(intervals)
        summary["interval_ms_mean"] = round(float(arr.mean()) * 1e3, 3)
        summary["interval_ms_p95"] = round(float(np.percentile(arr, 95)) * 1e3, 3)
        summary["interval_ms_max"] = round(float(arr.max()) * 1e3, 3)

    manifest = {
        "environment": environment(),
        "source": source.description,
        "note": args.note,
        "scene_mode": args.scene,
        "roi": None if roi is None else [roi.x, roi.y, roi.width, roi.height],
        "save_every": args.save_every,
        "config": {
            "pipeline": asdict(config.pipeline),
            "metrics_enabled": list(config.metrics.enabled),
            "normalization": asdict(config.normalization),
            "temporal": asdict(config.temporal),
            "focus_map": asdict(config.focus_map),
        },
        "summary": summary,
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, default=str), encoding="utf-8"
    )

    logger.info("wrote %d rows to %s", written, csv_path)
    logger.info("summary: %s", json.dumps(summary))
    print(f"\nrecording written to {out_dir}")
    print(f"  frames.csv     {written} rows")
    if writer is not None:
        print(f"  frames/        {writer.written} PNG files")
    print(f"  manifest.json  environment, config and summary")
    print(f"  collect.log    run log")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
