"""Data-collection mode: log every frame's full evaluation to CSV.

Records, per frame: the frame identifier (and image file, if saved), every raw
and normalised metric, the ensemble score, the weights and reliabilities, the
confidence, the image statistics, the processing time, and the focus actuator
position when an external system supplies one.

The actuator position can be supplied in three ways:

* ``--motor-file PATH`` - a file that an external controller keeps rewriting
  with the current position; it is read once per frame.  This is the loosest
  possible coupling to the servo system, and needs no import from it.
* ``--motor-from-meta`` - take it from ``Frame.meta['motor_position']``, which
  the synthetic sweep source provides.
* nothing - the column is left empty.

Examples::

    python3 tools/collect_dataset.py --backend gphoto2 --frames 300 \\
        --out data/run1.csv --save-frames data/run1_frames

    python3 tools/collect_dataset.py --backend synthetic --frames 200 \\
        --motor-from-meta --out data/sweep.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from capture import CaptureError, open_source  # noqa: E402
from sharpness import ROI, SharpnessEvaluator, load_config  # noqa: E402

logger = logging.getLogger("collect")

_stop_requested = False


def _handle_sigint(signum: int, frame: Any) -> None:  # noqa: ANN401, ARG001
    global _stop_requested
    _stop_requested = True
    logger.warning("interrupt received; finishing the current frame and closing the file")


def read_motor_position(path: Path | None) -> float | None:
    """Read the focus actuator position an external controller publishes."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--backend", default=None,
        choices=["auto", "v4l2", "gphoto2", "file", "synthetic"],
    )
    parser.add_argument("--path", default=None, help="path for the file backend")
    parser.add_argument("--frames", type=int, default=0, help="0 = until interrupted")
    parser.add_argument("--out", type=Path, default=None, help="CSV output path")
    parser.add_argument(
        "--save-frames", type=Path, default=None,
        help="also write each frame as PNG into this directory",
    )
    parser.add_argument("--roi", default=None, help="x,y,w,h in source pixels")
    parser.add_argument(
        "--motor-file", type=Path, default=None,
        help="file an external controller keeps updated with the focus position",
    )
    parser.add_argument(
        "--motor-from-meta", action="store_true",
        help="take the focus position from the frame metadata",
    )
    parser.add_argument("--note", default="", help="free-text note stored in every row")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    signal.signal(signal.SIGINT, _handle_sigint)

    config = load_config(args.config)
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

    out_path = args.out
    if out_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path("data") / f"dataset_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)

    evaluator = SharpnessEvaluator(config)
    try:
        source = open_source(config.capture)
    except CaptureError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    logger.info("source: %s", source.description)
    logger.info("writing %s", out_path)

    written = 0
    writer: csv.DictWriter | None = None
    started_at = time.perf_counter()
    try:
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            while not _stop_requested and (args.frames <= 0 or written < args.frames):
                frame = source.read()
                if frame is None:
                    logger.info("source exhausted")
                    break

                motor = read_motor_position(args.motor_file)
                if motor is None and args.motor_from_meta:
                    value = frame.meta.get("motor_position")
                    motor = float(value) if value is not None else None

                result = evaluator.evaluate(frame, roi=roi, motor_position=motor)
                row = result.as_row()
                row["wall_time"] = round(time.perf_counter() - started_at, 6)
                row["source"] = frame.source
                row["note"] = args.note
                row["image_file"] = ""

                if args.save_frames:
                    import cv2

                    name = f"frame_{frame.index:06d}.png"
                    cv2.imwrite(str(args.save_frames / name), frame.data)
                    row["image_file"] = name

                if writer is None:
                    writer = csv.DictWriter(handle, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                written += 1
                if written % 50 == 0:
                    handle.flush()
                    logger.info("%d frames written", written)
    finally:
        source.close()

    elapsed = time.perf_counter() - started_at
    rate = written / elapsed if elapsed > 0 else 0.0
    print(f"\nwrote {written} rows to {out_path} in {elapsed:.1f}s ({rate:.1f} fps)")
    if args.save_frames:
        print(f"frames saved to {args.save_frames}")
    if written == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
