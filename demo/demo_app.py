"""Live demonstration window.

Shows the video, the active ROI, the ensemble score, every individual metric
with its current weight, the confidence, a rolling plot of the score, and the
measured frame rate and latency.

This module is deliberately **not** imported by the library: OpenCV's GUI
support (highgui) is an optional runtime dependency, and nothing in
``sharpness`` or ``capture`` needs it.

Controls
--------
q / Esc   quit
r         reset the evaluator state
space     pause
c         cycle the ROI (whole frame -> centre 50% -> centre 25%)
h         toggle the help overlay

Headless machines can pass ``--headless --frames N`` to run the same pipeline
and print a summary instead of opening a window.

Examples::

    python3 demo/demo_app.py --backend gphoto2
    python3 demo/demo_app.py --backend synthetic --headless --frames 120
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_sharpness.capture import CaptureError, open_source  # noqa: E402
from adaptive_sharpness import ROI, SharpnessEvaluator, SharpnessResult, load_config  # noqa: E402

logger = logging.getLogger("demo")

PANEL_WIDTH = 330
PLOT_HEIGHT = 130
BACKGROUND = (24, 24, 28)
TEXT = (232, 232, 236)
MUTED = (150, 150, 158)
ACCENT = (120, 210, 120)
WARN = (90, 170, 250)
BAD = (90, 90, 240)

ROI_MODES: tuple[float | None, ...] = (None, 0.5, 0.25)


def confidence_colour(confidence: float) -> tuple[int, int, int]:
    if confidence >= 0.65:
        return ACCENT
    if confidence >= 0.35:
        return WARN
    return BAD


def draw_plot(
    canvas: np.ndarray,
    values: Sequence[float],
    confidences: Sequence[float],
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    """Draw the rolling score history, coloured by confidence."""
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (40, 40, 46), -1)
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = int(y0 + height - fraction * height)
        cv2.line(canvas, (x0, y), (x0 + width, y), (58, 58, 66), 1)
    if len(values) < 2:
        return
    step = width / float(len(values) - 1)
    for index in range(1, len(values)):
        p0 = (int(x0 + (index - 1) * step),
              int(y0 + height - float(values[index - 1]) * height))
        p1 = (int(x0 + index * step),
              int(y0 + height - float(values[index]) * height))
        cv2.line(canvas, p0, p1, confidence_colour(confidences[index]), 2, cv2.LINE_AA)


def draw_panel(
    result: SharpnessResult,
    height: int,
    fps: float,
    age_ms: float,
    history: Sequence[float],
    confidence_history: Sequence[float],
    dropped: int,
    paused: bool,
) -> np.ndarray:
    import cv2

    panel = np.full((height, PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def text(label: str, y: int, colour=TEXT, scale=0.45, x=12) -> None:  # noqa: ANN001
        cv2.putText(panel, label, (x, y), font, scale, colour, 1, cv2.LINE_AA)

    y = 28
    text("SHARPNESS", y, MUTED, 0.5)
    y += 34
    colour = confidence_colour(result.confidence)
    cv2.putText(panel, f"{result.score:.3f}", (12, y + 12), font, 1.3, colour, 2, cv2.LINE_AA)
    y += 34
    text(f"instant {result.instantaneous_score:.3f}", y, MUTED, 0.4)
    y += 22
    text(f"confidence {result.confidence:.3f}", y, colour, 0.45)
    y += 12

    # Confidence bar.
    cv2.rectangle(panel, (12, y), (PANEL_WIDTH - 12, y + 8), (48, 48, 56), -1)
    filled = int((PANEL_WIDTH - 24) * max(0.0, min(1.0, result.confidence)))
    cv2.rectangle(panel, (12, y), (12 + filled, y + 8), colour, -1)
    y += 30

    if result.score_change_detected:
        text("SCORE JUMPED", y, ACCENT, 0.45)
    y += 22

    text("metric        norm  weight", y, MUTED, 0.4)
    y += 6
    bar_left, bar_right = 12, PANEL_WIDTH - 12
    for sample in result.metrics:
        y += 20
        cv2.rectangle(panel, (bar_left, y - 9), (bar_right, y - 1), (44, 44, 52), -1)
        width = int((bar_right - bar_left) * max(0.0, min(1.0, sample.weight)) * 3.0)
        width = min(width, bar_right - bar_left)
        cv2.rectangle(panel, (bar_left, y - 9), (bar_left + width, y - 1), (70, 90, 130), -1)
        text(f"{sample.name:<12s}{sample.normalized:5.2f} {sample.weight:6.3f}",
             y - 2, TEXT, 0.4)
    y += 26

    stats = result.stats
    for line in (
        f"edges      {stats.edge_density:.4f} ({stats.edge_sufficiency:.2f})",
        f"contrast   {stats.local_contrast:.4f}",
        f"noise      {stats.noise_sigma:.2f} sigma",
        f"brightness {stats.brightness:.3f}",
        f"clipped    {stats.clipped_high:.3f} hi / {stats.clipped_low:.3f} lo",
        f"motion     {stats.motion_px:.2f} px",
    ):
        text(line, y, MUTED, 0.4)
        y += 18

    y += 10
    text(f"{fps:5.1f} fps", y, TEXT, 0.5)
    y += 20
    text(f"process  {result.processing_time_s * 1e3:5.1f} ms", y, MUTED, 0.4)
    y += 18
    text(f"capture  {result.capture_latency_s * 1e3:5.1f} ms", y, MUTED, 0.4)
    y += 18
    text(f"age      {age_ms:5.1f} ms", y, MUTED, 0.4)
    y += 18
    text(f"dropped  {dropped}", y, MUTED, 0.4)
    if paused:
        y += 22
        text("PAUSED", y, WARN, 0.5)

    plot_y = height - PLOT_HEIGHT - 12
    text("score history", plot_y - 8, MUTED, 0.4)
    draw_plot(panel, history, confidence_history, (12, plot_y),
              (PANEL_WIDTH - 24, PLOT_HEIGHT))
    return panel


def compose(frame_bgr: np.ndarray, panel: np.ndarray, roi: ROI | None) -> np.ndarray:
    import cv2

    display = frame_bgr.copy()
    if roi is not None:
        cv2.rectangle(display, (roi.x, roi.y), (roi.x2, roi.y2), ACCENT, 2)
        cv2.putText(display, "ROI", (roi.x + 6, roi.y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)
    if display.shape[0] != panel.shape[0]:
        scale = panel.shape[0] / display.shape[0]
        display = cv2.resize(
            display, (int(display.shape[1] * scale), panel.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
    return np.hstack([display, panel])


def run_headless(
    config, roi_fraction: float | None, frames: int, roi_arg: ROI | None
) -> int:
    """Run the same pipeline without a window and print a summary."""
    evaluator = SharpnessEvaluator(config)
    try:
        source = open_source(config.capture)
    except CaptureError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"source: {source.description}")
    scores: list[float] = []
    confidences: list[float] = []
    processing: list[float] = []
    started = time.perf_counter()
    last: SharpnessResult | None = None
    try:
        for _ in range(frames):
            frame = source.read()
            if frame is None:
                break
            roi = roi_arg
            if roi is None and roi_fraction is not None:
                roi = ROI.centered(frame.data.shape, roi_fraction)
            last = evaluator.evaluate(frame, roi=roi)
            scores.append(last.score)
            confidences.append(last.confidence)
            processing.append(last.processing_time_s)
    finally:
        source.close()

    if not scores:
        print("no frames processed", file=sys.stderr)
        return 1
    elapsed = time.perf_counter() - started
    print(f"frames          : {len(scores)}")
    print(f"throughput      : {len(scores) / elapsed:.2f} fps")
    print(f"score           : mean {np.mean(scores):.3f}  min {np.min(scores):.3f}  "
          f"max {np.max(scores):.3f}")
    print(f"confidence      : mean {np.mean(confidences):.3f}")
    print(f"processing time : mean {np.mean(processing) * 1e3:.2f} ms")
    if last is not None:
        print("final weights   :")
        for name, weight in last.weights.items():
            print(f"    {name:12s} {weight:.3f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML config; defaults to the packaged one")
    parser.add_argument(
        "--backend", default=None,
        choices=["auto", "v4l2", "gphoto2", "file", "synthetic"],
    )
    parser.add_argument("--path", default=None)
    parser.add_argument("--roi", default=None, help="x,y,w,h in source pixels")
    parser.add_argument("--history", type=int, default=240)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=200, help="headless frame count")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config) if args.config else load_default_config()
    overrides: dict[str, Any] = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.path:
        overrides["path"] = args.path
    if args.backend in ("synthetic", "file"):
        overrides["loop"] = True
    if overrides:
        config = config.with_overrides(capture=overrides)

    roi_arg = None
    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.split(","))
        except ValueError:
            parser.error("--roi must be four integers: x,y,w,h")
        roi_arg = ROI(x, y, w, h)

    if args.headless:
        return run_headless(config, None, args.frames, roi_arg)

    try:
        import cv2
    except ImportError:
        print("OpenCV is required for the demo window", file=sys.stderr)
        return 2
    if not hasattr(cv2, "imshow"):
        print(
            "This OpenCV build has no GUI support. Re-run with --headless, or "
            "install a build with highgui.",
            file=sys.stderr,
        )
        return 2

    evaluator = SharpnessEvaluator(config)
    try:
        source = open_source(config.capture)
    except CaptureError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    window = "Adaptive sharpness"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        print(f"could not open a window ({exc}); try --headless", file=sys.stderr)
        source.close()
        return 2

    history: deque[float] = deque(maxlen=args.history)
    confidence_history: deque[float] = deque(maxlen=args.history)
    frame_times: deque[float] = deque(maxlen=30)
    roi_mode = 0
    paused = False
    show_help = True
    last_result: SharpnessResult | None = None
    last_frame: np.ndarray | None = None
    previous = time.perf_counter()

    print(f"source: {source.description}")
    print("keys: q quit | r reset | space pause | c cycle ROI | h help")
    try:
        while True:
            if not paused:
                frame = source.read()
                if frame is None:
                    print("source exhausted")
                    break
                roi = roi_arg
                if roi is None and ROI_MODES[roi_mode] is not None:
                    roi = ROI.centered(frame.data.shape, ROI_MODES[roi_mode])
                result = evaluator.evaluate(frame, roi=roi)
                age_ms = (time.perf_counter() - frame.timestamp) * 1e3
                now = time.perf_counter()
                frame_times.append(now - previous)
                previous = now
                history.append(result.score)
                confidence_history.append(result.confidence)
                last_result, last_frame = result, frame.data

            if last_result is None or last_frame is None:
                continue

            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            height = max(last_frame.shape[0], 640)
            panel = draw_panel(
                last_result, height, fps, age_ms if not paused else 0.0,
                list(history), list(confidence_history),
                int(getattr(source, "dropped", 0)), paused,
            )
            canvas = compose(last_frame, panel, last_result.roi)
            if show_help:
                cv2.putText(
                    canvas, "q quit  r reset  space pause  c ROI  h help",
                    (12, canvas.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, MUTED, 1, cv2.LINE_AA,
                )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                evaluator.reset()
                history.clear()
                confidence_history.clear()
            elif key == ord(" "):
                paused = not paused
            elif key == ord("c"):
                roi_mode = (roi_mode + 1) % len(ROI_MODES)
                evaluator.reset()
                history.clear()
                confidence_history.clear()
            elif key == ord("h"):
                show_help = not show_help
    finally:
        source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
