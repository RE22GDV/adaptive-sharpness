"""Reproducible performance measurement.

Separates the three costs the project cares about:

* **capture latency** - time spent inside the capture backend waiting for the
  camera (for the GH6 this dominates, and it is *not* something the algorithm
  can improve);
* **processing time** - time inside the evaluator, broken down per metric;
* **end-to-end rate** - what the consumer actually sees, which with threaded
  capture is bounded by ``max(capture, processing)`` rather than their sum.

Examples::

    python3 tools/benchmark.py --backend synthetic --frames 300
    python3 tools/benchmark.py --backend gphoto2 --frames 200 --json data/bench.json
    python3 tools/benchmark.py --backend synthetic --sweep-analysis-width
"""
from __future__ import annotations

import argparse
import json
import logging
import platform
import statistics
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_sharpness.capture import CaptureError, open_source  # noqa: E402
from adaptive_sharpness import (  # noqa: E402
    ROI, SceneEvaluator, SharpnessConfig, SharpnessEvaluator, load_config,
    load_default_config,
)

logger = logging.getLogger("benchmark")


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q / 100.0 * (len(ordered) - 1)))))
    return ordered[index]


@dataclass
class Timings:
    """Collected per-frame timings, in seconds."""

    capture: list[float] = field(default_factory=list)
    processing: list[float] = field(default_factory=list)
    wall: list[float] = field(default_factory=list)
    # Age of the frame at the instant its result becomes available: this is the
    # end-to-end latency the autofocus loop actually reacts to.
    age: list[float] = field(default_factory=list)
    per_metric: dict[str, list[float]] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        def stats(values: list[float], name: str) -> dict[str, float]:
            if not values:
                return {}
            return {
                f"{name}_ms_mean": round(statistics.mean(values) * 1e3, 3),
                f"{name}_ms_median": round(statistics.median(values) * 1e3, 3),
                f"{name}_ms_p95": round(percentile(values, 95) * 1e3, 3),
                f"{name}_ms_max": round(max(values) * 1e3, 3),
            }

        out: dict[str, Any] = {"frames": len(self.wall)}
        out.update(stats(self.capture, "capture"))
        out.update(stats(self.processing, "processing"))
        out.update(stats(self.wall, "wall"))
        out.update(stats(self.age, "age"))
        if self.wall:
            mean_wall = statistics.mean(self.wall)
            out["fps_mean"] = round(1.0 / mean_wall, 2) if mean_wall > 0 else 0.0
            out["fps_p05"] = (
                round(1.0 / percentile(self.wall, 95), 2)
                if percentile(self.wall, 95) > 0 else 0.0
            )
        out["per_metric_ms_mean"] = {
            name: round(statistics.mean(times) * 1e3, 4)
            for name, times in sorted(self.per_metric.items())
            if times
        }
        return out


def run_benchmark(
    config: SharpnessConfig,
    frames: int,
    warmup: int,
    roi: ROI | None,
    extra_load_s: float = 0.0,
    scene: bool = False,
) -> dict[str, Any]:
    """Capture and evaluate ``frames`` frames, returning the timing summary."""
    evaluator = SceneEvaluator(config) if scene else SharpnessEvaluator(config)
    timings = Timings()
    source = open_source(config.capture)
    description = source.description

    scores: list[float] = []
    confidences: list[float] = []
    scene_stage: list[float] = []
    subject_labels: list[str] = []
    region_counts: list[int] = []
    dropped = 0
    try:
        previous = time.perf_counter()
        collected = 0
        attempts = 0
        max_attempts = (frames + warmup) * 3 + 30
        while collected < frames + warmup and attempts < max_attempts:
            attempts += 1
            frame = source.read()
            if frame is None:
                logger.warning("source exhausted after %d frames", collected)
                break
            if scene:
                scene_result = evaluator.evaluate(frame)
                result = scene_result.frame_result
                scene_stage.append(scene_result.scene_time_s)
                subject_labels.append(scene_result.subject_label)
                region_counts.append(len(scene_result.regions))
                # Charge the whole scene call, not just the frame ensemble.
                result = replace(result, processing_time_s=scene_result.total_time_s)
            else:
                result = evaluator.evaluate(frame, roi=roi)
            if extra_load_s > 0.0:
                # Stand-in for a heavier downstream consumer, used to test
                # whether threaded capture keeps the analysed frame fresher.
                time.sleep(extra_load_s)
            now = time.perf_counter()
            collected += 1
            if collected <= warmup:
                previous = now
                continue
            timings.capture.append(frame.capture_latency_s)
            timings.processing.append(result.processing_time_s)
            timings.wall.append(now - previous)
            timings.age.append(now - frame.timestamp)
            previous = now
            scores.append(result.score)
            confidences.append(result.confidence)
            for sample in result.metrics:
                timings.per_metric.setdefault(sample.name, []).append(sample.compute_time_s)
        dropped = int(getattr(source, "dropped", 0))
    finally:
        source.close()

    summary = timings.summary()
    summary["source"] = description
    summary["dropped_frames"] = dropped
    summary["analysis_width"] = config.pipeline.analysis_width
    summary["metrics"] = list(evaluator.metric_names)
    summary["roi"] = None if roi is None else [roi.x, roi.y, roi.width, roi.height]
    if scores:
        summary["score_mean"] = round(statistics.mean(scores), 4)
        summary["confidence_mean"] = round(statistics.mean(confidences), 4)
    if scene_stage:
        summary["scene_stage_ms_mean"] = round(statistics.mean(scene_stage) * 1e3, 3)
        summary["regions_mean"] = round(statistics.mean(region_counts), 2)
        top = max(set(subject_labels), key=subject_labels.count)
        summary["subject_mode"] = f"{top} ({subject_labels.count(top)}/{len(subject_labels)})"
    return summary


def environment() -> dict[str, Any]:
    import cv2
    import numpy as np

    info: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "cv2_threads": cv2.getNumThreads(),
    }
    model = Path("/proc/device-tree/model")
    if model.exists():
        try:
            info["device"] = model.read_text(errors="ignore").strip("\x00 \n")
        except OSError:
            pass
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:  # noqa: BLE001
        pass
    return info


def print_summary(title: str, summary: dict[str, Any]) -> None:
    print(f"\n--- {title} ---")
    print(f"  source          : {summary.get('source')}")
    print(f"  frames measured : {summary.get('frames')}")
    print(f"  analysis width  : {summary.get('analysis_width')} px")
    for key in ("capture", "processing", "wall", "age"):
        mean = summary.get(f"{key}_ms_mean")
        if mean is None:
            continue
        print(
            f"  {key:15s} : mean {mean:7.2f} ms   median "
            f"{summary.get(f'{key}_ms_median', 0):7.2f}   p95 "
            f"{summary.get(f'{key}_ms_p95', 0):7.2f}   max "
            f"{summary.get(f'{key}_ms_max', 0):7.2f}"
        )
    print(
        f"  throughput      : {summary.get('fps_mean', 0):.2f} fps mean, "
        f"{summary.get('fps_p05', 0):.2f} fps at the 5th percentile"
    )
    print(f"  dropped (stale) : {summary.get('dropped_frames', 0)}")
    if "scene_stage_ms_mean" in summary:
        print(f"  scene stage     : {summary['scene_stage_ms_mean']:.2f} ms "
              f"(map + regions), {summary.get('regions_mean')} regions/frame")
        print(f"  subject         : {summary.get('subject_mode')}")
    per_metric = summary.get("per_metric_ms_mean") or {}
    if per_metric:
        print("  per-metric mean :")
        total = sum(per_metric.values())
        for name, value in sorted(per_metric.items(), key=lambda kv: -kv[1]):
            share = 100.0 * value / total if total else 0.0
            print(f"      {name:12s} {value:7.3f} ms  ({share:4.1f}%)")
        print(f"      {'sum':12s} {total:7.3f} ms")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML config; defaults to the packaged one")
    parser.add_argument(
        "--backend", default=None,
        choices=["auto", "v4l2", "gphoto2", "file", "synthetic"],
    )
    parser.add_argument("--path", default=None, help="path for the file backend")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--analysis-width", type=int, default=None)
    parser.add_argument("--roi", default=None, help="x,y,w,h in source pixels")
    parser.add_argument("--no-threaded", action="store_true", help="disable threaded capture")
    parser.add_argument(
        "--sweep-analysis-width", action="store_true",
        help="repeat the run at several analysis widths",
    )
    parser.add_argument(
        "--compare-threading", action="store_true",
        help="run once threaded and once serial, to show what threading buys",
    )
    parser.add_argument(
        "--scene", action="store_true",
        help="benchmark the scene stage (focus map + regions + subject) too",
    )
    parser.add_argument(
        "--extra-load-ms", type=float, default=0.0,
        help="simulate a heavier consumer by sleeping this long per frame",
    )
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config) if args.config else load_default_config()
    capture_overrides: dict[str, Any] = {}
    if args.backend:
        capture_overrides["backend"] = args.backend
    if args.path:
        capture_overrides["path"] = args.path
    if args.no_threaded:
        capture_overrides["threaded"] = False
    # Offline sources are finite; loop them so --frames is what actually decides
    # the run length instead of the clip being silently cut short.
    if (args.backend or config.capture.backend) in ("synthetic", "file"):
        capture_overrides["loop"] = True
    if capture_overrides:
        config = config.with_overrides(capture=capture_overrides)
    if args.analysis_width:
        config = config.with_overrides(pipeline={"analysis_width": args.analysis_width})

    roi = None
    if args.roi:
        try:
            x, y, w, h = (int(v) for v in args.roi.split(","))
        except ValueError:
            parser.error("--roi must be four integers: x,y,w,h")
        roi = ROI(x, y, w, h)

    load = max(0.0, args.extra_load_ms) / 1000.0
    env = environment()
    print("=" * 72)
    print("SHARPNESS PIPELINE BENCHMARK")
    print("=" * 72)
    for key, value in env.items():
        print(f"  {key:15s} : {value}")

    results: dict[str, Any] = {"environment": env, "runs": {}}

    try:
        if args.sweep_analysis_width:
            for width in (160, 240, 320, 480, 640):
                cfg = config.with_overrides(pipeline={"analysis_width": width})
                summary = run_benchmark(cfg, args.frames, args.warmup, roi, load, args.scene)
                results["runs"][f"width_{width}"] = summary
                print_summary(f"analysis width {width}", summary)
        elif args.compare_threading:
            for threaded in (True, False):
                cfg = config.with_overrides(capture={"threaded": threaded})
                summary = run_benchmark(cfg, args.frames, args.warmup, roi, load, args.scene)
                label = "threaded" if threaded else "serial"
                results["runs"][label] = summary
                print_summary(f"capture: {label}", summary)
        else:
            summary = run_benchmark(config, args.frames, args.warmup, roi, load, args.scene)
            results["runs"]["default"] = summary
            print_summary("run", summary)
    except CaptureError as exc:
        print(f"\nCAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
