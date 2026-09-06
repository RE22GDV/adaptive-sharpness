"""Measure the distribution of the image statistics on real frames.

The reference constants in ``AnalysisConfig`` (``edge_ref_density``,
``noise_ref_sigma``, ``contrast_ref``, ``motion_ref_px``) decide when a frame
counts as "enough structure", "too noisy" and so on.  Guessing them produces a
confidence signal that is systematically wrong for a given camera, so this tool
measures them from actual frames and prints the percentiles to set them from.

    python3 tools/calibrate_stats.py --backend gphoto2 --frames 120
    python3 tools/calibrate_stats.py --backend file --path data/sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys

import numpy as np
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_sharpness.capture import CaptureError, open_source  # noqa: E402
from adaptive_sharpness import ROI, load_config, load_default_config  # noqa: E402
from adaptive_sharpness.analysis import ImageAnalyzer  # noqa: E402
from adaptive_sharpness.preprocess import Preprocessor  # noqa: E402

logger = logging.getLogger("calibrate")

FIELDS = (
    "edge_density",
    "local_contrast",
    "noise_sigma",
    "brightness",
    "clipped_high",
    "clipped_low",
    "motion_px",
    "anisotropy",
)


def quantiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    n = len(ordered)

    def at(q: float) -> float:
        return ordered[min(n - 1, max(0, int(round(q * (n - 1)))))]

    return {
        "min": ordered[0],
        "p05": at(0.05),
        "p25": at(0.25),
        "median": at(0.50),
        "p75": at(0.75),
        "p95": at(0.95),
        "max": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML config; defaults to the packaged one")
    parser.add_argument(
        "--backend", default="gphoto2",
        choices=["auto", "v4l2", "gphoto2", "file", "synthetic"],
    )
    parser.add_argument("--path", default=None)
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--roi", default=None, help="x,y,w,h in source pixels")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    config = load_config(args.config) if args.config else load_default_config()
    overrides: dict[str, Any] = {"backend": args.backend}
    if args.path:
        overrides["path"] = args.path
    config = config.with_overrides(capture=overrides)

    roi = None
    if args.roi:
        x, y, w, h = (int(v) for v in args.roi.split(","))
        roi = ROI(x, y, w, h)

    preprocessor = Preprocessor(config.pipeline)
    analyzer = ImageAnalyzer(config.analysis)
    collected: dict[str, list[float]] = {field: [] for field in FIELDS}

    # The confidence references are just as easy to miscalibrate as the image
    # ones, and a wrong dispersion reference silently destroys the confidence on
    # perfectly good frames, so measure those too.
    from adaptive_sharpness import SharpnessEvaluator  # noqa: PLC0415

    evaluator = SharpnessEvaluator(config)
    confidence_fields = ("confidence", "dispersion", "snr_component", "concordance")
    for name in confidence_fields:
        collected[name] = []

    try:
        source = open_source(config.capture)
    except CaptureError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"source: {source.description}")
    try:
        for index in range(args.frames + args.warmup):
            frame = source.read()
            if frame is None:
                break
            image = preprocessor.prepare(frame.data, roi)
            stats = analyzer.analyze(image.gray)
            result = evaluator.evaluate(frame, roi=roi)
            if index < args.warmup:
                continue
            data = stats.as_dict()
            for field in FIELDS:
                collected[field].append(data[field])

            normalised = np.array(list(result.normalized_metrics.values()))
            median = float(np.median(normalised))
            dispersion = 1.4826 * float(np.median(np.abs(normalised - median)))
            collected["confidence"].append(result.confidence)
            collected["dispersion"].append(dispersion)
            collected["snr_component"].append(
                min(1.0, stats.snr / max(config.ensemble.confidence_snr_ref, 1e-9))
            )
            collected["concordance"].append(
                1.0 - min(1.0, dispersion
                          / max(config.ensemble.confidence_dispersion_ref, 1e-9))
            )
    finally:
        source.close()

    count = len(collected["edge_density"])
    if count == 0:
        print("no frames captured", file=sys.stderr)
        return 1

    print(f"frames analysed: {count}\n")
    header = f"{'field':16s} {'min':>9} {'p05':>9} {'median':>9} {'p95':>9} {'max':>9}"
    print(header)
    print("-" * len(header))
    summary: dict[str, dict[str, float]] = {}
    all_fields = list(FIELDS) + ["confidence", "dispersion", "snr_component",
                                 "concordance"]
    for field in all_fields:
        q = quantiles(collected[field])
        summary[field] = q
        print(
            f"{field:16s} {q['min']:9.4f} {q['p05']:9.4f} {q['median']:9.4f} "
            f"{q['p95']:9.4f} {q['max']:9.4f}"
        )

    print("\nSuggested AnalysisConfig references for this camera and scene:")
    # A frame at the upper end of the observed structure should count as fully
    # sufficient, so anchor on a high percentile rather than the median.
    print(f"  edge_ref_density  = {summary['edge_density']['p75']:.4f}"
          f"   (75th percentile of the observed edge density)")
    print(f"  contrast_ref      = {summary['local_contrast']['p75']:.4f}"
          f"   (75th percentile of the observed RMS contrast)")
    # Noise: the reference is the level at which metrics are badly degraded, so
    # anchor above the routine noise floor.
    print(f"  noise_ref_sigma   = {max(1.0, summary['noise_sigma']['p95'] * 2.0):.2f}"
          f"   (twice the 95th percentile of the measured noise)")
    print(f"  motion_ref_px     = {max(1.0, summary['motion_px']['p95'] * 2.0):.2f}"
          f"   (twice the 95th percentile of the measured motion)")
    # The metrics never agree perfectly on a real scene.  A reference tighter
    # than the routine spread makes the concordance factor - and with it the
    # whole confidence - collapse on frames that are in fact perfectly good.
    print(f"  confidence_dispersion_ref = "
          f"{max(0.05, summary['dispersion']['p95'] * 1.5):.3f}"
          f"   (1.5x the 95th percentile of the observed metric spread)")
    print()
    print(f"  observed confidence: median {summary['confidence']['median']:.3f}, "
          f"p05 {summary['confidence']['p05']:.3f}")
    print(f"  observed concordance factor: median "
          f"{summary['concordance']['median']:.3f}, "
          f"p05 {summary['concordance']['p05']:.3f}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps({"frames": count, "source": source.description, "stats": summary},
                       indent=2),
            encoding="utf-8",
        )
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
