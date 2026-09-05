"""End-to-end smoke test on synthetic data: does the score follow the defocus?

Checks each metric individually as well as the ensemble, because a metric that
is silently non-monotone would still leave the ensemble looking healthy.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharpness import ROI, SharpnessEvaluator, available_metrics  # noqa: E402
from tools.synthetic import defocus, make_scene  # noqa: E402


def is_non_increasing(values: list[float], tolerance: float = 1e-9) -> bool:
    return all(a >= b - tolerance for a, b in zip(values, values[1:]))


def main() -> int:
    print("metrics:", ", ".join(available_metrics()))
    scene = make_scene()
    evaluator = SharpnessEvaluator()

    radii = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    # Two passes: the first warms the running normaliser, the second is scored.
    for _ in range(2):
        for radius in radii:
            evaluator.evaluate(defocus(scene, radius))

    print(f"\n{'radius':>6} {'score':>7} {'inst':>7} {'conf':>6}   raw metrics")
    scores: list[float] = []
    per_metric: dict[str, list[float]] = {name: [] for name in evaluator.metric_names}
    result = None
    for radius in radii:
        result = evaluator.evaluate(defocus(scene, radius))
        scores.append(result.instantaneous_score)
        for name, value in result.raw_metrics.items():
            per_metric[name].append(value)
        raws = "  ".join(f"{k}={v:.4g}" for k, v in result.raw_metrics.items())
        print(
            f"{radius:6.1f} {result.score:7.3f} {result.instantaneous_score:7.3f} "
            f"{result.confidence:6.3f}   {raws}"
        )

    assert result is not None
    print("\nweights on the last (most defocused) frame:")
    for name, weight in result.weights.items():
        print(f"  {name:12s} {weight:.3f}")
    print("\nstats on the last frame:")
    for key, value in result.stats.as_dict().items():
        print(f"  {key:18s} {value:.4f}")

    roi = ROI(160, 90, 320, 180)
    roi_result = SharpnessEvaluator().evaluate(scene, roi=roi)
    print(f"\nROI {roi}: score={roi_result.score:.3f} conf={roi_result.confidence:.3f}")

    print("\nmonotonicity (raw value must not rise as defocus grows):")
    failures = []
    for name, values in per_metric.items():
        ok = is_non_increasing(values)
        print(f"  {name:12s} {'OK' if ok else 'NON-MONOTONE'}")
        if not ok:
            failures.append(name)
    ensemble_ok = is_non_increasing(scores)
    print(f"  {'ensemble':12s} {'OK' if ensemble_ok else 'NON-MONOTONE'}")
    if not ensemble_ok:
        failures.append("ensemble")

    print(f"\nprocessing time last frame: {result.processing_time_s * 1e3:.2f} ms")
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
