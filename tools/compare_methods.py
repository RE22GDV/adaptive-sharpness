"""Controlled comparison of sharpness-scoring methods on a focus sweep.

Compares, on identical data and identical normalisation:

  1. each metric on its own;
  2. the plain (unweighted) mean of the metrics;
  3. a fixed weighted mean, using the configured prior weights;
  4. the proposed adaptive ensemble;
  5. the adaptive ensemble without noise compensation;
  6. the adaptive ensemble without motion compensation;
  7. the adaptive ensemble without temporal filtering.

Fair-comparison notes
---------------------
*Normalisation is fitted once over the whole recording and frozen*, so every
method sees exactly the same normalised inputs.  Using the live rolling
normaliser instead would give each method a different scale and make the
comparison meaningless.

*Raw metrics are computed once per frame* and reused by every method, so the
methods differ only in how they combine them - not in what they measured.

Criteria
--------
monotonicity
    Fraction of adjacent steps that move in the direction of the true focus.
    1.0 means a search can hill-climb without ever being misled.
peak_error
    Distance, in sweep steps, between the score's maximum and the true best
    focus position.
discrimination
    (peak - median) / (robust spread), i.e. how far the peak stands out of the
    background. A search needs this to be well above 1.
false_peaks
    Number of local maxima above 70% of the global peak. Extra peaks are where
    a hill-climbing search gets trapped.

Usage::

    python3 tools/compare_methods.py                     # synthetic, all cases
    python3 tools/compare_methods.py --frames-dir data/sweep --best-index 20
    python3 tools/compare_methods.py --json data/comparison.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharpness import ROI, SharpnessConfig, load_config  # noqa: E402
from sharpness.analysis import ImageAnalyzer  # noqa: E402
from sharpness.ensemble import AdaptiveEnsemble  # noqa: E402
from sharpness.metrics import build_metrics  # noqa: E402
from sharpness.normalize import NormalizerBank  # noqa: E402
from sharpness.preprocess import Preprocessor  # noqa: E402
from sharpness.temporal import TemporalFilter  # noqa: E402
from sharpness.types import ImageStats  # noqa: E402
from tools.synthetic import apply_exposure, focus_sweep, make_scene  # noqa: E402

logger = logging.getLogger("compare")


# --------------------------------------------------------------------------
# Data preparation
# --------------------------------------------------------------------------

@dataclass
class SweepData:
    """Raw metrics and statistics for every frame of one sweep."""

    name: str
    raw: dict[str, list[float]] = field(default_factory=dict)
    stats: list[ImageStats] = field(default_factory=list)
    degradations: list[dict[str, float]] = field(default_factory=list)
    true_best_index: int = 0
    positions: list[float] = field(default_factory=list)

    @property
    def length(self) -> int:
        return len(self.stats)


def analyse_frames(
    frames: Iterable[np.ndarray],
    config: SharpnessConfig,
    name: str,
    true_best_index: int,
    positions: Sequence[float] | None = None,
    roi: ROI | None = None,
) -> SweepData:
    """Compute the raw metrics and statistics for a sequence of frames."""
    preprocessor = Preprocessor(config.pipeline)
    analyzer = ImageAnalyzer(config.analysis)
    metrics = build_metrics(config.metrics)

    data = SweepData(name=name, true_best_index=true_best_index)
    data.raw = {metric.name: [] for metric in metrics}
    for index, image in enumerate(frames):
        prepared = preprocessor.prepare(image, roi)
        stats = analyzer.analyze(prepared.gray)
        data.stats.append(stats)
        data.degradations.append(analyzer.degradations(stats))
        for metric in metrics:
            data.raw[metric.name].append(metric.compute(prepared.gray))
    data.positions = list(positions) if positions is not None else list(
        range(data.length)
    )
    return data


def fit_normalizers(data: SweepData, config: SharpnessConfig) -> NormalizerBank:
    """Fit and freeze one normaliser per metric over the whole recording."""
    names = tuple(data.raw)
    # The window must cover the whole recording or the frozen anchors would
    # reflect only its tail.
    normalization = type(config.normalization)(
        window=max(config.normalization.window, data.length),
        warmup=config.normalization.warmup,
        log_compress=config.normalization.log_compress,
        low_percentile=config.normalization.low_percentile,
        high_percentile=config.normalization.high_percentile,
        min_range_ratio=config.normalization.min_range_ratio,
    )
    bank = NormalizerBank(names, normalization)
    bank.fit(data.raw)
    return bank


def normalized_series(data: SweepData, bank: NormalizerBank) -> list[dict[str, float]]:
    """Normalised metric values for every frame, using the frozen scale."""
    return [
        {
            name: bank[name].normalize(values[index], observe=False)
            for name, values in data.raw.items()
        }
        for index in range(data.length)
    ]


# --------------------------------------------------------------------------
# Methods under comparison
# --------------------------------------------------------------------------

ScoreFn = Callable[[SweepData, list[dict[str, float]], SharpnessConfig], list[float]]


def method_single(metric_name: str) -> ScoreFn:
    def score(data, normalized, config):  # noqa: ANN001
        return [row[metric_name] for row in normalized]

    return score


def method_plain_mean(data, normalized, config) -> list[float]:  # noqa: ANN001
    return [float(np.mean(list(row.values()))) for row in normalized]


def method_fixed_weighted(data, normalized, config) -> list[float]:  # noqa: ANN001
    priors = config.metrics.priors
    names = list(config.metrics.enabled)
    weights = np.array([max(0.0, priors.get(n, 1.0)) for n in names], dtype=np.float64)
    weights /= weights.sum()
    return [
        float(np.dot(weights, np.array([row[n] for n in names])))
        for row in normalized
    ]


def _adaptive(
    data: SweepData,
    normalized: list[dict[str, float]],
    config: SharpnessConfig,
    temporal: bool,
) -> list[float]:
    ensemble = AdaptiveEnsemble(
        tuple(config.metrics.enabled), config.ensemble, config.metrics.priors
    )
    filt = TemporalFilter(config.temporal) if temporal else None
    scores: list[float] = []
    for index, row in enumerate(normalized):
        out = ensemble.combine(row, data.stats[index], data.degradations[index])
        value = out.score
        if filt is not None:
            value = filt.update(value, out.confidence).value
        scores.append(value)
    return scores


def method_adaptive(data, normalized, config) -> list[float]:  # noqa: ANN001
    return _adaptive(data, normalized, config, temporal=True)


def method_adaptive_no_temporal(data, normalized, config) -> list[float]:  # noqa: ANN001
    return _adaptive(data, normalized, config, temporal=False)


def method_adaptive_no_noise(data, normalized, config) -> list[float]:  # noqa: ANN001
    ablated = config.with_overrides(ensemble={"use_noise_compensation": False})
    return _adaptive(data, normalized, ablated, temporal=True)


def method_adaptive_no_motion(data, normalized, config) -> list[float]:  # noqa: ANN001
    ablated = config.with_overrides(ensemble={"use_motion_compensation": False})
    return _adaptive(data, normalized, ablated, temporal=True)


def build_methods(config: SharpnessConfig) -> dict[str, ScoreFn]:
    methods: dict[str, ScoreFn] = {
        f"single:{name}": method_single(name) for name in config.metrics.enabled
    }
    methods["plain_mean"] = method_plain_mean
    methods["fixed_weighted"] = method_fixed_weighted
    methods["adaptive"] = method_adaptive
    methods["adaptive_no_noise_comp"] = method_adaptive_no_noise
    methods["adaptive_no_motion_comp"] = method_adaptive_no_motion
    methods["adaptive_no_temporal"] = method_adaptive_no_temporal
    return methods


# --------------------------------------------------------------------------
# Quality criteria
# --------------------------------------------------------------------------

def monotonicity(scores: Sequence[float], best_index: int) -> float:
    """Fraction of steps that move towards the true focus correctly."""
    correct = 0
    total = 0
    for index in range(len(scores) - 1):
        delta = scores[index + 1] - scores[index]
        if index + 1 <= best_index:
            expected_rise = True
        elif index >= best_index:
            expected_rise = False
        else:
            continue
        total += 1
        if abs(delta) < 1e-12:
            # A flat step is not misleading, but it is not informative either.
            correct += 0.5
        elif (delta > 0) == expected_rise:
            correct += 1
    return correct / total if total else 0.0


def peak_error(scores: Sequence[float], best_index: int) -> int:
    return abs(int(np.argmax(scores)) - best_index)


def discrimination(scores: Sequence[float]) -> float:
    """How far the peak stands out of the background, in robust sigmas."""
    values = np.asarray(scores, dtype=np.float64)
    median = float(np.median(values))
    spread = 1.4826 * float(np.median(np.abs(values - median)))
    if spread < 1e-9:
        return 0.0
    return float((values.max() - median) / spread)


def count_false_peaks(scores: Sequence[float], threshold: float = 0.7) -> int:
    """Distinct local maxima above ``threshold`` of the global peak, excluding it.

    Plateaus count once.  Treating every point of a flat maximum as a separate
    peak reported two false peaks even on a perfectly clean unimodal sweep,
    because the three frames nearest focus differ by less than a sub-pixel blur
    radius and are effectively identical.
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size < 3:
        return 0
    peak = float(values.max())
    floor = float(values.min())
    span = peak - floor
    if span < 1e-9:
        return 0
    level = floor + threshold * span
    # Differences far below the measurement resolution must not split one
    # plateau into several "peaks".
    tolerance = max(1e-6, 0.01 * span)
    best = int(np.argmax(values))

    plateaus: list[tuple[int, int]] = []
    index = 1
    while index < values.size - 1:
        if values[index] < level or values[index] < values[index - 1] - tolerance:
            index += 1
            continue
        end = index
        while end + 1 < values.size and abs(values[end + 1] - values[index]) <= tolerance:
            end += 1
        if end + 1 >= values.size or values[end + 1] < values[index] - tolerance:
            plateaus.append((index, end))
        index = end + 1

    return sum(1 for start, end in plateaus if not start <= best <= end)


def hill_climb_distance(scores: Sequence[float], best_index: int) -> float:
    """Mean distance from the true peak after a greedy search from both ends.

    This is the criterion that matters operationally: an autofocus loop never
    sees the whole curve, it takes steps and follows the gradient.  A score with
    good monotonicity but one misleading bump still fails here.
    """
    values = list(scores)
    size = len(values)
    if size < 2:
        return 0.0

    def climb(start: int) -> int:
        position = start
        direction = 1 if start < best_index else -1
        while 0 <= position + direction < size:
            if values[position + direction] > values[position]:
                position += direction
                continue
            # Try the other direction once before stopping, as a real search
            # would when a step fails to improve.
            other = -direction
            if 0 <= position + other < size and values[position + other] > values[position]:
                direction = other
                position += direction
                continue
            break
        return position

    return float(
        np.mean([abs(climb(0) - best_index), abs(climb(size - 1) - best_index)])
    )


def evaluate_series(scores: Sequence[float], best_index: int) -> dict[str, float]:
    return {
        "monotonicity": round(monotonicity(scores, best_index), 4),
        "peak_error": peak_error(scores, best_index),
        "hill_climb": round(hill_climb_distance(scores, best_index), 2),
        "discrimination": round(discrimination(scores), 3),
        "false_peaks": count_false_peaks(scores),
    }


# --------------------------------------------------------------------------
# Experiment drivers
# --------------------------------------------------------------------------

def build_synthetic_conditions(
    config: SharpnessConfig, steps: int, seed: int
) -> dict[str, SweepData]:
    """One clean sweep plus one per disturbance the criteria call for."""
    best_index = (steps - 1) // 2
    rich = make_scene(640, 360, seed=seed)
    sparse = make_scene(640, 360, seed=seed, detail=0.12)
    dim = apply_exposure(rich, 0.22)

    # The disturbance levels are deliberately harsh.  Mild ones leave every
    # method looking identical, which says nothing about which combination rule
    # is better.
    conditions: dict[str, tuple[np.ndarray, dict[str, Any]]] = {
        "clean": (rich, {}),
        "noise_sigma_20": (rich, {"noise_sigma": 20.0}),
        "dark_and_noisy": (dim, {"noise_sigma": 10.0}),
        "low_texture": (sparse, {}),
        "low_texture_noisy": (sparse, {"noise_sigma": 12.0}),
        "exposure_drift": (rich, {"exposure_drift": 0.25}),
        "clipped_highlights": (apply_exposure(rich, 2.2), {}),
        "motion_4px": (rich, {"motion_px": 4.0}),
        "noise_and_motion": (rich, {"noise_sigma": 20.0, "motion_px": 4.0}),
    }
    out: dict[str, SweepData] = {}
    for name, (scene, kwargs) in conditions.items():
        sweep = list(
            focus_sweep(scene, steps=steps, best_position=0.5, max_radius=8.0,
                        seed=seed + 1, **kwargs)
        )
        out[name] = analyse_frames(
            (f.image for f in sweep), config, name, best_index,
            positions=[f.motor_position for f in sweep],
        )
    return out


def run_comparison(
    data: SweepData, config: SharpnessConfig
) -> dict[str, dict[str, Any]]:
    bank = fit_normalizers(data, config)
    normalized = normalized_series(data, bank)
    results: dict[str, dict[str, Any]] = {}
    for name, method in build_methods(config).items():
        started = time.perf_counter()
        scores = method(data, normalized, config)
        elapsed = time.perf_counter() - started
        entry = evaluate_series(scores, data.true_best_index)
        entry["combine_us_per_frame"] = round(elapsed / max(1, data.length) * 1e6, 1)
        entry["scores"] = [round(float(s), 5) for s in scores]
        results[name] = entry
    return results


def repeatability(
    config: SharpnessConfig, steps: int, seeds: Sequence[int]
) -> dict[str, dict[str, float]]:
    """Peak position across independent noisy realisations of the same sweep."""
    per_method: dict[str, list[int]] = {}
    best_index = (steps - 1) // 2
    for seed in seeds:
        scene = make_scene(640, 360, seed=7)
        sweep = list(
            focus_sweep(scene, steps=steps, best_position=0.5, max_radius=8.0,
                        noise_sigma=8.0, motion_px=1.5, seed=seed)
        )
        data = analyse_frames(
            (f.image for f in sweep), config, f"seed{seed}", best_index
        )
        bank = fit_normalizers(data, config)
        normalized = normalized_series(data, bank)
        for name, method in build_methods(config).items():
            scores = method(data, normalized, config)
            per_method.setdefault(name, []).append(int(np.argmax(scores)))
    return {
        name: {
            "peak_indices": peaks,
            "peak_std_steps": round(
                statistics.pstdev(peaks) if len(peaks) > 1 else 0.0, 3
            ),
            "mean_peak_error": round(
                statistics.mean(abs(p - best_index) for p in peaks), 3
            ),
        }
        for name, peaks in per_method.items()
    }


def step_response(config: SharpnessConfig, hold: int = 30) -> dict[str, Any]:
    """How many frames each temporal setting needs to follow a focus step.

    Directly addresses the requirement that the temporal filter must not hide a
    real focus change.
    """
    scene = make_scene(640, 360, seed=7)
    from tools.synthetic import add_noise, defocus

    blurred = defocus(scene, 6.0)
    frames = [add_noise(blurred, 4.0, seed=i) for i in range(hold)]
    frames += [add_noise(scene, 4.0, seed=1000 + i) for i in range(hold)]
    data = analyse_frames(frames, config, "step", hold)
    bank = fit_normalizers(data, config)
    normalized = normalized_series(data, bank)

    instantaneous = _adaptive(data, normalized, config, temporal=False)
    target = instantaneous[-1]
    baseline = instantaneous[hold - 1]
    threshold = baseline + 0.9 * (target - baseline)

    def frames_to_settle(series: Sequence[float]) -> int | None:
        for offset in range(hold):
            if series[hold + offset] >= threshold:
                return offset + 1
        return None

    variants: dict[str, Any] = {
        "no_temporal_filter": instantaneous,
        "gated_filter": _adaptive(data, normalized, config, temporal=True),
    }
    plain_config = config.with_overrides(
        temporal={"gate_sigma": 1e6, "gate_full_sigma": 1e6 + 1, "gate_abs_jump": 2.0}
    )
    variants["plain_ema_same_alpha"] = _adaptive(
        data, normalized, plain_config, temporal=True
    )

    return {
        "threshold": round(float(threshold), 4),
        "settling_frames": {
            name: frames_to_settle(series) for name, series in variants.items()
        },
        "series": {name: [round(float(v), 5) for v in s] for name, s in variants.items()},
    }


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def print_table(title: str, results: dict[str, dict[str, Any]]) -> None:
    print(f"\n{title}")
    header = (
        f"  {'method':26s} {'monotonic':>10} {'peak_err':>9} {'climb':>7} "
        f"{'discrim':>8} {'false_pk':>9} {'us/frame':>9}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for name, entry in results.items():
        print(
            f"  {name:26s} {entry['monotonicity']:10.3f} {entry['peak_error']:9d} "
            f"{entry['hill_climb']:7.1f} {entry['discrimination']:8.2f} "
            f"{entry['false_peaks']:9d} {entry['combine_us_per_frame']:9.1f}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=41)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--frames-dir", type=Path, default=None,
        help="directory of recorded sweep frames (sorted by filename)",
    )
    parser.add_argument(
        "--best-index", type=int, default=None,
        help="index of the in-focus frame in --frames-dir",
    )
    parser.add_argument("--roi", default=None, help="x,y,w,h in source pixels")
    parser.add_argument("--repeat-seeds", type=int, default=5)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    config = load_config(args.config)

    roi = None
    if args.roi:
        x, y, w, h = (int(v) for v in args.roi.split(","))
        roi = ROI(x, y, w, h)

    report: dict[str, Any] = {"config_file": str(args.config) if args.config else None}
    print("=" * 78)
    print("METHOD COMPARISON")
    print("=" * 78)

    if args.frames_dir:
        if args.best_index is None:
            parser.error("--frames-dir requires --best-index")
        import cv2

        from capture.file_source import IMAGE_SUFFIXES

        files = sorted(
            p for p in args.frames_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            parser.error(f"no images found in {args.frames_dir}")
        images = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in files]
        if any(img is None for img in images):
            parser.error("some images could not be decoded")
        print(f"\nrecorded sweep: {len(images)} frames from {args.frames_dir}")
        print(f"true best focus index: {args.best_index}")
        data = analyse_frames(images, config, "recorded", args.best_index, roi=roi)
        results = run_comparison(data, config)
        report["recorded"] = results
        print_table("recorded sweep", results)
    else:
        print(f"\nsynthetic sweeps: {args.steps} steps, disc PSF, "
              f"true best focus at index {(args.steps - 1) // 2}")
        conditions = build_synthetic_conditions(config, args.steps, args.seed)
        report["conditions"] = {}
        for name, data in conditions.items():
            results = run_comparison(data, config)
            report["conditions"][name] = results
            print_table(f"condition: {name}", results)

        print("\n" + "=" * 78)
        print("REPEATABILITY (independent noise realisations of the same sweep)")
        print("=" * 78)
        seeds = list(range(100, 100 + args.repeat_seeds))
        rep = repeatability(config, args.steps, seeds)
        report["repeatability"] = rep
        print(f"  {'method':26s} {'peak_std':>9} {'mean_err':>9}   peaks")
        print("  " + "-" * 62)
        for name, entry in rep.items():
            print(
                f"  {name:26s} {entry['peak_std_steps']:9.3f} "
                f"{entry['mean_peak_error']:9.3f}   {entry['peak_indices']}"
            )

    print("\n" + "=" * 78)
    print("STEP RESPONSE (frames needed to follow a real focus change)")
    print("=" * 78)
    step = step_response(config)
    report["step_response"] = step
    for name, frames in step["settling_frames"].items():
        shown = "not within the window" if frames is None else f"{frames} frames"
        print(f"  {name:26s} {shown}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nJSON written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
