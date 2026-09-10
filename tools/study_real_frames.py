"""Experiments on a real recording, none of which need an external ground truth.

Every validation in this project so far used a synthetic scene generator with a
synthetic defocus model.  That leaves two questions unanswered: whether the
conclusions transfer to real image content, and whether the disc point-spread
function used for the synthetic sweeps resembles real optical defocus at all.

Five experiments, in increasing order of what they assume:

E1  Order dependence
    Raw metrics are stateless, so re-evaluating a shuffled sequence must give
    identical raw values and different normalised ones.  The size of that
    difference measures how much of the reported score is history rather than
    image content.  Assumes nothing.

E2  Held-position groups
    When the operator pauses on each focus position, the recording contains
    natural groups of frames at one constant (unknown) position.  Within-group
    against between-group variance gives a discriminability ratio per metric on
    real data.  Assumes only that the operator paused.

E3  Real-content defocus ladder
    Take the sharpest real frames and apply a *known* defocus.  Real image
    statistics, exact ground truth.  Re-runs the monotonicity and method
    comparison that were previously synthetic-only.

E4  Robustness on real content
    The same ladder under known added noise and exposure change - the
    conditions the adaptive weighting is supposed to react to.

E5  PSF model check
    The edge-width metric estimates blur radius in pixels from physics, not by
    analogy.  Comparing a really-defocused frame with a synthetically-defocused
    one at the *same measured edge width* tests whether the disc PSF resembles
    what the lens actually does.  This is the only experiment here that can
    invalidate the synthetic model.

    python3 tools/study_real_frames.py data/run1 --figures docs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import (  # noqa: E402
    SharpnessConfig,
    SharpnessEvaluator,
    load_config,
    load_default_config,
)
from adaptive_sharpness.analysis import ImageAnalyzer  # noqa: E402
from adaptive_sharpness.ensemble import AdaptiveEnsemble  # noqa: E402
from adaptive_sharpness.metrics import build_metrics  # noqa: E402
from adaptive_sharpness.normalize import NormalizerBank  # noqa: E402
from adaptive_sharpness.preprocess import Preprocessor  # noqa: E402
from adaptive_sharpness.synthetic import add_noise, apply_exposure, defocus  # noqa: E402

METRICS = ["laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class Recording:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        with (directory / "frames.csv").open(newline="", encoding="utf-8") as handle:
            self.rows = list(csv.DictReader(handle))
        if not self.rows:
            raise ValueError("empty recording")

    def column(self, name: str) -> np.ndarray:
        out = []
        for row in self.rows:
            try:
                out.append(float(row.get(name, "")))
            except (TypeError, ValueError):
                out.append(math.nan)
        return np.array(out)

    def saved_frames(self) -> list[tuple[dict[str, str], np.ndarray]]:
        """Rows that have an image on disk, with the image loaded."""
        out = []
        for row in self.rows:
            name = row.get("image_file", "")
            if not name:
                continue
            path = self.directory / "frames" / name
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is not None:
                out.append((row, image))
        return out


def sharpest_frames(
    recording: Recording, images: list[tuple[dict[str, str], np.ndarray]], count: int
) -> list[np.ndarray]:
    """The sharpest saved frames, by consensus rank across all metrics.

    Ranking by a single metric would bias every later experiment towards that
    metric's idea of sharpness, so the median rank over all six is used.
    """
    if not images:
        return []
    ranks = []
    for name in METRICS:
        values = np.array([float(row[f"raw_{name}"]) for row, _ in images])
        order = np.argsort(np.argsort(values))  # 0 = smallest
        ranks.append(order)
    consensus = np.median(np.vstack(ranks), axis=0)
    best = np.argsort(consensus)[::-1][:count]
    return [images[i][1] for i in best]


# ---------------------------------------------------------------------------
# E1 - order dependence
# ---------------------------------------------------------------------------

def experiment_order(
    images: list[np.ndarray], config: SharpnessConfig, trials: int = 12
) -> dict[str, Any]:
    def run(order: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        evaluator = SharpnessEvaluator(config)
        raw = {name: [] for name in METRICS}
        inst, filt = [], []
        for index in order:
            result = evaluator.evaluate(images[index])
            for name in METRICS:
                raw[name].append(result.raw_metrics[name])
            inst.append(result.instantaneous_score)
            filt.append(result.filtered_score)
        return ({k: np.array(v) for k, v in raw.items()},
                np.array(inst), np.array(filt))

    forward = list(range(len(images)))
    raw_f, inst_f, filt_f = run(forward)

    rng = np.random.default_rng(0)
    raw_worst = 0.0
    inst_diffs, filt_diffs, rank_corrs = [], [], []
    for _ in range(trials):
        shuffled = forward.copy()
        rng.shuffle(shuffled)
        raw_s, inst_s, filt_s = run(shuffled)
        back = np.empty(len(images), dtype=int)
        for position, frame in enumerate(shuffled):
            back[frame] = position
        for name in METRICS:
            a, b = raw_f[name], raw_s[name][back]
            raw_worst = max(raw_worst,
                            float(np.max(np.abs(a - b) / np.maximum(np.abs(a), 1e-12))))
        inst_diffs.append(np.abs(inst_f - inst_s[back]))
        filt_diffs.append(np.abs(filt_f - filt_s[back]))
        rank_corrs.append(float(np.corrcoef(
            np.argsort(np.argsort(inst_f)),
            np.argsort(np.argsort(inst_s[back])),
        )[0, 1]))

    inst_all = np.concatenate(inst_diffs)
    filt_all = np.concatenate(filt_diffs)
    return {
        "frames": len(images),
        "trials": trials,
        "raw_worst_relative_difference": raw_worst,
        "instantaneous_mean_abs_diff": float(inst_all.mean()),
        "instantaneous_p95_abs_diff": float(np.percentile(inst_all, 95)),
        "instantaneous_max_abs_diff": float(inst_all.max()),
        "filtered_mean_abs_diff": float(filt_all.mean()),
        "filtered_max_abs_diff": float(filt_all.max()),
        "rank_correlation_mean": float(np.mean(rank_corrs)),
        "rank_correlation_min": float(np.min(rank_corrs)),
    }


# ---------------------------------------------------------------------------
# E2 - held-position groups
# ---------------------------------------------------------------------------

def segment_held_groups(
    signal: np.ndarray, min_length: int = 5, quantile: float = 40.0
) -> list[tuple[int, int]]:
    """Runs of consecutive frames over which a stateless signal barely moves."""
    finite = np.isfinite(signal)
    if finite.sum() < min_length * 2:
        return []
    values = np.log1p(np.maximum(signal, 0.0))
    steps = np.abs(np.diff(values))
    threshold = float(np.percentile(steps[np.isfinite(steps)], quantile))
    groups, start = [], 0
    for index in range(1, len(values)):
        if steps[index - 1] > threshold:
            if index - start >= min_length:
                groups.append((start, index))
            start = index
    if len(values) - start >= min_length:
        groups.append((start, len(values)))
    return groups


def experiment_groups(recording: Recording) -> dict[str, Any]:
    # Segment on a stateless signal so the grouping does not inherit the
    # normaliser's memory.
    groups = segment_held_groups(recording.column("raw_tenengrad"))
    if not groups:
        return {"groups": 0}

    out: dict[str, Any] = {
        "groups": len(groups),
        "median_group_frames": int(np.median([b - a for a, b in groups])),
        "coverage_fraction": float(sum(b - a for a, b in groups) / len(recording.rows)),
        "per_metric": {},
    }
    for name in METRICS:
        values = recording.column(f"raw_{name}")
        within, means = [], []
        for a, b in groups:
            segment = values[a:b]
            segment = segment[np.isfinite(segment)]
            if segment.size < 2:
                continue
            scale = max(abs(float(segment.mean())), 1e-12)
            within.append(float(segment.std()) / scale)
            means.append(float(segment.mean()))
        if not means:
            continue
        means_arr = np.array(means)
        between = float(means_arr.std()) / max(abs(float(means_arr.mean())), 1e-12)
        within_median = float(np.median(within))
        out["per_metric"][name] = {
            "within_group_relative_sd": within_median,
            "between_group_relative_sd": between,
            # How many times better the metric separates positions than it
            # fluctuates at a fixed position. This is discriminability on real
            # data, with no ground truth assumed.
            "discriminability": between / max(within_median, 1e-12),
        }
    return out


# ---------------------------------------------------------------------------
# E3 / E4 - defocus ladder on real content
# ---------------------------------------------------------------------------

def evaluate_ladder(
    frames: list[np.ndarray], config: SharpnessConfig
) -> tuple[dict[str, np.ndarray], list[dict[str, float]], list[Any]]:
    """Raw metrics, normalised values and stats for a sequence of frames."""
    preprocessor = Preprocessor(config.pipeline)
    analyzer = ImageAnalyzer(config.analysis)
    metrics = build_metrics(config.metrics)

    raw: dict[str, list[float]] = {m.name: [] for m in metrics}
    stats, degradations = [], []
    for frame in frames:
        prepared = preprocessor.prepare(frame)
        image_stats = analyzer.analyze(prepared.gray)
        stats.append(image_stats)
        degradations.append(analyzer.degradations(image_stats))
        for metric in metrics:
            raw[metric.name].append(metric.compute(prepared.gray))

    bank = NormalizerBank(
        tuple(raw), type(config.normalization)(
            window=max(config.normalization.window, len(frames)),
            warmup=config.normalization.warmup,
            log_compress=config.normalization.log_compress,
            low_percentile=config.normalization.low_percentile,
            high_percentile=config.normalization.high_percentile,
            min_range_ratio=config.normalization.min_range_ratio,
        )
    )
    bank.fit(raw)
    normalized = [
        {name: bank[name].normalize(raw[name][i], observe=False) for name in raw}
        for i in range(len(frames))
    ]
    return {k: np.array(v) for k, v in raw.items()}, normalized, (stats, degradations)


def combine_methods(
    normalized: list[dict[str, float]],
    stats_and_degradations: tuple[list[Any], list[dict[str, float]]],
    config: SharpnessConfig,
) -> dict[str, np.ndarray]:
    stats, degradations = stats_and_degradations
    names = tuple(config.metrics.enabled)
    priors = np.array([config.metrics.priors[n] for n in names])
    priors = priors / priors.sum()

    plain, fixed, adaptive = [], [], []
    ensemble = AdaptiveEnsemble(names, config.ensemble, config.metrics.priors)
    for index, row in enumerate(normalized):
        vector = np.array([row[n] for n in names])
        plain.append(float(vector.mean()))
        fixed.append(float(np.dot(priors, vector)))
        adaptive.append(
            ensemble.combine(row, stats[index], degradations[index]).score
        )
    out = {"plain_mean": np.array(plain), "fixed_weighted": np.array(fixed),
           "adaptive": np.array(adaptive)}
    for name in names:
        out[f"single:{name}"] = np.array([row[name] for row in normalized])
    return out


def monotonicity(values: np.ndarray) -> float:
    """Fraction of steps that decrease, on a ladder of increasing defocus."""
    if values.size < 2:
        return 0.0
    steps = np.diff(values)
    correct = np.count_nonzero(steps < 0) + 0.5 * np.count_nonzero(steps == 0)
    return float(correct / steps.size)


def experiment_ladder(
    frames: list[np.ndarray],
    config: SharpnessConfig,
    radii: list[float],
    label: str,
    noise_sigma: float = 0.0,
    exposure_gain: float = 1.0,
    seed: int = 3,
) -> dict[str, Any]:
    """Apply a known defocus ladder to real frames and score every method."""
    results: dict[str, list[float]] = {}
    raw_all: dict[str, list[list[float]]] = {name: [] for name in METRICS}

    for base in frames:
        ladder = []
        for index, radius in enumerate(radii):
            image = defocus(base, radius)
            if exposure_gain != 1.0:
                image = apply_exposure(image, exposure_gain)
            if noise_sigma > 0.0:
                image = add_noise(image, noise_sigma, seed=seed + index)
            ladder.append(image)
        raw, normalized, extras = evaluate_ladder(ladder, config)
        for name in METRICS:
            raw_all[name].append(list(raw[name]))
        for method, values in combine_methods(normalized, extras, config).items():
            results.setdefault(method, []).append(monotonicity(values))

    per_metric_raw = {
        name: float(np.mean([monotonicity(np.array(v)) for v in series]))
        for name, series in raw_all.items()
    }
    return {
        "label": label,
        "base_frames": len(frames),
        "radii": radii,
        "noise_sigma": noise_sigma,
        "exposure_gain": exposure_gain,
        "raw_metric_monotonicity": per_metric_raw,
        "method_monotonicity": {
            method: float(np.mean(values)) for method, values in results.items()
        },
        "method_monotonicity_min": {
            method: float(np.min(values)) for method, values in results.items()
        },
    }


# ---------------------------------------------------------------------------
# E5 - PSF model check
# ---------------------------------------------------------------------------

def measured_blur_radius(image: np.ndarray, config: SharpnessConfig) -> float:
    """Blur radius in pixels, from the edge-width metric's physics.

    The metric reports 1 / median edge width; for a Gaussian of width sigma the
    edge width is sigma * sqrt(2*pi), so sigma follows directly.  This is an
    estimate from the image, not from any model of the lens.
    """
    preprocessor = Preprocessor(config.pipeline)
    metrics = {m.name: m for m in build_metrics(config.metrics)}
    value = metrics["edge_width"].compute(preprocessor.prepare(image).gray)
    if value <= 1e-9:
        return float("inf")
    return (1.0 / value) / math.sqrt(2.0 * math.pi)


def content_matches(
    a: np.ndarray, b: np.ndarray, config: SharpnessConfig
) -> tuple[bool, float]:
    """Do two frames show the same thing?

    Comparing metric values between different scene content is meaningless, and
    a handheld recording sweeps across a room.  A pair is accepted only when the
    global statistics agree and phase correlation puts the two views within a
    few pixels of each other.  Returns (accepted, displacement_px).
    """
    pre = Preprocessor(config.pipeline)
    ga = pre.prepare(a).gray.copy()
    gb = pre.prepare(b).gray.copy()
    if ga.shape != gb.shape:
        return False, math.inf

    mean_a, mean_b = float(ga.mean()), float(gb.mean())
    if abs(mean_a - mean_b) / max(mean_a, 1e-9) > 0.06:
        return False, math.inf
    # Contrast is allowed to differ: that is what defocus does to it.

    # Even dimensions, or phaseCorrelate carries a half-pixel bias.
    h = ga.shape[0] - ga.shape[0] % 2
    w = ga.shape[1] - ga.shape[1] % 2
    pa = np.ascontiguousarray(ga[:h, :w])
    pb = np.ascontiguousarray(gb[:h, :w])
    window = cv2.createHanningWindow((w, h), cv2.CV_32F)
    try:
        bias, _ = cv2.phaseCorrelate(pa.copy(), pa.copy(), window)
        (dx, dy), response = cv2.phaseCorrelate(pa.copy(), pb.copy(), window)
    except cv2.error:
        return False, math.inf
    shift = float(np.hypot(dx - bias[0], dy - bias[1]))
    return (shift < 6.0 and response > 0.25), shift


def experiment_psf(
    recording: Recording,
    images: list[tuple[dict[str, str], np.ndarray]],
    config: SharpnessConfig,
    max_frame_gap: int = 6,
) -> dict[str, Any]:
    """Does synthetic defocus degrade the metrics the way real defocus does?

    For every pair of saved frames that show the SAME view, the sharper one is
    synthetically defocused until its measured edge width matches the softer
    one, and the remaining five metrics are compared.  A systematic ratio away
    from 1.0 means the disc PSF is not what the lens does.

    The same-view requirement is what makes this a test of the PSF rather than
    of the scene: an earlier version compared one reference frame against the
    whole recording and measured mostly the difference between rooms.
    """
    if len(images) < 8:
        return {"pairs": 0, "rejected_different_content": 0}

    preprocessor = Preprocessor(config.pipeline)
    metrics = build_metrics(config.metrics)
    others = [m for m in metrics if m.name != "edge_width"]

    def vector(image: np.ndarray) -> dict[str, float]:
        gray = preprocessor.prepare(image).gray
        return {m.name: m.compute(gray) for m in others}

    sigmas = [measured_blur_radius(image, config) for _, image in images]

    pairs: list[dict[str, Any]] = []
    rejected = 0
    for i in range(len(images)):
        if not np.isfinite(sigmas[i]):
            continue
        for j in range(max(0, i - max_frame_gap), min(len(images), i + max_frame_gap + 1)):
            if i == j or not np.isfinite(sigmas[j]):
                continue
            # j must be the sharper one, i the softer target.
            if sigmas[j] >= sigmas[i]:
                continue
            extra = sigmas[i] ** 2 - sigmas[j] ** 2
            if extra <= 0.25 or math.sqrt(extra) > 12.0:
                continue
            ok, shift = content_matches(images[j][1], images[i][1], config)
            if not ok:
                rejected += 1
                continue
            radius = math.sqrt(extra)
            synthetic = defocus(images[j][1], radius)
            synthetic_sigma = measured_blur_radius(synthetic, config)
            if not np.isfinite(synthetic_sigma):
                continue
            real_v, synth_v = vector(images[i][1]), vector(synthetic)
            pairs.append({
                "real_sigma": sigmas[i],
                "source_sigma": sigmas[j],
                "applied_radius": radius,
                "synthetic_sigma": synthetic_sigma,
                "displacement_px": shift,
                "ratio": {
                    name: (synth_v[name] / real_v[name])
                    if real_v[name] > 1e-12 else math.nan
                    for name in real_v
                },
            })

    if not pairs:
        return {"pairs": 0, "rejected_different_content": rejected}

    summary: dict[str, Any] = {
        "pairs": len(pairs),
        "rejected_different_content": rejected,
        "median_displacement_px": float(np.median([p["displacement_px"] for p in pairs])),
        "median_applied_radius_px": float(np.median([p["applied_radius"] for p in pairs])),
        "edge_width_match_error_px": float(np.median(
            [abs(p["synthetic_sigma"] - p["real_sigma"]) for p in pairs]
        )),
        "metric_ratio_synthetic_over_real": {},
    }
    for name in [m.name for m in others]:
        values = np.array([p["ratio"][name] for p in pairs])
        values = values[np.isfinite(values)]
        if values.size:
            summary["metric_ratio_synthetic_over_real"][name] = {
                "median": float(np.median(values)),
                "p25": float(np.percentile(values, 25)),
                "p75": float(np.percentile(values, 75)),
            }
    return summary


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict[str, Any]) -> None:
    print("=" * 78)
    print("STUDY OF A REAL RECORDING")
    print("=" * 78)

    e1 = results["E1_order_dependence"]
    print("\nE1  ORDER DEPENDENCE  (how much of the score is history?)")
    print(f"    {e1['frames']} frames, {e1['trials']} shuffles")
    print(f"    raw metrics, worst relative difference : "
          f"{e1['raw_worst_relative_difference']:.3e}  (must be 0)")
    print(f"    same frame, instantaneous score        : "
          f"mean {e1['instantaneous_mean_abs_diff']:.3f}  "
          f"p95 {e1['instantaneous_p95_abs_diff']:.3f}  "
          f"max {e1['instantaneous_max_abs_diff']:.3f}")
    print(f"    same frame, filtered score             : "
          f"mean {e1['filtered_mean_abs_diff']:.3f}  "
          f"max {e1['filtered_max_abs_diff']:.3f}")
    print(f"    rank correlation across orders         : "
          f"mean {e1['rank_correlation_mean']:.3f}  "
          f"min {e1['rank_correlation_min']:.3f}")

    e2 = results["E2_held_groups"]
    print(f"\nE2  HELD-POSITION GROUPS  ({e2.get('groups', 0)} groups, "
          f"median {e2.get('median_group_frames', 0)} frames, "
          f"{100 * e2.get('coverage_fraction', 0):.0f}% of the run)")
    if e2.get("per_metric"):
        print(f"    {'metric':12s} {'within':>9} {'between':>9} {'discriminability':>18}")
        ranked = sorted(e2["per_metric"].items(),
                        key=lambda kv: -kv[1]["discriminability"])
        for name, values in ranked:
            print(f"    {name:12s} {values['within_group_relative_sd']:9.4f} "
                  f"{values['between_group_relative_sd']:9.4f} "
                  f"{values['discriminability']:18.1f}")

    print("\nE3/E4  KNOWN DEFOCUS APPLIED TO REAL FRAMES")
    print("    monotonicity: fraction of ladder steps in the correct direction")
    for key in sorted(k for k in results if k.startswith("E3_") or k.startswith("E4_")):
        block = results[key]
        print(f"\n    -- {block['label']} "
              f"({block['base_frames']} base frames, "
              f"{len(block['radii'])} defocus steps)")
        print(f"       {'raw metric':14s} mono | {'method':18s} mono   worst")
        methods = block["method_monotonicity"]
        worst = block["method_monotonicity_min"]
        ordered_methods = ["plain_mean", "fixed_weighted", "adaptive"]
        raw_items = list(block["raw_metric_monotonicity"].items())
        for index in range(max(len(raw_items), len(ordered_methods))):
            left = ""
            if index < len(raw_items):
                name, value = raw_items[index]
                left = f"{name:14s} {value:4.2f}"
            right = ""
            if index < len(ordered_methods):
                method = ordered_methods[index]
                right = f"{method:18s} {methods[method]:4.2f}   {worst[method]:4.2f}"
            print(f"       {left:21s}| {right}")

    e5 = results["E5_psf_model"]
    print("\nE5  PSF MODEL CHECK  (does synthetic defocus behave like real defocus?)")
    if not e5.get("pairs"):
        print("    not enough usable pairs in this recording")
    else:
        print(f"    {e5['pairs']} same-view pairs "
              f"({e5['rejected_different_content']} rejected: different content)")
        print(f"    median view displacement {e5['median_displacement_px']:.2f} px, "
              f"median applied radius {e5['median_applied_radius_px']:.2f} px")
        print(f"    edge width matched to within {e5['edge_width_match_error_px']:.3f} px")
        print("    ratio of synthetic to real, per metric (1.00 = the model matches):")
        for name, values in e5["metric_ratio_synthetic_over_real"].items():
            flag = "" if 0.7 <= values["median"] <= 1.4 else "   <-- model differs"
            print(f"      {name:12s} median {values['median']:5.2f}  "
                  f"[{values['p25']:5.2f}, {values['p75']:5.2f}]{flag}")


def make_figures(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    e2 = results["E2_held_groups"].get("per_metric", {})
    if e2:
        names = sorted(e2, key=lambda n: -e2[n]["discriminability"])
        ax.barh(names, [e2[n]["discriminability"] for n in names], color="#00798c")
        ax.invert_yaxis()
        ax.set_xlabel("between-group / within-group spread")
        ax.set_title("E2  Discriminability on real frames")
        ax.grid(axis="x", color="#e6e8eb")

    ax = axes[1]
    keys = sorted(k for k in results if k.startswith("E3_") or k.startswith("E4_"))
    methods = ["plain_mean", "fixed_weighted", "adaptive"]
    width = 0.25
    x = np.arange(len(keys))
    colours = ["#9aa0a6", "#edae49", "#111111"]
    for index, method in enumerate(methods):
        values = [results[k]["method_monotonicity"][method] for k in keys]
        ax.bar(x + (index - 1) * width, values, width, label=method,
               color=colours[index])
    ax.set_xticks(x)
    ax.set_xticklabels([results[k]["label"].replace(" ", "\n") for k in keys],
                       fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("monotonicity")
    ax.set_title("E3/E4  Known defocus on real content")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#e6e8eb")

    ax = axes[2]
    e5 = results["E5_psf_model"].get("metric_ratio_synthetic_over_real", {})
    if e5:
        names = list(e5)
        medians = [e5[n]["median"] for n in names]
        lows = [e5[n]["median"] - e5[n]["p25"] for n in names]
        highs = [e5[n]["p75"] - e5[n]["median"] for n in names]
        ax.errorbar(medians, names, xerr=[lows, highs], fmt="o", color="#d1495b",
                    capsize=4)
        ax.axvline(1.0, color="#111111", linestyle="--", linewidth=1.2)
        ax.set_xlabel("synthetic / real metric value")
        ax.set_title("E5  Does the disc PSF match the lens?")
        ax.grid(axis="x", color="#e6e8eb")

    fig.tight_layout()
    path = out_dir / "real_frame_study.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--base-frames", type=int, default=5,
                        help="how many real sharp frames to build ladders from")
    parser.add_argument("--shuffles", type=int, default=12)
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_default_config()
    recording = Recording(args.directory)
    images = recording.saved_frames()
    print(f"loaded {len(recording.rows)} rows, {len(images)} saved frames")
    if len(images) < 8:
        print("too few saved frames for this study", file=sys.stderr)
        return 2

    frame_list = [image for _, image in images]
    bases = sharpest_frames(recording, images, args.base_frames)
    radii = [0.0, 1.0, 2.0, 3.0, 4.5, 6.0, 8.0]

    results: dict[str, Any] = {
        "recording": str(args.directory),
        "E1_order_dependence": experiment_order(frame_list, config, args.shuffles),
        "E2_held_groups": experiment_groups(recording),
        "E3_clean": experiment_ladder(bases, config, radii, "clean"),
        "E4_noise20": experiment_ladder(bases, config, radii, "noise sigma 20",
                                        noise_sigma=20.0),
        "E4_dim": experiment_ladder(bases, config, radii, "exposure x0.5",
                                    exposure_gain=0.5),
        "E5_psf_model": experiment_psf(recording, images, config),
    }

    print_report(results)

    if args.figures:
        try:
            make_figures(results, args.figures)
        except ImportError:
            print("matplotlib not installed; skipping figures", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float),
                             encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
