"""Comprehensive comparison of this project's scheme against published methods.

Four parts, all run on frames from a real recording:

A  Speed.        Cost of every focus measure and every combination scheme,
                 and how each scales with analysis resolution.
B  Separability. How well each measure distinguishes real focus positions,
                 using held-position groups. Needs no ground truth.
C  Accuracy.     A defocus ladder with known ground truth, built by degrading
                 real frames. Monotonicity and peak-localisation error.
D  Combination.  This project's adaptive scheme against seven standard fusion
                 rules, three of which are offline and see the whole sequence.

The offline schemes are not competitors in the ordinary sense - a real-time
system cannot use them - but they bound what any fusion of these measures can
achieve on this data, which is the more useful comparison.

    python3 tools/comprehensive_study.py data/run1 --figures docs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import SharpnessConfig, load_config, load_default_config  # noqa: E402
from adaptive_sharpness.analysis import ImageAnalyzer  # noqa: E402
from adaptive_sharpness.ensemble import AdaptiveEnsemble  # noqa: E402
from adaptive_sharpness.metrics import build_metrics  # noqa: E402
from adaptive_sharpness.normalize import NormalizerBank  # noqa: E402
from adaptive_sharpness.preprocess import Preprocessor  # noqa: E402
from adaptive_sharpness.synthetic import add_noise, apply_exposure, defocus  # noqa: E402
from tools.baselines import COMBINERS, FOCUS_MEASURES, OFFLINE_COMBINERS  # noqa: E402
from tools.study_real_frames import (  # noqa: E402
    Recording,
    segment_held_groups,
    sharpest_frames,
)

OWN = ["laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width"]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def all_measures(config: SharpnessConfig):
    """This project's six plus the published baselines, behind one interface."""
    own = {m.name: m.compute for m in build_metrics(config.metrics)}
    return {**own, **FOCUS_MEASURES}


def normalise_columns(matrix: np.ndarray) -> np.ndarray:
    """Robust per-column scaling to [0, 1] over a whole sequence.

    Every measure is put on the same footing, including the baselines, so the
    comparison is of the measures and not of anyone's normalisation.
    """
    out = np.zeros_like(matrix, dtype=np.float64)
    for column in range(matrix.shape[1]):
        values = matrix[:, column].astype(np.float64)
        values = np.log1p(np.maximum(values - values.min(), 0.0))
        low, high = np.percentile(values, 5), np.percentile(values, 95)
        span = high - low
        out[:, column] = 0.5 if span < 1e-12 else np.clip((values - low) / span, 0, 1)
    return out


def monotonicity_to_peak(values: np.ndarray, peak: int) -> float:
    """Fraction of steps moving towards the peak from either side."""
    if values.size < 2:
        return 0.0
    correct = total = 0.0
    for index in range(values.size - 1):
        delta = values[index + 1] - values[index]
        if index + 1 <= peak:
            expected_rise = True
        elif index >= peak:
            expected_rise = False
        else:
            continue
        total += 1
        if abs(delta) < 1e-12:
            correct += 0.5
        elif (delta > 0) == expected_rise:
            correct += 1
    return correct / total if total else 0.0


# ---------------------------------------------------------------------------
# Part A - speed
# ---------------------------------------------------------------------------

def part_a_speed(
    frames: list[np.ndarray], config: SharpnessConfig, repeats: int = 3
) -> dict[str, Any]:
    preprocessor = Preprocessor(config.pipeline)
    grays = [preprocessor.prepare(f).gray.copy() for f in frames[:20]]
    measures = all_measures(config)

    per_measure: dict[str, float] = {}
    for name, function in measures.items():
        best = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            for gray in grays:
                function(gray)
            best = min(best, (time.perf_counter() - start) / len(grays))
        per_measure[name] = best * 1e3

    # Combination cost, measured on a realistic matrix.
    matrix = np.random.default_rng(0).uniform(0, 1, (len(grays), len(OWN)))
    priors = np.array([config.metrics.priors[n] for n in OWN])
    per_combiner: dict[str, float] = {}
    for name, function in COMBINERS.items():
        best = math.inf
        for _ in range(repeats):
            start = time.perf_counter()
            function(matrix, priors)
            best = min(best, (time.perf_counter() - start) / matrix.shape[0])
        per_combiner[name] = best * 1e3

    analyzer = ImageAnalyzer(config.analysis)
    ensemble = AdaptiveEnsemble(tuple(OWN), config.ensemble, config.metrics.priors)
    start = time.perf_counter()
    for gray in grays:
        stats = analyzer.analyze(gray)
        degradations = analyzer.degradations(stats)
        ensemble.combine({n: 0.5 for n in OWN}, stats, degradations)
    per_combiner["adaptive (this project)"] = (
        (time.perf_counter() - start) / len(grays) * 1e3
    )

    # Resolution scaling for this project's six.
    scaling: dict[int, float] = {}
    own_metrics = build_metrics(config.metrics)
    for width in (160, 240, 320, 480, 640):
        cfg = config.with_overrides(pipeline={"analysis_width": width})
        pre = Preprocessor(cfg.pipeline)
        prepared = [pre.prepare(f).gray.copy() for f in frames[:8]]
        start = time.perf_counter()
        for gray in prepared:
            for metric in own_metrics:
                metric.compute(gray)
        scaling[width] = (time.perf_counter() - start) / len(prepared) * 1e3

    return {
        "analysis_width": config.pipeline.analysis_width,
        "frames_timed": len(grays),
        "measure_ms": per_measure,
        "combiner_ms": per_combiner,
        "own_total_ms": sum(per_measure[n] for n in OWN),
        "resolution_scaling_ms": scaling,
    }


def low_texture_frames(
    recording: Recording, images: list[tuple[dict[str, str], np.ndarray]], count: int
) -> list[np.ndarray]:
    """Real frames with the least structure.

    Low texture is the condition under which the fusion rules diverged in the
    synthetic study, and a recording of an ordinary room contains such frames -
    they just have to be selected rather than generated.
    """
    scored = []
    for row, image in images:
        try:
            density = float(row["edge_density"])
            contrast = float(row["local_contrast"])
        except (KeyError, ValueError):
            continue
        scored.append((density * contrast, image))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0])
    return [image for _, image in scored[:count]]


# ---------------------------------------------------------------------------
# Part B - separability on real held-position groups
# ---------------------------------------------------------------------------

def fusion_series(
    matrix: np.ndarray,
    stats: list[Any],
    degradations: list[dict[str, float]],
    config: SharpnessConfig,
) -> dict[str, np.ndarray]:
    """Every fusion rule's output for a sequence, including this project's."""
    priors = np.array([config.metrics.priors[n] for n in OWN])
    out = {
        name: np.asarray(function(matrix, priors), dtype=np.float64)
        for name, function in COMBINERS.items()
    }
    ensemble = AdaptiveEnsemble(tuple(OWN), config.ensemble, config.metrics.priors)
    out["adaptive"] = np.array([
        ensemble.combine(
            {name: float(matrix[i, j]) for j, name in enumerate(OWN)},
            stats[i], degradations[i],
        ).score
        for i in range(matrix.shape[0])
    ])
    return out


def part_b_separability(
    recording: Recording,
    images: list[tuple[dict[str, str], np.ndarray]],
    config: SharpnessConfig,
) -> dict[str, Any]:
    """Separability of real focus positions, for measures AND fusion rules.

    The ratio is between-group standard deviation over within-group standard
    deviation, both in the signal's own units.  That is invariant to scale and
    to offset, which matters because the comparison mixes unbounded raw metrics
    with fusion outputs bounded to [0, 1]: an earlier version divided each
    standard deviation by its mean, which is scale-free but not shift-free, and
    therefore favoured the unbounded signals.
    """
    groups = segment_held_groups(recording.column("raw_tenengrad"))
    if not groups:
        return {"groups": 0}

    preprocessor = Preprocessor(config.pipeline)
    analyzer = ImageAnalyzer(config.analysis)
    measures = all_measures(config)
    own_metrics = build_metrics(config.metrics)

    values: dict[str, list[float]] = {name: [] for name in measures}
    stats, degradations = [], []
    own_raw: dict[str, list[float]] = {m.name: [] for m in own_metrics}
    for _, image in images:
        gray = preprocessor.prepare(image).gray
        image_stats = analyzer.analyze(gray)
        stats.append(image_stats)
        degradations.append(analyzer.degradations(image_stats))
        for name, function in measures.items():
            values[name].append(function(gray))
        for metric in own_metrics:
            own_raw[metric.name].append(values[metric.name][-1])

    # Fusion rules operate on normalised metric values, so build that matrix
    # once and score every rule from it.
    matrix = normalise_columns(
        np.column_stack([own_raw[n] for n in OWN]).astype(np.float64)
    )
    for name, series in fusion_series(matrix, stats, degradations, config).items():
        values[f"fusion:{name}"] = list(series)

    # Fusion outputs are normalised and clipped to [0, 1]; raw measures are
    # neither. Clipping saturates - measured at 18-22% of frames - which
    # compresses between-group spread without compressing within-group noise,
    # so comparing raw measures against fused ones penalises the fused side by
    # construction. Every individual measure therefore also gets a normalised
    # twin, put through exactly the same transform the fusion inputs went
    # through, and that is the column the two sides are compared on.
    measure_names_list = list(measures)
    normalised_measures = normalise_columns(
        np.column_stack([values[n] for n in measure_names_list]).astype(np.float64)
    )
    for column, name in enumerate(measure_names_list):
        values[f"norm:{name}"] = list(normalised_measures[:, column])

    csv_positions = [i for i, row in enumerate(recording.rows) if row.get("image_file")]
    group_of = {}
    for group_index, (start, stop) in enumerate(groups):
        for position in range(start, stop):
            group_of[position] = group_index
    membership: dict[int, list[int]] = {}
    for saved_index, csv_index in enumerate(csv_positions):
        group = group_of.get(csv_index)
        if group is not None:
            membership.setdefault(group, []).append(saved_index)
    usable = {g: idx for g, idx in membership.items() if len(idx) >= 2}

    out: dict[str, Any] = {
        "groups": len(groups),
        "groups_with_saved_frames": len(usable),
        "per_measure": {},
    }
    if not usable:
        return out

    for name, series_list in values.items():
        series = np.array(series_list, dtype=np.float64)
        if not np.all(np.isfinite(series)):
            continue
        within, means = [], []
        for indices in usable.values():
            segment = series[indices]
            within.append(float(segment.std()))
            means.append(float(segment.mean()))
        between = float(np.array(means).std())
        within_pooled = max(float(np.median(within)), 1e-12)
        out["per_measure"][name] = {
            "within": within_pooled,
            "between": between,
            "separability": between / within_pooled,
        }
    return out


# ---------------------------------------------------------------------------
# Part C - accuracy on a known ladder built from real frames
# ---------------------------------------------------------------------------

def build_ladder(
    base: np.ndarray, radii: list[float], noise: float, gain: float, seed: int
) -> list[np.ndarray]:
    frames = []
    for index, radius in enumerate(radii):
        image = defocus(base, radius)
        if gain != 1.0:
            image = apply_exposure(image, gain)
        if noise > 0.0:
            image = add_noise(image, noise, seed=seed + index)
        frames.append(image)
    return frames


def part_c_accuracy(
    bases: list[np.ndarray], config: SharpnessConfig
) -> dict[str, Any]:
    # A symmetric sweep through focus, made from real content.
    half = [8.0, 6.0, 4.5, 3.0, 2.0, 1.0]
    radii = half + [0.0] + half[::-1]
    peak = len(half)

    preprocessor = Preprocessor(config.pipeline)
    measures = all_measures(config)
    conditions = {
        "clean": (0.0, 1.0),
        "noise sigma 35": (35.0, 1.0),
        "dark x0.25 + noise 25": (25.0, 0.25),
        "clipped x2.5": (0.0, 2.5),
    }

    results: dict[str, Any] = {"radii": radii, "peak_index": peak, "conditions": {}}
    for label, (noise, gain) in conditions.items():
        mono: dict[str, list[float]] = {name: [] for name in measures}
        peak_err: dict[str, list[int]] = {name: [] for name in measures}
        for seed, base in enumerate(bases):
            frames = build_ladder(base, radii, noise, gain, seed * 100)
            grays = [preprocessor.prepare(f).gray.copy() for f in frames]
            for name, function in measures.items():
                series = np.array([function(g) for g in grays], dtype=np.float64)
                mono[name].append(monotonicity_to_peak(series, peak))
                peak_err[name].append(int(abs(int(np.argmax(series)) - peak)))
        results["conditions"][label] = {
            name: {
                "monotonicity": float(np.mean(mono[name])),
                "peak_error_mean": float(np.mean(peak_err[name])),
                "peak_error_max": int(np.max(peak_err[name])),
            }
            for name in measures
        }
    return results


# ---------------------------------------------------------------------------
# Part D - combination schemes head to head
# ---------------------------------------------------------------------------

def part_d_combination(
    bases: list[np.ndarray], config: SharpnessConfig
) -> dict[str, Any]:
    half = [8.0, 6.0, 4.5, 3.0, 2.0, 1.0]
    radii = half + [0.0] + half[::-1]
    peak = len(half)

    preprocessor = Preprocessor(config.pipeline)
    analyzer_template = config.analysis
    own_metrics = build_metrics(config.metrics)
    priors = np.array([config.metrics.priors[n] for n in OWN])

    conditions = {
        "clean": (0.0, 1.0),
        "noise sigma 35": (35.0, 1.0),
        "dark x0.25 + noise 25": (25.0, 0.25),
        "clipped x2.5": (0.0, 2.5),
    }
    scores: dict[str, dict[str, list[float]]] = {}
    errors: dict[str, dict[str, list[int]]] = {}

    for label, (noise, gain) in conditions.items():
        scores[label] = {}
        errors[label] = {}
        for seed, base in enumerate(bases):
            frames = build_ladder(base, radii, noise, gain, seed * 100)
            analyzer = ImageAnalyzer(analyzer_template)
            grays, stats, degradations = [], [], []
            raw = {m.name: [] for m in own_metrics}
            for frame in frames:
                gray = preprocessor.prepare(frame).gray
                grays.append(gray)
                image_stats = analyzer.analyze(gray)
                stats.append(image_stats)
                degradations.append(analyzer.degradations(image_stats))
                for metric in own_metrics:
                    raw[metric.name].append(metric.compute(gray))

            matrix = normalise_columns(
                np.column_stack([raw[n] for n in OWN]).astype(np.float64)
            )
            # Baseline fusion rules.
            for name, function in COMBINERS.items():
                series = np.asarray(function(matrix, priors), dtype=np.float64)
                scores[label].setdefault(name, []).append(
                    monotonicity_to_peak(series, peak)
                )
                errors[label].setdefault(name, []).append(
                    int(abs(int(np.argmax(series)) - peak))
                )
            # This project's scheme, frame by frame as it runs live.
            ensemble = AdaptiveEnsemble(tuple(OWN), config.ensemble, config.metrics.priors)
            adaptive = []
            for index in range(len(frames)):
                row = {name: float(matrix[index, j]) for j, name in enumerate(OWN)}
                adaptive.append(
                    ensemble.combine(row, stats[index], degradations[index]).score
                )
            series = np.array(adaptive)
            scores[label].setdefault("adaptive", []).append(
                monotonicity_to_peak(series, peak)
            )
            errors[label].setdefault("adaptive", []).append(
                int(abs(int(np.argmax(series)) - peak))
            )

    return {
        "conditions": {
            label: {
                name: {
                    "monotonicity": float(np.mean(values)),
                    "peak_error_mean": float(np.mean(errors[label][name])),
                    "offline": name in OFFLINE_COMBINERS,
                }
                for name, values in per_label.items()
            }
            for label, per_label in scores.items()
        }
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_report(results: dict[str, Any]) -> None:
    print("=" * 80)
    print("COMPREHENSIVE STUDY")
    print("=" * 80)

    a = results["A_speed"]
    print(f"\nA  SPEED  (analysis {a['analysis_width']} px, "
          f"{a['frames_timed']} real frames, best of 3)")
    print(f"\n   {'focus measure':14s} {'ms/frame':>9}   source")
    print("   " + "-" * 46)
    for name, value in sorted(a["measure_ms"].items(), key=lambda kv: kv[1]):
        origin = "this project" if name in OWN else "baseline"
        print(f"   {name:14s} {value:9.3f}   {origin}")
    print(f"\n   this project's six, total: {a['own_total_ms']:.2f} ms")

    print(f"\n   {'fusion rule':24s} {'ms/frame':>9}")
    print("   " + "-" * 36)
    for name, value in sorted(a["combiner_ms"].items(), key=lambda kv: kv[1]):
        print(f"   {name:24s} {value:9.4f}")

    print("\n   resolution scaling of the six:")
    for width, value in a["resolution_scaling_ms"].items():
        print(f"     {width:4d} px  {value:6.2f} ms")

    b = results["B_separability"]
    print(f"\nB  SEPARABILITY ON REAL FRAMES  "
          f"({b.get('groups_with_saved_frames', 0)} held-position groups)")
    if b.get("per_measure"):
        print(f"\n   {'measure':14s} {'within':>8} {'between':>9} {'separability':>13}  source")
        print("   " + "-" * 62)
        ranked = sorted(b["per_measure"].items(), key=lambda kv: -kv[1]["separability"])
        for name, values in ranked:
            if name.startswith("norm:"):
                continue  # shown in the like-for-like table below
            if name.startswith("fusion:"):
                rule = name.split(":", 1)[1]
                origin = ("FUSION, this project" if rule == "adaptive"
                          else "fusion, baseline")
                shown = rule + " (fused)"
            else:
                origin = "this project" if name in OWN else "baseline"
                shown = name
            print(f"   {shown:20s} {values['within']:8.4f} {values['between']:9.4f} "
                  f"{values['separability']:13.1f}  {origin}")


        # Like-for-like: every signal put through the same normalisation and
        # clipping the fusion inputs went through, so the bounded fused
        # scores are not compared against unbounded raw measures.
        print("")
        print("   like for like - all signals normalised and clipped alike:")
        print(f"   {'signal':24s} {'separability':>13}  source")
        print("   " + "-" * 52)
        fair = {}
        for name, entry in b["per_measure"].items():
            if name.startswith("norm:"):
                fair[name[5:]] = (entry["separability"], "measure")
            elif name.startswith("fusion:"):
                fair[name[7:] + " (fused)"] = (entry["separability"], "fusion")
        for name, (value, kind) in sorted(fair.items(), key=lambda kv: -kv[1][0]):
            base = name.replace(" (fused)", "")
            if kind == "fusion":
                origin = ("FUSION, this project" if base == "adaptive"
                          else "fusion, baseline")
            else:
                origin = "this project" if base in OWN else "baseline"
            print(f"   {name:24s} {value:13.1f}  {origin}")

    c = results["C_accuracy"]
    print(f"\nC  ACCURACY ON A KNOWN LADDER FROM REAL FRAMES  "
          f"(peak at index {c['peak_index']} of {len(c['radii'])})")
    for label, per_measure in c["conditions"].items():
        print(f"\n   -- {label}")
        print(f"      {'measure':14s} {'monotonic':>10} {'peak err':>9}  source")
        ranked = sorted(per_measure.items(),
                        key=lambda kv: (kv[1]["peak_error_mean"], -kv[1]["monotonicity"]))
        for name, values in ranked:
            origin = "this project" if name in OWN else "baseline"
            print(f"      {name:14s} {values['monotonicity']:10.3f} "
                  f"{values['peak_error_mean']:9.2f}  {origin}")

    for key, title in (
        ("D_combination", "D  FUSION RULES HEAD TO HEAD  (sharp real frames)"),
        ("D_combination_low_texture",
         "D2 FUSION RULES ON LOW-TEXTURE REAL FRAMES"),
    ):
        block = results.get(key, {})
        if block.get("conditions"):
            print_fusion(title, block)


def print_fusion(title: str, d: dict[str, Any]) -> None:
    """One fusion-rule table. Split out so the same layout serves both the
    sharp-frame and the low-texture comparison."""
    print("")
    print(title)
    for label, per_rule in d["conditions"].items():
        print(f"\n   -- {label}")
        print(f"      {'rule':24s} {'monotonic':>10} {'peak err':>9}  mode")
        ranked = sorted(per_rule.items(),
                        key=lambda kv: (kv[1]["peak_error_mean"], -kv[1]["monotonicity"]))
        for name, values in ranked:
            mode = "OFFLINE" if values["offline"] else "online"
            mark = "  <-- this project" if name == "adaptive" else ""
            print(f"      {name:24s} {values['monotonicity']:10.3f} "
                  f"{values['peak_error_mean']:9.2f}  {mode}{mark}")


def make_figures(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    own_colour, base_colour = "#00798c", "#9aa0a6"

    # --- Figure 1: speed ----------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    a = results["A_speed"]
    items = sorted(a["measure_ms"].items(), key=lambda kv: kv[1])
    names = [n for n, _ in items]
    values = [v for _, v in items]
    colours = [own_colour if n in OWN else base_colour for n in names]
    ax = axes[0]
    ax.barh(names, values, color=colours)
    ax.invert_yaxis()
    ax.set_xlabel("ms per frame")
    ax.set_title("Cost per focus measure (320 px, Raspberry Pi 5)")
    ax.grid(axis="x", color="#e6e8eb")
    handles = [plt.Rectangle((0, 0), 1, 1, color=own_colour),
               plt.Rectangle((0, 0), 1, 1, color=base_colour)]
    ax.legend(handles, ["this project", "published baseline"], fontsize=8)

    ax = axes[1]
    widths = list(a["resolution_scaling_ms"])
    ax.plot(widths, [a["resolution_scaling_ms"][w] for w in widths],
            marker="o", color="#111111", lw=2)
    ax.axhline(40.0, color="#d1495b", ls="--", lw=1.3)
    ax.annotate("40 ms camera frame interval", (widths[0], 41.5),
                fontsize=8, color="#d1495b")
    ax.set_xlabel("analysis width, px")
    ax.set_ylabel("ms per frame, six measures")
    ax.set_title("Resolution scaling")
    ax.grid(color="#e6e8eb")
    fig.tight_layout()
    fig.savefig(out_dir / "study_speed.png", dpi=130, facecolor="white")
    plt.close(fig)

    # --- Figure 2: separability + accuracy ----------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    b = results["B_separability"].get("per_measure", {})
    if b:
        # Like-for-like only: every signal normalised and clipped alike, so the
        # bounded fused scores are not plotted against unbounded raw measures.
        fair = {}
        for name, entry in b.items():
            if name.startswith("norm:"):
                fair[name[5:]] = (entry["separability"], "measure")
            elif name.startswith("fusion:"):
                fair[name[7:] + " (fused)"] = (entry["separability"], "fusion")
        ranked = sorted(fair.items(), key=lambda kv: -kv[1][0])
        names = [n for n, _ in ranked]
        values = [v[0] for _, v in ranked]
        colours = []
        for name, (_, kind) in ranked:
            base = name.replace(" (fused)", "")
            if kind == "fusion":
                colours.append("#d1495b" if base == "adaptive" else "#edae49")
            else:
                colours.append(own_colour if base in OWN else base_colour)
        ax = axes[0]
        ax.barh(names, values, color=colours)
        ax.invert_yaxis()
        ax.set_xlabel("between-group / within-group standard deviation")
        ax.set_title("Separability of real focus positions"
                     "\n(all signals normalised alike)")
        ax.grid(axis="x", color="#e6e8eb")
        legend = [
            plt.Rectangle((0, 0), 1, 1, color=own_colour),
            plt.Rectangle((0, 0), 1, 1, color=base_colour),
            plt.Rectangle((0, 0), 1, 1, color="#edae49"),
            plt.Rectangle((0, 0), 1, 1, color="#d1495b"),
        ]
        ax.legend(legend,
                  ["our measure", "published measure", "fusion rule",
                   "our adaptive fusion"], fontsize=7, loc="lower right")

    ax = axes[1]
    c = results["C_accuracy"]["conditions"]
    labels = list(c)
    measures = list(next(iter(c.values())))
    x = np.arange(len(measures))
    width = 0.8 / len(labels)
    palette = ["#111111", "#d1495b", "#edae49"]
    for index, label in enumerate(labels):
        values = [c[label][m]["peak_error_mean"] for m in measures]
        ax.bar(x + (index - (len(labels) - 1) / 2) * width, values, width,
               label=label, color=palette[index % len(palette)])
    ax.set_xticks(x)
    ax.set_xticklabels(measures, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("mean peak error, steps")
    ax.set_title("Peak localisation on a known ladder from real frames")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#e6e8eb")
    fig.tight_layout()
    fig.savefig(out_dir / "study_quality.png", dpi=130, facecolor="white")
    plt.close(fig)

    # --- Figure 3: fusion rules --------------------------------------------
    fig, ax = plt.subplots(figsize=(11, 5))
    d = results["D_combination"]["conditions"]
    labels = list(d)
    rules = list(next(iter(d.values())))
    x = np.arange(len(rules))
    width = 0.8 / len(labels)
    for index, label in enumerate(labels):
        values = [d[label][r]["peak_error_mean"] for r in rules]
        ax.bar(x + (index - (len(labels) - 1) / 2) * width, values, width,
               label=label, color=palette[index % len(palette)])
    ax.set_xticks(x)
    tick_labels = [
        r + ("\n(offline)" if d[labels[0]][r]["offline"] else "") for r in rules
    ]
    ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=8)
    for tick, rule in zip(ax.get_xticklabels(), rules):
        if rule == "adaptive":
            tick.set_color("#00798c")
            tick.set_fontweight("bold")
    ax.set_ylabel("mean peak error, steps")
    ax.set_title("Fusion rules on a known ladder from real frames "
                 "(lower is better; 'adaptive' is this project)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", color="#e6e8eb")
    fig.tight_layout()
    fig.savefig(out_dir / "study_fusion.png", dpi=130, facecolor="white")
    plt.close(fig)
    print(f"\nwrote three figures into {out_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--base-frames", type=int, default=4)
    args = parser.parse_args(argv)

    config = load_config(args.config) if args.config else load_default_config()
    recording = Recording(args.directory)
    images = recording.saved_frames()
    print(f"loaded {len(recording.rows)} rows, {len(images)} saved frames")
    if len(images) < 8:
        print("too few saved frames", file=sys.stderr)
        return 2

    frame_list = [image for _, image in images]
    bases = sharpest_frames(recording, images, args.base_frames)
    low_texture = low_texture_frames(recording, images, args.base_frames)
    print(f"base frames: {len(bases)} sharp, {len(low_texture)} low-texture")

    results = {
        "recording": str(args.directory),
        "A_speed": part_a_speed(frame_list, config),
        "B_separability": part_b_separability(recording, images, config),
        "C_accuracy": part_c_accuracy(bases, config),
        "D_combination": part_d_combination(bases, config),
        "D_combination_low_texture": part_d_combination(low_texture, config)
        if low_texture else {"conditions": {}},
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
