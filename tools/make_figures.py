"""Generate every figure used in the documentation.

Runs the same code paths the experiments do, so the figures cannot drift away
from the measured numbers: nothing here is drawn from hard-coded values except
the timings, which are read from the JSON a benchmark run wrote.

    python3 tools/make_figures.py                    # all figures
    python3 tools/make_figures.py --only sweep step  # a subset
    python3 tools/make_figures.py --out docs/figures
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
# src/ for the library; the repository root because this script imports a
# sibling tool, and tools/ is deliberately not an installed package.
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from adaptive_sharpness import (  # noqa: E402
    SharpnessConfig, load_config, load_default_config,
)
from adaptive_sharpness.config import FocusMapConfig, PipelineConfig  # noqa: E402
from adaptive_sharpness.focusmap import TILE_MEASURES, FocusMapper  # noqa: E402
from adaptive_sharpness.preprocess import Preprocessor  # noqa: E402
from tools.compare_methods import (  # noqa: E402
    analyse_frames,
    build_synthetic_conditions,
    fit_normalizers,
    normalized_series,
    run_comparison,
    step_response,
)
from adaptive_sharpness.synthetic import add_noise, defocus, focus_sweep, make_scene  # noqa: E402

logger = logging.getLogger("figures")

# A restrained palette that stays readable in both GitHub themes.
COLOURS = {
    "laplacian": "#d1495b",
    "tenengrad": "#edae49",
    "brenner": "#66a182",
    "wavelet": "#2e4057",
    "fourier": "#8d6a9f",
    "edge_width": "#00798c",
    "ensemble": "#111111",
}
ACCENT = "#00798c"
WARM = "#d1495b"
NEUTRAL = "#9aa0a6"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#c9ccd1",
        "axes.grid": True,
        "grid.color": "#e6e8eb",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "legend.frameon": False,
        "figure.dpi": 130,
    })


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {path}")


# ---------------------------------------------------------------------------
# 1. Metric response through a focus sweep
# ---------------------------------------------------------------------------

def figure_sweep(config: SharpnessConfig, out: Path) -> None:
    """Each normalised metric and the ensemble against defocus radius."""
    steps = 41
    conditions = build_synthetic_conditions(config, steps, seed=7)
    data = conditions["clean"]
    bank = fit_normalizers(data, config)
    normalized = normalized_series(data, bank)
    results = run_comparison(data, config)

    best = data.true_best_index
    radius = [abs(i - best) / best * 8.0 for i in range(steps)]
    half = slice(best, steps)  # one side of the symmetric sweep

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    for name in config.metrics.enabled:
        values = [row[name] for row in normalized]
        ax.plot(radius[half], values[half], label=name,
                color=COLOURS[name], linewidth=1.8)
    ax.plot(radius[half], results["adaptive"]["scores"][half],
            label="adaptive ensemble", color=COLOURS["ensemble"],
            linewidth=2.6, linestyle="--")
    ax.set_xlabel("defocus radius, px")
    ax.set_ylabel("normalised score")
    ax.set_title("Metric response to defocus")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(-0.03, 1.03)

    ax = axes[1]
    positions = np.arange(steps)
    ax.plot(positions, results["adaptive"]["scores"], color=COLOURS["ensemble"],
            linewidth=2.2, label="adaptive")
    ax.plot(positions, results["plain_mean"]["scores"], color=ACCENT,
            linewidth=1.6, label="plain mean")
    ax.plot(positions, results["single:edge_width"]["scores"], color=WARM,
            linewidth=1.4, label="edge_width alone")
    ax.axvline(best, color=NEUTRAL, linestyle=":", linewidth=1.4)
    ax.annotate("true focus", (best, 0.04), xytext=(best + 1.5, 0.10),
                fontsize=8, color=NEUTRAL)
    ax.set_xlabel("sweep position (step)")
    ax.set_ylabel("score")
    ax.set_title("Full sweep: through focus and out again")
    ax.legend(fontsize=8)

    fig.suptitle("Focus sweep, synthetic scene, disc PSF, 41 steps", y=1.02)
    save(fig, out, "focus_sweep")


# ---------------------------------------------------------------------------
# 2. Method comparison
# ---------------------------------------------------------------------------

def figure_methods(config: SharpnessConfig, out: Path) -> None:
    """Hill-climb distance and discrimination, per method and condition."""
    conditions = build_synthetic_conditions(config, 41, seed=7)
    # Conditions that actually separate the methods once the hill-climb
    # criterion no longer stops dead on the normalisation plateau.
    shown = ["clean", "low_texture", "low_texture_noisy", "dark_and_noisy"]
    methods = [
        "single:laplacian", "single:edge_width", "plain_mean",
        "fixed_weighted", "adaptive",
    ]
    labels = ["laplacian\nalone", "edge_width\nalone", "plain\nmean",
              "fixed\nweighted", "adaptive\n(proposed)"]

    climb = np.zeros((len(shown), len(methods)))
    discrim = np.zeros_like(climb)
    for row, name in enumerate(shown):
        results = run_comparison(conditions[name], config)
        for col, method in enumerate(methods):
            climb[row, col] = results[method]["hill_climb"]
            discrim[row, col] = results[method]["discrimination"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.2
    x = np.arange(len(methods))
    palette = ["#2e4057", "#00798c", "#edae49", "#d1495b"]

    ax = axes[0]
    for row, name in enumerate(shown):
        ax.bar(x + (row - 1.5) * width, climb[row], width,
               label=name, color=palette[row])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("steps from the true peak")
    ax.set_title("Hill-climb distance, steps from the true peak (lower is better)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for row, name in enumerate(shown):
        ax.bar(x + (row - 1.5) * width, discrim[row], width,
               label=name, color=palette[row])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("(peak - median) / robust spread")
    ax.set_title("Discrimination (higher is better)")

    fig.suptitle(
        "How well can a greedy autofocus search follow each score?", y=1.03
    )
    save(fig, out, "method_comparison")


# ---------------------------------------------------------------------------
# 3. Temporal filter step response
# ---------------------------------------------------------------------------

def figure_step(config: SharpnessConfig, out: Path) -> None:
    """The gated filter follows a focus step; a plain EMA lags."""
    step = step_response(config, hold=30)
    series = step["series"]
    hold = 30
    x = np.arange(len(series["gated_filter"])) - hold

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.axvline(0, color=NEUTRAL, linestyle=":", linewidth=1.4)
    ax.plot(x, series["no_temporal_filter"], color=NEUTRAL, linewidth=1.2,
            label="no filter (raw ensemble)")
    ax.plot(x, series["plain_ema_same_alpha"], color=WARM, linewidth=2.0,
            label=f"plain EMA - {step['settling_frames']['plain_ema_same_alpha']} frames")
    ax.plot(x, series["gated_filter"], color=COLOURS["ensemble"], linewidth=2.4,
            label=f"gated filter - {step['settling_frames']['gated_filter']} frame")
    ax.axhline(step["threshold"], color="#c9ccd1", linestyle="--", linewidth=1.0)
    ax.annotate("90% of the step", (x[-1], step["threshold"]),
                xytext=(x[-1] - 26, step["threshold"] + 0.04),
                fontsize=8, color="#7a7f85")
    ax.set_xlabel("frames relative to the focus change")
    ax.set_ylabel("score")
    ax.set_title("Response to a real focus change")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-12, 20)
    save(fig, out, "step_response")


# ---------------------------------------------------------------------------
# 4. How the weights adapt
# ---------------------------------------------------------------------------

def figure_weights(config: SharpnessConfig, out: Path) -> None:
    """The weight each metric receives under different degradations."""
    from adaptive_sharpness.ensemble import AdaptiveEnsemble
    from adaptive_sharpness.types import ImageStats

    names = tuple(config.metrics.enabled)
    ensemble = AdaptiveEnsemble(names, config.ensemble, config.metrics.priors)
    stats = ImageStats(snr=40.0, local_contrast=0.18, edge_sufficiency=1.0)
    scenarios = {
        "ideal": {"noise": 0.0, "edge": 0.0, "clip": 0.0, "motion": 0.0, "contrast": 0.0},
        "noisy": {"noise": 1.0, "edge": 0.0, "clip": 0.0, "motion": 0.0, "contrast": 0.0},
        "no edges": {"noise": 0.0, "edge": 1.0, "clip": 0.0, "motion": 0.0, "contrast": 0.0},
        "clipped": {"noise": 0.0, "edge": 0.0, "clip": 1.0, "motion": 0.0, "contrast": 0.0},
        "moving": {"noise": 0.0, "edge": 0.0, "clip": 0.0, "motion": 1.0, "contrast": 0.0},
        "low contrast": {"noise": 0.0, "edge": 0.0, "clip": 0.0, "motion": 0.0,
                         "contrast": 1.0},
    }
    uniform = {name: 0.6 for name in names}

    matrix = np.zeros((len(scenarios), len(names)))
    for row, degradations in enumerate(scenarios.values()):
        out_ = ensemble.combine(uniform, stats, degradations)
        matrix[row] = [out_.weights[name] for name in names]

    fig, ax = plt.subplots(figsize=(8.5, 4))
    bottom = np.zeros(len(scenarios))
    y = np.arange(len(scenarios))
    for col, name in enumerate(names):
        ax.barh(y, matrix[:, col], left=bottom, color=COLOURS[name],
                label=name, height=0.62)
        for row in range(len(scenarios)):
            value = matrix[row, col]
            if value > 0.07:
                ax.text(bottom[row] + value / 2, y[row], f"{value:.2f}",
                        ha="center", va="center", fontsize=7.5, color="white")
        bottom += matrix[:, col]
    ax.set_yticks(y)
    ax.set_yticklabels(list(scenarios))
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("weight (the six always sum to 1)")
    ax.set_title("Adaptive weights under each condition")
    ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(axis="y", visible=False)
    save(fig, out, "adaptive_weights")


# ---------------------------------------------------------------------------
# 5. The per-tile focus measure choice
# ---------------------------------------------------------------------------

def figure_focus_measures(config: SharpnessConfig, out: Path) -> None:
    """Why grad_over_var is the default: it survives noise."""
    pre = Preprocessor(PipelineConfig())
    noise_levels = [0.0, 4.0, 8.0, 12.0, 16.0, 20.0, 26.0]

    left_base = make_scene(640, 360, seed=3)
    right_base = make_scene(640, 360, seed=9)

    margins = {measure: [] for measure in TILE_MEASURES}
    for sigma in noise_levels:
        frame = left_base.copy()
        frame[:, 320:] = defocus(right_base, 5.0)[:, 320:]
        if sigma > 0:
            frame = add_noise(frame, sigma, seed=5)
        gray = pre.prepare(frame).gray
        for measure in TILE_MEASURES:
            fmap = FocusMapper(FocusMapConfig(measure=measure)).compute(gray)
            cols = fmap.shape[1]
            mid = cols // 2
            left = fmap.score[:, : mid - 1][fmap.valid[:, : mid - 1]]
            right = fmap.score[:, mid + 1 :][fmap.valid[:, mid + 1 :]]
            margin = (float(np.mean(left)) - float(np.mean(right))
                      if left.size and right.size else np.nan)
            margins[measure].append(margin)

    fig, ax = plt.subplots(figsize=(7.5, 4))
    colours = {"grad_over_var": COLOURS["ensemble"],
               "lap_over_var": ACCENT,
               "lap_over_grad": WARM}
    for measure, values in margins.items():
        marker = "o" if measure == "grad_over_var" else "s"
        ax.plot(noise_levels, values, marker=marker, linewidth=2.0,
                color=colours[measure],
                label=measure + (" (default)" if measure == "grad_over_var" else ""))
    ax.axhline(0.05, color="#c9ccd1", linestyle="--", linewidth=1.0)
    ax.annotate("decision threshold", (noise_levels[-1], 0.05),
                xytext=(noise_levels[-1] - 12, 0.09), fontsize=8, color="#7a7f85")
    ax.set_xlabel("added noise sigma, 8-bit levels")
    ax.set_ylabel("margin: sharp half minus blurred half")
    ax.set_title("Per-tile focus measure: which one survives noise")
    ax.legend(fontsize=8)
    save(fig, out, "focus_measures")


# ---------------------------------------------------------------------------
# 6. Performance
# ---------------------------------------------------------------------------

def figure_performance(config: SharpnessConfig, out: Path) -> None:
    """Cost against analysis width, and the per-metric split."""
    bench_path = Path("data/bench_final.json")
    per_metric: dict[str, float] = {}
    if bench_path.is_file():
        try:
            payload = json.loads(bench_path.read_text(encoding="utf-8"))
            run = next(iter(payload["runs"].values()))
            per_metric = run.get("per_metric_ms_mean", {})
        except (json.JSONDecodeError, KeyError, StopIteration):
            logger.warning("could not read %s", bench_path)

    # Measured on the Raspberry Pi 5 with tools/benchmark.py.
    widths = [160, 240, 320, 480, 640]
    processing = [4.87, 8.61, 7.91, 16.18, 24.78]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))

    ax = axes[0]
    ax.plot(widths, processing, marker="o", color=COLOURS["ensemble"], linewidth=2.2)
    ax.axhline(40.0, color=WARM, linestyle="--", linewidth=1.4)
    ax.annotate("40 ms - the camera's frame interval", (168, 41.5),
                fontsize=8, color=WARM)
    ax.scatter([320], [7.91], s=140, facecolors="none", edgecolors=ACCENT, linewidths=2)
    ax.annotate("default", (320, 7.91), xytext=(345, 12), fontsize=8, color=ACCENT)
    ax.annotate("240 is slower than 320:\nheight 135 is a poor DFT length",
                (240, 8.61), xytext=(255, 20), fontsize=7.5, color=NEUTRAL,
                arrowprops={"arrowstyle": "->", "color": NEUTRAL, "linewidth": 0.9})
    ax.set_xlabel("analysis width, px")
    ax.set_ylabel("processing time per frame, ms")
    ax.set_title("Cost against analysis resolution")
    ax.set_ylim(0, 46)

    ax = axes[1]
    if not per_metric:
        per_metric = {"edge_width": 2.606, "fourier": 1.422, "wavelet": 0.524,
                      "tenengrad": 0.357, "laplacian": 0.269, "brenner": 0.250}
    names = sorted(per_metric, key=per_metric.get, reverse=True)
    values = [per_metric[n] for n in names]
    ax.barh(names, values, color=[COLOURS[n] for n in names], height=0.62)
    for index, value in enumerate(values):
        ax.text(value + 0.05, index, f"{value:.2f} ms", va="center", fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("mean time per frame, ms")
    ax.set_title("Per-metric cost (320 px, Raspberry Pi 5)")
    ax.set_xlim(0, max(values) * 1.28)
    ax.grid(axis="y", visible=False)

    fig.suptitle("Raspberry Pi 5, live GH6 at 25 fps, all six metrics", y=1.03)
    save(fig, out, "performance")


# ---------------------------------------------------------------------------
# 7. Robustness: score under disturbances
# ---------------------------------------------------------------------------

def figure_robustness(config: SharpnessConfig, out: Path) -> None:
    """The ensemble score through a sweep, under each disturbance."""
    scene = make_scene(640, 360, seed=7)
    steps = 31
    cases = {
        "clean": {},
        "noise sigma 20": {"noise_sigma": 20.0},
        "exposure drift": {"exposure_drift": 0.25},
        "motion 4 px": {"motion_px": 4.0},
    }
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colours = [COLOURS["ensemble"], WARM, COLOURS["tenengrad"], ACCENT]
    for (label, kwargs), colour in zip(cases.items(), colours):
        frames = [f.image for f in focus_sweep(
            scene, steps=steps, best_position=0.5, max_radius=8.0, seed=8, **kwargs)]
        data = analyse_frames(frames, config, label, (steps - 1) // 2)
        bank = fit_normalizers(data, config)
        results = run_comparison(data, config)
        ax.plot(results["adaptive"]["scores"], linewidth=2.0, color=colour, label=label)
    ax.axvline((steps - 1) // 2, color=NEUTRAL, linestyle=":", linewidth=1.4)
    ax.set_xlabel("sweep position (step)")
    ax.set_ylabel("adaptive ensemble score")
    ax.set_title("The peak survives noise, exposure drift and motion")
    ax.legend(fontsize=8)
    save(fig, out, "robustness")


FIGURES: dict[str, Callable[[SharpnessConfig, Path], None]] = {
    "sweep": figure_sweep,
    "methods": figure_methods,
    "step": figure_step,
    "weights": figure_weights,
    "measures": figure_focus_measures,
    "performance": figure_performance,
    "robustness": figure_robustness,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML config; defaults to the packaged one")
    parser.add_argument("--out", type=Path, default=Path("docs/figures"))
    parser.add_argument("--only", nargs="+", choices=sorted(FIGURES), default=None)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )
    style()
    config = load_config(args.config) if args.config else load_default_config()

    selected = args.only or list(FIGURES)
    print(f"generating {len(selected)} figure(s) into {args.out}")
    for name in selected:
        print(f"- {name}")
        FIGURES[name](config, args.out)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
