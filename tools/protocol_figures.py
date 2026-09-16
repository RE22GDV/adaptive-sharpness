"""Figures for the protocol study.

Kept separate from :mod:`tools.protocol_study` so that the study runs without
matplotlib, which is not a dependency of the library.

Nothing here plots image content.  The recordings are photographs of wherever
the camera happened to be pointing, so every figure is built from measurements
only and is safe to publish.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

_KIND_COLOUR = {
    "measure": "#4C72B0",
    "fusion": "#DD8452",
    "online": "#55A868",
}

#: Models drawn as step profiles.  A curated few, or the plot is unreadable.
_PROFILE_MODELS = (
    "single:VOL4",
    "single:TENV",
    "single:tenengrad",
    "own6:adaptive",
    "own6:mean",
    "online:filtered",
)


def _kind(model: str) -> str:
    if model.startswith("single:"):
        return "measure"
    if model.startswith("online:"):
        return "online"
    return "fusion"


def _short(model: str) -> str:
    return model.split(":", 1)[1] if model.startswith("single:") else model


def make_figures(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    _ranking(results, out_dir, plt)
    _profiles(results, out_dir, plt)
    _ground_truth(results, out_dir, plt)
    _saturation(results, out_dir, plt)
    _per_run_heatmap(results, out_dir, plt)


def _ranking(results: dict[str, Any], out_dir: Path, plt) -> None:
    rows = results["aggregate"]
    order = sorted(
        (n for n in rows if np.isfinite(rows[n]["adj"])),
        key=lambda n: rows[n]["adj"],
    )
    values = [rows[n]["adj"] for n in order]
    colours = [_KIND_COLOUR[_kind(n)] for n in order]

    figure, axes = plt.subplots(figsize=(9, 11))
    axes.barh(range(len(order)), values, color=colours)
    axes.set_yticks(range(len(order)))
    axes.set_yticklabels([_short(n) for n in order], fontsize=8)
    axes.set_xlabel("adjacent-step discrimination  |2·AUC − 1|,  mean over recordings")
    axes.set_title(
        "Can the model tell one focus step from the next?\n"
        f"{len(results['runs'])} recordings, exact protocol labels",
        fontsize=11,
    )
    axes.set_xlim(0, 1)
    axes.grid(axis="x", alpha=0.3)
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colour) for colour in _KIND_COLOUR.values()
    ]
    axes.legend(handles, _KIND_COLOUR, loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(out_dir / "protocol_ranking.png", dpi=130)
    plt.close(figure)


def _profiles(results: dict[str, Any], out_dir: Path, plt) -> None:
    runs = {
        name: entry for name, entry in results["runs"].items()
        if entry["profiles"]
    }
    if not runs:
        return
    columns = min(len(runs), 4)
    rows_count = int(np.ceil(len(runs) / columns))
    figure, axes = plt.subplots(
        rows_count, columns, figsize=(4.2 * columns, 3.2 * rows_count), squeeze=False
    )
    for index, (name, entry) in enumerate(runs.items()):
        axis = axes[index // columns][index % columns]
        steps = entry["step_values"]
        for model in _PROFILE_MODELS:
            profile = entry["profiles"].get(model)
            if not profile:
                continue
            series = np.asarray(profile, dtype=float)
            span = np.nanmax(series) - np.nanmin(series)
            if span > 0:
                series = (series - np.nanmin(series)) / span
            axis.plot(steps, series, marker="o", markersize=3, linewidth=1.2,
                      label=_short(model))
        axis.set_title(f"{entry['preset']}\n{name[-6:]}", fontsize=9)
        axis.set_xlabel("protocol step")
        axis.set_ylabel("rescaled to [0,1]")
        axis.grid(alpha=0.3)
        axis.tick_params(labelsize=8)
    for index in range(len(runs), rows_count * columns):
        axes[index // columns][index % columns].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=9)
    figure.suptitle("Step profiles: each model over the swept focus range", fontsize=12)
    figure.tight_layout(rect=(0, 0.06, 1, 0.97))
    figure.savefig(out_dir / "protocol_profiles.png", dpi=130)
    plt.close(figure)


def _ground_truth(results: dict[str, Any], out_dir: Path, plt) -> None:
    point_runs = {
        name: entry for name, entry in results["runs"].items()
        if entry.get("spot_profile")
    }
    if not point_runs:
        return
    figure, axes = plt.subplots(
        1, len(point_runs), figsize=(5.2 * len(point_runs), 4.0), squeeze=False
    )
    for index, (name, entry) in enumerate(point_runs.items()):
        axis = axes[0][index]
        spot = entry["spot_profile"]
        axis.plot(spot["steps"], spot["radius_px"], marker="o", color="#333333",
                  linewidth=2, label="spot radius r50 (ground truth)")
        best = spot["steps"][int(np.argmin(spot["radius_px"]))]
        axis.axvline(best, color="#333333", linestyle="--", alpha=0.6)
        axis.set_xlabel("protocol step")
        axis.set_ylabel("encircled-energy radius, px")
        axis.grid(alpha=0.3)

        twin = axis.twinx()
        for model in ("single:VOL4", "own6:adaptive", "online:filtered"):
            profile = entry["profiles"].get(model)
            if not profile:
                continue
            series = np.asarray(profile, dtype=float)
            span = np.nanmax(series) - np.nanmin(series)
            if span > 0:
                series = (series - np.nanmin(series)) / span
            twin.plot(entry["step_values"], series, alpha=0.75, linewidth=1.2,
                      marker=".", label=_short(model))
        twin.set_ylabel("model score, rescaled")
        axis.set_title(f"{name[-6:]}   best focus at step {best:.0f}", fontsize=10)
        lines = axis.get_lines()[:1] + twin.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=8,
                    loc="upper left")
    figure.suptitle(
        "Point source: spot size is the only signal here that uses no focus measure",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(out_dir / "protocol_ground_truth.png", dpi=130)
    plt.close(figure)


def _saturation(results: dict[str, Any], out_dir: Path, plt) -> None:
    rows = results["aggregate"]
    names = [n for n in rows if np.isfinite(rows[n]["sat"]) and np.isfinite(rows[n]["sep"])]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for name in names:
        axes[0].scatter(rows[name]["sat"], rows[name]["adj"],
                        color=_KIND_COLOUR[_kind(name)], s=28)
    axes[0].set_xlabel("fraction of frames pinned at the signal's own extremes")
    axes[0].set_ylabel("adjacent-step discrimination")
    axes[0].set_title("Saturation costs discrimination", fontsize=10)
    axes[0].grid(alpha=0.3)

    for name in names:
        axes[1].scatter(rows[name]["sat"], rows[name]["sep"],
                        color=_KIND_COLOUR[_kind(name)], s=28)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("fraction of frames pinned at the signal's own extremes")
    axes[1].set_ylabel("between/within separability  (log scale)")
    axes[1].set_title(
        "...but inflates the separability ratio the earlier study ranked on",
        fontsize=10,
    )
    axes[1].grid(alpha=0.3)

    handles = [
        plt.Line2D([], [], marker="o", linestyle="", color=colour, label=kind)
        for kind, colour in _KIND_COLOUR.items()
    ]
    figure.legend(handles=handles, loc="lower center", ncol=3, fontsize=9)
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    figure.savefig(out_dir / "protocol_saturation.png", dpi=130)
    plt.close(figure)


def _per_run_heatmap(results: dict[str, Any], out_dir: Path, plt) -> None:
    runs = results["runs"]
    aggregate = results["aggregate"]
    models = sorted(
        (n for n in aggregate if np.isfinite(aggregate[n]["adj"])),
        key=lambda n: -aggregate[n]["adj"],
    )
    run_names = list(runs)
    grid = np.full((len(models), len(run_names)), np.nan)
    for row, model in enumerate(models):
        for column, run_name in enumerate(run_names):
            entry = runs[run_name]["models"].get(model)
            if entry and np.isfinite(entry["adj"]):
                grid[row, column] = entry["adj"]

    figure, axes = plt.subplots(figsize=(1.5 + 0.9 * len(run_names), 0.26 * len(models) + 2))
    image = axes.imshow(grid, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes.set_yticks(range(len(models)))
    axes.set_yticklabels([_short(m) for m in models], fontsize=7)
    axes.set_xticks(range(len(run_names)))
    axes.set_xticklabels(
        [f"{runs[n]['preset']}\n{n[-6:]}" for n in run_names], fontsize=7, rotation=45,
        ha="right",
    )
    axes.set_title("Adjacent-step discrimination, per recording", fontsize=11)
    figure.colorbar(image, ax=axes, shrink=0.7, label="|2·AUC − 1|")
    figure.tight_layout()
    figure.savefig(out_dir / "protocol_per_run.png", dpi=130)
    plt.close(figure)
