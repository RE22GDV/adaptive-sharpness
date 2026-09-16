"""Figures for the ablation.

Measurements only - no image content, so every figure here is safe to publish
even though the recordings themselves are private.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

#: Columns worth plotting, and whether more is better.
_PANELS: tuple[tuple[str, str, bool], ...] = (
    ("adj", "adjacent-step discrimination", True),
    ("sentinel", "frames on the 0.5 sentinel", False),
    ("confidence_zero", "frames with zero confidence", False),
    ("sat", "frames pinned at the extremes", False),
)

_GROUP_TITLE = {
    "fixes": "Each repair on its own, against what shipped before",
    "noise": "Where to read the noise estimator",
    "edges": "Edge-sufficiency reference",
    "fusion": "Does the adaptivity earn its keep?",
    "metrics": "Metric set and analysis resolution",
}


def make_figures(results: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    group = results.get("group", "fixes")
    _panels(results, out_dir, plt, group)
    if any("spearman_gt" in row for row in results["aggregate"].values()):
        _ground_truth(results, out_dir, plt, group)


def _panels(results: dict[str, Any], out_dir: Path, plt, group: str) -> None:
    summary = results["aggregate"]
    names = list(summary)
    figure, axes = plt.subplots(2, 2, figsize=(12, 7.5))

    for index, (field, label, higher_is_better) in enumerate(_PANELS):
        axis = axes[index // 2][index % 2]
        values = [summary[n].get(field, np.nan) for n in names]
        if not np.any(np.isfinite(values)):
            axis.axis("off")
            continue
        best = (
            int(np.nanargmax(values)) if higher_is_better else int(np.nanargmin(values))
        )
        colours = ["#BBBBBB"] * len(names)
        colours[best] = "#4C72B0" if higher_is_better else "#55A868"
        if names and names[0] in ("baseline", "six", "adaptive"):
            colours[0] = "#C44E52"
        axis.bar(range(len(names)), values, color=colours)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axis.set_title(
            f"{label}  ({'higher' if higher_is_better else 'lower'} is better)",
            fontsize=10,
        )
        axis.grid(axis="y", alpha=0.3)
        for position, value in enumerate(values):
            if np.isfinite(value):
                axis.text(position, value, f"{value:.3f}", ha="center",
                          va="bottom", fontsize=7)

    figure.suptitle(
        f"{_GROUP_TITLE.get(group, group)}\n"
        f"{len(results['runs'])} recordings, replayed through the shipped evaluator",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    figure.savefig(out_dir / f"ablation_{group}.png", dpi=130)
    plt.close(figure)


def _ground_truth(results: dict[str, Any], out_dir: Path, plt, group: str) -> None:
    """Agreement with the physical spot size, which uses no focus measure."""
    summary = results["aggregate"]
    names = [n for n in summary if np.isfinite(summary[n].get("spearman_gt", np.nan))]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    for axis, field, label, higher in (
        (axes[0], "spearman_gt", "rank correlation with spot size", True),
        (axes[1], "inversion", "step pairs ordered the wrong way", False),
    ):
        values = [summary[n][field] for n in names]
        best = int(np.argmax(values)) if higher else int(np.argmin(values))
        colours = ["#BBBBBB"] * len(names)
        colours[best] = "#4C72B0"
        if names and names[0] in ("baseline", "six", "adaptive"):
            colours[0] = "#C44E52"
        axis.bar(range(len(names)), values, color=colours)
        axis.set_xticks(range(len(names)))
        axis.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        axis.set_title(
            f"{label}  ({'higher' if higher else 'lower'} is better)", fontsize=10
        )
        axis.grid(axis="y", alpha=0.3)
        for position, value in enumerate(values):
            axis.text(position, value, f"{value:.3f}", ha="center", va="bottom",
                      fontsize=7)

    figure.suptitle(
        "Against the point-source ground truth (two recordings, 15 steps each)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(out_dir / f"ablation_{group}_truth.png", dpi=130)
    plt.close(figure)
