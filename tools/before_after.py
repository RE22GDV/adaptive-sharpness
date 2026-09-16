"""Before and after, against the point-source ground truth.

One figure, one question: does the score reported by the library track the
physical size of a defocused point?  The spot radius uses no focus measure, so
it is the only reference in this project that cannot be accused of agreeing
with the thing it is checking.

    python3 tools/before_after.py data --figures docs/figures
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import load_default_config  # noqa: E402
from tools.ablation import HISTORIC, REPAIRED, ground_truth_scores, spearman  # noqa: E402
from tools.evaluation import provenance, select_frames  # noqa: E402
from tools.protocol_study import step_profile  # noqa: E402
from tools.replay_configs import load_frames, load_labels, replay  # noqa: E402

logger = logging.getLogger("before_after")


def collect(directory: Path) -> dict[str, Any] | None:
    cache = directory / "measures_cache.npz"
    if not cache.exists():
        return None
    loaded = np.load(cache)
    if "spot_radius" not in loaded or loaded["spot_radius"].size == 0:
        return None

    files, step, hold = load_labels(directory)
    radius = loaded["spot_radius"]
    offset = loaded["spot_offset"]
    # The same selection every other tool uses, so that a number here can be
    # compared with a number there.
    selection = select_frames(
        step=step, hold=hold, spot_radius=radius, spot_offset=offset,
        require_reference=True,
    )
    if selection.count < 20:
        return None
    selected = selection.mask
    labels = step[selected]

    config = load_default_config()
    images = load_frames(directory, files)
    out: dict[str, Any] = {"name": directory.name, "selection": selection.summary()}

    steps, spot_medians, _ = step_profile(radius[selected], labels)
    out["steps"] = steps.tolist()
    out["spot_radius"] = spot_medians.tolist()
    out["best_step"] = float(steps[int(np.argmin(spot_medians))])

    for label, spec in (("before", HISTORIC), ("after", REPAIRED)):
        logger.info("%s: %s", directory.name, label)
        cfg = spec.build(config)
        result = replay(images, cfg, cfg.metrics.enabled)
        series = result["filtered"][selected]
        _, medians, _ = step_profile(series, labels)
        scores = ground_truth_scores(series, radius[selected], labels)
        out[label] = {
            "profile": medians.tolist(),
            "spearman": scores["spearman_gt"],
            "inversion": scores["inversion"],
            "peak_step": float(steps[int(np.nanargmax(medians))]),
            "peak_err": scores["peak_err"],
            "spec": {k: (list(v) if isinstance(v, tuple) else v)
                     for k, v in spec.__dict__.items()},
        }
    del images
    return out


def make_figure(runs: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, len(runs), figsize=(6.0 * len(runs), 4.4), squeeze=False)
    for index, run in enumerate(runs):
        axis = axes[0][index]
        axis.plot(run["steps"], run["spot_radius"], color="#222222", linewidth=2.4,
                  marker="o", markersize=4, label="spot radius r50 (ground truth)")
        axis.axvline(run["best_step"], color="#222222", linestyle="--", alpha=0.5)
        axis.set_xlabel("protocol step")
        axis.set_ylabel("encircled-energy radius, px")
        axis.grid(alpha=0.3)
        axis.invert_yaxis()

        twin = axis.twinx()
        for label, colour in (("before", "#C44E52"), ("after", "#4C72B0")):
            series = np.asarray(run[label]["profile"], dtype=float)
            span = np.nanmax(series) - np.nanmin(series)
            if span > 0:
                series = (series - np.nanmin(series)) / span
            twin.plot(run["steps"], series, color=colour, linewidth=1.8, marker=".",
                      label=f"{label}  (rank corr {run[label]['spearman']:+.2f})")
        twin.set_ylabel("reported score, rescaled")

        lines = axis.get_lines()[:1] + twin.get_lines()
        axis.legend(lines, [line.get_label() for line in lines], fontsize=8,
                    loc="lower center")
        axis.set_title(
            f"{run['name'][-6:]}   sharpest at step {run['best_step']:.0f}",
            fontsize=10,
        )
    figure.suptitle(
        "Does the reported score follow the physical spot size?\n"
        "(radius axis inverted, so up is sharper on both scales)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.9))
    figure.savefig(out_dir / "before_after_ground_truth.png", dpi=130)
    plt.close(figure)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    runs = []
    for directory in sorted(args.root.iterdir()):
        if not (directory.is_dir() and (directory / "frames.csv").exists()):
            continue
        collected = collect(directory)
        if collected:
            runs.append(collected)
    if not runs:
        parser.error("no point-source recordings with a measure cache found")

    print(f"{'recording':<32}{'before':>10}{'after':>10}{'best step':>11}"
          f"{'peak before':>13}{'peak after':>12}")
    print("-" * 88)
    for run in runs:
        print(
            f"{run['name']:<32}{run['before']['spearman']:>10.3f}"
            f"{run['after']['spearman']:>10.3f}{run['best_step']:>11.0f}"
            f"{run['before']['peak_step']:>13.0f}{run['after']['peak_step']:>12.0f}"
        )

    if args.json:
        payload = {
            "runs": runs,
            "provenance": provenance(
                load_default_config(), recordings=[r["name"] for r in runs]
            ),
        }
        args.json.write_text(json.dumps(payload, indent=2, default=float))
        print(f"\nwrote {args.json}")
    if args.figures:
        make_figure(runs, args.figures)
        print(f"wrote figure to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
