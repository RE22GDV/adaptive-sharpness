"""Two comparisons that must not be mixed up.

A ranking of *measures* and a ranking of *systems* answer different questions,
and this project has already been misled once by reading one as the other.

Signals
    The raw measure, with no normalisation, no history and no filter, scored
    directly against the point-source reference.  This says how much focus
    information the measure carries.  VOL4 keeps its negative values here:
    clamping it at zero - which the library's metric contract would do - turns
    it into a different method, and it is negative on 40-85% of these frames.

Pipelines
    The full streaming system, which is what a search would actually consume.
    Every entry gets the *same* normalisation, the same history, and the same
    temporal filter, so the only difference is what is being fused.

    python3 tools/fair_comparison.py data --json data/fair_comparison.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import load_default_config  # noqa: E402
from adaptive_sharpness.config import METRIC_NAMES  # noqa: E402
from tools.ablation import REPAIRED, ground_truth_scores  # noqa: E402
from tools.evaluation import provenance, select_frames  # noqa: E402
from tools.protocol_study import SPOT_PARAMETERS, adjacent_discrimination  # noqa: E402
from tools.replay_configs import load_context, load_frames, replay  # noqa: E402

logger = logging.getLogger("fair_comparison")

#: Raw signals to score directly, by their name in the measure cache.
RAW_SIGNALS: tuple[str, ...] = (
    "tenengrad", "brenner", "laplacian", "wavelet", "fourier", "edge_width",
    "TENV", "VOL4", "SML", "SMD", "GLVA", "NVAR", "LAPE", "VOL5", "DCTR",
)

#: Streaming pipelines.  Single measures use the library's own metric set of
#: one, so they inherit exactly the normaliser and filter the ensemble uses.
PIPELINES: dict[str, dict[str, Any]] = {
    "single:tenengrad": {"enabled": ("tenengrad",)},
    "single:brenner": {"enabled": ("brenner",)},
    "single:gradient_variance": {"enabled": ("gradient_variance",)},
    "six:adaptive": {},
    "six:fixed_weights": {"use_reliability": False, "use_agreement": False},
    "six:plain_mean": {
        "use_reliability": False, "use_agreement": False, "equal_priors": True,
    },
}


def load_reference(directory: Path, frames: int) -> tuple[np.ndarray, np.ndarray] | None:
    cache = directory / "measures_cache.npz"
    if not cache.exists():
        return None
    loaded = np.load(cache)
    if "spot_radius" not in loaded or loaded["spot_radius"].size != frames:
        return None
    expected = json.dumps(SPOT_PARAMETERS, sort_keys=True)
    if "spot_fingerprint" not in loaded or str(loaded["spot_fingerprint"]) != expected:
        logger.warning("%s: spot cache is stale or unfingerprinted", directory.name)
        return None
    return loaded["spot_radius"], loaded["spot_offset"]


def analyse(directory: Path) -> dict[str, Any] | None:
    context = load_context(directory)
    reference = load_reference(directory, len(context.files))
    if reference is None:
        return None
    radius, offset = reference

    selection = select_frames(
        step=context.step, hold=context.hold,
        spot_radius=radius, spot_offset=offset, require_reference=True,
    )
    if selection.count < 20:
        return None
    mask = selection.mask
    labels = context.step[mask]
    truth = radius[mask]

    cache = np.load(directory / "measures_cache.npz")
    out: dict[str, Any] = {
        "selection": selection.summary(),
        "signals": {},
        "pipelines": {},
    }

    for name in RAW_SIGNALS:
        key = f"raw_{name}"
        if key not in cache:
            continue
        series = cache[key][mask].astype(float)
        scores = ground_truth_scores(series, truth, labels)
        scores["adj"] = adjacent_discrimination(series, labels)
        scores["negative_fraction"] = float(np.mean(series < 0.0))
        out["signals"][name] = scores

    config = load_default_config()
    images = load_frames(directory, context.files)
    for name, overrides in PIPELINES.items():
        logger.info("%s: %s", directory.name, name)
        spec = REPAIRED
        for field_name, value in overrides.items():
            spec = spec.__class__(**{**spec.__dict__, field_name: value})
        variant = spec.build(config)
        result = replay(images, variant, variant.metrics.enabled)
        series = result["filtered"][mask]
        scores = ground_truth_scores(series, truth, labels)
        scores["adj"] = adjacent_discrimination(series, labels)
        scores["ms"] = float(np.median(result["elapsed"]) * 1000.0)
        out["pipelines"][name] = scores
    del images
    return out


def aggregate(runs: dict[str, Any], section: str) -> dict[str, Any]:
    names: set[str] = set()
    for entry in runs.values():
        names.update(entry[section])
    summary: dict[str, Any] = {}
    for name in names:
        rows = [entry[section][name] for entry in runs.values() if name in entry[section]]
        fields = {f for row in rows for f in row}
        summary[name] = {
            f: float(np.mean([r[f] for r in rows if np.isfinite(r.get(f, np.nan))]))
            if any(np.isfinite(r.get(f, np.nan)) for r in rows) else float("nan")
            for f in fields
        }
        summary[name]["runs"] = len(rows)
    return summary


def print_section(title: str, summary: dict[str, Any], extra: str | None = None) -> None:
    print(f"\n--- {title} ---\n")
    order = sorted(
        summary,
        key=lambda n: -summary[n]["spearman_gt"]
        if np.isfinite(summary[n]["spearman_gt"]) else 1e9,
    )
    header = (f"{'name':<26}{'spearman':>10}{'inversion':>11}{'strict':>8}"
              f"{'resolved':>10}{'adj':>7}{'peak err':>10}")
    if extra:
        header += f"{extra:>9}"
    print(header)
    print("-" * len(header))
    for name in order:
        row = summary[name]
        line = (
            f"{name:<26}{row['spearman_gt']:>10.3f}{row['inversion']:>11.3f}"
            f"{row['inversion_strict']:>8.3f}{row['resolved']:>10.3f}"
            f"{row['adj']:>7.3f}{row['peak_err']:>10.2f}"
        )
        if extra:
            value = row.get(extra, float("nan"))
            line += f"{value:>9.2f}" if np.isfinite(value) else f"{'-':>9}"
        print(line)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    runs: dict[str, Any] = {}
    for directory in sorted(args.root.iterdir()):
        if not (directory.is_dir() and (directory / "frames.csv").exists()):
            continue
        entry = analyse(directory)
        if entry:
            runs[directory.name] = entry
    if not runs:
        parser.error("no recordings with a usable point-source reference")

    results = {
        "runs": runs,
        "signals": aggregate(runs, "signals"),
        "pipelines": aggregate(runs, "pipelines"),
        "provenance": provenance(load_default_config(), recordings=list(runs)),
    }

    print("=" * 86)
    print("FAIR COMPARISON - raw signals and full pipelines, against the spot reference")
    print("=" * 86)
    print(f"\n{len(runs)} recordings with a usable reference\n")
    for name, entry in runs.items():
        info = entry["selection"]
        print(f"  {name:<32} {info['selected']} of {info['total_frames']} frames")
    print_section("Raw signals: no normalisation, no history, no filter",
                  results["signals"])
    print_section("Streaming pipelines: same normaliser, history and filter",
                  results["pipelines"], extra="ms")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
