"""Two comparisons that must not be mixed up.

A ranking of *measures* and a ranking of *systems* answer different questions,
and this project has already been misled once by reading one as the other.

Signals
    The raw measure, with no normalisation, no history and no filter, scored
    directly against the point-source reference.  This says how much focus
    information the measure carries.

    A raw measure is perfectly computable online - nothing about `wavelet`
    needs a future frame - so this is not an offline-only category.  What it
    lacks is a bounded scale, comparability between scenes, a confidence, and
    anything to fuse; whether those are worth paying for is the question the
    next section answers, not one this section settles.

    VOL4 keeps its negative values here: clamping it at zero, which the
    library's metric contract would do, turns it into a different method, and
    it is negative on 40-85% of these frames.

Pipelines
    The full streaming system, which is what a search would actually consume.
    Every entry gets the *same* normalisation, the same history and the same
    temporal filter, so the only difference is what is being fused.  Each
    single measure appears as a pipeline too, so that "what does the apparatus
    cost" is answered by comparing like with like rather than by comparing a
    pipeline against a raw number.

    python3 tools/fair_comparison.py data --json reports/fair_comparison.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import SharpnessConfig, load_default_config  # noqa: E402
from tools.ablation import (  # noqa: E402
    REPAIRED,
    VariantSpec,
    ground_truth_scores,
    spec_from_config,
)
from tools.evaluation import (  # noqa: E402
    load_measure_cache,
    provenance,
    select_frames,
)
from tools.protocol_study import adjacent_discrimination  # noqa: E402
from tools.replay_configs import (  # noqa: E402
    load_context,
    load_frames,
    replay_in_context,
)

logger = logging.getLogger("fair_comparison")

#: Raw signals to score directly, by their name in the measure cache.
RAW_SIGNALS: tuple[str, ...] = (
    "wavelet", "tenengrad", "brenner", "laplacian", "fourier", "edge_width",
    "TENV", "VOL4", "SML", "SMD", "GLVA", "NVAR", "LAPE", "VOL5", "DCTR",
)

#: Single measures that also get a full pipeline, so that the cost of the
#: apparatus is measured against the same measure rather than against a
#: different one.  `wavelet` matters most: it is the raw signal that agrees
#: best with the reference, so leaving its pipeline out of the table would make
#: any statement about what the pipeline costs unsupported.
SINGLE_PIPELINES: tuple[str, ...] = (
    "wavelet", "tenengrad", "brenner", "laplacian", "gradient_variance",
)


def build_pipelines(config: SharpnessConfig) -> dict[str, tuple[VariantSpec, str]]:
    """Every streaming pipeline in the comparison, and what it is.

    ``six:shipped`` is read back out of the library's own configuration rather
    than restated here, so it cannot drift from what actually ships.  An earlier
    version of this table hard-coded a base with the consensus kernel still on,
    months after the default turned it off, and therefore understated the
    shipped pipeline.
    """
    shipped = spec_from_config(config)
    pipelines: dict[str, tuple[VariantSpec, str]] = {
        "six:shipped": (shipped, "what this library ships"),
        "six:with_kernel": (REPAIRED, "as shipped, consensus kernel on"),
        "six:fixed_weights": (
            replace(shipped, use_reliability=False, use_agreement=False),
            "prior weights only",
        ),
        "six:plain_mean": (
            replace(
                shipped, use_reliability=False, use_agreement=False,
                equal_priors=True,
            ),
            "unweighted mean of the six",
        ),
    }
    for name in SINGLE_PIPELINES:
        pipelines[f"single:{name}"] = (
            replace(shipped, enabled=(name,)),
            "one measure, same normaliser and filter",
        )
    return pipelines


def analyse(directory: Path, config: SharpnessConfig) -> dict[str, Any] | None:
    context = load_context(directory)
    cached = load_measure_cache(directory, context.files)
    if not cached.has_reference:
        logger.info("%s: no usable reference - %s", directory.name, cached.note)
        return None
    radius, offset = cached.spot_radius, cached.spot_offset

    selection = select_frames(
        step=context.step, hold=context.hold,
        spot_radius=radius, spot_offset=offset, require_reference=True,
    )
    if selection.count < 20:
        return None
    mask = selection.mask
    labels = context.step[mask]
    truth = radius[mask]

    out: dict[str, Any] = {
        "selection": selection.summary(),
        "context": context.summary(),
        "cache": cached.note,
        "signals": {},
        "pipelines": {},
    }

    raw_cache = np.load(cached.path, allow_pickle=False)
    for name in RAW_SIGNALS:
        key = f"raw_{name}"
        if key not in raw_cache.files:
            continue
        series = raw_cache[key][mask].astype(float)
        scores = ground_truth_scores(series, truth, labels)
        scores["adj"] = adjacent_discrimination(series, labels)
        scores["negative_fraction"] = float(np.mean(series < 0.0))
        out["signals"][name] = scores

    images = load_frames(directory, context.files)
    for name, (spec, note) in build_pipelines(config).items():
        logger.info("%s: %s", directory.name, name)
        variant = spec.build(config)
        result = replay_in_context(images, context, variant)
        series = result["filtered"][mask]
        scores = ground_truth_scores(series, truth, labels)
        scores["adj"] = adjacent_discrimination(series, labels)
        scores["ms"] = float(np.median(result["elapsed"]) * 1000.0)
        scores["note"] = note
        out["pipelines"][name] = scores
    del images
    return out


def aggregate(runs: dict[str, Any], section: str) -> dict[str, Any]:
    names: set[str] = set()
    for entry in runs.values():
        names.update(entry[section])
    summary: dict[str, Any] = {}
    for name in names:
        rows = [
            entry[section][name] for entry in runs.values() if name in entry[section]
        ]
        fields = {f for row in rows for f in row if not isinstance(row[f], str)}
        summary[name] = {
            f: (
                float(np.mean([r[f] for r in rows if np.isfinite(r.get(f, np.nan))]))
                if any(np.isfinite(r.get(f, np.nan)) for r in rows)
                else float("nan")
            )
            for f in fields
        }
        summary[name]["runs"] = len(rows)
        note = next((r.get("note") for r in rows if r.get("note")), "")
        if note:
            summary[name]["note"] = note
    return summary


def print_section(title: str, summary: dict[str, Any], extra: str | None = None) -> None:
    print(f"\n--- {title} ---\n")
    order = sorted(
        summary,
        key=lambda n: (
            -summary[n]["spearman_gt"]
            if np.isfinite(summary[n]["spearman_gt"]) else 1e9
        ),
    )
    header = (
        f"{'name':<26}{'spearman':>10}{'inversion':>11}{'strict':>8}"
        f"{'resolved':>10}{'adj':>7}{'peak err':>10}"
    )
    if extra:
        header += f"{extra:>9}"
    header += "  note"
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
        print(line + "  " + str(row.get("note", "")))


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

    config = load_default_config()
    runs: dict[str, Any] = {}
    for directory in sorted(args.root.iterdir()):
        if not (directory.is_dir() and (directory / "frames.csv").exists()):
            continue
        entry = analyse(directory, config)
        if entry:
            runs[directory.name] = entry
    if not runs:
        parser.error("no recordings with a usable point-source reference")

    results = {
        "runs": runs,
        "signals": aggregate(runs, "signals"),
        "pipelines": aggregate(runs, "pipelines"),
        "provenance": provenance(config, recordings=list(runs)),
    }

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        print(f"wrote {args.json}")

    print("=" * 96)
    print("FAIR COMPARISON - raw signals and full pipelines, against the spot reference")
    print("=" * 96)
    print(f"\n{len(runs)} recordings with a usable reference\n")
    for name, entry in runs.items():
        info = entry["selection"]
        print(f"  {name:<32} {info['selected']} of {info['total_frames']} frames"
              f"   cache: {entry['cache']}")
    print_section(
        "Raw signals: no normalisation, no history, no filter", results["signals"]
    )
    print_section(
        "Streaming pipelines: same normaliser, history and filter",
        results["pipelines"], extra="ms",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
