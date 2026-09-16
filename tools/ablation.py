"""Ablation: measure each mechanism separately, on the real recordings.

Every variant runs the same code and differs only in configuration.  Each one
states **every** switch that can change a measurement, rather than inheriting
whatever the shipped defaults happen to be: when `use_agreement` moved to
`false`, inherited variants silently collapsed onto each other and `adaptive`
became a duplicate of `no_agreement` with nothing in the output to show it.

    python3 tools/ablation.py data --group fixes --json data/ablation_fixes.json
    python3 tools/ablation.py data --group factorial --json data/factorial.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import SharpnessConfig, load_default_config  # noqa: E402
from adaptive_sharpness.config import METRIC_NAMES  # noqa: E402
from tools.evaluation import (  # noqa: E402
    SPOT_PARAMETERS,
    load_measure_cache,
    ordering_against_truth,
    peak_interval,
    provenance,
    select_frames,
)
from tools.protocol_study import (  # noqa: E402
    _average_ranks,
    adjacent_discrimination,
    count_peaks,
    monotonicity,
    saturated_fraction,
    step_profile,
)
from tools.replay_configs import (  # noqa: E402
    load_context,
    load_frames,
    replay_in_context,
)

logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Variants, stated in full
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VariantSpec:
    """Every switch a variant sets.

    Nothing here is inherited from the shipped configuration, so a change of
    default cannot silently turn two variants into the same experiment.  The
    spec is written into the report next to the numbers it produced.
    """

    # normalisation
    window: int
    min_range_absolute: float
    long_window_multiple: int
    mapping: str
    auto_freeze: bool
    ready_informative_fraction: float
    # analysis
    noise_quantile: float
    edge_ref_density: float
    # ensemble
    use_agreement: bool
    agreement_scale: float
    use_reliability: bool
    equal_priors: bool
    # temporal
    temporal_enabled: bool
    confidence_coupling: bool
    # pipeline / metrics
    analysis_width: int
    enabled: tuple[str, ...]

    def build(self, base: SharpnessConfig) -> SharpnessConfig:
        sensitivity = base.ensemble.sensitivity
        if not self.use_reliability:
            sensitivity = {
                name: {key: 0.0 for key in row}
                for name, row in sensitivity.items()
            }
        priors = base.metrics.priors
        if self.equal_priors:
            priors = {name: 1.0 for name in priors}
        return replace(
            base,
            pipeline=replace(base.pipeline, analysis_width=self.analysis_width),
            metrics=replace(base.metrics, enabled=self.enabled, priors=priors),
            normalization=replace(
                base.normalization,
                window=self.window,
                min_range_absolute=self.min_range_absolute,
                long_window_multiple=self.long_window_multiple,
                mapping=self.mapping,
                auto_freeze=self.auto_freeze,
                auto_thaw=False,
                ready_informative_fraction=self.ready_informative_fraction,
            ),
            analysis=replace(
                base.analysis,
                noise_quantile=self.noise_quantile,
                edge_ref_density=self.edge_ref_density,
            ),
            ensemble=replace(
                base.ensemble,
                sensitivity=sensitivity,
                use_agreement=self.use_agreement,
                agreement_scale=self.agreement_scale,
            ),
            temporal=replace(
                base.temporal,
                enabled=self.temporal_enabled,
                confidence_coupling=self.confidence_coupling,
            ),
        )


def spec_from_config(config: SharpnessConfig) -> VariantSpec:
    """Read a VariantSpec back out of a configuration.

    So that "what ships" is taken from the library rather than restated in a
    second place.  A hand-written copy drifts: `fair_comparison` compared a
    "shipped" pipeline that still had the agreement kernel on, months after the
    default turned it off.
    """
    return VariantSpec(
        window=config.normalization.window,
        min_range_absolute=config.normalization.min_range_absolute,
        long_window_multiple=config.normalization.long_window_multiple,
        mapping=config.normalization.mapping,
        auto_freeze=config.normalization.auto_freeze,
        ready_informative_fraction=config.normalization.ready_informative_fraction,
        noise_quantile=config.analysis.noise_quantile,
        edge_ref_density=config.analysis.edge_ref_density,
        use_agreement=config.ensemble.use_agreement,
        agreement_scale=config.ensemble.agreement_scale,
        use_reliability=any(
            value
            for row in config.ensemble.sensitivity.values()
            for value in row.values()
        ),
        equal_priors=len(set(config.metrics.priors.values())) == 1,
        temporal_enabled=config.temporal.enabled,
        confidence_coupling=config.temporal.confidence_coupling,
        analysis_width=config.pipeline.analysis_width,
        enabled=tuple(config.metrics.enabled),
    )


#: The behaviour that shipped before the protocol study.
HISTORIC = VariantSpec(
    window=120,
    min_range_absolute=1.0,
    long_window_multiple=0,
    mapping="linear",
    auto_freeze=False,
    ready_informative_fraction=0.0,
    noise_quantile=50.0,
    edge_ref_density=0.03,
    use_agreement=True,
    agreement_scale=1.5,
    use_reliability=True,
    equal_priors=False,
    temporal_enabled=True,
    confidence_coupling=True,
    analysis_width=320,
    enabled=METRIC_NAMES,
)

#: The pipeline after the repairs, with the weighting model left exactly as it
#: was - agreement kernel included - so that the normalisation question and the
#: adaptivity question stay separate.
#:
#: This is deliberately NOT the shipped configuration, which also has the
#: agreement kernel off.  "all repairs" answers "what did fixing the
#: normalisation buy", and the factorial answers the rest; conflating them
#: would credit the repairs with a change the factorial made.
REPAIRED = replace(
    HISTORIC,
    window=960,
    min_range_absolute=1e-6,
    long_window_multiple=2,
    mapping="logistic",
    auto_freeze=True,
    ready_informative_fraction=0.5,
    noise_quantile=75.0,
    edge_ref_density=0.006,
)

_SEVEN = METRIC_NAMES + ("gradient_variance",)

FIX_VARIANTS: dict[str, VariantSpec] = {
    "baseline": HISTORIC,
    "range": replace(HISTORIC, min_range_absolute=1e-6),
    "horizon": replace(HISTORIC, long_window_multiple=8),
    "noise": replace(HISTORIC, noise_quantile=75.0),
    "edges": replace(HISTORIC, edge_ref_density=0.006),
    "ready": replace(HISTORIC, ready_informative_fraction=0.5),
    "window": replace(HISTORIC, window=960),
    "freeze": replace(HISTORIC, window=960, auto_freeze=True),
    "logistic": replace(HISTORIC, mapping="logistic"),
    "freeze+logistic": replace(
        HISTORIC, window=960, auto_freeze=True, mapping="logistic"
    ),
    "all": REPAIRED,
}

NORMALISATION_VARIANTS: dict[str, VariantSpec] = {
    "rolling+linear": replace(REPAIRED, auto_freeze=False, mapping="linear"),
    "rolling+logistic": replace(REPAIRED, auto_freeze=False, mapping="logistic"),
    "frozen+linear": replace(REPAIRED, auto_freeze=True, mapping="linear"),
    "frozen+logistic": REPAIRED,
    "short+rolling+linear": replace(
        REPAIRED, window=120, auto_freeze=False, mapping="linear"
    ),
    "short+frozen+logistic": replace(REPAIRED, window=120),
}

NOISE_VARIANTS: dict[str, VariantSpec] = {
    f"q{int(q)}": replace(REPAIRED, noise_quantile=q)
    for q in (50.0, 60.0, 75.0, 90.0, 95.0)
}

EDGE_VARIANTS: dict[str, VariantSpec] = {
    f"edge{v:g}": replace(REPAIRED, edge_ref_density=v)
    for v in (0.003, 0.006, 0.012, 0.02, 0.03)
}

METRIC_VARIANTS: dict[str, VariantSpec] = {
    "six": REPAIRED,
    "seven": replace(REPAIRED, enabled=_SEVEN),
    "w240": replace(REPAIRED, analysis_width=240),
    "w480": replace(REPAIRED, analysis_width=480),
    "w640": replace(REPAIRED, analysis_width=640),
}
#: Leave-one-out over the shipped six, to test each metric's contribution on
#: the repaired pipeline rather than on the broken one.
METRIC_VARIANTS.update({
    f"without_{name}": replace(
        REPAIRED, enabled=tuple(n for n in METRIC_NAMES if n != name)
    )
    for name in METRIC_NAMES
})

#: All eight combinations of the three mechanisms, so interactions are visible
#: rather than inferred from three one-at-a-time rows.
FACTORIAL_VARIANTS: dict[str, VariantSpec] = {
    f"{'R' if reliability else '-'}{'A' if agreement else '-'}{'F' if filt else '-'}":
        replace(
            REPAIRED,
            use_reliability=reliability,
            use_agreement=agreement,
            temporal_enabled=filt,
        )
    for reliability in (True, False)
    for agreement in (True, False)
    for filt in (True, False)
}
FACTORIAL_VARIANTS["RAF+nogate"] = replace(REPAIRED, confidence_coupling=False)
FACTORIAL_VARIANTS["plain_mean"] = replace(
    REPAIRED, use_reliability=False, use_agreement=False, equal_priors=True
)

#: How aggressive the consensus kernel is.  If the kernel is merely mistuned,
#: widening it should find a better setting than "off"; if it is the wrong idea
#: on this data, the result should approach "off" monotonically.
AGREEMENT_VARIANTS: dict[str, VariantSpec] = {
    "off": replace(REPAIRED, use_agreement=False),
    **{
        f"scale{scale:g}": replace(REPAIRED, use_agreement=True, agreement_scale=scale)
        for scale in (1.0, 1.5, 2.5, 4.0, 8.0)
    },
}

VARIANT_GROUPS: dict[str, dict[str, VariantSpec]] = {
    "agreement": AGREEMENT_VARIANTS,
    "fixes": FIX_VARIANTS,
    "normalisation": NORMALISATION_VARIANTS,
    "noise": NOISE_VARIANTS,
    "edges": EDGE_VARIANTS,
    "metrics": METRIC_VARIANTS,
    "factorial": FACTORIAL_VARIANTS,
}


def check_variants_differ(variants: dict[str, VariantSpec]) -> list[str]:
    """Report any two variants that are the same experiment under two names."""
    seen: dict[tuple, str] = {}
    duplicates = []
    for name, spec in variants.items():
        key = tuple(sorted(asdict(spec).items(), key=lambda kv: kv[0]))
        if key in seen:
            duplicates.append(f"{name} is identical to {seen[key]}")
        else:
            seen[key] = name
    return duplicates


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, without pulling in SciPy."""
    finite = np.isfinite(a) & np.isfinite(b)
    if finite.sum() < 4:
        return float("nan")
    ranks_a = _average_ranks(a[finite])
    ranks_b = _average_ranks(b[finite])
    ranks_a = ranks_a - ranks_a.mean()
    ranks_b = ranks_b - ranks_b.mean()
    denominator = float(np.sqrt((ranks_a ** 2).sum() * (ranks_b ** 2).sum()))
    if denominator <= 0:
        return float("nan")
    return float((ranks_a * ranks_b).sum() / denominator)


def ground_truth_scores(
    series: np.ndarray, radius: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Agreement with the physical spot size, which uses no focus measure."""
    truth = -radius  # smaller spot = sharper
    out: dict[str, float] = {"spearman_gt": spearman(series, truth)}

    steps, model_medians, _ = step_profile(series, labels)
    _, truth_medians, _ = step_profile(truth, labels)
    out.update(ordering_against_truth(model_medians, truth_medians).to_dict())

    model_first, model_last = peak_interval(model_medians)
    truth_first, truth_last = peak_interval(truth_medians)
    if model_first < 0 or truth_first < 0:
        out["peak_err"] = float("nan")
        out["peak_plateau"] = float("nan")
        return out
    model_centre = 0.5 * (steps[model_first] + steps[model_last])
    truth_centre = 0.5 * (steps[truth_first] + steps[truth_last])
    out["peak_err"] = abs(float(model_centre) - float(truth_centre))
    out["peak_plateau"] = float(steps[model_last] - steps[model_first] + 1)
    out["reference_plateau"] = float(steps[truth_last] - steps[truth_first] + 1)
    return out


def evaluate(
    output: dict[str, np.ndarray],
    selected: np.ndarray,
    labels: np.ndarray,
    radius: np.ndarray | None,
) -> dict[str, float]:
    filtered = output["filtered"][selected]
    steps, medians, scatters = step_profile(filtered, labels)
    peak_first, peak_last = peak_interval(medians)
    row = {
        "adj": adjacent_discrimination(filtered, labels),
        "adj_instant": adjacent_discrimination(
            output["instantaneous"][selected], labels
        ),
        "sat": saturated_fraction(filtered),
        "mono": monotonicity(medians, peak_first),
        "peaks": float(count_peaks(medians, scatters)),
        "peak_plateau_steps": float(
            steps[peak_last] - steps[peak_first] + 1 if peak_first >= 0 else np.nan
        ),
        "sentinel": float(np.mean(output["informative"][selected] < 0.5)),
        "confidence_zero": float(np.mean(output["confidence"][selected] <= 1e-9)),
        "ready": float(np.mean(output["ready"][selected])),
        "noise_zero": float(np.mean(output["noise_sigma"][selected] <= 1e-9)),
        "ms": float(np.median(output["elapsed"]) * 1000.0),
        "ms_p95": float(np.percentile(output["elapsed"], 95) * 1000.0),
        "ms_p99": float(np.percentile(output["elapsed"], 99) * 1000.0),
    }
    if radius is not None:
        row.update(ground_truth_scores(filtered, radius[selected], labels))
        instant = ground_truth_scores(
            output["instantaneous"][selected], radius[selected], labels
        )
        row.update({f"{k}_instant": v for k, v in instant.items()})
    return row


def alternative_reference(
    directory: Path, files: Sequence[str], variant_name: str
) -> tuple[np.ndarray, np.ndarray] | None:
    """Recompute the spot reference with a different measurement recipe.

    The reference has free parameters, and the variants split into two groups
    that are *anti*-correlated with each other (see
    tools/reference_sensitivity.py).  A comparison between two models is only
    meaningful if it survives changing which member of the coherent group is
    used, so the choice is exposed rather than hard-coded.
    """
    from tools.reference_sensitivity import SpotVariant, measure

    fraction, half, background = variant_name.split("_")
    spot = SpotVariant(
        fraction=int(fraction[1:]) / 100.0, half=int(half[1:]), background=background
    )
    radii, centres = [], []
    for name in files:
        image = cv2.imread(str(directory / "frames" / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            return None
        radius, _, _ = measure(image, spot)
        radii.append(radius)
        blurred = cv2.GaussianBlur(image, (9, 9), 0)
        _, _, _, peak = cv2.minMaxLoc(blurred)
        centres.append(peak)
    centre_array = np.asarray(centres, dtype=float)
    offset = np.linalg.norm(centre_array - np.median(centre_array, axis=0), axis=1)
    return np.asarray(radii, dtype=float), offset


def run(
    directories: Sequence[Path],
    config: SharpnessConfig,
    variants: dict[str, VariantSpec],
    reference_variant: str = "r50_w80_median",
) -> dict[str, Any]:
    duplicates = check_variants_differ(variants)
    if duplicates:
        raise ValueError(
            "variants must differ; " + "; ".join(duplicates)
        )

    results: dict[str, Any] = {
        "runs": {},
        "variants": {name: asdict(spec) for name, spec in variants.items()},
        "selection": {},
    }
    for directory in directories:
        context = load_context(directory)
        files, step, hold = context.files, context.step, context.hold

        cached = load_measure_cache(directory, files)
        radius, offset = cached.spot_radius, cached.spot_offset
        reference_note = cached.note
        if radius is None:
            logger.info("%s: no usable reference - %s", directory.name, cached.note)

        if radius is not None and reference_variant != "r50_w80_median":
            logger.info("%s: recomputing reference as %s",
                        directory.name, reference_variant)
            alternative = alternative_reference(directory, files, reference_variant)
            if alternative is None:
                radius = offset = None
                reference_note = "alternative reference could not be computed"
            else:
                radius, offset = alternative
                reference_note = f"spot size, recipe {reference_variant}"

        selection = select_frames(
            step=step, hold=hold, spot_radius=radius, spot_offset=offset
        )
        if selection.count < 20:
            logger.info("%s: too few measurable frames, skipped", directory.name)
            continue
        selected = selection.mask
        labels = step[selected]
        results["selection"][directory.name] = {
            **selection.summary(),
            "context": context.summary(),
            # Every number below is recomputed from the frames by a fresh
            # evaluator, not the score the recorder wrote at capture time.
            "source": "recomputed",
            "reference": reference_note,
        }

        logger.info("%s: decoding %d frames", directory.name, len(files))
        images = load_frames(directory, files)
        entry: dict[str, Any] = {}
        for name, spec in variants.items():
            logger.info("%s: %s", directory.name, name)
            variant_config = spec.build(config)
            output = replay_in_context(images, context, variant_config)
            entry[name] = evaluate(output, selected, labels, radius)
        results["runs"][directory.name] = entry
        del images

    results["aggregate"] = aggregate(results["runs"], variants)
    return results


def aggregate(
    runs: dict[str, Any], variants: dict[str, VariantSpec]
) -> dict[str, Any]:
    if not runs:
        return {}
    fields = sorted({f for entry in runs.values() for row in entry.values() for f in row})
    summary: dict[str, Any] = {}
    for variant in variants:
        summary[variant] = {}
        for field_name in fields:
            values = [
                runs[r][variant][field_name] for r in runs
                if field_name in runs[r][variant]
                and np.isfinite(runs[r][variant][field_name])
            ]
            summary[variant][field_name] = (
                float(np.mean(values)) if values else float("nan")
            )
            # Spread across recordings, so a mean is never read as a constant.
            summary[variant][f"{field_name}_spread"] = (
                float(np.max(values) - np.min(values)) if len(values) > 1 else 0.0
            )
            summary[variant][f"{field_name}_n"] = len(values)
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(results: dict[str, Any]) -> None:
    summary = results["aggregate"]
    meta = results.get("provenance", {})
    print("=" * 100)
    print(f"ABLATION [{results.get('group', '?')}] - one code path, "
          "every switch stated per variant")
    print("=" * 100)
    print(f"\ncommit {meta.get('commit', '?')[:12]}"
          f"{' (dirty)' if meta.get('working_tree_dirty') else ''}   "
          f"numpy {meta.get('numpy', '?')}   opencv {meta.get('opencv', '?')}")
    print(f"{len(results['runs'])} recordings\n")

    for name, info in results.get("selection", {}).items():
        excluded = ", ".join(f"{k} {v}" for k, v in info["excluded"].items()) or "none"
        print(f"  {name:<32} {info['selected']:>5} of {info['total_frames']:>5}"
              f"   excluded: {excluded}")

    header = (
        f"\n{'variant':<22}{'adj':>7}{'sat':>7}{'mono':>7}{'sentinel':>10}"
        f"{'conf=0':>8}{'ready':>7}{'noise=0':>9}{'ms':>7}{'p95':>7}"
    )
    print(header)
    print("-" * (len(header) - 1))
    for name, row in summary.items():
        print(
            f"{name:<22}{row['adj']:>7.3f}{row['sat']:>7.3f}{row['mono']:>7.3f}"
            f"{row['sentinel']:>10.3f}{row['confidence_zero']:>8.3f}"
            f"{row['ready']:>7.3f}{row['noise_zero']:>9.3f}{row['ms']:>7.2f}"
            f"{row['ms_p95']:>7.2f}"
        )

    if any(np.isfinite(r.get("spearman_gt", np.nan)) for r in summary.values()):
        header = (
            f"\n{'variant':<22}{'spearman':>10}{'+/-spread':>9}{'inversion':>11}"
            f"{'strict':>8}{'resolved':>10}{'peak err':>10}{'plateau':>9}"
        )
        print(header)
        print("-" * (len(header) - 1))
        for name, row in summary.items():
            if not np.isfinite(row.get("spearman_gt", float("nan"))):
                continue
            print(
                f"{name:<22}{row['spearman_gt']:>10.3f}"
                f"{row['spearman_gt_spread']:>9.3f}{row['inversion']:>11.3f}"
                f"{row['inversion_strict']:>8.3f}{row['resolved']:>10.3f}"
                f"{row['peak_err']:>10.2f}{row['peak_plateau']:>9.1f}"
            )

    print(f"\n--- Adjacent-step discrimination per recording ---\n")
    variants = list(results["variants"])
    header = f"{'recording':<32}" + "".join(f"{v[:11]:>12}" for v in variants)
    print(header)
    print("-" * len(header))
    for run_name, entry in results["runs"].items():
        cells = "".join(f"{entry[v]['adj']:>12.3f}" for v in variants)
        print(f"{run_name:<32}{cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument(
        "--group", choices=sorted(VARIANT_GROUPS), default="fixes",
        help="which set of variants to compare",
    )
    parser.add_argument(
        "--reference", default="r50_w80_median",
        help="how to measure the point-source reference, e.g. r25_w50_median",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )
    directories = sorted(
        d for d in args.root.iterdir()
        if d.is_dir() and (d / "frames.csv").exists() and (d / "frames").is_dir()
    )
    if args.only:
        directories = [d for d in directories if any(f in d.name for f in args.only)]
    if not directories:
        parser.error(f"no recordings found under {args.root}")

    config = load_default_config()
    variants = VARIANT_GROUPS[args.group]
    results = run(directories, config, variants, reference_variant=args.reference)
    results["group"] = args.group
    results["reference_variant"] = args.reference
    results["provenance"] = provenance(
        config,
        group=args.group,
        reference_variant=args.reference,
        recordings=[d.name for d in directories],
    )
    # Write the result before rendering it.  Printing a table is the cheapest
    # step and the only one that can fail on a terminal encoding - losing an
    # hour of replay to a character that would not encode is not acceptable.
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(results, indent=2, default=float), encoding="utf-8"
        )
        print(f"wrote {args.json}")
    print_report(results)
    if args.figures:
        from tools.ablation_figures import make_figures

        make_figures(results, args.figures)
        print(f"wrote figures to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
