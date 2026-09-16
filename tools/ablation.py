"""Ablation: measure each proposed fix separately, on the real recordings.

Every variant here runs the *same code* and differs only in configuration, so
a difference between two rows is caused by the setting and not by a code
version.  The baseline variant reproduces the behaviour this project shipped
before the protocol study, which is why each defect it targets is reachable
from configuration at all.

The defects being tested, all of them measured rather than supposed:

range      The degenerate-range guard used floor = ratio * max(|hi|, |lo|, 1.0).
           The 1.0 made it absolute at 1e-3, while the entire value of
           `brenner` at defocus is 2.9e-4, so four metrics of six returned the
           0.5 sentinel on every defocused frame.
horizon    A held focus position has no range of its own.  Falling back to a
           longer history scales the frame against the sweep instead of
           refusing to answer with a mid-range 0.5 that outranks genuine low
           scores.
noise      58-74% of the Haar HH coefficients of this stream are exactly zero,
           so the median is zero and the noise estimate is zero on 100% of real
           frames - which disabled the noise branch of the weighting model
           entirely.
edges      edge_ref_density was 0.03; no recording reaches it at defocus and
           four of eight never reach it at all, so edge sufficiency - a
           necessary confidence factor - was zero on 28-78% of frames.
ready      `ready` required one informative metric out of six.

    python3 tools/ablation.py data --json data/ablation.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import SharpnessConfig, load_default_config  # noqa: E402
from adaptive_sharpness.config import METRIC_NAMES  # noqa: E402
from tools.protocol_study import (  # noqa: E402
    _average_ranks,
    adjacent_discrimination,
    count_peaks,
    monotonicity,
    saturated_fraction,
    step_profile,
)
from tools.replay_configs import load_frames, load_labels, replay  # noqa: E402

logger = logging.getLogger("ablation")


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def _shipped_before(config: SharpnessConfig) -> SharpnessConfig:
    """Exactly the behaviour that shipped before this study."""
    return replace(
        config,
        normalization=replace(
            config.normalization,
            window=120,
            min_range_absolute=1.0,
            long_window_multiple=0,
            ready_informative_fraction=0.0,
            mapping="linear",
            auto_freeze=False,
        ),
        analysis=replace(config.analysis, noise_quantile=50.0, edge_ref_density=0.03),
    )


_SECTIONS = ("normalization", "analysis", "pipeline", "metrics")


def _with(config: SharpnessConfig, **changes: Any) -> SharpnessConfig:
    """Set fields by name, routed to whichever config section owns them."""
    routed: dict[str, dict[str, Any]] = {name: {} for name in _SECTIONS}
    for key, value in changes.items():
        owners = [s for s in _SECTIONS if hasattr(getattr(config, s), key)]
        if not owners:
            raise KeyError(f"unknown setting: {key}")
        if len(owners) > 1:
            raise KeyError(f"ambiguous setting {key}: owned by {', '.join(owners)}")
        routed[owners[0]][key] = value
    return replace(config, **{
        section: replace(getattr(config, section), **values)
        for section, values in routed.items() if values
    })


Builder = Callable[[SharpnessConfig], SharpnessConfig]

_ALL_FIXES = dict(
    window=960,
    min_range_absolute=1e-6,
    long_window_multiple=2,
    noise_quantile=75.0,
    edge_ref_density=0.006,
    ready_informative_fraction=0.5,
    mapping="logistic",
    auto_freeze=True,
)


def _fixed(config: SharpnessConfig, **overrides: Any) -> SharpnessConfig:
    """The repaired pipeline, optionally with one more setting changed."""
    return _with(_shipped_before(config), **{**_ALL_FIXES, **overrides})


#: One setting at a time, against the behaviour that shipped before the study.
FIX_VARIANTS: dict[str, Builder] = {
    "baseline": _shipped_before,
    "range": lambda c: _with(_shipped_before(c), min_range_absolute=1e-6),
    "horizon": lambda c: _with(_shipped_before(c), long_window_multiple=8),
    "range+horizon": lambda c: _with(
        _shipped_before(c), min_range_absolute=1e-6, long_window_multiple=8
    ),
    "noise": lambda c: _with(_shipped_before(c), noise_quantile=75.0),
    "edges": lambda c: _with(_shipped_before(c), edge_ref_density=0.006),
    "ready": lambda c: _with(_shipped_before(c), ready_informative_fraction=0.5),
    "window": lambda c: _with(_shipped_before(c), window=960),
    "freeze": lambda c: _with(_shipped_before(c), window=960, auto_freeze=True),
    "logistic": lambda c: _with(_shipped_before(c), mapping="logistic"),
    "freeze+logistic": lambda c: _with(
        _shipped_before(c), auto_freeze=True, mapping="logistic"
    ),
    "all": _fixed,
}

#: Where to read the noise estimator, on top of the repaired pipeline.  The
#: median is the classical choice and reports exactly zero on this stream; the
#: higher quantiles trade that blindness for a floor set by image structure.
NOISE_VARIANTS: dict[str, Builder] = {
    f"q{int(q)}": (lambda q: lambda c: _fixed(c, noise_quantile=q))(q)
    for q in (50.0, 60.0, 75.0, 90.0, 95.0)
}

#: The edge-sufficiency reference, which gates a necessary confidence factor.
EDGE_VARIANTS: dict[str, Builder] = {
    f"edge{value:g}": (lambda v: lambda c: _fixed(c, edge_ref_density=v))(value)
    for value in (0.003, 0.006, 0.012, 0.02, 0.03)
}


def _ensemble(config: SharpnessConfig, **changes: Any) -> SharpnessConfig:
    return replace(config, ensemble=replace(config.ensemble, **changes))


def _flat_sensitivity(config: SharpnessConfig) -> SharpnessConfig:
    """Zero every kappa, which turns the reliability stage into a constant."""
    flat = {
        name: {key: 0.0 for key in row}
        for name, row in config.ensemble.sensitivity.items()
    }
    return _ensemble(config, sensitivity=flat)


def _equal_priors(config: SharpnessConfig) -> SharpnessConfig:
    return replace(
        config,
        metrics=replace(
            config.metrics,
            priors={name: 1.0 for name in config.metrics.priors},
        ),
    )


#: Does the adaptivity earn its keep?  Each variant removes one mechanism the
#: project claims as its contribution, leaving everything else repaired.
FUSION_VARIANTS: dict[str, Builder] = {
    "adaptive": _fixed,
    "no_agreement": lambda c: _ensemble(_fixed(c), use_agreement=False),
    "no_reliability": lambda c: _flat_sensitivity(_fixed(c)),
    "fixed_weights": lambda c: _flat_sensitivity(
        _ensemble(_fixed(c), use_agreement=False)
    ),
    "plain_mean": lambda c: _equal_priors(
        _flat_sensitivity(_ensemble(_fixed(c), use_agreement=False))
    ),
    "no_filter": lambda c: replace(
        _fixed(c), temporal=replace(_fixed(c).temporal, enabled=False)
    ),
    "no_confidence_gate": lambda c: replace(
        _fixed(c), temporal=replace(_fixed(c).temporal, confidence_coupling=False)
    ),
}

#: Does a seventh metric help, and does a larger analysis image?  Both cost
#: time, so both have to earn it.  gradient_variance (TENV) separated adjacent
#: focus steps better than any of the project's own six in the offline study.
_SEVEN = METRIC_NAMES + ("gradient_variance",)

METRIC_VARIANTS: dict[str, Builder] = {
    "six": _fixed,
    "seven": lambda c: _with(_fixed(c), enabled=_SEVEN),
    "seven_w480": lambda c: _with(_fixed(c), enabled=_SEVEN, analysis_width=480),
    "six_w480": lambda c: _with(_fixed(c), analysis_width=480),
    "six_w640": lambda c: _with(_fixed(c), analysis_width=640),
    "six_w240": lambda c: _with(_fixed(c), analysis_width=240),
}

#: How long a memory should the normaliser have?
#:
#: The rolling window makes the score relative to recent history, so on a slow
#: sweep the score answers "is this sharper than the last few seconds" rather
#: than "how sharp is this".  Measured on the point-source runs: the score peaks
#: two thirds of the way through the approach and then collapses to 0.006 at the
#: step where the spot is physically smallest.  A longer memory should turn the
#: score back into a fraction of the best sharpness seen this session, which is
#: what a search actually needs.
WINDOW_VARIANTS: dict[str, Builder] = {
    f"w{int(n)}": (lambda n: lambda c: _fixed(c, window=n, long_window_multiple=0))(n)
    for n in (120, 300, 600, 1200, 2400)
}
#: The same, keeping the fallback horizon on top of the longer window.
WINDOW_VARIANTS["w120+horizon"] = lambda c: _fixed(c, window=120)
WINDOW_VARIANTS["w600+horizon"] = lambda c: _fixed(c, window=600)

VARIANT_GROUPS: dict[str, dict[str, Builder]] = {
    "window": WINDOW_VARIANTS,
    "metrics": METRIC_VARIANTS,
    "fixes": FIX_VARIANTS,
    "noise": NOISE_VARIANTS,
    "edges": EDGE_VARIANTS,
    "fusion": FUSION_VARIANTS,
}

#: Set by main() to the group under test; the report and aggregation read it.
VARIANTS: dict[str, Builder] = FIX_VARIANTS


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
    """Agreement with the physical spot size, which uses no focus measure.

    ``inversion`` is the fraction of step pairs the model orders the wrong way
    round against the spot: it is the direct measurement of the failure where a
    fully defocused frame scores above a mildly defocused one.
    """
    truth = -radius  # smaller spot = sharper
    out = {"spearman_gt": spearman(series, truth)}

    steps, model_medians, _ = step_profile(series, labels)
    _, truth_medians, _ = step_profile(truth, labels)
    wrong = total = 0
    for i in range(steps.size):
        for j in range(i + 1, steps.size):
            if abs(truth_medians[i] - truth_medians[j]) < 1e-12:
                continue
            total += 1
            sharper_is_i = truth_medians[i] > truth_medians[j]
            model_says_i = model_medians[i] > model_medians[j]
            if sharper_is_i != model_says_i:
                wrong += 1
    out["inversion"] = wrong / total if total else float("nan")
    out["peak_err"] = abs(
        float(steps[int(np.argmax(model_medians))])
        - float(steps[int(np.argmax(truth_medians))])
    )
    return out


def evaluate(
    output: dict[str, np.ndarray],
    selected: np.ndarray,
    labels: np.ndarray,
    radius: np.ndarray | None,
) -> dict[str, float]:
    filtered = output["filtered"][selected]
    steps, medians, scatters = step_profile(filtered, labels)
    row = {
        "adj": adjacent_discrimination(filtered, labels),
        "adj_instant": adjacent_discrimination(
            output["instantaneous"][selected], labels
        ),
        "sat": saturated_fraction(filtered),
        "mono": monotonicity(medians, int(np.argmax(medians))),
        "peaks": float(count_peaks(medians, scatters)),
        "sentinel": float(np.mean(output["informative"][selected] < 0.5)),
        "confidence_zero": float(np.mean(output["confidence"][selected] <= 1e-9)),
        "ready": float(np.mean(output["ready"][selected])),
        "noise_zero": float(np.mean(output["noise_sigma"][selected] <= 1e-9)),
        "ms": float(np.median(output["elapsed"]) * 1000.0),
    }
    if radius is not None:
        row.update(ground_truth_scores(filtered, radius[selected], labels))
    return row


def run(directories: Sequence[Path], config: SharpnessConfig) -> dict[str, Any]:
    results: dict[str, Any] = {"runs": {}, "variants": {}}
    for directory in directories:
        files, step, hold = load_labels(directory)
        selected = hold & (step > 0)
        if selected.sum() < 20:
            continue
        labels = step[selected]

        radius = None
        cache = directory / "measures_cache.npz"
        if cache.exists():
            loaded = np.load(cache)
            if "spot_radius" in loaded and loaded["spot_radius"].size == len(files):
                radius = loaded["spot_radius"]

        logger.info("%s: decoding %d frames", directory.name, len(files))
        images = load_frames(directory, files)
        entry: dict[str, Any] = {}
        for name, build in VARIANTS.items():
            logger.info("%s: %s", directory.name, name)
            variant_config = build(config)
            output = replay(
                images, variant_config, variant_config.metrics.enabled
            )
            entry[name] = evaluate(output, selected, labels, radius)
        results["runs"][directory.name] = entry
        del images

    results["aggregate"] = aggregate(results["runs"])
    results["variants"] = list(VARIANTS)
    return results


def aggregate(runs: dict[str, Any]) -> dict[str, Any]:
    if not runs:
        return {}
    fields = sorted({f for entry in runs.values() for row in entry.values() for f in row})
    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        summary[variant] = {}
        for field in fields:
            values = [
                runs[r][variant][field] for r in runs
                if field in runs[r][variant] and np.isfinite(runs[r][variant][field])
            ]
            summary[variant][field] = float(np.mean(values)) if values else float("nan")
    return summary


def print_report(results: dict[str, Any]) -> None:
    summary = results["aggregate"]
    print("=" * 96)
    print(f"ABLATION [{results.get('group', 'fixes')}] - one code path, "
          "one setting changed at a time")
    print("=" * 96)
    print(f"\n{len(results['runs'])} recordings\n")

    header = (
        f"{'variant':<16}{'adj':>7}{'sat':>7}{'mono':>7}{'sentinel':>10}"
        f"{'conf=0':>8}{'ready':>7}{'noise=0':>9}{'ms':>7}"
    )
    print(header)
    print("-" * len(header))
    for name, row in summary.items():
        print(
            f"{name:<16}{row['adj']:>7.3f}{row['sat']:>7.3f}{row['mono']:>7.3f}"
            f"{row['sentinel']:>10.3f}{row['confidence_zero']:>8.3f}"
            f"{row['ready']:>7.3f}{row['noise_zero']:>9.3f}{row['ms']:>7.2f}"
        )

    print("\n--- Against the physical spot-size ground truth (point-source runs) ---\n")
    header = f"{'variant':<16}{'spearman':>10}{'inversion':>11}{'peak_err':>10}"
    print(header)
    print("-" * len(header))
    for name, row in summary.items():
        if not np.isfinite(row.get("spearman_gt", float("nan"))):
            continue
        print(
            f"{name:<16}{row['spearman_gt']:>10.3f}{row['inversion']:>11.3f}"
            f"{row['peak_err']:>10.2f}"
        )

    print("\n--- Adjacent-step discrimination per recording (filtered score) ---\n")
    variants = list(VARIANTS)
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
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    global VARIANTS
    VARIANTS = VARIANT_GROUPS[args.group]

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

    results = run(directories, load_default_config())
    results["group"] = args.group
    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")
    if args.figures:
        from tools.ablation_figures import make_figures

        make_figures(results, args.figures)
        print(f"wrote figures to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
