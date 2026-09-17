"""Д07-Д20 of the research programme, one function each.

Every study here takes the recorded frames, replays them through the real
evaluator and writes `reports/study_dNN.json`.  None of them writes an image:
the recordings are photographs of a private room, and only measurements leave
the machine.

What each study is for is in docs/STUDY_UA_REPORT.md section 8.  What it found
is in docs/STUDY_UA_RESULTS.md.  The numbers quoted in prose are registered in
docs/claims.toml and checked against these files by tests/test_claims.py.
"""
from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adaptive_sharpness import load_default_config
from tools.programme import (
    CORPUS,
    WITH_REFERENCE,
    aggregate_variants,
    build_config,
    corpus_directories,
    load_recording,
    rich_replay,
    score_against,
    sweep,
)
from tools.protocol_study import adjacent_discrimination, step_profile
from tools.ablation import spearman

logger = logging.getLogger("studies")

SIX = ("laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width")

def no_reliability() -> dict[str, Any]:
    """Turn the reliability model off the way `tools/ablation.py` does.

    There is no `use_reliability` switch: the model is off when every
    sensitivity coefficient is zero, which makes every reliability exactly 1.
    Reproducing that here rather than inventing a second definition is what
    lets a number from this module sit next to one from the published
    factorial.
    """
    sensitivity = load_default_config().ensemble.sensitivity
    return {
        "ensemble.sensitivity": {
            name: {key: 0.0 for key in row} for name, row in sensitivity.items()
        }
    }


def flat_priors() -> dict[str, Any]:
    """Every metric equally important before conditions are considered."""
    priors = load_default_config().metrics.priors
    return {"metrics.priors": {name: 1.0 for name in priors}}


#: The three weighting schemes the report compares.  Named once here so that
#: Д07, Д12 and Д18 cannot quietly disagree about what "prior weights" means.
WEIGHTING: dict[str, dict[str, Any]] = {
    "adaptive": {"ensemble.use_agreement": False},
    "prior": {"ensemble.use_agreement": False, **no_reliability()},
    "equal": {"ensemble.use_agreement": False, **no_reliability(), **flat_priors()},
}


# ---------------------------------------------------------------------------
# Д07 - uncertainty of a paired difference
# ---------------------------------------------------------------------------

def _autocorrelation(series: np.ndarray, max_lag: int = 200) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]
    values = values - values.mean()
    if values.size < 10 or not np.any(values):
        return np.zeros(max_lag + 1)
    full = np.correlate(values, values, mode="full")[values.size - 1:]
    full = full / full[0]
    return full[:max_lag + 1]


def _first_crossing(acf: np.ndarray, threshold: float = 1.0 / np.e) -> int:
    below = np.flatnonzero(acf < threshold)
    return int(below[0]) if below.size else len(acf)


def _block_bootstrap_difference(
    a: np.ndarray, b: np.ndarray, labels: np.ndarray,
    block: int, draws: int, rng: np.random.Generator,
) -> np.ndarray:
    """Paired moving-block bootstrap of `adj(a) - adj(b)`.

    The pairing matters: both configurations saw the same frames, so resampling
    them together keeps the comparison paired and removes the between-frame
    variance that both share.  Frames inside a recording are a time series, so
    resampling single frames would treat neighbours as independent and give an
    interval several times too narrow - which is the whole reason this study
    exists rather than a t-test.
    """
    n = a.size
    if n <= block:
        return np.array([])
    starts_available = n - block + 1
    needed = int(np.ceil(n / block))
    out = np.empty(draws)
    for draw in range(draws):
        starts = rng.integers(0, starts_available, size=needed)
        index = np.concatenate([np.arange(s, s + block) for s in starts])[:n]
        out[draw] = (
            adjacent_discrimination(a[index], labels[index])
            - adjacent_discrimination(b[index], labels[index])
        )
    return out


def study_d07(data: Path, workers: int | None, draws: int = 400) -> dict[str, Any]:
    """How wide is the interval around "adaptive minus prior"?

    Section 5.3 of the report states a difference of -0.0050 in `adj` and
    +0.0145 in rank correlation without any interval at all, which makes both
    numbers descriptions rather than results.  This attaches an interval, and
    reports how it depends on the block length so the choice of block can be
    seen rather than assumed.
    """
    directories = corpus_directories(data)
    variants = [(name, overrides) for name, overrides in WEIGHTING.items()]
    runs = sweep(directories, variants, workers=workers, keep=("filtered",))

    blocks = (1, 5, 10, 25, 50, 100, 200)
    rng = np.random.default_rng(20260917)
    per_run: dict[str, Any] = {}

    for name, run in runs.items():
        series = {
            key: np.asarray(run["variants"][key]["series_filtered"], dtype=float)
            for key in WEIGHTING if key in run["variants"]
        }
        if len(series) < 2:
            continue
        recording = load_recording(data / name)
        labels = recording.labels
        difference = series["adaptive"] - series["prior"]
        acf = _autocorrelation(difference)
        entry: dict[str, Any] = {
            "frames": int(labels.size),
            "autocorrelation_lag_1": float(acf[1]) if acf.size > 1 else float("nan"),
            "decorrelation_lag": _first_crossing(acf),
            "blocks": {},
        }
        for pair in (("adaptive", "prior"), ("adaptive", "equal")):
            first, second = pair
            if first not in series or second not in series:
                continue
            point = (
                adjacent_discrimination(series[first], labels)
                - adjacent_discrimination(series[second], labels)
            )
            for block in blocks:
                draws_out = _block_bootstrap_difference(
                    series[first], series[second], labels, block, draws, rng
                )
                if draws_out.size == 0:
                    continue
                entry["blocks"].setdefault(f"{first}-{second}", {})[str(block)] = {
                    "point": float(point),
                    "low": float(np.percentile(draws_out, 2.5)),
                    "high": float(np.percentile(draws_out, 97.5)),
                    "width": float(
                        np.percentile(draws_out, 97.5) - np.percentile(draws_out, 2.5)
                    ),
                    "crosses_zero": bool(
                        np.percentile(draws_out, 2.5) < 0.0
                        < np.percentile(draws_out, 97.5)
                    ),
                }
        per_run[name] = entry

    # The corpus-level question is a difference of means across recordings, so
    # the recordings themselves are the unit and the interval comes from them.
    corpus: dict[str, Any] = {}
    for pair in ("adaptive-prior", "adaptive-equal"):
        points = [
            run["blocks"][pair]["25"]["point"]
            for run in per_run.values() if pair in run["blocks"]
        ]
        if not points:
            continue
        points = np.asarray(points)
        boot = rng.choice(points, size=(4000, points.size), replace=True).mean(axis=1)
        corpus[pair] = {
            "recordings": int(points.size),
            "mean": float(points.mean()),
            "wins": int(np.sum(points > 0)),
            "low": float(np.percentile(boot, 2.5)),
            "high": float(np.percentile(boot, 97.5)),
            "crosses_zero": bool(
                np.percentile(boot, 2.5) < 0.0 < np.percentile(boot, 97.5)
            ),
        }
    return {
        "runs": per_run,
        "corpus": corpus,
        "aggregate": aggregate_variants(runs),
        "draws": draws,
        "blocks": list(blocks),
    }


# ---------------------------------------------------------------------------
# Д09 - the normalisation, one stage at a time
# ---------------------------------------------------------------------------

def study_d09(data: Path, workers: int | None) -> dict[str, Any]:
    """What does each stage between the raw measure and the score cost?

    The report says the pipeline costs 0.023 against the best raw measure but
    never says which part of it does.  Every stage here is scored on the same
    frames with the same criterion, and stages 3 and 4 are the library's own
    output from the same replay rather than a reimplementation, so the ladder
    cannot drift from what actually ships.
    """
    base = load_default_config()
    out: dict[str, Any] = {"runs": {}}

    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        result = rich_replay(recording.images, recording.context, base)
        single = rich_replay(
            recording.images, recording.context,
            build_config(base, {"metrics.enabled": ("wavelet",)}),
        )
        mask, labels, truth = recording.mask, recording.labels, recording.truth
        names = list(result["names"])
        index = names.index("wavelet")

        raw = result["raw"][mask, index]
        stages: dict[str, np.ndarray] = {}
        stages["1_raw"] = raw
        stages["2_log"] = np.log1p(np.maximum(raw, 0.0))

        compressed = stages["2_log"]
        finite = compressed[np.isfinite(compressed)]
        low, high = (np.percentile(finite, 5), np.percentile(finite, 95)) if finite.size else (0.0, 1.0)
        span = max(high - low, 1e-12)
        stages["3_fixed_linear"] = np.clip((compressed - low) / span, 0.0, 1.0)
        centre = 0.5 * (low + high)
        gain = base.normalization.logistic_gain
        stages["4_fixed_logistic"] = 1.0 / (
            1.0 + np.exp(-np.clip(gain * (compressed - centre) / span, -60, 60))
        )
        # Stages 5 onwards are what the library produced, not a reimplementation.
        #
        # 5a exists because without it this ladder is confounded: stages 1-4
        # are `wavelet` alone, so a drop at "streaming ensemble" could be the
        # streaming normaliser or could be the other five metrics, and the two
        # would be indistinguishable.  5a is `wavelet` alone through the whole
        # streaming pipeline, which separates them.
        stages["5a_streaming_single"] = single["instantaneous"][mask]
        stages["5b_streaming_ensemble"] = result["instantaneous"][mask]
        stages["6_temporal_filter"] = result["filtered"][mask]

        rows: dict[str, Any] = {}
        previous_rank: np.ndarray | None = None
        for stage, series in stages.items():
            row = score_against(series, labels, truth)
            row["saturated"] = float(
                np.mean(
                    (series <= np.nanmin(series) + 1e-9)
                    | (series >= np.nanmax(series) - 1e-9)
                )
            )
            order = np.argsort(np.argsort(series))
            if previous_rank is not None:
                row["rank_shift_mean"] = float(np.mean(np.abs(order - previous_rank)))
                row["rank_spearman_vs_previous"] = spearman(
                    series.astype(float), previous_rank.astype(float)
                )
            previous_rank = order
            rows[stage] = row
        out["runs"][name] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "variants": rows,
        }
    out["aggregate"] = aggregate_variants(out["runs"])
    return out


# ---------------------------------------------------------------------------
# Д10 - how much of the score is the history
# ---------------------------------------------------------------------------

def study_d10(data: Path, workers: int | None) -> dict[str, Any]:
    """Does the same frame get the same score after a different history?

    The score is documented as belonging to one stream's history. That is a
    statement about the design; this measures the size of the effect. Every
    comparison is between the *same* frames, so a difference is the history and
    nothing else.
    """
    base = load_default_config()
    out: dict[str, Any] = {"runs": {}}

    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        total = len(recording.images)
        reference = rich_replay(recording.images, recording.context, base)

        variants: dict[str, dict[str, Any]] = {}
        for label, start in (
            ("start_10pct", int(total * 0.10)),
            ("start_25pct", int(total * 0.25)),
            ("start_50pct", int(total * 0.50)),
        ):
            other = rich_replay(recording.images, recording.context, base, start=start)
            variants[label] = _compare_histories(reference, other, recording)

        reversed_order = list(range(total))[::-1]
        other = rich_replay(
            recording.images, recording.context, base, order=reversed_order
        )
        variants["reversed"] = _compare_histories(reference, other, recording)

        ready = reference["ready"]
        first_ready = int(np.argmax(ready > 0)) if np.any(ready > 0) else -1
        out["runs"][name] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "frames": total,
            "first_ready_frame": first_ready,
            "variants": variants,
        }
    out["aggregate"] = aggregate_variants(out["runs"])
    return out


def _compare_histories(reference: dict, other: dict, recording) -> dict[str, float]:
    both = reference["fed"] & other["fed"] & recording.mask
    if both.sum() < 20:
        return {"comparable_frames": float(both.sum())}
    a = reference["filtered"][both]
    b = other["filtered"][both]
    labels = recording.context.step[both]
    row = {
        "comparable_frames": float(both.sum()),
        "max_abs_difference": float(np.nanmax(np.abs(a - b))),
        "mean_abs_difference": float(np.nanmean(np.abs(a - b))),
        "rank_agreement": spearman(a, b),
        "adj_reference": adjacent_discrimination(a, labels),
        "adj_variant": adjacent_discrimination(b, labels),
    }
    row["adj_change"] = row["adj_variant"] - row["adj_reference"]
    steps_a, med_a, _ = step_profile(a, labels)
    steps_b, med_b, _ = step_profile(b, labels)
    if len(steps_a) and len(steps_b):
        row["peak_moved_steps"] = float(
            abs(steps_a[int(np.argmax(med_a))] - steps_b[int(np.argmax(med_b))])
        )
    return row


# ---------------------------------------------------------------------------
# Д11 - the four constants of the freeze
# ---------------------------------------------------------------------------

def study_d11(data: Path, workers: int | None) -> dict[str, Any]:
    """Where do `auto_freeze`'s constants actually matter?

    They were fitted on the same recordings that evaluate them, which the
    report already admits. This at least shows the surface they sit on, and
    whether the freeze fires at all on a recording of ordinary length.
    """
    variants: list[tuple[str, dict[str, Any]]] = [("off", {"normalization.auto_freeze": False})]
    for minimum in (120, 240, 480):
        for patience in (60, 120, 240):
            for stability in (0.02, 0.05, 0.10):
                variants.append((
                    f"min{minimum}_pat{patience}_stab{stability:g}",
                    {
                        "normalization.auto_freeze": True,
                        "normalization.auto_freeze_min_samples": minimum,
                        "normalization.auto_freeze_patience": patience,
                        "normalization.auto_freeze_stability": stability,
                    },
                ))
    directories = corpus_directories(data)
    runs = sweep(directories, variants, workers=workers)
    return {"runs": runs, "aggregate": aggregate_variants(runs)}


# ---------------------------------------------------------------------------
# Д12 - what the weights actually do
# ---------------------------------------------------------------------------

#: The frame statistics the reliability model is supposed to react to.
DRIVERS = ("noise_level", "edge_density", "motion_px", "brightness",
           "local_contrast", "clipped_high", "clipped_low", "snr")


def study_d12(data: Path, workers: int | None) -> dict[str, Any]:
    """Section 5.3 proposed a mechanism.  This reads the weights.

    The claim was that adaptive weighting does worst on the two darkest
    recordings because the edge and exposure factors are pinned at their
    extremes there, so the model is weighting from inputs that have saturated
    and stopped carrying information.  Nothing had ever read the weights out,
    so the claim was an inference from the scores alone.
    """
    base = build_config(load_default_config(), WEIGHTING["adaptive"])
    floor = base.ensemble.weight_floor
    out: dict[str, Any] = {"runs": {}, "weight_floor": floor}

    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        result = rich_replay(recording.images, recording.context, base)
        mask = recording.mask
        names = list(result["names"])
        weight = result["weight"][mask]
        reliability = result["reliability"][mask]
        drivers = {d: result[d][mask] for d in DRIVERS}

        metrics: dict[str, Any] = {}
        for position, metric in enumerate(names):
            column = weight[:, position]
            row: dict[str, Any] = {
                "weight_mean": float(np.nanmean(column)),
                "weight_std": float(np.nanstd(column)),
                "weight_min": float(np.nanmin(column)),
                "weight_max": float(np.nanmax(column)),
                "reliability_mean": float(np.nanmean(reliability[:, position])),
                # A weight that never moves is a constant with extra steps.
                "weight_range": float(np.nanmax(column) - np.nanmin(column)),
                "correlations": {},
            }
            for driver, values in drivers.items():
                both = np.isfinite(column) & np.isfinite(values)
                row["correlations"][driver] = (
                    spearman(column[both], values[both])
                    if both.sum() > 20 and np.ptp(values[both]) > 0 else float("nan")
                )
            metrics[metric] = row

        # How saturated are the inputs the model is reading?
        pinned = {
            driver: float(np.mean(values <= np.nanmin(values) + 1e-12))
            for driver, values in drivers.items()
        }
        total = np.nansum(weight, axis=1, keepdims=True)
        share = weight / np.where(total > 0, total, np.nan)
        out["runs"][name] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "metrics": metrics,
            "drivers_at_minimum_fraction": pinned,
            "driver_means": {d: float(np.nanmean(v)) for d, v in drivers.items()},
            # One number for "did the weighting do anything at all here".
            "weight_variation": float(np.nanmean(np.nanstd(share, axis=0))),
            "max_weight_share": float(np.nanmax(np.nanmean(share, axis=0))),
            "frames": int(mask.sum()),
        }
    return out


# ---------------------------------------------------------------------------
# Д13 - is the confidence worth anything
# ---------------------------------------------------------------------------

def study_d13(data: Path, workers: int | None) -> dict[str, Any]:
    """Does discarding low-confidence frames actually reduce the error?

    The confidence is documented as an uncalibrated heuristic.  A heuristic can
    still be useful: if keeping only the most confident frames improves
    agreement with the reference, it is carrying information.  If the curve is
    flat, it is not, and the honest thing is to say so.
    """
    base = build_config(load_default_config(), WEIGHTING["adaptive"])
    out: dict[str, Any] = {"runs": {}}

    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        result = rich_replay(recording.images, recording.context, base)
        mask = recording.mask
        confidence = result["confidence"][mask]
        series = result["filtered"][mask]
        labels, truth = recording.labels, recording.truth

        curve = []
        for coverage in (1.0, 0.9, 0.75, 0.5, 0.25, 0.1):
            if coverage >= 1.0:
                keep = np.ones(confidence.size, dtype=bool)
                threshold = float(np.nanmin(confidence))
            else:
                threshold = float(np.nanquantile(confidence, 1.0 - coverage))
                keep = confidence >= threshold
            if keep.sum() < 20 or np.unique(labels[keep]).size < 3:
                continue
            point = {
                "coverage": float(keep.mean()),
                "threshold": threshold,
                "adj": adjacent_discrimination(series[keep], labels[keep]),
                "steps_remaining": int(np.unique(labels[keep]).size),
            }
            if truth is not None:
                point["spearman_gt"] = spearman(series[keep], truth[keep])
            curve.append(point)

        out["runs"][name] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "curve": curve,
            "within_step": _within_step_error_by_confidence(
                series, confidence, labels
            ),
            "confidence_mean": float(np.nanmean(confidence)),
            "confidence_zero_fraction": float(np.mean(confidence <= 1e-9)),
            # Does the confidence even vary here?  A constant cannot rank.
            "confidence_std": float(np.nanstd(confidence)),
        }
    return out


def _within_step_error_by_confidence(
    series: np.ndarray, confidence: np.ndarray, labels: np.ndarray,
) -> dict[str, Any]:
    """The coverage curve above cannot answer the question on its own.

    Two artefacts sit in it. Dropping low-confidence frames also drops whole
    protocol steps, and `adj` is defined over adjacent steps, so the criterion
    itself changes as coverage falls. And on a point source the confident
    frames are the in-focus ones, so a correlation against the reference is
    computed over a restricted range and falls for reasons that have nothing
    to do with measurement quality.

    This avoids both. Within one step the true focus is constant, so the
    spread of the score around that step's median is measurement error and
    nothing else. If the confidence carries information, high-confidence
    frames sit closer to their own step's median than low-confidence ones do.
    Comparing terciles *inside* each step keeps the step count fixed and the
    range unrestricted.
    """
    deviation = np.full(series.shape, np.nan)
    for step in np.unique(labels):
        at_step = labels == step
        if at_step.sum() < 9:
            continue
        deviation[at_step] = np.abs(series[at_step] - np.median(series[at_step]))

    usable = np.isfinite(deviation) & np.isfinite(confidence)
    if usable.sum() < 30 or np.ptp(confidence[usable]) <= 0:
        return {"usable_frames": int(usable.sum()), "separates": False}

    error, level = deviation[usable], confidence[usable]
    low, high = np.quantile(level, [1 / 3, 2 / 3])
    groups = {
        "low_confidence": error[level <= low],
        "mid_confidence": error[(level > low) & (level <= high)],
        "high_confidence": error[level > high],
    }
    out: dict[str, Any] = {
        "usable_frames": int(usable.sum()),
        "tercile_edges": [float(low), float(high)],
    }
    for label, values in groups.items():
        out[label] = float(np.mean(values)) if values.size else float("nan")
    lowest, highest = out["low_confidence"], out["high_confidence"]
    out["high_minus_low"] = highest - lowest
    # Useful means the confident frames are the accurate ones: negative.
    out["separates"] = bool(np.isfinite(highest) and np.isfinite(lowest)
                            and highest < lowest)
    out["rank_correlation_confidence_error"] = spearman(level, error)
    return out


# ---------------------------------------------------------------------------
# Д14 - the temporal filter
# ---------------------------------------------------------------------------

def study_d14(data: Path, workers: int | None) -> dict[str, Any]:
    """The filter was the largest positive contrast in the factorial.

    That measured on against off.  This measures the parameter: how much
    smoothing buys how much discrimination, and what it costs in lag when the
    focus actually moves.
    """
    variants: list[tuple[str, dict[str, Any]]] = [
        ("off", {"temporal.enabled": False, **WEIGHTING["adaptive"]}),
    ]
    for alpha in (0.1, 0.2, 0.35, 0.5, 0.75, 1.0):
        for coupling in (True, False):
            variants.append((
                f"alpha{alpha:g}_{'coupled' if coupling else 'plain'}",
                {
                    "temporal.enabled": True,
                    "temporal.alpha_base": alpha,
                    "temporal.confidence_coupling": coupling,
                    **WEIGHTING["adaptive"],
                },
            ))
    directories = corpus_directories(data)
    runs = sweep(directories, variants, workers=workers, keep=("filtered",))

    # Lag: after the operator moves to a new step, how many frames until the
    # score has travelled most of the way to that step's eventual level?
    for name, run in runs.items():
        recording = load_recording(data / name)
        labels = recording.labels
        for variant, row in run["variants"].items():
            series = np.asarray(row.pop("series_filtered"), dtype=float)
            row["settling_frames"] = _settling_frames(series, labels)
            row["hold_scatter"] = _hold_scatter(series, labels)
    return {"runs": runs, "aggregate": aggregate_variants(runs)}


def _settling_frames(series: np.ndarray, labels: np.ndarray) -> float:
    """Median frames to cover 90% of the move to a new step's median level."""
    boundaries = np.flatnonzero(np.diff(labels) != 0) + 1
    settles: list[float] = []
    for start in boundaries:
        end = start
        while end < labels.size and labels[end] == labels[start]:
            end += 1
        if end - start < 5:
            continue
        target = float(np.median(series[start:end]))
        origin = float(series[start - 1])
        if abs(target - origin) < 1e-6:
            continue
        travel = np.abs(series[start:end] - origin) / abs(target - origin)
        reached = np.flatnonzero(travel >= 0.9)
        settles.append(float(reached[0]) if reached.size else float(end - start))
    return float(np.median(settles)) if settles else float("nan")


def _hold_scatter(series: np.ndarray, labels: np.ndarray) -> float:
    """Median within-step standard deviation: the noise the filter removes."""
    scatters = [
        float(np.std(series[labels == step]))
        for step in np.unique(labels) if np.sum(labels == step) >= 5
    ]
    return float(np.median(scatters)) if scatters else float("nan")


# ---------------------------------------------------------------------------
# Д15 - which metrics, for the configuration that actually ships
# ---------------------------------------------------------------------------

def study_d15(data: Path, workers: int | None, max_size: int = 6) -> dict[str, Any]:
    """Every subset of the six, under the shipped configuration.

    The published leave-one-out ran on the intermediate configuration with the
    consensus kernel still on, so its conclusion could not be carried across.
    This is all 63 non-empty subsets with the kernel off, which answers the
    composition question and the speed-against-quality question at once.
    """
    variants: list[tuple[str, dict[str, Any]]] = []
    for size in range(1, max_size + 1):
        for subset in itertools.combinations(SIX, size):
            label = "six" if size == len(SIX) else "+".join(s[:4] for s in subset)
            variants.append((
                f"{size}:{label}",
                {"metrics.enabled": subset, **WEIGHTING["adaptive"]},
            ))
    directories = corpus_directories(data)
    runs = sweep(directories, variants, workers=workers)
    summary = aggregate_variants(runs)
    for name, row in summary.items():
        row["size"] = int(name.split(":", 1)[0])
    return {"runs": runs, "aggregate": summary, "subsets": len(variants)}


# ---------------------------------------------------------------------------
# Д16 - analysis width and region of interest
# ---------------------------------------------------------------------------

def study_d16(data: Path, workers: int | None) -> dict[str, Any]:
    """Resolution and region, for the shipped configuration.

    Width was tested before on the intermediate configuration; the region never
    was.  A centre crop is what a real focus loop would use, and it changes
    which content the measure sees, not only how much of it.
    """
    variants: list[tuple[str, dict[str, Any]]] = []
    for width in (160, 240, 320, 480, 640):
        for fraction, label in ((1.0, "full"), (0.5, "centre50"), (0.25, "centre25")):
            variants.append((
                f"w{width}_{label}",
                {
                    "pipeline.analysis_width": width,
                    "pipeline.default_roi_fraction": fraction,
                    **WEIGHTING["adaptive"],
                },
            ))
    directories = corpus_directories(data)
    runs = sweep(directories, variants, workers=workers)
    return {"runs": runs, "aggregate": aggregate_variants(runs)}


# ---------------------------------------------------------------------------
# Д17 - controlled degradations of real frames
# ---------------------------------------------------------------------------

def degrade(image: np.ndarray, kind: str, amount: float, seed: int) -> np.ndarray:
    """One transformation, applied to a real frame.

    These are model transformations, not new physical measurements. Adding
    Gaussian noise to an already-denoised 8-bit preview is not the same thing
    as raising the sensor's ISO, and a result here says how the measure reacts
    to the transformation, not how it would behave on a noisier camera.
    """
    if kind == "none":
        return image
    if kind == "noise":
        rng = np.random.default_rng(seed)
        noisy = image.astype(np.float32) + rng.normal(0.0, amount, image.shape)
        return np.clip(noisy, 0, 255).astype(np.uint8)
    if kind == "brightness":
        return np.clip(image.astype(np.float32) * amount, 0, 255).astype(np.uint8)
    if kind == "clip":
        # Push the highlights into saturation without changing the midtones
        # much, which is what an over-exposure does to a real frame.
        scaled = image.astype(np.float32) * (1.0 + amount)
        return np.clip(scaled, 0, 255).astype(np.uint8)
    if kind == "shift":
        pixels = int(round(amount))
        return np.roll(image, (pixels, pixels), axis=(0, 1))
    if kind == "blur":
        import cv2
        size = int(amount) * 2 + 1
        return cv2.GaussianBlur(image, (size, size), 0)
    raise ValueError(f"unknown degradation {kind!r}")


#: Each degradation and the amounts it is applied at.  The seed is fixed so a
#: re-run is the same experiment.
DEGRADATIONS: tuple[tuple[str, tuple[float, ...]], ...] = (
    ("noise", (2.0, 5.0, 10.0, 20.0)),
    ("brightness", (0.5, 0.75, 1.5, 2.0)),
    ("clip", (0.25, 0.5, 1.0)),
    ("shift", (1.0, 3.0, 8.0)),
    ("blur", (1.0, 2.0, 4.0)),
)
DEGRADATION_SEED = 20260917


def study_d17(data: Path, workers: int | None) -> dict[str, Any]:
    """How the shipped score reacts when a real frame is made worse.

    The blur family is the sanity check: a measure of focus that does not fall
    when the frame is blurred is broken, whatever else it does. The rest ask
    whether the score is measuring focus or measuring exposure.
    """
    base = build_config(load_default_config(), WEIGHTING["adaptive"])
    out: dict[str, Any] = {"runs": {}, "seed": DEGRADATION_SEED}

    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        mask, labels, truth = recording.mask, recording.labels, recording.truth
        variants: dict[str, Any] = {}

        clean = rich_replay(recording.images, recording.context, base)
        variants["none"] = score_against(clean["filtered"][mask], labels, truth)
        variants["none"]["score_mean"] = float(np.nanmean(clean["filtered"][mask]))
        reference_series = clean["filtered"][mask]

        for kind, amounts in DEGRADATIONS:
            for amount in amounts:
                images = [
                    degrade(image, kind, amount, DEGRADATION_SEED + index)
                    for index, image in enumerate(recording.images)
                ]
                result = rich_replay(images, recording.context, base)
                series = result["filtered"][mask]
                row = score_against(series, labels, truth)
                row["score_mean"] = float(np.nanmean(series))
                row["score_shift"] = row["score_mean"] - variants["none"]["score_mean"]
                row["rank_agreement_with_clean"] = spearman(series, reference_series)
                row["confidence_mean"] = float(np.nanmean(result["confidence"][mask]))
                row["noise_sigma_mean"] = float(np.nanmean(result["noise_sigma"][mask]))
                row["edge_density_mean"] = float(np.nanmean(result["edge_density"][mask]))
                variants[f"{kind}_{amount:g}"] = row
        out["runs"][name] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "variants": variants,
        }
    out["aggregate"] = aggregate_variants(out["runs"])
    return out


# ---------------------------------------------------------------------------
# Д18 - the sensitivity coefficients, fitted and evaluated apart
# ---------------------------------------------------------------------------

def scaled_sensitivity(factor: float) -> dict[str, Any]:
    """The whole kappa matrix multiplied by one number.

    Factor 0 is the reliability model off; 1 is what ships. The coefficients
    were set by hand and never estimated, so the first question is not which
    value is best but whether the result depends on them at all.
    """
    sensitivity = load_default_config().ensemble.sensitivity
    return {
        "ensemble.sensitivity": {
            name: {key: value * factor for key, value in row.items()}
            for name, row in sensitivity.items()
        }
    }


def study_d18(data: Path, workers: int | None) -> dict[str, Any]:
    """Vary the coefficients, and keep the fitting set apart from the test set.

    Picking the best factor on all eight recordings and then reporting that
    factor's score on the same eight is how a tuning exercise gets published as
    a result. The split here is by scene type, so the held-out recordings are
    not near-duplicates of the fitted ones.
    """
    factors = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
    variants = [
        (f"kappa{factor:g}", {**scaled_sensitivity(factor),
                              "ensemble.use_agreement": False})
        for factor in factors
    ]
    directories = corpus_directories(data)
    runs = sweep(directories, variants, workers=workers)
    summary = aggregate_variants(runs)

    # Fit on the condition recordings, evaluate on the sweeps and point sources.
    fit_names = [n for n in runs if n.startswith("cond_")]
    held_names = [n for n in runs if not n.startswith("cond_")]

    def mean_over(names: Sequence[str], variant: str, field: str) -> float:
        values = [
            runs[n]["variants"][variant][field]
            for n in names
            if variant in runs[n]["variants"]
            and np.isfinite(runs[n]["variants"][variant].get(field, np.nan))
        ]
        return float(np.mean(values)) if values else float("nan")

    fitted = max(
        (v for v, _ in variants),
        key=lambda v: mean_over(fit_names, v, "adj"),
    )
    split = {
        "fit_recordings": fit_names,
        "held_out_recordings": held_names,
        "chosen_on_fit": fitted,
        "fit_adj": mean_over(fit_names, fitted, "adj"),
        "held_out_adj": mean_over(held_names, fitted, "adj"),
        "shipped_fit_adj": mean_over(fit_names, "kappa1", "adj"),
        "shipped_held_out_adj": mean_over(held_names, "kappa1", "adj"),
        "best_on_held_out": max(
            (v for v, _ in variants), key=lambda v: mean_over(held_names, v, "adj")
        ),
    }
    return {"runs": runs, "aggregate": summary, "factors": list(factors),
            "held_out": split}


# ---------------------------------------------------------------------------
# Д19 - a search over a recorded pass
# ---------------------------------------------------------------------------

def _step_levels(series: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    steps, medians, _ = step_profile(series, labels)
    return np.asarray(steps), np.asarray(medians)


def _hill_climb(levels: np.ndarray, start: int, budget: int) -> dict[str, Any]:
    """Climb while the score improves, then stop.  Returns where it stopped.

    This is a search over an already-recorded profile, so it does not model the
    lens at all: no backlash, no settling, no cost of reversing direction. It
    answers "would this signal lead a search to the right step", not "how fast
    would the camera focus".
    """
    position = int(np.clip(start, 0, levels.size - 1))
    visited = [position]
    direction = 1
    evaluations = 1
    while evaluations < budget:
        nxt = position + direction
        if not 0 <= nxt < levels.size:
            if direction == -1:
                break
            direction = -1
            continue
        evaluations += 1
        visited.append(nxt)
        if levels[nxt] > levels[position]:
            position = nxt
        elif direction == 1:
            direction = -1
        else:
            break
    return {"stopped_at": position, "evaluations": evaluations,
            "visited": len(set(visited))}


def _ternary(levels: np.ndarray, budget: int) -> dict[str, Any]:
    low, high, evaluations = 0, levels.size - 1, 0
    while high - low > 2 and evaluations + 2 <= budget:
        first = low + (high - low) // 3
        second = high - (high - low) // 3
        evaluations += 2
        if levels[first] < levels[second]:
            low = first + 1
        else:
            high = second - 1
    window = np.arange(low, high + 1)
    best = int(window[int(np.argmax(levels[window]))]) if window.size else low
    return {"stopped_at": best, "evaluations": evaluations + window.size,
            "visited": int(window.size)}


def study_d19(data: Path, workers: int | None) -> dict[str, Any]:
    """Would a search driven by this score find the reference's best step?

    Replaying a recorded pass is not a closed-loop test and cannot become one:
    nothing here moves a lens or waits for it to settle. What it can say is
    whether the signal's shape is one a search can climb, and how the answer
    depends on where the search starts.
    """
    base = build_config(load_default_config(), WEIGHTING["adaptive"])
    out: dict[str, Any] = {"runs": {}}

    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        result = rich_replay(recording.images, recording.context, base)
        series = result["filtered"][recording.mask]
        labels = recording.labels
        steps, levels = _step_levels(series, labels)
        if steps.size < 5:
            continue
        best_index = int(np.argmax(levels))

        truth_index = None
        if recording.truth is not None:
            _, truth_levels = _step_levels(recording.truth, labels)
            truth_index = int(np.argmax(truth_levels))

        trials = []
        for start in range(steps.size):
            for budget in (6, 10, steps.size * 2):
                outcome = _hill_climb(levels, start, budget)
                trial = {
                    "strategy": "hill_climb",
                    "start_step": float(steps[start]),
                    "budget": budget,
                    "evaluations": outcome["evaluations"],
                    "error_vs_signal_peak": abs(outcome["stopped_at"] - best_index),
                }
                if truth_index is not None:
                    trial["error_vs_reference"] = abs(outcome["stopped_at"] - truth_index)
                trials.append(trial)
        for budget in (6, 10, steps.size * 2):
            outcome = _ternary(levels, budget)
            trial = {
                "strategy": "ternary", "start_step": float("nan"), "budget": budget,
                "evaluations": outcome["evaluations"],
                "error_vs_signal_peak": abs(outcome["stopped_at"] - best_index),
            }
            if truth_index is not None:
                trial["error_vs_reference"] = abs(outcome["stopped_at"] - truth_index)
            trials.append(trial)

        summary: dict[str, Any] = {}
        for strategy in ("hill_climb", "ternary"):
            for budget in sorted({t["budget"] for t in trials}):
                subset = [
                    t for t in trials
                    if t["strategy"] == strategy and t["budget"] == budget
                ]
                if not subset:
                    continue
                errors = np.array([t["error_vs_signal_peak"] for t in subset], float)
                entry = {
                    "trials": len(subset),
                    "mean_error_vs_signal_peak": float(errors.mean()),
                    "found_signal_peak_fraction": float(np.mean(errors == 0)),
                    "mean_evaluations": float(
                        np.mean([t["evaluations"] for t in subset])
                    ),
                }
                if truth_index is not None:
                    reference_errors = np.array(
                        [t["error_vs_reference"] for t in subset], float
                    )
                    entry["mean_error_vs_reference"] = float(reference_errors.mean())
                    entry["within_one_step_of_reference"] = float(
                        np.mean(reference_errors <= 1)
                    )
                summary[f"{strategy}_budget{budget}"] = entry

        out["runs"][name] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "steps": int(steps.size),
            "signal_peak_step": float(steps[best_index]),
            "reference_peak_step": (
                float(steps[truth_index]) if truth_index is not None else None
            ),
            "variants": summary,
        }
    out["aggregate"] = aggregate_variants(out["runs"])
    return out


# ---------------------------------------------------------------------------
# Д20 - do the criteria agree about the ranking
# ---------------------------------------------------------------------------

def study_d20(data: Path, workers: int | None, reports: Path | None = None) -> dict[str, Any]:
    """The conclusions depend on which criterion is read.  By how much?

    Section 5.3 found rank correlation and adjacent-step discrimination
    disagreeing about the sign of the adaptivity effect.  That could be a
    property of that one comparison or of the criteria.  This ranks a large set
    of configurations by each criterion and asks how much the rankings agree,
    and separately how much the plateau width depends on the tolerance used to
    define it.
    """
    variants: list[tuple[str, dict[str, Any]]] = []
    for size in (1, 2, 3, 6):
        for subset in itertools.combinations(SIX, size):
            if size == 2 and subset[0] != "laplacian":
                continue
            if size == 3 and subset[0] != "laplacian":
                continue
            variants.append((
                "+".join(s[:4] for s in subset),
                {"metrics.enabled": subset, **WEIGHTING["adaptive"]},
            ))
    for label, overrides in WEIGHTING.items():
        variants.append((f"weights_{label}", dict(overrides)))
    variants.append(("kernel_on", {"ensemble.use_agreement": True}))
    variants.append(("filter_off", {"temporal.enabled": False,
                                    **WEIGHTING["adaptive"]}))

    directories = corpus_directories(data)
    runs = sweep(directories, variants, workers=workers, keep=("filtered",))

    # Rank the variants by each criterion and compare the rankings.
    criteria = ("adj", "spearman_gt", "peak_err", "sat")
    summary = aggregate_variants(runs)
    names = [n for n in summary if "adj" in summary[n]]
    vectors: dict[str, np.ndarray] = {}
    for criterion in criteria:
        values = np.array([
            summary[n].get(criterion, np.nan) for n in names
        ], dtype=float)
        if np.isfinite(values).sum() >= 5:
            # peak_err and sat are better when smaller.
            vectors[criterion] = -values if criterion in ("peak_err", "sat") else values

    agreement: dict[str, dict[str, float]] = {}
    for first in vectors:
        agreement[first] = {}
        for second in vectors:
            both = np.isfinite(vectors[first]) & np.isfinite(vectors[second])
            agreement[first][second] = (
                spearman(vectors[first][both], vectors[second][both])
                if both.sum() >= 5 else float("nan")
            )

    # How much does the plateau width depend on the tolerance?
    tolerance_curve: dict[str, Any] = {}
    for tolerance in (0.0, 0.005, 0.01, 0.02, 0.05, 0.1):
        widths = []
        for name, run in runs.items():
            recording = load_recording(data / name)
            labels = recording.labels
            for variant in ("six", "weights_adaptive"):
                row = run["variants"].get(variant)
                if not row or "series_filtered" not in row:
                    continue
                series = np.asarray(row["series_filtered"], dtype=float)
                steps, medians, _ = step_profile(series, labels)
                if not len(steps):
                    continue
                top = np.nanmax(medians)
                span = np.nanmax(medians) - np.nanmin(medians)
                at_top = np.flatnonzero(medians >= top - tolerance * max(span, 1e-12))
                widths.append(float(steps[at_top[-1]] - steps[at_top[0]] + 1))
        if widths:
            tolerance_curve[str(tolerance)] = {
                "mean_plateau_steps": float(np.mean(widths)),
                "recordings": len(widths),
            }

    for run in runs.values():
        for row in run["variants"].values():
            row.pop("series_filtered", None)

    return {
        "runs": runs,
        "aggregate": summary,
        "criteria_agreement": agreement,
        "plateau_tolerance": tolerance_curve,
        "variants_ranked": len(names),
    }


# ---------------------------------------------------------------------------
# Д08 - the first commit, actually executed
# ---------------------------------------------------------------------------

FIRST_COMMIT = "1f1c75caf31d3b686dd594a4876b2d45ea6ccb3b"

#: Run inside a checkout of the first commit, in its own process.
#: That commit's package is called `sharpness` and its result object exposes a
#: single `score`, so it cannot be imported alongside the current
#: `adaptive_sharpness` and cannot be driven through the current replay code.
_FIRST_COMMIT_DRIVER = '''
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root))
import numpy as np, cv2
from sharpness import SharpnessEvaluator, load_config

config = load_config(str(root / "config" / "default.toml"))
evaluator = SharpnessEvaluator(config)
frames = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
scores, confidences = [], []
for path in frames:
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    result = evaluator.evaluate(image)
    score = getattr(result, "score", None)
    if score is None:
        score = getattr(result, "filtered_score", float("nan"))
    scores.append(float(score))
    confidences.append(float(getattr(result, "confidence", float("nan"))))
Path(sys.argv[3]).write_text(
    json.dumps({"score": scores, "confidence": confidences}), encoding="utf-8"
)
'''


def study_d08(data: Path, workers: int | None, worktree: Path | None = None) -> dict[str, Any]:
    """Run the first commit's own code, not a reconstruction of its settings.

    Every "before" number published so far came from `HISTORIC`, which is the
    first commit's *parameters* inside today's implementation. That is a fair
    way to ask what the parameters did and an unfair way to claim the first
    version behaved like that, because every bug fixed since is absent from it.
    This executes the original.

    If the original cannot run - a dependency it needs no longer installs, an
    API it uses is gone - that is the result, and it is reported rather than
    replaced by the reconstruction.
    """
    import json as _json
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[1]
    worktree = worktree or (root / ".worktrees" / "first-commit")
    out: dict[str, Any] = {"commit": FIRST_COMMIT, "runs": {}}

    if not worktree.exists():
        worktree.parent.mkdir(parents=True, exist_ok=True)
        created = subprocess.run(
            ["git", "worktree", "add", "--detach", str(worktree), FIRST_COMMIT],
            cwd=root, capture_output=True, text=True,
        )
        if created.returncode != 0:
            out["unavailable"] = f"git worktree failed: {created.stderr.strip()}"
            return out

    driver = worktree / "_d08_driver.py"
    driver.write_text(_FIRST_COMMIT_DRIVER, encoding="utf-8")

    base = build_config(load_default_config(), WEIGHTING["adaptive"])
    for name in CORPUS:
        recording = load_recording(data / name)
        if recording is None:
            continue
        paths = [
            str((data / name / "frames" / f).resolve()) for f in recording.context.files
        ]
        with tempfile.TemporaryDirectory() as tmp:
            listing = Path(tmp) / "frames.json"
            listing.write_text(_json.dumps(paths), encoding="utf-8")
            result_path = Path(tmp) / "out.json"
            completed = subprocess.run(
                [sys.executable, str(driver), str(worktree), str(listing),
                 str(result_path)],
                capture_output=True, text=True, cwd=str(worktree),
            )
            if completed.returncode != 0 or not result_path.exists():
                out.setdefault("failures", {})[name] = (
                    completed.stderr.strip()[-600:] or "no output"
                )
                continue
            payload = _json.loads(result_path.read_text(encoding="utf-8"))

        original = np.asarray(payload["score"], dtype=float)[recording.mask]
        current_result = rich_replay(recording.images, recording.context, base)
        current = current_result["filtered"][recording.mask]
        labels, truth = recording.labels, recording.truth

        entry: dict[str, Any] = {
            "selection": recording.selection,
            "has_reference": recording.has_reference,
            "variants": {
                "first_commit": score_against(original, labels, truth),
                "current": score_against(current, labels, truth),
            },
        }
        entry["variants"]["first_commit"]["confidence"] = float(
            np.nanmean(np.asarray(payload["confidence"], float)[recording.mask])
        )
        entry["variants"]["current"]["confidence"] = float(
            np.nanmean(current_result["confidence"][recording.mask])
        )
        entry["rank_agreement_between_versions"] = spearman(original, current)
        out["runs"][name] = entry

    if out["runs"]:
        out["aggregate"] = aggregate_variants(out["runs"])
    elif "unavailable" not in out:
        out["unavailable"] = "the first commit ran on no recording; see failures"
    return out


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

STUDIES = {
    "d07": study_d07, "d08": study_d08, "d09": study_d09, "d10": study_d10,
    "d11": study_d11, "d12": study_d12, "d13": study_d13, "d14": study_d14,
    "d15": study_d15, "d16": study_d16, "d17": study_d17, "d18": study_d18,
    "d19": study_d19, "d20": study_d20,
}

TITLES = {
    "d07": "Невизначеність парних різниць",
    "d08": "Відтворення першого коду",
    "d09": "Нормалізація поетапно",
    "d10": "Залежність від передісторії",
    "d11": "Параметри автофіксації",
    "d12": "Поведінка ваг",
    "d13": "Якість показника впевненості",
    "d14": "Часовий фільтр",
    "d15": "Підмножини метрик",
    "d16": "Масштаб та ROI",
    "d17": "Штучні деградації на реальних кадрах",
    "d18": "Чутливість коефіцієнтів надійності",
    "d19": "Пошук по записаному проходу",
    "d20": "Стійкість критеріїв оцінювання",
}
