"""Multi-run model comparison on protocol recordings with exact step labels.

Earlier studies in this project had to guess where the focus ring was held
still, by segmenting a stateless signal, and recovered only 30% of one handheld
recording.  Every ranking that came out of that was marked "indicative, not
settled".

The recordings this tool reads were made with the UI's protocol buttons: the
operator was prompted to hold or move, and the step number and phase were
written into every CSV row as they happened.  The grouping is therefore known
rather than inferred, coverage is ~70% of frames instead of 30%, and the camera
was on a tripod under manual exposure.

What it compares
----------------
Single measures      this project's six plus nine published baselines.
Fusion rules         eight rules over the project's six metrics.
Metric sets          the adaptive rule over four different metric subsets,
                     to test whether the two expensive metrics earn their place
                     and whether two published measures should be added.
Online output        the score the shipped system actually produced live,
                     which is the only model here that had no future knowledge.

How they are scored
-------------------
adj        Adjacent-step discrimination: mean |2*AUC - 1| over consecutive
           step pairs.  Telling a focus position from its neighbour is the job
           a search actually has to do, and it is the one that separates these
           models - the coarser measures below saturate.  This is the primary
           number.
eta2       Rank-ANOVA effect size: the fraction of rank variance explained by
           the step label.  Invariant to any monotone rescaling.  Measures
           separation of all positions, near and far, and saturates above 0.9
           for almost every model on these recordings.
sat        Fraction of frames pinned at the signal's own minimum or maximum,
           where it carries no gradient for a search to follow.
sep        Between-step spread over within-step scatter, on commonly
           normalised values.  Comparable with the earlier studies, and shown
           here mainly to demonstrate that it disagrees with the rank measures
           whenever ``sat`` is high.
mono       Fraction of step transitions moving the right way relative to the
           profile peak.
peaks      Local maxima in the step profile that stand above within-step
           noise.  A search can be trapped by any of them.
peak_err   Distance from the peak the model reports to the peak an independent
           physical measurement reports.  Only defined for the point-source
           runs, where spot size gives ground truth that uses no focus measure.

    python3 tools/protocol_study.py data --figures docs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import SharpnessConfig, load_default_config  # noqa: E402
from adaptive_sharpness.analysis import DEGRADATION_KEYS, ImageAnalyzer  # noqa: E402
from adaptive_sharpness.ensemble import AdaptiveEnsemble  # noqa: E402
from adaptive_sharpness.metrics import build_metrics  # noqa: E402
from adaptive_sharpness.preprocess import Preprocessor  # noqa: E402
from adaptive_sharpness.types import ImageStats  # noqa: E402
from tools.baselines import COMBINERS, FOCUS_MEASURES, OFFLINE_COMBINERS  # noqa: E402
from tools.comprehensive_study import all_measures, normalise_columns  # noqa: E402
from tools.evaluation import (  # noqa: E402
    SPOT_PARAMETERS,
    config_fingerprint,
    file_fingerprint,
    load_measure_cache,
    provenance,
)

logger = logging.getLogger("protocol_study")

#: Bump when the measurement code changes in a way that alters cached values.
_CACHE_VERSION = "2"

OWN: tuple[str, ...] = (
    "laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width",
)

#: Metric subsets the adaptive rule is run over.  ``own6`` is what ships.
#:
#: ``lean4`` drops the two measures the earlier comparison found to be both the
#: most expensive (76% of the measure budget) and the least separating.  If the
#: ensemble does not lose anything, they should go.
#:
#: ``plus8`` and ``core4`` add the two published measures that beat all six of
#: ours on the handheld recording.
METRIC_SETS: dict[str, tuple[str, ...]] = {
    "own6": OWN,
    "lean4": ("laplacian", "tenengrad", "brenner", "wavelet"),
    "plus8": OWN + ("VOL4", "TENV"),
    "core4": ("tenengrad", "brenner", "VOL4", "TENV"),
}

#: Sensitivity coefficients for the two added baselines.
#:
#: These are ASSUMED BY ANALOGY, not fitted: TENV is the variance of the Sobel
#: magnitude, so it is given the first-derivative behaviour of ``tenengrad``;
#: VOL4 is a first-order autocorrelation difference, structurally closest to
#: ``brenner``.  Both are raised slightly under noise, because neither carries
#: the contrast normalisation the project's own metrics apply.  Every number in
#: the shipped table has the same status - see EnsembleConfig's docstring - so
#: this does not put the added metrics on a different footing, but a result
#: that depends on these values is a result about an assumption.
ASSUMED_SENSITIVITY: dict[str, dict[str, float]] = {
    "TENV": {"noise": 1.4, "edge": 0.7, "clip": 0.6, "motion": 0.7, "contrast": 0.9},
    "VOL4": {"noise": 1.2, "edge": 0.9, "clip": 0.6, "motion": 0.9, "contrast": 1.0},
}

#: Fusion rules run over every metric set.  The full eight run over ``own6``.
CORE_RULES: tuple[str, ...] = ("mean", "median", "adaptive")

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class RunData:
    """Everything one recording contributes, already measured."""

    name: str
    preset: str
    note: str
    step: np.ndarray            # per frame, protocol step number (0 = unknown)
    hold: np.ndarray            # per frame, True while the ring was held still
    raw: dict[str, np.ndarray]  # measure name -> raw value per frame
    stats: list[ImageStats]
    degradations: list[dict[str, float]]
    online_score: np.ndarray
    online_filtered: np.ndarray
    spot_radius: np.ndarray | None   # point-source runs only
    spot_offset: np.ndarray | None   # spot displacement from the run median

    @property
    def n(self) -> int:
        return self.step.size

    @property
    def is_point_source(self) -> bool:
        return self.spot_radius is not None


def _float(row: dict[str, str], key: str, default: float = float("nan")) -> float:
    value = row.get(key, "")
    if value in ("", None):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def spot_radius(image: np.ndarray, half: int = 80, fraction: float = 0.5) -> tuple[float, tuple[int, int]]:
    """Radius enclosing half of the spot's background-subtracted pixel sum.

    This is the point-source reference.  It uses no focus measure: a point
    imaged through a defocused lens spreads into a disc, and the radius holding
    half the spot's signal tracks the radius of that disc.

    **Not an optical energy radius.**  The input is an 8-bit JPEG preview from
    the camera, with unknown tone curve and no radiometric calibration, so the
    pixel sum is a monotone-ish proxy for energy rather than energy.  For a
    *reference ordering* - which frame is more defocused - that is enough, and
    ordering is all this is used for.

    Background is taken from the border of the window rather than the frame, so
    that the dim room behind the source does not enter the integral.

    Caveats, neither of them established as harmless:

    * The source saturates near focus (measured peak 251-253 of 255).  Clipped
      core samples are missing from the sum, which inflates the radius where it
      is smallest.  Whether that displaces the minimum has not been shown; the
      sensitivity of the result to r25/r50/r75 and to the window size is the
      way to find out.
    * The window is fixed at +/-80 px.  A disc larger than that is truncated,
      which compresses the radius at heavy defocus.
    """
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    _, _, _, peak = cv2.minMaxLoc(blurred)
    cx, cy = int(peak[0]), int(peak[1])
    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half + 1)
    y0, y1 = max(0, cy - half), min(image.shape[0], cy + half + 1)
    window = image[y0:y1, x0:x1].astype(np.float64)

    border = np.concatenate([
        window[0, :], window[-1, :], window[:, 0], window[:, -1],
    ])
    signal = np.maximum(window - float(np.median(border)), 0.0)
    total = float(signal.sum())
    if total <= _EPS:
        return float("nan"), (cx, cy)

    grid_y, grid_x = np.mgrid[0:window.shape[0], 0:window.shape[1]]
    centre_y = float((signal * grid_y).sum() / total)
    centre_x = float((signal * grid_x).sum() / total)
    distance = np.sqrt((grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2)

    order = np.argsort(distance, axis=None)
    cumulative = np.cumsum(signal.ravel()[order])
    index = int(np.searchsorted(cumulative, total * fraction))
    index = min(index, order.size - 1)
    return float(distance.ravel()[order][index]), (x0 + int(round(centre_x)), y0 + int(round(centre_y)))


def extract(
    directory: Path, config: SharpnessConfig, *, refresh: bool = False
) -> RunData:
    """Measure every saved frame of one recording, with an on-disk cache."""
    rows = list(csv.DictReader((directory / "frames.csv").open(newline="")))
    rows = [r for r in rows if r.get("image_file")]
    if not rows:
        raise ValueError(f"{directory.name}: no saved frames")

    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    preset = rows[0].get("preset", "") or directory.name.rsplit("_", 2)[0]
    note = str(manifest.get("note", ""))
    point_source = "point" in preset.lower()

    step = np.array([int(_float(r, "step_index", 0.0) or 0) for r in rows], dtype=np.int32)
    hold = np.array([r.get("step_phase") == "hold" for r in rows], dtype=bool)
    online_score = np.array([_float(r, "instantaneous_score") for r in rows])
    online_filtered = np.array([_float(r, "filtered_score") for r in rows])

    cache_path = directory / "measures_cache.npz"
    names = tuple(all_measures(config))

    # A cache is only usable when the frames, the processing configuration and
    # the measurement code are all the same as when it was written.  The check
    # lives in tools/evaluation.py so that every reader applies the same one.
    files = [directory / "frames" / str(r["image_file"]) for r in rows]
    processing_fingerprint = "|".join((
        _CACHE_VERSION,
        config_fingerprint(config.pipeline),
        config_fingerprint(config.metrics),
        config_fingerprint(config.analysis),
        ",".join(names),
    ))
    spot_fingerprint = json.dumps(SPOT_PARAMETERS, sort_keys=True)

    if not refresh:
        cached_entry = load_measure_cache(
            directory, [str(r["image_file"]) for r in rows],
            processing_fingerprint=processing_fingerprint,
        )
        usable = cached_entry.raw is not None and (
            not point_source or cached_entry.has_reference
        )
        logger.info("%s: cache - %s", directory.name, cached_entry.note)
        if usable:
            cached = np.load(cache_path, allow_pickle=False)
            raw = {name: cached[f"raw_{name}"] for name in names}
            stats = [
                ImageStats(**{k: float(cached[f"stat_{k}"][i]) for k in _STAT_FIELDS})
                for i in range(len(rows))
            ]
            degradations = [
                {k: float(cached[f"deg_{k}"][i]) for k in DEGRADATION_KEYS}
                for i in range(len(rows))
            ]
            radius = cached["spot_radius"] if point_source else None
            offset = cached["spot_offset"] if point_source else None
            return RunData(
                directory.name, preset, note, step, hold, raw, stats,
                degradations, online_score, online_filtered, radius, offset,
            )

    preprocessor = Preprocessor(config.pipeline)
    analyzer = ImageAnalyzer(config.analysis)
    measures = all_measures(config)

    raw: dict[str, list[float]] = {name: [] for name in names}
    stats: list[ImageStats] = []
    degradations: list[dict[str, float]] = []
    radii: list[float] = []
    centres: list[tuple[int, int]] = []

    started = time.perf_counter()
    for index, row in enumerate(rows):
        path = directory / "frames" / str(row["image_file"])
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable frame: {path}")
        gray = preprocessor.prepare(image).gray
        image_stats = analyzer.analyze(gray)
        stats.append(image_stats)
        degradations.append(analyzer.degradations(image_stats))
        for name, function in measures.items():
            raw[name].append(float(function(gray)))
        if point_source:
            full = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            value, centre = spot_radius(full)
            radii.append(value)
            centres.append(centre)
        if index and index % 250 == 0:
            rate = index / (time.perf_counter() - started)
            logger.info("%s: %d/%d (%.0f frames/s)", directory.name, index, len(rows), rate)

    raw_arrays = {name: np.asarray(values, dtype=np.float64) for name, values in raw.items()}
    if point_source:
        centre_array = np.asarray(centres, dtype=np.float64)
        median_centre = np.median(centre_array, axis=0)
        offset = np.linalg.norm(centre_array - median_centre, axis=1)
        radius_array = np.asarray(radii, dtype=np.float64)
    else:
        offset = None
        radius_array = None

    payload: dict[str, Any] = {
        "n": len(rows),
        "names": np.array(names),
        "file_fingerprint": np.array(file_fingerprint(files)),
        "processing_fingerprint": np.array(processing_fingerprint),
        "spot_fingerprint": np.array(spot_fingerprint),
        "spot_radius": radius_array if radius_array is not None else np.zeros(0),
        "spot_offset": offset if offset is not None else np.zeros(0),
    }
    payload.update({f"raw_{name}": values for name, values in raw_arrays.items()})
    payload.update({
        f"stat_{field}": np.array([getattr(s, field) for s in stats])
        for field in _STAT_FIELDS
    })
    payload.update({
        f"deg_{key}": np.array([d[key] for d in degradations])
        for key in DEGRADATION_KEYS
    })
    np.savez_compressed(cache_path, **payload)

    return RunData(
        directory.name, preset, note, step, hold, raw_arrays, stats,
        degradations, online_score, online_filtered, radius_array, offset,
    )


_STAT_FIELDS: tuple[str, ...] = (
    "noise_sigma", "noise_level", "snr", "local_contrast", "brightness",
    "clipped_high", "clipped_low", "edge_density", "edge_sufficiency",
    "motion_px", "motion_level", "anisotropy",
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def _ensemble_for(names: Sequence[str], config: SharpnessConfig) -> AdaptiveEnsemble:
    """An adaptive ensemble over an arbitrary metric subset.

    Metrics outside the shipped six need a prior and a sensitivity row; see
    ASSUMED_SENSITIVITY for what those are and what they are worth.
    """
    priors = dict(config.metrics.priors)
    sensitivity = {k: dict(v) for k, v in config.ensemble.sensitivity.items()}
    for name in names:
        priors.setdefault(name, 1.0)
        if name not in sensitivity:
            sensitivity[name] = dict(ASSUMED_SENSITIVITY[name])
    ensemble_config = replace(config.ensemble, sensitivity=sensitivity)
    return AdaptiveEnsemble(tuple(names), ensemble_config, priors)


def build_models(run: RunData, config: SharpnessConfig) -> dict[str, np.ndarray]:
    """Every model's output for one recording, on a common footing.

    All measures go through the same offline normalisation before fusion, and
    every single measure also gets a normalised twin, so that no model is
    compared against another on a different scale.  The primary score (eta2) is
    rank-based and therefore immune to this choice anyway; ``sep`` is not, which
    is why the choice is made once, here, for everybody.
    """
    names = list(run.raw)
    normalised = normalise_columns(
        np.column_stack([run.raw[name] for name in names]).astype(np.float64)
    )
    column = {name: normalised[:, i] for i, name in enumerate(names)}

    models: dict[str, np.ndarray] = {
        f"single:{name}": column[name] for name in names
    }

    for set_name, metric_names in METRIC_SETS.items():
        matrix = np.column_stack([column[n] for n in metric_names])
        priors = np.array([
            config.metrics.priors.get(n, 1.0) for n in metric_names
        ])
        rules = COMBINERS if set_name == "own6" else {
            k: v for k, v in COMBINERS.items() if k in CORE_RULES
        }
        for rule_name, rule in rules.items():
            models[f"{set_name}:{rule_name}"] = np.asarray(
                rule(matrix, priors), dtype=np.float64
            )
        ensemble = _ensemble_for(metric_names, config)
        models[f"{set_name}:adaptive"] = np.array([
            ensemble.combine(
                {n: float(matrix[i, j]) for j, n in enumerate(metric_names)},
                run.stats[i],
                run.degradations[i],
            ).score
            for i in range(matrix.shape[0])
        ])

    models["online:instantaneous"] = run.online_score
    models["online:filtered"] = run.online_filtered
    return models


def is_offline(model: str) -> bool:
    """True for models that needed the whole recording before scoring a frame.

    Every fusion here is offline in one respect - the normalisation uses the
    whole sequence - which is why ``online:*`` is carried separately as the only
    strictly causal model in the comparison.
    """
    return model.split(":")[-1] in OFFLINE_COMBINERS


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def rank_eta_squared(values: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of rank variance explained by the step label.

    Rank-transforming first makes this invariant to any monotone rescaling, so
    a model cannot win by being normalised more favourably than another.  It is
    the Kruskal-Wallis effect size, bounded in [0, 1].
    """
    finite = np.isfinite(values)
    if finite.sum() < 4:
        return float("nan")
    values, labels = values[finite], labels[finite]
    # Ties must take the average rank.  A saturated signal is mostly ties, and
    # breaking them by array order would turn that saturation into apparent
    # within-group structure and flatter it.
    ranks = _average_ranks(values)
    grand = ranks.mean()
    total = float(((ranks - grand) ** 2).sum())
    if total <= _EPS:
        return 0.0
    between = 0.0
    for label in np.unique(labels):
        group = ranks[labels == label]
        between += group.size * (group.mean() - grand) ** 2
    return float(between / total)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks with ties averaged, as Mann-Whitney requires."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    ranks[order] = np.arange(1, values.size + 1, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    for index in range(1, values.size + 1):
        if index == values.size or sorted_values[index] != sorted_values[start]:
            if index - start > 1:
                ranks[order[start:index]] = ranks[order[start:index]].mean()
            start = index
    return ranks


def _auc(a: np.ndarray, b: np.ndarray) -> float:
    """P(a > b) + 0.5 P(a == b), via the Mann-Whitney U statistic."""
    combined = np.concatenate([a, b])
    ranks = _average_ranks(combined)
    u = ranks[: a.size].sum() - a.size * (a.size + 1) / 2.0
    return float(u / (a.size * b.size))


def adjacent_discrimination(values: np.ndarray, labels: np.ndarray) -> float:
    """How reliably one protocol step can be told from the next one.

    Separating far-apart focus positions is easy - on these recordings almost
    every signal does it, and the eta2 column saturates above 0.9 as a result.
    What an autofocus search actually needs is to tell a position from its
    neighbour, so this averages |2*AUC - 1| over consecutive step pairs.

    0 means the two steps are indistinguishable; 1 means every frame of one
    scores above every frame of the other.  The absolute value keeps it free of
    any assumption about which side of the peak a pair sits on, so no ground
    truth is needed and the measure cannot be made to look good by agreeing
    with the profile it is being scored against.
    """
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    steps = np.unique(labels)
    if steps.size < 2:
        return float("nan")
    scores = []
    for first, second in zip(steps[:-1], steps[1:]):
        a, b = values[labels == first], values[labels == second]
        if a.size < 2 or b.size < 2:
            continue
        scores.append(abs(2.0 * _auc(a, b) - 1.0))
    return float(np.mean(scores)) if scores else float("nan")


def saturated_fraction(values: np.ndarray) -> float:
    """Fraction of frames pinned at the extremes of the signal's own range.

    A signal that spends much of a sweep at exactly its own minimum or maximum
    has no gradient there for a search to follow.  Earlier work measured this
    at 18-22% for the fused score without being able to compare it against the
    alternatives; this column makes it comparable.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    low, high = float(finite.min()), float(finite.max())
    span = high - low
    if span <= _EPS:
        return 1.0
    tolerance = span * 1e-6
    pinned = (finite <= low + tolerance) | (finite >= high - tolerance)
    return float(pinned.mean())


def between_within(values: np.ndarray, labels: np.ndarray) -> float:
    """Between-step spread over the typical within-step scatter."""
    finite = np.isfinite(values)
    values, labels = values[finite], labels[finite]
    if values.size < 4:
        return float("nan")
    means, scatters = [], []
    for label in np.unique(labels):
        group = values[labels == label]
        if group.size < 2:
            continue
        means.append(float(group.mean()))
        scatters.append(float(group.std()))
    if len(means) < 2:
        return float("nan")
    return float(np.std(means) / max(float(np.median(scatters)), _EPS))


def step_profile(
    values: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-step median and within-step scatter, in step order."""
    steps = np.unique(labels)
    medians = np.array([float(np.nanmedian(values[labels == s])) for s in steps])
    scatters = np.array([float(np.nanstd(values[labels == s])) for s in steps])
    return steps, medians, scatters


def monotonicity(medians: np.ndarray, peak: int) -> float:
    """Fraction of step transitions moving towards the profile's own peak."""
    if medians.size < 2:
        return float("nan")
    correct = total = 0.0
    for index in range(medians.size - 1):
        delta = medians[index + 1] - medians[index]
        if index + 1 <= peak:
            expected_rise = True
        elif index >= peak:
            expected_rise = False
        else:
            continue
        total += 1.0
        if abs(delta) < _EPS:
            correct += 0.5
        elif (delta > 0) == expected_rise:
            correct += 1.0
    return correct / total if total else float("nan")


def count_peaks(medians: np.ndarray, scatters: np.ndarray) -> int:
    """Local maxima standing clear of within-step noise.

    A hill-climbing search stops at any of these, so the count is a direct
    measure of how trap-prone the signal is.  A bump only counts if it rises
    above the noise floor of the steps it sits between, which keeps scatter
    from being reported as structure.
    """
    if medians.size < 3:
        return 1
    floor = max(float(np.median(scatters)), _EPS)
    count = 0
    for index in range(medians.size):
        left = medians[index - 1] if index > 0 else -np.inf
        right = medians[index + 1] if index + 1 < medians.size else -np.inf
        if medians[index] >= left and medians[index] >= right:
            prominence = medians[index] - max(
                min(left, medians[index]), min(right, medians[index]), -np.inf
            )
            if not np.isfinite(prominence) or prominence >= floor:
                count += 1
    return max(count, 1)


def evaluate(
    series: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Every quality number for one model on one recording."""
    steps, medians, scatters = step_profile(series, labels)
    peak = int(np.argmax(medians))
    return {
        "adj": adjacent_discrimination(series, labels),
        "eta2": rank_eta_squared(series, labels),
        "sat": saturated_fraction(series),
        "sep": between_within(series, labels),
        "mono": monotonicity(medians, peak),
        "peaks": float(count_peaks(medians, scatters)),
        "peak_step": float(steps[peak]),
        "within_cv": float(
            np.median(scatters) / max(abs(float(np.median(medians))), _EPS)
        ),
    }


def ground_truth_peak(run: RunData) -> tuple[float, dict[str, float]] | None:
    """Best-focus step from spot size alone, for the point-source runs.

    Frames where the detected spot wandered away from its usual place are
    dropped: those are frames where the brightest thing in the window was not
    the source.
    """
    if not run.is_point_source:
        return None
    valid = run.hold & np.isfinite(run.spot_radius) & (run.spot_offset <= 20.0)
    if valid.sum() < 20:
        return None
    steps, medians, scatters = step_profile(run.spot_radius[valid], run.step[valid])
    best = int(np.argmin(medians))
    diagnostics = {
        "rejected_fraction": float(1.0 - valid.sum() / max(run.hold.sum(), 1)),
        "min_radius_px": float(medians[best]),
        "max_radius_px": float(np.max(medians)),
        "scatter_px": float(np.median(scatters)),
    }
    return float(steps[best]), diagnostics


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------

def measure_cost(
    directory: Path, config: SharpnessConfig, *, frames: int = 25, repeats: int = 3
) -> dict[str, Any]:
    """Per-frame cost of every measure, of the analyzer, and of every fusion.

    Best of several passes rather than the mean, because the Pi is a shared
    machine and a slow pass measures the scheduler, not the code.

    A model's total is then its metrics plus its fusion, and - for the adaptive
    rule only - the image analysis its weights depend on.  That analysis is the
    reason the adaptive rule costs three orders of magnitude more than an
    arithmetic mean, so charging it to the rule is the honest accounting.
    """
    rows = [
        r for r in csv.DictReader((directory / "frames.csv").open(newline=""))
        if r.get("image_file")
    ]
    step = max(1, len(rows) // frames)
    selected = rows[::step][:frames]

    preprocessor = Preprocessor(config.pipeline)
    analyzer = ImageAnalyzer(config.analysis)
    images = []
    for row in selected:
        image = cv2.imread(str(directory / "frames" / str(row["image_file"])), cv2.IMREAD_COLOR)
        images.append(preprocessor.prepare(image).gray)

    def timed(function) -> float:
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            for gray in images:
                function(gray)
            best = min(best, (time.perf_counter() - started) / len(images))
        return best * 1000.0

    measures = all_measures(config)
    per_measure = {name: timed(function) for name, function in measures.items()}

    def analyse(gray: np.ndarray) -> None:
        analyzer.degradations(analyzer.analyze(gray))

    analyzer.reset()
    analysis_ms = timed(analyse)

    # Fusion cost is measured on a matrix of the right shape, over the same
    # number of frames, so that it is per-frame like everything else.
    analyzer.reset()
    stats = [analyzer.analyze(gray) for gray in images]
    degradations = [analyzer.degradations(s) for s in stats]

    fusion_ms: dict[str, float] = {}
    for set_name, metric_names in METRIC_SETS.items():
        matrix = np.random.default_rng(0).random((len(images), len(metric_names)))
        priors = np.array([config.metrics.priors.get(n, 1.0) for n in metric_names])
        rules = COMBINERS if set_name == "own6" else {
            k: v for k, v in COMBINERS.items() if k in CORE_RULES
        }
        for rule_name, rule in rules.items():
            best = float("inf")
            for _ in range(repeats):
                started = time.perf_counter()
                rule(matrix, priors)
                best = min(best, (time.perf_counter() - started) / len(images))
            fusion_ms[f"{set_name}:{rule_name}"] = best * 1000.0

        ensemble = _ensemble_for(metric_names, config)
        best = float("inf")
        for _ in range(repeats):
            started = time.perf_counter()
            for index in range(len(images)):
                ensemble.combine(
                    {n: float(matrix[index, j]) for j, n in enumerate(metric_names)},
                    stats[index], degradations[index],
                )
            best = min(best, (time.perf_counter() - started) / len(images))
        fusion_ms[f"{set_name}:adaptive"] = best * 1000.0

    totals: dict[str, float] = {}
    for name in measures:
        totals[f"single:{name}"] = per_measure[name]
    for model, cost in fusion_ms.items():
        set_name = model.split(":")[0]
        metric_cost = sum(per_measure[n] for n in METRIC_SETS[set_name])
        overhead = analysis_ms if model.endswith(":adaptive") else 0.0
        totals[model] = metric_cost + cost + overhead
    totals["online:instantaneous"] = totals["own6:adaptive"]
    totals["online:filtered"] = totals["own6:adaptive"]

    return {
        "frames": len(images),
        "per_measure_ms": per_measure,
        "analysis_ms": analysis_ms,
        "per_fusion_ms": fusion_ms,
        "model_total_ms": totals,
    }


# ---------------------------------------------------------------------------
# Study
# ---------------------------------------------------------------------------

def study(
    directories: Sequence[Path], config: SharpnessConfig, *, refresh: bool = False,
    with_cost: bool = True,
) -> dict[str, Any]:
    runs: list[RunData] = []
    for directory in directories:
        logger.info("extracting %s", directory.name)
        runs.append(extract(directory, config, refresh=refresh))

    per_run: dict[str, Any] = {}
    truth: dict[str, Any] = {}
    for run in runs:
        selected = run.hold & (run.step > 0)
        if selected.sum() < 20:
            logger.warning("%s: too few held frames, skipped", run.name)
            continue
        labels = run.step[selected]
        models = build_models(run, config)
        scores = {
            name: evaluate(series[selected], labels)
            for name, series in models.items()
        }
        reference = ground_truth_peak(run)
        if reference is not None:
            best_step, diagnostics = reference
            truth[run.name] = {"peak_step": best_step, **diagnostics}
            for name, entry in scores.items():
                entry["peak_err"] = abs(entry["peak_step"] - best_step)
        step_values = np.unique(labels)
        profiles = {
            name: step_profile(series[selected], labels)[1].tolist()
            for name, series in models.items()
        }
        if run.is_point_source:
            valid = selected & np.isfinite(run.spot_radius) & (run.spot_offset <= 20.0)
            spot_steps, spot_medians, _ = step_profile(
                run.spot_radius[valid], run.step[valid]
            )
            spot_profile = {
                "steps": spot_steps.tolist(), "radius_px": spot_medians.tolist()
            }
        else:
            spot_profile = None

        per_run[run.name] = {
            "preset": run.preset,
            "note": run.note,
            "step_values": step_values.tolist(),
            "profiles": profiles,
            "spot_profile": spot_profile,
            "frames": int(run.n),
            "held": int(selected.sum()),
            "steps": int(np.unique(labels).size),
            "brightness": float(np.median([s.brightness for s in run.stats])),
            "edge_density": float(np.median([s.edge_density for s in run.stats])),
            "motion_px_p95": float(np.percentile([s.motion_px for s in run.stats], 95)),
            "models": scores,
        }

    cost: dict[str, Any] = {}
    if with_cost and directories:
        logger.info("measuring cost")
        cost = measure_cost(directories[0], config)

    return {
        "runs": per_run,
        "ground_truth": truth,
        "aggregate": aggregate(per_run),
        "cost": cost,
        "provenance": provenance(
            config, recordings=[d.name for d in directories]
        ),
    }


def aggregate(per_run: dict[str, Any]) -> dict[str, Any]:
    """Pool the per-run results into one ranking.

    Runs are pooled by averaging each model's *rank* within a run, not its raw
    score.  Recordings differ in how hard they are - a low-texture sweep scores
    everything lower - so averaging raw scores would weight the easy recordings
    most.  Averaging ranks asks the only question that transfers: on this
    recording, did this model beat that one?
    """
    model_names: set[str] = set()
    for entry in per_run.values():
        model_names.update(entry["models"])

    columns = ("adj", "eta2", "sat", "sep", "mono", "peaks", "within_cv", "peak_err")
    lower_is_better = {"sat", "peaks", "within_cv", "peak_err"}
    pooled: dict[str, dict[str, Any]] = {
        name: {"runs": 0, **{c: [] for c in columns}, **{f"{c}_rank": [] for c in columns}}
        for name in model_names
    }

    for entry in per_run.values():
        models = entry["models"]
        for column in columns:
            values = {
                name: models[name][column]
                for name in models
                if column in models[name] and np.isfinite(models[name][column])
            }
            if len(values) < 2:
                continue
            order = sorted(
                values, key=lambda n: values[n], reverse=column not in lower_is_better
            )
            for position, name in enumerate(order):
                pooled[name][f"{column}_rank"].append(position + 1)
                pooled[name][column].append(values[name])
        for name in models:
            pooled[name]["runs"] += 1

    summary: dict[str, Any] = {}
    for name, record in pooled.items():
        row: dict[str, Any] = {"runs": record["runs"]}
        for column in columns:
            values = record[column]
            ranks = record[f"{column}_rank"]
            row[column] = float(np.mean(values)) if values else float("nan")
            row[f"{column}_rank"] = float(np.mean(ranks)) if ranks else float("nan")
        summary[name] = row
    return summary


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _kind(model: str) -> str:
    if model.startswith("single:"):
        return "measure"
    if model.startswith("online:"):
        return "online"
    return "fusion"


def print_report(results: dict[str, Any]) -> None:
    runs = results["runs"]
    print("=" * 78)
    print("PROTOCOL STUDY - stepped sweeps with exact labels")
    print("=" * 78)
    print(f"\n{len(runs)} recordings\n")
    header = f"{'recording':<34}{'preset':<14}{'held':>6}{'steps':>7}{'bright':>8}{'edges':>8}"
    print(header)
    print("-" * len(header))
    for name, entry in runs.items():
        print(
            f"{name:<34}{entry['preset']:<14}{entry['held']:>6}{entry['steps']:>7}"
            f"{entry['brightness']:>8.3f}{entry['edge_density']:>8.4f}"
        )

    if results["ground_truth"]:
        print("\n--- Ground truth from spot size (uses no focus measure) ---\n")
        for name, entry in results["ground_truth"].items():
            print(
                f"{name:<34} best step {entry['peak_step']:.0f}   "
                f"r50 {entry['min_radius_px']:.1f} -> {entry['max_radius_px']:.1f} px   "
                f"rejected {100 * entry['rejected_fraction']:.0f}%"
            )

    aggregate_rows = results["aggregate"]
    order = sorted(
        aggregate_rows,
        key=lambda n: (
            aggregate_rows[n]["adj_rank"]
            if np.isfinite(aggregate_rows[n]["adj_rank"]) else 1e9
        ),
    )
    print("\n--- Pooled ranking, by mean rank of adjacent-step discrimination ---\n")
    header = (
        f"{'#':>3} {'model':<26}{'kind':<9}{'adj':>7}{'rank':>6}"
        f"{'eta2':>7}{'sat':>7}{'mono':>7}{'peaks':>7}{'perr':>7}"
    )
    print(header)
    print("-" * len(header))
    for position, name in enumerate(order, 1):
        row = aggregate_rows[name]
        error = row["peak_err"]
        print(
            f"{position:>3} {name:<26}{_kind(name):<9}{row['adj']:>7.3f}"
            f"{row['adj_rank']:>6.1f}{row['eta2']:>7.3f}{row['sat']:>7.3f}"
            f"{row['mono']:>7.3f}{row['peaks']:>7.2f}"
            + (f"{error:>7.2f}" if np.isfinite(error) else f"{'-':>7}")
        )

    cost = results.get("cost") or {}
    if cost:
        totals = cost["model_total_ms"]
        print("\n--- Cost on this hardware, and what it buys ---\n")
        header = f"{'model':<26}{'ms/frame':>9}{'adj':>8}{'rank':>7}{'adj/ms':>9}"
        print(header)
        print("-" * len(header))
        affordable = [
            n for n in order
            if n in totals and np.isfinite(aggregate_rows[n]["adj"])
        ]
        for name in affordable[:16]:
            ms = totals[name]
            row = aggregate_rows[name]
            print(
                f"{name:<26}{ms:>9.3f}{row['adj']:>8.3f}{row['adj_rank']:>7.1f}"
                f"{row['adj'] / max(ms, 1e-9):>9.2f}"
            )

    print("\n--- Per recording, best five by adjacent-step discrimination ---\n")
    for name, entry in runs.items():
        models = entry["models"]
        best = sorted(
            models, key=lambda n: -models[n]["adj"] if np.isfinite(models[n]["adj"]) else 1e9
        )[:5]
        rendered = ", ".join(
            f"{m.split(':', 1)[-1] if m.startswith('single') else m} {models[m]['adj']:.3f}"
            for m in best
        )
        print(f"{name:<34} {rendered}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="directory holding the recordings")
    parser.add_argument("--only", nargs="*", default=None, help="recording name filter")
    parser.add_argument("--refresh", action="store_true", help="ignore measure caches")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--figures", type=Path, default=None)
    parser.add_argument("--no-cost", action="store_true", help="skip the timing pass")
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
    results = study(
        directories, config, refresh=args.refresh, with_cost=not args.no_cost
    )
    print_report(results)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")
    if args.figures:
        from tools.protocol_figures import make_figures

        make_figures(results, args.figures)
        print(f"wrote figures to {args.figures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
