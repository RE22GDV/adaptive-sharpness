"""Tests for the scoring functions of the protocol study.

These are the functions every published ranking now rests on, so each one is
tested against a case whose answer is known by construction rather than by
running the code and recording what it said.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.protocol_study import (
    _auc,
    _average_ranks,
    adjacent_discrimination,
    aggregate,
    count_peaks,
    monotonicity,
    rank_eta_squared,
    saturated_fraction,
    spot_radius,
    step_profile,
)


# ---------------------------------------------------------------------------
# Rank helpers
# ---------------------------------------------------------------------------

def test_average_ranks_splits_ties() -> None:
    ranks = _average_ranks(np.array([10.0, 20.0, 20.0, 40.0]))
    # The tied pair occupies positions 2 and 3, so both take 2.5.
    assert ranks.tolist() == [1.0, 2.5, 2.5, 4.0]


def test_auc_is_one_when_fully_separated() -> None:
    assert _auc(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 1.0


def test_auc_is_zero_when_fully_separated_the_other_way() -> None:
    assert _auc(np.array([0.0, 1.0]), np.array([3.0, 4.0])) == 0.0


def test_auc_is_half_for_identical_samples() -> None:
    sample = np.array([1.0, 2.0, 3.0])
    assert _auc(sample, sample.copy()) == 0.5


# ---------------------------------------------------------------------------
# Adjacent-step discrimination
# ---------------------------------------------------------------------------

def test_adjacent_discrimination_is_one_for_cleanly_stepped_signal() -> None:
    labels = np.repeat([1, 2, 3], 10)
    values = np.concatenate([
        np.linspace(0.0, 0.1, 10), np.linspace(1.0, 1.1, 10), np.linspace(2.0, 2.1, 10),
    ])
    assert adjacent_discrimination(values, labels) == pytest.approx(1.0)


def test_adjacent_discrimination_is_near_zero_for_pure_noise() -> None:
    rng = np.random.default_rng(0)
    labels = np.repeat([1, 2, 3, 4], 200)
    values = rng.normal(size=labels.size)
    assert adjacent_discrimination(values, labels) < 0.15


def test_adjacent_discrimination_ignores_direction() -> None:
    """A descending sweep must score the same as the ascending one.

    The measure takes an absolute value precisely so that it needs no ground
    truth about which side of the peak a pair sits on.
    """
    labels = np.repeat([1, 2, 3], 10)
    values = np.concatenate([
        np.full(10, 0.0), np.full(10, 1.0), np.full(10, 2.0),
    ])
    assert adjacent_discrimination(values, labels) == pytest.approx(
        adjacent_discrimination(-values, labels)
    )


def test_adjacent_discrimination_is_harder_than_eta_squared() -> None:
    """Neighbouring steps overlap; distant ones do not.

    This is the whole reason the study ranks on ``adj``: eta2 saturates when
    far-apart positions are easy to tell apart, which they are on a full sweep.
    """
    rng = np.random.default_rng(1)
    labels = np.repeat([1, 2, 3, 4, 5], 100)
    centres = np.repeat([0.0, 1.0, 2.0, 3.0, 4.0], 100)
    values = centres + rng.normal(scale=0.9, size=labels.size)
    assert rank_eta_squared(values, labels) > adjacent_discrimination(values, labels)


# ---------------------------------------------------------------------------
# Rank ANOVA
# ---------------------------------------------------------------------------

def test_rank_eta_squared_is_one_for_perfect_grouping() -> None:
    labels = np.repeat([1, 2, 3], 5)
    values = np.repeat([0.0, 1.0, 2.0], 5)
    assert rank_eta_squared(values, labels) == pytest.approx(1.0)


def test_rank_eta_squared_is_small_for_unrelated_labels() -> None:
    rng = np.random.default_rng(2)
    labels = np.repeat([1, 2, 3, 4], 250)
    assert rank_eta_squared(rng.normal(size=labels.size), labels) < 0.05


def test_rank_eta_squared_survives_monotone_rescaling() -> None:
    """The point of ranking first: no normalisation can change the answer."""
    rng = np.random.default_rng(3)
    labels = np.repeat([1, 2, 3], 60)
    values = np.repeat([1.0, 2.0, 3.0], 60) + rng.normal(scale=0.3, size=180)
    plain = rank_eta_squared(values, labels)
    squashed = rank_eta_squared(np.log1p(values - values.min()), labels)
    assert plain == pytest.approx(squashed)


# ---------------------------------------------------------------------------
# Saturation
# ---------------------------------------------------------------------------

def test_saturated_fraction_counts_both_ends() -> None:
    values = np.array([0.0, 0.0, 0.3, 0.7, 1.0, 1.0])
    assert saturated_fraction(values) == pytest.approx(4 / 6)


def test_saturated_fraction_is_one_for_a_constant_signal() -> None:
    assert saturated_fraction(np.full(20, 0.5)) == 1.0


# ---------------------------------------------------------------------------
# Profile shape
# ---------------------------------------------------------------------------

def test_count_peaks_finds_one_on_a_unimodal_profile() -> None:
    medians = np.array([0.0, 0.3, 0.7, 1.0, 0.7, 0.3, 0.0])
    scatters = np.full(7, 0.01)
    assert count_peaks(medians, scatters) == 1


def test_count_peaks_finds_two_on_a_bimodal_profile() -> None:
    medians = np.array([0.0, 1.0, 0.2, 0.9, 0.0])
    scatters = np.full(5, 0.01)
    assert count_peaks(medians, scatters) == 2


def test_count_peaks_ignores_bumps_below_the_noise_floor() -> None:
    """A wobble smaller than within-step scatter is not a trap, it is noise."""
    medians = np.array([0.0, 1.0, 0.98, 0.99, 0.0])
    scatters = np.full(5, 0.2)
    assert count_peaks(medians, scatters) == 1


def test_monotonicity_is_one_on_a_clean_triangle() -> None:
    medians = np.array([0.0, 0.5, 1.0, 0.5, 0.0])
    assert monotonicity(medians, peak=2) == pytest.approx(1.0)


def test_monotonicity_penalises_a_step_going_the_wrong_way() -> None:
    medians = np.array([0.0, 0.5, 0.4, 1.0, 0.5, 0.0])
    assert monotonicity(medians, peak=3) < 1.0


def test_step_profile_returns_steps_in_order() -> None:
    labels = np.array([3, 1, 2, 1, 3, 2])
    values = np.array([30.0, 10.0, 20.0, 12.0, 32.0, 22.0])
    steps, medians, _ = step_profile(values, labels)
    assert steps.tolist() == [1, 2, 3]
    assert medians.tolist() == [11.0, 21.0, 31.0]


# ---------------------------------------------------------------------------
# Point-source ground truth
# ---------------------------------------------------------------------------

def _disc_image(radius: float, size: int = 240) -> np.ndarray:
    grid_y, grid_x = np.mgrid[0:size, 0:size]
    centre = size / 2.0
    distance = np.sqrt((grid_y - centre) ** 2 + (grid_x - centre) ** 2)
    image = np.where(distance <= radius, 200.0, 0.0)
    return image.astype(np.uint8)


def test_spot_radius_matches_the_analytic_value_for_a_disc() -> None:
    """Half the energy of a uniform disc of radius R lies inside R/sqrt(2)."""
    measured, _ = spot_radius(_disc_image(30.0), half=80)
    assert measured == pytest.approx(30.0 / np.sqrt(2.0), rel=0.1)


def test_spot_radius_grows_with_the_disc() -> None:
    small, _ = spot_radius(_disc_image(8.0), half=80)
    large, _ = spot_radius(_disc_image(24.0), half=80)
    assert large > small * 2.0


def test_spot_radius_reports_the_spot_centre() -> None:
    size = 240
    image = _disc_image(10.0, size)
    _, centre = spot_radius(image, half=80)
    assert centre[0] == pytest.approx(size / 2, abs=3)
    assert centre[1] == pytest.approx(size / 2, abs=3)


def test_spot_radius_ignores_a_constant_background() -> None:
    """The dim room behind the source must not enter the integral."""
    plain = _disc_image(12.0)
    lifted = np.clip(plain.astype(np.int32) + 20, 0, 255).astype(np.uint8)
    bare, _ = spot_radius(plain, half=80)
    raised, _ = spot_radius(lifted, half=80)
    assert raised == pytest.approx(bare, rel=0.05)


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------

def _run_entry(models: dict[str, dict[str, float]]) -> dict[str, object]:
    return {"models": models}


def test_aggregate_ranks_within_a_run_before_pooling() -> None:
    """Pooling must average ranks, not raw scores.

    One easy recording scoring everything highly should not be able to outvote
    a hard one; only the within-recording ordering carries over.
    """
    easy = _run_entry({
        "a": {"adj": 0.50, "eta2": 0.9, "sat": 0.0, "sep": 1.0, "mono": 1.0,
              "peaks": 1.0, "within_cv": 0.1},
        "b": {"adj": 0.99, "eta2": 0.9, "sat": 0.0, "sep": 1.0, "mono": 1.0,
              "peaks": 1.0, "within_cv": 0.1},
    })
    hard = _run_entry({
        "a": {"adj": 0.40, "eta2": 0.3, "sat": 0.0, "sep": 1.0, "mono": 1.0,
              "peaks": 1.0, "within_cv": 0.1},
        "b": {"adj": 0.35, "eta2": 0.3, "sat": 0.0, "sep": 1.0, "mono": 1.0,
              "peaks": 1.0, "within_cv": 0.1},
    })
    pooled = aggregate({"easy": easy, "hard": hard})
    # Each model wins one recording and loses the other, so the mean ranks tie,
    # even though b's mean raw score is far higher purely because the easy
    # recording inflates everything on it.
    assert pooled["a"]["adj_rank"] == pytest.approx(pooled["b"]["adj_rank"])
    assert pooled["b"]["adj"] > pooled["a"]["adj"]


def test_aggregate_treats_lower_as_better_for_cost_like_columns() -> None:
    run = _run_entry({
        "quiet": {"adj": 0.5, "eta2": 0.5, "sat": 0.1, "sep": 1.0, "mono": 1.0,
                  "peaks": 1.0, "within_cv": 0.1},
        "noisy": {"adj": 0.5, "eta2": 0.5, "sat": 0.9, "sep": 1.0, "mono": 1.0,
                  "peaks": 4.0, "within_cv": 0.9},
    })
    pooled = aggregate({"only": run})
    assert pooled["quiet"]["sat_rank"] < pooled["noisy"]["sat_rank"]
    assert pooled["quiet"]["peaks_rank"] < pooled["noisy"]["peaks_rank"]
