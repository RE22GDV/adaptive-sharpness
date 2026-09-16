"""Tests for the shared frame selection and ordering statistics.

Both were found by review to be wrong in ways that changed published numbers:
the ordering statistic resolved ties in favour of one direction, and the three
analysis tools each selected frames differently.  Each defect is pinned here.
"""
from __future__ import annotations

import numpy as np
import pytest

from tools.evaluation import (
    config_fingerprint,
    file_fingerprint,
    ordering_against_truth,
    peak_interval,
    provenance,
    select_frames,
)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_perfect_agreement(self) -> None:
        model = np.array([1.0, 2.0, 3.0, 4.0])
        result = ordering_against_truth(model, model.copy())
        assert result.wrong == 0
        assert result.tied == 0
        assert result.inversion == 0.0

    def test_perfect_disagreement(self) -> None:
        model = np.array([1.0, 2.0, 3.0, 4.0])
        result = ordering_against_truth(model, model[::-1].copy())
        assert result.correct == 0
        assert result.inversion == 1.0

    def test_a_constant_signal_scores_chance_whichever_way_truth_runs(self) -> None:
        """The defect: ties used to be resolved as "the second one is sharper".

        A constant signal therefore scored 0.000 inversions against one
        reference direction and 1.000 against the other, while in fact it
        distinguishes nothing at all.
        """
        constant = np.full(5, 0.5)
        rising = np.arange(5, dtype=float)
        falling = rising[::-1].copy()
        up = ordering_against_truth(constant, rising)
        down = ordering_against_truth(constant, falling)
        assert up.inversion == pytest.approx(0.5)
        assert down.inversion == pytest.approx(0.5)
        assert up.correct == up.wrong == 0
        assert up.tied == up.comparable

    def test_ties_are_counted_not_hidden(self) -> None:
        model = np.array([1.0, 1.0, 3.0])
        result = ordering_against_truth(model, np.array([1.0, 2.0, 3.0]))
        assert result.tied == 1
        assert result.resolved == pytest.approx(2 / 3)

    def test_strict_inversion_ignores_ties(self) -> None:
        model = np.array([1.0, 1.0, 3.0])
        result = ordering_against_truth(model, np.array([1.0, 2.0, 3.0]))
        assert result.inversion_strict == 0.0
        assert result.inversion > 0.0

    def test_reference_ties_are_excluded(self) -> None:
        result = ordering_against_truth(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 3.0])
        )
        assert result.reference_tied == 1
        assert result.comparable == 2

    def test_result_is_independent_of_pair_order(self) -> None:
        rng = np.random.default_rng(0)
        model = rng.normal(size=8)
        truth = rng.normal(size=8)
        forward = ordering_against_truth(model, truth)
        order = np.argsort(rng.normal(size=8))
        shuffled = ordering_against_truth(model[order], truth[order])
        assert forward.inversion == pytest.approx(shuffled.inversion)

    def test_non_finite_values_are_skipped(self) -> None:
        model = np.array([1.0, np.nan, 3.0])
        result = ordering_against_truth(model, np.array([1.0, 2.0, 3.0]))
        assert result.comparable == 1


class TestPeakInterval:
    def test_single_maximum(self) -> None:
        assert peak_interval(np.array([0.0, 1.0, 0.5])) == (1, 1)

    def test_plateau_reports_both_ends(self) -> None:
        """argmax answers a different question than "where is the peak"."""
        assert peak_interval(np.array([0.0, 1.0, 1.0, 1.0, 0.2])) == (1, 3)

    def test_all_nan(self) -> None:
        assert peak_interval(np.full(3, np.nan)) == (-1, -1)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelection:
    @staticmethod
    def _inputs(n: int = 10):
        step = np.arange(1, n + 1)
        hold = np.ones(n, dtype=bool)
        return step, hold

    def test_moving_frames_are_excluded(self) -> None:
        step, hold = self._inputs()
        hold[:3] = False
        selection = select_frames(step=step, hold=hold)
        assert selection.count == 7
        assert selection.reasons["moving"] == 3

    def test_unlabelled_frames_are_excluded(self) -> None:
        step, hold = self._inputs()
        step[:2] = 0
        selection = select_frames(step=step, hold=hold)
        assert selection.reasons["no_step_label"] == 2

    def test_a_wandering_spot_is_excluded(self) -> None:
        step, hold = self._inputs()
        radius = np.full(10, 5.0)
        offset = np.zeros(10)
        offset[:4] = 50.0
        selection = select_frames(
            step=step, hold=hold, spot_radius=radius, spot_offset=offset
        )
        assert selection.reasons["spot_wandered"] == 4
        assert selection.count == 6

    def test_reference_is_applied_when_present_even_if_not_required(self) -> None:
        """The two tools disagreed here, by 3.3% and 5.5% of frames."""
        step, hold = self._inputs()
        radius = np.full(10, 5.0)
        radius[0] = np.nan
        offset = np.zeros(10)
        selection = select_frames(
            step=step, hold=hold, spot_radius=radius, spot_offset=offset
        )
        assert selection.count == 9

    def test_reasons_do_not_double_count(self) -> None:
        step, hold = self._inputs()
        hold[:3] = False
        step[:3] = 0
        selection = select_frames(step=step, hold=hold)
        assert selection.reasons["moving"] == 3
        assert "no_step_label" not in selection.reasons
        assert selection.count == 7

    def test_summary_accounts_for_every_frame(self) -> None:
        step, hold = self._inputs()
        hold[:2] = False
        step[5:7] = 0
        selection = select_frames(step=step, hold=hold)
        summary = selection.summary()
        assert summary["selected"] + sum(summary["excluded"].values()) == summary["total_frames"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_config_fingerprint_changes_with_a_setting(self) -> None:
        from dataclasses import replace

        from adaptive_sharpness import load_default_config

        config = load_default_config()
        changed = replace(
            config, analysis=replace(config.analysis, noise_quantile=90.0)
        )
        assert config_fingerprint(config) != config_fingerprint(changed)

    def test_config_fingerprint_is_stable(self) -> None:
        from adaptive_sharpness import load_default_config

        assert config_fingerprint(load_default_config()) == config_fingerprint(
            load_default_config()
        )

    def test_file_fingerprint_notices_a_changed_file(self, tmp_path) -> None:
        first = tmp_path / "a.bin"
        first.write_bytes(b"hello")
        before = file_fingerprint([first])
        first.write_bytes(b"world")
        assert file_fingerprint([first]) != before

    def test_file_fingerprint_notices_a_reordering(self, tmp_path) -> None:
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"aaaa")
        b.write_bytes(b"bbbb")
        assert file_fingerprint([a, b]) != file_fingerprint([b, a])

    def test_provenance_records_what_produced_a_number(self) -> None:
        from adaptive_sharpness import load_default_config

        record = provenance(load_default_config(), group="test")
        for key in ("commit", "python", "numpy", "opencv", "config_fingerprint"):
            assert key in record
        assert record["group"] == "test"
