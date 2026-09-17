"""Tests for the adaptive weighting, the confidence, and the ablation switches."""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_sharpness.config import EnsembleConfig, MetricsConfig
from adaptive_sharpness.ensemble import AdaptiveEnsemble
from adaptive_sharpness.types import ImageStats

NAMES = ("laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width")

IDEAL = {"noise": 0.0, "edge": 0.0, "clip": 0.0, "motion": 0.0, "contrast": 0.0}
NOISY = {**IDEAL, "noise": 1.0}
NO_EDGES = {**IDEAL, "edge": 1.0}
MOVING = {**IDEAL, "motion": 1.0}

GOOD_STATS = ImageStats(
    noise_sigma=0.5, noise_level=0.08, snr=40.0, local_contrast=0.18,
    brightness=0.45, edge_density=0.08, edge_sufficiency=1.0,
    motion_px=0.1, motion_level=0.03,
)


def make(config: EnsembleConfig | None = None) -> AdaptiveEnsemble:
    return AdaptiveEnsemble(NAMES, config or EnsembleConfig(), MetricsConfig().priors)


def uniform_scores(value: float = 0.6) -> dict[str, float]:
    return {name: value for name in NAMES}


class TestWeightInvariants:
    @pytest.mark.parametrize("degradations", [IDEAL, NOISY, NO_EDGES, MOVING])
    def test_weights_sum_to_one(self, degradations) -> None:
        out = make().combine(uniform_scores(), GOOD_STATS, degradations)
        assert sum(out.weights.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("degradations", [IDEAL, NOISY, NO_EDGES, MOVING])
    def test_weights_are_non_negative(self, degradations) -> None:
        out = make().combine(uniform_scores(), GOOD_STATS, degradations)
        assert all(w >= 0.0 for w in out.weights.values())

    def test_no_metric_is_ever_fully_silenced(self) -> None:
        """The floor keeps the ensemble able to recover when conditions improve."""
        brutal = {"noise": 1.0, "edge": 1.0, "clip": 1.0, "motion": 1.0, "contrast": 1.0}
        out = make().combine(uniform_scores(), GOOD_STATS, brutal)
        assert all(w > 0.0 for w in out.weights.values())

    def test_score_is_bounded(self) -> None:
        out = make().combine(uniform_scores(1.0), GOOD_STATS, IDEAL)
        assert 0.0 <= out.score <= 1.0

    def test_uniform_metrics_give_that_value(self) -> None:
        """With every metric agreeing, the weighted sum must return that value."""
        out = make().combine(uniform_scores(0.42), GOOD_STATS, IDEAL)
        assert out.score == pytest.approx(0.42)

    def test_reliabilities_are_bounded(self) -> None:
        out = make().combine(uniform_scores(), GOOD_STATS, NOISY)
        assert all(0.0 < r <= 1.0 for r in out.reliabilities.values())


class TestAdaptiveBehaviour:
    def test_noise_penalises_the_laplacian_most(self) -> None:
        """The second-derivative operator amplifies noise more than the others."""
        ensemble = make()
        clean = ensemble.combine(uniform_scores(), GOOD_STATS, IDEAL).weights
        noisy = ensemble.combine(uniform_scores(), GOOD_STATS, NOISY).weights
        laplacian_drop = noisy["laplacian"] / clean["laplacian"]
        brenner_drop = noisy["brenner"] / clean["brenner"]
        assert laplacian_drop < brenner_drop, (
            "the Laplacian should lose more weight under noise than Brenner"
        )

    def test_missing_edges_penalise_the_edge_metric_most(self) -> None:
        ensemble = make()
        clean = ensemble.combine(uniform_scores(), GOOD_STATS, IDEAL).weights
        bare = ensemble.combine(uniform_scores(), GOOD_STATS, NO_EDGES).weights
        drops = {name: bare[name] / clean[name] for name in NAMES}
        assert drops["edge_width"] == min(drops.values())

    def test_reliability_decreases_monotonically_with_degradation(self) -> None:
        ensemble = make()
        previous = None
        for level in (0.0, 0.25, 0.5, 0.75, 1.0):
            reliabilities = ensemble.reliabilities({**IDEAL, "noise": level})
            current = float(reliabilities[0])
            if previous is not None:
                assert current <= previous + 1e-12
            previous = current

    def test_an_outlier_metric_is_down_weighted_when_enabled(self) -> None:
        """One metric fooled by a highlight must not drag the score with it.

        This is what the consensus stage is *for*.  It is off by default
        because it cost measurably on eight real recordings and none of them
        contained this failure - see EnsembleConfig.use_agreement - so the
        mechanism is tested here with it switched on deliberately.
        """
        scores = {name: 0.5 for name in NAMES}
        scores["fourier"] = 0.05
        out = make(EnsembleConfig(use_agreement=True)).combine(
            scores, GOOD_STATS, IDEAL
        )
        others = [out.weights[n] for n in NAMES if n != "fourier"]
        assert out.weights["fourier"] < min(others)
        # The consensus value must dominate the outlier.
        assert out.score > 0.4

    def test_agreement_is_off_by_default(self) -> None:
        scores = {name: 0.5 for name in NAMES}
        scores["fourier"] = 0.05
        out = make().combine(scores, GOOD_STATS, IDEAL)
        assert all(a == pytest.approx(1.0) for a in out.agreements.values())

    def test_disabling_agreement_does_not_blind_the_confidence(self) -> None:
        """Two mechanisms share one calculation; only one of them is off.

        The kernel reweights the metrics; the dispersion feeds the concordance
        factor of the confidence.  Switching off the first must leave the
        second reporting disagreement.
        """
        # Broad disagreement, not one outlier: the dispersion is a MAD and is
        # deliberately unmoved by a single dissenting metric.
        scores = dict(zip(NAMES, [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]))
        disagreeing = make().combine(scores, GOOD_STATS, IDEAL)
        agreeing = make().combine(uniform_scores(0.5), GOOD_STATS, IDEAL)
        assert disagreeing.dispersion > agreeing.dispersion
        assert disagreeing.confidence < agreeing.confidence

    def test_a_single_outlier_does_not_move_the_dispersion(self) -> None:
        """Documents the limit of the MAD: one dissenter is invisible to it."""
        scores = {name: 0.5 for name in NAMES}
        scores["fourier"] = 0.05
        assert make().combine(scores, GOOD_STATS, IDEAL).dispersion == 0.0

    def test_dispersion_reports_disagreement(self) -> None:
        agree = make().combine(uniform_scores(), GOOD_STATS, IDEAL)
        scores = dict(zip(NAMES, [0.1, 0.9, 0.2, 0.8, 0.15, 0.85]))
        disagree = make().combine(scores, GOOD_STATS, IDEAL)
        assert disagree.dispersion > agree.dispersion


class TestAblationSwitches:
    def test_noise_compensation_off_ignores_noise(self) -> None:
        config = EnsembleConfig(use_noise_compensation=False)
        ensemble = make(config)
        clean = ensemble.combine(uniform_scores(), GOOD_STATS, IDEAL).weights
        noisy = ensemble.combine(uniform_scores(), GOOD_STATS, NOISY).weights
        for name in NAMES:
            assert noisy[name] == pytest.approx(clean[name])

    def test_motion_compensation_off_ignores_motion(self) -> None:
        config = EnsembleConfig(use_motion_compensation=False)
        ensemble = make(config)
        still = ensemble.combine(uniform_scores(), GOOD_STATS, IDEAL).weights
        moving = ensemble.combine(uniform_scores(), GOOD_STATS, MOVING).weights
        for name in NAMES:
            assert moving[name] == pytest.approx(still[name])

    def test_switches_are_independent(self) -> None:
        """Turning noise compensation off must not disable the motion term."""
        ensemble = make(EnsembleConfig(use_noise_compensation=False))
        still = ensemble.combine(uniform_scores(), GOOD_STATS, IDEAL).weights
        moving = ensemble.combine(uniform_scores(), GOOD_STATS, MOVING).weights
        assert any(
            moving[name] != pytest.approx(still[name]) for name in NAMES
        )


class TestConfidence:
    def test_good_conditions_give_high_confidence(self) -> None:
        out = make().combine(uniform_scores(), GOOD_STATS, IDEAL)
        assert out.confidence > 0.8

    def test_no_edges_gives_low_confidence(self) -> None:
        stats = ImageStats(edge_sufficiency=0.0, snr=40.0, local_contrast=0.01)
        out = make().combine(uniform_scores(), stats, NO_EDGES)
        assert out.confidence < 0.2

    def test_disagreement_lowers_confidence(self) -> None:
        agreeing = make().combine(uniform_scores(), GOOD_STATS, IDEAL)
        scores = dict(zip(NAMES, [0.05, 0.95, 0.1, 0.9, 0.15, 0.85]))
        disagreeing = make().combine(scores, GOOD_STATS, IDEAL)
        assert disagreeing.confidence < agreeing.confidence

    def test_saturated_motion_does_not_annihilate_confidence(self) -> None:
        """Motion degrades a measurement; it does not make one impossible.

        Regression: the weakest-link cap applied to every factor, so a single
        saturated soft factor produced exactly 0.0. Measured on a 1288-frame
        handheld recording that fired on 39% of frames, 97% of them purely
        because the motion factor had saturated while edges, SNR, exposure and
        contrast were all fine.
        """
        stats = ImageStats(
            snr=40.0, local_contrast=0.18, edge_sufficiency=1.0, motion_level=1.0
        )
        out = make().combine(uniform_scores(), stats, MOVING)
        assert out.confidence > 0.15, "saturated motion must not zero the confidence"
        assert out.confidence < 0.6, "but it must still reduce it substantially"

    def test_missing_edges_still_annihilates_confidence(self) -> None:
        """The other half of the same rule: a necessary factor still caps hard."""
        stats = ImageStats(edge_sufficiency=0.0, snr=40.0, local_contrast=0.01)
        assert make().combine(uniform_scores(), stats, NO_EDGES).confidence < 0.1

    def test_motion_lowers_confidence(self) -> None:
        still = make().combine(uniform_scores(), GOOD_STATS, IDEAL)
        moving_stats = ImageStats(
            snr=40.0, local_contrast=0.18, edge_sufficiency=1.0, motion_level=1.0
        )
        moving = make().combine(uniform_scores(), moving_stats, MOVING)
        assert moving.confidence < still.confidence

    def test_warmup_lowers_confidence(self) -> None:
        warm = make().combine(uniform_scores(), GOOD_STATS, IDEAL, warmed_up=True)
        cold = make().combine(uniform_scores(), GOOD_STATS, IDEAL, warmed_up=False)
        assert cold.confidence < warm.confidence

    def test_confidence_is_bounded(self) -> None:
        for degradations in (IDEAL, NOISY, NO_EDGES, MOVING):
            out = make().combine(uniform_scores(), GOOD_STATS, degradations)
            assert 0.0 <= out.confidence <= 1.0

    def test_components_are_reported(self) -> None:
        out = make().combine(uniform_scores(), GOOD_STATS, IDEAL)
        assert {"edges", "snr", "exposure", "contrast", "concordance", "motion"} <= set(
            out.confidence_components
        )


class TestValidation:
    def test_rejects_empty_metric_list(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            AdaptiveEnsemble((), EnsembleConfig(), {})

    def test_rejects_metric_without_a_profile(self) -> None:
        with pytest.raises(KeyError, match="sensitivity profile"):
            AdaptiveEnsemble(("mystery",), EnsembleConfig(), {"mystery": 1.0})

    def test_rejects_unknown_degradation_key(self) -> None:
        config = EnsembleConfig(sensitivity={"laplacian": {"gremlins": 1.0}})
        with pytest.raises(KeyError, match="gremlins"):
            AdaptiveEnsemble(("laplacian",), config, {"laplacian": 1.0})

    def test_rejects_negative_sensitivity(self) -> None:
        config = EnsembleConfig(sensitivity={"laplacian": {"noise": -1.0}})
        with pytest.raises(ValueError, match="non-negative"):
            AdaptiveEnsemble(("laplacian",), config, {"laplacian": 1.0})

    def test_rejects_all_zero_priors(self) -> None:
        with pytest.raises(ValueError, match="positive prior"):
            AdaptiveEnsemble(NAMES, EnsembleConfig(), {n: 0.0 for n in NAMES})

    def test_missing_metric_value_raises(self) -> None:
        partial = {name: 0.5 for name in NAMES[:-1]}
        with pytest.raises(KeyError, match="missing normalised"):
            make().combine(partial, GOOD_STATS, IDEAL)


class TestSingleMetric:
    def test_one_metric_gets_all_the_weight(self) -> None:
        ensemble = AdaptiveEnsemble(("laplacian",), EnsembleConfig(), {"laplacian": 1.0})
        out = ensemble.combine({"laplacian": 0.7}, GOOD_STATS, IDEAL)
        assert out.weights["laplacian"] == pytest.approx(1.0)
        assert out.score == pytest.approx(0.7)
        assert out.dispersion == 0.0
