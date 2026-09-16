"""Tests for the running normaliser, and for the defects found on real data.

Each class below locks in one behaviour that the protocol recordings showed was
broken.  The numbers in the docstrings are measured, not illustrative: they come
from data/*/frames.csv and the caches built by tools/protocol_study.py.
"""
from __future__ import annotations

from dataclasses import replace

import math

import numpy as np
import pytest

from adaptive_sharpness.analysis import ImageAnalyzer, _half_normal_scale
from adaptive_sharpness.config import AnalysisConfig, NormalizationConfig
from adaptive_sharpness.normalize import NormalizerBank, RunningNormalizer


@pytest.fixture
def config() -> NormalizationConfig:
    return NormalizationConfig()


@pytest.fixture
def rolling() -> NormalizationConfig:
    """A short, never-frozen scale.

    The shipped defaults are a long window that freezes once the range settles,
    which is the point of the change.  Tests about the *rolling* behaviour have
    to ask for it explicitly or they measure the frozen scale instead.
    """
    return NormalizationConfig(window=120, auto_freeze=False)


# ---------------------------------------------------------------------------
# The degenerate-range guard
# ---------------------------------------------------------------------------

class TestDegenerateRangeGuard:
    """The guard must be relative to the metric, not to the number 1.

    It used to be ``ratio * max(|hi|, |lo|, 1.0)``.  Every metric here is
    contrast-normalised by mean(I)**2 and therefore lives far below 1, so the
    1.0 made the guard an absolute 1e-3.  Measured at defocus, the entire value
    of `brenner` is 2.9e-4 - smaller than the floor meant to guard it - so the
    normaliser returned its neutral sentinel on every defocused frame.
    """

    def test_small_metric_with_real_range_is_not_called_degenerate(
        self, config: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer("brenner", config)
        # Values of the magnitude actually recorded for brenner over a sweep.
        for value in np.linspace(1e-5, 6e-4, 60):
            normalizer.observe(float(value))
        assert not normalizer.is_degenerate

    def test_the_old_absolute_floor_would_have_called_it_degenerate(
        self, config: NormalizationConfig
    ) -> None:
        """Guards the regression, by reproducing the old setting explicitly."""
        old = replace(config, min_range_absolute=1.0, long_window_multiple=0)
        normalizer = RunningNormalizer("brenner", old)
        for value in np.linspace(1e-5, 6e-4, 60):
            normalizer.observe(float(value))
        assert normalizer.is_degenerate

    def test_a_genuinely_static_signal_is_still_degenerate(
        self, config: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer("stuck", config)
        for _ in range(60):
            normalizer.observe(0.25)
        assert normalizer.is_degenerate
        assert normalizer.normalize(0.25) == 0.5

    def test_small_metric_spans_the_full_output_range(
        self, config: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer("brenner", config)
        values = np.linspace(1e-5, 6e-4, 60)
        for value in values:
            normalizer.observe(float(value))
        assert normalizer.normalize(float(values[-1]), observe=False) > 0.9
        assert normalizer.normalize(float(values[0]), observe=False) < 0.1


# ---------------------------------------------------------------------------
# The long-horizon fallback
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("rolling")
class TestLongHorizonFallback:
    """A held focus position has no range; the sweep it belongs to does.

    This is the defect the operator reported: at full defocus the score read
    0.5, while a *sharper* frame at mild defocus read 0.001.  0.5 outranks
    0.001, so a hill-climbing search is pushed away from focus.
    """

    @staticmethod
    def _sweep_then_hold(normalizer: RunningNormalizer) -> None:
        # A sweep from sharp to blurred, then a long hold at the blurred end,
        # long enough to fill the rolling window with the held value alone.
        for value in np.linspace(1.0, 0.01, 200):
            normalizer.observe(float(value))
        for _ in range(normalizer.config.window):
            normalizer.observe(0.01)

    def test_held_defocus_scores_low_not_neutral(
        self, rolling: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer("tenengrad", rolling)
        self._sweep_then_hold(normalizer)
        assert normalizer.normalize(0.01, observe=False) < 0.2

    def test_without_the_fallback_it_returns_the_sentinel(
        self, rolling: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer(
            "tenengrad", replace(rolling, long_window_multiple=0)
        )
        self._sweep_then_hold(normalizer)
        assert normalizer.normalize(0.01, observe=False) == 0.5

    def test_deep_defocus_never_outranks_mild_defocus(
        self, rolling: NormalizationConfig
    ) -> None:
        """The inversion the operator reported, stated as a test.

        Both values are scored against the *same* normaliser state, which is
        the situation a search is in: it holds at deep defocus and has to know
        that a milder defocus would be better.
        """
        normalizer = RunningNormalizer("tenengrad", rolling)
        self._sweep_then_hold(normalizer)
        deep = normalizer.normalize(0.01, observe=False)
        mild = normalizer.normalize(0.2, observe=False)
        assert deep < mild

    def test_the_old_behaviour_loses_that_ordering(
        self, rolling: NormalizationConfig
    ) -> None:
        """Without the fallback both values collapse onto the same sentinel."""
        normalizer = RunningNormalizer(
            "tenengrad", replace(rolling, long_window_multiple=0)
        )
        self._sweep_then_hold(normalizer)
        assert normalizer.normalize(0.01, observe=False) == 0.5
        assert normalizer.normalize(0.2, observe=False) == 0.5

    def test_fallback_does_not_override_a_usable_window(
        self, rolling: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer("tenengrad", rolling)
        for value in np.linspace(0.0, 1.0, 400):
            normalizer.observe(float(value))
        # The rolling window covers only the top of the ramp, so 0.8 sits low
        # within it (~0.33).  Scaled against the whole history it would land
        # near 0.8, so the threshold below discriminates between the two.
        assert normalizer.normalize(0.8, observe=False) < 0.5

    def test_reset_clears_the_long_history_too(
        self, rolling: NormalizationConfig
    ) -> None:
        normalizer = RunningNormalizer("tenengrad", rolling)
        self._sweep_then_hold(normalizer)
        normalizer.reset()
        assert normalizer.sample_count == 0
        assert normalizer.is_degenerate


# ---------------------------------------------------------------------------
# The noise estimator
# ---------------------------------------------------------------------------

class TestNoiseQuantile:
    """The median of |HH| is zero on a quantised stream, so sigma read zero.

    Measured: 58-74% of the Haar HH coefficients of the GH6 live view are
    exactly zero, and noise_sigma was exactly 0 on 100% of frames in six of
    eight recordings.
    """

    def test_median_setting_reproduces_the_classical_constant(self) -> None:
        assert 1.0 / _half_normal_scale(50.0) == pytest.approx(1.482602218505602)

    def test_estimator_is_unbiased_on_continuous_noise(self) -> None:
        rng = np.random.default_rng(4)
        analyzer = ImageAnalyzer(AnalysisConfig())
        field = rng.normal(128.0, 5.0, (256, 256)).astype(np.float32)
        assert analyzer.estimate_noise_sigma(field) == pytest.approx(5.0, rel=0.1)

    @staticmethod
    def _zero_inflated(zero_fraction: float = 0.70) -> np.ndarray:
        """A band with the same defect as the camera's: mostly exact zeros.

        Measured on real GH6 frames: 58-74% of |HH| is exactly zero.  0.70 sits
        inside that range, so the median is zero and any quantile below it
        reports no noise at all.
        """
        rng = np.random.default_rng(5)
        image = np.zeros((200, 200), dtype=np.float32)
        noisy_rows = int(round(image.shape[0] * (1.0 - zero_fraction)))
        image[:noisy_rows] = rng.normal(128.0, 8.0, (noisy_rows, image.shape[1]))
        return image

    def test_median_variant_is_blind_to_a_zero_inflated_band(self) -> None:
        analyzer = ImageAnalyzer(AnalysisConfig(noise_quantile=50.0))
        assert analyzer.estimate_noise_sigma(self._zero_inflated()) == 0.0

    def test_a_high_quantile_still_sees_it(self) -> None:
        analyzer = ImageAnalyzer(AnalysisConfig(noise_quantile=90.0))
        assert analyzer.estimate_noise_sigma(self._zero_inflated()) > 1.0

    def test_rejects_a_degenerate_quantile(self) -> None:
        with pytest.raises(ValueError):
            ImageAnalyzer(AnalysisConfig(noise_quantile=0.0))
        with pytest.raises(ValueError):
            ImageAnalyzer(AnalysisConfig(noise_quantile=100.0))


# ---------------------------------------------------------------------------
# The bank
# ---------------------------------------------------------------------------

class TestInformativeFraction:
    def test_reports_the_share_of_metrics_carrying_information(self) -> None:
        config = NormalizationConfig()
        bank = NormalizerBank(("a", "b"), config)
        for index in range(60):
            bank.normalize({"a": float(index) * 1e-5, "b": 0.25})
        assert bank.informative_fraction == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The output mapping
# ---------------------------------------------------------------------------

class TestMapping:
    """Clipping a linear map destroys the ordering a focus search needs.

    Measured against the point-source ground truth: a frozen linear map pins
    41% of frames at 0 or 1 and reaches rank correlation 0.88 and 0.84, while
    the logistic map pins 0.3% and reaches 0.94 and 0.97.
    """

    @staticmethod
    def _ramped(mapping: str) -> RunningNormalizer:
        config = NormalizationConfig(mapping=mapping, auto_freeze=False)
        normalizer = RunningNormalizer("tenengrad", config)
        for value in np.linspace(0.0, 1.0, 400):
            normalizer.observe(float(value))
        return normalizer

    def test_linear_ties_everything_past_the_anchor(self) -> None:
        normalizer = self._ramped("linear")
        assert normalizer.normalize(5.0, observe=False) == pytest.approx(
            normalizer.normalize(50.0, observe=False)
        )

    def test_logistic_keeps_them_apart(self) -> None:
        normalizer = self._ramped("logistic")
        assert (
            normalizer.normalize(1.5, observe=False)
            < normalizer.normalize(2.0, observe=False)
        )

    def test_logistic_stays_inside_the_unit_interval(self) -> None:
        normalizer = self._ramped("logistic")
        for value in (0.0, 1e-9, 0.5, 1e6):
            assert 0.0 <= normalizer.normalize(float(value), observe=False) <= 1.0

    def test_logistic_is_monotone_across_the_range(self) -> None:
        normalizer = self._ramped("logistic")
        values = [
            normalizer.normalize(float(v), observe=False)
            for v in np.linspace(0.0, 2.0, 50)
        ]
        assert all(a <= b for a, b in zip(values, values[1:]))

    def test_anchors_land_where_the_gain_says(self) -> None:
        """At gain 4 the two anchors sit at 0.12 and 0.88, as documented."""
        normalizer = self._ramped("logistic")
        anchors = normalizer._percentiles(normalizer._history)
        assert anchors is not None
        assert normalizer._map(anchors[0], anchors) == pytest.approx(0.119, abs=0.01)
        assert normalizer._map(anchors[1], anchors) == pytest.approx(0.881, abs=0.01)

    def test_rejects_an_unknown_mapping(self) -> None:
        with pytest.raises(ValueError, match="unknown normalization mapping"):
            RunningNormalizer("x", NormalizationConfig(mapping="bogus"))

    def test_rejects_a_non_positive_gain(self) -> None:
        with pytest.raises(ValueError, match="logistic_gain"):
            RunningNormalizer("x", NormalizationConfig(logistic_gain=0.0))


class TestAutoFreeze:
    """A rolling scale rescores the same frame differently over time.

    Measured: the raw metric tracks the physical spot size at rank correlation
    0.97-0.98, while the rolling scale reaches -0.24 and -0.38.  Freezing the
    anchors once the observed range stops growing is what recovers it.
    """

    @staticmethod
    def _sweep(normalizer: RunningNormalizer, samples: int = 900) -> None:
        # Blurred, through focus, and out again - what a real sweep looks like.
        ramp = np.concatenate([
            np.linspace(0.01, 1.0, samples // 2),
            np.linspace(1.0, 0.01, samples - samples // 2),
        ])
        for value in ramp:
            normalizer.observe(float(value))

    def test_freezes_after_the_range_settles(self) -> None:
        normalizer = RunningNormalizer("tenengrad", NormalizationConfig())
        self._sweep(normalizer)
        assert normalizer.frozen

    def test_does_not_freeze_when_switched_off(self) -> None:
        normalizer = RunningNormalizer(
            "tenengrad", NormalizationConfig(auto_freeze=False)
        )
        self._sweep(normalizer)
        assert not normalizer.frozen

    def test_does_not_freeze_while_the_range_keeps_growing(self) -> None:
        """Slow sustained growth must not read as stability.

        The test used to compare each span against a maximum updated on every
        sample, so growth below the threshold was never accumulated: at 1% a
        sample against a 5% threshold the scale froze after 119 samples with
        the span 3.3x larger than when the interval began.
        """
        normalizer = RunningNormalizer("tenengrad", NormalizationConfig())
        high = 1.0
        for _ in range(400):
            high *= 1.01
            for value in np.linspace(0.0, high, 3):
                normalizer.observe(float(value))
            if normalizer.frozen:
                break
        assert not normalizer.frozen

    def test_does_not_freeze_before_enough_samples(self) -> None:
        normalizer = RunningNormalizer("tenengrad", NormalizationConfig())
        for value in np.linspace(0.0, 1.0, 100):
            normalizer.observe(float(value))
        assert not normalizer.frozen

    def test_a_frozen_scale_scores_a_repeated_frame_identically(self) -> None:
        """The property the whole change exists for."""
        normalizer = RunningNormalizer("tenengrad", NormalizationConfig())
        self._sweep(normalizer)
        first = normalizer.normalize(0.4)
        for _ in range(200):
            normalizer.normalize(0.9)
        assert normalizer.normalize(0.4) == pytest.approx(first, abs=1e-6)

    def test_a_rolling_scale_does_not(self) -> None:
        normalizer = RunningNormalizer(
            "tenengrad", NormalizationConfig(auto_freeze=False)
        )
        self._sweep(normalizer)
        first = normalizer.normalize(0.4)
        for _ in range(200):
            normalizer.normalize(0.9)
        assert normalizer.normalize(0.4) != pytest.approx(first, abs=1e-3)

    def test_thawing_is_off_by_default(self) -> None:
        """No threshold tested could tell a focus change from a scene change."""
        normalizer = RunningNormalizer("tenengrad", NormalizationConfig())
        self._sweep(normalizer)
        for _ in range(normalizer.config.window):
            normalizer.normalize(500.0)
        assert normalizer.frozen

    def test_thaws_and_recalibrates_when_switched_on(self) -> None:
        """Thawing is not an end state: it re-freezes on the new scene.

        What matters is that the anchors end up describing where the signal now
        lives, not that the normaliser is caught mid-thaw.
        """
        normalizer = RunningNormalizer(
            "tenengrad", NormalizationConfig(auto_thaw=True)
        )
        self._sweep(normalizer)
        before = normalizer._frozen_anchors
        assert before is not None
        for _ in range(normalizer.config.window):
            normalizer.normalize(500.0)
        after = normalizer._frozen_anchors
        assert after is not None and after[1] > before[1] * 5

    def test_keeps_the_ordering_within_the_new_scene_after_thawing(self) -> None:
        """The re-frozen anchors still span the old scene as well as the new,
        so the new range occupies only part of the output.  Ordering is the
        property that has to survive; absolute placement is not."""
        normalizer = RunningNormalizer(
            "tenengrad", NormalizationConfig(auto_thaw=True)
        )
        self._sweep(normalizer)
        for value in np.linspace(300.0, 700.0, normalizer.config.window):
            normalizer.normalize(float(value))
        assert (
            normalizer.normalize(300.0, observe=False)
            < normalizer.normalize(700.0, observe=False)
        )

    def test_unfreeze_clears_the_automatic_state(self) -> None:
        normalizer = RunningNormalizer("tenengrad", NormalizationConfig())
        self._sweep(normalizer)
        normalizer.unfreeze()
        assert not normalizer.frozen


class TestExplicitFreeze:
    """An explicit freeze is a statement about the scale, not a guess.

    The automatic rule must not overrule it, and it must recompute the anchors
    rather than read back whatever the automatic rule already installed - which
    would make freeze() a silent no-op and make fit() ignore its own batch.
    """

    def test_fit_uses_the_whole_batch_not_the_tail(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        # 1500 samples into a 960-deep history: the first third does not fit,
        # and must still shape the anchors.
        normalizer.fit(np.linspace(0.0, 10.0, 1500))
        low, high = normalizer._frozen_anchors  # type: ignore[misc]
        assert low == pytest.approx(math.log1p(0.5), abs=0.02)
        assert high == pytest.approx(math.log1p(9.5), abs=0.02)

    def test_fit_after_an_automatic_freeze_still_uses_the_batch(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        # A sweep out and back, so the observed range settles and the automatic
        # rule fires.  A one-way ramp never settles and never freezes, which is
        # the point of the stability test.
        ramp = np.concatenate([np.linspace(0.01, 1.0, 450), np.linspace(1.0, 0.01, 450)])
        for value in ramp:
            normalizer.observe(float(value))
        assert normalizer.frozen
        normalizer.fit(np.linspace(0.0, 10.0, 1500))
        assert normalizer._frozen_anchors[1] == pytest.approx(  # type: ignore[index]
            math.log1p(9.5), abs=0.02
        )

    def test_explicit_freeze_survives_later_data(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        for value in np.linspace(0.0, 1.0, 400):
            normalizer.observe(float(value))
        normalizer.freeze()
        before = normalizer._frozen_anchors
        for value in np.linspace(50.0, 100.0, 2000):
            normalizer.observe(float(value))
        assert normalizer._frozen_anchors == before

    def test_fit_rejects_too_few_samples(self) -> None:
        with pytest.raises(RuntimeError, match="at least 2 samples"):
            RunningNormalizer("m", NormalizationConfig()).fit([1.0])
