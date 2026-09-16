"""Unit tests for the individual sharpness metrics."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from adaptive_sharpness.config import MetricsConfig
from adaptive_sharpness.metrics import (
    Brenner,
    EdgeWidth,
    FourierHighFrequency,
    LaplacianVariance,
    Tenengrad,
    WaveletEnergy,
    available_metrics,
    build_metrics,
    haar_decompose,
)
from adaptive_sharpness.metrics.base import MetricError, normalization_factor
from adaptive_sharpness.synthetic import apply_exposure, defocus


ALL_METRIC_CLASSES = (
    LaplacianVariance,
    Tenengrad,
    Brenner,
    WaveletEnergy,
    FourierHighFrequency,
    EdgeWidth,
)


def to_analysis(image: np.ndarray) -> np.ndarray:
    small = cv2.resize(image, (320, 180), interpolation=cv2.INTER_AREA)
    if small.ndim == 3:
        small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return small.astype(np.float32)


class TestContract:
    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_rejects_colour_input(self, cls) -> None:
        metric = cls()
        with pytest.raises(MetricError, match="2-D"):
            metric.compute(np.zeros((64, 64, 3), dtype=np.float32))

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_rejects_wrong_dtype(self, cls) -> None:
        metric = cls()
        with pytest.raises(MetricError, match="float32"):
            metric.compute(np.zeros((64, 64), dtype=np.uint8))

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_rejects_tiny_image(self, cls) -> None:
        metric = cls()
        with pytest.raises(MetricError, match="smaller than"):
            metric.compute(np.zeros((4, 4), dtype=np.float32))

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_non_negative_and_finite(self, cls, gray_scene: np.ndarray) -> None:
        value = cls().compute(gray_scene)
        assert value >= 0.0
        assert np.isfinite(value)

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_flat_image_is_not_sharp(self, cls, flat_gray: np.ndarray) -> None:
        """A featureless frame must not be reported as sharp."""
        assert cls().compute(flat_gray) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_deterministic(self, cls, gray_scene: np.ndarray) -> None:
        metric = cls()
        assert metric.compute(gray_scene) == metric.compute(gray_scene)


class TestFocusResponse:
    """Every metric must fall as the defocus grows - the core requirement."""

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_monotone_in_defocus(self, cls, defocus_series) -> None:
        metric = cls()
        values = [metric.compute(to_analysis(frame)) for _, frame in defocus_series]
        for previous, current in zip(values, values[1:]):
            assert current <= previous + 1e-9, (
                f"{cls.__name__} rose with defocus: {values}"
            )

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_has_usable_dynamic_range(self, cls, defocus_series) -> None:
        """A metric that barely moves cannot drive an autofocus search."""
        metric = cls()
        values = [metric.compute(to_analysis(frame)) for _, frame in defocus_series]
        assert values[0] > values[-1] * 1.5, (
            f"{cls.__name__} has too little dynamic range: {values}"
        )


class TestExposureInvariance:
    """Contrast normalisation must absorb a pure illumination change."""

    @pytest.mark.parametrize("cls", ALL_METRIC_CLASSES)
    def test_gain_barely_moves_the_metric(self, cls, scene: np.ndarray) -> None:
        metric = cls(contrast_normalize=True)
        # 0.75 gain: darker, but far from clipping either end.
        base = metric.compute(to_analysis(scene))
        dimmed = metric.compute(to_analysis(apply_exposure(scene, 0.75)))
        assert base > 0
        assert dimmed == pytest.approx(base, rel=0.30), (
            f"{cls.__name__} moved {dimmed / base:.2f}x under a pure exposure change"
        )

    def test_without_normalisation_energy_metrics_track_exposure(
        self, scene: np.ndarray
    ) -> None:
        """Sanity check that the normalisation is what provides the invariance."""
        metric = LaplacianVariance(contrast_normalize=False)
        base = metric.compute(to_analysis(scene))
        dimmed = metric.compute(to_analysis(apply_exposure(scene, 0.5)))
        # Squared operator on a halved signal: roughly a quarter of the value.
        assert dimmed < base * 0.5


class TestNormalizationFactor:
    def test_disabled_returns_one(self, gray_scene: np.ndarray) -> None:
        assert normalization_factor(gray_scene, False) == 1.0

    def test_guards_against_black_frame(self) -> None:
        black = np.zeros((32, 32), dtype=np.float32)
        assert normalization_factor(black, True) >= 1.0


class TestHaar:
    def test_shapes_halve(self) -> None:
        image = np.random.default_rng(0).normal(128, 20, (64, 48)).astype(np.float32)
        ll, lh, hl, hh = haar_decompose(image)
        for band in (ll, lh, hl, hh):
            assert band.shape == (32, 24)

    def test_odd_dimensions_are_cropped(self) -> None:
        image = np.random.default_rng(0).normal(128, 20, (65, 49)).astype(np.float32)
        ll, _, _, _ = haar_decompose(image)
        assert ll.shape == (32, 24)

    def test_energy_is_preserved(self) -> None:
        """The transform is orthonormal, so total energy must be unchanged."""
        rng = np.random.default_rng(1)
        image = rng.normal(0, 30, (64, 64)).astype(np.float32)
        bands = haar_decompose(image)
        before = float(np.sum(image.astype(np.float64) ** 2))
        after = sum(float(np.sum(b.astype(np.float64) ** 2)) for b in bands)
        assert after == pytest.approx(before, rel=1e-5)

    def test_constant_image_has_no_detail(self) -> None:
        image = np.full((32, 32), 100.0, dtype=np.float32)
        ll, lh, hl, hh = haar_decompose(image)
        assert np.allclose(lh, 0.0, atol=1e-4)
        assert np.allclose(hl, 0.0, atol=1e-4)
        assert np.allclose(hh, 0.0, atol=1e-4)
        # The LL band has a DC gain of 2 per level.
        assert float(ll.mean()) == pytest.approx(200.0, rel=1e-5)


class TestEdgeWidth:
    def test_measures_a_plausible_width(self) -> None:
        """A Gaussian-blurred step of sigma s has width about s*sqrt(2*pi)."""
        image = np.full((200, 200), 40.0, dtype=np.float32)
        image[:, 100:] = 200.0
        sigma = 3.0
        blurred = cv2.GaussianBlur(image, (0, 0), sigma)
        value = EdgeWidth(range_window=21).compute(blurred)
        assert value > 0
        measured_width = 1.0 / value
        expected = sigma * np.sqrt(2.0 * np.pi)
        assert measured_width == pytest.approx(expected, rel=0.45), (
            f"measured {measured_width:.2f} px, expected about {expected:.2f} px"
        )

    def test_returns_zero_without_edges(self, flat_gray: np.ndarray) -> None:
        assert EdgeWidth().compute(flat_gray) == 0.0

    def test_rejects_even_window(self) -> None:
        with pytest.raises(ValueError, match="odd"):
            EdgeWidth(range_window=10)


class TestFourier:
    def test_is_a_bounded_ratio(self, gray_scene: np.ndarray) -> None:
        assert 0.0 <= FourierHighFrequency().compute(gray_scene) <= 1.0

    def test_rejects_bad_cutoff(self) -> None:
        with pytest.raises(ValueError, match="high_cutoff"):
            FourierHighFrequency(high_cutoff=1.5)

    def test_cache_survives_a_shape_change(self, gray_scene: np.ndarray) -> None:
        metric = FourierHighFrequency()
        first = metric.compute(gray_scene)
        other = cv2.resize(gray_scene, (160, 90)).astype(np.float32)
        metric.compute(other)
        assert metric.compute(gray_scene) == pytest.approx(first)


class TestBrenner:
    def test_rejects_zero_shift(self) -> None:
        with pytest.raises(ValueError, match="shift"):
            Brenner(shift=0)

    def test_larger_shift_still_monotone(self, defocus_series) -> None:
        metric = Brenner(shift=4)
        values = [metric.compute(to_analysis(f)) for _, f in defocus_series]
        assert all(b <= a + 1e-9 for a, b in zip(values, values[1:]))


class TestRegistry:
    def test_all_names_build(self) -> None:
        config = MetricsConfig()
        metrics = build_metrics(config)
        assert [m.name for m in metrics] == list(config.enabled)

    def test_every_enabled_metric_is_available(self) -> None:
        """The registry may hold more than the default set, but never less.

        `gradient_variance` ships built but switched off: it earned its place
        on measurement rather than on argument, so turning it on is a
        configuration change the user makes deliberately.
        """
        assert set(MetricsConfig().enabled) <= set(available_metrics())

    def test_every_available_metric_can_be_built(self) -> None:
        for name in available_metrics():
            assert build_metrics(MetricsConfig(enabled=(name,)))[0].name == name

    def test_every_available_metric_has_a_prior_and_a_sensitivity(self) -> None:
        """A metric that cannot be switched on from configuration is a trap."""
        from adaptive_sharpness import load_default_config

        config = load_default_config()
        for name in available_metrics():
            assert name in config.metrics.priors, f"{name} has no prior weight"
            assert name in config.ensemble.sensitivity, f"{name} has no sensitivity"

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown metric"):
            build_metrics(MetricsConfig(enabled=("laplacian", "does_not_exist")))

    def test_empty_selection_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            build_metrics(MetricsConfig(enabled=()))

    def test_config_reaches_the_metric(self) -> None:
        metrics = build_metrics(MetricsConfig(enabled=("brenner",), brenner_shift=5))
        assert metrics[0].shift == 5
