"""Tests for the image-characteristic estimators that drive the weighting."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from sharpness.analysis import DEGRADATION_KEYS, ImageAnalyzer
from sharpness.config import AnalysisConfig
from sharpness.types import ImageStats
from tools.synthetic import translate


@pytest.fixture
def analyzer() -> ImageAnalyzer:
    return ImageAnalyzer(AnalysisConfig())


class TestNoiseEstimator:
    @pytest.mark.parametrize("sigma", [1.0, 3.0, 6.0, 12.0])
    def test_recovers_known_sigma_on_a_flat_field(
        self, analyzer: ImageAnalyzer, sigma: float
    ) -> None:
        rng = np.random.default_rng(4)
        image = np.full((256, 256), 128.0, dtype=np.float32)
        image += rng.normal(0.0, sigma, image.shape)
        estimate = analyzer.estimate_noise_sigma(image.astype(np.float32))
        assert estimate == pytest.approx(sigma, rel=0.12)

    def test_structure_does_not_masquerade_as_noise(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        """A clean, textured but noiseless frame must read as low noise."""
        smooth = cv2.GaussianBlur(gray_scene, (0, 0), 2.0)
        assert analyzer.estimate_noise_sigma(smooth) < 1.0

    def test_noise_adds_on_top_of_structure(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        rng = np.random.default_rng(5)
        noisy = gray_scene + rng.normal(0.0, 8.0, gray_scene.shape).astype(np.float32)
        clean_estimate = analyzer.estimate_noise_sigma(gray_scene)
        noisy_estimate = analyzer.estimate_noise_sigma(noisy)
        assert noisy_estimate > clean_estimate + 4.0


class TestMotionEstimator:
    def test_first_frame_reports_no_motion(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        assert analyzer.estimate_motion_px(gray_scene) == 0.0

    def test_static_scene_reports_near_zero(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        """Regression: phaseCorrelate has a size-dependent sub-pixel offset that
        previously showed up as ~2 px of motion on a completely static scene."""
        analyzer.estimate_motion_px(gray_scene)
        for _ in range(4):
            assert analyzer.estimate_motion_px(gray_scene) < 0.35

    @pytest.mark.parametrize("shift", [4.0, 8.0, 16.0])
    def test_tracks_a_known_translation(
        self, analyzer: ImageAnalyzer, scene: np.ndarray, shift: float
    ) -> None:
        small = cv2.resize(scene, (320, 180), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        moved = cv2.cvtColor(
            translate(small, shift, 0.0), cv2.COLOR_BGR2GRAY
        ).astype(np.float32)
        analyzer.estimate_motion_px(gray)
        measured = analyzer.estimate_motion_px(moved)
        assert measured == pytest.approx(shift, abs=1.5)

    def test_defocus_alone_is_not_read_as_motion(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        """A focus change moves every pixel value but nothing translates."""
        analyzer.estimate_motion_px(gray_scene)
        blurred = cv2.GaussianBlur(gray_scene, (0, 0), 3.0)
        assert analyzer.estimate_motion_px(blurred) < 1.0

    def test_reset_clears_the_reference(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        analyzer.estimate_motion_px(gray_scene)
        analyzer.reset()
        assert analyzer.estimate_motion_px(gray_scene) == 0.0


class TestStats:
    def test_flat_frame_has_no_structure(
        self, analyzer: ImageAnalyzer, flat_gray: np.ndarray
    ) -> None:
        stats = analyzer.analyze(flat_gray)
        assert stats.edge_density == pytest.approx(0.0)
        assert stats.edge_sufficiency == pytest.approx(0.0)
        assert stats.local_contrast == pytest.approx(0.0)
        assert stats.noise_sigma == pytest.approx(0.0)

    def test_textured_frame_has_structure(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        stats = analyzer.analyze(gray_scene)
        assert stats.edge_density > 0.01
        assert stats.local_contrast > 0.05
        assert 0.0 < stats.brightness < 1.0

    def test_detects_blown_highlights(self, analyzer: ImageAnalyzer) -> None:
        image = np.full((180, 320), 128.0, dtype=np.float32)
        image[:90, :] = 255.0
        stats = analyzer.analyze(image)
        assert stats.clipped_high == pytest.approx(0.5, abs=0.01)
        assert stats.clipped_low == pytest.approx(0.0)

    def test_detects_crushed_shadows(self, analyzer: ImageAnalyzer) -> None:
        image = np.full((180, 320), 128.0, dtype=np.float32)
        image[:45, :] = 0.0
        stats = analyzer.analyze(image)
        assert stats.clipped_low == pytest.approx(0.25, abs=0.01)

    def test_anisotropy_is_high_for_parallel_edges(self, analyzer: ImageAnalyzer) -> None:
        image = np.zeros((180, 320), dtype=np.float32)
        image[:, ::8] = 255.0  # vertical stripes only
        assert analyzer.estimate_anisotropy(image) > 0.7

    def test_anisotropy_is_low_for_isotropic_texture(self, analyzer: ImageAnalyzer) -> None:
        rng = np.random.default_rng(6)
        image = rng.normal(128, 30, (180, 320)).astype(np.float32)
        assert analyzer.estimate_anisotropy(image) < 0.2

    def test_snr_stays_finite_without_noise(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        stats = analyzer.analyze(gray_scene)
        assert np.isfinite(stats.snr)

    def test_as_dict_is_complete(self, analyzer: ImageAnalyzer, gray_scene: np.ndarray) -> None:
        data = analyzer.analyze(gray_scene).as_dict()
        assert set(data) == set(ImageStats().as_dict())
        assert all(isinstance(v, float) for v in data.values())


class TestDegradations:
    def test_keys_match_the_contract(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        degradations = analyzer.degradations(analyzer.analyze(gray_scene))
        assert set(degradations) == set(DEGRADATION_KEYS)

    def test_all_values_are_bounded(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        degradations = analyzer.degradations(analyzer.analyze(gray_scene))
        assert all(0.0 <= v <= 1.0 for v in degradations.values())

    def test_flat_frame_is_maximally_degraded(
        self, analyzer: ImageAnalyzer, flat_gray: np.ndarray
    ) -> None:
        degradations = analyzer.degradations(analyzer.analyze(flat_gray))
        assert degradations["edge"] == pytest.approx(1.0)
        assert degradations["contrast"] == pytest.approx(1.0)

    def test_dark_frame_penalises_exposure(self, analyzer: ImageAnalyzer) -> None:
        rng = np.random.default_rng(7)
        dark = np.clip(rng.normal(12, 3, (180, 320)), 0, 255).astype(np.float32)
        degradations = analyzer.degradations(analyzer.analyze(dark))
        assert degradations["clip"] > 0.3

    def test_noise_raises_the_noise_degradation(
        self, analyzer: ImageAnalyzer, gray_scene: np.ndarray
    ) -> None:
        rng = np.random.default_rng(8)
        noisy = gray_scene + rng.normal(0, 10, gray_scene.shape).astype(np.float32)
        clean = analyzer.degradations(analyzer.analyze(gray_scene))["noise"]
        analyzer.reset()
        dirty = analyzer.degradations(analyzer.analyze(noisy))["noise"]
        assert dirty > clean
        assert dirty == pytest.approx(1.0, abs=0.2)


class TestConfigValidation:
    def test_rejects_bad_downscale(self) -> None:
        with pytest.raises(ValueError, match="motion_downscale"):
            ImageAnalyzer(AnalysisConfig(motion_downscale=0))
