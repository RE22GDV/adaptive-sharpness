"""Tests for the documented input contract and the result's state fields.

Every case here used to pass silently and produce plausible but wrong numbers,
which is worse than an exception: the caller has no way to notice.
"""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_sharpness import SharpnessConfig, SharpnessEvaluator
from adaptive_sharpness.preprocess import Preprocessor
from adaptive_sharpness.synthetic import defocus, make_scene


@pytest.fixture
def pipeline() -> Preprocessor:
    return Preprocessor(SharpnessConfig().pipeline)


class TestAcceptedInput:
    @pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
    def test_supported_dtypes(self, pipeline: Preprocessor, dtype) -> None:
        frame = (np.random.default_rng(0).uniform(0, 200, (120, 160))).astype(dtype)
        assert pipeline.prepare(frame).gray.dtype == np.float32

    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_supported_channel_counts(self, pipeline: Preprocessor, channels: int) -> None:
        frame = np.full((120, 160, channels), 100, dtype=np.uint8)
        assert pipeline.prepare(frame).gray.ndim == 2

    def test_two_dimensional_greyscale(self, pipeline: Preprocessor) -> None:
        assert pipeline.prepare(np.full((120, 160), 100, np.uint8)).gray.ndim == 2

    def test_geometric_scale_is_reported(self, pipeline: Preprocessor) -> None:
        """Regression: an intensity-scaling variable once shadowed this one,
        which reported every tile and region coordinate in analysis pixels."""
        prepared = pipeline.prepare(make_scene(640, 360, seed=3))
        assert prepared.gray.shape == (180, 320)
        assert prepared.scale == pytest.approx(0.5)


class TestRejectedInput:
    def test_rejects_a_non_array(self, pipeline: Preprocessor) -> None:
        with pytest.raises(TypeError, match="ndarray"):
            pipeline.prepare([[1, 2], [3, 4]])  # type: ignore[arg-type]

    def test_rejects_four_dimensions(self, pipeline: Preprocessor) -> None:
        with pytest.raises(ValueError, match="2-D"):
            pipeline.prepare(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_rejects_two_channels(self, pipeline: Preprocessor) -> None:
        """A 2-channel array previously reached cvtColor and raised there."""
        with pytest.raises(ValueError, match="1, 3 or 4 channels"):
            pipeline.prepare(np.zeros((120, 160, 2), dtype=np.uint8))

    def test_rejects_unsupported_dtype(self, pipeline: Preprocessor) -> None:
        with pytest.raises(ValueError, match="unsupported dtype"):
            pipeline.prepare(np.zeros((120, 160), dtype=np.int32))

    def test_rejects_an_empty_frame(self, pipeline: Preprocessor) -> None:
        with pytest.raises(ValueError, match="empty"):
            pipeline.prepare(np.zeros((0, 0), dtype=np.uint8))

    def test_rejects_float_normalised_to_unit_range(self, pipeline: Preprocessor) -> None:
        """[0, 1] float input would silently wreck every intensity statistic."""
        frame = np.random.default_rng(0).uniform(0.0, 1.0, (120, 160)).astype(np.float32)
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            pipeline.prepare(frame)

    def test_unit_range_is_accepted_when_declared(self) -> None:
        config = SharpnessConfig().with_overrides(pipeline={"input_max": 1.0})
        frame = np.random.default_rng(0).uniform(0.0, 1.0, (120, 160)).astype(np.float32)
        gray = Preprocessor(config.pipeline).prepare(frame).gray
        assert gray.max() > 100.0, "declared unit-range input must be scaled to [0, 255]"

    def test_rejects_bad_colour_order(self) -> None:
        with pytest.raises(ValueError, match="color_order"):
            Preprocessor(
                SharpnessConfig().with_overrides(pipeline={"color_order": "YUV"}).pipeline
            )


class TestColourOrder:
    def test_rgb_and_bgr_differ_on_a_colour_frame(self) -> None:
        """Silent misinterpretation is the risk, so the two must not coincide."""
        frame = np.zeros((120, 160, 3), dtype=np.uint8)
        frame[:, :, 0] = 200   # first channel only
        bgr = Preprocessor(SharpnessConfig().pipeline).prepare(frame).gray
        rgb_config = SharpnessConfig().with_overrides(pipeline={"color_order": "RGB"})
        rgb = Preprocessor(rgb_config.pipeline).prepare(frame).gray
        assert abs(float(bgr.mean()) - float(rgb.mean())) > 5.0

    def test_grey_input_is_unaffected_by_colour_order(self) -> None:
        frame = np.full((120, 160), 128, dtype=np.uint8)
        rgb_config = SharpnessConfig().with_overrides(pipeline={"color_order": "RGB"})
        a = Preprocessor(SharpnessConfig().pipeline).prepare(frame).gray
        b = Preprocessor(rgb_config.pipeline).prepare(frame).gray
        assert np.allclose(a, b)


class TestInputMax:
    def test_twelve_bit_data_in_uint16(self) -> None:
        """12-bit samples in a uint16 container must not be scaled as 16-bit."""
        frame = np.full((120, 160), 4095, dtype=np.uint16)
        default = Preprocessor(SharpnessConfig().pipeline).prepare(frame).gray
        declared = Preprocessor(
            SharpnessConfig().with_overrides(pipeline={"input_max": 4095}).pipeline
        ).prepare(frame).gray
        # Treated as full-range 16-bit, white reads as almost black.
        assert float(default.mean()) < 20.0
        assert float(declared.mean()) == pytest.approx(255.0, abs=1.0)

    def test_rejects_non_positive_input_max(self) -> None:
        config = SharpnessConfig().with_overrides(pipeline={"input_max": 0.0})
        with pytest.raises(ValueError, match="input_max"):
            Preprocessor(config.pipeline).prepare(np.zeros((32, 32), np.uint8))


class TestResultState:
    def test_reports_not_ready_before_warm_up(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene)
        assert result.ready is False
        assert result.warmup_samples == 1

    def test_reports_ready_once_the_scale_is_real(self, scene: np.ndarray) -> None:
        """Readiness needs variation, not merely a number of frames: a frozen
        scene leaves every normaliser degenerate however long it runs."""
        evaluator = SharpnessEvaluator()
        for radius in (0.0, 1.5, 3.0, 4.5, 6.0, 7.5, 9.0):
            result = evaluator.evaluate(defocus(scene, radius))
        assert result.ready is True
        assert result.informative_fraction == pytest.approx(1.0)
        assert result.warmup_samples >= 5

    def test_static_stream_never_becomes_ready(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        for _ in range(30):
            result = evaluator.evaluate(scene)
        assert result.informative_fraction == 0.0
        assert result.ready is False
        assert result.confidence < 0.05

    def test_reset_clears_readiness(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        for radius in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
            evaluator.evaluate(defocus(scene, radius))
        evaluator.reset()
        assert evaluator.evaluate(scene).ready is False


class TestScoreSemantics:
    def test_score_is_an_alias_of_filtered_score(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene)
        assert result.score == result.filtered_score

    def test_focus_change_alias(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene)
        assert result.focus_change_detected == result.score_change_detected

    def test_confidence_does_not_touch_the_instantaneous_score(
        self, scene: np.ndarray
    ) -> None:
        """Documented guarantee: only the filtered score is confidence-coupled."""
        coupled = SharpnessConfig().with_overrides(
            temporal={"confidence_coupling": True}
        )
        uncoupled = SharpnessConfig().with_overrides(
            temporal={"confidence_coupling": False}
        )
        frames = [defocus(scene, r) for r in (0.0, 2.0, 4.0, 6.0, 8.0, 1.0, 3.0)]
        a = [SharpnessEvaluator(coupled).evaluate(f).instantaneous_score for f in frames]
        b = [SharpnessEvaluator(uncoupled).evaluate(f).instantaneous_score for f in frames]
        assert a == pytest.approx(b)

    def test_confidence_does_change_the_filtered_score(self, scene: np.ndarray) -> None:
        """The opposite guarantee, which the documentation used to deny."""
        frames = [defocus(scene, r) for r in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)] * 3

        def run(coupling: bool) -> list[float]:
            config = SharpnessConfig().with_overrides(
                temporal={"confidence_coupling": coupling, "alpha_base": 0.3}
            )
            evaluator = SharpnessEvaluator(config)
            return [evaluator.evaluate(f).filtered_score for f in frames]

        coupled, uncoupled = run(True), run(False)
        assert any(
            abs(x - y) > 1e-6 for x, y in zip(coupled, uncoupled)
        ), "confidence coupling had no effect on the filtered score"
