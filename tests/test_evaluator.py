"""Integration tests for the full evaluation pipeline."""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_sharpness import (
    ROI,
    Frame,
    RegionEvaluator,
    SharpnessConfig,
    SharpnessEvaluator,
)
from adaptive_sharpness.synthetic import add_noise, apply_exposure, defocus, make_scene


def warm(evaluator: SharpnessEvaluator, frames, passes: int = 2) -> None:
    """Prime the running normalisers over the range of frames to be scored."""
    for _ in range(passes):
        for frame in frames:
            evaluator.evaluate(frame)


class TestBasicEvaluation:
    def test_returns_a_complete_result(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene)
        assert 0.0 <= result.score <= 1.0
        assert 0.0 <= result.confidence <= 1.0
        assert len(result.metrics) == 6
        assert result.processing_time_s > 0.0
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_accepts_a_frame_object(self, scene: np.ndarray) -> None:
        frame = Frame(data=scene, index=42, capture_latency_s=0.01)
        result = SharpnessEvaluator().evaluate(frame)
        assert result.frame_index == 42
        assert result.capture_latency_s == pytest.approx(0.01)

    def test_accepts_a_greyscale_frame(self, gray_scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(gray_scene.astype(np.uint8))
        assert 0.0 <= result.score <= 1.0

    def test_does_not_mutate_the_input(self, scene: np.ndarray) -> None:
        original = scene.copy()
        SharpnessEvaluator().evaluate(scene)
        assert np.array_equal(scene, original)

    def test_rejects_a_bad_shape(self) -> None:
        with pytest.raises(ValueError, match="2-D or 3-D"):
            SharpnessEvaluator().evaluate(np.zeros((4, 4, 4, 4), dtype=np.uint8))

    def test_metric_names_are_stable(self) -> None:
        evaluator = SharpnessEvaluator()
        assert evaluator.metric_names == tuple(SharpnessConfig().metrics.enabled)


class TestFocusResponse:
    def test_score_falls_with_defocus(self, scene: np.ndarray) -> None:
        frames = [defocus(scene, r) for r in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)]
        evaluator = SharpnessEvaluator()
        warm(evaluator, frames)
        scores = [evaluator.evaluate(f).instantaneous_score for f in frames]
        for previous, current in zip(scores, scores[1:]):
            assert current <= previous + 1e-9, scores

    def test_finds_the_true_best_focus(self, scene: np.ndarray) -> None:
        """A sweep through focus must peak at the in-focus frame."""
        radii = [6.0, 4.5, 3.0, 1.5, 0.0, 1.5, 3.0, 4.5, 6.0]
        frames = [defocus(scene, r) for r in radii]
        evaluator = SharpnessEvaluator()
        warm(evaluator, frames)
        scores = [evaluator.evaluate(f).instantaneous_score for f in frames]
        assert int(np.argmax(scores)) == 4

    def test_confidence_is_high_on_a_good_frame(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        # Warm on *varied* frames: feeding one frame repeatedly leaves the
        # normalisers with no observed range, which is correctly reported as
        # zero confidence (see test_static_scene_has_no_confidence).
        warm(evaluator, [defocus(scene, r) for r in (0.0, 1.5, 3.0, 5.0)], passes=3)
        assert evaluator.warmed_up
        assert evaluator.evaluate(scene).confidence > 0.6

    def test_confidence_is_penalised_before_warm_up(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        cold = evaluator.evaluate(scene).confidence
        warm(evaluator, [defocus(scene, r) for r in (0.0, 1.5, 3.0, 5.0)], passes=3)
        assert evaluator.evaluate(scene).confidence > cold

    def test_static_scene_has_no_confidence(self, scene: np.ndarray) -> None:
        """A frozen scene drives every metric to its neutral value, so the score
        carries no information - and must not be reported as trustworthy.

        Without this the metrics all read exactly 0.5, which the concordance
        factor sees as perfect agreement and rewards with 0.99 confidence.
        """
        evaluator = SharpnessEvaluator()
        for _ in range(20):
            result = evaluator.evaluate(scene)
        assert result.confidence < 0.05

    def test_confidence_collapses_on_a_blank_frame(self) -> None:
        blank = np.full((360, 640, 3), 128, dtype=np.uint8)
        result = SharpnessEvaluator().evaluate(blank)
        assert result.confidence < 0.1


class TestRobustness:
    def test_noise_shifts_weight_away_from_the_laplacian(self, scene: np.ndarray) -> None:
        clean = SharpnessEvaluator().evaluate(scene)
        noisy = SharpnessEvaluator().evaluate(add_noise(scene, 12.0, seed=1))
        assert noisy.weights["laplacian"] < clean.weights["laplacian"]

    def test_exposure_change_barely_moves_the_score(self, scene: np.ndarray) -> None:
        """A non-clipping exposure change must not look like a focus change.

        The normaliser is calibrated on a focus sweep first, because that is
        what sets its scale in real use.  Warming it on the exposure variants
        themselves would make the running percentiles stretch those few frames
        across the whole [0, 1] range and guarantee a large spread whatever the
        metrics did.
        """
        sweep = [defocus(scene, r) for r in (0.0, 1.0, 2.0, 3.0, 4.0, 6.0)]
        evaluator = SharpnessEvaluator()
        warm(evaluator, sweep, passes=3)
        base = evaluator.evaluate(scene).instantaneous_score
        for gain in (0.7, 0.85):
            score = evaluator.evaluate(apply_exposure(scene, gain)).instantaneous_score
            assert abs(score - base) < 0.2, f"gain {gain}: {score} vs {base}"

    def test_clipping_exposure_lowers_confidence(self, scene: np.ndarray) -> None:
        """Blowing the highlights destroys real detail, so the score genuinely
        drops - what the system must do is report that it is less trustworthy."""
        sweep = [defocus(scene, r) for r in (0.0, 1.5, 3.0, 5.0)]

        clean = SharpnessEvaluator()
        warm(clean, sweep, passes=3)
        normal = clean.evaluate(scene)

        overexposed = SharpnessEvaluator()
        warm(overexposed, [apply_exposure(f, 3.0) for f in sweep], passes=3)
        blown = overexposed.evaluate(apply_exposure(scene, 3.0))

        assert blown.stats.clipped_high > 0.1
        assert blown.confidence < normal.confidence

    def test_handles_a_fully_black_frame(self) -> None:
        black = np.zeros((360, 640, 3), dtype=np.uint8)
        result = SharpnessEvaluator().evaluate(black)
        assert np.isfinite(result.score)
        assert result.confidence < 0.1

    def test_handles_a_fully_white_frame(self) -> None:
        white = np.full((360, 640, 3), 255, dtype=np.uint8)
        result = SharpnessEvaluator().evaluate(white)
        assert np.isfinite(result.score)
        assert result.confidence < 0.1

    def test_handles_a_tiny_frame(self) -> None:
        tiny = make_scene(64, 48, seed=2)
        result = SharpnessEvaluator().evaluate(tiny)
        assert np.isfinite(result.score)


class TestROI:
    def test_roi_is_reported_back(self, scene: np.ndarray) -> None:
        roi = ROI(100, 50, 320, 180)
        result = SharpnessEvaluator().evaluate(scene, roi=roi)
        assert result.roi == roi

    def test_roi_is_clipped_to_the_frame(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene, roi=ROI(600, 300, 400, 400))
        assert result.roi is not None
        assert result.roi.x2 <= scene.shape[1]
        assert result.roi.y2 <= scene.shape[0]

    def test_roi_sees_only_its_own_region(self, scene: np.ndarray) -> None:
        """Blurring outside the ROI must not change the ROI score."""
        roi = ROI(0, 0, 200, 200)
        blurred_outside = scene.copy()
        blurred_outside[:, 260:] = defocus(scene, 8.0)[:, 260:]
        evaluator = SharpnessEvaluator()
        a = evaluator.evaluate(scene, roi=roi).raw_metrics
        b = evaluator.evaluate(blurred_outside, roi=roi).raw_metrics
        for name in a:
            assert b[name] == pytest.approx(a[name], rel=1e-6)

    def test_roi_detects_local_defocus(self, scene: np.ndarray) -> None:
        roi = ROI(160, 90, 320, 180)
        sharp = SharpnessEvaluator()
        blurry = SharpnessEvaluator()
        blurred = defocus(scene, 6.0)
        for _ in range(3):
            sharp.evaluate(scene, roi=roi)
            sharp.evaluate(blurred, roi=roi)
            blurry.evaluate(scene, roi=roi)
            blurry.evaluate(blurred, roi=roi)
        assert (
            sharp.evaluate(scene, roi=roi).instantaneous_score
            > blurry.evaluate(blurred, roi=roi).instantaneous_score
        )


class TestState:
    def test_reset_clears_history(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        for _ in range(10):
            evaluator.evaluate(scene)
        assert evaluator.warmed_up
        evaluator.reset()
        assert not evaluator.warmed_up

    def test_warms_up_after_enough_frames(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        assert not evaluator.warmed_up
        for _ in range(SharpnessConfig().normalization.warmup):
            evaluator.evaluate(scene)
        assert evaluator.warmed_up

    def test_frame_index_advances_for_raw_arrays(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        indices = [evaluator.evaluate(scene).frame_index for _ in range(3)]
        assert indices == [0, 1, 2]

    def test_evaluate_raw_only_does_not_disturb_state(self, scene: np.ndarray) -> None:
        evaluator = SharpnessEvaluator()
        for _ in range(10):
            evaluator.evaluate(scene)
        before = evaluator.evaluate(scene).score
        for _ in range(5):
            evaluator.evaluate_raw_only(defocus(scene, 8.0))
        after = evaluator.evaluate(scene).score
        assert after == pytest.approx(before, abs=0.02)

    def test_motor_position_is_carried_through(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene, motor_position=0.42)
        assert result.motor_position == pytest.approx(0.42)
        assert result.as_row()["motor_position"] == pytest.approx(0.42)


class TestSerialisation:
    def test_row_has_every_metric(self, scene: np.ndarray) -> None:
        result = SharpnessEvaluator().evaluate(scene)
        row = result.as_row()
        for name in result.weights:
            assert f"raw_{name}" in row
            assert f"norm_{name}" in row
            assert f"w_{name}" in row
            assert f"rel_{name}" in row
        assert "score" in row and "confidence" in row

    def test_row_values_are_serialisable(self, scene: np.ndarray) -> None:
        import json

        row = SharpnessEvaluator().evaluate(scene).as_row()
        json.dumps(row)


class TestRegionEvaluator:
    def test_returns_both_results(self, scene: np.ndarray) -> None:
        full, region = RegionEvaluator().evaluate(scene, roi=ROI(100, 50, 300, 200))
        assert full.roi is None
        assert region is not None
        assert region.roi is not None

    def test_roi_result_is_none_without_a_roi(self, scene: np.ndarray) -> None:
        full, region = RegionEvaluator().evaluate(scene)
        assert region is None
        assert full.score >= 0.0

    def test_reset_clears_both(self, scene: np.ndarray) -> None:
        evaluator = RegionEvaluator()
        for _ in range(10):
            evaluator.evaluate(scene, roi=ROI(0, 0, 200, 200))
        evaluator.reset()
        assert not evaluator.full.warmed_up
        assert not evaluator.region.warmed_up


class TestConfigurationEffects:
    def test_analysis_width_changes_the_work(self, scene: np.ndarray) -> None:
        narrow = SharpnessConfig().with_overrides(pipeline={"analysis_width": 160})
        wide = SharpnessConfig().with_overrides(pipeline={"analysis_width": 640})
        assert (
            SharpnessEvaluator(wide).evaluate(scene).processing_time_s
            > SharpnessEvaluator(narrow).evaluate(scene).processing_time_s
        )

    def test_metric_subset_is_honoured(self, scene: np.ndarray) -> None:
        config = SharpnessConfig().with_overrides(
            metrics={"enabled": ("laplacian", "tenengrad")}
        )
        result = SharpnessEvaluator(config).evaluate(scene)
        assert len(result.metrics) == 2
        assert sum(result.weights.values()) == pytest.approx(1.0)

    def test_temporal_filter_can_be_disabled(self, scene: np.ndarray) -> None:
        config = SharpnessConfig().with_overrides(temporal={"enabled": False})
        evaluator = SharpnessEvaluator(config)
        for _ in range(5):
            evaluator.evaluate(scene)
        result = evaluator.evaluate(defocus(scene, 5.0))
        assert result.score == pytest.approx(result.instantaneous_score)
