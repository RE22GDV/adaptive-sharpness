"""Tests for the focus map, region proposals and scene evaluation."""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from sharpness import SceneEvaluator, SharpnessConfig
from sharpness.config import FocusMapConfig, PipelineConfig, RegionsConfig
from sharpness.focusmap import TILE_MEASURES, FocusMapper
from sharpness.preprocess import Preprocessor
from sharpness.regions import Region, RegionProposer
from tools.synthetic import add_noise, defocus, make_scene

WIDTH, HEIGHT = 640, 360


def split_frame(
    left_blur: float,
    right_blur: float,
    left_detail: float = 1.0,
    right_detail: float = 1.0,
    left_flat: bool = False,
    noise: float = 0.0,
) -> np.ndarray:
    """A frame whose two halves are independently textured and defocused."""
    left = (
        np.full((HEIGHT, WIDTH, 3), 110, dtype=np.uint8)
        if left_flat
        else make_scene(WIDTH, HEIGHT, seed=3, detail=left_detail)
    )
    right = make_scene(WIDTH, HEIGHT, seed=9, detail=right_detail)
    combined = defocus(left, left_blur)
    combined[:, WIDTH // 2 :] = defocus(right, right_blur)[:, WIDTH // 2 :]
    if noise > 0:
        combined = add_noise(combined, noise, seed=5)
    return combined


@pytest.fixture
def prepared():
    pre = Preprocessor(PipelineConfig())

    def _prepare(image: np.ndarray):
        return pre.prepare(image)

    return _prepare


def half_means(fmap, mid_guard: int = 1) -> tuple[float, float]:
    cols = fmap.shape[1]
    mid = cols // 2
    left = fmap.score[:, : mid - mid_guard][fmap.valid[:, : mid - mid_guard]]
    right = fmap.score[:, mid + mid_guard :][fmap.valid[:, mid + mid_guard :]]
    return (
        float(np.mean(left)) if left.size else float("nan"),
        float(np.mean(right)) if right.size else float("nan"),
    )


class TestFocusMapContract:
    def test_rejects_colour_input(self) -> None:
        mapper = FocusMapper(FocusMapConfig())
        with pytest.raises(ValueError, match="2-D"):
            mapper.compute(np.zeros((64, 64, 3), dtype=np.float32))

    def test_rejects_wrong_dtype(self) -> None:
        mapper = FocusMapper(FocusMapConfig())
        with pytest.raises(ValueError, match="float32"):
            mapper.compute(np.zeros((64, 64), dtype=np.uint8))

    def test_rejects_unknown_measure(self) -> None:
        with pytest.raises(ValueError, match="unknown tile measure"):
            FocusMapper(FocusMapConfig(measure="vibes"))

    def test_rejects_tiny_tiles(self) -> None:
        with pytest.raises(ValueError, match="tile_size"):
            FocusMapper(FocusMapConfig(tile_size=4))

    def test_scores_are_bounded(self, prepared, gray_scene) -> None:
        fmap = FocusMapper(FocusMapConfig()).compute(gray_scene)
        assert np.all(fmap.score >= 0.0)
        assert np.all(fmap.score <= 1.0)

    def test_grid_shape_follows_tile_size(self, gray_scene) -> None:
        fmap = FocusMapper(FocusMapConfig(tile_size=32)).compute(gray_scene)
        height, width = gray_scene.shape
        assert fmap.shape == (height // 32, width // 32)


class TestFocusMapCorrectness:
    """The map must locate sharpness, and must not confuse it with texture."""

    @pytest.mark.parametrize("measure", TILE_MEASURES)
    def test_finds_the_sharp_half(self, prepared, measure) -> None:
        image = split_frame(left_blur=0.0, right_blur=5.0)
        fmap = FocusMapper(FocusMapConfig(measure=measure)).compute(prepared(image).gray)
        left, right = half_means(fmap)
        assert left > right + 0.05, f"{measure}: left {left:.3f} right {right:.3f}"

    @pytest.mark.parametrize("measure", TILE_MEASURES)
    def test_sparse_but_sharp_beats_dense_but_blurred(self, prepared, measure) -> None:
        """The adversarial case: more texture must not be mistaken for more focus."""
        image = split_frame(
            left_blur=0.0, right_blur=5.0, left_detail=0.18, right_detail=1.0
        )
        fmap = FocusMapper(FocusMapConfig(measure=measure)).compute(prepared(image).gray)
        left, right = half_means(fmap)
        assert left > right + 0.05, f"{measure}: left {left:.3f} right {right:.3f}"

    def test_default_measure_survives_heavy_noise(self, prepared) -> None:
        """Regression: lap_over_grad collapses here, which is why it is not the
        default. Noise puts high-frequency energy back into the blurred half."""
        image = split_frame(left_blur=0.0, right_blur=5.0, noise=20.0)
        fmap = FocusMapper(FocusMapConfig()).compute(prepared(image).gray)
        left, right = half_means(fmap)
        assert left > right + 0.15, f"left {left:.3f} right {right:.3f}"

    def test_featureless_area_is_invalid_not_blurry(self, prepared) -> None:
        """A blank wall has no focus information; it must not be ranked."""
        image = split_frame(left_blur=0.0, right_blur=0.0, left_flat=True)
        fmap = FocusMapper(FocusMapConfig()).compute(prepared(image).gray)
        cols = fmap.shape[1]
        left_valid = fmap.valid[:, : cols // 2 - 1]
        right_valid = fmap.valid[:, cols // 2 + 1 :]
        assert np.count_nonzero(left_valid) / left_valid.size < 0.15
        assert np.count_nonzero(right_valid) / right_valid.size > 0.7

    def test_uniformly_sharp_frame_has_no_strong_winner(self, prepared, scene) -> None:
        fmap = FocusMapper(FocusMapConfig()).compute(prepared(scene).gray)
        values = fmap.score[fmap.valid]
        assert values.size > 0
        assert float(np.std(values)) < 0.45

    def test_best_cell_lands_in_the_sharp_half(self, prepared) -> None:
        image = split_frame(left_blur=0.0, right_blur=6.0)
        fmap = FocusMapper(FocusMapConfig()).compute(prepared(image).gray)
        cell = fmap.best_cell()
        assert cell is not None
        assert cell[1] < fmap.shape[1] // 2

    def test_best_cell_is_none_without_texture(self) -> None:
        flat = np.full((180, 320), 128.0, dtype=np.float32)
        fmap = FocusMapper(FocusMapConfig()).compute(flat)
        assert fmap.best_cell() is None
        assert fmap.valid_fraction == 0.0

    def test_cell_bounds_map_back_to_the_source(self, prepared) -> None:
        image = split_frame(left_blur=0.0, right_blur=5.0)
        result = prepared(image)
        fmap = FocusMapper(FocusMapConfig()).compute(result.gray, result.scale)
        x, y, w, h = fmap.cell_bounds_source(1, 2)
        assert 0 <= x < WIDTH and 0 <= y < HEIGHT
        assert x + w <= WIDTH + fmap.tile_size


class TestRegions:
    def test_iou_is_symmetric_and_bounded(self) -> None:
        a = Region(0, 0, 100, 100)
        b = Region(50, 50, 100, 100)
        assert a.iou(b) == pytest.approx(b.iou(a))
        assert 0.0 <= a.iou(b) <= 1.0
        assert a.iou(a) == pytest.approx(1.0)
        assert a.iou(Region(500, 500, 10, 10)) == 0.0

    def test_proposes_a_region_on_the_sharp_half(self, prepared) -> None:
        image = split_frame(left_blur=0.0, right_blur=6.0)
        result = prepared(image)
        fmap = FocusMapper(FocusMapConfig()).compute(result.gray, result.scale)
        proposer = RegionProposer(RegionsConfig(detectors=("sharp_blobs",)))
        regions = proposer.propose(fmap, result.gray, image.shape)
        assert regions
        best = regions[0]
        assert best.center[0] < WIDTH // 2
        assert best.map_score > 0.0

    def test_unknown_detector_raises(self, prepared, gray_scene) -> None:
        fmap = FocusMapper(FocusMapConfig()).compute(gray_scene)
        proposer = RegionProposer(RegionsConfig(detectors=("telepathy",)))
        with pytest.raises(KeyError, match="unknown region detector"):
            proposer.propose(fmap, gray_scene, (360, 640, 3))

    def test_no_regions_without_texture(self) -> None:
        flat = np.full((180, 320), 128.0, dtype=np.float32)
        fmap = FocusMapper(FocusMapConfig()).compute(flat)
        proposer = RegionProposer(RegionsConfig(detectors=("sharp_blobs",)))
        assert proposer.propose(fmap, flat, (360, 640, 3)) == []

    def test_respects_the_region_cap(self, prepared, scene) -> None:
        result = prepared(scene)
        fmap = FocusMapper(FocusMapConfig(tile_size=16)).compute(result.gray, result.scale)
        proposer = RegionProposer(
            RegionsConfig(detectors=("sharp_blobs",), max_regions=2,
                          blob_min_tiles=1, min_area_fraction=0.0)
        )
        assert len(proposer.propose(fmap, result.gray, scene.shape)) <= 2

    def test_overlapping_proposals_are_merged(self) -> None:
        proposer = RegionProposer(RegionsConfig(merge_iou=0.4))
        regions = [
            Region(0, 0, 100, 100, label="face", source="faces", map_score=0.5,
                   map_coverage=1.0),
            Region(10, 10, 100, 100, label="sharp area", source="sharp_blobs",
                   map_score=0.9, map_coverage=1.0),
        ]
        merged = proposer._merge(regions)
        assert len(merged) == 1
        # The named detection survives, since it is more useful to a viewer.
        assert merged[0].label == "face"


class TestSceneEvaluator:
    def test_identifies_the_in_focus_side(self) -> None:
        image = split_frame(left_blur=0.0, right_blur=6.0)
        result = SceneEvaluator().evaluate(image)
        assert result.has_subject
        assert result.subject is not None
        assert result.subject.center[0] < WIDTH // 2, "picked the blurred half"

    def test_follows_the_focus_when_it_moves(self) -> None:
        evaluator = SceneEvaluator()
        left_sharp = split_frame(left_blur=0.0, right_blur=6.0)
        right_sharp = split_frame(left_blur=6.0, right_blur=0.0)
        first = evaluator.evaluate(left_sharp)
        second = evaluator.evaluate(right_sharp)
        assert first.subject is not None and second.subject is not None
        assert first.subject.center[0] < WIDTH // 2
        assert second.subject.center[0] > WIDTH // 2

    def test_reports_no_subject_on_a_blank_frame(self) -> None:
        blank = np.full((HEIGHT, WIDTH, 3), 128, dtype=np.uint8)
        result = SceneEvaluator().evaluate(blank)
        assert not result.has_subject
        assert result.subject_label == "nothing in focus"
        assert result.subject_result is None

    def test_frame_result_is_always_produced(self, scene) -> None:
        result = SceneEvaluator().evaluate(scene)
        assert result.frame_result is not None
        assert 0.0 <= result.frame_result.score <= 1.0

    def test_subject_result_scores_the_subject_region(self) -> None:
        image = split_frame(left_blur=0.0, right_blur=6.0)
        result = SceneEvaluator().evaluate(image)
        assert result.subject_result is not None
        assert result.subject_result.roi is not None
        assert sum(result.subject_result.weights.values()) == pytest.approx(1.0)

    def test_subject_change_is_flagged_and_resets_history(self) -> None:
        evaluator = SceneEvaluator()
        left_sharp = split_frame(left_blur=0.0, right_blur=6.0)
        right_sharp = split_frame(left_blur=6.0, right_blur=0.0)
        first = evaluator.evaluate(left_sharp)
        assert first.subject_changed  # nothing was tracked before
        stable = evaluator.evaluate(left_sharp)
        assert not stable.subject_changed
        moved = evaluator.evaluate(right_sharp)
        assert moved.subject_changed

    def test_separation_reports_a_weak_decision(self, scene) -> None:
        """A uniformly sharp frame gives no strong reason to prefer one region."""
        result = SceneEvaluator().evaluate(scene)
        assert 0.0 <= result.separation() <= 1.0

    def test_reset_clears_tracking(self) -> None:
        evaluator = SceneEvaluator()
        image = split_frame(left_blur=0.0, right_blur=6.0)
        evaluator.evaluate(image)
        evaluator.reset()
        assert evaluator.evaluate(image).subject_changed

    def test_timing_fields_are_populated(self, scene) -> None:
        result = SceneEvaluator().evaluate(scene)
        assert result.scene_time_s > 0.0
        assert result.total_time_s >= result.scene_time_s

    def test_motor_position_propagates(self, scene) -> None:
        result = SceneEvaluator().evaluate(scene, motor_position=0.31)
        assert result.frame_result is not None
        assert result.frame_result.motor_position == pytest.approx(0.31)

    def test_does_not_mutate_the_input(self) -> None:
        image = split_frame(left_blur=0.0, right_blur=6.0)
        original = image.copy()
        SceneEvaluator().evaluate(image)
        assert np.array_equal(image, original)

    def test_config_disables_faces(self, scene) -> None:
        config = SharpnessConfig().with_overrides(
            regions={"detectors": ("sharp_blobs",)}
        )
        result = SceneEvaluator(config).evaluate(scene)
        assert all(r.source == "sharp_blobs" for r in result.regions)
