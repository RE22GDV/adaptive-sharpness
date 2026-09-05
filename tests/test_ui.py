"""Tests for the UI's run-time switches and rendering.

The rendering tests do not check that the panel looks good - they check that it
is produced at all, for every state the operator can put the UI into, including
the awkward ones (no subject, every optional stage off).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

from focus_ui import MIN_CANVAS_HEIGHT, PANEL_WIDTH, UiState, describe, render  # noqa: E402

from sharpness import METRIC_NAMES, SceneEvaluator, SharpnessConfig  # noqa: E402
from tools.synthetic import defocus, make_scene  # noqa: E402


@pytest.fixture(scope="module")
def base() -> SharpnessConfig:
    return SharpnessConfig()


@pytest.fixture(scope="module")
def split_frame() -> np.ndarray:
    left = make_scene(640, 360, seed=3)
    right = make_scene(640, 360, seed=9)
    frame = left.copy()
    frame[:, 320:] = defocus(right, 6.0)[:, 320:]
    return frame


class TestMetricSwitches:
    def test_starts_with_every_metric(self) -> None:
        assert UiState().metrics == list(METRIC_NAMES)

    def test_toggle_removes_and_restores(self) -> None:
        state = UiState()
        assert state.toggle_metric(0)
        assert METRIC_NAMES[0] not in state.metrics
        assert state.toggle_metric(0)
        assert METRIC_NAMES[0] in state.metrics

    def test_order_is_preserved_after_restoring(self) -> None:
        state = UiState()
        state.toggle_metric(2)
        state.toggle_metric(2)
        assert state.metrics == list(METRIC_NAMES)

    def test_refuses_to_disable_the_last_metric(self) -> None:
        """An empty ensemble would raise deep inside the evaluator instead."""
        state = UiState()
        for index in range(len(METRIC_NAMES)):
            state.toggle_metric(index)
        assert len(state.metrics) == 1

    def test_out_of_range_index_is_ignored(self) -> None:
        state = UiState()
        assert not state.toggle_metric(99)
        assert state.metrics == list(METRIC_NAMES)

    def test_selection_reaches_the_config(self, base: SharpnessConfig) -> None:
        state = UiState()
        state.toggle_metric(0)
        assert state.apply(base).metrics.enabled == tuple(state.metrics)


class TestStageSwitches:
    def test_defaults_keep_every_stage_on(self, base: SharpnessConfig) -> None:
        config = UiState().apply(base)
        assert config.ensemble.use_noise_compensation
        assert config.ensemble.use_motion_compensation
        assert config.ensemble.use_agreement
        assert config.temporal.enabled

    def test_adaptive_off_zeroes_every_sensitivity(self, base: SharpnessConfig) -> None:
        """With no sensitivity and no agreement, the ensemble is the fixed
        weighted mean of the priors - the documented baseline."""
        config = UiState(adaptive=False).apply(base)
        assert all(
            value == 0.0
            for profile in config.ensemble.sensitivity.values()
            for value in profile.values()
        )
        assert not config.ensemble.use_agreement

    def test_adaptive_off_does_not_mutate_the_base(self, base: SharpnessConfig) -> None:
        UiState(adaptive=False).apply(base)
        assert any(
            value > 0.0
            for profile in base.ensemble.sensitivity.values()
            for value in profile.values()
        )

    @pytest.mark.parametrize(
        "field,path",
        [
            ("noise_compensation", "use_noise_compensation"),
            ("motion_compensation", "use_motion_compensation"),
            ("agreement", "use_agreement"),
        ],
    )
    def test_ensemble_switches(self, base: SharpnessConfig, field: str, path: str) -> None:
        state = UiState(**{field: False})
        assert getattr(state.apply(base).ensemble, path) is False

    def test_temporal_switch(self, base: SharpnessConfig) -> None:
        assert UiState(temporal=False).apply(base).temporal.enabled is False

    def test_faces_switch(self, base: SharpnessConfig) -> None:
        assert UiState(faces=False).apply(base).regions.detectors == ("sharp_blobs",)
        assert "faces" in UiState(faces=True).apply(base).regions.detectors

    def test_tile_size_switch(self, base: SharpnessConfig) -> None:
        state = UiState()
        first = state.apply(base).focus_map.tile_size
        state.tile_index = (state.tile_index + 1) % 4
        assert state.apply(base).focus_map.tile_size != first

    def test_summary_reports_disabled_stages(self) -> None:
        summary = UiState(adaptive=False, temporal=False).summary()
        assert "adapt" in summary and "temporal" in summary

    def test_summary_reports_all_on(self) -> None:
        assert "all stages on" in UiState().summary()


class TestSwitchesChangeResults:
    def test_disabling_metrics_changes_the_breakdown(
        self, base: SharpnessConfig, split_frame: np.ndarray
    ) -> None:
        full = SceneEvaluator(UiState().apply(base)).evaluate(split_frame)
        state = UiState()
        state.toggle_metric(0)
        state.toggle_metric(1)
        reduced = SceneEvaluator(state.apply(base)).evaluate(split_frame)
        assert full.subject_result is not None and reduced.subject_result is not None
        assert len(reduced.subject_result.metrics) == len(full.subject_result.metrics) - 2

    def test_adaptive_off_gives_the_prior_weights(
        self, base: SharpnessConfig, split_frame: np.ndarray
    ) -> None:
        config = UiState(adaptive=False).apply(base)
        result = SceneEvaluator(config).evaluate(split_frame)
        assert result.subject_result is not None
        weights = result.subject_result.weights
        priors = base.metrics.priors
        total = sum(priors[name] for name in weights)
        for name, weight in weights.items():
            assert weight == pytest.approx(priors[name] / total, abs=0.01)

    def test_weights_still_sum_to_one_with_stages_off(
        self, base: SharpnessConfig, split_frame: np.ndarray
    ) -> None:
        state = UiState(adaptive=False, noise_compensation=False,
                        motion_compensation=False, agreement=False, temporal=False)
        result = SceneEvaluator(state.apply(base)).evaluate(split_frame)
        assert result.subject_result is not None
        assert sum(result.subject_result.weights.values()) == pytest.approx(1.0)


class TestRendering:
    def test_renders_a_full_canvas(
        self, base: SharpnessConfig, split_frame: np.ndarray
    ) -> None:
        state = UiState()
        result = SceneEvaluator(state.apply(base)).evaluate(split_frame)
        canvas = render(split_frame, result, state, fps=25.0)
        assert canvas.ndim == 3 and canvas.shape[2] == 3
        assert canvas.shape[0] >= MIN_CANVAS_HEIGHT
        assert canvas.shape[1] > PANEL_WIDTH

    def test_renders_without_a_subject(self, base: SharpnessConfig) -> None:
        blank = np.full((360, 640, 3), 128, dtype=np.uint8)
        state = UiState()
        result = SceneEvaluator(state.apply(base)).evaluate(blank)
        assert not result.has_subject
        assert render(blank, result, state).shape[0] >= MIN_CANVAS_HEIGHT

    @pytest.mark.parametrize(
        "state",
        [
            UiState(heatmap=False),
            UiState(boxes=False),
            UiState(grid=True),
            UiState(adaptive=False),
            UiState(temporal=False, agreement=False),
            UiState(faces=False),
        ],
    )
    def test_renders_in_every_display_state(
        self, base: SharpnessConfig, split_frame: np.ndarray, state: UiState
    ) -> None:
        result = SceneEvaluator(state.apply(base)).evaluate(split_frame)
        canvas = render(split_frame, result, state, fps=25.0, history=[0.5, 0.6],
                        confidence_history=[0.8, 0.8])
        assert canvas.shape[0] >= MIN_CANVAS_HEIGHT

    def test_renders_with_a_reduced_metric_set(
        self, base: SharpnessConfig, split_frame: np.ndarray
    ) -> None:
        state = UiState()
        state.toggle_metric(0)
        result = SceneEvaluator(state.apply(base)).evaluate(split_frame)
        assert render(split_frame, result, state).shape[0] >= MIN_CANVAS_HEIGHT


class TestDescribe:
    def test_names_the_subject(self, base: SharpnessConfig, split_frame: np.ndarray) -> None:
        result = SceneEvaluator(UiState().apply(base)).evaluate(split_frame)
        assert "IN FOCUS" in describe(result)

    def test_says_so_when_nothing_is_in_focus(self, base: SharpnessConfig) -> None:
        blank = np.full((360, 640, 3), 128, dtype=np.uint8)
        result = SceneEvaluator(UiState().apply(base)).evaluate(blank)
        assert "nothing in focus" in describe(result)
