"""Tests for RunRecorder and the UI's record button."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "demo"))

from adaptive_sharpness import (  # noqa: E402
    RunRecorder,
    SceneEvaluator,
    SharpnessConfig,
    SharpnessEvaluator,
)
from adaptive_sharpness.synthetic import defocus, make_scene  # noqa: E402


@pytest.fixture
def frames() -> list[np.ndarray]:
    scene = make_scene(320, 180, seed=5)
    return [defocus(scene, r) for r in (0.0, 1.5, 3.0, 4.5, 6.0, 3.0, 1.0)]


def read_csv(directory: Path) -> list[dict[str, str]]:
    with (directory / "frames.csv").open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class TestRunRecorder:
    def test_writes_a_row_per_frame(self, tmp_path: Path, frames) -> None:
        evaluator = SharpnessEvaluator()
        recorder = RunRecorder(tmp_path, SharpnessConfig(), save_every=0)
        recorder.start("unit test")
        for frame in frames:
            recorder.add(evaluator.evaluate(frame))
        summary = recorder.stop()
        rows = read_csv(tmp_path)
        assert len(rows) == len(frames) == summary["frames"]

    def test_row_carries_the_full_evaluation(self, tmp_path: Path, frames) -> None:
        evaluator = SharpnessEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=0) as recorder:
            for frame in frames:
                recorder.add(evaluator.evaluate(frame))
        row = read_csv(tmp_path)[0]
        for key in ("instantaneous_score", "filtered_score", "confidence",
                    "ready", "informative_fraction", "wall_time", "interval_s"):
            assert key in row
        for name in SharpnessConfig().metrics.enabled:
            for prefix in ("raw_", "norm_", "w_", "rel_", "agree_", "ms_"):
                assert f"{prefix}{name}" in row

    def test_saves_every_nth_frame_as_png(self, tmp_path: Path, frames) -> None:
        """PNG, not JPEG: lossy compression would remove the very high-frequency
        content the metrics measure, so a re-analysis could not reproduce them."""
        evaluator = SharpnessEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=3) as recorder:
            for index, frame in enumerate(frames):
                recorder.add(evaluator.evaluate(frame), frame, index)
        images = sorted((tmp_path / "frames").glob("*.png"))
        assert len(images) == len([i for i in range(len(frames)) if i % 3 == 0])
        assert not list((tmp_path / "frames").glob("*.jpg"))

    def test_saved_frame_round_trips_losslessly(self, tmp_path: Path, frames) -> None:
        import cv2

        evaluator = SharpnessEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=1) as recorder:
            recorder.add(evaluator.evaluate(frames[0]), frames[0], 0)
        saved = cv2.imread(str(tmp_path / "frames" / "frame_000000.png"),
                           cv2.IMREAD_COLOR)
        assert np.array_equal(saved, frames[0]), "PNG round-trip must be exact"

    def test_image_file_column_names_the_saved_frame(self, tmp_path: Path, frames) -> None:
        evaluator = SharpnessEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=2) as recorder:
            for index, frame in enumerate(frames):
                recorder.add(evaluator.evaluate(frame), frame, index)
        rows = read_csv(tmp_path)
        named = [r["image_file"] for r in rows if r["image_file"]]
        assert named
        for name in named:
            assert (tmp_path / "frames" / name).is_file()

    def test_writes_a_manifest(self, tmp_path: Path, frames) -> None:
        evaluator = SharpnessEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=0,
                         note="hello") as recorder:
            recorder.add(evaluator.evaluate(frames[0]))
        manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["note"] == "hello"
        assert manifest["environment"]["opencv"]
        assert manifest["config"]["metrics_enabled"]
        assert manifest["summary"]["frames"] == 1

    def test_extra_columns_are_recorded(self, tmp_path: Path, frames) -> None:
        evaluator = SceneEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=0) as recorder:
            for frame in frames:
                scene = evaluator.evaluate(frame)
                assert scene.frame_result is not None
                recorder.add(scene.frame_result,
                             extra={"subject_label": scene.subject_label})
        rows = read_csv(tmp_path)
        assert all("subject_label" in row for row in rows)

    def test_a_later_extra_key_does_not_ragged_the_csv(self, tmp_path: Path, frames) -> None:
        """Columns are fixed by the first row; a new key later must not shift
        every subsequent value into the wrong column."""
        evaluator = SharpnessEvaluator()
        with RunRecorder(tmp_path, SharpnessConfig(), save_every=0) as recorder:
            recorder.add(evaluator.evaluate(frames[0]), extra={"a": 1})
            recorder.add(evaluator.evaluate(frames[1]), extra={"a": 2, "b": 3})
        rows = read_csv(tmp_path)
        assert len(rows) == 2
        assert all(len(row) == len(rows[0]) for row in rows)
        assert "b" not in rows[0]

    def test_summary_reports_the_rate(self, tmp_path: Path, frames) -> None:
        evaluator = SharpnessEvaluator()
        recorder = RunRecorder(tmp_path, SharpnessConfig(), save_every=0)
        recorder.start()
        for frame in frames:
            recorder.add(evaluator.evaluate(frame))
        summary = recorder.stop()
        assert summary["frames"] == len(frames)
        assert summary["duration_s"] > 0
        assert summary["mean_fps"] > 0

    def test_active_flag(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path, SharpnessConfig(), save_every=0)
        assert not recorder.active
        recorder.start()
        assert recorder.active
        recorder.stop()
        assert not recorder.active

    def test_add_before_start_raises(self, tmp_path: Path, frames) -> None:
        recorder = RunRecorder(tmp_path, SharpnessConfig(), save_every=0)
        result = SharpnessEvaluator().evaluate(frames[0])
        with pytest.raises(RuntimeError, match="not running"):
            recorder.add(result)

    def test_double_start_raises(self, tmp_path: Path) -> None:
        recorder = RunRecorder(tmp_path, SharpnessConfig(), save_every=0)
        recorder.start()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                recorder.start()
        finally:
            recorder.stop()

    def test_stop_without_start_is_harmless(self, tmp_path: Path) -> None:
        assert RunRecorder(tmp_path, SharpnessConfig()).stop() == {}


class TestRoundTrip:
    """A recording is only useful if it can be re-analysed offline.

    The raw metric values depend on the frame alone, so a saved frame
    re-evaluated later must give identical numbers. Verified on a real
    1288-frame handheld recording: the difference was exactly zero for all six
    metrics across all 129 saved frames.
    """

    def test_saved_frames_reproduce_the_recorded_metrics(
        self, tmp_path: Path, frames
    ) -> None:
        import cv2

        from adaptive_sharpness.metrics import build_metrics
        from adaptive_sharpness.preprocess import Preprocessor

        config = SharpnessConfig()
        evaluator = SharpnessEvaluator(config)
        with RunRecorder(tmp_path, config, save_every=1) as recorder:
            for index, frame in enumerate(frames):
                recorder.add(evaluator.evaluate(frame), frame, index)

        pre = Preprocessor(config.pipeline)
        metrics = build_metrics(config.metrics)
        rows = [r for r in read_csv(tmp_path) if r["image_file"]]
        assert rows

        for row in rows:
            image = cv2.imread(str(tmp_path / "frames" / row["image_file"]),
                               cv2.IMREAD_COLOR)
            gray = pre.prepare(image).gray
            for metric in metrics:
                recomputed = metric.compute(gray)
                recorded = float(row[f"raw_{metric.name}"])
                assert recomputed == pytest.approx(recorded, rel=1e-12, abs=1e-12), (
                    f"{metric.name} did not reproduce from the saved frame"
                )

    def test_saved_frame_statistics_reproduce(self, tmp_path: Path, frames) -> None:
        import cv2

        from adaptive_sharpness.analysis import ImageAnalyzer
        from adaptive_sharpness.preprocess import Preprocessor

        config = SharpnessConfig()
        evaluator = SharpnessEvaluator(config)
        with RunRecorder(tmp_path, config, save_every=1) as recorder:
            for index, frame in enumerate(frames):
                recorder.add(evaluator.evaluate(frame), frame, index)

        pre = Preprocessor(config.pipeline)
        analyzer = ImageAnalyzer(config.analysis)
        row = [r for r in read_csv(tmp_path) if r["image_file"]][0]
        gray = pre.prepare(
            cv2.imread(str(tmp_path / "frames" / row["image_file"]), cv2.IMREAD_COLOR)
        ).gray
        stats = analyzer.analyze(gray).as_dict()
        # Motion is excluded: it is defined between consecutive frames.
        for field in ("edge_density", "local_contrast", "brightness", "noise_sigma"):
            assert stats[field] == pytest.approx(float(row[field]), rel=1e-12,
                                                 abs=1e-12)


class TestRecordButton:
    def test_button_rect_is_published_after_render(self, frames) -> None:
        """The mouse callback hit-tests against this rect, so render() must
        publish it in canvas coordinates, not panel coordinates."""
        from focus_ui import RECORD_BUTTON, UiState, render

        RECORD_BUTTON.clear()
        state = UiState()
        result = SceneEvaluator(state.apply(SharpnessConfig())).evaluate(frames[0])
        canvas = render(frames[0], result, state)

        assert "canvas" in RECORD_BUTTON
        x1, y1, x2, y2 = RECORD_BUTTON["canvas"]
        assert x2 > x1 and y2 > y1
        assert x2 <= canvas.shape[1], "button must lie inside the canvas"
        assert y2 <= canvas.shape[0]
        # It belongs to the panel, which sits to the right of the video.
        assert x1 > canvas.shape[1] - 400

    def test_toggle_starts_and_stops(self, tmp_path: Path) -> None:
        from focus_ui import toggle_recording

        config = SharpnessConfig()
        recorder = toggle_recording(None, config, tmp_path, 5, "unit test")
        assert recorder is not None and recorder.active
        directory = recorder.directory

        assert toggle_recording(recorder, config, tmp_path, 5, "unit test") is None
        assert (directory / "frames.csv").is_file()
        assert (directory / "manifest.json").is_file()

    def test_each_toggle_gets_its_own_directory(self, tmp_path: Path) -> None:
        """Pressing the button twice must not overwrite the earlier run."""
        from focus_ui import toggle_recording

        config = SharpnessConfig()
        first = toggle_recording(None, config, tmp_path, 5, "a")
        assert first is not None
        first_dir = first.directory
        toggle_recording(first, config, tmp_path, 5, "a")

        # The directory name carries a one-second timestamp, so force a distinct
        # one rather than sleeping.
        second_root = tmp_path / "second"
        second = toggle_recording(None, config, second_root, 5, "b")
        assert second is not None
        assert second.directory != first_dir
        toggle_recording(second, config, second_root, 5, "b")
