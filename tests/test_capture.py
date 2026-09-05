"""Tests for the capture abstraction.

No hardware is required: the synthetic and file sources exercise the same
interface the camera backends implement.
"""
from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import pytest

from capture import (
    CaptureError,
    ImageDirectorySource,
    SyntheticSweepSource,
    ThreadedSource,
    VideoFileSource,
    open_source,
)
from capture.base import FrameSource
from sharpness.config import CaptureConfig
from sharpness.types import Frame


class SlowStubSource(FrameSource):
    """A deterministic source that takes a known time per frame."""

    def __init__(self, delay: float = 0.01, total: int | None = None) -> None:
        self.delay = delay
        self.total = total
        self.opened = False
        self.closed = False
        self.produced = 0
        self.description = "stub"

    @property
    def is_live(self) -> bool:
        return True

    def open(self) -> None:
        self.opened = True

    def read(self) -> Frame | None:
        if self.total is not None and self.produced >= self.total:
            return None
        time.sleep(self.delay)
        data = np.full((16, 16), self.produced % 256, dtype=np.uint8)
        frame = Frame(data=data, index=self.produced, source="stub")
        self.produced += 1
        return frame

    def close(self) -> None:
        self.closed = True


class TestSyntheticSource:
    def test_produces_the_expected_frames(self) -> None:
        source = SyntheticSweepSource(width=320, height=180, steps=11, loop=False)
        with source:
            frames = list(source.frames())
        assert len(frames) == 11
        assert frames[0].data.shape == (180, 320, 3)

    def test_carries_ground_truth_metadata(self) -> None:
        source = SyntheticSweepSource(width=320, height=180, steps=11, loop=False)
        with source:
            frame = source.read()
        assert frame is not None
        assert "motor_position" in frame.meta
        assert "defocus_radius" in frame.meta

    def test_best_focus_index_is_sharpest(self) -> None:
        from sharpness import SharpnessEvaluator

        source = SyntheticSweepSource(width=640, height=360, steps=21, loop=False)
        with source:
            frames = [f.data for f in source.frames()]
        evaluator = SharpnessEvaluator()
        for _ in range(2):
            for frame in frames:
                evaluator.evaluate(frame)
        scores = [evaluator.evaluate(f).instantaneous_score for f in frames]
        assert int(np.argmax(scores)) == source.best_focus_index

    def test_loops_when_asked(self) -> None:
        source = SyntheticSweepSource(width=160, height=90, steps=4, loop=True)
        with source:
            frames = list(source.frames(limit=10))
        assert len(frames) == 10

    def test_indices_are_monotone(self) -> None:
        source = SyntheticSweepSource(width=160, height=90, steps=5, loop=True)
        with source:
            indices = [f.index for f in source.frames(limit=8)]
        assert indices == list(range(8))


class TestFileSources:
    @pytest.fixture
    def image_dir(self, tmp_path: Path, scene: np.ndarray) -> Path:
        from tools.synthetic import defocus

        for index, radius in enumerate((0.0, 2.0, 4.0)):
            cv2.imwrite(str(tmp_path / f"frame_{index:03d}.png"), defocus(scene, radius))
        return tmp_path

    def test_reads_images_in_order(self, image_dir: Path) -> None:
        with ImageDirectorySource(image_dir) as source:
            names = [f.meta["name"] for f in source.frames()]
        assert names == ["frame_000.png", "frame_001.png", "frame_002.png"]

    def test_missing_directory_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CaptureError, match="not found"):
            ImageDirectorySource(tmp_path / "nope").open()

    def test_empty_directory_raises(self, tmp_path: Path) -> None:
        (tmp_path / "notes.txt").write_text("no images here")
        with pytest.raises(CaptureError, match="no images"):
            ImageDirectorySource(tmp_path).open()

    def test_missing_video_raises(self, tmp_path: Path) -> None:
        with pytest.raises(CaptureError, match="not found"):
            VideoFileSource(tmp_path / "missing.mp4").open()

    def test_replay_is_reproducible(self, image_dir: Path) -> None:
        def run() -> list[float]:
            from sharpness import SharpnessEvaluator

            evaluator = SharpnessEvaluator()
            with ImageDirectorySource(image_dir) as source:
                return [evaluator.evaluate(f).score for f in source.frames()]

        assert run() == run()


class TestThreadedSource:
    def test_delivers_frames(self) -> None:
        source = ThreadedSource(SlowStubSource(delay=0.005))
        try:
            source.open()
            frames = [source.read() for _ in range(5)]
        finally:
            source.close()
        assert all(f is not None for f in frames)

    def test_drops_stale_frames_for_a_slow_consumer(self) -> None:
        """The consumer must always get the newest frame, not a backlog."""
        stub = SlowStubSource(delay=0.005)
        source = ThreadedSource(stub)
        try:
            source.open()
            source.read()
            time.sleep(0.12)  # let the producer run far ahead
            frame = source.read()
        finally:
            source.close()
        assert frame is not None
        assert source.dropped > 0
        # The delivered frame is a recent one, not the second frame produced.
        assert frame.index >= stub.produced - 2

    def test_reports_exhaustion(self) -> None:
        source = ThreadedSource(SlowStubSource(delay=0.001, total=3))
        try:
            source.open()
            frames = []
            while True:
                frame = source.read()
                if frame is None:
                    break
                frames.append(frame)
        finally:
            source.close()
        assert len(frames) <= 3

    def test_close_stops_the_thread(self) -> None:
        stub = SlowStubSource(delay=0.002)
        source = ThreadedSource(stub)
        source.open()
        source.read()
        source.close()
        assert stub.closed
        assert source._thread is None

    def test_propagates_producer_errors(self) -> None:
        class Exploding(SlowStubSource):
            def read(self) -> Frame | None:
                raise RuntimeError("camera unplugged")

        source = ThreadedSource(Exploding())
        try:
            source.open()
            with pytest.raises(CaptureError, match="camera unplugged"):
                source.read()
        finally:
            source.close()

    def test_times_out_on_a_silent_producer(self) -> None:
        class Silent(SlowStubSource):
            def read(self) -> Frame | None:
                time.sleep(10.0)
                return None

        source = ThreadedSource(Silent(), poll_timeout=0.3)
        try:
            source.open()
            started = time.perf_counter()
            assert source.read() is None
            assert time.perf_counter() - started < 2.0
        finally:
            source.close()


class TestFactory:
    def test_builds_a_synthetic_source(self) -> None:
        config = CaptureConfig(backend="synthetic", width=160, height=90)
        source = open_source(config)
        try:
            assert source.read() is not None
        finally:
            source.close()

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(CaptureError, match="unknown capture backend"):
            open_source(CaptureConfig(backend="telepathy"))

    def test_file_backend_needs_a_path(self) -> None:
        with pytest.raises(CaptureError, match="capture.path"):
            open_source(CaptureConfig(backend="file", path=""))

    def test_file_backend_rejects_a_missing_path(self) -> None:
        with pytest.raises(CaptureError, match="does not exist"):
            open_source(CaptureConfig(backend="file", path="/nonexistent/xyz"))

    def test_offline_sources_are_not_threaded(self) -> None:
        """Wrapping a finite source in a thread would only add latency."""
        source = open_source(CaptureConfig(backend="synthetic", threaded=True))
        try:
            assert not isinstance(source, ThreadedSource)
        finally:
            source.close()
