"""Offline frame sources: a video file, or a directory of still images.

These make every experiment reproducible.  A recorded focus sweep replayed
through :class:`ImageDirectorySource` yields byte-identical frames on every run,
which is what lets the method comparison in ``tools/compare_methods.py`` be a
controlled experiment rather than an anecdote.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Sequence

from ..types import Frame

from .base import CaptureError, FrameSource

logger = logging.getLogger(__name__)

__all__ = ["VideoFileSource", "ImageDirectorySource", "IMAGE_SUFFIXES"]

IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".ppm", ".pgm")


class VideoFileSource(FrameSource):
    """Frames decoded from a video file."""

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        self.loop = loop
        self._capture = None
        self._index = 0
        self.description = f"video {self.path.name} (not opened)"

    def open(self) -> None:
        import cv2

        if not self.path.is_file():
            raise CaptureError(f"video file not found: {self.path}")
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise CaptureError(f"could not decode video: {self.path}")
        self._capture = capture
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        self.description = f"video {self.path.name} ({count} frames @ {fps:.1f} fps)"
        logger.info("opened %s", self.description)

    def read(self) -> Frame | None:
        if self._capture is None:
            raise CaptureError("source is not open; call open() first")
        started = time.perf_counter()
        ok, image = self._capture.read()
        if not ok or image is None:
            if not self.loop:
                return None
            import cv2

            self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, image = self._capture.read()
            if not ok or image is None:
                return None
        delivered = time.perf_counter()
        frame = Frame(
            data=image,
            index=self._index,
            timestamp=delivered,
            capture_latency_s=delivered - started,
            source="video",
            meta={"path": str(self.path)},
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


class ImageDirectorySource(FrameSource):
    """Frames read from a directory of images, in sorted filename order.

    Filenames are exposed in ``Frame.meta['name']`` so the data collector can
    record which file each row came from.
    """

    def __init__(self, path: str | Path, loop: bool = False) -> None:
        self.path = Path(path)
        self.loop = loop
        self._files: Sequence[Path] = ()
        self._position = 0
        self._index = 0
        self.description = f"images {self.path} (not opened)"

    def open(self) -> None:
        if not self.path.is_dir():
            raise CaptureError(f"image directory not found: {self.path}")
        files = sorted(
            p for p in self.path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
        )
        if not files:
            raise CaptureError(
                f"no images in {self.path} "
                f"(looked for {', '.join(IMAGE_SUFFIXES)})"
            )
        self._files = files
        self._position = 0
        self.description = f"images {self.path.name} ({len(files)} files)"
        logger.info("opened %s", self.description)

    def read(self) -> Frame | None:
        import cv2

        if not self._files:
            raise CaptureError("source is not open; call open() first")
        if self._position >= len(self._files):
            if not self.loop:
                return None
            self._position = 0

        path = self._files[self._position]
        self._position += 1
        started = time.perf_counter()
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("skipping unreadable image: %s", path)
            return self.read()
        delivered = time.perf_counter()
        frame = Frame(
            data=image,
            index=self._index,
            timestamp=delivered,
            capture_latency_s=delivered - started,
            source="images",
            meta={"path": str(path), "name": path.name},
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._files = ()
        self._position = 0
