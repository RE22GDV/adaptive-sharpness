"""V4L2 / UVC capture through OpenCV.

Used when the camera (or an external HDMI-to-USB capture stick) presents a real
``/dev/video*`` capture node.  The Panasonic GH6 does **not** in its tethering
USB mode - see :mod:`capture.gphoto_source` - so this backend exists for the
fallback hardware path and for ordinary webcams during development.

``tools/probe_camera.py`` distinguishes real capture nodes from the Raspberry
Pi's own ISP/codec nodes (``pispbe``, ``rpi-hevc-dec``), which also appear as
``/dev/video*`` but cannot capture anything.
"""
from __future__ import annotations

import logging
import time

from ..types import Frame

from .base import CaptureError, FrameSource

logger = logging.getLogger(__name__)

__all__ = ["V4L2Source"]


class V4L2Source(FrameSource):
    """Frames from a V4L2 capture device."""

    def __init__(
        self,
        device: str | int = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        fourcc: str = "MJPG",
        buffer_size: int = 1,
    ) -> None:
        self.device = device
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.buffer_size = buffer_size
        self._capture = None
        self._index = 0
        self.description = f"v4l2 {device} (not opened)"

    @property
    def is_live(self) -> bool:
        return True

    def open(self) -> None:
        import cv2

        target: str | int = self.device
        if isinstance(target, str) and target.isdigit():
            target = int(target)

        capture = cv2.VideoCapture(target, cv2.CAP_V4L2)
        if not capture.isOpened():
            # Fall back to the default backend, which helps on non-Linux hosts.
            capture = cv2.VideoCapture(target)
        if not capture.isOpened():
            raise CaptureError(
                f"could not open V4L2 device {self.device!r}. "
                "Run tools/probe_camera.py to list the real capture nodes."
            )

        if self.fourcc:
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self.fourcc))
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        capture.set(cv2.CAP_PROP_FPS, self.fps)
        # A one-frame driver queue keeps the newest frame closest to real time;
        # not every driver honours it, which is why ThreadedSource also drops.
        try:
            capture.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)
        except cv2.error:  # pragma: no cover - backend dependent
            pass

        actual_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = capture.get(cv2.CAP_PROP_FPS)
        self._capture = capture
        self.description = (
            f"v4l2 {self.device} {actual_w}x{actual_h} @ {actual_fps:.0f} fps"
        )
        if (actual_w, actual_h) != (self.width, self.height):
            logger.warning(
                "device gave %dx%d instead of the requested %dx%d",
                actual_w, actual_h, self.width, self.height,
            )
        logger.info("opened %s", self.description)

    def read(self) -> Frame | None:
        if self._capture is None:
            raise CaptureError("source is not open; call open() first")
        started = time.perf_counter()
        ok, image = self._capture.read()
        if not ok or image is None:
            logger.info("V4L2 device returned no frame; treating as end of stream")
            return None
        delivered = time.perf_counter()
        elapsed = delivered - started
        frame = Frame(
            data=image,
            index=self._index,
            timestamp=delivered,
            capture_latency_s=elapsed,
            source="v4l2",
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
            logger.info("released %s", self.description)
