"""PTP live-view capture through libgphoto2.

This is the working path for the Panasonic GH6 over USB-C.  The camera enumerates
as USB class 6 (Imaging) / subclass 1 (Still Image Capture) / protocol 1 (PTP)
and exposes **no** UVC interface in either of its two USB configurations, so
there is no ``/dev/video*`` node and V4L2 capture is not possible in that mode.
libgphoto2's ``capture_preview`` returns the camera's live-view JPEG, which is
what this backend decodes.

Measured on a Raspberry Pi 5 with a GH6 in "PC(Tether)" USB mode:
640x360 JPEG, ~28 kB per frame, ~38.6 ms per grab and ~1.4 ms to decode, giving
a sustained 25 fps.  The grab time is dominated by waiting on the camera, which
is why :class:`~capture.base.ThreadedSource` is worth using here.

If another process holds the camera the init call fails; on a Raspberry Pi OS
desktop the usual culprit is the gvfs PTP volume monitor::

    pkill -f gvfsd-gphoto2
    pkill -f gvfs-gphoto2-volume-monitor
"""
from __future__ import annotations

import logging
import time

import numpy as np

from sharpness.types import Frame

from .base import CaptureError, FrameSource

logger = logging.getLogger(__name__)

__all__ = ["GPhotoLiveViewSource"]


class GPhotoLiveViewSource(FrameSource):
    """Live-view frames from a PTP camera via libgphoto2."""

    def __init__(self, decode_color: bool = True, max_consecutive_errors: int = 5) -> None:
        self._camera = None
        self._gp = None
        self._cv2 = None
        self._decode_flag: int | None = None
        self._index = 0
        self._errors = 0
        self._max_errors = max_consecutive_errors
        self._decode_color = decode_color
        self.description = "gphoto2 live view (not opened)"

    @property
    def is_live(self) -> bool:
        return True

    def open(self) -> None:
        try:
            import gphoto2 as gp
        except ImportError as exc:
            raise CaptureError(
                "python-gphoto2 is not installed. On Raspberry Pi OS:\n"
                "  sudo apt install gphoto2 libgphoto2-dev python3-gphoto2"
            ) from exc
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - opencv is a hard dep
            raise CaptureError("OpenCV is required to decode preview JPEGs") from exc

        self._gp = gp
        self._cv2 = cv2
        self._decode_flag = cv2.IMREAD_COLOR if self._decode_color else cv2.IMREAD_GRAYSCALE

        camera = gp.Camera()
        try:
            camera.init()
        except gp.GPhoto2Error as exc:
            raise CaptureError(
                f"could not claim the camera over PTP ({exc}). "
                "Check that it is switched on, that its USB mode is the tethering "
                "mode (not mass storage), and that no other process holds it "
                "(pkill -f gvfsd-gphoto2)."
            ) from exc
        self._camera = camera

        try:
            summary = str(camera.get_summary())
            model = summary.splitlines()[0].strip() if summary else "unknown"
        except Exception:  # noqa: BLE001 - the summary is informational only
            model = "unknown"
        self.description = f"gphoto2 live view [{model}]"
        logger.info("opened %s", self.description)

    def read(self) -> Frame | None:
        if self._camera is None or self._gp is None or self._cv2 is None:
            raise CaptureError("source is not open; call open() first")
        gp = self._gp

        started = time.perf_counter()
        try:
            capture = self._camera.capture_preview()
            payload = capture.get_data_and_size()
        except gp.GPhoto2Error as exc:
            self._errors += 1
            logger.warning("capture_preview failed (%d/%d): %s",
                           self._errors, self._max_errors, exc)
            if self._errors >= self._max_errors:
                raise CaptureError(
                    f"live view failed {self._errors} times in a row: {exc}"
                ) from exc
            return self.read()

        # np.frombuffer wraps the libgphoto2 buffer without copying; imdecode
        # then produces the only real allocation per frame.
        buffer = np.frombuffer(memoryview(payload), dtype=np.uint8)
        image = self._cv2.imdecode(buffer, self._decode_flag)
        if image is None:
            self._errors += 1
            logger.warning("preview JPEG failed to decode (%d bytes)", buffer.size)
            if self._errors >= self._max_errors:
                raise CaptureError("preview JPEGs are not decodable")
            return self.read()

        self._errors = 0
        delivered = time.perf_counter()
        elapsed = delivered - started
        frame = Frame(
            data=image,
            index=self._index,
            timestamp=delivered,
            capture_latency_s=elapsed,
            source="gphoto2",
            meta={"jpeg_bytes": int(buffer.size)},
        )
        self._index += 1
        return frame

    def close(self) -> None:
        if self._camera is not None:
            try:
                self._camera.exit()
            except Exception as exc:  # noqa: BLE001 - best-effort release
                logger.debug("camera.exit() raised: %s", exc)
            self._camera = None
            logger.info("released the PTP camera")
