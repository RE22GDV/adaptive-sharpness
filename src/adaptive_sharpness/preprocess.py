"""Frame preparation for metric computation.

Design goals, in order:

1. **Do the smallest amount of work.**  When a ROI is supplied the crop happens
   *before* the downscale, so a small ROI never pays for resizing the whole
   frame and keeps its full detail up to ``analysis_width``.
2. **Avoid copies.**  The ROI crop is a NumPy view; the colour conversion and
   the resize each produce exactly one new array, written into a reused buffer
   where OpenCV allows it.
3. **Keep the display frame untouched.**  The caller's array is never modified.

Because the analysis scale differs between whole-frame mode and ROI mode, the
raw metric values are not comparable across a mid-run mode switch.  The running
normaliser adapts within a few frames, but a deliberate switch should reset the
evaluator (see :meth:`sharpness.evaluator.SharpnessEvaluator.reset`).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from .config import PipelineConfig
from .types import ROI

logger = logging.getLogger(__name__)

__all__ = ["AnalysisImage", "Preprocessor"]

_INTERPOLATIONS: Final[dict[str, int]] = {
    "INTER_NEAREST": cv2.INTER_NEAREST,
    "INTER_LINEAR": cv2.INTER_LINEAR,
    "INTER_AREA": cv2.INTER_AREA,
    "INTER_CUBIC": cv2.INTER_CUBIC,
}


@dataclass(frozen=True)
class AnalysisImage:
    """The float32 single-channel image the metrics actually see."""

    #: Analysis image in [0, 255], float32, 2-D.
    gray: np.ndarray
    #: Ratio analysis_size / source_size (< 1 when downscaled).
    scale: float
    #: ROI in source coordinates, or ``None`` for whole-frame analysis.
    roi: ROI | None
    #: Shape of the source frame, for reference.
    source_shape: tuple[int, ...]

    @property
    def size(self) -> tuple[int, int]:
        """``(width, height)`` of the analysis image."""
        return self.gray.shape[1], self.gray.shape[0]


class Preprocessor:
    """Turns a captured frame into an :class:`AnalysisImage`.

    A single instance is stateful only in its scratch buffers, so it is cheap to
    reuse across frames but must not be shared between threads.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if config.analysis_width < 16:
            raise ValueError("analysis_width must be at least 16 px")
        try:
            self._interpolation = _INTERPOLATIONS[config.interpolation]
        except KeyError:
            raise ValueError(
                f"unknown interpolation {config.interpolation!r}; "
                f"choose from {', '.join(_INTERPOLATIONS)}"
            ) from None
        # Reused output buffers; OpenCV writes in place when the shape matches.
        self._gray_buf: np.ndarray | None = None
        self._resize_buf: np.ndarray | None = None
        self._float_buf: np.ndarray | None = None

    def _to_gray(self, data: np.ndarray) -> np.ndarray:
        """Single-channel uint8/float view of ``data`` without copying if possible."""
        if data.ndim == 2:
            return data
        channels = data.shape[2]
        if channels == 1:
            return data[:, :, 0]
        code = cv2.COLOR_BGRA2GRAY if channels == 4 else cv2.COLOR_BGR2GRAY
        shape = data.shape[:2]
        if (
            self._gray_buf is None
            or self._gray_buf.shape != shape
            or self._gray_buf.dtype != data.dtype
        ):
            self._gray_buf = np.empty(shape, dtype=data.dtype)
        return cv2.cvtColor(data, code, dst=self._gray_buf)

    def _target_size(self, width: int, height: int) -> tuple[int, int] | None:
        """Target ``(w, h)`` for the analysis downscale, or ``None`` to keep as is."""
        target_w = self.config.analysis_width
        if width == target_w:
            return None
        if width < target_w and not self.config.allow_upscale:
            return None
        scale = target_w / float(width)
        target_h = max(8, int(round(height * scale)))
        return target_w, target_h

    def prepare(self, data: np.ndarray, roi: ROI | None = None) -> AnalysisImage:
        """Prepare ``data`` (an H x W [x C] array) for metric computation."""
        if data.ndim not in (2, 3):
            raise ValueError(f"expected a 2-D or 3-D frame, got shape {data.shape}")
        source_shape = data.shape

        gray = self._to_gray(data)

        # Crop first: a small ROI must not pay to resize the whole frame.
        clipped_roi: ROI | None = None
        if roi is not None:
            clipped_roi = roi.clipped_to(gray.shape)
            gray = gray[clipped_roi.y : clipped_roi.y2, clipped_roi.x : clipped_roi.x2]
        elif self.config.default_roi_fraction < 1.0:
            clipped_roi = ROI.centered(gray.shape, self.config.default_roi_fraction)
            gray = gray[clipped_roi.y : clipped_roi.y2, clipped_roi.x : clipped_roi.x2]

        height, width = gray.shape[:2]
        target = self._target_size(width, height)
        scale = 1.0
        if target is not None:
            target_w, target_h = target
            scale = target_w / float(width)
            if (
                self._resize_buf is None
                or self._resize_buf.shape != (target_h, target_w)
                or self._resize_buf.dtype != gray.dtype
            ):
                self._resize_buf = np.empty((target_h, target_w), dtype=gray.dtype)
            gray = cv2.resize(
                gray, (target_w, target_h),
                dst=self._resize_buf,
                interpolation=self._interpolation,
            )

        if gray.dtype != np.float32:
            shape = gray.shape
            if self._float_buf is None or self._float_buf.shape != shape:
                self._float_buf = np.empty(shape, dtype=np.float32)
            # Bring integer sensor data onto the documented [0, 255] scale.
            if gray.dtype == np.uint16:
                np.multiply(gray, 255.0 / 65535.0, out=self._float_buf, casting="unsafe")
            else:
                self._float_buf[...] = gray
            gray = self._float_buf
        elif gray.base is not None and not gray.flags["C_CONTIGUOUS"]:
            # A float32 non-contiguous view: OpenCV handles row-strided
            # submatrices, but np.fft does not benefit, so materialise it once.
            gray = np.ascontiguousarray(gray)

        if self.config.prefilter_sigma > 0.0:
            gray = cv2.GaussianBlur(gray, (0, 0), self.config.prefilter_sigma)

        return AnalysisImage(
            gray=gray,
            scale=scale,
            roi=clipped_roi,
            source_shape=source_shape,
        )
