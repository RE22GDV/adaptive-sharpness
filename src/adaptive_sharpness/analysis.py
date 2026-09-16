"""Measurement of the image properties that drive the adaptive weighting.

Each quantity is chosen so that it is *independent of focus*: the weighting must
react to the shooting conditions, not to the thing being measured.  That rules
out, for example, using the temporal intensity difference as a motion cue, since
a focus change alters every pixel without anything moving.  Global translation
estimated by phase correlation has the required property and is used instead.

Estimators
----------
noise
    Median-absolute-deviation of the finest Haar detail band,
    ``sigma = median(|HH|) / 0.6745``.  This is the standard robust estimator
    from Donoho & Johnstone (1994): the HH band is dominated by noise, and the
    median makes the sparse edge contributions negligible.
contrast
    RMS contrast, ``std(I) / 255``.
exposure
    Fraction of pixels pinned at either end of the range, plus a penalty when
    the mean intensity leaves the well-exposed band.
edges
    Fraction of Canny edge pixels, compared against a reference density.
motion
    Magnitude of the global translation between consecutive frames, from
    :func:`cv2.phaseCorrelate` on a decimated copy.  Non-rigid motion (a subject
    moving inside a static frame) is *not* captured — see docs/LIMITATIONS.md.
anisotropy
    ``(l1 - l2) / (l1 + l2)`` of the gradient structure tensor, i.e. how
    strongly the gradients favour one orientation.
"""
from __future__ import annotations

import logging

import cv2
import numpy as np

from statistics import NormalDist

from .config import AnalysisConfig
from .metrics.frequency import haar_decompose
from .types import ImageStats

logger = logging.getLogger(__name__)

__all__ = ["ImageAnalyzer", "DEGRADATION_KEYS"]

# Order matters only for readability; the ensemble looks these up by name.
DEGRADATION_KEYS = ("noise", "edge", "clip", "motion", "contrast")

# MAD-to-sigma conversion factor for Gaussian data.
_MAD_SCALE = 1.0 / 0.6744897501960817


def _half_normal_scale(percent: float) -> float:
    """Divisor turning the p-th quantile of |X| into sigma, for X ~ N(0, sigma).

    The quantile of the folded normal at probability p is sigma * Phi^-1((1+p)/2),
    so dividing by that factor recovers sigma whatever quantile is used.  At
    p = 50 this reproduces the classical Donoho-Johnstone MAD constant, which
    is asserted in the tests.
    """
    if not 0.0 < percent < 100.0:
        raise ValueError("noise_quantile must lie strictly between 0 and 100")
    return NormalDist().inv_cdf((1.0 + percent / 100.0) / 2.0)


class ImageAnalyzer:
    """Computes :class:`ImageStats` for successive analysis images.

    The analyzer keeps the previous frame in order to estimate motion, so a
    single instance corresponds to a single video stream.  Call :meth:`reset`
    when the stream is discontinuous.
    """

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config
        if config.motion_downscale < 1:
            raise ValueError("motion_downscale must be >= 1")
        self._prev_small: np.ndarray | None = None
        self._hann: np.ndarray | None = None
        self._hann_shape: tuple[int, int] | None = None
        self._bias: tuple[float, float] = (0.0, 0.0)
        self._noise_scale = _half_normal_scale(config.noise_quantile)

    def reset(self) -> None:
        """Forget the previous frame (used when the stream restarts)."""
        self._prev_small = None

    # -- individual estimators -------------------------------------------------

    def estimate_noise_sigma(self, gray: np.ndarray) -> float:
        """Robust noise standard deviation in 8-bit intensity units."""
        if min(gray.shape) < 4:
            return 0.0
        _, _, _, hh = haar_decompose(gray)
        if hh.size == 0:
            return 0.0
        # The Donoho-Johnstone estimator, evaluated at a configurable quantile
        # rather than always at the median.  On an 8-bit stream the majority of
        # HH coefficients can be exactly zero, which pins the median at zero and
        # reports no noise however much there is; a higher quantile reads the
        # same distribution where it is not degenerate.  See
        # AnalysisConfig.noise_quantile.
        quantile = float(
            np.percentile(np.abs(hh), self.config.noise_quantile)
        )
        return quantile / self._noise_scale

    def estimate_edges(self, gray: np.ndarray) -> float:
        """Fraction of pixels that Canny marks as an edge."""
        u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        edges = cv2.Canny(u8, 50, 150, L2gradient=True)
        return float(np.count_nonzero(edges)) / float(edges.size)

    def estimate_motion_px(self, gray: np.ndarray) -> float:
        """Global translation magnitude against the previous frame, in pixels."""
        step = self.config.motion_downscale
        height, width = gray.shape
        if step > 1 and min(height // step, width // step) >= 8:
            # Area-averaged resize, not strided slicing: plain decimation
            # aliases, and the aliasing pattern changes with the blur level, so
            # phase correlation reports a spurious shift when only the focus
            # moved.
            small = cv2.resize(
                gray, (width // step, height // step), interpolation=cv2.INTER_AREA
            )
        else:
            small = gray
            step = 1
        small = np.ascontiguousarray(small, dtype=np.float32)

        prev = self._prev_small
        # cv2.phaseCorrelate MODIFIES the arrays it is given (it windows them in
        # place for contiguous CV_32F input).  Keeping a pristine copy for the
        # next frame is essential: passing the stored array straight in silently
        # corrupted the reference, which showed up as a drifting ~0.5 px of
        # phantom motion on a perfectly static scene.
        self._prev_small = small.copy()
        if prev is None or prev.shape != small.shape:
            return 0.0

        shape = small.shape
        if self._hann_shape != shape:
            self._hann = cv2.createHanningWindow((shape[1], shape[0]), cv2.CV_32F)
            self._hann_shape = shape
            # cv2.phaseCorrelate's sub-pixel peak refinement carries a constant
            # offset that depends on the transform size, not on the content:
            # correlating an array with ITSELF returns e.g. (0.0, 0.5) at 44x80
            # but (0.0, 0.0) at 46x82.  Left uncorrected it reported ~2 px of
            # motion on a completely static scene.  The offset is measured once
            # per shape and subtracted.
            try:
                # Throwaway copies: see the in-place modification note above.
                probe = small.copy()
                self._bias, _ = cv2.phaseCorrelate(probe, probe.copy(), self._hann)
            except cv2.error:  # pragma: no cover - degenerate frames only
                self._bias = (0.0, 0.0)
            logger.debug("phase-correlation bias for %s: %s", shape, self._bias)

        try:
            (dx, dy), _response = cv2.phaseCorrelate(prev, small, self._hann)
        except cv2.error as exc:  # pragma: no cover - degenerate frames only
            logger.debug("phaseCorrelate failed: %s", exc)
            return 0.0
        dx -= self._bias[0]
        dy -= self._bias[1]
        return float(np.hypot(dx, dy) * step)

    def estimate_anisotropy(self, gray: np.ndarray) -> float:
        """Directional bias of the gradient field, in [0, 1]."""
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        jxx = float(np.mean(gx * gx))
        jyy = float(np.mean(gy * gy))
        jxy = float(np.mean(gx * gy))
        trace = jxx + jyy
        if trace <= 1e-9:
            return 0.0
        # Closed-form eigenvalue difference of the 2x2 structure tensor.
        diff = float(np.sqrt(max(0.0, (jxx - jyy) ** 2 + 4.0 * jxy * jxy)))
        return min(1.0, diff / trace)

    # -- public API ------------------------------------------------------------

    def analyze(self, gray: np.ndarray) -> ImageStats:
        """Measure every content descriptor for one analysis image."""
        cfg = self.config

        mean = float(gray.mean())
        std = float(gray.std())
        brightness = mean / 255.0
        local_contrast = std / 255.0

        total = float(gray.size)
        clipped_high = float(np.count_nonzero(gray >= cfg.clip_high_level)) / total
        clipped_low = float(np.count_nonzero(gray <= cfg.clip_low_level)) / total

        noise_sigma = self.estimate_noise_sigma(gray)
        noise_level = min(1.0, noise_sigma / max(cfg.noise_ref_sigma, 1e-6))
        snr = (std / noise_sigma) if noise_sigma > 1e-6 else float("inf")
        if not np.isfinite(snr):
            # No measurable noise: report a large but finite SNR so downstream
            # arithmetic and CSV serialisation stay well defined.
            snr = 1e3

        edge_density = self.estimate_edges(gray)
        edge_sufficiency = min(1.0, edge_density / max(cfg.edge_ref_density, 1e-9))

        motion_px = self.estimate_motion_px(gray)
        motion_level = min(1.0, motion_px / max(cfg.motion_ref_px, 1e-9))

        anisotropy = self.estimate_anisotropy(gray)

        return ImageStats(
            noise_sigma=noise_sigma,
            noise_level=noise_level,
            snr=float(snr),
            local_contrast=local_contrast,
            brightness=brightness,
            clipped_high=clipped_high,
            clipped_low=clipped_low,
            edge_density=edge_density,
            edge_sufficiency=edge_sufficiency,
            motion_px=motion_px,
            motion_level=motion_level,
            anisotropy=anisotropy,
        )

    def degradations(self, stats: ImageStats) -> dict[str, float]:
        """Map :class:`ImageStats` onto the degradation factors ``d_j``.

        Every value lies in ``[0, 1]``, where 0 means "ideal conditions for the
        metrics" and 1 means "this condition is as bad as the model accounts
        for".
        """
        cfg = self.config

        exposure = min(1.0, stats.clipped_total / max(cfg.clip_ref_fraction, 1e-9))
        # Add a penalty for a mean intensity outside the well-exposed band.
        if stats.brightness < cfg.brightness_low:
            deficit = (cfg.brightness_low - stats.brightness) / max(cfg.brightness_low, 1e-9)
            exposure = max(exposure, min(1.0, deficit))
        elif stats.brightness > cfg.brightness_high:
            excess = (stats.brightness - cfg.brightness_high) / max(
                1.0 - cfg.brightness_high, 1e-9
            )
            exposure = max(exposure, min(1.0, excess))

        contrast_sufficiency = min(
            1.0, stats.local_contrast / max(cfg.contrast_ref, 1e-9)
        )

        return {
            "noise": stats.noise_level,
            "edge": 1.0 - stats.edge_sufficiency,
            "clip": exposure,
            "motion": stats.motion_level,
            "contrast": 1.0 - contrast_sufficiency,
        }
