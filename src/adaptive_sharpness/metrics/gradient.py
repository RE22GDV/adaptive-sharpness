"""Spatial-derivative sharpness metrics.

These three operators all measure the energy of the image derivatives, which
grows as the optical blur kernel narrows.  They differ in derivative order and
support, which is exactly why they degrade differently under noise:

* :class:`LaplacianVariance` uses a second derivative and therefore has the
  steepest high-frequency response — the most focus-sensitive of the three, and
  the most noise-sensitive.
* :class:`Tenengrad` uses first-order Sobel derivatives with 3x3 smoothing
  built in, which suppresses part of the sensor noise.
* :class:`Brenner` uses a plain two-pixel difference: cheap, but it needs real
  structure at that specific scale.

References
----------
Brenner et al. (1976), *An automated microscope for cytologic research*.
Krotkov (1987), *Focusing* (Tenengrad).
Pech-Pacheco et al. (2000), *Diatom autofocusing in brightfield microscopy*
(variance of the Laplacian).
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import SharpnessMetric, normalization_factor

__all__ = ["LaplacianVariance", "Tenengrad", "Brenner"]


class LaplacianVariance(SharpnessMetric):
    """Variance of the 3x3 Laplacian response.

    A sharp image has a broad distribution of second-derivative values; blur
    collapses that distribution towards zero.
    """

    name = "laplacian"

    def _compute(self, gray: np.ndarray) -> float:
        lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
        # Using the mean of squares rather than cv2.meanStdDev keeps this a
        # single pass over the data.
        value = float(np.mean(lap * lap) - np.mean(lap) ** 2)
        return value / normalization_factor(gray, self.contrast_normalize)


class Tenengrad(SharpnessMetric):
    """Mean squared Sobel gradient magnitude.

    The classical Tenengrad only accumulates gradients above a threshold; the
    threshold is kept optional here (``gradient_threshold``) because on a
    downscaled analysis image the unthresholded mean is both cheaper and less
    scene-dependent.
    """

    name = "tenengrad"

    def __init__(
        self,
        contrast_normalize: bool = True,
        gradient_threshold: float = 0.0,
        ksize: int = 3,
    ) -> None:
        super().__init__(contrast_normalize)
        self.gradient_threshold = gradient_threshold
        self.ksize = ksize

    def _compute(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.ksize)
        energy = gx * gx + gy * gy
        if self.gradient_threshold > 0.0:
            thr = self.gradient_threshold * self.gradient_threshold
            energy = np.where(energy >= thr, energy, 0.0)
        value = float(np.mean(energy))
        return value / normalization_factor(gray, self.contrast_normalize)


class Brenner(SharpnessMetric):
    """Squared difference between pixels ``shift`` apart.

    The textbook operator is horizontal only.  This implementation averages the
    horizontal and vertical responses so the metric does not favour one edge
    orientation, which matters because the ensemble separately measures gradient
    anisotropy and would otherwise double-count it.
    """

    name = "brenner"

    def __init__(self, contrast_normalize: bool = True, shift: int = 2) -> None:
        super().__init__(contrast_normalize)
        if shift < 1:
            raise ValueError("Brenner shift must be >= 1")
        self.shift = shift
        self.min_size = max(8, shift + 2)

    def _compute(self, gray: np.ndarray) -> float:
        k = self.shift
        dh = gray[:, k:] - gray[:, :-k]
        dv = gray[k:, :] - gray[:-k, :]
        value = 0.5 * (float(np.mean(dh * dh)) + float(np.mean(dv * dv)))
        return value / normalization_factor(gray, self.contrast_normalize)


class GradientVariance(SharpnessMetric):
    """Variance of the Sobel gradient magnitude (Pertuz et al. 2013, "TENV").

    Closely related to :class:`Tenengrad`, which averages the *squared*
    magnitude.  Subtracting the mean makes this one respond to the spread of
    edge strengths rather than to their total energy, so a frame that is
    uniformly textured scores lower than one carrying a few sharp edges of the
    same total energy.

    Added on the evidence of the protocol study rather than by analogy: over
    eight tripod recordings with exact focus-step labels it separated adjacent
    focus positions better than any of the project's own six measures, at
    0.31 ms a frame.  See docs/CALIBRATION.md.

    Unlike the baseline implementation in ``tools/baselines.py`` this one
    carries the library's contrast normalisation, so that a pure exposure
    change does not move it.
    """

    name = "gradient_variance"

    def __init__(self, contrast_normalize: bool = True, ksize: int = 3) -> None:
        super().__init__(contrast_normalize)
        self.ksize = ksize

    def _compute(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=self.ksize)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=self.ksize)
        magnitude = cv2.magnitude(gx, gy)
        value = float(magnitude.var())
        return value / normalization_factor(gray, self.contrast_normalize)
