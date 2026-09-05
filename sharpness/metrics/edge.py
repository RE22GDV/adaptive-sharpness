"""Edge-profile sharpness metric.

Unlike the energy metrics, this one estimates a physically meaningful quantity:
the average *width* of the intensity transitions in the image, in pixels.

For a step edge of amplitude ``A`` convolved with a Gaussian of width ``sigma``,
the peak gradient is ``A / (sigma * sqrt(2*pi))``.  Therefore

    sigma * sqrt(2*pi) = A / max|grad I|

so dividing the local intensity range by the local gradient magnitude gives a
per-edge width estimate.  The metric reports the reciprocal of the median width,
so that — like every other metric here — larger means sharper.

Two properties matter for the ensemble:

* the result is a ratio of intensities, so it is inherently invariant to
  contrast scaling (``contrast_normalize`` does not apply);
* it is meaningless when the frame contains no edges, which is why the
  weighting model gives it by far the largest edge-deficiency sensitivity.

References
----------
Marziliano et al. (2002), *A no-reference perceptual blur metric*.
Ferzli & Karam (2009), *A no-reference objective image sharpness metric based
on the notion of Just Noticeable Blur (JNB)*.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import EPS, SharpnessMetric

__all__ = ["EdgeWidth"]


class EdgeWidth(SharpnessMetric):
    """Reciprocal of the median edge width, in inverse pixels."""

    name = "edge_width"

    #: Sobel's 3x3 kernel approximates 8 * d/dx, so its output must be divided
    #: by 8 to become an intensity-per-pixel derivative.  Skipping this makes
    #: every width estimate 8x too small.
    _SOBEL_SCALE = 1.0 / 8.0

    def __init__(
        self,
        contrast_normalize: bool = True,
        canny_percentile: float = 97.0,
        canny_ratio: float = 0.4,
        min_gradient: float = 1.0,
        max_width_px: float = 32.0,
        min_edge_pixels: int = 24,
        range_window: int = 11,
    ) -> None:
        super().__init__(contrast_normalize)
        if not 50.0 <= canny_percentile < 100.0:
            raise ValueError("canny_percentile must lie in [50, 100)")
        if not 0.0 < canny_ratio < 1.0:
            raise ValueError("canny_ratio must lie in (0, 1)")
        self.canny_percentile = canny_percentile
        self.canny_ratio = canny_ratio
        self.min_gradient = min_gradient
        self.max_width_px = max_width_px
        self.min_edge_pixels = min_edge_pixels
        if range_window < 3 or range_window % 2 == 0:
            raise ValueError("range_window must be an odd number >= 3")
        # The window must span the whole transition, otherwise the measured
        # range shrinks in step with the gradient and their ratio - the width -
        # comes out constant no matter how defocused the image is.  A window of
        # w pixels can measure widths up to roughly w - 2.
        self.range_window = range_window
        self.min_size = max(16, range_window * 2)
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (range_window, range_window)
        )

    def _compute(self, gray: np.ndarray) -> float:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=self._SOBEL_SCALE)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=self._SOBEL_SCALE)
        magnitude = cv2.magnitude(gx, gy)

        # Fixed Canny thresholds make this metric non-monotone: as defocus
        # grows, fewer and fewer edges clear the threshold, and the survivors
        # are the highest-contrast (apparently sharpest) ones, so the median
        # width can *fall* while the image blurs.  Measured on a defocus ramp,
        # fixed thresholds were non-monotone at every window size tested, while
        # percentile-adaptive thresholds were monotone at all of them and gave
        # a larger dynamic range.  Anchoring the thresholds to a percentile of
        # this frame's own gradient distribution keeps the edge population
        # roughly constant instead.
        high = float(np.percentile(magnitude, self.canny_percentile))
        if high <= self.min_gradient:
            return 0.0
        # Canny's internal Sobel is unscaled, so undo _SOBEL_SCALE for it.
        high_thr = int(high / self._SOBEL_SCALE)
        low_thr = int(high_thr * self.canny_ratio)

        u8 = np.clip(gray, 0.0, 255.0).astype(np.uint8)
        edges = cv2.Canny(u8, low_thr, high_thr, L2gradient=True)
        edge_mask = edges > 0
        edge_count = int(np.count_nonzero(edge_mask))
        if edge_count < self.min_edge_pixels:
            # Not enough structure to measure a width at all.  Returning 0 is
            # honest; the ensemble's reliability term will also collapse this
            # metric's weight because edge_sufficiency will be low.
            return 0.0

        # Local intensity range = morphological gradient over a window wide
        # enough to contain the full edge transition.
        local_range = cv2.dilate(gray, self._kernel) - cv2.erode(gray, self._kernel)

        mag_edges = magnitude[edge_mask]
        range_edges = local_range[edge_mask]

        # Keep only edges with a gradient strong enough for the width estimate
        # to be numerically meaningful.
        keep = mag_edges >= self.min_gradient
        if int(np.count_nonzero(keep)) < self.min_edge_pixels:
            return 0.0
        mag_edges = mag_edges[keep]
        range_edges = range_edges[keep]

        widths = range_edges / (mag_edges + EPS)
        np.clip(widths, 0.0, self.max_width_px, out=widths)
        median_width = float(np.median(widths))
        if median_width <= EPS:
            return 0.0
        return 1.0 / median_width
