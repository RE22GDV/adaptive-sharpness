"""Dense per-tile focus map: *where* in the frame is sharp.

The evaluator answers "how sharp is this region". To say which object is in
focus, the frame has to be scored everywhere at once, cheaply, and then compared
region against region.

The hard part is not speed, it is that **low gradient energy has two completely
different causes**: the region is defocused, or the region has no texture to
begin with. A blank wall and a badly defocused wall look almost identical to any
plain gradient measure. Reporting a featureless tile as "out of focus" would
make the whole display misleading, so every tile carries a validity flag and
only tiles with enough contrast are ranked.

To reduce the dependence on *how much* texture a region has, the tile measures
are all ratios. Three are implemented and selectable, because which one is best
is an empirical question rather than an obvious one:

``grad_over_var``
    ``mean(|grad I|^2) / var(I)`` - gradient energy per unit contrast.
``lap_over_var``
    ``mean(lap I ^2) / var(I)`` - the same with a second derivative, which has
    a steeper frequency response.
``lap_over_grad``
    ``mean(lap I ^2) / mean(|grad I|^2)`` - the ratio of second- to
    first-derivative energy. In the frequency domain this is a
    contrast-independent estimate of the characteristic spatial frequency of the
    region, so it is closest to a direct estimate of inverse blur radius, at the
    cost of being the most noise-sensitive of the three.

``tools/compare_focus_measures.py`` measures all three on controlled split-focus
scenes; see docs/ALGORITHM.md for the result that picked the default.

Every tile statistic is a block mean, computed with a single ``cv2.resize`` in
``INTER_AREA`` mode over the whole frame, so the cost is a handful of full-frame
passes regardless of the tile count.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

import cv2
import numpy as np

from .config import FocusMapConfig

logger = logging.getLogger(__name__)

__all__ = ["FocusMap", "FocusMapper", "TILE_MEASURES"]

TILE_MEASURES: Final[tuple[str, ...]] = (
    "grad_over_var",
    "lap_over_var",
    "lap_over_grad",
)

# Sobel's 3x3 kernel approximates 8 * d/dx.
_SOBEL_SCALE: Final[float] = 1.0 / 8.0
_EPS: Final[float] = 1e-6


@dataclass(frozen=True)
class FocusMap:
    """Per-tile focus estimate over one frame."""

    #: Raw tile measure, larger meaning sharper. Shape (rows, cols).
    raw: np.ndarray
    #: Tile measure rescaled to [0, 1] across the *valid* tiles of this frame.
    #: This is a within-frame comparison, which is exactly the question being
    #: asked - it does not claim anything about absolute sharpness.
    score: np.ndarray
    #: True where the tile has enough contrast for the measure to mean anything.
    valid: np.ndarray
    #: RMS contrast per tile, in [0, 1].
    contrast: np.ndarray
    #: Size of one tile in analysis-image pixels.
    tile_size: int
    #: Scale from analysis image to source frame coordinates.
    source_scale: float
    #: How much the raw tile measure actually varies across the frame, relative
    #: to its own median.  Near 0 means the frame is uniformly sharp (or
    #: uniformly soft) and there is genuinely nothing to choose between regions.
    relative_spread: float = 0.0
    #: How far the rescale was allowed to stretch, in [0, 1].  Below 1 the map
    #: has been deliberately pulled towards a neutral 0.5 because the underlying
    #: variation was too small to be meaningful.
    stretch: float = 1.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.raw.shape  # type: ignore[return-value]

    @property
    def valid_fraction(self) -> float:
        return float(np.count_nonzero(self.valid)) / float(max(1, self.valid.size))

    def best_cell(self) -> tuple[int, int] | None:
        """Row and column of the sharpest valid tile, or ``None`` if none is."""
        if not np.any(self.valid):
            return None
        masked = np.where(self.valid, self.score, -np.inf)
        index = int(np.argmax(masked))
        return divmod(index, self.raw.shape[1])

    def cell_bounds_source(self, row: int, col: int) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` of a tile in *source frame* pixels."""
        size = self.tile_size / max(self.source_scale, _EPS)
        return (
            int(round(col * size)),
            int(round(row * size)),
            max(1, int(round(size))),
            max(1, int(round(size))),
        )


class FocusMapper:
    """Computes a :class:`FocusMap` for successive analysis images."""

    def __init__(self, config: FocusMapConfig) -> None:
        if config.tile_size < 8:
            raise ValueError("tile_size must be at least 8 px")
        if config.measure not in TILE_MEASURES:
            raise ValueError(
                f"unknown tile measure {config.measure!r}; "
                f"choose from {', '.join(TILE_MEASURES)}"
            )
        self.config = config

    def _block_mean(self, image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
        """Mean of ``image`` over a (rows, cols) block grid."""
        rows, cols = shape
        return cv2.resize(image, (cols, rows), interpolation=cv2.INTER_AREA)

    def compute(self, gray: np.ndarray, source_scale: float = 1.0) -> FocusMap:
        """Build the focus map of one analysis image.

        ``source_scale`` is the analysis-to-source scale factor, used only so
        the caller can map tiles back to display coordinates.
        """
        if gray.ndim != 2:
            raise ValueError(f"expected a 2-D image, got shape {gray.shape}")
        if gray.dtype != np.float32:
            raise ValueError(f"expected float32 input, got {gray.dtype}")

        height, width = gray.shape
        tile = self.config.tile_size
        rows = max(1, height // tile)
        cols = max(1, width // tile)
        shape = (rows, cols)

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=_SOBEL_SCALE)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=_SOBEL_SCALE)
        grad_energy = cv2.magnitude(gx, gy)
        np.square(grad_energy, out=grad_energy)

        mean_i = self._block_mean(gray, shape)
        mean_i2 = self._block_mean(cv2.multiply(gray, gray), shape)
        # var = E[I^2] - E[I]^2, clipped because floating-point cancellation can
        # make it very slightly negative on a uniform tile.
        variance = np.maximum(mean_i2 - mean_i * mean_i, 0.0)
        mean_grad = self._block_mean(grad_energy, shape)

        measure = self.config.measure
        if measure == "grad_over_var":
            raw = mean_grad / (variance + _EPS)
        else:
            laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
            np.square(laplacian, out=laplacian)
            mean_lap = self._block_mean(laplacian, shape)
            if measure == "lap_over_var":
                raw = mean_lap / (variance + _EPS)
            else:  # lap_over_grad
                raw = mean_lap / (mean_grad + _EPS)

        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        contrast = np.sqrt(variance, dtype=np.float32) / 255.0
        valid = contrast >= self.config.min_contrast
        if self.config.min_gradient_energy > 0.0:
            valid &= mean_grad >= self.config.min_gradient_energy

        score, relative_spread, stretch = self._rescale(raw, valid)

        if self.config.smooth:
            # A light blur over the tile grid suppresses single-tile noise
            # without merging genuinely separate subjects.
            score = cv2.GaussianBlur(score, (3, 3), 0.8)
            score = np.where(valid, score, 0.0).astype(np.float32)

        return FocusMap(
            raw=raw,
            score=score,
            valid=valid,
            contrast=contrast.astype(np.float32),
            tile_size=tile,
            source_scale=source_scale,
            relative_spread=relative_spread,
            stretch=stretch,
        )

    def _rescale(self, raw: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, float, float]:
        """Map the valid tiles onto [0, 1] using robust within-frame anchors.

        A plain min-max rescale is dangerous here: on a frame that is uniformly
        sharp, it stretches whatever tiny numerical differences exist across the
        full range and turns measurement noise into a confident-looking winner.
        The stretch is therefore scaled by how much the raw measure actually
        varies *relative to its own median*, so a frame with nothing to choose
        between regions produces a map that stays near a neutral 0.5.

        Returns ``(score, relative_spread, stretch)``.
        """
        score = np.zeros_like(raw, dtype=np.float32)
        values = raw[valid]
        if values.size == 0:
            return score, 0.0, 0.0

        low = float(np.percentile(values, self.config.low_percentile))
        high = float(np.percentile(values, self.config.high_percentile))
        median = float(np.median(values))
        span = high - low
        if span <= _EPS:
            score[valid] = 0.5
            return score, 0.0, 0.0

        relative_spread = span / (abs(median) + _EPS)
        stretch = float(
            min(1.0, relative_spread / max(self.config.min_relative_spread, _EPS))
        )

        normalised = (raw - low) / span
        np.clip(normalised, 0.0, 1.0, out=normalised)
        # Pull towards 0.5 in proportion to how little real variation there is.
        normalised = 0.5 + (normalised - 0.5) * stretch
        score[valid] = normalised[valid]
        return score, relative_spread, stretch
