"""Frequency-domain sharpness metrics.

Defocus is a low-pass operation, so both of these metrics ask the same question
in different bases: how much energy survives at high spatial frequencies?

* :class:`WaveletEnergy` uses a Haar decomposition, which localises the answer
  in space as well as frequency and so tolerates a scene where only part of the
  field is textured.
* :class:`FourierHighFrequency` uses a global FFT and reports the *fraction* of
  spectral energy above a cut-off radius, which makes it naturally invariant to
  overall contrast.

The Haar transform is implemented directly with strided slicing rather than
pulling in PyWavelets: it is a handful of array operations, it avoids a
dependency that is not packaged for this platform, and it keeps the analysis
allocation-light on the Raspberry Pi.

References
----------
Yang & Nelson (2003), *Wavelet-based autofocusing and unsupervised
segmentation of microscopic images*.
Kautsky et al. (2002), *A new wavelet-based measure of image focus*.
"""
from __future__ import annotations

import cv2
import numpy as np

from .base import EPS, SharpnessMetric, normalization_factor

__all__ = ["WaveletEnergy", "FourierHighFrequency", "haar_decompose"]

_SQRT_HALF = float(np.sqrt(0.5))


def haar_decompose(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One level of the orthonormal 2-D Haar transform.

    Returns ``(LL, LH, HL, HH)``.  Odd rows/columns are dropped, so each output
    has shape ``(floor(h/2), floor(w/2))``.  The transform is orthonormal, hence
    energy-preserving, which is what lets the detail energies be compared across
    levels.
    """
    h, w = image.shape
    view = image[: h - (h % 2), : w - (w % 2)]
    # Horizontal pass.
    even = view[:, 0::2]
    odd = view[:, 1::2]
    low = (even + odd) * _SQRT_HALF
    high = (even - odd) * _SQRT_HALF
    # Vertical pass.
    ll = (low[0::2, :] + low[1::2, :]) * _SQRT_HALF
    lh = (low[0::2, :] - low[1::2, :]) * _SQRT_HALF
    hl = (high[0::2, :] + high[1::2, :]) * _SQRT_HALF
    hh = (high[0::2, :] - high[1::2, :]) * _SQRT_HALF
    return ll, lh, hl, hh


class WaveletEnergy(SharpnessMetric):
    """Mean energy of the Haar detail sub-bands, summed over levels."""

    name = "wavelet"

    def __init__(self, contrast_normalize: bool = True, levels: int = 2) -> None:
        super().__init__(contrast_normalize)
        if levels < 1:
            raise ValueError("wavelet levels must be >= 1")
        self.levels = levels
        self.min_size = max(8, 2 ** (levels + 1))

    def _compute(self, gray: np.ndarray) -> float:
        current = gray
        total = 0.0
        # The orthonormal Haar LL band has a DC gain of 2 per level, so its
        # energy grows by 4 per level.  Without dividing that out, the deeper
        # levels are inflated and the metric stops being monotone in the blur:
        # level-1 detail collapses under blur while the (4x amplified) level-2
        # detail does not, so the sum can *increase* with defocus.
        level_gain = 1.0
        for _ in range(self.levels):
            if min(current.shape) < 2:
                break
            ll, lh, hl, hh = haar_decompose(current)
            energy = float(np.mean(lh * lh) + np.mean(hl * hl) + np.mean(hh * hh))
            total += energy / level_gain
            current = ll
            level_gain *= 4.0
        return total / normalization_factor(gray, self.contrast_normalize)


class FourierHighFrequency(SharpnessMetric):
    """Fraction of spectral energy above a cut-off radius.

    A Hann window is applied first.  Without it the implicit discontinuity at
    the image border injects broadband energy that swamps the focus signal — an
    error that makes the metric look almost focus-independent.

    Because the result is a ratio of energies it is already invariant to
    contrast scaling, so ``contrast_normalize`` does not apply to this metric.
    """

    name = "fourier"

    def __init__(self, contrast_normalize: bool = True, high_cutoff: float = 0.25) -> None:
        super().__init__(contrast_normalize)
        if not 0.0 < high_cutoff < 1.0:
            raise ValueError("fourier high_cutoff must lie in (0, 1)")
        self.high_cutoff = high_cutoff
        self.min_size = 16
        # Shape-keyed caches: building the window and the radial mask on every
        # frame would cost more than the FFT itself.
        self._window: np.ndarray | None = None
        self._mask: np.ndarray | None = None
        self._cached_shape: tuple[int, int] | None = None

    def _ensure_cache(self, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        if self._cached_shape == shape and self._window is not None and self._mask is not None:
            return self._window, self._mask
        h, w = shape
        win_y = np.hanning(h).astype(np.float32)
        win_x = np.hanning(w).astype(np.float32)
        window = np.outer(win_y, win_x).astype(np.float32)

        # rfft2 keeps only the non-redundant half of the horizontal axis.
        fy = np.fft.fftfreq(h).astype(np.float32)[:, None]
        fx = np.fft.rfftfreq(w).astype(np.float32)[None, :]
        # Normalised radius where 1.0 is the Nyquist frequency.
        radius = np.sqrt((fy * 2.0) ** 2 + (fx * 2.0) ** 2)
        mask = (radius >= self.high_cutoff).astype(np.float32)
        # Never count the DC term as signal.
        mask[0, 0] = 0.0

        self._window, self._mask, self._cached_shape = window, mask, shape
        return window, mask

    def _compute(self, gray: np.ndarray) -> float:
        window, mask = self._ensure_cache(gray.shape)
        # Remove the mean before windowing so DC leakage does not dominate.
        windowed = (gray - float(gray.mean())) * window
        spectrum = np.fft.rfft2(windowed)
        power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
        total = float(power.sum())
        if total <= EPS:
            return 0.0
        high = float((power * mask).sum())
        return high / total
