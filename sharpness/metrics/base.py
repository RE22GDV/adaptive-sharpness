"""Common interface for the individual sharpness metrics.

Every metric receives the same input — a single-channel ``float32`` array whose
values lie in ``[0, 255]`` — and returns a single scalar where **larger means
sharper**.  Keeping the direction uniform lets the ensemble treat all metrics
identically.

Contrast normalisation
----------------------
Textbook definitions of the gradient/energy metrics are not invariant to
illumination scaling: doubling the exposure quadruples a squared-gradient sum
even though the focus has not changed.  Because the evaluation criteria for this
project explicitly include robustness to exposure changes, the energy-type
metrics are divided by ``mean(I)**2`` when ``contrast_normalize`` is enabled
(the default).  This makes them invariant to a multiplicative illumination
change while leaving their focus response unchanged.  Set the flag to ``False``
to recover the classical textbook forms.
"""
from __future__ import annotations

import abc
from typing import Final

import numpy as np

__all__ = ["SharpnessMetric", "MetricError", "normalization_factor", "EPS"]

EPS: Final[float] = 1e-9


class MetricError(RuntimeError):
    """Raised when a metric cannot be evaluated on the given input."""


def normalization_factor(gray: np.ndarray, enabled: bool) -> float:
    """Return the divisor implementing illumination-scale invariance.

    Returns ``mean(I)**2`` when enabled (guarded against a black frame), else
    ``1.0``.
    """
    if not enabled:
        return 1.0
    mean = float(gray.mean())
    return max(mean * mean, 1.0)


class SharpnessMetric(abc.ABC):
    """Base class for all sharpness metrics."""

    #: Stable identifier used in configuration, logs and CSV headers.
    name: str = "unnamed"

    #: Smallest image side the metric can handle.
    min_size: int = 8

    def __init__(self, contrast_normalize: bool = True) -> None:
        self.contrast_normalize = contrast_normalize

    def __call__(self, gray: np.ndarray) -> float:
        return self.compute(gray)

    def validate(self, gray: np.ndarray) -> None:
        """Check the input contract, raising :class:`MetricError` on violation."""
        if gray.ndim != 2:
            raise MetricError(
                f"{self.name}: expected a 2-D single-channel image, got shape {gray.shape}"
            )
        if gray.dtype != np.float32:
            raise MetricError(
                f"{self.name}: expected float32 input, got {gray.dtype}"
            )
        if min(gray.shape) < self.min_size:
            raise MetricError(
                f"{self.name}: image {gray.shape} is smaller than the minimum "
                f"side of {self.min_size} px"
            )

    def compute(self, gray: np.ndarray) -> float:
        """Evaluate the metric, returning a non-negative float."""
        self.validate(gray)
        value = self._compute(gray)
        if not np.isfinite(value):
            return 0.0
        return max(0.0, float(value))

    @abc.abstractmethod
    def _compute(self, gray: np.ndarray) -> float:
        """Metric-specific implementation, called with a validated image."""

    def __repr__(self) -> str:  # pragma: no cover - debugging helper
        return f"{type(self).__name__}(name={self.name!r})"
