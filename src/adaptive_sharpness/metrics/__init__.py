"""Metric registry and factory.

Adding a new metric means writing a :class:`~sharpness.metrics.base.SharpnessMetric`
subclass and registering its builder in :data:`METRIC_BUILDERS`; the evaluator,
the ensemble and the tooling pick it up without further changes.
"""
from __future__ import annotations

import logging
from typing import Callable, Mapping, Sequence

from ..config import MetricsConfig
from .base import EPS, MetricError, SharpnessMetric, normalization_factor
from .edge import EdgeWidth
from .frequency import FourierHighFrequency, WaveletEnergy, haar_decompose
from .gradient import Brenner, GradientVariance, LaplacianVariance, Tenengrad

logger = logging.getLogger(__name__)

__all__ = [
    "SharpnessMetric",
    "MetricError",
    "EPS",
    "normalization_factor",
    "haar_decompose",
    "LaplacianVariance",
    "Tenengrad",
    "GradientVariance",
    "Brenner",
    "WaveletEnergy",
    "FourierHighFrequency",
    "EdgeWidth",
    "METRIC_BUILDERS",
    "build_metrics",
    "available_metrics",
]

#: Maps a metric name to a factory taking the metrics configuration.
METRIC_BUILDERS: Mapping[str, Callable[[MetricsConfig], SharpnessMetric]] = {
    "laplacian": lambda c: LaplacianVariance(
        contrast_normalize=c.contrast_normalize,
    ),
    "tenengrad": lambda c: Tenengrad(
        contrast_normalize=c.contrast_normalize,
    ),
    "brenner": lambda c: Brenner(
        contrast_normalize=c.contrast_normalize,
        shift=c.brenner_shift,
    ),
    "wavelet": lambda c: WaveletEnergy(
        contrast_normalize=c.contrast_normalize,
        levels=c.wavelet_levels,
    ),
    "fourier": lambda c: FourierHighFrequency(
        contrast_normalize=c.contrast_normalize,
        high_cutoff=c.fourier_high_cutoff,
    ),
    "gradient_variance": lambda c: GradientVariance(
        contrast_normalize=c.contrast_normalize,
    ),
    "edge_width": lambda c: EdgeWidth(
        contrast_normalize=c.contrast_normalize,
        canny_percentile=c.edge_canny_percentile,
        canny_ratio=c.edge_canny_ratio,
        min_gradient=c.edge_min_gradient,
        range_window=c.edge_range_window,
    ),
}


def available_metrics() -> tuple[str, ...]:
    """Names of every metric this build knows how to construct."""
    return tuple(METRIC_BUILDERS)


def build_metrics(config: MetricsConfig) -> tuple[SharpnessMetric, ...]:
    """Instantiate the metrics listed in ``config.enabled``.

    Raises ``KeyError`` for an unknown name so that a configuration typo fails
    at construction time rather than producing a silently smaller ensemble.
    """
    if not config.enabled:
        raise ValueError("at least one metric must be enabled")
    unknown = [name for name in config.enabled if name not in METRIC_BUILDERS]
    if unknown:
        raise KeyError(
            f"unknown metric(s): {', '.join(unknown)}. "
            f"Available: {', '.join(available_metrics())}"
        )
    metrics = tuple(METRIC_BUILDERS[name](config) for name in config.enabled)
    logger.debug("built %d metrics: %s", len(metrics), [m.name for m in metrics])
    return metrics
