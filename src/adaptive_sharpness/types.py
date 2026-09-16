"""Core data types shared across the sharpness evaluation pipeline.

All types are plain dataclasses so that they can be serialised to CSV/JSON by
the data-collection tools without any extra dependency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ROI",
    "Frame",
    "ImageStats",
    "MetricSample",
    "SharpnessResult",
]


@dataclass(frozen=True)
class ROI:
    """Rectangular region of interest in *source frame* pixel coordinates.

    The ROI is expressed against the full-resolution frame; the preprocessing
    stage rescales it internally when the frame is downscaled for metric
    computation, so callers never have to think about the analysis scale.
    """

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                f"ROI must have positive size, got {self.width}x{self.height}"
            )
        if self.x < 0 or self.y < 0:
            raise ValueError(f"ROI origin must be non-negative, got ({self.x}, {self.y})")

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def clipped_to(self, shape: tuple[int, ...]) -> "ROI":
        """Clip the ROI to an ``(height, width)`` image shape."""
        h, w = int(shape[0]), int(shape[1])
        x = min(max(0, self.x), max(0, w - 1))
        y = min(max(0, self.y), max(0, h - 1))
        x2 = min(self.x2, w)
        y2 = min(self.y2, h)
        if x2 <= x or y2 <= y:
            raise ValueError(f"ROI {self} does not intersect image of shape {shape}")
        return ROI(x, y, x2 - x, y2 - y)

    def scaled(self, factor: float) -> "ROI":
        """Scale the ROI by ``factor`` (used when the frame is downscaled)."""
        if factor <= 0:
            raise ValueError("scale factor must be positive")
        return ROI(
            x=int(round(self.x * factor)),
            y=int(round(self.y * factor)),
            width=max(1, int(round(self.width * factor))),
            height=max(1, int(round(self.height * factor))),
        )

    @classmethod
    def centered(cls, shape: tuple[int, ...], fraction: float = 0.5) -> "ROI":
        """A centred ROI covering ``fraction`` of each image dimension."""
        if not 0.0 < fraction <= 1.0:
            raise ValueError("fraction must lie in (0, 1]")
        h, w = int(shape[0]), int(shape[1])
        rw = max(1, int(round(w * fraction)))
        rh = max(1, int(round(h * fraction)))
        return cls((w - rw) // 2, (h - rh) // 2, rw, rh)


@dataclass(frozen=True)
class Frame:
    """One captured frame plus the timing information needed for latency stats.

    ``data`` is kept as the original array delivered by the capture backend; the
    pipeline never mutates it and only copies when a backend cannot guarantee
    buffer ownership.
    """

    data: np.ndarray
    index: int = 0
    # Instant (perf_counter domain) at which the backend *delivered* this
    # frame, i.e. when read() finished.  Anchoring on delivery rather than on
    # the start of the grab is what makes "frame age" measure queuing and
    # processing delay only, instead of also charging the unavoidable wait for
    # the camera to produce the frame.
    timestamp: float = field(default_factory=time.perf_counter)
    # Seconds spent inside the capture backend for this frame.
    capture_latency_s: float = 0.0
    source: str = "unknown"
    meta: Mapping[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def is_color(self) -> bool:
        return self.data.ndim == 3 and self.data.shape[2] >= 3


@dataclass(frozen=True)
class ImageStats:
    """Content descriptors that drive the adaptive weighting.

    Every field is normalised to a documented range so that the weighting model
    stays interpretable and unit-free.
    """

    # Robust noise sigma estimate in [0, 255] intensity units.
    noise_sigma: float = 0.0
    # Noise level mapped to [0, 1] through the configured reference sigma.
    noise_level: float = 0.0
    # Signal-to-noise ratio proxy (local contrast / noise sigma), unbounded >= 0.
    snr: float = 0.0
    # RMS local contrast in [0, 1] (std of the intensity, /255).
    local_contrast: float = 0.0
    # Mean intensity in [0, 1].
    brightness: float = 0.0
    # Fraction of pixels at/near the top of the range, in [0, 1].
    clipped_high: float = 0.0
    # Fraction of pixels at/near the bottom of the range, in [0, 1].
    clipped_low: float = 0.0
    # Fraction of pixels belonging to significant edges, in [0, 1].
    edge_density: float = 0.0
    # Edge sufficiency score in [0, 1]: 1 = plenty of structure to focus on.
    edge_sufficiency: float = 0.0
    # Estimated inter-frame motion in pixels (0 when no previous frame).
    motion_px: float = 0.0
    # Motion mapped to [0, 1] through the configured reference displacement.
    motion_level: float = 0.0
    # Directional anisotropy of the gradient field in [0, 1].
    anisotropy: float = 0.0

    @property
    def clipped_total(self) -> float:
        return min(1.0, self.clipped_high + self.clipped_low)

    def as_dict(self) -> dict[str, float]:
        return {
            "noise_sigma": self.noise_sigma,
            "noise_level": self.noise_level,
            "snr": self.snr,
            "local_contrast": self.local_contrast,
            "brightness": self.brightness,
            "clipped_high": self.clipped_high,
            "clipped_low": self.clipped_low,
            "edge_density": self.edge_density,
            "edge_sufficiency": self.edge_sufficiency,
            "motion_px": self.motion_px,
            "motion_level": self.motion_level,
            "anisotropy": self.anisotropy,
        }


@dataclass(frozen=True)
class MetricSample:
    """A single metric evaluated on a single frame."""

    name: str
    raw: float
    normalized: float
    weight: float
    reliability: float
    agreement: float
    compute_time_s: float = 0.0


@dataclass(frozen=True)
class SharpnessResult:
    """Full outcome of evaluating one frame.

    Score semantics
    ---------------
    **Both scores are relative, not absolute.**  They are normalised against a
    rolling history held by one evaluator instance, so a value of 0.8 means
    "near the top of what this evaluator has seen recently on this stream", not
    "80% sharp".  Scores are **not comparable** across different scenes, ROIs,
    configurations, metric selections or evaluator instances, and they depend on
    the order frames arrive in.

    Two scores are reported because they answer different questions:

    ``instantaneous_score``
        The ensemble output for *this frame alone*.  Confidence never affects
        it.  Use it when you want the raw per-frame measurement.
    ``filtered_score``
        ``instantaneous_score`` after the temporal filter.  When
        ``temporal.confidence_coupling`` is enabled the confidence changes the
        filter's gain, so **confidence does affect this value**.  Use it for a
        control loop that wants the jitter suppressed.
    """

    # Ensemble score for this frame alone, before temporal filtering, in [0, 1].
    # Not affected by the confidence.
    instantaneous_score: float
    # Score after temporal filtering, in [0, 1].  Affected by the confidence
    # when confidence coupling is enabled.
    filtered_score: float
    # Heuristic indicator of whether the frame carried enough information to
    # measure focus at all, in [0, 1].  NOT a calibrated probability.
    confidence: float
    metrics: Sequence[MetricSample] = field(default_factory=tuple)
    stats: ImageStats = field(default_factory=ImageStats)
    frame_index: int = 0
    timestamp: float = 0.0
    # Seconds spent in the evaluator (excludes capture).
    processing_time_s: float = 0.0
    capture_latency_s: float = 0.0
    # The temporal filter saw a large jump in the score.  A focus change is one
    # cause; subject motion, a ROI change, an exposure change or a new object
    # entering the frame are others.
    score_change_detected: bool = False
    roi: ROI | None = None
    # Externally supplied focus actuator position, forwarded to the logs.
    motor_position: float | None = None
    # True once the normalisers have a usable observed range.  Before that the
    # scores are placeholders and the confidence is deliberately low.
    ready: bool = False
    # Fraction of metrics whose normalisation is currently meaningful, in
    # [0, 1].  0 means every metric is pinned at its neutral value because
    # nothing in the stream has varied.
    informative_fraction: float = 0.0
    # How many frames the normalisers have observed.
    warmup_samples: int = 0
    #: True once every metric is on a frozen scale.  While it is False the
    #: score is still relative to a history that keeps moving, so readings
    #: taken far apart are not strictly comparable.
    scale_frozen: bool = False

    @property
    def score(self) -> float:
        """Deprecated alias for :attr:`filtered_score`.

        Kept so existing callers keep working; the name was ambiguous about
        which of the two scores it meant.
        """
        return self.filtered_score

    @property
    def focus_change_detected(self) -> bool:
        """Deprecated alias for :attr:`score_change_detected`.

        Renamed because the filter detects a change in the *score*, which a
        focus change is only one possible cause of.
        """
        return self.score_change_detected

    @property
    def weights(self) -> dict[str, float]:
        return {m.name: m.weight for m in self.metrics}

    @property
    def raw_metrics(self) -> dict[str, float]:
        return {m.name: m.raw for m in self.metrics}

    @property
    def normalized_metrics(self) -> dict[str, float]:
        return {m.name: m.normalized for m in self.metrics}

    def as_row(self) -> dict[str, Any]:
        """Flat, CSV-friendly representation of everything computed.

        Nothing the evaluator worked out is dropped: the agreement term and the
        per-metric timings used to be computed and then discarded here, which
        made a recorded run impossible to analyse afterwards without re-running
        it.
        """
        row: dict[str, Any] = {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "instantaneous_score": self.instantaneous_score,
            "filtered_score": self.filtered_score,
            "confidence": self.confidence,
            "ready": int(self.ready),
            "informative_fraction": self.informative_fraction,
            "warmup_samples": self.warmup_samples,
            "scale_frozen": int(self.scale_frozen),
            "processing_time_s": self.processing_time_s,
            "capture_latency_s": self.capture_latency_s,
            "score_change_detected": int(self.score_change_detected),
            "motor_position": "" if self.motor_position is None else self.motor_position,
            "roi_x": "" if self.roi is None else self.roi.x,
            "roi_y": "" if self.roi is None else self.roi.y,
            "roi_w": "" if self.roi is None else self.roi.width,
            "roi_h": "" if self.roi is None else self.roi.height,
        }
        for m in self.metrics:
            row[f"raw_{m.name}"] = m.raw
            row[f"norm_{m.name}"] = m.normalized
            row[f"w_{m.name}"] = m.weight
            row[f"rel_{m.name}"] = m.reliability
            row[f"agree_{m.name}"] = m.agreement
            row[f"ms_{m.name}"] = m.compute_time_s * 1e3
        row.update(self.stats.as_dict())
        return row

    def to_dict(self) -> dict[str, Any]:
        """Nested representation suitable for JSON, keeping the structure."""
        return {
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "scores": {
                "instantaneous": self.instantaneous_score,
                "filtered": self.filtered_score,
            },
            "confidence": self.confidence,
            "state": {
                "ready": self.ready,
                "informative_fraction": self.informative_fraction,
                "warmup_samples": self.warmup_samples,
                "scale_frozen": self.scale_frozen,
                "score_change_detected": self.score_change_detected,
            },
            "timing": {
                "processing_s": self.processing_time_s,
                "capture_latency_s": self.capture_latency_s,
            },
            "roi": None if self.roi is None else {
                "x": self.roi.x, "y": self.roi.y,
                "width": self.roi.width, "height": self.roi.height,
            },
            "motor_position": self.motor_position,
            "metrics": {
                m.name: {
                    "raw": m.raw,
                    "normalized": m.normalized,
                    "weight": m.weight,
                    "reliability": m.reliability,
                    "agreement": m.agreement,
                    "compute_time_s": m.compute_time_s,
                }
                for m in self.metrics
            },
            "stats": self.stats.as_dict(),
        }

    def to_json(self, **kwargs: Any) -> str:
        """JSON form of :meth:`to_dict`."""
        import json

        return json.dumps(self.to_dict(), **kwargs)
