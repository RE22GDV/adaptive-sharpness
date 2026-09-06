"""The public entry point: :class:`SharpnessEvaluator`.

Wires the stages together::

    frame -> Preprocessor -> ImageAnalyzer -> metrics -> NormalizerBank
          -> AdaptiveEnsemble -> TemporalFilter -> SharpnessResult

An evaluator instance owns per-stream state (the normaliser histories, the
previous frame used for motion estimation, the temporal filter).  It is
therefore **not thread-safe** and must not be shared between concurrent
streams; construct one per stream and call :meth:`reset` on a discontinuity.
"""
from __future__ import annotations

import logging
import time
from typing import Mapping, Sequence

import numpy as np

from .analysis import ImageAnalyzer
from .config import SharpnessConfig
from .ensemble import AdaptiveEnsemble
from .metrics import build_metrics
from .normalize import NormalizerBank
from .preprocess import Preprocessor
from .temporal import TemporalFilter
from .types import Frame, ImageStats, MetricSample, SharpnessResult, ROI

logger = logging.getLogger(__name__)

__all__ = ["SharpnessEvaluator", "RegionEvaluator"]


class SharpnessEvaluator:
    """Evaluates the sharpness of successive frames from one stream."""

    def __init__(self, config: SharpnessConfig | None = None) -> None:
        self.config = config or SharpnessConfig()
        self._preprocessor = Preprocessor(self.config.pipeline)
        self._analyzer = ImageAnalyzer(self.config.analysis)
        self._metrics = build_metrics(self.config.metrics)
        names = tuple(m.name for m in self._metrics)
        self._normalizers = NormalizerBank(names, self.config.normalization)
        self._ensemble = AdaptiveEnsemble(names, self.config.ensemble, self.config.metrics.priors)
        self._filter = TemporalFilter(self.config.temporal)
        self._frame_counter = 0
        logger.debug("evaluator ready with metrics: %s", ", ".join(names))

    # -- introspection ---------------------------------------------------------

    @property
    def metric_names(self) -> tuple[str, ...]:
        return tuple(m.name for m in self._metrics)

    @property
    def warmed_up(self) -> bool:
        """True once the normalisers have a meaningful scale."""
        return self._normalizers.warmed_up

    @property
    def normalizers(self) -> NormalizerBank:
        """Exposed so offline tools can freeze or refit the scaling."""
        return self._normalizers

    def reset(self) -> None:
        """Clear all per-stream state."""
        self._normalizers.reset()
        self._analyzer.reset()
        self._filter.reset()
        self._frame_counter = 0
        logger.debug("evaluator state reset")

    # -- evaluation ------------------------------------------------------------

    def evaluate(
        self,
        frame: Frame | np.ndarray,
        roi: ROI | None = None,
        motor_position: float | None = None,
    ) -> SharpnessResult:
        """Evaluate one frame.

        ``frame`` may be a :class:`~sharpness.types.Frame` or a bare NumPy
        array.  ``roi`` is given in *source frame* pixel coordinates; passing
        ``None`` evaluates the whole frame (subject to
        ``pipeline.default_roi_fraction``).  ``motor_position`` is carried
        through to the result untouched, for the benefit of the data collector.
        """
        started = time.perf_counter()

        if isinstance(frame, Frame):
            data = frame.data
            index = frame.index
            timestamp = frame.timestamp
            capture_latency = frame.capture_latency_s
        else:
            data = frame
            index = self._frame_counter
            timestamp = started
            capture_latency = 0.0
        self._frame_counter += 1

        image = self._preprocessor.prepare(data, roi)
        gray = image.gray

        stats = self._analyzer.analyze(gray)
        degradations = self._analyzer.degradations(stats)

        raw: dict[str, float] = {}
        durations: dict[str, float] = {}
        for metric in self._metrics:
            t0 = time.perf_counter()
            raw[metric.name] = metric.compute(gray)
            durations[metric.name] = time.perf_counter() - t0

        normalized = self._normalizers.normalize(raw)
        warmed_up = self._normalizers.warmed_up
        output = self._ensemble.combine(
            normalized, stats, degradations, warmed_up,
            self._normalizers.informative_fraction,
        )
        filtered = self._filter.update(output.score, output.confidence)

        samples = tuple(
            MetricSample(
                name=metric.name,
                raw=raw[metric.name],
                normalized=normalized[metric.name],
                weight=output.weights[metric.name],
                reliability=output.reliabilities[metric.name],
                agreement=output.agreements[metric.name],
                compute_time_s=durations[metric.name],
            )
            for metric in self._metrics
        )

        informative = self._normalizers.informative_fraction
        return SharpnessResult(
            instantaneous_score=output.score,
            filtered_score=filtered.value,
            confidence=output.confidence,
            metrics=samples,
            stats=stats,
            frame_index=index,
            timestamp=timestamp,
            processing_time_s=time.perf_counter() - started,
            capture_latency_s=capture_latency,
            score_change_detected=filtered.score_change_detected,
            roi=image.roi,
            motor_position=motor_position,
            # Carried on every result so a host application does not have to
            # poll the evaluator separately and keep the two in sync.
            ready=warmed_up and informative > 0.0,
            informative_fraction=informative,
            warmup_samples=self._normalizers.sample_count,
        )

    def evaluate_raw_only(
        self, frame: Frame | np.ndarray, roi: ROI | None = None
    ) -> tuple[dict[str, float], ImageStats]:
        """Compute the raw metrics and stats without touching any filter state.

        Useful for offline dataset building, where the normalisation has to be
        fitted over the whole recording rather than incrementally.
        """
        data = frame.data if isinstance(frame, Frame) else frame
        image = self._preprocessor.prepare(data, roi)
        stats = self._analyzer.analyze(image.gray)
        raw = {m.name: m.compute(image.gray) for m in self._metrics}
        return raw, stats


class RegionEvaluator:
    """Evaluates the whole frame and a ROI side by side.

    Each region needs its own normalisation history and temporal filter (their
    raw metric ranges differ), so this holds two independent evaluators and
    keeps their configuration in sync.
    """

    def __init__(self, config: SharpnessConfig | None = None) -> None:
        self.config = config or SharpnessConfig()
        self.full = SharpnessEvaluator(self.config)
        self.region = SharpnessEvaluator(self.config)

    @property
    def metric_names(self) -> tuple[str, ...]:
        return self.full.metric_names

    def reset(self) -> None:
        self.full.reset()
        self.region.reset()

    def evaluate(
        self,
        frame: Frame | np.ndarray,
        roi: ROI | None = None,
        motor_position: float | None = None,
    ) -> tuple[SharpnessResult, SharpnessResult | None]:
        """Return ``(whole_frame_result, roi_result)``.

        ``roi_result`` is ``None`` when no ROI was supplied.
        """
        full_result = self.full.evaluate(frame, None, motor_position)
        if roi is None:
            return full_result, None
        roi_result = self.region.evaluate(frame, roi, motor_position)
        return full_result, roi_result
