"""Adaptive ensemble sharpness evaluation for autofocus systems.

Typical library use::

    from adaptive_sharpness import SharpnessEvaluator, ROI, load_config

    evaluator = SharpnessEvaluator(load_config("config/default.toml"))
    result = evaluator.evaluate(bgr_frame, roi=ROI(320, 180, 640, 360))
    print(result.score, result.confidence, result.weights)

The package core depends only on NumPy and OpenCV.  Capture backends, the demo
UI and the experiment tooling live in separate packages so that importing the
library never pulls in a camera driver or a plotting stack.
"""
from __future__ import annotations

from .analysis import ImageAnalyzer
from .config import (
    METRIC_NAMES,
    AnalysisConfig,
    CaptureConfig,
    EnsembleConfig,
    FocusMapConfig,
    MetricsConfig,
    NormalizationConfig,
    PipelineConfig,
    RegionsConfig,
    SharpnessConfig,
    TemporalConfig,
    default_config_path,
    load_config,
    load_default_config,
)
from .ensemble import AdaptiveEnsemble, EnsembleOutput
from .focusmap import FocusMap, FocusMapper
from .evaluator import RegionEvaluator, SharpnessEvaluator
from .metrics import available_metrics, build_metrics
from .normalize import NormalizerBank, RunningNormalizer
from .regions import Region, RegionProposer
from .scene import SceneEvaluator, SceneResult
from .preprocess import AnalysisImage, Preprocessor
from .recording import RunRecorder
from .temporal import TemporalFilter
from .types import ROI, Frame, ImageStats, MetricSample, SharpnessResult

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # main entry points
    "SharpnessEvaluator",
    "RegionEvaluator",
    "SceneEvaluator",
    "SceneResult",
    "FocusMap",
    "FocusMapper",
    "Region",
    "RegionProposer",
    # data types
    "ROI",
    "Frame",
    "ImageStats",
    "MetricSample",
    "SharpnessResult",
    # configuration
    "SharpnessConfig",
    "PipelineConfig",
    "MetricsConfig",
    "NormalizationConfig",
    "AnalysisConfig",
    "EnsembleConfig",
    "TemporalConfig",
    "CaptureConfig",
    "FocusMapConfig",
    "RegionsConfig",
    "load_config",
    "load_default_config",
    "default_config_path",
    "METRIC_NAMES",
    # components
    "Preprocessor",
    "AnalysisImage",
    "ImageAnalyzer",
    "AdaptiveEnsemble",
    "EnsembleOutput",
    "RunRecorder",
    "NormalizerBank",
    "RunningNormalizer",
    "TemporalFilter",
    "available_metrics",
    "build_metrics",
]
