"""Configuration model for the sharpness evaluation pipeline.

The configuration is a tree of frozen dataclasses with documented defaults.  It
can be loaded from a TOML file (``tomllib`` is in the standard library since
Python 3.11, so this adds no dependency) and every field can also be overridden
programmatically when the package is used as a library.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "PipelineConfig",
    "MetricsConfig",
    "NormalizationConfig",
    "AnalysisConfig",
    "FocusMapConfig",
    "RegionsConfig",
    "EnsembleConfig",
    "TemporalConfig",
    "CaptureConfig",
    "SharpnessConfig",
    "load_config",
    "load_default_config",
    "default_config_path",
]

# Names of the six built-in metrics, in a stable order.
METRIC_NAMES: tuple[str, ...] = (
    "laplacian",
    "tenengrad",
    "brenner",
    "wavelet",
    "fourier",
    "edge_width",
)


@dataclass(frozen=True)
class PipelineConfig:
    """Preprocessing and scheduling options."""

    # Longest side of the image actually fed to the metrics.  The full frame is
    # only used for display; metrics run on this downscaled copy.
    analysis_width: int = 320
    # Downscale only, never upscale a small frame.
    allow_upscale: bool = False
    # Interpolation used for the analysis downscale (cv2 flag name).
    interpolation: str = "INTER_AREA"
    # Default ROI as a fraction of the frame when the caller supplies none.
    # 1.0 means "use the whole frame".
    default_roi_fraction: float = 1.0
    # Apply a light blur before the metrics to tame sensor noise.  Off by
    # default because it also removes part of the signal the metrics measure.
    prefilter_sigma: float = 0.0
    # Channel order of colour input: "BGR" (the OpenCV convention, and the
    # default) or "RGB".  Getting this wrong is silent - the frame still
    # converts to greyscale, just with the red and blue weights swapped - so it
    # is stated explicitly rather than guessed.
    color_order: str = "BGR"
    # Pixel value that represents white.  ``None`` infers it from the dtype:
    # 255 for uint8 and float, 65535 for uint16.  Set it explicitly for sensor
    # data that occupies only part of its container, e.g. 4095 for 12-bit
    # samples stored in uint16 - otherwise the frame is scaled as if it were
    # full-range and every brightness-dependent statistic is wrong.
    input_max: float | None = None


@dataclass(frozen=True)
class MetricsConfig:
    """Which metrics run, and their static prior weights."""

    enabled: tuple[str, ...] = METRIC_NAMES
    # Prior weights p_i.  They are multiplied by the adaptive terms and the
    # product is renormalised, so only the ratios matter.
    priors: Mapping[str, float] = field(
        default_factory=lambda: {
            "laplacian": 1.0,
            "tenengrad": 1.0,
            "brenner": 1.0,
            "wavelet": 1.0,
            "fourier": 0.8,
            "edge_width": 0.8,
            "gradient_variance": 1.0,
        }
    )
    # Divide the energy-type metrics by mean(I)**2 so that a pure exposure
    # change does not move the score.  See sharpness/metrics/base.py.
    contrast_normalize: bool = True
    # Fourier metric: cut-off of the "high frequency" band as a fraction of the
    # Nyquist radius.
    fourier_high_cutoff: float = 0.25
    # Wavelet metric: number of Haar decomposition levels.
    wavelet_levels: int = 2
    # Brenner metric: pixel displacement used by the difference operator.
    brenner_shift: int = 2
    # Edge-width metric: the Canny high threshold is taken at this percentile
    # of the frame's own gradient magnitude, and the low threshold is that
    # times edge_canny_ratio.  Fixed absolute thresholds were measured to make
    # the metric non-monotone in the defocus; see sharpness/metrics/edge.py.
    edge_canny_percentile: float = 97.0
    edge_canny_ratio: float = 0.4
    # Morphological window used to measure the edge amplitude.  It bounds the
    # largest measurable edge width (roughly window - 2 pixels).
    edge_range_window: int = 11
    # Minimum gradient magnitude (intensity levels per pixel, on the true
    # derivative scale) for the frame to carry a measurable edge at all.
    edge_min_gradient: float = 1.0


@dataclass(frozen=True)
class NormalizationConfig:
    """Running normalisation that maps heterogeneous raw metrics to [0, 1]."""

    # Length of the rolling history used to estimate the observed range.
    #
    # 38 seconds at 25 fps.  It used to be 120 frames, under five seconds,
    # which made the score relative to the immediate past: on a slow approach
    # to focus the window filled with high values and the sharpest frames of
    # all were scored as no better than recent history.  Measured against the
    # point-source ground truth, rank correlation rose monotonically with the
    # window - 0.77, 0.83, 0.88 at 120, 480 and 960 frames on one run - before
    # freezing was introduced on top.
    window: int = 960
    # Number of frames before the normaliser is considered warmed up.
    warmup: int = 5
    # Compress heavy-tailed metrics with log1p before scaling.
    log_compress: bool = True
    # How the compressed value is mapped onto [0, 1] between the anchors.
    #
    # "linear" is the textbook min-max map, clipped at both ends.  The clipping
    # is the problem: every frame beyond an anchor gets exactly the same score,
    # so the ordering that a focus search depends on is destroyed wherever the
    # anchors do not span the working range.  Measured on the point-source
    # recordings, a frozen linear map pins 41% of frames at 0 or 1.
    #
    # "logistic" maps the same anchors through 1/(1+exp(-gain*(x-mid)/span)).
    # The function is strictly monotone, so it introduces no ties of its own -
    # but the implementation is float64 and saturates numerically: past roughly
    # 9 spans from the midpoint the result rounds to exactly 1.0, so two very
    # different frames out in that tail still collide.  In practice the tails
    # are far outside the working range; the claim is "no ties over the range
    # the anchors describe", not "no ties ever".  Measured on the same
    # recordings: 0.3% of frames pinned against 40.7% for a frozen linear map.
    mapping: str = "logistic"
    # Logistic steepness.  At 4.0 the two anchors land on 0.12 and 0.88, so the
    # configured percentiles still occupy most of the output range while the
    # tails stay distinguishable.
    logistic_gain: float = 4.0
    # Robust percentiles used as the scaling anchors.
    low_percentile: float = 5.0
    high_percentile: float = 95.0
    # Guard against a degenerate (near-zero) observed range.
    min_range_ratio: float = 1e-3
    # Absolute term in the degenerate-range guard, which is
    #     floor = min_range_ratio * max(|high|, |low|, min_range_absolute)
    # This used to be a hard-coded 1.0, which made the guard absolute rather
    # than relative for every metric whose values sit below 1.0 - which is all
    # of them, because contrast normalisation divides by mean(I)**2.  Measured
    # on the protocol recordings: the whole value of `brenner` at defocus is
    # 0.00029 while the floor was 0.001, so the guard could never be satisfied
    # and four metrics of six returned the 0.5 sentinel on every defocused
    # frame.  See docs/CALIBRATION.md.
    min_range_absolute: float = 1e-6
    # When the rolling window's range is degenerate, fall back to a longer
    # history that is this many windows deep before giving up and returning
    # the neutral sentinel.  A held defocus position has no range of its own,
    # but the sweep it belongs to does, and scaling against the sweep puts the
    # frame near 0 instead of at a misleading mid-range 0.5.  Set to 0 to
    # restore the previous sentinel-only behaviour.  Now that `window` is
    # itself long, this is only a backstop for a scene held perfectly still
    # for longer than the window.
    long_window_multiple: int = 2
    # Fraction of metrics that must be informative before a result is reported
    # ready.  The original test was "more than none", so a result with five of
    # six metrics pinned at the sentinel still announced itself as ready.
    ready_informative_fraction: float = 0.5
    # Freeze the scaling anchors once the observed range has stopped growing.
    #
    # A rolling window makes the score relative to recent history, which breaks
    # the property a focus search needs: that within one stream a higher score
    # means a sharper frame.  Freezing does not *guarantee* that - the fused
    # score is a weighted sum of six metrics that need not agree, and the
    # weights themselves move - it removes the one cause that was provably
    # breaking it.  Measured against the point-source reference, the raw
    # metric tracks true focus at rank correlation 0.97-0.98, the rolling
    # window at 120 frames scores -0.24 and -0.38 - worse than useless - and a
    # frozen scale recovers 0.92 and 0.88.  On a slow approach to focus the
    # window fills with high values, so the sharpest frames of all are scored
    # as "no better than recent history" and the score collapses.
    #
    # Freezing waits for a representative range rather than freezing early on a
    # narrow one, which would clip everything afterwards.  A host that performs
    # its own coarse sweep should call NormalizerBank.freeze() itself instead.
    auto_freeze: bool = True
    # Samples required before freezing is even considered.
    auto_freeze_min_samples: int = 240
    # Freeze when the range has grown by less than this fraction ...
    auto_freeze_stability: float = 0.05
    # ... over this many consecutive samples.
    auto_freeze_patience: int = 120
    # Thaw a frozen scale again when the scene appears to have changed.
    #
    # OFF by default, because no threshold tested could tell a focus change
    # from a scene change.  A sweep legitimately runs past the anchors - they
    # are percentiles of an earlier window - so the rule fires mid-sweep and
    # throws away the fixed scale exactly when it is doing its job.  Measured
    # on the point-source runs, rank correlation with the physical spot size:
    #
    #     thawing off          0.940   0.967
    #     margin 1 span        0.913   0.277
    #     margin 3 spans       0.930   0.329
    #     margin 5 spans       0.937   0.354
    #
    # A host that knows the scene changed - the camera was repointed, the lens
    # swapped, the exposure altered - should call unfreeze() and say so, which
    # is information no statistic in the stream can recover.
    auto_thaw: bool = False
    # Fraction of the recent window that must lie outside the frozen range,
    # widened by auto_thaw_margin spans on each side, before thawing.
    auto_thaw_outside: float = 0.5
    auto_thaw_margin: float = 3.0


@dataclass(frozen=True)
class AnalysisConfig:
    """References used to map raw image measurements onto [0, 1] levels."""

    # Noise sigma (8-bit units) that maps to noise_level = 1.
    noise_ref_sigma: float = 6.0
    # Quantile of |HH| used by the noise estimator, in percent.  The classical
    # Donoho-Johnstone estimator uses the median (50).  Measured on the
    # protocol recordings, 58-74% of the Haar HH coefficients of this camera's
    # live-view stream are *exactly* zero after 8-bit quantisation, so the
    # median is zero and the estimate is zero on 100% of real frames - which
    # silently disabled the entire noise branch of the weighting model.  A
    # higher quantile is the same estimator evaluated where the distribution is
    # not degenerate; the scale factor is derived from the quantile, so the
    # estimate remains unbiased for Gaussian noise.
    noise_quantile: float = 75.0
    # Inter-frame displacement (pixels, at analysis scale) mapping to level 1.
    # Measured on a 1288-frame handheld GH6 recording while the focus ring was
    # being turned: median 1.4 px, p75 5.8, p90 17.9, p95 29.0. The original
    # guess of 3.0 saturated on 38% of frames, so ordinary hand-held shake read
    # as "maximum motion".
    motion_ref_px: float = 15.0
    # Edge-pixel fraction at/above which structure is considered sufficient.
    # Measured with tools/calibrate_stats.py rather than guessed: a rich,
    # high-contrast synthetic scene reaches ~0.045 and a real GH6 live-view
    # frame of an ordinary indoor subject sits at 0.006-0.009, so the original
    # guess of 0.06 was unreachable and capped the confidence for every frame.
    edge_ref_density: float = 0.006
    # RMS contrast at/above which the scene is considered well textured.
    contrast_ref: float = 0.08
    # Intensities at/above (below) these bounds count as clipped.
    clip_high_level: int = 250
    clip_low_level: int = 5
    # Clipped fraction that maps to a fully degraded exposure score.
    clip_ref_fraction: float = 0.15
    # Brightness band considered well exposed (mean intensity, normalised).
    brightness_low: float = 0.15
    brightness_high: float = 0.85
    # Downscale factor applied on top of the analysis image for motion
    # estimation, which does not need full analysis resolution.
    motion_downscale: int = 4


@dataclass(frozen=True)
class FocusMapConfig:
    """Dense per-tile focus map used to locate what is in focus."""

    # Tile side in analysis-image pixels.  At the default 320 px analysis width
    # a 32 px tile gives a 10 x 5 grid on a 16:9 frame.
    tile_size: int = 32
    # Which per-tile measure to use; see sharpness/focusmap.py.
    # Chosen by measurement (tools/compare_focus_measures.py): all three
    # candidates separate sharp from blurred on clean split-focus scenes, but at
    # noise sigma 20 "lap_over_grad" collapses to a 0.039 margin - a ratio of two
    # high-pass measures is dominated by noise - while "grad_over_var" keeps a
    # 0.257 margin and also has the smallest false margin when both halves are
    # equally sharp.
    measure: str = "grad_over_var"
    # A tile below this RMS contrast carries no usable focus information and is
    # marked invalid rather than being reported as "out of focus".
    min_contrast: float = 0.02
    # Optional additional floor on the tile's gradient energy.
    min_gradient_energy: float = 0.0
    # Robust anchors for the within-frame rescale to [0, 1].
    low_percentile: float = 10.0
    high_percentile: float = 90.0
    # How much the raw tile measure must vary across the frame, relative to its
    # own median, before the rescale is allowed to stretch to the full range.
    # Without this, a uniformly sharp frame has its numerical noise amplified
    # into a confident-looking winner.
    min_relative_spread: float = 0.8
    # Light smoothing across the tile grid.
    smooth: bool = True


@dataclass(frozen=True)
class RegionsConfig:
    """Region proposals for the "what is in focus" display."""

    # Detectors to run, in order.  "sharp_blobs" clusters the focus map itself;
    # "faces" uses the Haar cascade bundled with OpenCV; "contours" proposes
    # generic high-structure blobs.
    detectors: tuple[str, ...] = ("faces", "sharp_blobs")
    # Cascade file name inside cv2.data.haarcascades.
    face_cascade: str = "haarcascade_frontalface_default.xml"
    # Haar detector parameters.
    face_scale_factor: float = 1.15
    face_min_neighbors: int = 5
    face_min_size_fraction: float = 0.08
    # Run the cascade only every Nth frame and reuse its boxes in between; it is
    # by far the most expensive step in the region stage.
    face_interval: int = 3
    # Fraction of the peak tile score above which a tile joins a sharp blob.
    blob_relative_threshold: float = 0.55
    # Smallest blob, in tiles.
    blob_min_tiles: int = 2
    # Largest number of regions scored with the full ensemble per frame.
    max_regions: int = 5
    # Regions smaller than this fraction of the frame are discarded.
    min_area_fraction: float = 0.004
    # Merge proposals whose intersection-over-union exceeds this.
    merge_iou: float = 0.45
    # IoU above which the winning region is treated as the same subject as on
    # the previous frame, so its normalisation history is kept.
    subject_iou_match: float = 0.30


@dataclass(frozen=True)
class EnsembleConfig:
    """The adaptive weighting model.

    ``sensitivity`` holds the per-metric coefficients kappa_ij of the
    log-linear reliability model

        r_i = exp( - sum_j kappa_ij * d_j )

    where d_j are the degradation factors measured by :mod:`sharpness.analysis`
    (noise, edge deficiency, clipping, motion, low contrast).  Larger kappa
    means the metric degrades faster under that condition.

    The defaults encode the qualitative behaviour documented in the literature
    and in ``docs/ALGORITHM.md``: second-derivative operators amplify noise most,
    the edge-width metric collapses without edges, and so on.  They are starting
    points to be refined by the calibration experiment, not measured constants.
    """

    sensitivity: Mapping[str, Mapping[str, float]] = field(
        default_factory=lambda: {
            #            noise  edge   clip   motion lowcontrast
            "laplacian": {"noise": 2.2, "edge": 0.6, "clip": 0.5, "motion": 0.6, "contrast": 0.7},
            "tenengrad": {"noise": 1.2, "edge": 0.7, "clip": 0.6, "motion": 0.7, "contrast": 0.8},
            "brenner": {"noise": 1.0, "edge": 0.9, "clip": 0.6, "motion": 0.9, "contrast": 0.9},
            "wavelet": {"noise": 1.6, "edge": 0.5, "clip": 0.5, "motion": 0.6, "contrast": 0.6},
            "fourier": {"noise": 1.9, "edge": 0.4, "clip": 0.7, "motion": 0.5, "contrast": 0.6},
            "edge_width": {"noise": 1.4, "edge": 2.2, "clip": 0.9, "motion": 1.0, "contrast": 1.0},
            # Same family as tenengrad - a first-derivative operator - so it is
            # given the same row.  Assumed by analogy like every other row
            # here; see the class docstring for what that is worth.
            "gradient_variance": {"noise": 1.3, "edge": 0.7, "clip": 0.6, "motion": 0.7, "contrast": 0.8},
        }
    )
    # Enable the consensus-agreement reweighting term a_i.
    #
    # OFF by default.  It was one of the three mechanisms this project claimed
    # as its contribution, and it is the one that did not survive measurement.
    # On eight protocol recordings, against the physical point-source ground
    # truth, switching it off improved every criterion: rank correlation
    # 0.946 -> 0.964, wrongly ordered step pairs 0.057 -> 0.052, adjacent-step
    # discrimination 0.816 -> 0.818, and 0.08 ms a frame cheaper.  Widening the
    # kernel only walks the result back towards "off" (0.958 at scale 1.0,
    # 0.963 at scale 8.0, 0.965 off), so this is not a tuning failure.
    #
    # What that does NOT establish: the term exists to protect the score when a
    # single metric is fooled - a specular highlight, a blown region - and none
    # of these recordings contains that failure.  It is switched off because it
    # costs measurably and its benefit is untested, not because the idea is
    # wrong.  The reliability stage, by contrast, clearly earns its place:
    # zeroing kappa doubles the wrong orderings (0.057 -> 0.105).
    use_agreement: bool = False
    # Scale of the agreement kernel, in units of the robust spread of the
    # normalised scores.  Smaller = more aggressive outlier suppression.
    agreement_scale: float = 1.5
    # Floor applied to every weight before renormalisation, so that a metric is
    # never fully silenced (keeps the ensemble responsive if conditions change).
    weight_floor: float = 0.02
    # Master switch for the noise-compensation term (ablation study).
    use_noise_compensation: bool = True
    # Master switch for the motion term (ablation study).
    use_motion_compensation: bool = True
    # SNR at/above which the SNR confidence factor saturates at 1.
    confidence_snr_ref: float = 12.0
    # Robust spread of the normalised metrics that maps to zero concordance.
    # Measured with tools/calibrate_stats.py on live GH6 frames: over a
    # subject-sized ROI the routine spread is 0.18 median and 0.33 at the 95th
    # percentile, so the original 0.25 zeroed the concordance - and with it the
    # whole confidence - on about a quarter of perfectly good frames.
    confidence_dispersion_ref: float = 0.45
    # Confidence factor applied while the normalisers are still warming up.
    warmup_confidence: float = 0.3


@dataclass(frozen=True)
class TemporalConfig:
    """Innovation-gated exponential smoothing.

    A plain EMA hides genuine focus transitions, which is unacceptable for an
    autofocus loop.  The filter therefore raises its gain when the innovation is
    large relative to the recent noise of the score, so real focus moves pass
    through nearly unfiltered while small fluctuations stay smoothed.
    """

    enabled: bool = True
    # Baseline smoothing factor (0 = frozen, 1 = no smoothing).
    alpha_base: float = 0.35
    # Innovation magnitude, in robust sigmas of the score history, at which the
    # gate starts opening.
    gate_sigma: float = 2.5
    # Innovation, in sigmas, at which the filter is fully open (alpha -> 1).
    gate_full_sigma: float = 5.0
    # Absolute innovation that always counts as a real focus change, even when
    # the score history is very quiet.
    gate_abs_jump: float = 0.12
    # Length of the history used to estimate the score noise.
    history: int = 30
    # Scale the gain by the confidence: unreliable frames are smoothed harder.
    confidence_coupling: bool = True
    # Lowest gain multiplier the confidence coupling may impose.
    min_confidence_gain: float = 0.3


@dataclass(frozen=True)
class CaptureConfig:
    """Frame-source selection for the CLI tools."""

    # One of: auto, v4l2, gphoto2, file, synthetic.
    backend: str = "auto"
    # V4L2 device index or path; ignored by the other backends.
    device: str = "0"
    # Path for the ``file`` backend (video file or directory of images).
    path: str = ""
    width: int = 1280
    height: int = 720
    fps: int = 30
    # FOURCC requested from V4L2 (MJPG keeps USB bandwidth sane).
    fourcc: str = "MJPG"
    # Capture in a worker thread and always hand the newest frame to the
    # consumer, dropping stale ones.
    threaded: bool = True
    # Loop file sources forever (useful for the demo).
    loop: bool = False


@dataclass(frozen=True)
class SharpnessConfig:
    """Root configuration object."""

    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    focus_map: FocusMapConfig = field(default_factory=FocusMapConfig)
    regions: RegionsConfig = field(default_factory=RegionsConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    capture: CaptureConfig = field(default_factory=CaptureConfig)

    def with_overrides(self, **sections: Mapping[str, Any]) -> "SharpnessConfig":
        """Return a copy with per-section keyword overrides applied."""
        updated: dict[str, Any] = {}
        for name, values in sections.items():
            if not hasattr(self, name):
                raise KeyError(f"unknown configuration section: {name!r}")
            updated[name] = replace(getattr(self, name), **dict(values))
        return replace(self, **updated)


def _coerce_section(section_cls: type, values: Mapping[str, Any], path: str) -> Any:
    """Build a dataclass instance from a mapping, validating the key names."""
    known = {f.name: f for f in fields(section_cls)}
    kwargs: dict[str, Any] = {}
    for key, value in values.items():
        if key not in known:
            raise KeyError(f"unknown configuration key: {path}.{key}")
        target = known[key]
        # tuple-typed fields arrive from TOML as lists.
        if isinstance(value, list) and "tuple" in str(target.type):
            value = tuple(value)
        kwargs[key] = value
    return section_cls(**kwargs)


def default_config_path() -> Path:
    """Path to the configuration file that ships inside the package.

    The repository also has ``config/default.toml``, but that copy is not part
    of the wheel: after a plain ``pip install`` it does not exist.  Anything
    that needs the shipped defaults must go through here.
    """
    return Path(__file__).resolve().parent / "data" / "default.toml"


def load_default_config() -> SharpnessConfig:
    """Load the packaged default configuration.

    Falls back to the dataclass defaults if the file is missing, so an unusual
    installation degrades instead of failing.  ``tests/test_config.py`` keeps
    the two in agreement.
    """
    path = default_config_path()
    if not path.is_file():
        logger.warning(
            "packaged default config not found at %s; using built-in defaults", path
        )
        return SharpnessConfig()
    return load_config(path)


def load_config(path: str | Path | None = None) -> SharpnessConfig:
    """Load a :class:`SharpnessConfig` from a TOML file.

    Passing ``None`` returns the defaults.  Unknown keys raise ``KeyError`` so
    that typos in a config file fail loudly instead of being silently ignored.
    """
    if path is None:
        return SharpnessConfig()

    import tomllib

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"configuration file not found: {file_path}")

    with file_path.open("rb") as handle:
        raw = tomllib.load(handle)

    sections: dict[str, Any] = {}
    root_fields = {f.name: f.type for f in fields(SharpnessConfig)}
    section_types = {
        "pipeline": PipelineConfig,
        "metrics": MetricsConfig,
        "normalization": NormalizationConfig,
        "analysis": AnalysisConfig,
        "focus_map": FocusMapConfig,
        "regions": RegionsConfig,
        "ensemble": EnsembleConfig,
        "temporal": TemporalConfig,
        "capture": CaptureConfig,
    }
    for name, values in raw.items():
        if name not in root_fields:
            raise KeyError(f"unknown configuration section: {name!r}")
        if not isinstance(values, Mapping):
            raise TypeError(f"section {name!r} must be a table")
        sections[name] = _coerce_section(section_types[name], values, name)

    logger.debug("loaded configuration from %s (%d sections)", file_path, len(sections))
    return SharpnessConfig(**sections)
