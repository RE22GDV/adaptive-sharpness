"""Live UI: which object is in focus right now, with every stage switchable.

Shows the video with a per-tile focus heat map, boxes around the candidate
subjects, the winner called out by name, and a panel with the full breakdown -
the ensemble score and confidence for the subject, every metric with its current
weight, the image statistics, a rolling history plot, and the measured frame
rate and latency.

Every stage of the algorithm can be switched on and off at run time, so the
effect of each one is visible directly rather than only in an offline ablation
table: individual metrics, the adaptive weighting, noise compensation, motion
compensation, the consensus-agreement stage, and the temporal filter.

Two honest distinctions the display keeps visible, because collapsing them would
make it lie:

* **"no texture" is not "out of focus".** A tile without enough contrast carries
  no focus information at all, so it is left untinted rather than painted as
  blurred. The panel reports what fraction of the frame was even measurable.
* **Confidence in the score is not confidence in the decision.** A subject can
  have a well-measured score while barely beating the runner-up. The panel shows
  both: `conf` for the measurement, `decision margin` for the ranking.

Controls
--------
q / Esc  quit                  1..6  toggle individual metrics
space    pause                 a     adaptive weighting on/off
r        reset state           n     noise compensation on/off
s        save a PNG snapshot   m     motion compensation on/off
h        heat map on/off       c     consensus agreement on/off
b        candidate boxes       e     temporal filter on/off
t        tile grid             f     face detector on/off
g        cycle tile size       0     restore all defaults

Headless use (no window needed)::

    python3 demo/focus_ui.py --backend gphoto2 --headless --frames 60
    python3 demo/focus_ui.py --backend synthetic --snapshot docs/figures/ui.png
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import deque
from datetime import datetime
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_sharpness.capture import CaptureError, open_source  # noqa: E402
from adaptive_sharpness import (  # noqa: E402
    METRIC_NAMES,
    RunRecorder,
    SceneEvaluator,
    SceneResult,
    SharpnessConfig,
    load_config,
    load_default_config,
)

logger = logging.getLogger("focus_ui")

PANEL_WIDTH = 360
PLOT_HEIGHT = 100
MIN_CANVAS_HEIGHT = 800

BACKGROUND = (26, 26, 30)
TEXT = (234, 234, 238)
MUTED = (146, 146, 156)
DIM = (96, 96, 106)
GOOD = (120, 210, 120)
OKAY = (90, 180, 250)
POOR = (92, 92, 240)
WINNER = (110, 235, 130)
CANDIDATE = (170, 170, 90)
OFF = (78, 78, 88)
REC_IDLE = (70, 70, 200)
REC_LIVE = (60, 60, 235)

#: Canvas-space rectangle of the record button, refreshed by render() so the
#: mouse callback can hit-test it. Kept module level because OpenCV's
#: highgui has no widget model to attach it to.
RECORD_BUTTON: dict[str, tuple[int, int, int, int]] = {}

TILE_SIZES = (16, 24, 32, 48)
#: Short labels for the metric toggles, in the order of METRIC_NAMES.
METRIC_KEYS = ("Lap", "Ten", "Bre", "Wav", "Fou", "Edg")


def focus_lut() -> np.ndarray:
    """A 256-entry BGR palette where bright green literally means "in focus".

    The obvious choice, COLORMAP_TURBO, runs blue -> green -> red, so its
    *sharpest* value is red and green is merely the middle of the scale.  On real
    footage the in-focus areas happened to land in that middle, which reads as
    "green = sharp" by accident and would teach a viewer the wrong rule.  This
    palette is monotone in one hue instead: dark and desaturated where the image
    is soft, bright green where it is sharp - the same convention as focus
    peaking in a camera.
    """
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    ramp = np.linspace(0.0, 1.0, 256)
    lut[:, 0, 0] = np.clip(90 * (1.0 - ramp) ** 1.5, 0, 255)   # blue
    lut[:, 0, 1] = np.clip(40 + 215 * ramp ** 1.3, 0, 255)     # green
    lut[:, 0, 2] = np.clip(70 * (1.0 - ramp) ** 1.8, 0, 255)   # red
    return lut


_FOCUS_LUT = focus_lut()


def quality_colour(value: float) -> tuple[int, int, int]:
    if value >= 0.65:
        return GOOD
    if value >= 0.35:
        return OKAY
    return POOR


# ---------------------------------------------------------------------------
# Run-time switches
# ---------------------------------------------------------------------------

@dataclass
class UiState:
    """Every switch the operator can flip, and how it maps to the config.

    Keeping this separate from :class:`SharpnessConfig` means the UI can rebuild
    a fresh evaluator from a known-good base configuration on every change,
    rather than mutating a frozen dataclass in place.
    """

    metrics: list[str] = field(default_factory=lambda: list(METRIC_NAMES))
    adaptive: bool = True
    noise_compensation: bool = True
    motion_compensation: bool = True
    agreement: bool = True
    temporal: bool = True
    faces: bool = True
    tile_index: int = TILE_SIZES.index(32)
    # Display-only switches; they do not affect the computation.
    heatmap: bool = True
    boxes: bool = True
    grid: bool = False

    def toggle_metric(self, index: int) -> bool:
        """Flip metric ``index``; refuses to leave the ensemble empty."""
        if not 0 <= index < len(METRIC_NAMES):
            return False
        name = METRIC_NAMES[index]
        if name in self.metrics:
            if len(self.metrics) == 1:
                logger.warning("at least one metric must stay enabled")
                return False
            self.metrics.remove(name)
        else:
            self.metrics.append(name)
        # Keep the canonical order so the panel rows never jump around.
        self.metrics = [n for n in METRIC_NAMES if n in self.metrics]
        return True

    def apply(self, base: SharpnessConfig) -> SharpnessConfig:
        """Build the configuration these switches describe."""
        ensemble = replace(
            base.ensemble,
            use_noise_compensation=self.noise_compensation,
            use_motion_compensation=self.motion_compensation,
            use_agreement=self.agreement,
        )
        if not self.adaptive:
            # "Adaptive off" means the fixed weighted mean of the priors: zero
            # sensitivity to every condition, and no agreement reweighting.
            ensemble = replace(
                ensemble,
                sensitivity={
                    name: {key: 0.0 for key in profile}
                    for name, profile in base.ensemble.sensitivity.items()
                },
                use_agreement=False,
            )
        detectors = ("faces", "sharp_blobs") if self.faces else ("sharp_blobs",)
        return replace(
            base,
            metrics=replace(base.metrics, enabled=tuple(self.metrics)),
            ensemble=ensemble,
            temporal=replace(base.temporal, enabled=self.temporal),
            regions=replace(base.regions, detectors=detectors),
            focus_map=replace(base.focus_map, tile_size=TILE_SIZES[self.tile_index]),
        )

    def summary(self) -> str:
        flags = [
            ("adapt", self.adaptive),
            ("noise", self.noise_compensation),
            ("motion", self.motion_compensation),
            ("agree", self.agreement),
            ("temporal", self.temporal),
        ]
        off = [name for name, on in flags if not on]
        parts = [f"{len(self.metrics)}/{len(METRIC_NAMES)} metrics"]
        parts.append("all stages on" if not off else "off: " + ",".join(off))
        return "; ".join(parts)


# ---------------------------------------------------------------------------
# Overlay on the video
# ---------------------------------------------------------------------------

def draw_heatmap(frame: np.ndarray, result: SceneResult, alpha: float = 0.30) -> np.ndarray:
    """Tint the frame by per-tile focus, leaving untextured tiles untouched.

    Two things keep this from lying to the viewer: tiles without enough texture
    are not tinted at all, so "no information" never looks like "out of focus";
    and the tint strength follows how decisive the reading was, so a frame with
    no real focus variation stays almost untinted instead of being painted a
    confident colour.
    """
    import cv2

    fmap = result.focus_map
    height, width = frame.shape[:2]
    score = (np.clip(fmap.score, 0.0, 1.0) * 255).astype(np.uint8)
    coloured = cv2.applyColorMap(score, _FOCUS_LUT)
    coloured = cv2.resize(coloured, (width, height), interpolation=cv2.INTER_LINEAR)

    strength = np.abs(fmap.score - 0.5) * 2.0
    mask = fmap.valid.astype(np.float32) * (0.35 + 0.65 * strength)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_LINEAR)
    mask = (mask[:, :, None] * alpha).astype(np.float32)

    blended = frame.astype(np.float32) * (1.0 - mask) + coloured.astype(np.float32) * mask
    return blended.astype(np.uint8)


def draw_tile_grid(frame: np.ndarray, result: SceneResult) -> None:
    import cv2

    rows, cols = result.focus_map.shape
    height, width = frame.shape[:2]
    for col in range(1, cols):
        x = int(col * width / cols)
        cv2.line(frame, (x, 0), (x, height), (60, 60, 68), 1)
    for row in range(1, rows):
        y = int(row * height / rows)
        cv2.line(frame, (0, y), (width, y), (60, 60, 68), 1)


def draw_regions(frame: np.ndarray, result: SceneResult, scale: float) -> None:
    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    for rank, region in enumerate(result.regions):
        x, y = int(region.x * scale), int(region.y * scale)
        w, h = int(region.width * scale), int(region.height * scale)
        winner = rank == 0
        colour = WINNER if winner else CANDIDATE
        cv2.rectangle(frame, (x, y), (x + w, y + h), colour, 3 if winner else 1)

        label = f"{region.label}  {region.map_score:.2f}"
        if winner:
            label = "IN FOCUS: " + label
        (tw, th), _ = cv2.getTextSize(label, font, 0.5, 1)
        ty = max(th + 6, y - 6)
        cv2.rectangle(frame, (x, ty - th - 5), (x + tw + 8, ty + 4), colour, -1)
        cv2.putText(frame, label, (x + 4, ty), font, 0.5, (20, 20, 24), 1, cv2.LINE_AA)


def compose_video(
    frame: np.ndarray, result: SceneResult, state: UiState, target_height: int
) -> np.ndarray:
    import cv2

    display = draw_heatmap(frame, result) if state.heatmap else frame.copy()
    if state.grid:
        draw_tile_grid(display, result)
    scale = target_height / display.shape[0]
    display = cv2.resize(
        display,
        (int(round(display.shape[1] * scale)), target_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    if state.boxes:
        draw_regions(display, result, scale)
    return display


# ---------------------------------------------------------------------------
# Side panel
# ---------------------------------------------------------------------------

def draw_plot(
    canvas: np.ndarray,
    values: Sequence[float],
    colours: Sequence[float],
    origin: tuple[int, int],
    size: tuple[int, int],
) -> None:
    import cv2

    x0, y0 = origin
    width, height = size
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (44, 44, 50), -1)
    for fraction in (0.25, 0.5, 0.75):
        y = int(y0 + height - fraction * height)
        cv2.line(canvas, (x0, y), (x0 + width, y), (60, 60, 68), 1)
    if len(values) < 2:
        return
    step = width / float(len(values) - 1)
    for index in range(1, len(values)):
        p0 = (int(x0 + (index - 1) * step), int(y0 + height - values[index - 1] * height))
        p1 = (int(x0 + index * step), int(y0 + height - values[index] * height))
        cv2.line(canvas, p0, p1, quality_colour(colours[index]), 2, cv2.LINE_AA)


def draw_panel(
    result: SceneResult,
    state: UiState,
    height: int,
    fps: float,
    age_ms: float,
    history: Sequence[float],
    confidence_history: Sequence[float],
    dropped: int,
    paused: bool,
    recorder: RunRecorder | None = None,
) -> np.ndarray:
    import cv2

    panel = np.full((height, PANEL_WIDTH, 3), BACKGROUND, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    left, right = 14, PANEL_WIDTH - 14
    column = left + 176

    def text(label: str, y: int, colour=TEXT, scale=0.42, x=left) -> None:  # noqa: ANN001
        cv2.putText(panel, label, (x, y), font, scale, colour, 1, cv2.LINE_AA)

    def bar(y: int, value: float, colour, h: int = 6) -> None:  # noqa: ANN001
        cv2.rectangle(panel, (left, y), (right, y + h), (52, 52, 60), -1)
        filled = int((right - left) * max(0.0, min(1.0, value)))
        cv2.rectangle(panel, (left, y), (left + filled, y + h), colour, -1)

    def rule(y: int) -> int:
        cv2.line(panel, (left, y), (right, y), (54, 54, 62), 1)
        return y + 17

    subject = result.subject
    subject_result = result.subject_result
    fmap = result.focus_map

    # --- record button ------------------------------------------------------
    bx1, by1 = PANEL_WIDTH - 104, 12
    bx2, by2 = PANEL_WIDTH - 14, 40
    recording = recorder is not None and recorder.active
    cv2.rectangle(panel, (bx1, by1), (bx2, by2),
                  REC_LIVE if recording else (52, 52, 60), -1)
    cv2.rectangle(panel, (bx1, by1), (bx2, by2),
                  REC_LIVE if recording else REC_IDLE, 1)
    if recording:
        cv2.rectangle(panel, (bx1 + 9, by1 + 10), (bx1 + 17, by1 + 18), (255, 255, 255), -1)
        text("STOP", by2 - 9, (255, 255, 255), 0.42, x=bx1 + 24)
    else:
        cv2.circle(panel, (bx1 + 13, by1 + 14), 5, REC_IDLE, -1)
        text("REC", by2 - 9, TEXT, 0.42, x=bx1 + 24)
    # Panel-space rect; render() converts it to canvas space for hit-testing.
    RECORD_BUTTON["panel"] = (bx1, by1, bx2, by2)

    # --- what is in focus --------------------------------------------------
    y = 24
    text("IN FOCUS", y, MUTED, 0.44)
    y += 27
    if subject is None:
        cv2.putText(panel, "nothing", (left, y), font, 0.76, POOR, 2, cv2.LINE_AA)
        y += 18
        text("no region has enough texture", y, MUTED, 0.37)
        y += 16
    else:
        cv2.putText(panel, subject.label, (left, y), font, 0.76, WINNER, 2, cv2.LINE_AA)
        y += 18
        text(f"at ({subject.center[0]}, {subject.center[1]})   "
             f"{subject.width}x{subject.height} px", y, MUTED, 0.37)
        y += 16

    separation = result.separation()
    runner = result.runner_up
    text("decision margin", y, MUTED, 0.39)
    y += 11
    if runner is None:
        bar(y, 0.0, DIM)
        y += 15
        text("sole candidate - nothing to compare with", y, MUTED, 0.36)
    else:
        bar(y, min(1.0, separation * 4.0), quality_colour(min(1.0, separation * 4.0)))
        y += 15
        text(f"lead over '{runner.label}': {separation:+.3f}", y, MUTED, 0.36)
    y += 15
    text(f"focus variation {fmap.relative_spread:4.2f}   "
         f"stretch {fmap.stretch:.2f}", y, MUTED, 0.36)
    y = rule(y + 8)

    # --- subject measurement ----------------------------------------------
    if subject_result is not None:
        colour = quality_colour(subject_result.confidence)
        cv2.putText(panel, f"{subject_result.score:.3f}", (left, y + 8), font,
                    0.88, colour, 2, cv2.LINE_AA)
        text(f"conf {subject_result.confidence:.2f}", y + 3, colour, 0.43, x=column)
        text("subject sharpness", y + 17, MUTED, 0.36, x=column)
        y += 22
        bar(y, subject_result.score, colour, 5)
        y += 16
        if subject_result.score_change_detected:
            text("SCORE JUMPED", y, GOOD, 0.41)
            y += 14

        text("metric         norm  weight", y, MUTED, 0.37)
        for sample in subject_result.metrics:
            y += 15
            text(f"{sample.name:<13s}{sample.normalized:5.2f}  {sample.weight:5.3f}",
                 y, TEXT, 0.37)
            cv2.rectangle(panel, (left, y + 2), (right, y + 4), (46, 46, 54), -1)
            width = int((right - left) * min(1.0, sample.weight * 3.0))
            cv2.rectangle(panel, (left, y + 2), (left + width, y + 4), (78, 104, 150), -1)
        # Metrics switched off are listed, so the display never silently shrinks.
        disabled = [n for n in METRIC_NAMES if n not in state.metrics]
        if disabled:
            y += 15
            text("off: " + ", ".join(disabled), y, OFF, 0.36)
        y += 17

        stats = subject_result.stats
        for a, b in (
            (f"edges  {stats.edge_density:.4f}", f"suff {stats.edge_sufficiency:.2f}"),
            (f"contr  {stats.local_contrast:.4f}", f"noise {stats.noise_sigma:.2f}"),
            (f"clip   {stats.clipped_high:.3f}", f"motion {stats.motion_px:.2f} px"),
        ):
            y += 14
            text(a, y, MUTED, 0.36)
            text(b, y, MUTED, 0.36, x=column)
        y += 8
    y = rule(y + 6)

    # --- frame-level context -----------------------------------------------
    if result.frame_result is not None:
        text(f"whole frame {result.frame_result.score:.3f}", y, MUTED, 0.36)
        text(f"conf {result.frame_result.confidence:.2f}", y, MUTED, 0.36, x=column)
        y += 14
    text(f"measurable  {fmap.valid_fraction * 100:3.0f}%", y, MUTED, 0.36)
    text(f"grid {fmap.shape[1]}x{fmap.shape[0]} @{fmap.tile_size}px", y, MUTED, 0.36,
         x=column)
    y += 14
    text(f"candidates  {len(result.regions)}", y, MUTED, 0.36)
    text("faces on" if state.faces else "faces OFF", y,
         MUTED if state.faces else OFF, 0.36, x=column)
    y = rule(y + 8)

    # --- algorithm switches -------------------------------------------------
    text("stages   (1-6 metrics, a n m c e)", y, MUTED, 0.36)
    y += 15
    x = left
    for index, short in enumerate(METRIC_KEYS):
        on = METRIC_NAMES[index] in state.metrics
        cv2.rectangle(panel, (x, y - 9), (x + 50, y + 2),
                      (58, 58, 66) if on else (40, 40, 46), -1)
        text(f"{index + 1}{short}", y, TEXT if on else OFF, 0.36, x=x + 4)
        x += 55
    y += 18
    x = left
    for label, on in (
        ("adapt", state.adaptive),
        ("noise", state.noise_compensation),
        ("motion", state.motion_compensation),
    ):
        text(label, y, GOOD if on else OFF, 0.36, x=x)
        x += 62
    y += 14
    x = left
    for label, on in (("agree", state.agreement), ("temporal", state.temporal)):
        text(label, y, GOOD if on else OFF, 0.36, x=x)
        x += 62
    y = rule(y + 8)

    # --- timing -------------------------------------------------------------
    text(f"{fps:5.1f} fps", y + 2, TEXT, 0.46)
    text(f"age {age_ms:5.1f} ms", y + 2, MUTED, 0.36, x=column)
    y += 16
    text(f"scene {result.scene_time_s * 1e3:5.1f} ms", y, MUTED, 0.35)
    text(f"total {result.total_time_s * 1e3:5.1f} ms", y, MUTED, 0.35, x=column)
    y += 14
    text(f"dropped {dropped}", y, MUTED, 0.35)
    if paused:
        text("PAUSED", y, OKAY, 0.43, x=column)
    if recorder is not None and recorder.active:
        y += 15
        text(f"REC {recorder.elapsed:5.1f} s   {recorder.frames} frames",
             y, REC_LIVE, 0.4)
        y += 14
        text(f"-> {recorder.directory}", y, MUTED, 0.34)

    # --- legend and history -------------------------------------------------
    legend_y = height - PLOT_HEIGHT - 76
    text("green = in focus, dark = soft, untinted = no texture",
         legend_y, MUTED, 0.35)
    gradient = np.linspace(0, 255, right - left).astype(np.uint8)[None, :]
    strip = cv2.applyColorMap(gradient, _FOCUS_LUT)
    panel[legend_y + 6 : legend_y + 15, left:right] = np.repeat(strip, 9, axis=0)
    text("soft", legend_y + 25, DIM, 0.33)
    text("sharp", legend_y + 25, DIM, 0.33, x=right - 32)

    plot_y = height - PLOT_HEIGHT - 24
    draw_plot(panel, list(history), list(confidence_history),
              (left, plot_y), (right - left, PLOT_HEIGHT))
    text("q quit  h heat  b boxes  g grid  s save  R record  0 reset",
         height - 8, DIM, 0.33)
    return panel


def compose(video: np.ndarray, panel: np.ndarray) -> np.ndarray:
    height = max(video.shape[0], panel.shape[0])
    if video.shape[0] != height:
        pad = np.full((height - video.shape[0], video.shape[1], 3), BACKGROUND, np.uint8)
        video = np.vstack([video, pad])
    if panel.shape[0] != height:
        pad = np.full((height - panel.shape[0], PANEL_WIDTH, 3), BACKGROUND, np.uint8)
        panel = np.vstack([panel, pad])
    return np.hstack([video, panel])


def render(
    frame: np.ndarray,
    result: SceneResult,
    state: UiState,
    fps: float = 0.0,
    age_ms: float = 0.0,
    history: Sequence[float] = (),
    confidence_history: Sequence[float] = (),
    dropped: int = 0,
    paused: bool = False,
    recorder: RunRecorder | None = None,
) -> np.ndarray:
    """Build the complete UI image for one frame."""
    target_height = max(MIN_CANVAS_HEIGHT, frame.shape[0])
    video = compose_video(frame, result, state, target_height)
    panel = draw_panel(result, state, video.shape[0], fps, age_ms,
                       history, confidence_history, dropped, paused, recorder)
    # The panel sits to the right of the video, so shift the button rect by the
    # video width to get canvas coordinates for hit-testing.
    if "panel" in RECORD_BUTTON:
        bx1, by1, bx2, by2 = RECORD_BUTTON["panel"]
        offset = video.shape[1]
        RECORD_BUTTON["canvas"] = (bx1 + offset, by1, bx2 + offset, by2)
    return compose(video, panel)


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

def toggle_recording(
    recorder: RunRecorder | None,
    config: SharpnessConfig,
    out_root: Path,
    save_every: int,
    source_description: str,
) -> RunRecorder | None:
    """Start a new recording, or stop the running one.

    Each recording gets its own timestamped directory, so pressing the button
    twice never overwrites an earlier run.
    """
    if recorder is not None and recorder.active:
        summary = recorder.stop()
        print(f"recording stopped: {summary.get('frames', 0)} frames, "
              f"{summary.get('duration_s', 0):.1f} s, "
              f"{summary.get('images_saved', 0)} images -> {recorder.directory}")
        return None

    directory = out_root / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    new = RunRecorder(directory, config, save_every=save_every,
                      note="recorded from the UI")
    new.start(source_description)
    print(f"recording to {directory}")
    return new


def describe(result: SceneResult) -> str:
    subject = result.subject
    if subject is None:
        return "nothing in focus (no region has enough texture)"
    lead = ("sole candidate" if result.runner_up is None
            else f"lead={result.separation():+.3f}")
    parts = [
        f"IN FOCUS: {subject.label}",
        f"at ({subject.center[0]}, {subject.center[1]})",
        f"map={subject.map_score:.3f}",
        lead,
    ]
    if result.subject_result is not None:
        parts.append(f"score={result.subject_result.score:.3f}")
        parts.append(f"conf={result.subject_result.confidence:.2f}")
    return "  ".join(parts)


def run_headless(
    base: SharpnessConfig, state: UiState, frames: int, snapshot: Path | None
) -> int:
    config = state.apply(base)
    evaluator = SceneEvaluator(config)
    try:
        source = open_source(config.capture)
    except CaptureError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"source: {source.description}")
    print(f"switches: {state.summary()}")
    history: deque[float] = deque(maxlen=240)
    confidence_history: deque[float] = deque(maxlen=240)
    labels: list[str] = []
    totals: list[float] = []
    last_frame = None
    last_result = None
    started = time.perf_counter()
    try:
        for index in range(frames):
            frame = source.read()
            if frame is None:
                break
            result = evaluator.evaluate(frame)
            labels.append(result.subject_label)
            totals.append(result.total_time_s)
            score = result.subject_result.score if result.subject_result else 0.0
            conf = result.subject_result.confidence if result.subject_result else 0.0
            history.append(score)
            confidence_history.append(conf)
            last_frame, last_result = frame.data, result
            if index % 20 == 0:
                print(f"  [{index:4d}] {describe(result)}")
    finally:
        source.close()

    if not labels:
        print("no frames processed", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    print(f"\nframes            : {len(labels)}")
    print(f"throughput        : {len(labels) / elapsed:.2f} fps")
    print(f"total per frame   : {np.mean(totals) * 1e3:.2f} ms mean, "
          f"{np.percentile(totals, 95) * 1e3:.2f} ms p95")
    print("subject stability :")
    for label in sorted(set(labels), key=labels.count, reverse=True):
        print(f"    {label:24s} {labels.count(label):4d} frames "
              f"({100.0 * labels.count(label) / len(labels):.0f}%)")

    if snapshot is not None and last_frame is not None and last_result is not None:
        import cv2

        canvas = render(last_frame, last_result, state, len(labels) / elapsed, 0.0,
                        history, confidence_history)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(snapshot), canvas)
        print(f"\nsnapshot written to {snapshot} ({canvas.shape[1]}x{canvas.shape[0]})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None,
                        help="TOML config; defaults to the packaged one")
    parser.add_argument(
        "--backend", default=None,
        choices=["auto", "v4l2", "gphoto2", "file", "synthetic"],
    )
    parser.add_argument("--path", default=None)
    parser.add_argument("--history", type=int, default=240)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument(
        "--snapshot", type=Path, default=None,
        help="render the final frame's UI to this PNG (implies --headless)",
    )
    parser.add_argument("--no-faces", action="store_true", help="start with faces off")
    parser.add_argument(
        "--no-adaptive", action="store_true",
        help="start with fixed weights instead of adaptive ones",
    )
    parser.add_argument("--no-temporal", action="store_true")
    parser.add_argument(
        "--record-dir", type=Path, default=Path("data"),
        help="where the record button writes runs (default: data/)",
    )
    parser.add_argument(
        "--save-every", type=int, default=10,
        help="record every Nth frame as PNG",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    base = load_config(args.config) if args.config else load_default_config()
    overrides: dict[str, Any] = {}
    if args.backend:
        overrides["backend"] = args.backend
    if args.path:
        overrides["path"] = args.path
    if args.backend in ("synthetic", "file"):
        overrides["loop"] = True
    if overrides:
        base = base.with_overrides(capture=overrides)

    state = UiState(
        faces=not args.no_faces,
        adaptive=not args.no_adaptive,
        temporal=not args.no_temporal,
    )
    if base.focus_map.tile_size in TILE_SIZES:
        state.tile_index = TILE_SIZES.index(base.focus_map.tile_size)

    if args.headless or args.snapshot is not None:
        return run_headless(base, state, args.frames, args.snapshot)

    try:
        import cv2
    except ImportError:
        print("OpenCV is required", file=sys.stderr)
        return 2
    if not hasattr(cv2, "imshow"):
        print("This OpenCV build has no GUI support; use --headless", file=sys.stderr)
        return 2

    evaluator = SceneEvaluator(state.apply(base))
    try:
        source = open_source(base.capture)
    except CaptureError as exc:
        print(f"CAPTURE ERROR: {exc}", file=sys.stderr)
        return 2

    window = "What is in focus"
    try:
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    except cv2.error as exc:
        print(f"could not open a window ({exc}); use --headless", file=sys.stderr)
        source.close()
        return 2

    history: deque[float] = deque(maxlen=args.history)
    confidence_history: deque[float] = deque(maxlen=args.history)
    frame_times: deque[float] = deque(maxlen=30)
    paused = False
    last_frame = None
    last_result = None
    age_ms = 0.0
    previous = time.perf_counter()
    snapshots = 0
    recorder: RunRecorder | None = None
    record_root = args.record_dir
    save_every = max(1, args.save_every)

    def on_mouse(event: int, x: int, y: int, flags: int, param: Any) -> None:  # noqa: ARG001
        nonlocal recorder
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        rect = RECORD_BUTTON.get("canvas")
        if rect and rect[0] <= x <= rect[2] and rect[1] <= y <= rect[3]:
            recorder = toggle_recording(
                recorder, state.apply(base), record_root, save_every,
                source.description,
            )

    cv2.setMouseCallback(window, on_mouse)

    def rebuild() -> SceneEvaluator:
        """Recreate the evaluator after a switch changed the configuration."""
        history.clear()
        confidence_history.clear()
        logger.info("switches: %s", state.summary())
        return SceneEvaluator(state.apply(base))

    print(f"source: {source.description}")
    print("keys: q quit | 1-6 metrics | a adapt | n noise | m motion | c agree | "
          "e temporal | f faces | h heat | b boxes | g grid | t tiles | s save | "
          "R record | 0 reset")
    try:
        while True:
            if not paused:
                frame = source.read()
                if frame is None:
                    print("source exhausted")
                    break
                result = evaluator.evaluate(frame)
                age_ms = (time.perf_counter() - frame.timestamp) * 1e3
                now = time.perf_counter()
                frame_times.append(now - previous)
                previous = now
                score = result.subject_result.score if result.subject_result else 0.0
                conf = result.subject_result.confidence if result.subject_result else 0.0
                history.append(score)
                confidence_history.append(conf)
                last_frame, last_result = frame.data, result

                if recorder is not None and recorder.active and result.frame_result:
                    subject = result.subject
                    recorder.add(
                        result.frame_result, frame.data, frame.index,
                        extra={
                            "subject_label": result.subject_label,
                            "subject_x": "" if subject is None else subject.center[0],
                            "subject_y": "" if subject is None else subject.center[1],
                            "subject_map_score": "" if subject is None else subject.map_score,
                            "subject_score": (
                                "" if result.subject_result is None
                                else result.subject_result.filtered_score
                            ),
                            "subject_confidence": (
                                "" if result.subject_result is None
                                else result.subject_result.confidence
                            ),
                            "decision_margin": result.separation(),
                            "region_count": len(result.regions),
                            "map_valid_fraction": result.focus_map.valid_fraction,
                            "scene_time_s": result.scene_time_s,
                            "total_time_s": result.total_time_s,
                        },
                    )

            if last_frame is None or last_result is None:
                continue

            fps = 1.0 / (sum(frame_times) / len(frame_times)) if frame_times else 0.0
            canvas = render(
                last_frame, last_result, state, fps, age_ms,
                history, confidence_history,
                dropped=int(getattr(source, "dropped", 0)), paused=paused,
                recorder=recorder,
            )
            cv2.imshow(window, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if ord("1") <= key <= ord("6"):
                if state.toggle_metric(key - ord("1")):
                    evaluator = rebuild()
            elif key == ord("a"):
                state.adaptive = not state.adaptive
                evaluator = rebuild()
            elif key == ord("n"):
                state.noise_compensation = not state.noise_compensation
                evaluator = rebuild()
            elif key == ord("m"):
                state.motion_compensation = not state.motion_compensation
                evaluator = rebuild()
            elif key == ord("c"):
                state.agreement = not state.agreement
                evaluator = rebuild()
            elif key == ord("e"):
                state.temporal = not state.temporal
                evaluator = rebuild()
            elif key == ord("f"):
                state.faces = not state.faces
                evaluator = rebuild()
            elif key == ord("g"):
                state.tile_index = (state.tile_index + 1) % len(TILE_SIZES)
                evaluator = rebuild()
            elif key == ord("0"):
                state = UiState()
                evaluator = rebuild()
            elif key == ord("h"):
                state.heatmap = not state.heatmap
            elif key == ord("b"):
                state.boxes = not state.boxes
            elif key == ord("t"):
                state.grid = not state.grid
            elif key == ord("r"):
                evaluator.reset()
                history.clear()
                confidence_history.clear()
            elif key == ord(" "):
                paused = not paused
            elif key == ord("R"):
                recorder = toggle_recording(
                    recorder, state.apply(base), record_root, save_every,
                    source.description,
                )
            elif key == ord("s"):
                path = Path(f"focus_ui_{snapshots:03d}.png")
                cv2.imwrite(str(path), canvas)
                snapshots += 1
                print(f"saved {path}")
    finally:
        # A recording must be closed even on an exception, or the CSV is left
        # unflushed and the manifest never written.
        if recorder is not None and recorder.active:
            summary = recorder.stop()
            print(f"recording stopped: {summary.get('frames', 0)} frames "
                  f"-> {recorder.directory}")
        source.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
