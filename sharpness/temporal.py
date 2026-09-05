"""Innovation-gated temporal smoothing of the sharpness score.

A plain exponential moving average is the wrong tool for an autofocus loop: the
smoothing that suppresses frame-to-frame noise also delays and attenuates the
one thing the loop is looking for — a genuine change of focus.

This filter keeps the EMA form but makes its gain depend on how surprising the
new sample is:

    z          = |s_t - s_hat_{t-1}| / sigma_r
    g          = smoothstep(z; gate_sigma, gate_full_sigma)     in [0, 1]
    alpha_base'= alpha_base * confidence_gain
    alpha_t    = alpha_base' + (1 - alpha_base') * g
    s_hat_t    = s_hat_{t-1} + alpha_t * (s_t - s_hat_{t-1})

``sigma_r`` is a robust (MAD) estimate of the recent frame-to-frame variation,
so the gate measures surprise in units of the score's own noise rather than in
absolute units that would need retuning per scene.  When ``g`` reaches 1 the
filter is fully transparent — a real focus transition passes through with no
lag at all — and when the score is merely jittering, ``g`` stays near 0 and the
full smoothing applies.

An absolute escape hatch (``gate_abs_jump``) covers the case of a very quiet
score history, where ``sigma_r`` would otherwise be so small that ordinary
jitter looks significant.

Confidence coupling scales only the *baseline* gain, never the gated term, so
an unreliable frame is smoothed harder while a genuine large jump is still
never suppressed.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np

from .config import TemporalConfig

logger = logging.getLogger(__name__)

__all__ = ["TemporalFilterOutput", "TemporalFilter"]

# Smallest robust scale used in the gate, to avoid dividing by ~0 on a frozen
# score. Expressed in normalised score units.
_MIN_SIGMA = 5e-3


@dataclass(frozen=True)
class TemporalFilterOutput:
    """Result of one filter update."""

    value: float
    alpha: float
    gate: float
    innovation: float
    sigma: float
    focus_change_detected: bool


def _smoothstep(x: float) -> float:
    """Hermite smoothstep on [0, 1]; 0 and 1 have zero derivative."""
    x = min(1.0, max(0.0, x))
    return x * x * (3.0 - 2.0 * x)


class TemporalFilter:
    """Adaptive-gain EMA over the ensemble score."""

    def __init__(self, config: TemporalConfig) -> None:
        if not 0.0 < config.alpha_base <= 1.0:
            raise ValueError("alpha_base must lie in (0, 1]")
        if config.gate_full_sigma <= config.gate_sigma:
            raise ValueError("gate_full_sigma must be greater than gate_sigma")
        if config.history < 2:
            raise ValueError("history must be at least 2")
        self.config = config
        self._value: float | None = None
        self._deltas: deque[float] = deque(maxlen=config.history)

    @property
    def value(self) -> float | None:
        """Current filtered score, or ``None`` before the first update."""
        return self._value

    def reset(self) -> None:
        """Drop all state (use when the stream restarts or the ROI changes)."""
        self._value = None
        self._deltas.clear()

    def _sigma(self) -> float:
        """Robust scale of the recent frame-to-frame variation."""
        if len(self._deltas) < 2:
            return _MIN_SIGMA
        data = np.fromiter(self._deltas, dtype=np.float64, count=len(self._deltas))
        # 1.4826 * MAD, taken about zero since these are already differences.
        sigma = 1.4826 * float(np.median(np.abs(data)))
        return max(sigma, _MIN_SIGMA)

    def update(self, score: float, confidence: float = 1.0) -> TemporalFilterOutput:
        """Filter one score sample.

        ``confidence`` in ``[0, 1]`` scales the baseline gain when
        ``confidence_coupling`` is enabled.
        """
        score = float(min(1.0, max(0.0, score)))
        cfg = self.config

        if not cfg.enabled:
            self._value = score
            return TemporalFilterOutput(score, 1.0, 1.0, 0.0, _MIN_SIGMA, False)

        if self._value is None:
            self._value = score
            return TemporalFilterOutput(score, 1.0, 0.0, 0.0, _MIN_SIGMA, False)

        innovation = score - self._value
        magnitude = abs(innovation)
        sigma = self._sigma()

        z = magnitude / sigma
        span = cfg.gate_full_sigma - cfg.gate_sigma
        gate = _smoothstep((z - cfg.gate_sigma) / span)
        if magnitude >= cfg.gate_abs_jump:
            # A large absolute step is a focus change regardless of how quiet
            # the recent history has been.
            gate = 1.0

        alpha_base = cfg.alpha_base
        if cfg.confidence_coupling:
            gain = max(cfg.min_confidence_gain, min(1.0, float(confidence)))
            alpha_base *= gain
        alpha = alpha_base + (1.0 - alpha_base) * gate
        alpha = min(1.0, max(0.0, alpha))

        self._value += alpha * innovation
        # The history tracks the *observed* variation, not the filtered one, so
        # the gate keeps a faithful picture of the measurement noise.
        self._deltas.append(innovation)

        detected = gate >= 0.5
        if detected:
            logger.debug(
                "focus change: innovation=%.4f sigma=%.4f z=%.2f alpha=%.2f",
                innovation, sigma, z, alpha,
            )

        return TemporalFilterOutput(
            value=float(self._value),
            alpha=float(alpha),
            gate=float(gate),
            innovation=float(innovation),
            sigma=float(sigma),
            focus_change_detected=detected,
        )
