"""Tests for the innovation-gated temporal filter.

The central requirement is a tension: smooth the jitter, but never hide a real
focus change.  These tests pin both halves of that down.
"""
from __future__ import annotations

import numpy as np
import pytest

from adaptive_sharpness.config import TemporalConfig
from adaptive_sharpness.temporal import TemporalFilter


def feed_noise(
    filt: TemporalFilter, level: float, count: int, sigma: float, seed: int = 0
) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(count):
        out.append(filt.update(level + float(rng.normal(0.0, sigma))).value)
    return out


class TestBasics:
    def test_first_sample_passes_through(self) -> None:
        filt = TemporalFilter(TemporalConfig())
        assert filt.update(0.7).value == pytest.approx(0.7)

    def test_value_is_none_before_first_update(self) -> None:
        assert TemporalFilter(TemporalConfig()).value is None

    def test_output_is_clamped(self) -> None:
        filt = TemporalFilter(TemporalConfig())
        assert 0.0 <= filt.update(5.0).value <= 1.0
        assert 0.0 <= filt.update(-5.0).value <= 1.0

    def test_reset_clears_state(self) -> None:
        filt = TemporalFilter(TemporalConfig())
        filt.update(0.8)
        filt.reset()
        assert filt.value is None
        assert filt.update(0.2).value == pytest.approx(0.2)

    def test_disabled_filter_is_transparent(self) -> None:
        filt = TemporalFilter(TemporalConfig(enabled=False))
        for value in (0.1, 0.9, 0.2, 0.8):
            out = filt.update(value)
            assert out.value == pytest.approx(value)
            assert out.alpha == pytest.approx(1.0)


class TestSmoothing:
    def test_suppresses_jitter(self) -> None:
        """Around a stable level, the output must be quieter than the input."""
        filt = TemporalFilter(TemporalConfig())
        rng = np.random.default_rng(1)
        inputs = [0.5 + float(rng.normal(0, 0.02)) for _ in range(120)]
        outputs = [filt.update(v).value for v in inputs]
        assert np.std(outputs[20:]) < np.std(inputs[20:]) * 0.8

    def test_converges_to_a_constant(self) -> None:
        filt = TemporalFilter(TemporalConfig())
        filt.update(0.2)
        for _ in range(80):
            filt.update(0.9)
        assert filt.value == pytest.approx(0.9, abs=0.02)


class TestFocusChangeGate:
    def test_large_jump_is_not_suppressed(self) -> None:
        """A genuine focus transition must reach the output quickly."""
        filt = TemporalFilter(TemporalConfig())
        feed_noise(filt, 0.30, 40, 0.01, seed=2)
        before = filt.value
        out = filt.update(0.85)
        assert out.score_change_detected
        assert out.gate == pytest.approx(1.0)
        assert out.alpha > 0.9
        # Nearly the whole step lands in a single frame.
        assert out.value > before + 0.9 * (0.85 - before)

    def test_small_jitter_does_not_trip_the_gate(self) -> None:
        filt = TemporalFilter(TemporalConfig())
        results = []
        rng = np.random.default_rng(3)
        for _ in range(60):
            results.append(filt.update(0.5 + float(rng.normal(0, 0.01))))
        detections = sum(r.score_change_detected for r in results[10:])
        assert detections == 0

    def test_absolute_escape_hatch_on_a_quiet_history(self) -> None:
        """With a perfectly flat history the robust sigma collapses; the absolute
        jump threshold must still let a real change through."""
        config = TemporalConfig(gate_abs_jump=0.12)
        filt = TemporalFilter(config)
        for _ in range(30):
            filt.update(0.40)
        out = filt.update(0.40 + 0.15)
        assert out.score_change_detected
        assert out.alpha > 0.9

    def test_step_response_is_fast(self) -> None:
        """The filter must not add many frames of lag to a focus transition."""
        filt = TemporalFilter(TemporalConfig())
        feed_noise(filt, 0.2, 40, 0.008, seed=4)
        frames_to_settle = 0
        for _ in range(30):
            frames_to_settle += 1
            if filt.update(0.8).value > 0.78:
                break
        assert frames_to_settle <= 3, f"took {frames_to_settle} frames to follow a step"

    def test_a_plain_ema_would_be_slower(self) -> None:
        """Justifies the gate: the same alpha_base without gating lags badly."""
        alpha = TemporalConfig().alpha_base
        value = 0.2
        plain_frames = 0
        for _ in range(60):
            plain_frames += 1
            value += alpha * (0.8 - value)
            if value > 0.78:
                break

        gated = TemporalFilter(TemporalConfig())
        feed_noise(gated, 0.2, 40, 0.008, seed=5)
        gated_frames = 0
        for _ in range(60):
            gated_frames += 1
            if gated.update(0.8).value > 0.78:
                break
        assert gated_frames < plain_frames


class TestConfidenceCoupling:
    def test_low_confidence_smooths_harder(self) -> None:
        """With the gate closed, the gain must scale with the confidence.

        The history is warmed with realistic jitter first: against a perfectly
        flat history the robust sigma sits at its floor, every innovation looks
        enormous, the gate opens fully and the gain saturates at 1 regardless of
        confidence - which is correct behaviour, but tests nothing here.
        """
        config = TemporalConfig(confidence_coupling=True)
        alphas = {}
        for confidence in (1.0, 0.1):
            filt = TemporalFilter(config)
            feed_noise(filt, 0.5, 50, 0.02, seed=7)
            level = filt.value or 0.5
            out = filt.update(level + 0.005, confidence=confidence)
            assert out.gate < 0.5, "the gate should stay closed for a small step"
            alphas[confidence] = out.alpha
        assert alphas[0.1] < alphas[1.0]
        assert alphas[0.1] == pytest.approx(
            config.alpha_base * config.min_confidence_gain, rel=0.2
        )

    def test_low_confidence_still_lets_a_real_jump_through(self) -> None:
        """Confidence must scale the baseline gain only, never gate a real step."""
        filt = TemporalFilter(TemporalConfig(confidence_coupling=True))
        feed_noise(filt, 0.3, 40, 0.01, seed=6)
        out = filt.update(0.9, confidence=0.05)
        assert out.alpha > 0.9
        assert out.score_change_detected

    def test_coupling_can_be_disabled(self) -> None:
        config = TemporalConfig(confidence_coupling=False)
        a = TemporalFilter(config)
        b = TemporalFilter(config)
        a.update(0.5)
        b.update(0.5)
        assert a.update(0.6, confidence=1.0).alpha == pytest.approx(
            b.update(0.6, confidence=0.0).alpha
        )


class TestValidation:
    def test_rejects_bad_alpha(self) -> None:
        with pytest.raises(ValueError, match="alpha_base"):
            TemporalFilter(TemporalConfig(alpha_base=0.0))

    def test_rejects_inverted_gate(self) -> None:
        with pytest.raises(ValueError, match="gate_full_sigma"):
            TemporalFilter(TemporalConfig(gate_sigma=5.0, gate_full_sigma=2.0))

    def test_rejects_short_history(self) -> None:
        with pytest.raises(ValueError, match="history"):
            TemporalFilter(TemporalConfig(history=1))
