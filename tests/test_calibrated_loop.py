"""The complete usage sequence, checked.

`demo/calibrated_focus_loop.py` is the answer to "how do I actually use this":
calibrate, verify, freeze, compare, reset. The measurements say that sequence
is where the result is won - freezing the scale is worth +0.46 in rank
correlation (Д11) and a different warm-up moves the same frame's score by up to
0.64 (Д10) - so the sequence needs a test rather than only a paragraph.

The heavy check runs against a real recording and skips when there is none, so
this passes in CI, where the frames are absent by design: they are photographs
of a private room.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "demo",):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

from adaptive_sharpness import SharpnessEvaluator, load_default_config  # noqa: E402

from calibrated_focus_loop import (  # noqa: E402
    Calibration,
    calibrate,
    compare_positions,
    hold_the_scale,
    scene_changed,
)


def _textured(seed: int, blur: int = 0) -> np.ndarray:
    """A frame with structure, optionally softened."""
    import cv2

    rng = np.random.default_rng(seed)
    image = rng.integers(0, 255, (180, 320, 3), dtype=np.uint8)
    if blur:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    return image


def _sweep(count: int = 400) -> list[np.ndarray]:
    """A pass that goes out of focus and back, which is what calibration is."""
    radii = list(range(6, 0, -1)) + [0] + list(range(1, 7))
    return [
        _textured(index, radii[index % len(radii)]) for index in range(count)
    ]


class TestUsability:
    """`Calibration.usable` decides whether the pass may be frozen on."""

    def test_a_pass_that_never_moved_is_refused(self) -> None:
        """Freezing on a still pass locks in a scale that cannot separate
        anything afterwards, which is the failure mode the span guards."""
        still = Calibration(
            frames_seen=500, ready_fraction=1.0, informative_fraction=1.0,
            frozen=False, span=0.02,
        )
        assert not still.usable
        assert any("spanned only" in reason for reason in still.why_not())

    def test_a_pass_that_never_became_ready_is_refused(self) -> None:
        cold = Calibration(
            frames_seen=50, ready_fraction=0.1, informative_fraction=1.0,
            frozen=False, span=0.9,
        )
        assert not cold.usable
        assert any("ready" in reason for reason in cold.why_not())

    def test_a_good_pass_is_accepted(self) -> None:
        good = Calibration(
            frames_seen=500, ready_fraction=0.99, informative_fraction=1.0,
            frozen=False, span=0.9,
        )
        assert good.usable
        assert good.why_not() == []

    def test_auto_freeze_is_not_required(self) -> None:
        """Measured on a real sweep: 781 frames, ready on 99%, and the
        automatic rule never fired - because during a sweep the metric range
        grows the whole time, which is what the sweep is for. Requiring it here
        would reject every valid calibration pass."""
        good = Calibration(
            frames_seen=781, ready_fraction=0.99, informative_fraction=1.0,
            frozen=False, span=0.9,
        )
        assert good.usable


class TestSequence:
    def test_calibrating_a_sweep_produces_a_usable_scale(self) -> None:
        evaluator = SharpnessEvaluator(load_default_config())
        calibration = calibrate(evaluator, _sweep())
        assert calibration.frames_seen == 400
        assert calibration.usable, calibration.why_not()

    def test_the_explicit_freeze_takes(self) -> None:
        evaluator = SharpnessEvaluator(load_default_config())
        calibrate(evaluator, _sweep())
        hold_the_scale(evaluator)
        assert evaluator.evaluate(_textured(1, 0)).scale_frozen

    def test_a_frozen_scale_normalises_a_frame_identically(self) -> None:
        """The property the sequence exists to obtain, stated exactly.

        Once frozen, the *normalised value of each metric* for a given frame is
        bit-identical however many other frames pass through in between. That
        is the scale holding still, and it is what makes two positions
        comparable.
        """
        evaluator = SharpnessEvaluator(load_default_config())
        calibrate(evaluator, _sweep())
        hold_the_scale(evaluator)

        probe = _textured(999, 0)
        first = evaluator.evaluate(probe)
        for index in range(60):
            evaluator.evaluate(_textured(2000 + index, index % 7))
        second = evaluator.evaluate(probe)

        for before, after in zip(first.metrics, second.metrics):
            assert before.normalized == pytest.approx(after.normalized, abs=1e-12), (
                f"{before.name} normalised differently under a frozen scale"
            )

    def test_the_weights_still_depend_on_frame_order(self) -> None:
        """And the part that does *not* hold still, with its cause.

        The ensemble score moves slightly even with the scale frozen, by about
        0.0015 here. The normalised values above are identical, so it is not
        the scale: it is the adaptive weights. Motion is estimated by phase
        correlation against the *previous* frame, so the same frame preceded by
        a different one gets a different motion estimate - 13.3 px against 62.1
        px in this test - and therefore slightly different weights.

        It is small, but it means the adaptive path cannot give bit-exact
        repeatability for one frame. Fixed weights can, and study Д07 found no
        measurable quality difference between the two, so that is a real option
        for a caller who needs determinism.
        """
        evaluator = SharpnessEvaluator(load_default_config())
        calibrate(evaluator, _sweep())
        hold_the_scale(evaluator)

        probe = _textured(999, 0)
        first = evaluator.evaluate(probe)
        for index in range(60):
            evaluator.evaluate(_textured(2000 + index, index % 7))
        second = evaluator.evaluate(probe)

        drift = abs(first.instantaneous_score - second.instantaneous_score)
        assert drift < 0.01, f"weights moved the score by {drift}, more than expected"
        assert first.stats.motion_px != second.stats.motion_px, (
            "this test is meaningless if the motion estimate did not change"
        )
        # The normalised values are identical, so the difference is the weights.
        weights_before = [m.weight for m in first.metrics]
        weights_after = [m.weight for m in second.metrics]
        assert weights_before != weights_after

    def test_reset_puts_the_score_on_a_different_scale(self) -> None:
        """The counterpart: after a reset the same frame is not comparable.

        On a real recording this reads 0.157 before the reset and 0.500 after -
        0.5 being the neutral sentinel for "no informative range yet". The
        sequence documents this so that nobody compares across a reset.
        """
        evaluator = SharpnessEvaluator(load_default_config())
        calibrate(evaluator, _sweep())
        hold_the_scale(evaluator)

        probe = _textured(1234, 3)
        before = evaluator.evaluate(probe).instantaneous_score
        scene_changed(evaluator)
        after = evaluator.evaluate(probe).instantaneous_score
        assert not evaluator.evaluate(probe).scale_frozen
        assert abs(before - after) > 1e-3, (
            "a reset should put the score on a different scale"
        )

    def test_comparison_returns_one_score_per_position(self) -> None:
        evaluator = SharpnessEvaluator(load_default_config())
        calibrate(evaluator, _sweep())
        hold_the_scale(evaluator)
        positions = [(float(i), _textured(500 + i, i % 5)) for i in range(8)]
        scored = compare_positions(evaluator, positions)
        assert [p for p, _ in scored] == [float(i) for i in range(8)]
        assert all(0.0 <= s <= 1.0 for _, s in scored)


def _recordings() -> list[Path]:
    roots = [_ROOT / "data", Path("D:/camera_focus_data")]
    out: list[Path] = []
    for root in roots:
        if root.exists():
            out += [d for d in sorted(root.iterdir()) if (d / "frames.csv").exists()]
    return out


class TestAgainstARealRecording:
    def test_the_sequence_runs_end_to_end(self) -> None:
        """The whole point of a documented sequence is that it runs."""
        from calibrated_focus_loop import run_sequence

        candidates = [
            d for d in _recordings()
            if d.name.startswith(("sweep_", "cond_", "point_source_"))
        ]
        if not candidates:
            pytest.skip("no recordings available; frames are not in the repository")
        assert run_sequence(candidates[-1]) == 0


class TestDocumentedExamples:
    """The integration guide's code has to work against the installed package.

    It did not: every example imported `from sharpness`, the package name from
    the very first commit, and read `result.score`, a field that has not
    existed since the API split the instantaneous and filtered scores. A reader
    following the guide would have hit an ImportError on line one.
    """

    def _guide(self) -> str:
        return (_ROOT / "docs" / "INTEGRATION.md").read_text(encoding="utf-8")

    def test_no_example_uses_the_old_package_name(self) -> None:
        assert "from sharpness" not in self._guide()

    def test_no_example_uses_the_removed_score_field(self) -> None:
        import re

        assert not re.search(r"\bresult\.score\b", self._guide())

    def test_every_documented_import_resolves(self) -> None:
        import importlib
        import re

        problems: list[str] = []
        for match in re.finditer(
            r"^from (adaptive_sharpness[\w.]*) import ([^\n]+)$",
            self._guide(), re.M,
        ):
            module_name, names = match.group(1), match.group(2)
            try:
                module = importlib.import_module(module_name)
            except Exception as error:  # noqa: BLE001 - reported, not raised
                problems.append(f"{module_name}: {error}")
                continue
            for name in (n.strip() for n in names.split(",")):
                if not hasattr(module, name):
                    problems.append(f"{module_name} has no {name}")
        assert not problems, problems

    def test_the_guide_points_at_the_worked_sequence(self) -> None:
        assert "calibrated_focus_loop.py" in self._guide()
