"""One complete usage sequence, end to end, with the checks that make it work.

Calibrate -> verify the scale is usable -> freeze it -> compare focus
positions -> reset when the scene or ROI changes.

Why this file exists
--------------------
`README.md` shows how to score a frame and `docs/INTEGRATION.md` shows how to
drive a servo, but nothing showed the sequence around them, and the sequence is
where the measurements say the result is won or lost:

*   Study Д11 measured that freezing the normalisation scale is worth **+0.46**
    in rank correlation with a physical reference - 0.50 without it, 0.97 with
    it.  It is the largest effect anywhere in this project.
*   Study Д10 measured that the *same frames*, scored after a different warm-up,
    differ by up to **0.64** with rank agreement falling to 0.56.  Comparing two
    positions across a reset, or before the scale settles, compares two
    different scales.

So a caller who evaluates frames without calibrating first is not using a
worse configuration of this library - they are reading numbers off a ruler
whose markings are still moving.

Run it against a recording:

    python3 demo/calibrated_focus_loop.py --recording data/sweep_texture_...

It prints a short report and exits non-zero if the sequence did not hold, which
is what `tests/test_calibrated_loop.py` checks on a real recording.
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src",):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import (  # noqa: E402
    ROI,
    SharpnessEvaluator,
    load_default_config,
)


@dataclass
class Calibration:
    """What a calibration pass concluded, and whether it is usable."""

    frames_seen: int
    ready_fraction: float
    informative_fraction: float
    #: Whether `auto_freeze` fired during the pass.  **Expect False here**, and
    #: that is not a fault: the automatic rule waits for the metric range to
    #: stop growing, and during a deliberate sweep through focus the range grows
    #: the whole time - that is what the sweep is for.  `auto_freeze` is for a
    #: stream that settles on its own; a calibration pass freezes explicitly,
    #: which is step 3.  Measured on `sweep_texture_20260916_110108`: 781 frames
    #: of sweep, ready on 99% of them, and the automatic rule never fired.
    frozen: bool
    span: float

    @property
    def usable(self) -> bool:
        """Is the pass good enough to freeze a scale on?

        Three conditions, and all three are necessary:

        ``ready_fraction``
            The evaluator's own readiness flag, which requires warm-up to have
            finished *and* at least half the metrics to be informative.
        ``informative_fraction``
            How many metrics had a usable range.  A metric with none returns
            the 0.5 sentinel, which is not a measurement.
        ``span``
            The pass has to have contained a real focus change.  Freezing on a
            pass where nothing moved locks in a scale that cannot distinguish
            anything afterwards.

        ``frozen`` is deliberately **not** one of them - see the note on the
        field below.
        """
        return (
            self.ready_fraction >= 0.9
            and self.informative_fraction >= 0.5
            and self.span >= 0.2
        )

    def why_not(self) -> list[str]:
        reasons = []
        if self.ready_fraction < 0.9:
            reasons.append(f"only {self.ready_fraction:.0%} of frames were ready")
        if self.informative_fraction < 0.5:
            reasons.append(
                f"only {self.informative_fraction:.0%} of metrics were informative"
            )
        if self.span < 0.2:
            reasons.append(
                f"the calibration pass spanned only {self.span:.2f} of the "
                "score range; it needs to sweep through focus, not sit still"
            )
        return reasons


def calibrate(
    evaluator: SharpnessEvaluator,
    frames: Iterable[np.ndarray],
    roi: ROI | None = None,
) -> Calibration:
    """Step 1 and 2: sweep through focus, then ask whether the scale is usable.

    The caller drives the lens across its range while this consumes the frames.
    Nothing here decides anything about focus - the pass exists only to show
    the normaliser what range of values this scene produces.
    """
    scores: list[float] = []
    ready: list[bool] = []
    informative: list[float] = []
    frozen = False

    for frame in frames:
        result = evaluator.evaluate(frame, roi)
        scores.append(result.instantaneous_score)
        ready.append(bool(result.ready))
        informative.append(result.informative_fraction)
        frozen = frozen or bool(result.scale_frozen)

    if not scores:
        return Calibration(0, 0.0, 0.0, False, 0.0)
    return Calibration(
        frames_seen=len(scores),
        ready_fraction=float(np.mean(ready)),
        informative_fraction=float(np.mean(informative)),
        frozen=frozen,
        span=float(np.max(scores) - np.min(scores)),
    )


def hold_the_scale(evaluator: SharpnessEvaluator) -> None:
    """Step 3: freeze explicitly, rather than waiting for the automatic rule.

    `auto_freeze` fires on its own after enough stable observations, and on a
    calibration pass you know you are done, so there is no reason to wait.
    Freezing recomputes the anchors from the whole history rather than reusing
    whatever the automatic rule last stored.
    """
    evaluator.normalizers.freeze()


def compare_positions(
    evaluator: SharpnessEvaluator,
    positions: Sequence[tuple[float, np.ndarray]],
    roi: ROI | None = None,
) -> list[tuple[float, float]]:
    """Step 4: score candidate positions on one frozen scale.

    This is the only step whose numbers may be compared with each other, and
    they may be compared *only* with each other: they belong to this scale, on
    this scene, in this evaluator.
    """
    out = []
    for position, frame in positions:
        result = evaluator.evaluate(frame, roi)
        out.append((position, result.filtered_score))
    return out


def scene_changed(evaluator: SharpnessEvaluator) -> None:
    """Step 5: the scene or the ROI changed, so the scale no longer applies.

    `reset()` clears the history and the freeze.  Everything measured before it
    is on a different scale from everything measured after, and the two must not
    be compared - which means recalibrating before the next comparison.
    """
    evaluator.reset()


# ---------------------------------------------------------------------------
# Running the sequence against a recording
# ---------------------------------------------------------------------------

def load_recording(directory: Path) -> tuple[list[np.ndarray], np.ndarray]:
    """Frames and their protocol step, in capture order."""
    rows = list(csv.DictReader((directory / "frames.csv").open(encoding="utf-8")))
    images, steps = [], []
    for row in rows:
        image = cv2.imread(
            str(directory / "frames" / row["image_file"]), cv2.IMREAD_COLOR
        )
        if image is None:
            continue
        images.append(image)
        # `step_index` is the operator's protocol step; `step_phase` says
        # whether the ring was being moved or held.  Only held frames describe
        # a position, so a moving frame gets no label.
        held = (row.get("step_phase") or "").strip().lower() == "hold"
        steps.append(float(row.get("step_index") or "nan") if held else float("nan"))
    return images, np.asarray(steps, dtype=float)


def run_sequence(directory: Path) -> int:
    images, steps = load_recording(directory)
    if len(images) < 200:
        print(f"{directory.name}: too few frames ({len(images)})")
        return 1

    evaluator = SharpnessEvaluator(load_default_config())

    # 1 + 2 - calibrate on the first half, then check.
    half = len(images) // 2
    calibration = calibrate(evaluator, images[:half])
    print(f"calibration over {calibration.frames_seen} frames")
    print(f"  ready       {calibration.ready_fraction:.0%}")
    print(f"  informative {calibration.informative_fraction:.0%}")
    print(f"  span        {calibration.span:.2f}")
    print(f"  auto-froze  {calibration.frozen}"
          "   <- False is normal during a sweep")
    if not calibration.usable:
        for reason in calibration.why_not():
            print(f"  NOT USABLE: {reason}")
        return 2

    # 3 - hold it, and check that it took.
    hold_the_scale(evaluator)
    check = evaluator.evaluate(images[half - 1])
    if not check.scale_frozen:
        print("  freeze did not take; refusing to compare on a moving scale")
        return 4
    print("  scale frozen explicitly"
          + ("" if calibration.frozen else " (auto_freeze had not fired)"))

    # 4 - compare the remaining frames on that frozen scale.
    rest = [(steps[i], images[i]) for i in range(half, len(images))]
    scored = compare_positions(evaluator, rest)
    usable = [(p, s) for p, s in scored if np.isfinite(p)]
    if not usable:
        print("no labelled positions in the second half")
        return 3
    best_position = max(usable, key=lambda item: item[1])[0]
    print(f"comparison over {len(usable)} labelled frames")
    print(f"  best step {best_position:.0f}")

    # 5 - a scene change invalidates the scale.
    before = evaluator.evaluate(images[-1]).instantaneous_score
    scene_changed(evaluator)
    after = evaluator.evaluate(images[-1]).instantaneous_score
    print(f"same frame before reset {before:.3f}, immediately after {after:.3f}")
    print("  (these are on different scales and must not be compared)")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recording", type=Path, required=True)
    args = parser.parse_args(argv)
    if not (args.recording / "frames.csv").exists():
        parser.error(f"not a recording: {args.recording}")
    return run_sequence(args.recording)


if __name__ == "__main__":
    raise SystemExit(main())
