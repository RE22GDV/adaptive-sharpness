"""Scene-level evaluation: *what* is in focus, not just *how* sharp the frame is.

Three stages per frame:

1. a dense per-tile :class:`~sharpness.focusmap.FocusMap`, cheap enough to run
   over the whole frame;
2. region proposals (faces, sharp blobs) ranked by their focus-map score;
3. the full adaptive ensemble on the whole frame and on the winning region.

Why the ranking comes from the map and not from the ensemble
------------------------------------------------------------
The ensemble's score is normalised against a *rolling history* of previous
frames.  That is right for tracking one region over time and wrong for comparing
several regions inside one frame: each region would need its own history, those
histories would be built from different content, and the resulting numbers would
not be comparable.  Worse, regions come and go between frames, so a per-region
history has nothing stable to attach to.

The focus map avoids all of that: it is rescaled *within the frame*, so
comparing tile against tile is exactly the comparison it supports.  The full
ensemble is then run only on the winner, where a persistent history does make
sense - and it is reset when the subject changes identity, tracked by
intersection-over-union between frames.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .config import SharpnessConfig
from .evaluator import SharpnessEvaluator
from .focusmap import FocusMap, FocusMapper
from .preprocess import Preprocessor
from .regions import Region, RegionProposer
from .types import Frame, SharpnessResult

logger = logging.getLogger(__name__)

__all__ = ["SceneResult", "SceneEvaluator"]


@dataclass(frozen=True)
class SceneResult:
    """Everything known about one frame's focus distribution."""

    #: Per-tile focus map over the whole frame.
    focus_map: FocusMap
    #: Candidate subjects, best first.
    regions: Sequence[Region] = field(default_factory=tuple)
    #: The region judged to be in focus, or ``None`` if nothing qualified.
    subject: Region | None = None
    #: Full ensemble result for the subject region.
    subject_result: SharpnessResult | None = None
    #: Full ensemble result for the whole frame.
    frame_result: SharpnessResult | None = None
    #: True on the frame where the subject changed identity.
    subject_changed: bool = False
    #: Seconds spent in the scene stage, on top of the ensemble evaluations.
    scene_time_s: float = 0.0
    #: Total seconds for the whole call.
    total_time_s: float = 0.0

    @property
    def subject_label(self) -> str:
        if self.subject is None:
            return "nothing in focus"
        return self.subject.label

    @property
    def has_subject(self) -> bool:
        return self.subject is not None

    @property
    def runner_up(self) -> Region | None:
        return self.regions[1] if len(self.regions) > 1 else None

    def separation(self) -> float:
        """Gap between the best and second-best region scores.

        A small gap means the claim about which object is in focus is weakly
        supported, even when each individual score is measured confidently.

        Returns 0.0 when there is no runner-up: a single candidate is not
        evidence of a wide margin, it is an absence of comparison.  Check
        :attr:`runner_up` to tell the two situations apart.
        """
        if len(self.regions) < 2:
            return 0.0
        return float(self.regions[0].map_score - self.regions[1].map_score)


class SceneEvaluator:
    """Locates the in-focus subject and scores it in full.

    Holds per-stream state (two evaluators, the motion reference, the face
    cache), so construct one per stream and do not share it between threads.
    """

    def __init__(self, config: SharpnessConfig | None = None) -> None:
        self.config = config or SharpnessConfig()
        self._preprocessor = Preprocessor(self.config.pipeline)
        self._mapper = FocusMapper(self.config.focus_map)
        self._proposer = RegionProposer(self.config.regions)
        # One evaluator for the whole frame, one for the tracked subject.
        self._frame_evaluator = SharpnessEvaluator(self.config)
        self._subject_evaluator = SharpnessEvaluator(self.config)
        self._previous_subject: Region | None = None

    @property
    def metric_names(self) -> tuple[str, ...]:
        return self._frame_evaluator.metric_names

    def reset(self) -> None:
        self._frame_evaluator.reset()
        self._subject_evaluator.reset()
        self._proposer.reset()
        self._previous_subject = None

    def _same_subject(self, current: Region) -> bool:
        previous = self._previous_subject
        if previous is None:
            return False
        if previous.iou(current) >= self.config.regions.subject_iou_match:
            return True
        # A face that moved fast can miss on IoU while still being the same
        # subject; accept a same-label detection whose centre barely moved.
        if previous.label == current.label and previous.source == current.source:
            px, py = previous.center
            cx, cy = current.center
            travel = float(np.hypot(cx - px, cy - py))
            return travel < 0.5 * max(previous.width, previous.height)
        return False

    def evaluate(
        self,
        frame: Frame | np.ndarray,
        motor_position: float | None = None,
    ) -> SceneResult:
        """Evaluate one frame and report which region is in focus."""
        started = time.perf_counter()
        data = frame.data if isinstance(frame, Frame) else frame

        # The focus map and the proposals share one analysis image.
        prepared = self._preprocessor.prepare(data, None)
        focus_map = self._mapper.compute(prepared.gray, prepared.scale)
        regions = self._proposer.propose(focus_map, prepared.gray, data.shape)
        scene_time = time.perf_counter() - started

        frame_result = self._frame_evaluator.evaluate(frame, None, motor_position)

        subject = regions[0] if regions else None
        subject_result = None
        subject_changed = False
        if subject is not None:
            subject_changed = not self._same_subject(subject)
            if subject_changed:
                # A different subject has a different raw metric range; keeping
                # the old normalisation history would corrupt the score for
                # several frames.
                self._subject_evaluator.reset()
                logger.debug("subject changed to %s", subject.label)
            subject_result = self._subject_evaluator.evaluate(
                frame, subject.to_roi(), motor_position
            )
        self._previous_subject = subject

        return SceneResult(
            focus_map=focus_map,
            regions=tuple(regions),
            subject=subject,
            subject_result=subject_result,
            frame_result=frame_result,
            subject_changed=subject_changed,
            scene_time_s=scene_time,
            total_time_s=time.perf_counter() - started,
        )
