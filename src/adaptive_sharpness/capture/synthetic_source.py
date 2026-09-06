"""A synthetic frame source that replays a simulated focus sweep.

It exists so that the full pipeline - including the demo and the data collector
- can be exercised end to end with no camera attached, and so that the
regression tests have a deterministic stream with a *known* best-focus position.
"""
from __future__ import annotations

import logging
import time

from ..types import Frame

from .base import FrameSource

logger = logging.getLogger(__name__)

__all__ = ["SyntheticSweepSource"]


class SyntheticSweepSource(FrameSource):
    """Replays a defocus ramp built by :mod:`tools.synthetic`."""

    def __init__(
        self,
        width: int = 640,
        height: int = 360,
        steps: int = 41,
        max_radius: float = 8.0,
        best_position: float = 0.5,
        noise_sigma: float = 0.0,
        exposure_drift: float = 0.0,
        motion_px: float = 0.0,
        seed: int = 7,
        loop: bool = True,
        pace_fps: float | None = None,
    ) -> None:
        self.width = width
        self.height = height
        self.steps = steps
        self.max_radius = max_radius
        self.best_position = best_position
        self.noise_sigma = noise_sigma
        self.exposure_drift = exposure_drift
        self.motion_px = motion_px
        self.seed = seed
        self.loop = loop
        # When set, read() sleeps so the source imitates a camera's frame rate.
        self.pace_fps = pace_fps
        self._frames: list = []
        self._position = 0
        self._index = 0
        self._last_read = 0.0
        self.description = "synthetic sweep (not opened)"

    def open(self) -> None:
        # Imported here so the capture package does not depend on the tools
        # package unless a synthetic source is actually used.
        from ..synthetic import focus_sweep, make_scene

        scene = make_scene(self.width, self.height, seed=self.seed)
        self._frames = list(
            focus_sweep(
                scene,
                steps=self.steps,
                best_position=self.best_position,
                max_radius=self.max_radius,
                noise_sigma=self.noise_sigma,
                exposure_drift=self.exposure_drift,
                motion_px=self.motion_px,
                seed=self.seed + 1,
            )
        )
        self._position = 0
        self.description = (
            f"synthetic sweep ({self.steps} steps, best at {self.best_position:.2f}, "
            f"max radius {self.max_radius} px)"
        )
        logger.info("opened %s", self.description)

    @property
    def best_focus_index(self) -> int:
        """Index of the in-focus frame, for tests that need ground truth."""
        return int(round(self.best_position * (self.steps - 1)))

    def read(self) -> Frame | None:
        if not self._frames:
            return None
        if self._position >= len(self._frames):
            if not self.loop:
                return None
            self._position = 0

        if self.pace_fps:
            interval = 1.0 / self.pace_fps
            elapsed = time.perf_counter() - self._last_read
            if 0.0 < elapsed < interval:
                time.sleep(interval - elapsed)
        self._last_read = time.perf_counter()

        sweep_frame = self._frames[self._position]
        self._position += 1
        frame = Frame(
            data=sweep_frame.image,
            index=self._index,
            timestamp=self._last_read,
            capture_latency_s=0.0,
            source="synthetic",
            meta={
                "motor_position": sweep_frame.motor_position,
                "defocus_radius": sweep_frame.defocus_radius,
                "sweep_index": sweep_frame.index,
            },
        )
        self._index += 1
        return frame

    def close(self) -> None:
        self._frames = []
        self._position = 0
