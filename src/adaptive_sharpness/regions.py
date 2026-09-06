"""Region proposals: the candidate "objects" whose focus is compared.

Two detectors, both offline and dependency-free:

``faces``
    OpenCV's bundled Haar cascade.  It gives a *named* subject, which is what
    makes a display legible - "the face is in focus" beats "tile (3, 7) is in
    focus".  It is run on the analysis image and only every ``face_interval``
    frames, because the cascade is by far the most expensive thing here.
``sharp_blobs``
    Connected components of the focus map above a fraction of its own peak.
    This needs no model and finds whatever is actually sharp, including objects
    no classifier knows about.  It is the fallback that always works.

Proposals are merged by intersection-over-union so that a face and the sharp
blob covering it are reported once, with the face's label kept - a named region
is more useful to a viewer than an anonymous one.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from .config import RegionsConfig
from .focusmap import FocusMap
from .types import ROI

logger = logging.getLogger(__name__)

__all__ = ["Region", "RegionProposer"]


@dataclass(frozen=True)
class Region:
    """A candidate subject, in source-frame pixel coordinates."""

    x: int
    y: int
    width: int
    height: int
    #: Human-readable name for the display, e.g. "face" or "sharp area".
    label: str = "region"
    #: Which detector produced it.
    source: str = "unknown"
    #: Mean focus-map score over the region, in [0, 1]. The within-frame
    #: ranking key.
    map_score: float = 0.0
    #: Fraction of the region's tiles that carried usable texture.
    map_coverage: float = 0.0
    #: Peak tile score inside the region.
    map_peak: float = 0.0

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return self.x + self.width // 2, self.y + self.height // 2

    def to_roi(self) -> ROI:
        return ROI(self.x, self.y, self.width, self.height)

    def iou(self, other: "Region") -> float:
        """Intersection over union with another region."""
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        intersection = (ix2 - ix1) * (iy2 - iy1)
        union = self.area + other.area - intersection
        return intersection / union if union > 0 else 0.0


class RegionProposer:
    """Produces ranked :class:`Region` candidates for one frame."""

    def __init__(self, config: RegionsConfig) -> None:
        self.config = config
        self._cascade: cv2.CascadeClassifier | None = None
        self._cascade_failed = False
        self._face_cache: list[Region] = []
        self._frame_counter = 0

    # -- face detection --------------------------------------------------------

    def _load_cascade(self) -> cv2.CascadeClassifier | None:
        if self._cascade is not None or self._cascade_failed:
            return self._cascade
        try:
            path = cv2.data.haarcascades + self.config.face_cascade
        except AttributeError:
            logger.warning("this OpenCV build ships no cascade data; faces disabled")
            self._cascade_failed = True
            return None
        cascade = cv2.CascadeClassifier(path)
        if cascade.empty():
            logger.warning("could not load cascade %s; faces disabled", path)
            self._cascade_failed = True
            return None
        self._cascade = cascade
        logger.debug("loaded face cascade from %s", path)
        return cascade

    def detect_faces(self, gray: np.ndarray, scale_to_source: float) -> list[Region]:
        """Detect faces on the analysis image, returned in source coordinates."""
        cascade = self._load_cascade()
        if cascade is None:
            return []
        u8 = np.clip(gray, 0, 255).astype(np.uint8)
        min_side = int(min(u8.shape) * self.config.face_min_size_fraction)
        try:
            boxes = cascade.detectMultiScale(
                u8,
                scaleFactor=self.config.face_scale_factor,
                minNeighbors=self.config.face_min_neighbors,
                minSize=(max(12, min_side), max(12, min_side)),
            )
        except cv2.error as exc:  # pragma: no cover - degenerate frames only
            logger.debug("cascade failed: %s", exc)
            return []
        regions = []
        for index, (x, y, w, h) in enumerate(boxes):
            regions.append(
                Region(
                    x=int(round(x * scale_to_source)),
                    y=int(round(y * scale_to_source)),
                    width=max(1, int(round(w * scale_to_source))),
                    height=max(1, int(round(h * scale_to_source))),
                    label=f"face {index + 1}" if len(boxes) > 1 else "face",
                    source="faces",
                )
            )
        return regions

    # -- focus-map blobs -------------------------------------------------------

    def detect_sharp_blobs(self, focus_map: FocusMap) -> list[Region]:
        """Connected components of the focus map above a fraction of its peak."""
        valid = focus_map.valid
        if not np.any(valid):
            return []
        peak = float(np.max(np.where(valid, focus_map.score, 0.0)))
        if peak <= 0.0:
            return []
        threshold = peak * self.config.blob_relative_threshold
        mask = ((focus_map.score >= threshold) & valid).astype(np.uint8)
        if not np.any(mask):
            return []

        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        tile = focus_map.tile_size / max(focus_map.source_scale, 1e-6)
        regions: list[Region] = []
        for index in range(1, count):
            area_tiles = int(stats[index, cv2.CC_STAT_AREA])
            if area_tiles < self.config.blob_min_tiles:
                continue
            col = int(stats[index, cv2.CC_STAT_LEFT])
            row = int(stats[index, cv2.CC_STAT_TOP])
            cols = int(stats[index, cv2.CC_STAT_WIDTH])
            rows = int(stats[index, cv2.CC_STAT_HEIGHT])
            regions.append(
                Region(
                    x=int(round(col * tile)),
                    y=int(round(row * tile)),
                    width=max(1, int(round(cols * tile))),
                    height=max(1, int(round(rows * tile))),
                    label="sharp area",
                    source="sharp_blobs",
                )
            )
        return regions

    # -- scoring and merging ---------------------------------------------------

    def score_region(self, region: Region, focus_map: FocusMap) -> Region:
        """Attach the focus-map statistics of the tiles a region covers."""
        tile = focus_map.tile_size / max(focus_map.source_scale, 1e-6)
        rows, cols = focus_map.shape
        c0 = max(0, min(cols - 1, int(region.x // tile)))
        r0 = max(0, min(rows - 1, int(region.y // tile)))
        c1 = max(c0 + 1, min(cols, int(np.ceil(region.x2 / tile))))
        r1 = max(r0 + 1, min(rows, int(np.ceil(region.y2 / tile))))

        window = focus_map.score[r0:r1, c0:c1]
        window_valid = focus_map.valid[r0:r1, c0:c1]
        coverage = float(np.count_nonzero(window_valid)) / float(max(1, window_valid.size))
        if not np.any(window_valid):
            return Region(
                **{**region.__dict__, "map_score": 0.0, "map_coverage": 0.0, "map_peak": 0.0}
            )
        values = window[window_valid]
        return Region(
            **{
                **region.__dict__,
                "map_score": float(np.mean(values)),
                "map_coverage": coverage,
                "map_peak": float(np.max(values)),
            }
        )

    def _merge(self, regions: Sequence[Region]) -> list[Region]:
        """Drop overlapping duplicates, keeping the more informative label."""
        # Named detections first, so a face survives its overlapping blob.
        ordered = sorted(
            regions,
            key=lambda r: (r.source != "faces", -r.map_score),
        )
        kept: list[Region] = []
        for region in ordered:
            if any(region.iou(existing) > self.config.merge_iou for existing in kept):
                continue
            kept.append(region)
        return kept

    def propose(
        self,
        focus_map: FocusMap,
        gray: np.ndarray,
        frame_shape: tuple[int, ...],
    ) -> list[Region]:
        """Return the ranked region candidates for one frame.

        ``gray`` is the analysis image; ``frame_shape`` is the source frame
        shape, used to filter out regions that are too small to be a subject.
        """
        self._frame_counter += 1
        scale_to_source = 1.0 / max(focus_map.source_scale, 1e-6)
        proposals: list[Region] = []

        for detector in self.config.detectors:
            if detector == "faces":
                interval = max(1, self.config.face_interval)
                if (self._frame_counter - 1) % interval == 0:
                    self._face_cache = self.detect_faces(gray, scale_to_source)
                proposals.extend(self._face_cache)
            elif detector == "sharp_blobs":
                proposals.extend(self.detect_sharp_blobs(focus_map))
            else:
                raise KeyError(
                    f"unknown region detector {detector!r}; "
                    "choose from 'faces', 'sharp_blobs'"
                )

        frame_area = float(frame_shape[0] * frame_shape[1])
        scored = [self.score_region(r, focus_map) for r in proposals]
        scored = [
            r for r in scored
            if r.area / frame_area >= self.config.min_area_fraction
            and r.map_coverage > 0.0
        ]
        merged = self._merge(scored)
        merged.sort(key=lambda r: r.map_score, reverse=True)
        return merged[: self.config.max_regions]

    def reset(self) -> None:
        self._face_cache = []
        self._frame_counter = 0
