"""Recording a run to disk: per-frame CSV, sampled frames, and a manifest.

This lives in the library rather than in a script because two callers need
exactly the same format - the batch collector and the UI's record button - and
two separate implementations of a file format drift apart.

It writes result rows, not camera data: nothing here opens a device.

Layout produced::

    <directory>/
        frames.csv      one row per frame, everything the evaluator computed
        frames/         every Nth frame as PNG
        manifest.json   environment, configuration and run summary

Frames are stored as **PNG**.  JPEG would remove exactly the high-frequency
content the metrics measure, so a re-analysis of JPEG frames could not reproduce
the recorded numbers.  Encoding runs on a background thread, because doing it
inline lands inside the measured inter-frame interval and would make the
recorded timing describe the recorder rather than the pipeline.
"""
from __future__ import annotations

import csv
import json
import logging
import platform
import queue
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from .config import SharpnessConfig
from .types import SharpnessResult

logger = logging.getLogger(__name__)

__all__ = ["FrameWriter", "RunRecorder", "environment"]


def environment() -> dict[str, Any]:
    """Machine and library versions, recorded so a run stays interpretable."""
    import cv2

    from . import __version__

    info: dict[str, Any] = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "library_version": __version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "numpy": np.__version__,
        "opencv": cv2.__version__,
    }
    model = Path("/proc/device-tree/model")
    if model.exists():
        try:
            info["device"] = model.read_text(errors="ignore").strip("\x00 \n")
        except OSError:
            pass
    return info


class FrameWriter:
    """Writes sampled frames to disk on a background thread."""

    def __init__(self, directory: Path, max_pending: int = 64) -> None:
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[tuple[str, np.ndarray] | None] = queue.Queue(max_pending)
        self._thread = threading.Thread(target=self._run, name="frame-writer", daemon=True)
        self.written = 0
        self.dropped = 0
        self._thread.start()

    def _run(self) -> None:
        import cv2

        while True:
            item = self._queue.get()
            if item is None:
                return
            name, image = item
            try:
                cv2.imwrite(str(self.directory / name), image)
                self.written += 1
            except Exception as exc:  # noqa: BLE001 - a bad write must not stop a run
                logger.error("could not write %s: %s", name, exc)

    def submit(self, name: str, image: np.ndarray) -> bool:
        """Queue a frame; returns False when the queue is full and it is skipped."""
        try:
            # A copy is required: the capture backend may reuse its buffer.
            self._queue.put_nowait((name, image.copy()))
            return True
        except queue.Full:
            self.dropped += 1
            logger.warning("frame-writer queue full; skipped %s", name)
            return False

    def close(self, timeout: float = 30.0) -> None:
        self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("frame writer did not finish within %.0fs", timeout)


class RunRecorder:
    """Records evaluated frames into one directory.

    Usage::

        recorder = RunRecorder(Path("data/run1"), config, save_every=10)
        recorder.start(source_description="gphoto2 live view")
        ...
        recorder.add(result, frame.data, frame.index, extra={"subject": "face"})
        ...
        summary = recorder.stop()

    ``add`` is cheap: one dict build, one CSV row, and at most one queue push.
    """

    def __init__(
        self,
        directory: Path,
        config: SharpnessConfig,
        save_every: int = 10,
        note: str = "",
    ) -> None:
        self.directory = Path(directory)
        self.config = config
        self.save_every = max(0, int(save_every))
        self.note = note
        self.frames = 0
        self.started_at: float | None = None

        self._handle: TextIO | None = None
        self._writer: csv.DictWriter | None = None
        self._images: FrameWriter | None = None
        self._intervals: list[float] = []
        self._previous: float = 0.0
        self._source_description = ""

    @property
    def active(self) -> bool:
        return self._handle is not None

    @property
    def elapsed(self) -> float:
        if self.started_at is None:
            return 0.0
        return time.perf_counter() - self.started_at

    def start(self, source_description: str = "") -> None:
        if self.active:
            raise RuntimeError("recorder is already running")
        self.directory.mkdir(parents=True, exist_ok=True)
        self._source_description = source_description
        self._handle = (self.directory / "frames.csv").open(
            "w", newline="", encoding="utf-8"
        )
        self._writer = None
        self._images = FrameWriter(self.directory / "frames") if self.save_every else None
        self.frames = 0
        self._intervals.clear()
        self.started_at = time.perf_counter()
        self._previous = self.started_at
        logger.info("recording to %s", self.directory)

    def add(
        self,
        result: SharpnessResult,
        image: np.ndarray | None = None,
        frame_index: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Record one evaluated frame."""
        if self._handle is None:
            raise RuntimeError("recorder is not running; call start() first")

        row = result.as_row()
        if extra:
            row.update(extra)

        now = time.perf_counter()
        row["wall_time"] = round(now - (self.started_at or now), 6)
        row["interval_s"] = round(now - self._previous, 6)
        self._intervals.append(now - self._previous)
        self._previous = now
        row["image_file"] = ""

        if (
            self._images is not None
            and image is not None
            and self.frames % self.save_every == 0
        ):
            index = frame_index if frame_index is not None else self.frames
            name = f"frame_{index:06d}.png"
            if self._images.submit(name, image):
                row["image_file"] = name

        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
            self._writer.writeheader()
        # Later rows must not introduce new columns, or the CSV goes ragged.
        self._writer.writerow({key: row.get(key, "") for key in self._writer.fieldnames})
        self.frames += 1
        if self.frames % 100 == 0:
            self._handle.flush()

    def stop(self) -> dict[str, Any]:
        """Close the files, write the manifest, and return the run summary."""
        if self._handle is None:
            return {}
        elapsed = self.elapsed
        self._handle.close()
        self._handle = None
        self._writer = None
        if self._images is not None:
            self._images.close()

        summary: dict[str, Any] = {
            "frames": self.frames,
            "duration_s": round(elapsed, 3),
            "mean_fps": round(self.frames / elapsed, 3) if elapsed > 0 else 0.0,
            "images_saved": self._images.written if self._images else 0,
            "images_skipped": self._images.dropped if self._images else 0,
        }
        if len(self._intervals) > 1:
            arr = np.array(self._intervals[1:])
            summary["interval_ms_mean"] = round(float(arr.mean()) * 1e3, 3)
            summary["interval_ms_p95"] = round(float(np.percentile(arr, 95)) * 1e3, 3)
            summary["interval_ms_max"] = round(float(arr.max()) * 1e3, 3)

        manifest = {
            "environment": environment(),
            "source": self._source_description,
            "note": self.note,
            "save_every": self.save_every,
            "config": {
                "pipeline": asdict(self.config.pipeline),
                "metrics_enabled": list(self.config.metrics.enabled),
                "normalization": asdict(self.config.normalization),
                "temporal": asdict(self.config.temporal),
                "focus_map": asdict(self.config.focus_map),
            },
            "summary": summary,
        }
        (self.directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        self._images = None
        self.started_at = None
        logger.info("recording stopped: %s", json.dumps(summary))
        return summary

    def __enter__(self) -> "RunRecorder":
        self.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop()
