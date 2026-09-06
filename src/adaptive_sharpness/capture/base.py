"""Capture abstraction.

The library never talks to a camera directly; it consumes
:class:`~sharpness.types.Frame` objects from a :class:`FrameSource`.  That keeps
the sharpness code testable without hardware and lets the same pipeline run on a
live camera, a recorded video, or a synthetic sweep.

:class:`ThreadedSource` implements the "always process the freshest frame"
requirement: a worker thread grabs continuously into a single-slot buffer, so a
slow consumer drops intermediate frames instead of falling further and further
behind a growing queue.
"""
from __future__ import annotations

import abc
import logging
import threading
import time
from typing import Iterator

from ..types import Frame

logger = logging.getLogger(__name__)

__all__ = ["FrameSource", "CaptureError", "ThreadedSource"]


class CaptureError(RuntimeError):
    """Raised when a capture backend cannot be opened or used."""


class FrameSource(abc.ABC):
    """A source of frames.

    Implementations must be safe to :meth:`close` twice and must not raise from
    :meth:`close` when they were never opened.
    """

    #: Human-readable description, filled in by :meth:`open`.
    description: str = "unknown source"

    @abc.abstractmethod
    def open(self) -> None:
        """Acquire the device or file.  Raises :class:`CaptureError` on failure."""

    @abc.abstractmethod
    def read(self) -> Frame | None:
        """Return the next frame, or ``None`` when the source is exhausted."""

    @abc.abstractmethod
    def close(self) -> None:
        """Release the device or file."""

    @property
    def is_live(self) -> bool:
        """True for real-time sources, where dropping stale frames is correct."""
        return False

    def __enter__(self) -> "FrameSource":
        self.open()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def frames(self, limit: int | None = None) -> Iterator[Frame]:
        """Iterate frames until exhaustion or ``limit`` frames."""
        count = 0
        while limit is None or count < limit:
            frame = self.read()
            if frame is None:
                return
            count += 1
            yield frame


class ThreadedSource(FrameSource):
    """Runs a :class:`FrameSource` in a worker thread, keeping only the newest frame.

    This is what makes the end-to-end rate capture-bound rather than
    capture-plus-processing-bound: the USB wait for the next preview JPEG
    overlaps with the metric computation of the previous one.  When the consumer
    is slower than the camera, intermediate frames are discarded (and counted in
    :attr:`dropped`) so the frame being analysed is always the most recent one.
    """

    def __init__(self, source: FrameSource, poll_timeout: float = 5.0) -> None:
        self._source = source
        self._poll_timeout = poll_timeout
        self._thread: threading.Thread | None = None
        self._condition = threading.Condition()
        self._latest: Frame | None = None
        self._latest_serial = 0
        self._consumed_serial = 0
        self._stop = threading.Event()
        self._error: BaseException | None = None
        self._exhausted = False
        self.dropped = 0

    @property
    def description(self) -> str:  # type: ignore[override]
        return f"threaded({self._source.description})"

    @property
    def is_live(self) -> bool:
        return self._source.is_live

    def open(self) -> None:
        self._source.open()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run, name="capture", daemon=True
        )
        self._thread.start()
        logger.debug("threaded capture started for %s", self._source.description)

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                frame = self._source.read()
                if frame is None:
                    with self._condition:
                        self._exhausted = True
                        self._condition.notify_all()
                    return
                with self._condition:
                    if self._latest is not None and self._latest_serial > self._consumed_serial:
                        # The consumer never picked up the previous frame.
                        self.dropped += 1
                    self._latest = frame
                    self._latest_serial += 1
                    self._condition.notify_all()
        except BaseException as exc:  # noqa: BLE001 - propagated to the consumer
            logger.exception("capture thread failed")
            with self._condition:
                self._error = exc
                self._condition.notify_all()

    def read(self) -> Frame | None:
        """Return the newest unconsumed frame, waiting up to ``poll_timeout``."""
        deadline = time.perf_counter() + self._poll_timeout
        with self._condition:
            while True:
                if self._error is not None:
                    error = self._error
                    self._error = None
                    raise CaptureError(f"capture thread failed: {error}") from error
                if self._latest is not None and self._latest_serial > self._consumed_serial:
                    self._consumed_serial = self._latest_serial
                    return self._latest
                if self._exhausted:
                    return None
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    logger.warning(
                        "no frame within %.1fs from %s", self._poll_timeout,
                        self._source.description,
                    )
                    return None
                self._condition.wait(remaining)

    def close(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=self._poll_timeout + 1.0)
            if thread.is_alive():
                logger.warning("capture thread did not stop cleanly")
        self._thread = None
        self._source.close()
