"""Frame sources and the factory that picks one.

``auto`` follows the priority the project requires:

1. a real V4L2/UVC capture node, if one exists;
2. PTP live view via libgphoto2;
3. nothing - the caller is told explicitly rather than being handed a silent
   fallback that would fake a working camera.
"""
from __future__ import annotations

import logging
from pathlib import Path

from sharpness.config import CaptureConfig

from .base import CaptureError, FrameSource, ThreadedSource
from .file_source import IMAGE_SUFFIXES, ImageDirectorySource, VideoFileSource
from .gphoto_source import GPhotoLiveViewSource
from .synthetic_source import SyntheticSweepSource
from .v4l2_source import V4L2Source

logger = logging.getLogger(__name__)

__all__ = [
    "FrameSource",
    "CaptureError",
    "ThreadedSource",
    "V4L2Source",
    "GPhotoLiveViewSource",
    "VideoFileSource",
    "ImageDirectorySource",
    "SyntheticSweepSource",
    "open_source",
    "find_capture_nodes",
]

# Drivers whose /dev/video* nodes belong to the SoC, not to a camera.
_INTERNAL_DRIVERS = {"pispbe", "rpi-hevc-dec", "bcm2835-codec", "bcm2835-isp", "rpivid"}


def find_capture_nodes() -> list[str]:
    """Return the ``/dev/video*`` nodes that are real capture devices.

    On a Raspberry Pi 5 most video nodes belong to the ISP and the hardware
    codecs; treating them as cameras is the classic way to end up "opening" a
    device that can never produce a frame.
    """
    import shutil
    import subprocess

    nodes = sorted(Path("/dev").glob("video*"), key=lambda p: p.name)
    if not nodes:
        return []
    if shutil.which("v4l2-ctl") is None:
        logger.debug("v4l2-ctl is unavailable; cannot classify video nodes")
        return []

    capture_nodes: list[str] = []
    for node in nodes:
        try:
            proc = subprocess.run(
                ["v4l2-ctl", "-d", str(node), "--info"],
                capture_output=True, text=True, timeout=5,
                env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"},
            )
        except (OSError, subprocess.SubprocessError):
            continue
        if proc.returncode != 0:
            continue
        driver = ""
        for line in proc.stdout.splitlines():
            if "Driver name" in line:
                driver = line.split(":", 1)[1].strip()
                break
        if driver and driver not in _INTERNAL_DRIVERS:
            capture_nodes.append(str(node))
    return capture_nodes


def _build(config: CaptureConfig) -> FrameSource:
    backend = config.backend.lower()

    if backend == "v4l2":
        return V4L2Source(config.device, config.width, config.height, config.fps, config.fourcc)
    if backend == "gphoto2":
        return GPhotoLiveViewSource()
    if backend == "synthetic":
        return SyntheticSweepSource(config.width, config.height, loop=config.loop)
    if backend == "file":
        if not config.path:
            raise CaptureError("the 'file' backend needs capture.path to be set")
        path = Path(config.path)
        if path.is_dir():
            return ImageDirectorySource(path, loop=config.loop)
        if path.is_file():
            if path.suffix.lower() in IMAGE_SUFFIXES:
                raise CaptureError(
                    f"{path} is a single image; point capture.path at its directory"
                )
            return VideoFileSource(path, loop=config.loop)
        raise CaptureError(f"capture.path does not exist: {path}")

    if backend != "auto":
        raise CaptureError(
            f"unknown capture backend {config.backend!r}; "
            "choose auto, v4l2, gphoto2, file or synthetic"
        )

    nodes = find_capture_nodes()
    if nodes:
        logger.info("auto-selected V4L2 capture node %s", nodes[0])
        return V4L2Source(nodes[0], config.width, config.height, config.fps, config.fourcc)

    try:
        import gphoto2  # noqa: F401
    except ImportError:
        raise CaptureError(
            "no V4L2 capture node was found and python-gphoto2 is not installed, "
            "so no camera can be opened. Run tools/probe_camera.py for details, or "
            "set capture.backend to 'file' or 'synthetic'."
        ) from None
    logger.info("auto-selected PTP live view (no V4L2 capture node present)")
    return GPhotoLiveViewSource()


def open_source(config: CaptureConfig) -> FrameSource:
    """Build a frame source from the configuration and open it.

    Live sources are wrapped in :class:`ThreadedSource` when
    ``config.threaded`` is set, so the consumer always gets the newest frame.
    """
    source = _build(config)
    if config.threaded and source.is_live:
        source = ThreadedSource(source)
    source.open()
    return source
