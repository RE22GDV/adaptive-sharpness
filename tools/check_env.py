"""Quick environment probe: interpreter, packages, and OpenCV video backends."""
from __future__ import annotations

import importlib
import platform
import sys

PACKAGES = [
    "numpy",
    "cv2",
    "scipy",
    "pywt",
    "gphoto2",
    "pyrealsense2",
    "pytest",
    "matplotlib",
    "psutil",
]


def main() -> int:
    print(f"python      : {sys.version.split()[0]} ({platform.machine()})")
    print(f"platform    : {platform.platform()}")
    print("packages:")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - report any import failure
            print(f"  {name:14s} MISSING ({type(exc).__name__})")
            continue
        version = getattr(mod, "__version__", "?")
        print(f"  {name:14s} {version}")

    try:
        import cv2
    except Exception:
        print("opencv      : unavailable")
        return 0

    print("opencv video backends:")
    try:
        names = [
            cv2.videoio_registry.getBackendName(b)
            for b in cv2.videoio_registry.getCameraBackends()
        ]
        print(f"  camera    : {', '.join(names)}")
        names = [
            cv2.videoio_registry.getBackendName(b)
            for b in cv2.videoio_registry.getStreamBackends()
        ]
        print(f"  stream    : {', '.join(names)}")
    except Exception as exc:  # noqa: BLE001
        print(f"  registry unavailable: {exc}")
    print(f"cv2 threads : {cv2.getNumThreads()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
