"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
# src/ holds the installed package; the repository root is added as well so the
# tests can reach tools/ and demo/, which are scripts rather than packages.
for entry in (ROOT / "src", ROOT):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from adaptive_sharpness.synthetic import defocus, make_scene  # noqa: E402


@pytest.fixture(scope="session")
def scene() -> np.ndarray:
    """A deterministic, realistically anti-aliased BGR test scene."""
    return make_scene(640, 360, seed=7)


@pytest.fixture(scope="session")
def gray_scene(scene: np.ndarray) -> np.ndarray:
    """The same scene as a float32 single-channel analysis image."""
    import cv2

    small = cv2.resize(scene, (320, 180), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)


@pytest.fixture(scope="session")
def defocus_series(scene: np.ndarray) -> list[tuple[float, np.ndarray]]:
    """(radius, frame) pairs spanning sharp to heavily defocused."""
    radii = [0.0, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    return [(r, defocus(scene, r)) for r in radii]


@pytest.fixture
def flat_gray() -> np.ndarray:
    """A featureless mid-grey image: no edges, no texture, no noise."""
    return np.full((180, 320), 128.0, dtype=np.float32)
