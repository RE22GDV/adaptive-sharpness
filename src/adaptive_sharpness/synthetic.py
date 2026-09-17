"""Synthetic scenes and simulated focus sweeps.

Why this module exists
----------------------
A naive test scene — axis-aligned rectangles drawn straight onto the pixel grid
— is actively misleading for sharpness work.  A step edge that lands exactly on
a pixel boundary occupies *zero* pixels of transition, so a perfectly sharp
image of it has **zero** fine-scale detail energy; blurring such an edge
*creates* detail before destroying it, and several metrics then rise with
defocus instead of falling.  This was measured, not assumed: the Haar level-1
detail energy of a grid-aligned checkerboard is 0.0000 when sharp and 485 at
sigma = 2.

Every scene here is therefore rendered at ``supersample`` times the target
resolution and then area-averaged down, which puts edges at sub-pixel positions
with realistic anti-aliasing — the way a real sensor sees them.

Defocus is modelled with a **disc** point-spread function by default, which is
the geometric-optics circle of confusion, rather than a Gaussian.  A Gaussian is
available for comparison because much of the autofocus literature uses it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal, Sequence

import cv2
import numpy as np

__all__ = [
    "make_scene",
    "defocus",
    "disc_kernel",
    "add_noise",
    "apply_exposure",
    "translate",
    "SweepFrame",
    "focus_sweep",
]

PsfKind = Literal["disc", "gauss"]


def make_scene(
    width: int = 640,
    height: int = 360,
    seed: int = 7,
    supersample: int = 4,
    texture_strength: float = 14.0,
    detail: float = 1.0,
) -> np.ndarray:
    """A deterministic, realistically anti-aliased test scene (BGR uint8).

    Contains structure at several scales — large blocks, mid-size shapes, thin
    lines and broadband texture — so that every metric has something to measure.

    ``detail`` scales how much structure the scene carries.  1.0 is a richly
    textured subject; small values give the sparse, low-contrast frames where
    edge-based metrics start to fail and the weighting has to react.
    """
    if supersample < 1:
        raise ValueError("supersample must be >= 1")
    if detail <= 0.0:
        raise ValueError("detail must be positive")
    rng = np.random.default_rng(seed)
    texture_strength *= detail
    sw, sh = width * supersample, height * supersample
    canvas = np.full((sh, sw), 60.0, dtype=np.float32)

    # Large blocks at deliberately non-integer positions.
    for _ in range(max(1, int(round(9 * detail)))):
        x = rng.uniform(0, sw * 0.85)
        y = rng.uniform(0, sh * 0.85)
        w = rng.uniform(sw * 0.06, sw * 0.22)
        h = rng.uniform(sh * 0.06, sh * 0.25)
        angle = rng.uniform(0, 180)
        value = float(rng.uniform(90, 235))
        box = cv2.boxPoints(((x + w / 2, y + h / 2), (w, h), angle))
        cv2.fillPoly(canvas, [np.int32(box * 16)], value, lineType=cv2.LINE_AA, shift=4)

    # Circles: curved edges at every orientation.
    for _ in range(max(1, int(round(14 * detail)))):
        cx = rng.uniform(0, sw)
        cy = rng.uniform(0, sh)
        radius = rng.uniform(sw * 0.01, sw * 0.05)
        value = float(rng.uniform(70, 245))
        cv2.circle(
            canvas,
            (int(cx * 16), int(cy * 16)),
            int(radius * 16),
            value,
            thickness=-1,
            lineType=cv2.LINE_AA,
            shift=4,
        )

    # Thin lines: the highest-frequency deterministic content.
    for _ in range(max(1, int(round(18 * detail)))):
        p0 = (rng.uniform(0, sw), rng.uniform(0, sh))
        p1 = (p0[0] + rng.uniform(-sw * 0.3, sw * 0.3), p0[1] + rng.uniform(-sh * 0.3, sh * 0.3))
        value = float(rng.uniform(30, 250))
        cv2.line(
            canvas,
            (int(p0[0] * 16), int(p0[1] * 16)),
            (int(p1[0] * 16), int(p1[1] * 16)),
            value,
            thickness=max(1, supersample),
            lineType=cv2.LINE_AA,
            shift=4,
        )

    # Broadband texture with a 1/f-ish spectrum, closer to a natural surface
    # than white noise.
    texture = np.zeros((sh, sw), dtype=np.float32)
    amplitude = texture_strength
    scale = 1.0
    for _ in range(4):
        layer = rng.normal(0.0, amplitude, size=(sh, sw)).astype(np.float32)
        if scale > 1.0:
            layer = cv2.GaussianBlur(layer, (0, 0), scale)
        texture += layer
        amplitude *= 0.8
        scale *= 2.2
    canvas += texture

    np.clip(canvas, 0.0, 255.0, out=canvas)
    # Area-average down: this is what puts the edges at sub-pixel positions.
    small = cv2.resize(canvas, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(small.astype(np.uint8), cv2.COLOR_GRAY2BGR)


def disc_kernel(radius: float) -> np.ndarray:
    """Normalised filled-disc PSF of the given radius, in pixels."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    size = int(np.ceil(radius)) * 2 + 1
    centre = (size - 1) / 2.0
    ys, xs = np.mgrid[0:size, 0:size]
    # Soft edge over one pixel keeps the kernel from aliasing at small radii.
    distance = np.sqrt((xs - centre) ** 2 + (ys - centre) ** 2)
    kernel = np.clip(radius + 0.5 - distance, 0.0, 1.0).astype(np.float32)
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError(f"degenerate disc kernel for radius {radius}")
    return kernel / total


def defocus(image: np.ndarray, radius: float, kind: PsfKind = "disc") -> np.ndarray:
    """Apply a defocus PSF of the given radius (0 leaves the image unchanged)."""
    if radius <= 1e-6:
        return image.copy()
    if kind == "gauss":
        return cv2.GaussianBlur(image, (0, 0), radius)
    if kind != "disc":
        raise ValueError(f"unknown PSF kind: {kind!r}")
    return cv2.filter2D(image, -1, disc_kernel(radius), borderType=cv2.BORDER_REFLECT)


def add_noise(image: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Additive Gaussian sensor noise, in 8-bit intensity units."""
    if sigma <= 0:
        return image.copy()
    rng = np.random.default_rng(seed)
    noisy = image.astype(np.float32) + rng.normal(0.0, sigma, size=image.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_exposure(image: np.ndarray, gain: float) -> np.ndarray:
    """Multiplicative exposure change; values above 255 clip, as on a sensor."""
    scaled = image.astype(np.float32) * gain
    return np.clip(scaled, 0, 255).astype(np.uint8)


def translate(image: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """Sub-pixel translation, used to simulate camera motion."""
    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return cv2.warpAffine(
        image,
        matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


@dataclass(frozen=True)
class SweepFrame:
    """One frame of a simulated focus sweep."""

    index: int
    #: Simulated focus actuator position, in the same 0..1 units the servo uses.
    motor_position: float
    #: Defocus radius applied to this frame, in pixels.
    defocus_radius: float
    image: np.ndarray


def focus_sweep(
    scene: np.ndarray,
    steps: int = 41,
    best_position: float = 0.5,
    max_radius: float = 8.0,
    psf: PsfKind = "disc",
    noise_sigma: float = 0.0,
    exposure_drift: float = 0.0,
    motion_px: float = 0.0,
    seed: int = 11,
) -> Iterator[SweepFrame]:
    """Generate a focus sweep from fully defocused to sharp and back.

    ``best_position`` is where the sweep is in focus; the defocus radius grows
    linearly with the distance from it, which is the standard thin-lens
    approximation for a small range of positions.

    The optional ``noise_sigma``, ``exposure_drift`` and ``motion_px`` arguments
    inject the disturbances the evaluation protocol calls for, so the same
    generator produces both the clean reference sweep and the perturbed ones.
    """
    if steps < 2:
        raise ValueError("a sweep needs at least 2 steps")
    rng = np.random.default_rng(seed)
    reach = max(best_position, 1.0 - best_position)
    for index in range(steps):
        position = index / (steps - 1)
        radius = abs(position - best_position) / max(reach, 1e-9) * max_radius
        frame = defocus(scene, radius, psf)

        if motion_px > 0.0:
            dx, dy = rng.normal(0.0, motion_px, size=2)
            frame = translate(frame, float(dx), float(dy))
        if exposure_drift > 0.0:
            gain = 1.0 + rng.normal(0.0, exposure_drift)
            frame = apply_exposure(frame, float(max(0.05, gain)))
        if noise_sigma > 0.0:
            frame = add_noise(frame, noise_sigma, seed=int(rng.integers(0, 2**31)))

        yield SweepFrame(
            index=index,
            motor_position=position,
            defocus_radius=float(radius),
            image=frame,
        )
