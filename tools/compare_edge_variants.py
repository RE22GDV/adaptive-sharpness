"""Compare formulations of the edge-width metric across a defocus ramp.

Variant A: Canny edge set + median width (the current implementation).
Variant B: gradient-energy-weighted global width, no edge set at all.
Variant C: Canny with percentile-adaptive thresholds + median width.

Also sweeps the morphological range window, which sets the largest measurable
edge width.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adaptive_sharpness.config import PipelineConfig  # noqa: E402
from adaptive_sharpness.preprocess import Preprocessor  # noqa: E402
from adaptive_sharpness.synthetic import defocus, make_scene  # noqa: E402

SOBEL_SCALE = 1.0 / 8.0
EPS = 1e-9


def gradients(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=SOBEL_SCALE)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=SOBEL_SCALE)
    return cv2.magnitude(gx, gy), gx


def local_range(gray: np.ndarray, window: int) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window, window))
    return cv2.dilate(gray, kernel) - cv2.erode(gray, kernel)


def variant_a(gray: np.ndarray, window: int) -> tuple[float, int]:
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    edges = cv2.Canny(u8, 50, 150, L2gradient=True) > 0
    n = int(np.count_nonzero(edges))
    if n < 24:
        return 0.0, n
    mag, _ = gradients(gray)
    rng = local_range(gray, window)
    widths = rng[edges] / (mag[edges] + EPS)
    np.clip(widths, 0, 32, out=widths)
    median = float(np.median(widths))
    return (1.0 / median if median > EPS else 0.0), n


def variant_b(gray: np.ndarray, window: int, min_gradient: float = 1.0) -> tuple[float, int]:
    mag, _ = gradients(gray)
    rng = local_range(gray, window)
    mask = mag >= min_gradient
    n = int(np.count_nonzero(mask))
    if n < 24:
        return 0.0, n
    m = mag[mask]
    r = rng[mask]
    # Gradient-energy-weighted mean of (range / gradient).
    denom = float(np.sum(m * m))
    if denom <= EPS:
        return 0.0, n
    width = float(np.sum(r * m)) / denom
    return (1.0 / width if width > EPS else 0.0), n


def variant_c(gray: np.ndarray, window: int) -> tuple[float, int]:
    mag, _ = gradients(gray)
    high = float(np.percentile(mag, 97.0))
    low = 0.4 * high
    u8 = np.clip(gray, 0, 255).astype(np.uint8)
    # Canny thresholds are on its own internal gradient scale (Sobel, unscaled),
    # so scale the percentile back up by 8.
    edges = cv2.Canny(u8, int(low * 8), int(high * 8), L2gradient=True) > 0
    n = int(np.count_nonzero(edges))
    if n < 24:
        return 0.0, n
    rng = local_range(gray, window)
    widths = rng[edges] / (mag[edges] + EPS)
    np.clip(widths, 0, 32, out=widths)
    median = float(np.median(widths))
    return (1.0 / median if median > EPS else 0.0), n


def main() -> int:
    scene = make_scene()
    pre = Preprocessor(PipelineConfig())
    radii = [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0]

    for window in (7, 11, 15, 21):
        print("=" * 78)
        print(f"range_window = {window}")
        print("=" * 78)
        print(f"{'radius':>7} | {'A value':>9} {'A n':>6} | {'B value':>9} {'B n':>6} "
              f"| {'C value':>9} {'C n':>6}")
        series: dict[str, list[float]] = {"A": [], "B": [], "C": []}
        for radius in radii:
            gray = pre.prepare(defocus(scene, radius)).gray
            a, na = variant_a(gray, window)
            b, nb = variant_b(gray, window)
            c, nc = variant_c(gray, window)
            series["A"].append(a)
            series["B"].append(b)
            series["C"].append(c)
            print(f"{radius:7.1f} | {a:9.4f} {na:6d} | {b:9.4f} {nb:6d} | {c:9.4f} {nc:6d}")
        for name, values in series.items():
            ok = all(x >= y - 1e-9 for x, y in zip(values, values[1:]))
            drop = values[0] / values[-1] if values[-1] > 1e-9 else float("inf")
            print(f"  variant {name}: monotone={ok}  dynamic_range={drop:.2f}x")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
