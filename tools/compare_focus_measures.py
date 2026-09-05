"""Pick the per-tile focus measure by measurement, not by argument.

Builds split-focus scenes where the *correct* answer is known, including the
adversarial case that matters: a sparsely textured but SHARP half next to a
densely textured but BLURRED half.  A plain gradient-energy measure picks the
blurred half there, because it has more texture.  A good measure must not.

Cases
-----
same_texture      both halves equally textured, left sharp   -> left must win
sparse_vs_dense   left sparse and SHARP, right dense and BLURRED -> left must win
dense_vs_sparse   left dense and SHARP, right sparse and BLURRED -> left must win
flat_vs_textured  left featureless, right textured and sharp -> left must be INVALID
both_sharp        both sharp, different texture -> neither should win by much
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sharpness.config import FocusMapConfig, PipelineConfig  # noqa: E402
from sharpness.focusmap import TILE_MEASURES, FocusMapper  # noqa: E402
from sharpness.preprocess import Preprocessor  # noqa: E402
from tools.synthetic import add_noise, defocus, make_scene  # noqa: E402

WIDTH, HEIGHT = 640, 360


def split_scene(
    left_detail: float,
    right_detail: float,
    left_blur: float,
    right_blur: float,
    left_flat: bool = False,
    noise: float = 0.0,
) -> np.ndarray:
    """Two independently textured and independently defocused halves.

    Noise is added *after* the blur, as a sensor does - which is exactly what
    makes it dangerous here: it puts high-frequency energy back into the
    defocused half.
    """
    left = (
        np.full((HEIGHT, WIDTH, 3), 110, dtype=np.uint8)
        if left_flat
        else make_scene(WIDTH, HEIGHT, seed=3, detail=left_detail)
    )
    right = make_scene(WIDTH, HEIGHT, seed=9, detail=right_detail)
    left = defocus(left, left_blur)
    right = defocus(right, right_blur)
    combined = left.copy()
    combined[:, WIDTH // 2 :] = right[:, WIDTH // 2 :]
    if noise > 0.0:
        combined = add_noise(combined, noise, seed=5)
    return combined


def half_scores(mapper: FocusMapper, pre: Preprocessor, image: np.ndarray):
    prepared = pre.prepare(image)
    fmap = mapper.compute(prepared.gray, prepared.scale)
    cols = fmap.shape[1]
    mid = cols // 2
    # Ignore the two columns straddling the seam.
    left_mask = fmap.valid[:, : max(1, mid - 1)]
    right_mask = fmap.valid[:, mid + 1 :]
    left_vals = fmap.score[:, : max(1, mid - 1)][left_mask]
    right_vals = fmap.score[:, mid + 1 :][right_mask]
    return (
        float(np.mean(left_vals)) if left_vals.size else float("nan"),
        float(np.mean(right_vals)) if right_vals.size else float("nan"),
        float(np.count_nonzero(left_mask)) / max(1, left_mask.size),
        float(np.count_nonzero(right_mask)) / max(1, right_mask.size),
    )


CASES = {
    # name: (scene kwargs, expected winner)
    "same_texture": (dict(left_detail=1.0, right_detail=1.0, left_blur=0.0, right_blur=5.0), "left"),
    "sparse_sharp_vs_dense_blur": (
        dict(left_detail=0.18, right_detail=1.0, left_blur=0.0, right_blur=5.0), "left"),
    "dense_sharp_vs_sparse_blur": (
        dict(left_detail=1.0, right_detail=0.18, left_blur=0.0, right_blur=5.0), "left"),
    "flat_vs_textured_sharp": (
        dict(left_detail=1.0, right_detail=1.0, left_blur=0.0, right_blur=0.0,
             left_flat=True), "invalid_left"),
    "both_sharp_diff_texture": (
        dict(left_detail=1.0, right_detail=0.18, left_blur=0.0, right_blur=0.0), "tie"),
    # Noise puts high-frequency energy back into the defocused half, which is
    # where a ratio of two high-pass measures is most likely to break down.
    "same_texture_noise8": (
        dict(left_detail=1.0, right_detail=1.0, left_blur=0.0, right_blur=5.0,
             noise=8.0), "left"),
    "same_texture_noise20": (
        dict(left_detail=1.0, right_detail=1.0, left_blur=0.0, right_blur=5.0,
             noise=20.0), "left"),
    "sparse_sharp_vs_dense_blur_noise20": (
        dict(left_detail=0.18, right_detail=1.0, left_blur=0.0, right_blur=5.0,
             noise=20.0), "left"),
    "both_sharp_noise20": (
        dict(left_detail=1.0, right_detail=0.18, left_blur=0.0, right_blur=0.0,
             noise=20.0), "tie"),
}


def main() -> int:
    pre = Preprocessor(PipelineConfig())
    print(f"{'case':30s} {'measure':16s} {'left':>7} {'right':>7} "
          f"{'valid_L':>8} {'margin':>8}  verdict")
    print("-" * 92)

    tally: dict[str, int] = {m: 0 for m in TILE_MEASURES}
    win_margins: dict[str, list[float]] = {m: [] for m in TILE_MEASURES}
    tie_margins: dict[str, list[float]] = {m: [] for m in TILE_MEASURES}
    for case_name, (kwargs, expected) in CASES.items():
        image = split_scene(**kwargs)
        for measure in TILE_MEASURES:
            mapper = FocusMapper(FocusMapConfig(measure=measure))
            left, right, valid_l, valid_r = half_scores(mapper, pre, image)
            margin = (left - right) if np.isfinite(left) and np.isfinite(right) else float("nan")

            if expected == "left":
                ok = np.isfinite(margin) and margin > 0.05
                if np.isfinite(margin):
                    win_margins[measure].append(margin)
            elif expected == "invalid_left":
                # The featureless half must be rejected, not ranked.
                ok = valid_l < 0.15
            else:  # tie
                ok = np.isfinite(margin) and abs(margin) < 0.30
                if np.isfinite(margin):
                    tie_margins[measure].append(abs(margin))

            tally[measure] += int(ok)
            print(
                f"{case_name:30s} {measure:16s} {left:7.3f} {right:7.3f} "
                f"{valid_l:8.2f} {margin:8.3f}  {'PASS' if ok else 'FAIL'}"
            )
        print()

    print("score out of", len(CASES))
    for measure, passed in sorted(tally.items(), key=lambda kv: -kv[1]):
        print(f"  {measure:16s} {passed}/{len(CASES)}")

    best = max(tally.items(), key=lambda kv: kv[1])
    print(f"\nbest measure: {best[0]} ({best[1]}/{len(CASES)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
