"""How much does the point-source reference depend on how it is measured?

Every accuracy claim in this project is checked against one number: the radius
enclosing half of a spot's background-subtracted pixel sum.  That number has
free parameters - which fraction, how big a window, how the background is
estimated - and the source saturates near focus, which removes signal from the
core exactly where the radius is smallest.

If the best step moves when those choices move, the reference carries an
uncertainty that every result built on it inherits.  This measures that.

    python3 tools/reference_sensitivity.py data --json data/reference_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from tools.evaluation import peak_interval, provenance, select_frames  # noqa: E402
from tools.protocol_study import step_profile  # noqa: E402
from tools.replay_configs import load_context  # noqa: E402

logger = logging.getLogger("reference_sensitivity")


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    def ranks(x: np.ndarray) -> np.ndarray:
        order = np.argsort(x, kind="mergesort")
        out = np.empty(x.size)
        out[order] = np.arange(1, x.size + 1)
        return out

    ra, rb = ranks(a) - (a.size + 1) / 2, ranks(b) - (b.size + 1) / 2
    denominator = float(np.sqrt((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / denominator) if denominator > 0 else float("nan")

_EPS = 1e-12


@dataclass(frozen=True)
class SpotVariant:
    """One way of turning a frame into a spot size."""

    fraction: float
    half: int
    background: str  # "median" or "p25"

    @property
    def name(self) -> str:
        return f"r{int(self.fraction * 100)}_w{self.half}_{self.background}"


VARIANTS: tuple[SpotVariant, ...] = tuple(
    SpotVariant(fraction=f, half=h, background=b)
    for f in (0.25, 0.5, 0.75)
    for h in (50, 80, 120)
    for b in ("median",)
) + (
    SpotVariant(0.5, 80, "p25"),
)


def measure(image: np.ndarray, variant: SpotVariant) -> tuple[float, float, float]:
    """Spot radius, clipped-core fraction, and how concentrated the window is.

    The third number is what says whether the measurement is about the source
    at all.  For a compact spot almost all of the window's signal lies within a
    couple of radii of the centroid; when the window is large enough to include
    other structure - the edge of the screen the source sits on - the signal
    spreads out, the "radius" describes the window rather than the optics, and
    it can move the *wrong way* with focus because that other structure sharpens
    too.  Measured: a 120 px window never falls below 20 px here and its
    ordering is anti-correlated with the smaller windows at -0.79.
    """
    blurred = cv2.GaussianBlur(image, (9, 9), 0)
    _, _, _, peak = cv2.minMaxLoc(blurred)
    cx, cy = int(peak[0]), int(peak[1])
    half = variant.half
    x0, x1 = max(0, cx - half), min(image.shape[1], cx + half + 1)
    y0, y1 = max(0, cy - half), min(image.shape[0], cy + half + 1)
    window = image[y0:y1, x0:x1].astype(np.float64)

    border = np.concatenate([
        window[0, :], window[-1, :], window[:, 0], window[:, -1],
    ])
    level = (
        float(np.median(border)) if variant.background == "median"
        else float(np.percentile(border, 25))
    )
    signal = np.maximum(window - level, 0.0)
    total = float(signal.sum())
    saturated = float(np.mean(window >= 250.0))
    if total <= _EPS:
        return float("nan"), saturated, 0.0

    grid_y, grid_x = np.mgrid[0:window.shape[0], 0:window.shape[1]]
    centre_y = float((signal * grid_y).sum() / total)
    centre_x = float((signal * grid_x).sum() / total)
    distance = np.sqrt((grid_y - centre_y) ** 2 + (grid_x - centre_x) ** 2)

    order = np.argsort(distance, axis=None)
    cumulative = np.cumsum(signal.ravel()[order])
    index = int(np.searchsorted(cumulative, total * variant.fraction))
    index = min(index, order.size - 1)
    radius = float(distance.ravel()[order][index])

    # Signal reaching the window border means the window is not isolating the
    # source: either the disc is truncated, or something else bright is inside
    # it.  Either way the number describes the window, not the optics, so it is
    # not a radius and must not be reported as one.  Measured: a 120 px window
    # on these recordings never falls below 20 px and its ordering is
    # *anti*-correlated with the smaller windows at -0.79.
    border_share = float(signal[0, :].sum() + signal[-1, :].sum()
                         + signal[:, 0].sum() + signal[:, -1].sum()) / total
    concentration = float(signal[distance <= 2.0 * radius].sum() / total) if radius > 0 else 0.0
    if border_share >= 0.02:
        return float("nan"), saturated, concentration
    return radius, saturated, concentration


def analyse(directory: Path) -> dict[str, Any] | None:
    context = load_context(directory)
    selection = select_frames(step=context.step, hold=context.hold)
    if selection.count < 20:
        return None
    labels = context.step[selection.mask]
    chosen = [f for f, keep in zip(context.files, selection.mask) if keep]

    values: dict[str, list[float]] = {v.name: [] for v in VARIANTS}
    concentrations: dict[str, list[float]] = {v.name: [] for v in VARIANTS}
    saturation: list[float] = []
    for index, name in enumerate(chosen):
        image = cv2.imread(str(directory / "frames" / name), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"unreadable frame: {name}")
        for variant in VARIANTS:
            radius, saturated, concentration = measure(image, variant)
            values[variant.name].append(radius)
            concentrations[variant.name].append(concentration)
            if variant.fraction == 0.5 and variant.half == 80 and variant.background == "median":
                saturation.append(saturated)
        if index and index % 250 == 0:
            logger.info("%s: %d/%d", directory.name, index, len(chosen))

    # How much the reference depends on its own parameters, measured rather
    # than argued: the rank correlation between every pair of variants.  This
    # is the ceiling on how finely any model can be scored against it.
    cross: dict[str, dict[str, float]] = {}
    for first in VARIANTS:
        cross[first.name] = {}
        a = np.asarray(values[first.name], dtype=float)
        for second in VARIANTS:
            b = np.asarray(values[second.name], dtype=float)
            ok = np.isfinite(a) & np.isfinite(b)
            cross[first.name][second.name] = (
                _spearman(a[ok], b[ok]) if ok.sum() > 4 else float("nan")
            )

    saturated_mask = np.asarray(saturation) > 0.0
    out: dict[str, Any] = {
        "frames": len(chosen),
        "cross_correlation": cross,
        "frames_with_clipped_core": float(np.mean(saturated_mask)),
        "variants": {},
    }

    for variant in VARIANTS:
        series = np.asarray(values[variant.name], dtype=float)
        steps, medians, scatter = step_profile(-series, labels)  # -r: up is sharper
        first, last = peak_interval(medians)
        usable = float(np.mean(np.isfinite(series)))
        entry = {
            "usable_fraction": usable,
            "best_step_first": float(steps[first]) if first >= 0 else float("nan"),
            "best_step_last": float(steps[last]) if last >= 0 else float("nan"),
            "min_radius_px": float(-np.nanmax(medians)),
            "max_radius_px": float(-np.nanmin(medians)),
            "within_step_scatter_px": float(np.nanmedian(scatter)),
        }
        # The same, with frames whose core is clipped removed.
        if saturated_mask.any() and (~saturated_mask).sum() > 20:
            keep = ~saturated_mask
            steps_c, medians_c, _ = step_profile(-series[keep], labels[keep])
            first_c, _ = peak_interval(medians_c)
            entry["best_step_unclipped"] = (
                float(steps_c[first_c]) if first_c >= 0 else float("nan")
            )
        out["variants"][variant.name] = entry
    return out


def print_report(results: dict[str, Any]) -> None:
    print("=" * 84)
    print("REFERENCE SENSITIVITY - does the best step depend on how the spot is measured?")
    print("=" * 84)
    meta = results.get("provenance", {})
    print(f"\ncommit {meta.get('commit','?')[:12]}   numpy {meta.get('numpy','?')}\n")

    for name, entry in results["runs"].items():
        print(f"== {name}   {entry['frames']} frames, "
              f"{100 * entry['frames_with_clipped_core']:.1f}% with a clipped core")
        header = (f"{'variant':<20}{'best step':>11}{'plateau':>9}"
                  f"{'r min':>8}{'r max':>8}{'unclipped':>11}")
        print(header)
        print("-" * len(header))
        for variant, row in entry["variants"].items():
            plateau = row["best_step_last"] - row["best_step_first"] + 1
            unclipped = row.get("best_step_unclipped", float("nan"))
            print(
                f"{variant:<20}"
                f"{row['best_step_first']:>11.0f}{plateau:>9.0f}"
                f"{row['min_radius_px']:>8.1f}{row['max_radius_px']:>8.1f}"
                + (f"{unclipped:>11.0f}" if np.isfinite(unclipped) else f"{'-':>11}")
            )
        steps = [r["best_step_first"] for r in entry["variants"].values()]
        finite = [s for s in steps if np.isfinite(s)]
        if finite:
            print(f"\n  best step across variants: {min(finite):.0f} - {max(finite):.0f}"
                  f"   (spread {max(finite) - min(finite):.0f} steps)")

        matrix = entry.get("cross_correlation") or {}
        if matrix:
            names = list(matrix)
            print("\n  Rank correlation between reference variants.  This is the")
            print("  ceiling: no model can be resolved against this reference more")
            print("  finely than the reference agrees with itself.\n")
            print("    " + " " * 20 + "".join(f"{n[:9]:>10}" for n in names))
            for row_name in names:
                cells = "".join(
                    f"{matrix[row_name].get(col, float('nan')):>10.3f}" for col in names
                )
                print(f"    {row_name:<20}{cells}")
            off_diagonal = [
                matrix[a][b] for a in names for b in names
                if a != b and np.isfinite(matrix[a].get(b, float("nan")))
            ]
            if off_diagonal:
                print(f"\n    lowest agreement between two variants: "
                      f"{min(off_diagonal):.3f}")
        print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    results: dict[str, Any] = {"runs": {}}
    for directory in sorted(args.root.iterdir()):
        if not (directory.is_dir() and (directory / "frames.csv").exists()):
            continue
        if "point" not in directory.name:
            continue
        logger.info("measuring %s", directory.name)
        entry = analyse(directory)
        if entry:
            results["runs"][directory.name] = entry
    if not results["runs"]:
        parser.error("no point-source recordings found")

    results["provenance"] = provenance(
        None, variants=[v.name for v in VARIANTS],
        recordings=list(results["runs"]),
    )
    print_report(results)
    if args.json:
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
