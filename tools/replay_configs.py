"""Replay recordings through the real evaluator under several metric sets.

The offline comparison in :mod:`tools.protocol_study` normalises every measure
over the whole recording.  That is the right way to compare *measures*, but it
is not what the library does: online it fits a rolling window, warms up, and
runs a confidence-gated temporal filter.  A metric set that looks better
offline could easily be worse online, because the rolling normaliser and the
filter interact with how noisy each metric is.

This tool therefore answers the only question that can justify changing a
default: feed the same frames, in the same order, to the shipped
:class:`SharpnessEvaluator` under different ``metrics.enabled`` sets, and score
what actually comes out.

Every configuration sees exactly the same frames in the same order, with fresh
state per recording, so the comparison isolates the metric set.

    python3 tools/replay_configs.py data --json data/replay_configs.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import SharpnessConfig, SharpnessEvaluator, load_default_config  # noqa: E402
from adaptive_sharpness.types import ROI  # noqa: E402
from tools.protocol_study import (  # noqa: E402
    adjacent_discrimination,
    count_peaks,
    monotonicity,
    rank_eta_squared,
    saturated_fraction,
    step_profile,
)

logger = logging.getLogger("replay_configs")

#: Metric sets to replay.  Only subsets of the library's own six are possible
#: here: the published baselines live in tools/baselines.py and are not part of
#: the package, so they cannot be switched on through configuration.
CONFIGS: dict[str, tuple[str, ...]] = {
    "own6": ("laplacian", "tenengrad", "brenner", "wavelet", "fourier", "edge_width"),
    "no_edge_width": ("laplacian", "tenengrad", "brenner", "wavelet", "fourier"),
    "no_fourier": ("laplacian", "tenengrad", "brenner", "wavelet", "edge_width"),
    "lean4": ("laplacian", "tenengrad", "brenner", "wavelet"),
    "trio": ("tenengrad", "brenner", "wavelet"),
    "solo_tenengrad": ("tenengrad",),
}


@dataclass(frozen=True)
class RunContext:
    """What a recording says about itself, beyond the pixels.

    A replay that ignores this is not reproducing the run, it is running the
    same frames through a fresh evaluator - which is a different experiment and
    has to be labelled as one.
    """

    files: list[str]
    step: np.ndarray
    hold: np.ndarray
    rois: list[ROI | None]
    timestamps: np.ndarray
    live_instantaneous: np.ndarray
    live_filtered: np.ndarray

    @property
    def has_roi(self) -> bool:
        return any(roi is not None for roi in self.rois)

    def summary(self) -> dict[str, object]:
        gaps = np.diff(self.timestamps[np.isfinite(self.timestamps)])
        return {
            "frames": len(self.files),
            "roi_recorded": self.has_roi,
            "median_interval_s": float(np.median(gaps)) if gaps.size else float("nan"),
            "max_interval_s": float(np.max(gaps)) if gaps.size else float("nan"),
            "monotone_timestamps": bool(gaps.size == 0 or np.all(gaps > 0)),
        }


def _maybe_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value) if value not in ("", None) else float("nan")
    except ValueError:
        return float("nan")


def load_context(directory: Path) -> RunContext:
    """Frame names, protocol labels, ROI, timestamps and the live scores."""
    rows = [
        r for r in csv.DictReader((directory / "frames.csv").open(newline=""))
        if r.get("image_file")
    ]
    rois: list[ROI | None] = []
    for row in rows:
        values = [_maybe_float(row, f"roi_{k}") for k in ("x", "y", "w", "h")]
        rois.append(
            ROI(int(values[0]), int(values[1]), int(values[2]), int(values[3]))
            if all(np.isfinite(values)) and values[2] > 0 and values[3] > 0
            else None
        )
    return RunContext(
        files=[str(r["image_file"]) for r in rows],
        step=np.array([int(r.get("step_index") or 0) for r in rows], dtype=np.int32),
        hold=np.array([r.get("step_phase") == "hold" for r in rows], dtype=bool),
        rois=rois,
        timestamps=np.array([_maybe_float(r, "timestamp") for r in rows]),
        live_instantaneous=np.array([
            _maybe_float(r, "instantaneous_score") for r in rows
        ]),
        live_filtered=np.array([_maybe_float(r, "filtered_score") for r in rows]),
    )


def load_labels(directory: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Backwards-compatible view of :func:`load_context`."""
    context = load_context(directory)
    return context.files, context.step, context.hold


def load_frames(directory: Path, files: Sequence[str]) -> list[np.ndarray]:
    """Decode a recording once, so that every configuration replays the same
    pixels without paying for the decode again."""
    images = []
    for name in files:
        image = cv2.imread(str(directory / "frames" / name), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"unreadable frame: {directory / 'frames' / name}")
        images.append(image)
    return images


def replay(
    images: Sequence[np.ndarray],
    config: SharpnessConfig,
    names: Sequence[str],
    rois: Sequence[ROI | None] | None = None,
) -> dict[str, np.ndarray]:
    """Run one recording through a freshly initialised evaluator.

    This is a **recomputation**, not the live result: the evaluator starts with
    no history, so warm-up and the scale calibration happen again.  The live
    scores are carried separately in :class:`RunContext`, and the two must not
    be conflated in a report.
    """
    evaluator = SharpnessEvaluator(
        replace(config, metrics=replace(config.metrics, enabled=tuple(names)))
    )
    fields = (
        "instantaneous", "filtered", "confidence", "elapsed",
        "informative", "ready", "noise_sigma",
    )
    out = {field: np.empty(len(images)) for field in fields}

    for index, image in enumerate(images):
        roi = rois[index] if rois is not None else None
        started = time.perf_counter()
        result = evaluator.evaluate(image, roi)
        out["elapsed"][index] = time.perf_counter() - started
        out["instantaneous"][index] = result.instantaneous_score
        out["filtered"][index] = result.filtered_score
        out["confidence"][index] = result.confidence
        out["informative"][index] = result.informative_fraction
        out["ready"][index] = float(result.ready)
        out["noise_sigma"][index] = result.stats.noise_sigma

    return out


def score(series: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    steps, medians, scatters = step_profile(series, labels)
    peak = int(np.argmax(medians))
    return {
        "adj": adjacent_discrimination(series, labels),
        "eta2": rank_eta_squared(series, labels),
        "sat": saturated_fraction(series),
        "mono": monotonicity(medians, peak),
        "peaks": float(count_peaks(medians, scatters)),
        "peak_step": float(steps[peak]),
    }


def run(directories: Sequence[Path], config: SharpnessConfig) -> dict[str, Any]:
    results: dict[str, Any] = {"runs": {}, "configs": {k: list(v) for k, v in CONFIGS.items()}}
    for directory in directories:
        files, step, hold = load_labels(directory)
        selected = hold & (step > 0)
        if selected.sum() < 20:
            logger.info("%s: no protocol labels, skipped", directory.name)
            continue
        labels = step[selected]
        logger.info("%s: decoding %d frames", directory.name, len(files))
        images = load_frames(directory, files)
        entry: dict[str, Any] = {}
        for config_name, names in CONFIGS.items():
            logger.info("%s: replaying %s", directory.name, config_name)
            output = replay(images, config, names)
            entry[config_name] = {
                "instantaneous": score(output["instantaneous"][selected], labels),
                "filtered": score(output["filtered"][selected], labels),
                "ms_per_frame": float(np.median(output["elapsed"]) * 1000.0),
                "ms_p95": float(np.percentile(output["elapsed"], 95) * 1000.0),
                "confidence_median": float(np.median(output["confidence"])),
                "confidence_zero_fraction": float(np.mean(output["confidence"] <= 1e-9)),
                "ready_fraction": float(np.mean(np.isfinite(output["filtered"]))),
            }
        results["runs"][directory.name] = entry
    results["aggregate"] = aggregate(results["runs"])
    return results


def aggregate(runs: dict[str, Any]) -> dict[str, Any]:
    """Mean over recordings, plus the mean within-recording rank.

    Same reasoning as the protocol study: recordings differ in difficulty, so
    the rank is what transfers and the mean is only context.
    """
    summary: dict[str, Any] = {}
    if not runs:
        return summary
    config_names = list(next(iter(runs.values())))
    for stage in ("instantaneous", "filtered"):
        for config_name in config_names:
            key = f"{config_name}:{stage}"
            summary[key] = {
                field: float(np.mean([
                    runs[r][config_name][stage][field] for r in runs
                ]))
                for field in ("adj", "eta2", "sat", "mono", "peaks")
            }
            summary[key]["ms_per_frame"] = float(np.mean([
                runs[r][config_name]["ms_per_frame"] for r in runs
            ]))
            summary[key]["confidence_zero_fraction"] = float(np.mean([
                runs[r][config_name]["confidence_zero_fraction"] for r in runs
            ]))

    for field in ("adj", "eta2"):
        for run_name in runs:
            values = {
                f"{c}:{s}": runs[run_name][c][s][field]
                for c in config_names for s in ("instantaneous", "filtered")
            }
            order = sorted(values, key=lambda k: -values[k])
            for position, key in enumerate(order, 1):
                summary[key].setdefault(f"{field}_ranks", []).append(position)
    for key, row in summary.items():
        for field in ("adj", "eta2"):
            ranks = row.pop(f"{field}_ranks", [])
            row[f"{field}_rank"] = float(np.mean(ranks)) if ranks else float("nan")
    return summary


def print_report(results: dict[str, Any]) -> None:
    print("=" * 78)
    print("REPLAY - the shipped evaluator, online, under different metric sets")
    print("=" * 78)
    print(f"\n{len(results['runs'])} recordings\n")
    for name, members in results["configs"].items():
        print(f"  {name:<16} {', '.join(members)}")

    summary = results["aggregate"]
    order = sorted(summary, key=lambda k: summary[k]["adj_rank"])
    print("\n--- Pooled, by mean rank of adjacent-step discrimination ---\n")
    header = (
        f"{'config:stage':<28}{'adj':>7}{'rank':>6}{'eta2':>7}{'sat':>7}"
        f"{'mono':>7}{'peaks':>7}{'ms':>8}{'conf=0':>8}"
    )
    print(header)
    print("-" * len(header))
    for key in order:
        row = summary[key]
        print(
            f"{key:<28}{row['adj']:>7.3f}{row['adj_rank']:>6.1f}{row['eta2']:>7.3f}"
            f"{row['sat']:>7.3f}{row['mono']:>7.3f}{row['peaks']:>7.2f}"
            f"{row['ms_per_frame']:>8.2f}{row['confidence_zero_fraction']:>8.3f}"
        )

    print("\n--- Filtered score, adjacent-step discrimination per recording ---\n")
    config_names = list(results["configs"])
    header = f"{'recording':<32}" + "".join(f"{c[:12]:>13}" for c in config_names)
    print(header)
    print("-" * len(header))
    for run_name, entry in results["runs"].items():
        cells = "".join(
            f"{entry[c]['filtered']['adj']:>13.3f}" for c in config_names
        )
        print(f"{run_name:<32}{cells}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )

    directories = sorted(
        d for d in args.root.iterdir()
        if d.is_dir() and (d / "frames.csv").exists() and (d / "frames").is_dir()
    )
    if args.only:
        directories = [d for d in directories if any(f in d.name for f in args.only)]
    if not directories:
        parser.error(f"no recordings found under {args.root}")

    results = run(directories, load_default_config())
    print_report(results)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2, default=float))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
