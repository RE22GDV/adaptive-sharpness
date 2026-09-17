"""The research programme of docs/STUDY_UA_REPORT.md, one subcommand each.

Д01-Д06 were answered from the reports already published.  This module runs
the rest, Д07-Д20, which all need the recorded frames rather than a summary of
them.  Each writes `reports/study_dNN.json` with its own provenance, and
`tools/render_results.py` turns those into tables.

Frames are private
    The recordings are photographs of the operator's room.  Nothing here writes
    an image, and every output is a measurement.  `--data` points at wherever
    the frames are; a local copy is roughly six times faster than the share
    they live on, which matters when a study replays the corpus forty times.

Two things this module does that the older tools do not:

*   **It captures the weights.**  `replay_in_context` returns per-frame scores
    only, so the reliability model could be argued about but not watched.  Д12
    asks what the weights actually do, so `rich_replay` keeps every metric's
    weight, reliability and normalised value alongside the frame statistics
    that are supposed to drive them.

*   **It runs variants in parallel.**  A study with forty configurations over
    eight recordings is twenty-five minutes in one process and a couple of
    minutes across every core, with OpenCV's own threading turned off inside
    each worker so the processes do not fight over the same cores.

    python3 tools/programme.py d12 --data D:/camera_focus_data
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

import cv2  # noqa: E402

from adaptive_sharpness import (  # noqa: E402
    SharpnessConfig,
    SharpnessEvaluator,
    load_default_config,
)
from tools.evaluation import (  # noqa: E402
    load_measure_cache,
    provenance,
    select_frames,
)
from tools.ablation import spearman  # noqa: E402
from tools.protocol_study import (  # noqa: E402
    adjacent_discrimination,
    saturated_fraction,
    step_profile,
)
from tools.replay_configs import RunContext, load_context, load_frames  # noqa: E402

logger = logging.getLogger("programme")

#: Where the frames are.  The default is the share; a local copy is passed with
#: --data and is the same bytes, verified by file count and total size.
DEFAULT_DATA = _ROOT / "data"

#: The eight recordings of the main corpus, in the order the report lists them.
#: `run_20260909_104332` and `selftest` are earlier and smaller and are not
#: part of it; a study that silently included them would not be comparable
#: with anything already published.
CORPUS: tuple[str, ...] = (
    "cond_20260916_110502", "cond_20260916_110600",
    "cond_20260916_110656", "cond_20260916_110746",
    "point_source_20260916_111545", "point_source_20260916_111724",
    "sweep_plain_20260916_110955", "sweep_texture_20260916_110108",
)

#: Recordings carrying the point-source reference, so a study can say which of
#: its numbers rest on two recordings and which on eight.
WITH_REFERENCE: tuple[str, ...] = (
    "point_source_20260916_111545", "point_source_20260916_111724",
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

@dataclass
class Recording:
    """One recording, decoded once and replayed many times."""

    name: str
    directory: Path
    context: RunContext
    images: list[np.ndarray]
    mask: np.ndarray
    labels: np.ndarray
    truth: np.ndarray | None          # -radius, so larger is sharper
    selection: dict[str, Any]

    @property
    def has_reference(self) -> bool:
        return self.truth is not None


def load_recording(directory: Path, *, require_reference: bool = False) -> Recording | None:
    """Decode a recording and work out which of its frames are measurable.

    Frame selection goes through `tools.evaluation.select_frames`, the same
    definition every published study used, so a number from here can be put
    next to a number from there.
    """
    context = load_context(directory)
    cached = load_measure_cache(directory, context.files)
    radius = cached.spot_radius
    offset = cached.spot_offset
    selection = select_frames(
        step=context.step, hold=context.hold,
        spot_radius=radius, spot_offset=offset,
        require_reference=require_reference,
    )
    if selection.count < 20:
        logger.warning("%s: only %d measurable frames", directory.name, selection.count)
        return None
    if require_reference and radius is None:
        return None
    mask = selection.mask
    return Recording(
        name=directory.name,
        directory=directory,
        context=context,
        images=load_frames(directory, context.files),
        mask=mask,
        labels=context.step[mask],
        truth=(-np.asarray(radius)[mask]) if radius is not None else None,
        selection=selection.summary(),
    )


def corpus_directories(data: Path, names: Sequence[str] = CORPUS) -> list[Path]:
    out = []
    for name in names:
        directory = data / name
        if not (directory / "frames.csv").exists():
            raise SystemExit(f"missing recording: {directory}")
        out.append(directory)
    return out


# ---------------------------------------------------------------------------
# Replay
# ---------------------------------------------------------------------------

#: Per-frame fields every study gets.
_SCALAR_FIELDS = (
    "instantaneous", "filtered", "confidence", "informative", "ready",
    "noise_sigma", "noise_level", "snr", "local_contrast", "brightness",
    "clipped_high", "clipped_low", "edge_density", "motion_px", "elapsed",
    "scale_frozen", "score_change",
)

#: Per-metric fields, captured as (frames, metrics) arrays.  These are what
#: the reliability model actually produced, as opposed to what it is supposed
#: to produce, and no published study had ever looked at them.
_METRIC_FIELDS = ("normalized", "weight", "reliability", "agreement", "raw")


def rich_replay(
    images: Sequence[np.ndarray],
    context: RunContext,
    config: SharpnessConfig,
    *,
    start: int = 0,
    order: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Replay with the weights kept.

    ``start`` drops the first N frames instead of feeding them, which is how
    Д10 asks what the same frame scores when the evaluator has seen a different
    history.  ``order`` feeds the frames in an explicit order, for the reversed
    diagnostic.  Both leave the output indexed by *original* frame position, so
    a comparison is always between the same frames.
    """
    import time

    evaluator = SharpnessEvaluator(config)
    names = tuple(config.metrics.enabled)
    total = len(images)
    indices = list(order) if order is not None else list(range(start, total))

    out: dict[str, Any] = {
        field: np.full(total, np.nan) for field in _SCALAR_FIELDS
    }
    for field in _METRIC_FIELDS:
        out[field] = np.full((total, len(names)), np.nan)
    out["names"] = names
    out["fed"] = np.zeros(total, dtype=bool)

    for index in indices:
        roi = context.rois[index] if context.has_roi else None
        started = time.perf_counter()
        result = evaluator.evaluate(images[index], roi)
        elapsed = time.perf_counter() - started

        out["fed"][index] = True
        out["elapsed"][index] = elapsed
        out["instantaneous"][index] = result.instantaneous_score
        out["filtered"][index] = result.filtered_score
        out["confidence"][index] = result.confidence
        out["informative"][index] = result.informative_fraction
        out["ready"][index] = float(result.ready)
        out["scale_frozen"][index] = float(result.scale_frozen)
        out["score_change"][index] = float(result.score_change_detected)
        stats = result.stats
        for field in ("noise_sigma", "noise_level", "snr", "local_contrast",
                      "brightness", "clipped_high", "clipped_low",
                      "edge_density", "motion_px"):
            out[field][index] = float(getattr(stats, field, np.nan))
        for position, sample in enumerate(result.metrics):
            if position >= len(names):
                break
            for field in _METRIC_FIELDS:
                out[field][index, position] = float(getattr(sample, field))
    return out


def score_against(series: np.ndarray, labels: np.ndarray,
                  truth: np.ndarray | None) -> dict[str, float]:
    """The same three criteria every published table uses."""
    row: dict[str, float] = {
        "adj": adjacent_discrimination(series, labels),
        "sat": saturated_fraction(series),
    }
    steps, medians, _ = step_profile(series, labels)
    row["peak_step"] = float(steps[int(np.argmax(medians))]) if len(steps) else float("nan")
    if truth is not None:
        row["spearman_gt"] = spearman(series, truth)
        _, truth_medians, _ = step_profile(truth, labels)
        row["peak_err"] = abs(
            float(steps[int(np.argmax(medians))])
            - float(steps[int(np.argmax(truth_medians))])
        ) if len(steps) else float("nan")
    return row


# ---------------------------------------------------------------------------
# Parallel execution
# ---------------------------------------------------------------------------

def _worker_init() -> None:
    # Each worker replays one variant of one recording.  Letting OpenCV also
    # spawn a thread per core inside every worker oversubscribes the machine
    # badly and made a parallel run slower than a serial one.
    cv2.setNumThreads(1)
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"


def parallel_map(
    function: Callable[..., Any],
    jobs: Iterable[tuple],
    *,
    workers: int | None = None,
) -> list[Any]:
    """Run independent replays across processes, in submission order."""
    jobs = list(jobs)
    if not jobs:
        return []
    workers = workers or min(len(jobs), max(1, os.cpu_count() or 4))
    if workers == 1:
        _worker_init()
        return [function(*job) for job in jobs]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init
    ) as pool:
        return list(pool.map(function, *zip(*jobs)))


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_report(
    name: str, payload: dict[str, Any], config: SharpnessConfig, *,
    reports: Path, recordings: Sequence[str],
) -> Path:
    """Write the JSON *before* printing anything.

    An hour of computation was once lost because a table was printed first and
    the console could not encode a character in the header.
    """
    reports.mkdir(parents=True, exist_ok=True)
    payload = dict(payload)
    payload["provenance"] = provenance(config, recordings=list(recordings))
    path = reports / f"{name}.json"
    path.write_text(
        json.dumps(payload, indent=2, default=_jsonable), encoding="utf-8"
    )
    return path


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return float(value)


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA,
                        help="directory holding the recordings")
    parser.add_argument("--reports", type=Path, default=_ROOT / "reports")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(message)s",
    )


# ---------------------------------------------------------------------------
# Variants
# ---------------------------------------------------------------------------

def build_config(base: SharpnessConfig, overrides: dict[str, Any]) -> SharpnessConfig:
    """Apply dotted overrides such as ``{"metrics.enabled": ("wavelet",)}``.

    Studies are described by what they change rather than by a whole
    configuration object, for two reasons: a dict survives being sent to a
    worker process, and a variant that differs from the default in one field
    reads as one field in the report instead of as a wall of settings.
    """
    sections: dict[str, dict[str, Any]] = {}
    for path, value in overrides.items():
        section, _, field = path.partition(".")
        if not field:
            raise ValueError(f"override must be section.field, got {path!r}")
        if not hasattr(base, section):
            raise ValueError(f"no configuration section {section!r}")
        if not hasattr(getattr(base, section), field):
            raise ValueError(f"{section} has no field {field!r}")
        sections.setdefault(section, {})[field] = value
    updated = {
        name: replace(getattr(base, name), **fields)
        for name, fields in sections.items()
    }
    return replace(base, **updated)


def run_variants_for_recording(
    directory: Path,
    variants: Sequence[tuple[str, dict[str, Any]]],
    *,
    require_reference: bool = False,
    keep: Sequence[str] = ("filtered",),
) -> dict[str, Any]:
    """Replay several configurations against one recording, decoding it once.

    This is the unit of parallel work.  Splitting by variant instead would
    decode the same recording once per variant, which for the metric-subset
    study is 63 decodes of the same 1656 frames.
    """
    _worker_init()
    recording = load_recording(directory, require_reference=require_reference)
    if recording is None:
        return {"name": directory.name, "skipped": "too few measurable frames"}
    base = load_default_config()
    out: dict[str, Any] = {
        "name": recording.name,
        "selection": recording.selection,
        "has_reference": recording.has_reference,
        "variants": {},
    }
    for name, overrides in variants:
        config = build_config(base, overrides)
        result = rich_replay(recording.images, recording.context, config)
        series = result["filtered"][recording.mask]
        row = score_against(series, recording.labels, recording.truth)
        row["ms"] = float(np.nanmedian(result["elapsed"]) * 1000.0)
        row["ready"] = float(np.nanmean(result["ready"][recording.mask]))
        row["confidence"] = float(np.nanmean(result["confidence"][recording.mask]))
        frozen = result["scale_frozen"]
        fired = np.flatnonzero(np.nan_to_num(frozen) > 0)
        row["froze"] = float(fired.size > 0)
        row["freeze_frame"] = float(fired[0]) if fired.size else float("nan")
        row["frozen_fraction"] = float(np.nanmean(np.nan_to_num(frozen)))
        for field in keep:
            if field in result and result[field].ndim == 1:
                row[f"series_{field}"] = result[field][recording.mask].tolist()
        out["variants"][name] = row
    return out


def _chunk(items: Sequence[Any], parts: int) -> list[list[Any]]:
    parts = max(1, min(parts, len(items)))
    size = (len(items) + parts - 1) // parts
    return [list(items[i:i + size]) for i in range(0, len(items), size)]


def sweep(
    directories: Sequence[Path],
    variants: Sequence[tuple[str, dict[str, Any]]],
    *,
    workers: int | None = None,
    require_reference: bool = False,
    keep: Sequence[str] = (),
) -> dict[str, Any]:
    """Every variant against every recording, spread over the whole machine."""
    workers = workers or (os.cpu_count() or 8)
    per_recording = max(1, workers // max(1, len(directories)))
    jobs: list[tuple] = []
    for directory in directories:
        for part in _chunk(variants, per_recording):
            jobs.append((directory, part, require_reference, keep))

    logger.info("%d jobs over %d workers", len(jobs), workers)
    results = parallel_map(_sweep_job, jobs, workers=min(workers, len(jobs)))

    runs: dict[str, Any] = {}
    for entry in results:
        if "skipped" in entry:
            continue
        run = runs.setdefault(entry["name"], {
            "selection": entry["selection"],
            "has_reference": entry["has_reference"],
            "variants": {},
        })
        run["variants"].update(entry["variants"])
    return runs


def _sweep_job(directory, variants, require_reference, keep):
    return run_variants_for_recording(
        directory, variants, require_reference=require_reference, keep=keep
    )


def aggregate_variants(runs: dict[str, Any]) -> dict[str, Any]:
    """Average within recordings first, and record how many contributed.

    Thousands of frames from one recording are not independent repetitions, and
    a criterion defined only where the point-source reference exists covers two
    recordings while the rest cover eight.  Both counts are carried per field,
    because a published table of this project once mixed them.
    """
    names: list[str] = []
    for run in runs.values():
        for name in run["variants"]:
            if name not in names:
                names.append(name)

    summary: dict[str, Any] = {}
    for name in names:
        rows = [r["variants"][name] for r in runs.values() if name in r["variants"]]
        fields = {
            key for row in rows for key, value in row.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        entry: dict[str, Any] = {}
        for field in sorted(fields):
            values = [
                float(row[field]) for row in rows
                if field in row and np.isfinite(row[field])
            ]
            if values:
                entry[field] = float(np.mean(values))
                entry[f"{field}_spread"] = float(np.max(values) - np.min(values))
                entry[f"{field}_n"] = len(values)
        summary[name] = entry
    return summary


def main(argv: list[str] | None = None) -> int:
    from tools.studies import STUDIES, TITLES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "studies", nargs="+",
        help="study codes such as d12, or 'all' for every one of them",
    )
    add_common_arguments(parser)
    args = parser.parse_args(argv)
    configure_logging(args.verbose)

    names = sorted(STUDIES) if "all" in args.studies else list(args.studies)
    unknown = [n for n in names if n not in STUDIES]
    if unknown:
        parser.error(f"unknown studies: {unknown}; have {sorted(STUDIES)}")

    config = load_default_config()
    for name in names:
        logger.info("running %s - %s", name, TITLES[name])
        payload = STUDIES[name](args.data, args.workers)
        payload["study"] = name
        payload["title"] = TITLES[name]
        payload["data_root"] = str(args.data)
        path = write_report(
            f"study_{name}", payload, config,
            reports=args.reports, recordings=list(payload.get("runs", {})),
        )
        # ASCII only: a console that cannot encode the title must not be able
        # to destroy a finished run, which is how an hour of compute was lost.
        print(f"{name}: wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
