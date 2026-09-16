"""Shared frame selection, ordering statistics and run provenance.

Three tools used to answer the same question with quietly different code:
`protocol_study.py`, `ablation.py` and `before_after.py` each decided for
themselves which frames counted and how to score an ordering.  They disagreed
on both, which meant a number from one could not be compared with a number from
another - measured at 3.3% and 5.5% of frames on the point-source runs.

Everything that decides *what is measured* now lives here, so that a difference
between two reported numbers is a difference between the methods and not
between the scripts.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "SPOT_PARAMETERS",
    "CachedMeasures",
    "load_measure_cache",
    "FrameSelection",
    "select_frames",
    "OrderingResult",
    "ordering_against_truth",
    "provenance",
    "config_fingerprint",
    "file_fingerprint",
]


# ---------------------------------------------------------------------------
# Cache loading
# ---------------------------------------------------------------------------

#: Parameters the point-source measurement depends on.  Stored with the cache,
#: because changing any of them changes the reference.
SPOT_PARAMETERS: dict[str, float] = {"half": 80, "fraction": 0.5, "blur": 9}


@dataclass(frozen=True)
class CachedMeasures:
    """A verified measure cache, or a reason it could not be used."""

    path: Path
    frames: int
    raw: Mapping[str, Any] | None
    spot_radius: Any | None
    spot_offset: Any | None
    note: str

    @property
    def has_reference(self) -> bool:
        return self.spot_radius is not None


def load_measure_cache(
    directory: Path,
    files: Sequence[str],
    *,
    processing_fingerprint: str | None = None,
    require_raw: bool = False,
) -> CachedMeasures:
    """Load a measure cache only if it describes these frames and this code.

    Three scripts used to do this themselves and check different things: one
    verified the frame count and the spot parameters, another only the frame
    count.  A cache written before a change to `analysis_width` or
    `noise_quantile` therefore passed one reader and failed another, and the
    numbers from the two could not be compared.

    ``processing_fingerprint`` is required to trust the *raw metric* arrays,
    which depend on the whole processing configuration.  The *spot* reference
    does not: it is measured from the source frames directly, so it needs only
    the frames and its own parameters.  Keeping the two separate means a change
    to the metrics does not needlessly invalidate the reference.
    """
    import numpy as np

    path = directory / "measures_cache.npz"
    if not path.exists():
        return CachedMeasures(path, len(files), None, None, None, "no cache")

    loaded = np.load(path, allow_pickle=False)
    if int(loaded["n"]) != len(files):
        return CachedMeasures(
            path, len(files), None, None, None,
            f"cache holds {int(loaded['n'])} frames, recording has {len(files)}",
        )

    expected_files = file_fingerprint([directory / "frames" / name for name in files])
    stored = str(loaded["file_fingerprint"]) if "file_fingerprint" in loaded else None
    if stored is None:
        return CachedMeasures(
            path, len(files), None, None, None,
            "cache predates input fingerprinting; rebuild with --refresh",
        )
    if stored != expected_files:
        return CachedMeasures(
            path, len(files), None, None, None,
            "cache was built from different frames or a different order",
        )

    spot_expected = json.dumps(SPOT_PARAMETERS, sort_keys=True)
    spot_stored = (
        str(loaded["spot_fingerprint"]) if "spot_fingerprint" in loaded else None
    )
    spot_ok = (
        "spot_radius" in loaded
        and loaded["spot_radius"].size == len(files)
        and spot_stored == spot_expected
    )

    raw = None
    note_parts = []
    if processing_fingerprint is not None:
        stored_processing = (
            str(loaded["processing_fingerprint"])
            if "processing_fingerprint" in loaded else None
        )
        if stored_processing == processing_fingerprint:
            raw = {
                key[4:]: loaded[key] for key in loaded.files if key.startswith("raw_")
            }
            note_parts.append("raw metrics verified")
        else:
            note_parts.append("raw metrics stale, not used")
    if require_raw and raw is None:
        return CachedMeasures(
            path, len(files), None, None, None,
            "raw metrics required but stale or absent",
        )

    note_parts.append(
        "reference verified" if spot_ok else "reference stale or absent"
    )
    return CachedMeasures(
        path=path,
        frames=len(files),
        raw=raw,
        spot_radius=loaded["spot_radius"] if spot_ok else None,
        spot_offset=loaded["spot_offset"] if spot_ok else None,
        note="; ".join(note_parts),
    )


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FrameSelection:
    """Which frames are being measured, and why the others are not."""

    mask: np.ndarray
    reasons: Mapping[str, int]
    total: int

    @property
    def count(self) -> int:
        return int(self.mask.sum())

    def summary(self) -> dict[str, Any]:
        return {
            "total_frames": self.total,
            "selected": self.count,
            "excluded": dict(self.reasons),
        }


def select_frames(
    *,
    step: np.ndarray,
    hold: np.ndarray,
    spot_radius: np.ndarray | None = None,
    spot_offset: np.ndarray | None = None,
    max_spot_offset: float = 20.0,
    require_reference: bool = False,
) -> FrameSelection:
    """The one definition of a measurable frame.

    A frame counts when the operator was holding the ring still at a known
    protocol step.  When a run carries the point-source reference, a frame also
    has to have a usable reference: a finite radius, and a spot that has not
    wandered away from where it normally sits - if the brightest thing in the
    window was not the source, the reference describes something else.

    ``require_reference`` makes the reference mandatory, so that a comparison
    involving ground truth uses exactly the same frames as one that does not.
    """
    total = int(step.size)
    reasons: dict[str, int] = {}

    mask = np.ones(total, dtype=bool)

    def drop(name: str, keep: np.ndarray) -> None:
        nonlocal mask
        removed = int(np.sum(mask & ~keep))
        if removed:
            reasons[name] = reasons.get(name, 0) + removed
        mask = mask & keep

    drop("moving", hold.astype(bool))
    drop("no_step_label", step > 0)

    has_reference = spot_radius is not None and spot_radius.size == total
    if has_reference:
        assert spot_radius is not None
        drop("reference_not_finite", np.isfinite(spot_radius))
        if spot_offset is not None and spot_offset.size == total:
            drop("spot_wandered", spot_offset <= max_spot_offset)
    elif require_reference:
        drop("no_reference", np.zeros(total, dtype=bool))

    return FrameSelection(mask=mask, reasons=reasons, total=total)


# ---------------------------------------------------------------------------
# Ordering against a reference
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class OrderingResult:
    """How a model orders pairs of steps against a reference.

    Ties are counted, not silently resolved.  The previous implementation took
    ``model[i] > model[j]`` as the model's answer, so a tie always read as "j is
    sharper": a constant signal scored 0.000 wrong pairs against one reference
    direction and 1.000 against the other, while in truth it distinguishes
    nothing at all.
    """

    correct: int
    wrong: int
    tied: int
    comparable: int
    reference_tied: int

    @property
    def inversion(self) -> float:
        """Fraction of comparable pairs ordered the wrong way, ties at half.

        Ties are charged 0.5 because a tie is neither right nor wrong and a
        signal that ties everything should land at chance, not at either
        extreme.  ``inversion_strict`` and ``resolved`` are reported alongside
        so that the two effects can be separated.
        """
        if self.comparable == 0:
            return float("nan")
        return (self.wrong + 0.5 * self.tied) / self.comparable

    @property
    def inversion_strict(self) -> float:
        """Wrong pairs among those the model actually ordered."""
        decided = self.correct + self.wrong
        if decided == 0:
            return float("nan")
        return self.wrong / decided

    @property
    def resolved(self) -> float:
        """Fraction of comparable pairs the model gave any answer to."""
        if self.comparable == 0:
            return float("nan")
        return (self.correct + self.wrong) / self.comparable

    def to_dict(self) -> dict[str, float]:
        return {
            "inversion": self.inversion,
            "inversion_strict": self.inversion_strict,
            "resolved": self.resolved,
            "pairs_correct": float(self.correct),
            "pairs_wrong": float(self.wrong),
            "pairs_tied": float(self.tied),
            "pairs_comparable": float(self.comparable),
            "pairs_reference_tied": float(self.reference_tied),
        }


def ordering_against_truth(
    model: np.ndarray, reference: np.ndarray, *, tolerance: float = 1e-12
) -> OrderingResult:
    """Compare every pair of per-step values against a reference ordering.

    Both arrays are per-step summaries in the same step order, and both are
    oriented so that larger means sharper.
    """
    correct = wrong = tied = reference_tied = 0
    size = model.size
    for i in range(size):
        for j in range(i + 1, size):
            if not (np.isfinite(model[i]) and np.isfinite(model[j])):
                continue
            if not (np.isfinite(reference[i]) and np.isfinite(reference[j])):
                continue
            if abs(reference[i] - reference[j]) <= tolerance:
                reference_tied += 1
                continue
            if abs(model[i] - model[j]) <= tolerance:
                tied += 1
                continue
            if (reference[i] > reference[j]) == (model[i] > model[j]):
                correct += 1
            else:
                wrong += 1
    return OrderingResult(
        correct=correct, wrong=wrong, tied=tied,
        comparable=correct + wrong + tied, reference_tied=reference_tied,
    )


def peak_interval(values: np.ndarray, *, tolerance: float = 1e-9) -> tuple[int, int]:
    """First and last index within ``tolerance`` of the maximum.

    ``argmax`` answers "where is the first maximum", which is a different
    question from "where is the peak" whenever the profile has a plateau - and
    a flat top is exactly what a well-behaved focus curve has near best focus.
    """
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return -1, -1
    top = float(np.nanmax(values))
    at_top = np.flatnonzero(np.isfinite(values) & (values >= top - tolerance))
    return int(at_top[0]), int(at_top[-1])


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str, separators=(",", ":"))


def config_fingerprint(config: Any) -> str:
    """A hash of every configuration field that can change a measurement."""
    if hasattr(config, "__dataclass_fields__"):
        payload = {
            name: config_fingerprint(getattr(config, name))
            if hasattr(getattr(config, name), "__dataclass_fields__")
            else getattr(config, name)
            for name in sorted(config.__dataclass_fields__)
        }
    else:
        payload = config
    return hashlib.sha256(_stable_json(payload).encode()).hexdigest()[:16]


def file_fingerprint(paths: Sequence[Path], *, sample_bytes: int = 65536) -> str:
    """A hash of the input files: names, sizes, and the head of each.

    Full content hashing of 9 000 PNGs costs more than it is worth on every
    run; name, size and the first block catch a replaced or re-encoded frame,
    which is the failure this is guarding against.
    """
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.name).encode())
        try:
            digest.update(str(path.stat().st_size).encode())
            with path.open("rb") as handle:
                digest.update(handle.read(sample_bytes))
        except OSError:
            digest.update(b"<unreadable>")
    return digest.hexdigest()[:16]


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parents[1],
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _git_state() -> tuple[bool, int]:
    """Whether tracked files differ from the commit, and how many are untracked.

    Untracked files are counted, not treated as dirt.  A tool writes its own
    report while it runs, and that report is untracked until it is committed -
    so counting it made every run flag itself as dirty and the flag stopped
    meaning anything.  What matters for reproducing a number is whether the
    *tracked* source differed from the commit.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        tracked = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip()
        untracked = subprocess.run(
            ["git", "ls-files", "--others", "--exclude-standard"],
            capture_output=True, text=True, check=True, cwd=root,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return False, 0
    return bool(tracked), len([line for line in untracked.splitlines() if line])


def provenance(config: Any = None, **extra: Any) -> dict[str, Any]:
    """Everything needed to say what produced a report.

    Without this a JSON report is a table of numbers with no way to tell which
    code and which settings made them - which is how two ablation groups came
    to be run under different defaults with nothing in the output to show it.
    """
    import cv2

    tracked_modified, untracked = _git_state()
    record: dict[str, Any] = {
        "commit": _git_commit(),
        "tracked_files_modified": tracked_modified,
        "untracked_files": untracked,
        # Kept under its old name so that older reports stay readable.
        "working_tree_dirty": tracked_modified,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "opencv": cv2.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    if config is not None:
        record["config_fingerprint"] = config_fingerprint(config)
        record["config"] = _describe(config)
    record.update(extra)
    return record


def _describe(config: Any) -> Any:
    if hasattr(config, "__dataclass_fields__"):
        return {
            name: _describe(getattr(config, name))
            for name in sorted(config.__dataclass_fields__)
        }
    if isinstance(config, Mapping):
        return {str(k): _describe(v) for k, v in sorted(config.items())}
    if isinstance(config, (list, tuple)):
        return [_describe(v) for v in config]
    return config
