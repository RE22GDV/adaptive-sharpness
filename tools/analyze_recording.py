"""Analyse a recording made by tools/collect_dataset.py and report problems.

Reads ``frames.csv`` (and ``manifest.json`` when present) and runs a fixed set of
checks, each of which looks for a specific way the pipeline can be wrong in a way
that is invisible in a live view:

* the frame rate is unstable, or frames are being dropped;
* the normalisers never became informative, so the scores mean nothing;
* metrics are saturated at 0 or 1, so the score cannot resolve near-focus frames;
* a metric is constant, i.e. contributing nothing;
* the metrics disagree persistently, which collapses the confidence;
* weights swing frame to frame, which adds variance to the score;
* score-change detections fire on a static scene;
* processing time has outliers that would break a real-time loop.

Each check reports OK / WARN / FAIL with the measured number, so the output can
go straight into a report.

    python3 tools/analyze_recording.py data/run1
    python3 tools/analyze_recording.py data/run1 --figures docs/figures
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for _entry in (_ROOT / "src", _ROOT):
    if str(_entry) not in sys.path:
        sys.path.insert(0, str(_entry))

OK, WARN, FAIL = "OK", "WARN", "FAIL"


@dataclass
class Finding:
    level: str
    check: str
    detail: str

    def __str__(self) -> str:
        mark = {OK: "  ok  ", WARN: " WARN ", FAIL: " FAIL "}[self.level]
        return f"[{mark}] {self.check:34s} {self.detail}"


class Recording:
    """A recording loaded into columns."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        csv_path = directory / "frames.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(f"no frames.csv in {directory}")
        with csv_path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            raise ValueError(f"{csv_path} has no rows")
        self.rows = rows
        self.fields = list(rows[0])

        manifest_path = directory / "manifest.json"
        self.manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.rows)

    def column(self, name: str) -> np.ndarray:
        """Numeric column; non-numeric and empty cells become NaN."""
        values = []
        for row in self.rows:
            raw = row.get(name, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(math.nan)
        return np.array(values, dtype=np.float64)

    def text_column(self, name: str) -> list[str]:
        return [row.get(name, "") for row in self.rows]

    @property
    def metric_names(self) -> list[str]:
        return [f[4:] for f in self.fields if f.startswith("raw_")]

    def has(self, name: str) -> bool:
        return name in self.fields


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

def check_frame_rate(rec: Recording) -> list[Finding]:
    out: list[Finding] = []
    if not rec.has("interval_s"):
        return [Finding(WARN, "frame rate", "no interval_s column")]
    intervals = rec.column("interval_s")[1:]
    intervals = intervals[np.isfinite(intervals)]
    if intervals.size < 2:
        return [Finding(WARN, "frame rate", "too few frames")]
    mean_ms = intervals.mean() * 1e3
    p95_ms = float(np.percentile(intervals, 95)) * 1e3
    max_ms = intervals.max() * 1e3
    fps = 1.0 / intervals.mean()
    jitter = p95_ms / mean_ms if mean_ms else 0.0
    level = OK if jitter < 1.25 else (WARN if jitter < 2.0 else FAIL)
    out.append(Finding(
        level, "frame rate",
        f"{fps:.2f} fps, interval {mean_ms:.1f} ms mean / {p95_ms:.1f} p95 / "
        f"{max_ms:.1f} max (p95/mean = {jitter:.2f})",
    ))

    stalls = int(np.count_nonzero(intervals > 3 * intervals.mean()))
    if stalls:
        out.append(Finding(
            WARN, "frame-rate stalls",
            f"{stalls} interval(s) over 3x the mean - longest {max_ms:.0f} ms",
        ))
    return out


def check_readiness(rec: Recording) -> list[Finding]:
    if not rec.has("ready"):
        return [Finding(WARN, "readiness", "no ready column (old recording)")]
    ready = rec.column("ready")
    informative = rec.column("informative_fraction")
    ready_fraction = float(np.nanmean(ready))
    first_ready = int(np.argmax(ready > 0)) if np.any(ready > 0) else -1

    out = []
    if first_ready < 0:
        out.append(Finding(
            FAIL, "readiness",
            "the evaluator never became ready: every score in this recording is "
            "a placeholder. The scene did not vary enough to give the "
            "normalisers a scale.",
        ))
    else:
        level = OK if ready_fraction > 0.9 else WARN
        out.append(Finding(
            level, "readiness",
            f"ready from frame {first_ready}, {ready_fraction * 100:.0f}% of frames",
        ))
    mean_inf = float(np.nanmean(informative))
    out.append(Finding(
        OK if mean_inf > 0.99 else WARN, "normaliser informativeness",
        f"mean informative_fraction {mean_inf:.2f}",
    ))
    return out


def check_confidence(rec: Recording) -> list[Finding]:
    confidence = rec.column("confidence")
    confidence = confidence[np.isfinite(confidence)]
    if confidence.size == 0:
        return [Finding(WARN, "confidence", "no data")]
    low = float(np.mean(confidence < 0.35))
    median = float(np.median(confidence))
    level = OK if low < 0.2 else (WARN if low < 0.5 else FAIL)
    return [Finding(
        level, "confidence",
        f"median {median:.2f}, below 0.35 on {low * 100:.0f}% of frames",
    )]


def check_saturation(rec: Recording) -> list[Finding]:
    """Normalised values pinned at 0 or 1 cannot resolve anything."""
    out: list[Finding] = []
    worst = 0.0
    worst_name = ""
    for name in rec.metric_names:
        values = rec.column(f"norm_{name}")
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        pinned = float(np.mean((values >= 0.999) | (values <= 0.001)))
        if pinned > worst:
            worst, worst_name = pinned, name
    level = OK if worst < 0.15 else (WARN if worst < 0.4 else FAIL)
    out.append(Finding(
        level, "normalisation saturation",
        f"worst metric '{worst_name}' pinned at 0 or 1 on {worst * 100:.0f}% of "
        f"frames (clipping plateau)",
    ))
    return out


def check_constant_metrics(rec: Recording) -> list[Finding]:
    out: list[Finding] = []
    for name in rec.metric_names:
        raw = rec.column(f"raw_{name}")
        raw = raw[np.isfinite(raw)]
        if raw.size < 5:
            continue
        spread = float(raw.std() / (abs(raw.mean()) + 1e-12))
        if spread < 1e-4:
            out.append(Finding(
                FAIL, f"metric '{name}' constant",
                f"relative spread {spread:.2e} - contributes nothing",
            ))
    if not out:
        out.append(Finding(OK, "metric variability", "every metric varies"))
    return out


def check_agreement(rec: Recording) -> list[Finding]:
    names = rec.metric_names
    if len(names) < 2:
        return []
    matrix = np.column_stack([rec.column(f"norm_{n}") for n in names])
    valid = np.all(np.isfinite(matrix), axis=1)
    if not np.any(valid):
        return [Finding(WARN, "metric agreement", "no complete rows")]
    matrix = matrix[valid]
    median = np.median(matrix, axis=1, keepdims=True)
    dispersion = 1.4826 * np.median(np.abs(matrix - median), axis=1)
    mean_d = float(dispersion.mean())
    p95_d = float(np.percentile(dispersion, 95))
    level = OK if p95_d < 0.35 else (WARN if p95_d < 0.5 else FAIL)
    return [Finding(
        level, "metric agreement",
        f"robust spread {mean_d:.3f} mean / {p95_d:.3f} p95 "
        f"(confidence_dispersion_ref is 0.45)",
    )]


def check_weight_stability(rec: Recording) -> list[Finding]:
    names = rec.metric_names
    columns = [rec.column(f"w_{n}") for n in names if rec.has(f"w_{n}")]
    if not columns:
        return []
    matrix = np.column_stack(columns)
    valid = np.all(np.isfinite(matrix), axis=1)
    matrix = matrix[valid]
    if matrix.shape[0] < 3:
        return []
    # Total absolute weight moved between consecutive frames.
    churn = np.abs(np.diff(matrix, axis=0)).sum(axis=1)
    mean_churn = float(churn.mean())
    level = OK if mean_churn < 0.05 else (WARN if mean_churn < 0.15 else FAIL)
    sums = matrix.sum(axis=1)
    findings = [Finding(
        level, "weight stability",
        f"mean total weight moved per frame {mean_churn:.3f}",
    )]
    if not np.allclose(sums, 1.0, atol=1e-6):
        findings.append(Finding(
            FAIL, "weights sum to one",
            f"min {sums.min():.6f}, max {sums.max():.6f}",
        ))
    return findings


def check_change_detections(rec: Recording) -> list[Finding]:
    if not rec.has("score_change_detected"):
        return []
    flags = rec.column("score_change_detected")
    flags = flags[np.isfinite(flags)]
    if flags.size == 0:
        return []
    rate = float(flags.mean())
    level = OK if rate < 0.15 else (WARN if rate < 0.35 else FAIL)
    return [Finding(
        level, "score-change detections",
        f"{int(flags.sum())} of {flags.size} frames ({rate * 100:.0f}%)",
    )]


def check_timing(rec: Recording) -> list[Finding]:
    column = "total_time_s" if rec.has("total_time_s") else "processing_time_s"
    values = rec.column(column)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return []
    mean_ms = values.mean() * 1e3
    p95_ms = float(np.percentile(values, 95)) * 1e3
    max_ms = values.max() * 1e3
    ratio = max_ms / mean_ms if mean_ms else 0.0
    level = OK if ratio < 3.0 else WARN
    return [Finding(
        level, f"processing time ({column})",
        f"{mean_ms:.1f} ms mean / {p95_ms:.1f} p95 / {max_ms:.1f} max",
    )]


def check_exposure_and_content(rec: Recording) -> list[Finding]:
    out: list[Finding] = []
    if rec.has("clipped_high"):
        clipped = rec.column("clipped_high")
        clipped = clipped[np.isfinite(clipped)]
        if clipped.size and clipped.mean() > 0.05:
            out.append(Finding(
                WARN, "clipped highlights",
                f"mean {clipped.mean() * 100:.1f}% of pixels - detail is lost, "
                "so the score genuinely falls",
            ))
    if rec.has("edge_sufficiency"):
        edges = rec.column("edge_sufficiency")
        edges = edges[np.isfinite(edges)]
        if edges.size:
            level = OK if edges.mean() > 0.6 else WARN
            out.append(Finding(
                level, "edge sufficiency",
                f"mean {edges.mean():.2f} - scene texture for focus measurement",
            ))
    if rec.has("noise_sigma"):
        noise = rec.column("noise_sigma")
        noise = noise[np.isfinite(noise)]
        if noise.size and noise.mean() < 0.05:
            out.append(Finding(
                WARN, "noise adaptation inactive",
                f"measured sigma {noise.mean():.3f} - the noise term of the "
                "weighting never engages on this source",
            ))
    return out


def check_scene(rec: Recording) -> list[Finding]:
    if not rec.has("subject_label"):
        return []
    labels = [v for v in rec.text_column("subject_label") if v]
    out: list[Finding] = []
    if labels:
        top = max(set(labels), key=labels.count)
        share = labels.count(top) / len(labels)
        level = OK if share > 0.8 else WARN
        out.append(Finding(
            level, "subject stability",
            f"'{top}' on {share * 100:.0f}% of frames "
            f"({len(set(labels))} distinct label(s))",
        ))
    if rec.has("decision_margin"):
        margin = rec.column("decision_margin")
        margin = margin[np.isfinite(margin)]
        if margin.size:
            level = OK if margin.mean() > 0.05 else WARN
            out.append(Finding(
                level, "decision margin",
                f"mean {margin.mean():.3f} - lead of the winner over the runner-up",
            ))
    if rec.has("map_valid_fraction"):
        valid = rec.column("map_valid_fraction")
        valid = valid[np.isfinite(valid)]
        if valid.size:
            level = OK if valid.mean() > 0.5 else WARN
            out.append(Finding(
                level, "measurable area",
                f"mean {valid.mean() * 100:.0f}% of tiles carried enough texture",
            ))
    if rec.has("subject_x"):
        x = rec.column("subject_x")
        x = x[np.isfinite(x)]
        if x.size > 2:
            out.append(Finding(
                OK, "subject motion",
                f"centre x sd {x.std():.1f} px over the run",
            ))
    return out


CHECKS: list[Callable[[Recording], list[Finding]]] = [
    check_frame_rate,
    check_timing,
    check_readiness,
    check_confidence,
    check_saturation,
    check_constant_metrics,
    check_agreement,
    check_weight_stability,
    check_change_detections,
    check_exposure_and_content,
    check_scene,
]


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def summarise(rec: Recording) -> list[str]:
    lines = [f"frames        : {len(rec)}"]
    if rec.has("wall_time"):
        wall = rec.column("wall_time")
        lines.append(f"duration      : {np.nanmax(wall):.1f} s")
    manifest = rec.manifest
    if manifest:
        env = manifest.get("environment", {})
        lines.append(f"source        : {manifest.get('source', '?')}")
        lines.append(f"device        : {env.get('device', env.get('platform', '?'))}")
        lines.append(f"opencv/numpy  : {env.get('opencv', '?')} / {env.get('numpy', '?')}")
        summary = manifest.get("summary", {})
        if summary:
            lines.append(
                f"recorder said : {summary.get('mean_fps', '?')} fps, "
                f"{summary.get('images_saved', 0)} images saved, "
                f"{summary.get('capture_dropped_stale', 0)} stale frames dropped"
            )
    images = sorted((rec.directory / "frames").glob("*.png"))
    lines.append(f"saved frames  : {len(images)}")
    return lines


def make_figures(rec: Recording, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    wall = rec.column("wall_time") if rec.has("wall_time") else np.arange(len(rec))

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax = axes[0]
    if rec.has("instantaneous_score"):
        ax.plot(wall, rec.column("instantaneous_score"), lw=1.0,
                color="#9aa0a6", label="instantaneous")
    if rec.has("filtered_score"):
        ax.plot(wall, rec.column("filtered_score"), lw=1.8,
                color="#111111", label="filtered")
    if rec.has("subject_score"):
        ax.plot(wall, rec.column("subject_score"), lw=1.4,
                color="#00798c", label="subject")
    ax.set_ylabel("score")
    ax.set_title("Recorded run")
    ax.legend(fontsize=8)
    ax.grid(color="#e6e8eb")

    ax = axes[1]
    ax.plot(wall, rec.column("confidence"), lw=1.4, color="#d1495b", label="confidence")
    if rec.has("informative_fraction"):
        ax.plot(wall, rec.column("informative_fraction"), lw=1.0,
                color="#66a182", label="informative fraction")
    ax.set_ylabel("confidence")
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=8)
    ax.grid(color="#e6e8eb")

    ax = axes[2]
    for name in rec.metric_names:
        ax.plot(wall, rec.column(f"w_{name}"), lw=1.1, label=name)
    ax.set_ylabel("weight")
    ax.set_xlabel("time, s")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(color="#e6e8eb")

    fig.tight_layout()
    path = out_dir / "recording_timeline.png"
    fig.savefig(path, dpi=130, facecolor="white")
    plt.close(fig)
    print(f"\nwrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="recording directory")
    parser.add_argument("--figures", type=Path, default=None,
                        help="also write a timeline figure into this directory")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        rec = Recording(args.directory)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"RECORDING ANALYSIS - {args.directory}")
    print("=" * 78)
    for line in summarise(rec):
        print("  " + line)

    findings: list[Finding] = []
    for check in CHECKS:
        try:
            findings.extend(check(rec))
        except Exception as exc:  # noqa: BLE001 - one bad check must not hide the rest
            findings.append(Finding(WARN, check.__name__, f"check raised: {exc}"))

    print("\n" + "-" * 78)
    print("CHECKS")
    print("-" * 78)
    for finding in findings:
        print(finding)

    problems = [f for f in findings if f.level in (WARN, FAIL)]
    print("\n" + "-" * 78)
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for finding in problems:
            print(f"  {finding.level}: {finding.check} - {finding.detail}")
    else:
        print("no problems found")
    print("-" * 78)

    if args.figures:
        try:
            make_figures(rec, args.figures)
        except ImportError:
            print("matplotlib not installed; skipping figures", file=sys.stderr)

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps({
            "directory": str(args.directory),
            "frames": len(rec),
            "findings": [
                {"level": f.level, "check": f.check, "detail": f.detail}
                for f in findings
            ],
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 1 if any(f.level == FAIL for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())
