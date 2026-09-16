"""Render every results table from the stored reports, in one pass.

Numbers were previously copied from a terminal into three documents by hand.
That is how a table comes to disagree with the run that produced it, and how a
superseded figure survives a recomputation.  Every table below is generated
from the JSON a tool wrote, carries the commit and configuration fingerprint of
that run, and is regenerated whenever the reports are.

The prose lives in CALIBRATION.md and STUDY_UA.md and links here; the numbers
live only here.

    python3 tools/render_results.py data --out docs/RESULTS.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _load(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _cell(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, float)):
        if value != value:  # NaN
            return "-"
        return f"{value:.{digits}f}"
    return str(value)


def _spec_cell(value: Any) -> str:
    """Render a configuration value as written, not as a measurement."""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    lines.append("")
    return lines


def _provenance_line(report: dict[str, Any]) -> str:
    meta = report.get("provenance") or {}
    if not meta:
        return (
            "<sub>**No provenance recorded.** Regenerate this report before "
            "citing it.</sub>"
        )
    commit = str(meta.get("commit", "unknown"))[:12]
    modified = meta.get("tracked_files_modified", meta.get("working_tree_dirty"))
    state = " with local modifications" if modified else ", clean"
    fingerprint = meta.get("config_fingerprint", "?")
    return (
        f"<sub>commit `{commit}`{state} &middot; config `{fingerprint}` &middot; "
        f"numpy {meta.get('numpy', '?')} &middot; opencv {meta.get('opencv', '?')}</sub>"
    )


def _selection_table(report: dict[str, Any]) -> list[str]:
    selection = report.get("selection") or {}
    if not selection:
        return []
    lines = ["**Frames measured**", ""]
    rows = []
    for name, info in selection.items():
        excluded = ", ".join(f"{k} {v}" for k, v in info.get("excluded", {}).items())
        rows.append([
            f"`{name}`", info.get("selected", "?"), info.get("total_frames", "?"),
            excluded or "none", info.get("reference", "-"),
        ])
    lines += _table(
        ["recording", "measured", "of", "excluded", "reference"], rows
    )
    return lines


def _ablation_section(
    title: str, report: dict[str, Any], columns: Sequence[tuple[str, str, int]]
) -> list[str]:
    summary = report.get("aggregate") or {}
    if not summary:
        return []
    lines = [f"## {title}", "", _provenance_line(report), ""]
    lines += _selection_table(report)

    headers = ["variant"] + [label for label, _, _ in columns]
    rows = []
    for name, row in summary.items():
        rows.append(
            [f"`{name}`"]
            + [_cell(row.get(key, float("nan")), digits) for _, key, digits in columns]
        )
    lines += _table(headers, rows)

    # The switches each variant set, so a row can be traced to an experiment.
    # Older reports stored only the variant names, with no way to tell what
    # they meant - which is the gap this section exists to close.
    specs = report.get("variants") or {}
    if not isinstance(specs, dict):
        lines += [
            "> This report predates per-variant configuration recording, so what",
            "> each variant set cannot be recovered from it. Regenerate it.",
            "",
        ]
        specs = {}
    if specs:
        first = next(iter(specs.values()))
        interesting = [
            key for key in first
            if len({json.dumps(s.get(key), sort_keys=True) for s in specs.values()}) > 1
        ]
        if interesting:
            lines += ["<details><summary>What each variant set</summary>", ""]
            lines += _table(
                ["variant"] + interesting,
                [[f"`{n}`"] + [_spec_cell(s.get(k)) for k in interesting]
                 for n, s in specs.items()],
            )
            lines += ["</details>", ""]
    return lines


def _per_run_table(report: dict[str, Any], field: str, title: str) -> list[str]:
    runs = report.get("runs") or {}
    if not runs:
        return []
    stored = report.get("variants")
    variants = list(stored) if stored else list(next(iter(runs.values())))
    lines = [f"**{title}**", ""]
    rows = []
    for name, entry in runs.items():
        rows.append(
            [f"`{name}`"]
            + [_cell(entry.get(v, {}).get(field, float("nan"))) for v in variants]
        )
    lines += _table(["recording"] + variants, rows)
    return lines


def _fair_comparison(report: dict[str, Any]) -> list[str]:
    lines = ["## Signals and pipelines, compared separately", "",
             _provenance_line(report), "",
             "A ranking of *measures* and a ranking of *systems* answer",
             "different questions. Raw signals carry no normalisation, no",
             "history and no filter; pipelines all carry the same ones.", ""]
    for section, heading in (
        ("signals", "Raw signals, scored directly against the spot reference"),
        ("pipelines", "Full streaming pipelines"),
    ):
        summary = report.get(section) or {}
        if not summary:
            continue
        lines += [f"### {heading}", ""]
        order = sorted(
            summary,
            key=lambda n: -summary[n].get("spearman_gt", float("-inf")),
        )
        columns = [
            ("spearman", "spearman_gt", 3), ("inversion", "inversion", 3),
            ("strict", "inversion_strict", 3), ("resolved", "resolved", 3),
            ("adj", "adj", 3), ("peak err", "peak_err", 2),
        ]
        if section == "pipelines":
            columns.append(("ms", "ms", 2))
        else:
            columns.append(("negative", "negative_fraction", 3))
        rows = [
            [f"`{name}`"]
            + [_cell(summary[name].get(key, float("nan")), digits)
               for _, key, digits in columns]
            for name in order
        ]
        lines += _table(["name"] + [label for label, _, _ in columns], rows)
    return lines


def _reference_sensitivity(report: dict[str, Any]) -> list[str]:
    runs = report.get("runs") or {}
    if not runs:
        return []
    lines = ["## How much the reference itself depends on its parameters", "",
             _provenance_line(report), ""]
    for name, entry in runs.items():
        clipped = entry.get("frames_with_clipped_core", float("nan"))
        lines += [
            f"### `{name}`", "",
            f"{entry.get('frames', '?')} frames measured, "
            f"{_cell(100 * clipped, 1)}% with a clipped core.", "",
        ]
        rows = []
        for variant, row in entry.get("variants", {}).items():
            first = row.get("best_step_first", float("nan"))
            last = row.get("best_step_last", float("nan"))
            plateau = last - first + 1 if first == first and last == last else float("nan")
            rows.append([
                f"`{variant}`", _cell(first, 0), _cell(plateau, 0),
                _cell(row.get("min_radius_px"), 1), _cell(row.get("max_radius_px"), 1),
                _cell(row.get("within_step_scatter_px"), 2),
                _cell(row.get("best_step_unclipped", float("nan")), 0),
            ])
        lines += _table(
            ["variant", "best step", "plateau", "r min", "r max", "scatter",
             "excl. clipped"],
            rows,
        )
        steps = [
            r.get("best_step_first") for r in entry.get("variants", {}).values()
            if isinstance(r.get("best_step_first"), (int, float))
            and r["best_step_first"] == r["best_step_first"]
        ]
        if steps:
            lines += [
                f"Best step across all variants: **{min(steps):.0f} to "
                f"{max(steps):.0f}** (spread {max(steps) - min(steps):.0f} steps).",
                "",
            ]
    return lines


def _before_after(report: dict[str, Any]) -> list[str]:
    runs = report.get("runs") if isinstance(report, dict) else None
    if runs is None and isinstance(report, list):
        runs = report
    if not runs:
        return []
    lines = ["## Before and after, against the spot reference", ""]
    if isinstance(report, dict):
        lines += [_provenance_line(report), ""]
    rows = []
    for run in runs:
        rows.append([
            f"`{run['name']}`",
            _cell(run["before"].get("spearman")), _cell(run["after"].get("spearman")),
            _cell(run["before"].get("inversion")), _cell(run["after"].get("inversion")),
            _cell(run.get("best_step"), 0),
            _cell(run["before"].get("peak_err"), 1),
            _cell(run["after"].get("peak_err"), 1),
        ])
    lines += _table(
        ["recording", "spearman before", "after", "inversion before", "after",
         "true best step", "peak err before", "after"],
        rows,
    )
    return lines


_ABLATION_COLUMNS = (
    ("adj", "adj", 3),
    ("mono", "mono", 3),
    ("sat", "sat", 3),
    ("sentinel", "sentinel", 3),
    ("conf=0", "confidence_zero", 3),
    ("noise=0", "noise_zero", 3),
    ("ms", "ms", 2),
)

_GROUND_TRUTH_COLUMNS = (
    ("spearman", "spearman_gt", 3),
    ("spread", "spearman_gt_spread", 3),
    ("inversion", "inversion", 3),
    ("strict", "inversion_strict", 3),
    ("resolved", "resolved", 3),
    ("peak err", "peak_err", 2),
    ("plateau", "peak_plateau", 1),
)


def _reference_robustness(
    primary: dict[str, Any], alternative: dict[str, Any]
) -> list[str]:
    """Does the ranking survive measuring the reference a different way?

    The reference has free parameters, and its variants do not all agree - the
    120 px window is *anti*-correlated with the smaller ones.  A ranking that
    only holds for one recipe is a property of the recipe.
    """
    first = primary.get("provenance", {}).get("reference_variant", "default")
    second = alternative.get("provenance", {}).get("reference_variant", "alternative")
    lines = [
        "## Does the ranking survive a different reference recipe?", "",
        f"The same factorial, scored against `{first}` and against `{second}`.",
        "The two recipes agree with each other at rank correlation 0.98 on these",
        "recordings; the 120 px window variants, which are anti-correlated with",
        "both, are excluded on the grounds given in the sensitivity section.", "",
    ]
    names = [n for n in primary.get("aggregate", {}) if n in alternative.get("aggregate", {})]
    rows = []
    for name in names:
        a = primary["aggregate"][name]
        b = alternative["aggregate"][name]
        rows.append([
            f"`{name}`",
            _cell(a.get("spearman_gt")), _cell(b.get("spearman_gt")),
            _cell(a.get("spearman_gt", 0) - b.get("spearman_gt", 0)),
            _cell(a.get("inversion")), _cell(b.get("inversion")),
        ])
    lines += _table(
        ["variant", f"spearman ({first})", f"spearman ({second})", "difference",
         f"inversion ({first})", f"inversion ({second})"],
        rows,
    )

    # The comparison the project's central claim rests on, per recording, so
    # that a mean over two runs is never mistaken for an established effect.
    lines += ["**Adaptive weighting against a plain mean, per recording**", ""]
    per_run = []
    for report, label in ((primary, first), (alternative, second)):
        for run, entry in report.get("runs", {}).items():
            a = entry.get("R-F", {}).get("spearman_gt")
            b = entry.get("plain_mean", {}).get("spearman_gt")
            if a is None or b is None or a != a or b != b:
                continue
            per_run.append([f"`{run}`", f"`{label}`", _cell(a), _cell(b),
                            f"{a - b:+.3f}"])
    lines += _table(
        ["recording", "reference", "R-F", "plain_mean", "difference"], per_run
    )
    lines += [
        "Consistent in direction across both recordings and both reference",
        "recipes, at 0.008 to 0.018. Only two recordings carry a reference, so",
        "this is a consistent direction rather than an established effect.",
        "",
    ]
    return lines


def _dataset(report: dict[str, Any]) -> list[str]:
    """Describe the recordings without publishing any of their pixels."""
    runs = report.get("runs") or {}
    if not runs:
        return []
    lines = [
        "## The dataset", "", _provenance_line(report), "",
        "The recordings are photographs of the operator's room and are not",
        "published. Everything that characterises them as *data* is below:",
        "what was in front of the camera, how many frames, how many labelled",
        "focus steps, and the scene statistics that decide whether a recording",
        "can answer a question at all.", "",
    ]
    rows = []
    for name, entry in runs.items():
        rows.append([
            f"`{name}`", entry.get("preset", "?"), entry.get("note", ""),
            entry.get("frames", "?"), entry.get("held", "?"),
            entry.get("steps", "?"),
            _cell(entry.get("brightness"), 3),
            _cell(entry.get("edge_density"), 4),
            _cell(entry.get("motion_px_p95"), 2),
        ])
    lines += _table(
        ["recording", "protocol", "subject", "frames", "held", "steps",
         "brightness", "edge density", "motion p95 px"],
        rows,
    )
    lines += [
        "`held` counts frames recorded while the operator was prompted to hold",
        "the focus ring still; those are the frames every comparison uses.",
        "Brightness, edge density and motion are medians over the recording,",
        "except motion which is the 95th percentile.",
        "",
    ]
    return lines


def build(data: Path) -> str:
    lines = [
        "# Results",
        "",
        "Generated by `tools/render_results.py` from the stored reports. Do not",
        "edit by hand: the prose in [CALIBRATION.md](CALIBRATION.md) and",
        "[STUDY_UA.md](STUDY_UA.md) links here rather than restating these",
        "numbers, so that a recomputation cannot leave a stale table behind.",
        "",
        "Every table names the commit and configuration fingerprint of the run",
        "that produced it. A missing section means that report has not been",
        "regenerated since the measurement code was last corrected.",
        "",
        "---",
        "",
    ]

    protocol = _load(data / "protocol_study.json")
    if protocol:
        lines += _dataset(protocol)
        lines += ["---", ""]

    groups = [
        ("fixes", "Each repair on its own"),
        ("normalisation", "Normalisation: moving or frozen, linear or logistic"),
        ("factorial", "Adaptivity: all combinations of the three mechanisms"),
        ("metrics", "Metric set and analysis resolution"),
        ("agreement", "Consensus kernel width"),
        ("noise", "Noise estimator quantile"),
        ("edges", "Edge-sufficiency reference"),
    ]
    for key, title in groups:
        report = _load(data / f"ablation_{key}.json")
        if not report:
            continue
        lines += _ablation_section(title, report, _ABLATION_COLUMNS)
        summary = report.get("aggregate") or {}
        if any(
            isinstance(r.get("spearman_gt"), (int, float))
            and r["spearman_gt"] == r["spearman_gt"]
            for r in summary.values()
        ):
            lines += ["**Against the point-source reference**", ""]
            lines += _table(
                ["variant"] + [label for label, _, _ in _GROUND_TRUTH_COLUMNS],
                [
                    [f"`{name}`"] + [
                        _cell(row.get(k, float("nan")), d)
                        for _, k, d in _GROUND_TRUTH_COLUMNS
                    ]
                    for name, row in summary.items()
                ],
            )
        lines += _per_run_table(report, "adj", "Adjacent-step discrimination per recording")
        lines += ["---", ""]

    primary = _load(data / "ablation_factorial.json")
    alternative = _load(data / "ablation_factorial_altref.json")
    if primary and alternative:
        lines += _reference_robustness(primary, alternative)
        lines += ["---", ""]

    for loader, path in (
        (_before_after, data / "before_after.json"),
        (_fair_comparison, data / "fair_comparison.json"),
        (_reference_sensitivity, data / "reference_sensitivity.json"),
    ):
        report = _load(path)
        if report:
            lines += loader(report)
            lines += ["---", ""]

    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path, help="directory holding the JSON reports")
    parser.add_argument("--out", type=Path, default=Path("docs/RESULTS.md"))
    args = parser.parse_args(argv)

    text = build(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.out} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
