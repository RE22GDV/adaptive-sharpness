"""Every number quoted in prose, checked against the report that produced it.

A generated table cannot drift from its data, because it is written from it.
A sentence can, and did: `docs/CALIBRATION.md` summarised the singles ablation
in a table headed "across all eight recordings" while taking its plateau column
from `peak_plateau`, which is defined only on the two recordings that carry the
point-source reference.  The number was off by a factor of three, the sentence
built on it ("a single measure does not localise the maximum at all") was
wrong, and `docs/RESULTS.md` - generated, correct - sat two files away
disagreeing with it silently.

So each claim registers not just a value but the sample it was averaged over,
and both are checked.  A claim that reads the right field with the wrong `n`
fails as loudly as one whose value has moved.
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_CLAIMS = _ROOT / "docs" / "claims.toml"
_REPORTS = _ROOT / "reports"


def _load_claims() -> list[dict[str, Any]]:
    if not _CLAIMS.exists():
        return []
    return tomllib.loads(_CLAIMS.read_text(encoding="utf-8"))["claim"]


CLAIMS = _load_claims()


def _report(name: str) -> dict[str, Any] | None:
    path = _REPORTS / f"{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def test_registry_is_not_empty() -> None:
    assert CLAIMS, f"{_CLAIMS} is missing or has no claims"


def test_claim_ids_are_unique() -> None:
    ids = [c["id"] for c in CLAIMS]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate claim ids: {sorted(duplicates)}"


@pytest.mark.parametrize("claim", CLAIMS, ids=[c["id"] for c in CLAIMS])
def test_claim_matches_its_report(claim: dict[str, Any]) -> None:
    report = _report(claim["report"])
    if report is None:
        pytest.skip(f"reports/{claim['report']}.json not present")

    aggregate = report.get("aggregate") or {}
    assert claim["variant"] in aggregate, (
        f"{claim['id']}: report has no variant {claim['variant']!r}; "
        f"it has {sorted(aggregate)}"
    )
    row = aggregate[claim["variant"]]

    field = claim["field"]
    assert field in row, f"{claim['id']}: variant has no field {field!r}"

    where = ", ".join(claim.get("where", [])) or "(unrecorded)"

    # The sample size first: a value that is right for the wrong sample is the
    # failure this file was written for, and reporting it as a value mismatch
    # would send the reader looking in the wrong place.
    stated_n = claim.get("n")
    if stated_n is not None:
        actual_n = row.get(f"{field}_n")
        assert actual_n is not None, (
            f"{claim['id']}: report does not record a sample size for {field!r}, "
            f"so the claim of n={stated_n} in {where} cannot be checked"
        )
        assert int(actual_n) == int(stated_n), (
            f"{claim['id']}: {where} presents {field!r} as covering "
            f"{stated_n} recordings, but the report averaged it over "
            f"{int(actual_n)}. Either the prose or the field is wrong - "
            f"'{field}' and '{field}_steps'-style variants are averaged over "
            f"different samples on purpose."
        )

    digits = int(claim["digits"])
    actual = round(float(row[field]), digits)
    expected = round(float(claim["value"]), digits)
    assert actual == expected, (
        f"{claim['id']}: {where} states {field}={expected} but "
        f"reports/{claim['report']}.json now holds {actual}. "
        f"Regenerate the documents (tools/render_results.py) and update the "
        f"prose, or correct docs/claims.toml if the claim was wrong."
    )


def test_every_claim_names_a_document() -> None:
    """A registered number nobody quotes is dead weight; one nobody can trace
    is worse, because a failure cannot say which file to fix."""
    orphans = [c["id"] for c in CLAIMS if not c.get("where")]
    assert not orphans, f"claims with no 'where': {orphans}"


def test_claimed_documents_exist() -> None:
    missing = {
        c["id"]: [w for w in c.get("where", []) if not (_ROOT / w).exists()]
        for c in CLAIMS
    }
    missing = {k: v for k, v in missing.items() if v}
    assert not missing, f"claims pointing at files that do not exist: {missing}"


# ---------------------------------------------------------------------------
# Diagrams
# ---------------------------------------------------------------------------

_SCHEMES = _ROOT / "docs" / "figures" / "ua" / "schemes"


def _extract():
    from tools.figures_ua import SCHEME_SOURCE, _SCHEME_HEADING  # noqa: F401
    import tools.figures_ua as mod

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        return mod.extract_schemes(_ROOT / mod.SCHEME_SOURCE, Path(tmp)), tmp


def test_every_diagram_in_the_report_has_a_file() -> None:
    """The `.mmd` files are generated from the report, never edited beside it.

    A draft of this work shipped hand-kept copies of the same five diagrams
    next to the document they came from. Two copies of one diagram is two
    diagrams as soon as either is touched, which is the same failure as a
    number quoted in prose drifting from the report it came from.
    """
    import tools.figures_ua as mod

    source = _ROOT / mod.SCHEME_SOURCE
    if not source.exists():
        pytest.skip(f"{mod.SCHEME_SOURCE} not present")
    in_document = mod._SCHEME_HEADING.findall(source.read_text(encoding="utf-8"))
    assert in_document, "the report declares no diagrams"
    if not _SCHEMES.exists():
        pytest.skip("diagrams not generated yet; run tools/figures_ua.py")
    on_disk = sorted(p.name for p in _SCHEMES.glob("*.mmd"))
    assert len(on_disk) == len(in_document), (
        f"the report has {len(in_document)} diagrams and {len(on_disk)} files "
        f"exist; run 'python3 tools/figures_ua.py'"
    )


def test_the_files_still_match_the_report() -> None:
    """Regenerating must be a no-op, or one of the two has been edited alone."""
    import tempfile

    import tools.figures_ua as mod

    source = _ROOT / mod.SCHEME_SOURCE
    if not (source.exists() and _SCHEMES.exists()):
        pytest.skip("nothing to compare")
    with tempfile.TemporaryDirectory() as tmp:
        mod.extract_schemes(source, Path(tmp))
        for fresh in sorted(Path(tmp).glob("*.mmd")):
            committed = _SCHEMES / fresh.name
            assert committed.exists(), f"{fresh.name} was never written out"
            assert committed.read_text(encoding="utf-8") == fresh.read_text(
                encoding="utf-8"
            ), (
                f"{fresh.name} differs from the report. Edit the diagram in "
                f"{mod.SCHEME_SOURCE} and re-run tools/figures_ua.py; the "
                f"'.mmd' file is generated and edits to it are lost."
            )


def test_diagrams_are_numbered_in_reading_order() -> None:
    """A reader who meets 'Схема 5' before 'Схема 4' has found a defect; this
    document did, inherited from a draft where the two appeared reversed."""
    import tools.figures_ua as mod

    source = _ROOT / mod.SCHEME_SOURCE
    if not source.exists():
        pytest.skip(f"{mod.SCHEME_SOURCE} not present")
    numbers = [
        int(n) for n, _ in mod._SCHEME_HEADING.findall(
            source.read_text(encoding="utf-8")
        )
    ]
    assert numbers == sorted(numbers), (
        f"diagrams appear in the order {numbers}; renumber them so they read "
        f"in sequence"
    )


def test_svg_output_is_reproducible() -> None:
    """Regenerating an unchanged figure must produce an unchanged file.

    Matplotlib stamps SVGs with the wall clock and derives element ids from a
    per-process random salt, so two runs of the same code over the same data
    differed in 1356 lines across 19 files. A diff that always changes cannot
    show that something changed.
    """
    import tools.figures_ua as mod

    assert plt_rcparam("svg.hashsalt"), (
        "svg.hashsalt is unset, so element ids are random per process"
    )
    assert mod._SVG_METADATA.get("Date", "unset") is None, (
        "the SVG date stamp is not being suppressed"
    )


def plt_rcparam(key: str):
    import matplotlib.pyplot as plt

    import tools.figures_ua  # noqa: F401  (applies the rcParams on import)

    return plt.rcParams.get(key)
