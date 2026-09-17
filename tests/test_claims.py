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
