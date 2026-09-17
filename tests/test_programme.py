"""The research-programme machinery, pinned where it could silently change.

A study that quietly measures something other than what its name says is worse
than no study, because it still produces a number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]


class TestBuildConfig:
    def test_a_dotted_override_reaches_the_right_field(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config

        base = load_default_config()
        changed = build_config(base, {"pipeline.analysis_width": 480})
        assert changed.pipeline.analysis_width == 480
        assert base.pipeline.analysis_width == 320, "the base must not be mutated"

    def test_an_unknown_field_is_refused(self) -> None:
        """A typo silently ignored would produce a variant identical to the
        default while being reported under another name."""
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config

        with pytest.raises(ValueError, match="no field"):
            build_config(load_default_config(), {"pipeline.with": 1})

    def test_an_unknown_section_is_refused(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config

        with pytest.raises(ValueError, match="section"):
            build_config(load_default_config(), {"nosuch.field": 1})

    def test_a_bare_key_is_refused(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config

        with pytest.raises(ValueError, match="section.field"):
            build_config(load_default_config(), {"analysis_width": 320})


class TestWeightingSchemes:
    """These must mean the same thing they mean in `tools/ablation.py`.

    The published factorial calls the reliability model off when every
    sensitivity coefficient is zero. A second definition here would produce
    numbers that cannot be placed next to the published ones - and they were,
    so it was checked: this module reproduces the factorial's 0.8218 and
    0.9653 exactly.
    """

    def test_prior_weights_zero_every_sensitivity(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config
        from tools.studies import WEIGHTING

        config = build_config(load_default_config(), WEIGHTING["prior"])
        assert all(
            value == 0.0
            for row in config.ensemble.sensitivity.values()
            for value in row.values()
        )

    def test_adaptive_keeps_the_sensitivity(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config
        from tools.studies import WEIGHTING

        config = build_config(load_default_config(), WEIGHTING["adaptive"])
        assert any(
            value > 0.0
            for row in config.ensemble.sensitivity.values()
            for value in row.values()
        )

    def test_equal_weights_flatten_the_priors(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config
        from tools.studies import WEIGHTING

        config = build_config(load_default_config(), WEIGHTING["equal"])
        assert len(set(config.metrics.priors.values())) == 1

    def test_every_scheme_turns_the_kernel_off(self) -> None:
        """The kernel is off in the shipped configuration; a study that left it
        on would be measuring the intermediate pipeline instead."""
        from adaptive_sharpness import load_default_config
        from tools.programme import build_config
        from tools.studies import WEIGHTING

        for name, overrides in WEIGHTING.items():
            config = build_config(load_default_config(), overrides)
            assert config.ensemble.use_agreement is False, name


class TestDegradations:
    def test_blur_actually_blurs(self) -> None:
        import numpy as np

        from tools.studies import degrade

        rng = np.random.default_rng(0)
        image = rng.integers(0, 255, (64, 64, 3), dtype=np.uint8)
        blurred = degrade(image, "blur", 4.0, 0)
        assert np.std(blurred.astype(float)) < np.std(image.astype(float))

    def test_noise_is_reproducible_from_the_seed(self) -> None:
        import numpy as np

        from tools.studies import degrade

        image = np.full((32, 32, 3), 128, dtype=np.uint8)
        first = degrade(image, "noise", 10.0, 7)
        second = degrade(image, "noise", 10.0, 7)
        assert np.array_equal(first, second)
        assert not np.array_equal(first, degrade(image, "noise", 10.0, 8))

    def test_none_is_the_identity(self) -> None:
        import numpy as np

        from tools.studies import degrade

        image = np.full((8, 8, 3), 42, dtype=np.uint8)
        assert np.array_equal(degrade(image, "none", 0.0, 0), image)

    def test_an_unknown_degradation_is_refused(self) -> None:
        import numpy as np

        from tools.studies import degrade

        with pytest.raises(ValueError, match="unknown degradation"):
            degrade(np.zeros((8, 8, 3), dtype="uint8"), "sharpen", 1.0, 0)


class TestRegistry:
    def test_every_study_has_a_title(self) -> None:
        from tools.studies import STUDIES, TITLES

        assert set(STUDIES) == set(TITLES)

    def test_the_programme_is_complete(self) -> None:
        from tools.studies import STUDIES

        assert set(STUDIES) == {f"d{n:02d}" for n in range(7, 21)}


class TestPublishedStudyReports:
    """Each committed study must carry the provenance to be traced back."""

    @pytest.mark.parametrize("study", [f"d{n:02d}" for n in range(7, 21)])
    def test_report_has_provenance_and_a_title(self, study: str) -> None:
        path = _ROOT / "reports" / f"study_{study}.json"
        if not path.exists():
            pytest.skip(f"{path.name} not published")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("study") == study
        assert payload.get("title")
        provenance = payload.get("provenance") or {}
        assert provenance.get("commit")
        assert "config_fingerprint" in provenance

    def test_studies_agree_about_the_corpus(self) -> None:
        """A study that quietly ran on a different set of recordings cannot be
        compared with one that did not."""
        from tools.programme import CORPUS

        for study in (f"d{n:02d}" for n in range(7, 21)):
            path = _ROOT / "reports" / f"study_{study}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            runs = payload.get("runs") or {}
            if not runs:
                continue
            assert set(runs) <= set(CORPUS), f"{study} used recordings outside the corpus"


class TestSearchBudget:
    """A budget that is not enforced turns the budget axis into decoration.

    Д19 published a row labelled "budget 6" whose mean was ten evaluations:
    the ternary narrowing respected the limit and the sweep of the remaining
    window that follows it did not. The rows were then read as a
    cost-against-success curve, which they were not.
    """

    @pytest.mark.parametrize("budget", [1, 2, 4, 6, 8, 10, 12, 16])
    def test_ternary_never_exceeds_its_budget(self, budget: int) -> None:
        import numpy as np

        from tools.studies import _ternary

        rng = np.random.default_rng(20260918)
        for _ in range(40):
            levels = rng.random(int(rng.integers(5, 25)))
            assert _ternary(levels, budget)["evaluations"] <= budget

    @pytest.mark.parametrize("budget", [1, 2, 4, 6, 8, 10, 12, 16])
    def test_hill_climb_never_exceeds_its_budget(self, budget: int) -> None:
        import numpy as np

        from tools.studies import _hill_climb

        rng = np.random.default_rng(20260918)
        for _ in range(40):
            size = int(rng.integers(5, 25))
            levels = rng.random(size)
            start = int(rng.integers(0, size))
            assert _hill_climb(levels, start, budget)["evaluations"] <= budget

    def test_a_larger_budget_never_finds_a_worse_peak(self) -> None:
        """Ternary keeps the best value it has probed, so more probes cannot
        hurt. This is what makes the budget axis readable as a curve."""
        import numpy as np

        from tools.studies import _ternary

        rng = np.random.default_rng(7)
        for _ in range(60):
            levels = rng.random(int(rng.integers(6, 20)))
            small = _ternary(levels, 4)["stopped_at"]
            large = _ternary(levels, len(levels) + 4)["stopped_at"]
            assert levels[large] >= levels[small] - 1e-12

    def test_every_recording_gets_the_same_budgets(self) -> None:
        """Rows averaged over different recordings are not comparable, and the
        published aggregate had a 'budget 30' row over three recordings next to
        a 'budget 6' row over eight."""
        import json

        from tools.studies import SEARCH_BUDGETS

        path = _ROOT / "reports" / "study_d19.json"
        if not path.exists():
            pytest.skip("study_d19.json not published")
        payload = json.loads(path.read_text(encoding="utf-8"))
        counts = {
            key: row.get("mean_evaluations_n")
            for key, row in payload["aggregate"].items()
        }
        assert len(set(counts.values())) == 1, (
            f"rows cover different numbers of recordings: {counts}"
        )
        for budget in SEARCH_BUDGETS:
            assert f"ternary_budget{budget}" in payload["aggregate"]

    def test_published_rows_respect_their_own_budget(self) -> None:
        import json

        path = _ROOT / "reports" / "study_d19.json"
        if not path.exists():
            pytest.skip("study_d19.json not published")
        payload = json.loads(path.read_text(encoding="utf-8"))
        for key, row in payload["aggregate"].items():
            budget = int(key.rsplit("budget", 1)[1])
            assert row["mean_evaluations"] <= budget, (
                f"{key} averages {row['mean_evaluations']} evaluations"
            )
