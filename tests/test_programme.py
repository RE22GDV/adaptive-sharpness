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
