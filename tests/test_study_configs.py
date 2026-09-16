"""The studies must compare what the library actually ships.

A configuration restated in a second place drifts from the first.  It did:
`fair_comparison` compared a "shipped" pipeline that still had the consensus
kernel on, months after the default turned it off, and therefore understated
the shipped pipeline by 0.016 against the reference.
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from adaptive_sharpness import load_default_config
from tools.ablation import (
    HISTORIC,
    REPAIRED,
    VARIANT_GROUPS,
    check_variants_differ,
    spec_from_config,
)
from tools.fair_comparison import build_pipelines


class TestShippedSpec:
    def test_round_trips_through_the_config(self) -> None:
        config = load_default_config()
        assert spec_from_config(config).build(config) == config

    def test_tracks_a_changed_default(self) -> None:
        config = load_default_config()
        flipped = replace(
            config,
            ensemble=replace(
                config.ensemble, use_agreement=not config.ensemble.use_agreement
            ),
        )
        assert (
            spec_from_config(flipped).use_agreement
            != spec_from_config(config).use_agreement
        )

    def test_notices_a_disabled_reliability_model(self) -> None:
        config = load_default_config()
        flat = replace(
            config,
            ensemble=replace(
                config.ensemble,
                sensitivity={
                    name: {k: 0.0 for k in row}
                    for name, row in config.ensemble.sensitivity.items()
                },
            ),
        )
        assert spec_from_config(config).use_reliability
        assert not spec_from_config(flat).use_reliability


class TestPipelines:
    def test_the_shipped_configuration_is_in_the_comparison(self) -> None:
        config = load_default_config()
        pipelines = build_pipelines(config)
        assert "six:shipped" in pipelines
        assert pipelines["six:shipped"][0].build(config) == config

    def test_the_kernel_variant_actually_differs_from_shipped(self) -> None:
        config = load_default_config()
        pipelines = build_pipelines(config)
        shipped = pipelines["six:shipped"][0]
        with_kernel = pipelines["six:with_kernel"][0]
        assert shipped.use_agreement != with_kernel.use_agreement

    def test_the_best_raw_signal_has_a_pipeline(self) -> None:
        """`wavelet` agrees best with the reference of any raw measure.

        Without its pipeline in the table, "the pipeline costs 0.04" would be
        comparing a fused pipeline against a *different* measure's raw value.
        """
        assert "single:wavelet" in build_pipelines(load_default_config())

    def test_every_pipeline_is_a_distinct_experiment(self) -> None:
        pipelines = build_pipelines(load_default_config())
        specs = {name: spec for name, (spec, _) in pipelines.items()}
        assert not check_variants_differ(specs)

    def test_every_pipeline_carries_a_description(self) -> None:
        for name, (_, note) in build_pipelines(load_default_config()).items():
            assert note, f"{name} has no description"


class TestAblationGroups:
    @pytest.mark.parametrize("group", sorted(VARIANT_GROUPS))
    def test_no_group_contains_duplicate_experiments(self, group: str) -> None:
        assert not check_variants_differ(VARIANT_GROUPS[group])

    def test_repaired_is_not_silently_the_shipped_configuration(self) -> None:
        """REPAIRED answers "what did the repairs buy", with the weighting model
        left as it was.  Conflating it with what ships would credit the repairs
        with a change the factorial made."""
        shipped = spec_from_config(load_default_config())
        assert REPAIRED != shipped
        assert REPAIRED.use_agreement and not shipped.use_agreement

    def test_historic_differs_from_repaired_only_in_the_normalisation(self) -> None:
        differences = {
            field
            for field in HISTORIC.__dataclass_fields__
            if getattr(HISTORIC, field) != getattr(REPAIRED, field)
        }
        assert differences == {
            "window", "min_range_absolute", "long_window_multiple", "mapping",
            "auto_freeze", "ready_informative_fraction", "noise_quantile",
            "edge_ref_density",
        }
