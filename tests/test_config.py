"""Tests for configuration loading, validation and the running normaliser."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sharpness.config import (
    METRIC_NAMES,
    NormalizationConfig,
    SharpnessConfig,
    load_config,
)
from sharpness.normalize import NormalizerBank, RunningNormalizer


class TestDefaults:
    def test_defaults_are_self_consistent(self) -> None:
        config = SharpnessConfig()
        assert set(config.metrics.enabled) <= set(METRIC_NAMES)
        for name in config.metrics.enabled:
            assert name in config.metrics.priors
            assert name in config.ensemble.sensitivity

    def test_load_none_gives_defaults(self) -> None:
        assert load_config(None) == SharpnessConfig()

    def test_config_is_frozen(self) -> None:
        config = SharpnessConfig()
        with pytest.raises(Exception):
            config.pipeline.analysis_width = 999  # type: ignore[misc]


class TestOverrides:
    def test_applies_a_section_override(self) -> None:
        config = SharpnessConfig().with_overrides(pipeline={"analysis_width": 480})
        assert config.pipeline.analysis_width == 480

    def test_leaves_other_sections_alone(self) -> None:
        base = SharpnessConfig()
        config = base.with_overrides(pipeline={"analysis_width": 480})
        assert config.temporal == base.temporal

    def test_unknown_section_raises(self) -> None:
        with pytest.raises(KeyError, match="unknown configuration section"):
            SharpnessConfig().with_overrides(nonsense={"a": 1})

    def test_unknown_key_raises(self) -> None:
        with pytest.raises(TypeError):
            SharpnessConfig().with_overrides(pipeline={"not_a_field": 1})


class TestTomlLoading:
    def test_loads_a_file(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.toml"
        path.write_text(
            "[pipeline]\nanalysis_width = 240\n\n"
            "[temporal]\nalpha_base = 0.5\nenabled = false\n",
            encoding="utf-8",
        )
        config = load_config(path)
        assert config.pipeline.analysis_width == 240
        assert config.temporal.alpha_base == 0.5
        assert config.temporal.enabled is False
        # Untouched sections keep their defaults.
        assert config.metrics.enabled == SharpnessConfig().metrics.enabled

    def test_list_becomes_tuple(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.toml"
        path.write_text(
            '[metrics]\nenabled = ["laplacian", "brenner"]\n', encoding="utf-8"
        )
        config = load_config(path)
        assert config.metrics.enabled == ("laplacian", "brenner")

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "absent.toml")

    def test_typo_in_a_key_fails_loudly(self, tmp_path: Path) -> None:
        """A silently ignored typo would be far worse than an error."""
        path = tmp_path / "cfg.toml"
        path.write_text("[pipeline]\nanalisys_width = 240\n", encoding="utf-8")
        with pytest.raises(KeyError, match="analisys_width"):
            load_config(path)

    def test_unknown_section_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "cfg.toml"
        path.write_text("[pipelin]\nanalysis_width = 240\n", encoding="utf-8")
        with pytest.raises(KeyError, match="unknown configuration section"):
            load_config(path)

    def test_shipped_default_config_loads(self) -> None:
        path = Path(__file__).resolve().parents[1] / "config" / "default.toml"
        if not path.is_file():
            pytest.skip("config/default.toml is not present")
        config = load_config(path)
        assert isinstance(config, SharpnessConfig)


class TestRunningNormalizer:
    def test_output_is_bounded(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        rng = np.random.default_rng(0)
        for value in rng.uniform(0, 1000, 200):
            assert 0.0 <= normalizer.normalize(float(value)) <= 1.0

    def test_neutral_before_any_history(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        assert normalizer.normalize(5.0) == 0.5

    def test_preserves_ordering(self) -> None:
        """The mapping must be monotone or it would reorder focus positions."""
        normalizer = RunningNormalizer("m", NormalizationConfig())
        for value in np.linspace(1.0, 100.0, 60):
            normalizer.observe(float(value))
        outputs = [normalizer.normalize(float(v), observe=False)
                   for v in np.linspace(1.0, 100.0, 40)]
        assert all(b >= a - 1e-9 for a, b in zip(outputs, outputs[1:]))

    def test_constant_input_gives_neutral(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        for _ in range(50):
            normalizer.observe(7.0)
        assert normalizer.normalize(7.0, observe=False) == 0.5

    def test_warmup_flag(self) -> None:
        config = NormalizationConfig(warmup=5)
        normalizer = RunningNormalizer("m", config)
        for index in range(4):
            normalizer.observe(float(index))
            assert not normalizer.warmed_up
        normalizer.observe(4.0)
        assert normalizer.warmed_up

    def test_window_is_bounded(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig(window=10))
        for index in range(50):
            normalizer.observe(float(index))
        assert normalizer.sample_count == 10

    def test_freeze_locks_the_scale(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        for value in np.linspace(0.0, 100.0, 60):
            normalizer.observe(float(value))
        normalizer.freeze()
        frozen = normalizer.normalize(50.0, observe=False)
        for _ in range(60):
            normalizer.observe(1000.0)
        assert normalizer.normalize(50.0, observe=False) == pytest.approx(frozen)

    def test_unfreeze_restores_adaptation(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        for value in np.linspace(0.0, 100.0, 60):
            normalizer.observe(float(value))
        normalizer.freeze()
        normalizer.unfreeze()
        # Varied values, not a constant: a constant history is a degenerate
        # range and the normaliser deliberately returns a neutral 0.5 for it.
        for value in np.linspace(900.0, 1100.0, 150):
            normalizer.observe(float(value))
        assert normalizer.normalize(50.0, observe=False) < 0.1

    def test_freeze_without_data_raises(self) -> None:
        with pytest.raises(RuntimeError, match="too few samples"):
            RunningNormalizer("m", NormalizationConfig()).freeze()

    def test_fit_from_a_batch(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        normalizer.fit(np.linspace(0.0, 10.0, 50).tolist())
        assert normalizer.normalize(0.0, observe=False) < 0.2
        assert normalizer.normalize(10.0, observe=False) > 0.8

    def test_reset_clears_everything(self) -> None:
        normalizer = RunningNormalizer("m", NormalizationConfig())
        for value in range(50):
            normalizer.observe(float(value))
        normalizer.reset()
        assert normalizer.sample_count == 0
        assert normalizer.normalize(5.0, observe=False) == 0.5

    def test_rejects_bad_window(self) -> None:
        with pytest.raises(ValueError, match="window"):
            RunningNormalizer("m", NormalizationConfig(window=1))

    def test_rejects_inverted_percentiles(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            RunningNormalizer("m", NormalizationConfig(low_percentile=90, high_percentile=10))


class TestNormalizerBank:
    def test_normalises_every_metric(self) -> None:
        bank = NormalizerBank(METRIC_NAMES, NormalizationConfig())
        values = {name: float(index + 1) for index, name in enumerate(METRIC_NAMES)}
        out = bank.normalize(values)
        assert set(out) == set(METRIC_NAMES)

    def test_unknown_metric_raises(self) -> None:
        bank = NormalizerBank(("laplacian",), NormalizationConfig())
        with pytest.raises(KeyError, match="no normalizer"):
            bank.normalize({"mystery": 1.0})

    def test_warmed_up_needs_all_metrics(self) -> None:
        config = NormalizationConfig(warmup=3)
        bank = NormalizerBank(("a", "b"), config)
        for _ in range(3):
            bank["a"].observe(1.0)
        assert not bank.warmed_up
        for _ in range(3):
            bank["b"].observe(1.0)
        assert bank.warmed_up

    def test_reset_clears_all(self) -> None:
        bank = NormalizerBank(METRIC_NAMES, NormalizationConfig())
        for _ in range(10):
            bank.normalize({name: 1.0 for name in METRIC_NAMES})
        bank.reset()
        assert not bank.warmed_up
