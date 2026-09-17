"""Tests for the shared frame selection and ordering statistics.

Both were found by review to be wrong in ways that changed published numbers:
the ordering statistic resolved ties in favour of one direction, and the three
analysis tools each selected frames differently.  Each defect is pinned here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tools.evaluation import (
    config_fingerprint,
    file_fingerprint,
    ordering_against_truth,
    peak_interval,
    provenance,
    select_frames,
)


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------

class TestOrdering:
    def test_perfect_agreement(self) -> None:
        model = np.array([1.0, 2.0, 3.0, 4.0])
        result = ordering_against_truth(model, model.copy())
        assert result.wrong == 0
        assert result.tied == 0
        assert result.inversion == 0.0

    def test_perfect_disagreement(self) -> None:
        model = np.array([1.0, 2.0, 3.0, 4.0])
        result = ordering_against_truth(model, model[::-1].copy())
        assert result.correct == 0
        assert result.inversion == 1.0

    def test_a_constant_signal_scores_chance_whichever_way_truth_runs(self) -> None:
        """The defect: ties used to be resolved as "the second one is sharper".

        A constant signal therefore scored 0.000 inversions against one
        reference direction and 1.000 against the other, while in fact it
        distinguishes nothing at all.
        """
        constant = np.full(5, 0.5)
        rising = np.arange(5, dtype=float)
        falling = rising[::-1].copy()
        up = ordering_against_truth(constant, rising)
        down = ordering_against_truth(constant, falling)
        assert up.inversion == pytest.approx(0.5)
        assert down.inversion == pytest.approx(0.5)
        assert up.correct == up.wrong == 0
        assert up.tied == up.comparable

    def test_ties_are_counted_not_hidden(self) -> None:
        model = np.array([1.0, 1.0, 3.0])
        result = ordering_against_truth(model, np.array([1.0, 2.0, 3.0]))
        assert result.tied == 1
        assert result.resolved == pytest.approx(2 / 3)

    def test_strict_inversion_ignores_ties(self) -> None:
        model = np.array([1.0, 1.0, 3.0])
        result = ordering_against_truth(model, np.array([1.0, 2.0, 3.0]))
        assert result.inversion_strict == 0.0
        assert result.inversion > 0.0

    def test_reference_ties_are_excluded(self) -> None:
        result = ordering_against_truth(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 1.0, 3.0])
        )
        assert result.reference_tied == 1
        assert result.comparable == 2

    def test_result_is_independent_of_pair_order(self) -> None:
        rng = np.random.default_rng(0)
        model = rng.normal(size=8)
        truth = rng.normal(size=8)
        forward = ordering_against_truth(model, truth)
        order = np.argsort(rng.normal(size=8))
        shuffled = ordering_against_truth(model[order], truth[order])
        assert forward.inversion == pytest.approx(shuffled.inversion)

    def test_non_finite_values_are_skipped(self) -> None:
        model = np.array([1.0, np.nan, 3.0])
        result = ordering_against_truth(model, np.array([1.0, 2.0, 3.0]))
        assert result.comparable == 1


class TestPeakInterval:
    def test_single_maximum(self) -> None:
        assert peak_interval(np.array([0.0, 1.0, 0.5])) == (1, 1)

    def test_plateau_reports_both_ends(self) -> None:
        """argmax answers a different question than "where is the peak"."""
        assert peak_interval(np.array([0.0, 1.0, 1.0, 1.0, 0.2])) == (1, 3)

    def test_all_nan(self) -> None:
        assert peak_interval(np.full(3, np.nan)) == (-1, -1)


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------

class TestSelection:
    @staticmethod
    def _inputs(n: int = 10):
        step = np.arange(1, n + 1)
        hold = np.ones(n, dtype=bool)
        return step, hold

    def test_moving_frames_are_excluded(self) -> None:
        step, hold = self._inputs()
        hold[:3] = False
        selection = select_frames(step=step, hold=hold)
        assert selection.count == 7
        assert selection.reasons["moving"] == 3

    def test_unlabelled_frames_are_excluded(self) -> None:
        step, hold = self._inputs()
        step[:2] = 0
        selection = select_frames(step=step, hold=hold)
        assert selection.reasons["no_step_label"] == 2

    def test_a_wandering_spot_is_excluded(self) -> None:
        step, hold = self._inputs()
        radius = np.full(10, 5.0)
        offset = np.zeros(10)
        offset[:4] = 50.0
        selection = select_frames(
            step=step, hold=hold, spot_radius=radius, spot_offset=offset
        )
        assert selection.reasons["spot_wandered"] == 4
        assert selection.count == 6

    def test_reference_is_applied_when_present_even_if_not_required(self) -> None:
        """The two tools disagreed here, by 3.3% and 5.5% of frames."""
        step, hold = self._inputs()
        radius = np.full(10, 5.0)
        radius[0] = np.nan
        offset = np.zeros(10)
        selection = select_frames(
            step=step, hold=hold, spot_radius=radius, spot_offset=offset
        )
        assert selection.count == 9

    def test_reasons_do_not_double_count(self) -> None:
        step, hold = self._inputs()
        hold[:3] = False
        step[:3] = 0
        selection = select_frames(step=step, hold=hold)
        assert selection.reasons["moving"] == 3
        assert "no_step_label" not in selection.reasons
        assert selection.count == 7

    def test_summary_accounts_for_every_frame(self) -> None:
        step, hold = self._inputs()
        hold[:2] = False
        step[5:7] = 0
        selection = select_frames(step=step, hold=hold)
        summary = selection.summary()
        assert summary["selected"] + sum(summary["excluded"].values()) == summary["total_frames"]


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class TestProvenance:
    def test_config_fingerprint_changes_with_a_setting(self) -> None:
        from dataclasses import replace

        from adaptive_sharpness import load_default_config

        config = load_default_config()
        changed = replace(
            config, analysis=replace(config.analysis, noise_quantile=90.0)
        )
        assert config_fingerprint(config) != config_fingerprint(changed)

    def test_config_fingerprint_is_stable(self) -> None:
        from adaptive_sharpness import load_default_config

        assert config_fingerprint(load_default_config()) == config_fingerprint(
            load_default_config()
        )

    def test_file_fingerprint_notices_a_changed_file(self, tmp_path) -> None:
        first = tmp_path / "a.bin"
        first.write_bytes(b"hello")
        before = file_fingerprint([first])
        first.write_bytes(b"world")
        assert file_fingerprint([first]) != before

    def test_file_fingerprint_notices_a_reordering(self, tmp_path) -> None:
        a, b = tmp_path / "a.bin", tmp_path / "b.bin"
        a.write_bytes(b"aaaa")
        b.write_bytes(b"bbbb")
        assert file_fingerprint([a, b]) != file_fingerprint([b, a])

    def test_provenance_records_what_produced_a_number(self) -> None:
        from adaptive_sharpness import load_default_config

        record = provenance(load_default_config(), group="test")
        for key in ("commit", "python", "numpy", "opencv", "config_fingerprint"):
            assert key in record
        assert record["group"] == "test"


class TestReportsSurviveANarrowTerminal:
    """A table that will not encode must not be able to destroy a run.

    An hour of replay was lost to a single "+/-" in a header: the report was
    printed before the JSON was written, the print raised UnicodeEncodeError
    under the Pi's ASCII stdout, and nothing was saved.
    """

    @staticmethod
    def _sources() -> list[Path]:
        root = Path(__file__).resolve().parents[1] / "tools"
        return [
            root / name for name in (
                "ablation.py", "before_after.py", "evaluation.py",
                "fair_comparison.py", "protocol_study.py",
                "reference_sensitivity.py", "replay_configs.py",
            )
        ]

    def test_tool_sources_are_ascii(self) -> None:
        offenders = {}
        for path in self._sources():
            text = path.read_text(encoding="utf-8")
            bad = sorted({c for c in text if ord(c) > 127})
            if bad:
                offenders[path.name] = bad
        assert not offenders, f"non-ASCII in printing tools: {offenders}"

    def test_the_result_is_saved_before_it_is_printed(self) -> None:
        """Order matters: the expensive artefact must not depend on the cheap one."""
        source = (Path(__file__).resolve().parents[1] / "tools" / "ablation.py").read_text(
            encoding="utf-8"
        )
        write = source.index("args.json.write_text")
        render = source.index("print_report(results)", source.index("def main"))
        assert write < render


class TestCacheValidation:
    """One check, shared, or the readers disagree about what is stale.

    Three scripts each wrote their own: one verified the frame count and the
    spot parameters, another only the frame count.  A cache written before a
    configuration change therefore passed one reader and failed another, and
    numbers from the two could not be compared.
    """

    @staticmethod
    def _write(tmp_path, *, files, file_fp, processing_fp=None, spot_fp=None,
               n=None):
        import numpy as np

        frames = tmp_path / "frames"
        frames.mkdir(exist_ok=True)
        for name in files:
            (frames / name).write_bytes(name.encode())
        payload = {
            "n": n if n is not None else len(files),
            "file_fingerprint": np.array(file_fp),
            "raw_tenengrad": np.zeros(len(files)),
            "spot_radius": np.ones(len(files)),
            "spot_offset": np.zeros(len(files)),
        }
        if processing_fp is not None:
            payload["processing_fingerprint"] = np.array(processing_fp)
        if spot_fp is not None:
            payload["spot_fingerprint"] = np.array(spot_fp)
        np.savez_compressed(tmp_path / "measures_cache.npz", **payload)

    @staticmethod
    def _current(tmp_path, files):
        return file_fingerprint([tmp_path / "frames" / n for n in files])

    def test_a_matching_cache_is_accepted(self, tmp_path) -> None:
        import json

        from tools.evaluation import SPOT_PARAMETERS, load_measure_cache

        files = ["a.png", "b.png", "c.png"]
        self._write(
            tmp_path, files=files, file_fp="placeholder",
            processing_fp="proc", spot_fp=json.dumps(SPOT_PARAMETERS, sort_keys=True),
        )
        self._write(
            tmp_path, files=files, file_fp=self._current(tmp_path, files),
            processing_fp="proc", spot_fp=json.dumps(SPOT_PARAMETERS, sort_keys=True),
        )
        cached = load_measure_cache(tmp_path, files, processing_fingerprint="proc")
        assert cached.has_reference
        assert cached.raw is not None

    def test_a_changed_frame_invalidates_it(self, tmp_path) -> None:
        import json

        from tools.evaluation import SPOT_PARAMETERS, load_measure_cache

        files = ["a.png", "b.png"]
        self._write(tmp_path, files=files, file_fp="x")
        self._write(
            tmp_path, files=files, file_fp=self._current(tmp_path, files),
            processing_fp="proc", spot_fp=json.dumps(SPOT_PARAMETERS, sort_keys=True),
        )
        (tmp_path / "frames" / "a.png").write_bytes(b"different")
        cached = load_measure_cache(tmp_path, files, processing_fingerprint="proc")
        assert not cached.has_reference
        assert cached.raw is None

    def test_stale_processing_invalidates_the_raw_metrics_only(self, tmp_path) -> None:
        """The reference is measured from the frames, so a metric change does
        not invalidate it - but it must not silently validate the metrics."""
        import json

        from tools.evaluation import SPOT_PARAMETERS, load_measure_cache

        files = ["a.png", "b.png"]
        self._write(tmp_path, files=files, file_fp="x")
        self._write(
            tmp_path, files=files, file_fp=self._current(tmp_path, files),
            processing_fp="old", spot_fp=json.dumps(SPOT_PARAMETERS, sort_keys=True),
        )
        cached = load_measure_cache(tmp_path, files, processing_fingerprint="new")
        assert cached.raw is None
        assert cached.has_reference
        assert "stale" in cached.note

    def test_a_cache_without_fingerprints_is_refused(self, tmp_path) -> None:
        import numpy as np

        from tools.evaluation import load_measure_cache

        frames = tmp_path / "frames"
        frames.mkdir()
        for name in ("a.png", "b.png"):
            (frames / name).write_bytes(b"x")
        np.savez_compressed(
            tmp_path / "measures_cache.npz", n=2,
            spot_radius=np.ones(2), spot_offset=np.zeros(2),
        )
        cached = load_measure_cache(tmp_path, ["a.png", "b.png"])
        assert not cached.has_reference
        assert "predates" in cached.note

    def test_a_wrong_frame_count_is_refused(self, tmp_path) -> None:
        from tools.evaluation import load_measure_cache

        files = ["a.png", "b.png"]
        self._write(tmp_path, files=files, file_fp="x", n=99)
        cached = load_measure_cache(tmp_path, files)
        assert not cached.has_reference
        assert "frames" in cached.note


class TestProvenanceFlagMeansSomething:
    """A run that records itself must not report itself as a modification.

    The flag fired on every single run, twice for different reasons: first the
    report was untracked and untracked files counted as dirt, then the reports
    were committed and writing one made a tracked file modified.  Either way it
    stopped carrying information.  It is now restricted to the paths whose
    contents can change what a run computes.
    """

    def test_only_source_paths_are_checked(self) -> None:
        from tools.evaluation import SOURCE_PATHS

        for output in ("reports", "docs", "data"):
            assert output not in SOURCE_PATHS

    def test_source_paths_cover_the_code_that_runs(self) -> None:
        from tools.evaluation import SOURCE_PATHS

        for source in ("src", "tools", "tests"):
            assert source in SOURCE_PATHS

    def test_a_modified_source_file_is_named_not_just_counted(self) -> None:
        """A bare boolean cannot be acted on; the file list can."""
        from adaptive_sharpness import load_default_config
        from tools.evaluation import provenance

        record = provenance(load_default_config())
        assert isinstance(record["source_modified_files"], list)
        assert record["source_modified"] == bool(record["source_modified_files"])

    def test_paths_survive_the_porcelain_format(self) -> None:
        """Porcelain leaves the first column blank for an unstaged change, so
        stripping the whole output eats a character of the first path."""
        from adaptive_sharpness import load_default_config
        from tools.evaluation import provenance

        for name in provenance(load_default_config())["source_modified_files"]:
            assert name.split("/")[0] in {
                "src", "tools", "tests", "demo", "config", "pyproject.toml",
            }, f"path looks truncated: {name}"


class TestProcessingFingerprint:
    """A cache is written by one tool and read by another.

    `protocol_study.py` writes the raw metric arrays; `fair_comparison.py`
    compares them against streaming pipelines built from the *current* config.
    The fingerprint that says the two describe the same experiment used to be
    built inside the writer, so the reader had no way to ask for the same
    string - and did not check at all, reopening the `.npz` directly.  It now
    lives in one place and both call it.
    """

    def test_a_changed_analysis_width_changes_the_fingerprint(self) -> None:
        from dataclasses import replace

        from adaptive_sharpness import load_default_config
        from tools.evaluation import processing_fingerprint

        config = load_default_config()
        names = ("wavelet", "tenengrad")
        wider = replace(
            config, pipeline=replace(config.pipeline, analysis_width=640)
        )
        assert processing_fingerprint(config, names) != processing_fingerprint(
            wider, names
        )

    def test_a_changed_noise_quantile_changes_the_fingerprint(self) -> None:
        """The defect this guards is not hypothetical: the noise quantile moved
        from 50 to 75, and a cache written before that describes a different
        measurement."""
        from dataclasses import replace

        from adaptive_sharpness import load_default_config
        from tools.evaluation import processing_fingerprint

        config = load_default_config()
        names = ("wavelet",)
        median = replace(
            config, analysis=replace(config.analysis, noise_quantile=50.0)
        )
        assert processing_fingerprint(config, names) != processing_fingerprint(
            median, names
        )

    def test_a_different_measure_set_changes_the_fingerprint(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.evaluation import processing_fingerprint

        config = load_default_config()
        assert processing_fingerprint(config, ("wavelet",)) != processing_fingerprint(
            config, ("wavelet", "brenner")
        )

    def test_the_same_configuration_gives_the_same_fingerprint(self) -> None:
        from adaptive_sharpness import load_default_config
        from tools.evaluation import processing_fingerprint

        names = ("wavelet", "brenner")
        assert processing_fingerprint(
            load_default_config(), names
        ) == processing_fingerprint(load_default_config(), names)


class TestFairComparisonChecksItsCache:
    """The raw signals and the pipelines in that report are compared against
    each other, so raw arrays from a different configuration would make the
    whole table incoherent.  The tool read them without checking."""

    def test_it_asks_for_the_fingerprint(self) -> None:
        source = (
            Path(__file__).resolve().parents[1] / "tools" / "fair_comparison.py"
        ).read_text(encoding="utf-8")
        assert "processing_fingerprint(config, names)" in source

    def test_it_does_not_reopen_the_cache_behind_the_loader(self) -> None:
        """Reopening the `.npz` is what skipped the check."""
        source = (
            Path(__file__).resolve().parents[1] / "tools" / "fair_comparison.py"
        ).read_text(encoding="utf-8")
        assert "np.load(cached.path" not in source
        assert "cached.raw" in source

    def test_unverified_raw_metrics_stop_the_run(self) -> None:
        """A stale cache must produce no row rather than a wrong one."""
        from tools.evaluation import CachedMeasures

        stale = CachedMeasures(
            path=Path("nowhere.npz"), frames=10, raw=None,
            spot_radius=None, spot_offset=None,
            note="raw metrics stale, not used",
        )
        assert stale.raw is None
        assert not stale.has_reference
