# Release 1.0.0 — the research stage, closed

This is the state at which the measurement programme finished. It records what
was measured, on what, with which code, and — the part that matters most for
anyone picking this up — **what it is not entitled to claim**.

---

## What this release contains

| | |
| --- | --- |
| Library | `adaptive_sharpness`, six focus measures, streaming normalisation, condition-based weighting, temporal filter |
| Corpus | 8 recordings, 9017 frames, one lens, one room, one day |
| Independent reference | point-source encircled-energy radius, on **2** of the 8 recordings |
| Studies | 20 (Д01–Д06 from published reports, Д07–Д20 on the frames) |
| Reports | 25 JSON files in [`reports/`](../reports/), each with its own provenance |
| Tests | 600 |

Every number quoted in prose is registered in [`claims.toml`](claims.toml) and
checked against the reports by [`tests/test_claims.py`](../tests/test_claims.py),
so a document cannot drift from the data that produced it.

---

## Reproducing it

```bash
pip install -e .
python3 -m pytest tests/ -q
```

With the recordings present (they are **not** in the repository — they are
photographs of a private room):

```bash
# the eleven Pi reports; run on the target hardware, timings depend on it
python3 tools/protocol_study.py data --json reports/protocol_study.json
python3 tools/ablation.py data --group factorial --json reports/ablation_factorial.json
python3 tools/fair_comparison.py data --json reports/fair_comparison.json

# the fourteen studies; machine-independent conclusions, run anywhere
python3 tools/programme.py all --data <recordings> --workers 32

# everything derived
python3 tools/render_results.py reports --out docs/RESULTS.md --figures docs/figures
python3 tools/figures_ua.py
python3 tools/figures_studies_ua.py
python3 tools/report_pdf_ua.py
```

A local copy of the frames reads about six times faster than a network share,
which matters when a study replays the corpus forty times.

**Provenance.** Every report records the commit, whether the source tree was
modified, the library versions, the platform and a fingerprint of the full
configuration. All fourteen study reports carry `source_modified: false` at
commit `4bb5031`. The eleven earlier reports were measured on the Raspberry Pi
5 and their timing columns are the ones to use; the study reports were computed
on a desktop and theirs are not comparable.

---

## Configuration that these results describe

| Setting | Value | Why |
| --- | --- | --- |
| `metrics.enabled` | six measures | Д15 found a five-measure subset slightly better, inside the corpus's resolution — not enough to change a default |
| `pipeline.analysis_width` | 320 | best on both criteria; wider is slower and agrees with the reference *less* (Д16) |
| `normalization.mapping` | logistic | removes the linear map's 10 % clipping at no cost (Д09) |
| `normalization.auto_freeze` | on, 240 / 120 / 0.05 | the freeze is the largest single effect measured here; these thresholds fire on 7 recordings of 8 (Д11) |
| `analysis.noise_quantile` | 75 | the median is pinned at zero on an 8-bit stream |
| `ensemble.use_agreement` | **off** | negative on both criteria; widening the kernel only approaches "off" |
| `temporal.alpha_base` | 0.35 | mid-point of a smoothing-against-latency trade-off that has no optimum without an actuator (Д14) |

---

## Limits of applicability — read before using any of this

**The score is relative to one stream.** It is not absolute sharpness and is
not comparable across scenes, cameras, ROIs or evaluator instances. Two
evaluators both reporting 0.8 are not looking at equally sharp images.

**It cannot tell you the stream is defocused.** Blur every frame and the mean
score does not move: the normaliser refits to what it is shown, so the
least-blurred of a blurred set still reads near the top.

**It depends on when you switched the evaluator on.** The same frames scored
after a different warm-up differ by up to 0.64, with rank agreement falling to
0.56. Calibrate, freeze, then compare — the sequence is
[`demo/calibrated_focus_loop.py`](../demo/calibrated_focus_loop.py).

**Eight recordings resolve about 0.04.** Any difference smaller than that is
not established by this corpus, in either direction. In particular, adaptive
weighting is **not shown to beat fixed weighting**, and that is not a
demonstration that they are equivalent.

**The confidence is a diagnostic, not a gate.** Its components say what is
wrong with a frame; the scalar does not say the frame is less accurate.
Measured within protocol steps, confident frames were the accurate ones on four
recordings of eight.

**No closed loop was measured.** The lens position was never recorded, so
nothing here says anything about autofocus latency. A search *simulated over
recorded profiles* lands within one step of the reference, which is a statement
about the signal's shape and not about a camera.

---

## What would change the picture

Everything that needs **new recordings** rather than new computation on the
same frames:

- scenes with speculars and genuinely high ISO — the only way to test what the
  reliability model and the agreement kernel exist for;
- other lenses and rooms — whether "five measures beat six" and "320 px" travel;
- a closed loop with the lens position recorded;
- simply more scenes: separating an effect of size 0.01 needs tens of
  independent scenes, not eight.

---

## Where to read further

| Document | Contents |
| --- | --- |
| [STUDY_UA_REPORT.md](STUDY_UA_REPORT.md) | how the module works, five diagrams, change history, and the synthesis in §9 |
| [STUDY_UA_RESULTS.md](STUDY_UA_RESULTS.md) | all fourteen studies: what was tested, how, what came out, what it does not prove |
| [ZVIT_UA.pdf](ZVIT_UA.pdf) | four-page Ukrainian summary, generated from the reports |
| [RESULTS.md](RESULTS.md) | every table, generated from the stored reports |
| [CALIBRATION.md](CALIBRATION.md) | the English account of what the recordings changed |
| [LIMITATIONS.md](LIMITATIONS.md) | the full list, with reasoning |
| [INTEGRATION.md](INTEGRATION.md) | using the library inside a focus loop |
