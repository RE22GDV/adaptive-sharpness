# Field test: a 70-second handheld recording

The first recording of a real session on the reference hardware, and what it
found. Everything below is measured; the raw recording is reproducible with the
commands at the end.

## The run

| | |
| --- | --- |
| Source | Panasonic GH6 PTP live view, threaded | 
| Host | Raspberry Pi 5, Python 3.11.2, NumPy 1.26.4, OpenCV 4.13.0 |
| Duration | 70.7 s, 1288 frames |
| Sampled frames | 129 PNG (every 10th) |
| Operation | handheld, focus ring turned back and forth by hand |
| Recorded by | the UI's REC button |

```bash
python3 tools/analyze_recording.py data/run_20260909_104332
```

## First: the recording chain is trustworthy

Before drawing any conclusion from a recording, the recording itself has to be
verified. Every one of the 129 saved frames was re-evaluated offline and
compared against the numbers written at capture time:

| quantity | worst relative difference |
| --- | --- |
| all six raw metrics | **0.000e+00** |
| edge density, contrast, brightness, noise sigma | **0.000e+00** |

Exactly zero, not merely small. The saved PNGs are bit-identical to what was
measured, so offline re-analysis of a recording is sound. This is now a
permanent test (`tests/test_recording.py::TestRoundTrip`).

Motion is excluded from that check by construction: it is defined between
consecutive frames, and only every 10th frame is saved.

## What the analysis found

```
[  ok  ] frame rate               18.22 fps, interval 54.9 ms mean / 66.1 p95
[  ok  ] processing time          30.4 ms mean / 41.6 p95 / 49.5 max
[  ok  ] score smoothness         lag-1 autocorrelation 0.986, median step 0.0094
[  ok  ] readiness                ready from frame 0, 100% of frames
[  ok  ] normaliser informative   mean informative_fraction 1.00
[ WARN ] confidence               median 0.58, below 0.35 on 42% of frames
[ WARN ] normalisation saturation 'fourier' pinned at 0 or 1 on 22% of frames
[  ok  ] metric agreement         robust spread 0.082 mean / 0.273 p95
[ WARN ] weight stability         mean total weight moved per frame 0.110
[ WARN ] score-change detections  225 of 1288 frames (17%)
[  ok  ] edge sufficiency         mean 0.73
[  ok  ] subject stability        'sharp area' on 100% of frames
[  ok  ] decision margin          mean 0.076 over 469 contested frames
[  ok  ] measurable area          mean 81% of tiles
```

---

## Defect 1 — the confidence collapsed to exactly zero on 39% of frames

**Symptom.** Median confidence 0.58, and 496 of 1288 frames reported a hard
0.000.

**Cause, in two parts.** Reconstructing the six confidence factors from the
recorded statistics showed that on those frames the motion factor was zero on
**97%** of them, while edges, SNR, exposure and contrast were all healthy.

Measured inter-frame motion during handheld operation:

| percentile | motion, px |
| --- | --- |
| p50 | 1.37 |
| p75 | 5.82 |
| p90 | 17.93 |
| p95 | 28.99 |
| p99 | 56.62 |

`motion_ref_px` was **3.0** — a guess. Ordinary hand shake while turning a focus
ring saturated it on 38% of frames.

That alone would have been a calibration error. The second part is a design
error: the weakest-link cap

```
C = min(geometric_mean(factors), sqrt(min(factors)))
```

was applied to **every** factor. It was designed for *necessary* conditions —
with no edges in the frame there is nothing whose sharpness could be measured.
But motion is a *degradation*, not an impossibility: a moving frame is still
measurable, just less reliably. Capping on it turns a saturated soft factor into
exactly zero.

**Fix.** The cap now applies only to the necessary factors (edges, contrast,
warm-up, normaliser informativeness). Motion, SNR, exposure and concordance
enter through the geometric mean alone. `motion_ref_px` is raised to 15.0 from
the measured distribution.

**Verified on the same recording**, by recomputing the confidence from the
recorded statistics:

| variant | median | exactly 0 | below 0.35 | above 0.6 |
| --- | --- | --- | --- | --- |
| as recorded | 0.583 | **39%** | 42% | 47% |
| new reference only | 0.809 | 16% | 20% | 68% |
| necessary-only cap only | 0.676 | 5% | 39% | 53% |
| **both fixes** | **0.857** | **5%** | **18%** | **71%** |

The safety property survives — the fix does not simply inflate the number:

| frames | median confidence after the fix |
| --- | --- |
| edge sufficiency below 0.2 (142 frames) | **0.189** |
| motion above 30 px (59 frames) | **0.193** |
| motion below 2 px (720 frames) | **0.938** |

Two regression tests pin both directions: saturated motion must not zero the
confidence, and missing edges still must.

---

## Defect 2 — the analysis tool misreported the decision margin

**Symptom.** `decision margin mean 0.028` flagged as a WARN.

**Cause.** `SceneResult.separation()` returns 0 when there is no runner-up, and
819 of the 1288 frames (64%) had a single candidate region. Averaging those
forced zeros in reported a weak decision where there was simply nothing to
decide.

**Fix.** The check now averages only over frames that had a runner-up and
reports the count of single-candidate frames separately. The same data then
reads `mean 0.076 over the 469 frames with a runner-up`, which is OK.

---

## Defect 3 — the timeline figure invited a wrong conclusion

**Symptom.** The generated timeline showed the score apparently oscillating
between 0 and 1 on almost every frame. That reads as a badly unstable pipeline.

**Cause.** 1288 points drawn into a narrow axis. The data is smooth:

- lag-1 autocorrelation **0.986**
- median frame-to-frame change **0.009**
- exactly **1** step larger than 0.5, out of 1287

The figure was wrong, not the pipeline. This was nearly reported as a defect in
the algorithm.

**Fix.** The figure now widens with the length of the run rather than
compressing, and a `score smoothness` check states the autocorrelation as a
number so the fact does not have to be read off a plot.

---

## Not defects, but worth recording

**18.2 fps, not the benchmarked 25.** Accounted for in full:

| | ms |
| --- | --- |
| capture latency | 39.5 |
| evaluator total (incl. scene stage 6.2) | 30.4 |
| measured interval | 54.9 |
| unaccounted — UI render, `imshow`, recording | **24.4** |

The pipeline is unchanged; the UI and the recorder cost 24 ms per frame. A
headless recording via `tools/collect_dataset.py` does not pay that. This is a
property of recording *from the UI*, and it is the honest number for that mode.

**Score-change detections on 17% of frames.** The flag is
`score_change_detected`, not "focus change" — precisely because motion, a ROI
change or an exposure change trigger it too. During a handheld session with the
focus ring being turned, a rate this high is expected behaviour rather than a
fault. The WARN threshold (15%) is conservative for handheld use.

**Weight churn 0.110 per frame.** The adaptive weights move by 11% of their
total per frame, and slightly more on low-confidence frames (0.124 against
0.099). This is the same variance cost already measured in the synthetic
repeatability experiment: the weights are computed from noisy measurements, so
they add noise of their own. Not fixed.

---

## Still open

**Normalisation saturation, 18–22% per metric.** Every metric spends about a
fifth of its frames pinned at 0 or 1, because the running normaliser clips to
`[0, 1]`. Near best focus this destroys the ability to resolve adjacent frames —
the same effect that made the offline hill-climb criterion meaningless until it
was corrected, and the reason `test_best_focus_index_is_sharpest` had to be
rewritten around a plateau.

The proper fix is to keep an unsaturated internal focus signal and expose the
clipped `[0, 1]` value only for display. That is not done.

---

## Reproducing

```bash
# record (or press REC in the UI)
python3 tools/collect_dataset.py --backend gphoto2 --duration 60 \
    --save-every 10 --scene --out data/run1

# analyse
python3 tools/analyze_recording.py data/run1 --figures docs/figures
```

Turn the focus ring during the recording. A static scene leaves the normalisers
with no observed range, and every score in the recording is then a placeholder —
the analyser reports that as `FAIL: readiness`.
