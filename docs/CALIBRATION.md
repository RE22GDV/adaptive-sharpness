# Calibration: what the protocol recordings changed

Every constant in this project used to come from one of two places: a
qualitative argument, or a single 70-second handheld recording. This document
records what eight tripod recordings with exact focus-step labels said instead,
which defaults changed as a result, and — after a review of the tooling — which
of those answers the measurements can actually support.

**Every number lives in [RESULTS.md](RESULTS.md)**, generated from the stored
reports with the commit and configuration fingerprint of the run that produced
it. Nothing is transcribed here by hand; this file explains the mechanisms and
states the conclusions.

Source: 9 017 saved frames across eight recordings made with the UI's protocol
buttons — four exposure strata, two full sweeps (rich and low texture), and two
point-source runs. Hardware: Raspberry Pi 5, Panasonic GH6 live view at
640×360, 320-wide analysis. The recordings are not published: they are
photographs of the operator's room.

```bash
python3 tools/protocol_study.py data --refresh --json data/protocol_study.json
python3 tools/ablation.py data --group fixes         --json data/ablation_fixes.json
python3 tools/ablation.py data --group normalisation --json data/ablation_normalisation.json
python3 tools/ablation.py data --group factorial     --json data/ablation_factorial.json
python3 tools/ablation.py data --group metrics       --json data/ablation_metrics.json
python3 tools/fair_comparison.py data        --json data/fair_comparison.json
python3 tools/reference_sensitivity.py data  --json data/reference_sensitivity.json
python3 tools/render_results.py data --out docs/RESULTS.md
```

---

## The instruments were wrong too

Before any result below could be trusted, the code producing it had to be
repaired. Each defect was reproduced before being fixed, and each invalidated
some published figure.

| defect | what it did | how it was shown |
| --- | --- | --- |
| Ablation variants inherited the shipped defaults | moving `use_agreement` to false made `adaptive` a duplicate of `no_agreement`, and `no_reliability` a duplicate of `fixed_weights` | the built configurations compared equal |
| Ties in the ordering statistic resolved one way | a constant signal scored 1.000 inversions against one reference direction and 0.000 against the other | reproduced on a synthetic constant |
| Each tool selected its own frames | two tools reporting the same quantity described different samples | 3.3% and 5.5% of frames differed |
| The measure cache was keyed on frame count and metric names | a changed `analysis_width` or `noise_quantile` silently reused stale statistics | the cache stored nothing else |
| The freeze test compared against a running maximum | growth slower than the threshold never accumulated: at 1% a sample against a 5% threshold the scale froze after 119 samples with the range 3.3x wider | reproduced on a synthetic ramp |
| Reports carried no provenance | two ablation groups ran under different defaults with nothing in the output to show it | the JSON held no configuration at all |

Every variant now states all fifteen switches and a run refuses to start if two
variants describe the same experiment; correct, wrong and tied pairs are counted
separately; one `select_frames` serves every tool and reports why each frame was
excluded; caches carry a fingerprint of inputs, configuration and code version;
the stability test compares against a reference fixed at the start of the
interval; and every report carries the commit, the working-tree state, the
library versions and the full configuration.

**All inversion figures published before this repair are superseded.** The rank
correlations moved by less than 0.06.

---

## 1. The score did not track focus

The headline finding. Against the point-source reference — which uses no focus
measure — the raw metric tracks focus almost perfectly while the score the
system produced was *negatively* correlated with it. Numbers:
[before and after](RESULTS.md#before-and-after-against-the-spot-reference).

### Why

The normaliser scaled each metric against the 5th and 95th percentiles of a
rolling 120-frame window — under five seconds. That makes the score answer
*"is this frame sharper than the last few seconds"* rather than *"how sharp is
this frame"*. On a slow approach to focus the window fills with high values, so
the sharpest frames of the whole sweep read as no better than their immediate
past, and the score collapses exactly where it should peak.

On `point_source_20260916_111545` the raw metric peaks at step 14 while the
score peaked at step 11 and was at its *minimum* where the image was sharpest.

### What fixed it

**A fixed scale.** A time-varying map cannot preserve the property a focus
search depends on — that within one stream a higher score means a sharper frame.
Freezing does not *guarantee* that property either: the fused score is a
weighted sum of six metrics that need not agree, and the weights move too. It
removes the one cause that was demonstrably breaking it. The normaliser waits
until the observed range stops growing, then freezes (`auto_freeze`); a host
running its own coarse sweep should call `NormalizerBank.freeze()` instead.

**A logistic map instead of linear-with-clipping.** A clipped linear map gives
every frame beyond an anchor the same score, so wherever the anchors do not span
the working range the ordering is destroyed. Measured: a frozen linear map pins
41% of frames at 0 or 1 and leaves a six-step plateau at the peak. The logistic
map is strictly monotone and adds no ties of its own over the range the anchors
describe — it is not tie-free in the abstract, since the float64 implementation
saturates past about nine spans.

The window was also lengthened to 960 frames, which matters for the period
before the scale freezes.

The [normalisation group](RESULTS.md#normalisation-moving-or-frozen-linear-or-logistic)
separates these: the longer window, the freeze and the logistic map each recover
part of it, and only the combination recovers all of it.

### Thawing is off, deliberately

A frozen scale should ideally re-calibrate when the *scene* changes. No
threshold tested could tell a scene change from a focus change: a sweep
legitimately runs past anchors that are percentiles of an earlier window, so the
rule fires mid-sweep and discards the fixed scale exactly when it is doing its
job. `auto_thaw` therefore defaults to `false`. A host that *knows* the scene
changed should call `unfreeze()` — information no statistic in the stream can
recover.

---

## 2. The degenerate-range guard could never be satisfied

```
floor = min_range_ratio * max(|high|, |low|, 1.0)
```

The `1.0` makes it absolute at `1e-3` for any metric whose values sit below 1 —
which is all of them, because contrast normalisation divides by `mean(I)^2`.

| metric | typical anchor magnitude at defocus | floor | span/floor |
| --- | --- | --- | --- |
| `brenner` | 0.00029 | 0.001 | **0.09** |
| `fourier` | 0.00060 | 0.001 | **0.11** |
| `laplacian` | 0.00137 | 0.001 | **0.15** |
| `wavelet` | 0.00064 | 0.001 | **0.22** |

For `brenner` the floor is larger than the largest value the metric ever takes,
so the guard could not be satisfied at any focus position. Four metrics of six
returned the neutral 0.5 sentinel on every defocused frame — 33% of all frames
had at least three pinned. On `cond_20260916_110656`, step 2 scored 0.500 while
step 8, sixteen times sharper by the raw measure, scored 0.001.

`min_range_absolute` is now `1e-6`. The sentinel rate falls from 0.395 to 0.002.

---

## 3. The noise estimator reported zero on every real frame

`noise_sigma` was exactly 0.0000 on 100% of frames in six of the eight
recordings. Consequently `snr` sat at its sentinel, the SNR confidence factor
was pinned at 1, and the noise branch of the reliability model — whose
coefficients are the largest and most argued-for in the design — **never once
activated on real camera data**.

The estimator itself is correct: on continuous data it is unbiased (1.0 to 1.05,
3.0 to 2.96, 10.0 to 9.95). The failure is the input. 58–74% of the Haar HH
coefficients of this camera's live view are **exactly zero** after 8-bit
quantisation, so the median is zero by definition and `median(|HH|)*1.4826`
returns zero whatever the true noise is. This is a breakdown of the median under
heavy quantisation, not a coding error.

`noise_quantile` now defaults to 75, reading the same distribution where it is
not degenerate, with the matching Gaussian scale factor. The zero-noise rate
falls from 0.976 to 0.002.

**What this does not fix.** Every high quantile carries a floor of 0.3–0.44 set
by image structure rather than noise, and the reported value runs about 35% low
on the quantised path. It is a usable *relative* noise signal on this camera,
not an absolute measurement, and its scene-dependent floor means the noise level
partly tracks texture. A temporal estimator was tried and was no better, being
quantised the same way.

---

## 4. Edge sufficiency was unreachable, and is circular

`edge_ref_density` was 0.03. No recording reaches it at defocus and four of
eight never reach it at all. Edge sufficiency is a *necessary* confidence
factor, subject to the weakest-link cap, so a zero there zeroes the confidence:
it was exactly zero on 28–78% of frames.

Lowering the reference to 0.006 did not move it, and cannot: the measured edge
density is *exactly* zero on four recordings, because the gradient magnitude of
a defocused frame on this stream never exceeds about 3 intensity units per pixel
and Canny marks nothing at any sane threshold.

This is not a threshold to be tuned. **Edge density is itself a sharpness
measure**, so "not enough edges" and "out of focus" are the same observation,
and using it as a precondition for trusting a focus score is circular. A sound
version would ask whether the *scene* has structure — the highest edge density
seen recently, rather than this frame's. That is a design change, and it is
listed as open.

---

## 5. `ready` did not mean ready

```python
ready = warmed_up and informative > 0.0
```

One informative metric out of six was enough. On frames where three or more
metrics sat at the sentinel, `ready` was still true 75–100% of the time while
`informative_fraction` correctly reported 0.22–0.40. `ready` now requires a
majority; the ready rate falls from 0.949 to 0.605, and the frames removed are
the ones that were never usable.

---

## 6. `Preprocessor.prepare` hands back a scratch buffer

Not a calibration issue, but found the same way and equally silent: the returned
`gray` array is overwritten by the next call, so a caller that keeps the
reference and later compares two "different" frames gets a difference of exactly
zero and no error. It cost one wrong measurement during this study. Documented
at the call site and pinned by tests.

---

## What the repairs are worth

[RESULTS.md](RESULTS.md#each-repair-on-its-own) has the table. Three things are
worth reading out of it.

**No single repair is sufficient, and two are useless alone.** The logistic map
on its own is the worst row: a monotone map cannot help while the scale
underneath it keeps moving. The range guard alone removes the sentinel entirely
and still leaves the correlation negative.

**The repaired pipeline is also the fastest.** Freezing the scale stops the
percentile computation, which more than pays for the longer history before the
freeze.

**Rank correlation with the reference goes from clearly negative to 0.949, and
wrongly ordered step pairs from 0.53 to 0.05.** That is the number that matters
to a search: how often the model would send it the wrong way.

Note what `all` means here: **the repairs only**, with the weighting model left
exactly as it was, agreement kernel included. It is deliberately not the shipped
configuration, so that the normalisation question and the adaptivity question do
not credit each other. Turning the kernel off as well - which is what ships -
adds a further 0.016, and that belongs to the factorial below, not to the
repairs.

---

## Does the adaptivity earn its keep?

The [factorial](RESULTS.md#adaptivity-all-combinations-of-the-three-mechanisms)
runs all eight combinations of reliability model, agreement kernel and temporal
filter on the repaired pipeline, so interactions are visible rather than
inferred from three one-at-a-time rows.

**The reliability model earns its place.** Removing it is the worst quadrant on
every criterion, by a margin larger than any other effect in the table.

**The agreement kernel does not.** Switching it off is better on every
criterion, and widening the kernel only walks the result back towards "off"
(0.949 at the shipped scale of 1.5, 0.952 at 1.0, 0.958 at 4.0, 0.960 at 8.0,
0.965 off - see [the scale sweep](RESULTS.md#consensus-kernel-width)), so this
is not a tuning failure. It is now off by default. What that does *not* establish: the kernel
exists to protect the score when one metric is fooled by a specular highlight or
a blown region, and none of these recordings contains that failure.

**There is an interaction, and it changes how the reliability model should be
described.** Its apparent benefit is much larger with the agreement kernel on
than with it off — a large part of what it was doing was repairing damage the
kernel caused. With the kernel off, its remaining contribution is small.

**The temporal filter is worth more than any weighting choice** for
adjacent-step discrimination, at no cost in agreement with the reference.

**Against a plain mean, the advantage is consistent but not established.** The
shipped configuration is ahead on both point-source recordings and under both
reference recipes, by 0.008 to 0.018 in rank correlation — four measurements,
all the same sign. But only two recordings carry a reference at all, and the
difference is smaller than the spread between recordings. This is a consistent
direction, not a demonstrated effect, and it is reported that way.

---

## The pipeline is worse than its own best input

The [fair comparison](RESULTS.md#signals-and-pipelines-compared-separately)
scores raw measures — no normalisation, no history, no filter — directly against
the reference, and separately scores whole streaming pipelines that all share
the same normaliser, history and filter.

The raw `wavelet` measure reaches 0.988. The best full pipeline reaches 0.951.
**The normalisation, fusion and filtering together cost about 0.04 of agreement
with the reference relative to simply using the best raw measure.**

That is not an argument for shipping a raw measure. A raw measure has no bounded
scale, no cross-scene comparability, no confidence and nothing to fuse, and this
comparison is offline on a recording whose whole range is known in advance. But
it bounds what the apparatus is buying, and the honest framing is that the
pipeline trades accuracy for the properties a live system needs rather than
improving on its inputs.

Also worth recording: three published measures — NVAR, GLVA and VOL5 — score
about 0.10 against this reference. They are variance-family measures and respond
to contrast rather than to concentration, which is the wrong thing on a point
source. A ranking of measures depends on the target.

---

## How far the reference itself can be trusted

Every accuracy claim rests on one number, and
[it has free parameters](RESULTS.md#how-much-the-reference-itself-depends-on-its-parameters).

The variants split into two groups. Radii taken in a 50 px or 80 px window at
the 25th or 50th percentile of enclosed signal agree with each other at rank
correlation **0.98 to 0.997**. Every variant using a 120 px window is
**anti-correlated** with them, at -0.74 to -0.89: that window is large enough to
include the edge of the screen the source sits on, whose sharpness also changes,
so the measurement stops being about the source. Its radius never falls below
20 px, which is not a plausible in-focus spot on a 640x360 frame.

Consequences, stated wherever the reference is used:

- The best step spans 12–14 on one run and 9–11 on the other across the coherent
  variants: **the reference locates best focus to about one step, not exactly.**
  Reported `peak_err` values of 1.0–1.5 are inside that, so they cannot separate
  models.
- Rank correlations are more robust: the factorial ranking is unchanged under
  the alternative recipe, with every variant shifting down by 0.004–0.011.
- Excluding frames whose core is clipped (11.8% and 12.4%) moves the best step
  by at most one.
- A different background estimator moves it by one step and costs 0.07 of
  self-agreement.

---

## What did *not* change, and why

**The metric set stays at six.** The offline comparison in
[COMPARATIVE_STUDY.md](COMPARATIVE_STUDY.md) suggested dropping `fourier` and
`edge_width`. Replaying through the real online evaluator says otherwise, and
the [leave-one-out table](RESULTS.md#metric-set-and-analysis-resolution) says
something more uncomfortable: **no single metric is essential, removing
`tenengrad` slightly improves agreement with the reference, and removing
`fourier` or `edge_width` costs the most adjacent-step discrimination.** The
differences are within the between-recording spread, so the set is defensible
but not optimal, and no reduction is justified by this data.

**A seventh metric is available but off.** `gradient_variance` (TENV, Pertuz et
al. 2013) separated adjacent focus steps better than any of the project's own
six in the offline study and costs 0.31 ms. Online it is slightly worse on every
criterion. It ships built, with a prior and a sensitivity row, and stays out of
`metrics.enabled`.

**320 px stays the analysis width.** 240 is worse; 480 and 640 cost two and
three times as much and agree with the reference *less*. Higher resolution also
returns the noise estimator to reporting zero on 34% and 69% of frames, because
the 2x2 averaging in the downscale is what lifts the Haar band off the
quantisation floor.

**VOL4 is not available in the library.** It ranks among the best raw signals,
but it is negative on 40–85% of frames here and the metric contract clamps at
zero, which would flatten it across most of a sweep. Supporting it needs a
contract change, not a registration. The fair comparison keeps its negative
values, which is why it appears there and not in the metric set.

---

## What is still open

- **Edge sufficiency is circular.** The candidate fix is to measure scene
  structure over a history rather than per frame, judged by a coverage-versus-
  error curve rather than by having fewer zeros.
- **`auto_freeze` has four constants**, fitted on the same recordings that
  evaluate them.
- **The noise estimate has a structure-dependent floor** and reads about 35%
  low. It is a relative signal, not a measurement.
- **The ground-truth comparison rests on two recordings.** Any difference
  smaller than about 0.02 in rank correlation is a direction, not a result.
- **No closed-loop test.** Everything here measures the signal a search would
  consume, never a search. Nothing measures autofocus latency, because the lens
  position was never recorded.
- **The protocol step is a manual step number**, not a calibrated lens position,
  and steps are not equally spaced in defocus.

---

## Threats to validity

- **One camera, one room, one operator.** The quantisation finding in
  particular is a property of this stream.
- **Two point-source runs**, and they are the only independent reference in the
  project.
- **The spot saturates near focus** (peak 251–253 of 255), so some signal is
  missing from the core exactly where the radius is smallest. Excluding clipped
  frames moves the best step by at most one, which bounds but does not remove
  the concern.
- **`adj` and `mono` are auxiliary.** `adj` measures whether neighbouring steps
  can be told apart, not whether the direction is right; `mono` is computed
  against the profile's own maximum, which can be in the wrong place. Where a
  reference exists, rank correlation and pair ordering are the primary numbers.
- **Degradations applied after capture are not the same as capturing under
  those conditions.** Nothing here changes ISO or physical lighting.
