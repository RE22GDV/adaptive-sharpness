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
python3 tools/protocol_study.py data --refresh --json reports/protocol_study.json
for g in fixes normalisation factorial metrics agreement; do
  python3 tools/ablation.py data --group $g --json reports/ablation_$g.json
done
python3 tools/ablation.py data --group factorial --reference r25_w50_median     --json reports/ablation_factorial_altref.json
python3 tools/before_after.py data           --json reports/before_after.json
python3 tools/fair_comparison.py data        --json reports/fair_comparison.json
python3 tools/reference_sensitivity.py data  --json reports/reference_sensitivity.json
python3 tools/render_results.py reports --out docs/RESULTS.md --figures docs/figures
```

The reports in `reports/` are committed. Each carries the commit it ran from,
whether the source differed from it, the library versions, the full
configuration and its fingerprint. `docs/RESULTS.md` and every figure are
generated from those files and from nothing else.

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

**Against a plain mean, the advantage depends on which criterion is used, and
the two criteria disagree.** This is the least comfortable result in the
document and it is the one most easily misread, because the two criteria are
also measured on different numbers of recordings.

| comparison | rank correlation, n=2 | `adj`, n=8 |
| --- | --- | --- |
| adaptive - prior weights | **+0.0145** | **-0.0050** |
| adaptive - equal weights | **+0.0155** | **-0.0036** |

By rank correlation against the reference, the shipped configuration is ahead
on both point-source recordings and under both reference recipes - four
measurements, all the same sign, and the paired difference survives a change of
reference recipe with 0.003 to 0.005 of movement. By adjacent-step
discrimination across all eight recordings, it is behind.

The per-recording breakdown says why, and it is not a wash:

| recording | adaptive | prior | difference |
| --- | --- | --- | --- |
| `cond_20260916_110502` | 0.704 | 0.827 | **-0.123** |
| `cond_20260916_110746` | 0.772 | 0.829 | **-0.056** |
| `sweep_plain_20260916_110955` | 0.735 | 0.738 | -0.003 |
| `sweep_texture_20260916_110108` | 0.797 | 0.790 | +0.007 |
| `point_source_20260916_111545` | 0.934 | 0.923 | +0.011 |
| `point_source_20260916_111724` | 0.828 | 0.801 | +0.027 |
| `cond_20260916_110600` | 0.856 | 0.814 | +0.042 |
| `cond_20260916_110656` | 0.948 | 0.894 | +0.054 |

**Adaptive weights win on five recordings of eight and still lose on average**,
because the two losses are several times larger than any of the wins. The mean
is carried by two recordings.

Those two are the darkest in the corpus - mean brightness 0.2, measured edge
density 0.0000 against an `edge_ref_density` of 0.006. Both the edge and the
exposure degradation factors are pinned at their extremes there, so the
reliability model is producing weights from inputs that have saturated and
stopped carrying information. `cond_20260916_110656`, at brightness 0.8, is the
largest win.

That association is two recordings and a plausible mechanism, not a
demonstrated cause - the two point-source recordings also have near-zero edge
density and adaptivity wins on both. It is recorded here because it is a
testable prediction and because it points at the same defect as
[edge sufficiency](#what-is-still-open) below: a per-frame edge count that
bottoms out is not a usable condition signal.

**What this means for the claim.** "Fusing six measures beats one" and
"adapting the weights beats fixing them" are separate claims with separate
evidence. The first holds on all eight recordings and on both criteria. The
second does not hold.

> **Settled by the research programme.** Two studies finished this argument;
> full write-ups in [STUDY_UA_RESULTS.md](STUDY_UA_RESULTS.md).
>
> **The difference is not resolvable.** A paired moving-block bootstrap inside
> each recording, with the block length taken from the autocorrelation of the
> paired difference (decorrelation lag 3-74 frames), gives a 95% interval of
> **`[-0.046, +0.029]`** on a point estimate of -0.005 - about nine times the
> estimate. Adaptive weights are indistinguishable from fixed ones here, in
> either direction. The per-recording table above is a description, not a
> result.
>
> Resampling frames as if they were independent would have given 0.042 instead
> of 0.135 and produced an effect that is not there. That is a caution for any
> future comparison on this corpus.
>
> **The weights were read out, and they barely move.** Over an entire recording
> a metric's weight travels 0.01 to 0.05, on means between 0.04 and 0.24. There
> is almost nothing for a bootstrap to have found. The saturation mechanism
> proposed above survives only partly: the two worst recordings do have the
> highest edge-density saturation, 79% and 76% of frames, but across all eight
> the association is -0.41 and a recording at 66% is the biggest *win*.

---

## What the pipeline costs, and whether that is the price of real time

The [fair comparison](RESULTS.md#signals-and-pipelines-compared-separately)
scores raw measures - no normalisation, no history, no filter - directly against
the reference, and separately scores whole streaming pipelines that all share
the same normaliser, history and filter.

An earlier version of this section claimed the pipeline costs about 0.04 and
called that the price of running in real time. **Both were wrong**, because the
table compared a pipeline against a *different* measure: neither the shipped
configuration nor a pipeline for `wavelet` was in it.

Like for like:

| | rank correlation | ms |
| --- | --- | --- |
| raw `wavelet` | **0.988** | - |
| its own pipeline, `single:wavelet` | 0.937 | 2.39 |
| `six:shipped` | **0.965** | 6.85 |
| `six:plain_mean` | 0.950 | 6.80 |

Two steps, each measured on its own:

- **The streaming apparatus costs 0.051** on a single measure (0.988 to 0.937):
  that is the normalisation plus the temporal filter.
- **Fusing six recovers 0.028** (0.937 to 0.965), for 4.5 ms.

**The net gap to the best raw measure is 0.023**, and it is a real gap.

An earlier version of this section explained it away: the reference's coherent
variants agree with each other at rank correlation 0.98, so 0.023 was called
"below what this reference can resolve". That inference does not hold. A
correlation *between two recipes* is not an error bar on a correlation
*between a method and a recipe*, and the two need not have anything like the
same magnitude.

The question can be answered instead of argued, because the factorial was run
against both recipes ([table](RESULTS.md#does-the-ranking-survive-a-different-reference-recipe)).
Changing `r50_w80_median` to `r25_w50_median` moves every variant's rank
correlation down by 0.004 to 0.011 - a common-mode shift - while the
*differences between variants* move by only 0.002 to 0.005:

| paired difference | `r50_w80` | `r25_w50` | moved by |
| --- | --- | --- | --- |
| adaptive - prior weights | +0.0145 | +0.0114 | -0.0031 |
| adaptive - equal weights | +0.0155 | +0.0104 | -0.0050 |
| kernel off - kernel on | +0.0159 | +0.0136 | -0.0022 |

So paired differences are far more robust to the recipe than absolute
correlations are, and 0.023 is roughly five times the largest recipe-induced
movement. It is not noise in the reference. The honest reading is that the
pipeline does cost about that much against the best raw measure on these two
recordings, and the case for it has to be made on the other eight.

> **Localised precisely.** A stage-by-stage ladder on the same frames
> ([Д09](STUDY_UA_RESULTS.md#д09-нормалізація-поетапно)) shows the split is not
> where this section assumed. Offline scaling is free - the log is exactly
> neutral and the logistic map removes the linear map's 10% clipping at no
> cost. The whole loss is **being online**: a single measure through the
> streaming pipeline drops adjacent-step discrimination from 0.890 to 0.697.
> Fusion returns +0.077 of that and the temporal filter +0.048, and fusion also
> removes the 8.9% saturation that streaming a single measure introduces.
>
> So "the pipeline costs 0.023" is the residue after that repair, not its
> price. The gross cost of running online is 0.193, and the two mechanisms this
> section is about exist to recover it - which they do, by about two thirds.

So most of the loss comes not from running in real time but from using *one*
measure; fusion recovers more than half of it; and what is left is not
measurable here. "The price of real time" is not supported by this data.

One difference that only became visible once single measures got pipelines of
their own: a single measure through the pipeline **ties** protocol steps
(`resolved` 0.814-0.952) while the six-measure pipelines do not (1.000). Fusion
removes ties.

### Is one measure enough?

The comparison above runs only where the point-source reference exists, which
is two recordings of a bright point on black - the easiest possible target for
a high-frequency measure, and the one place where the robustness fusion is for
cannot be tested.

Across all eight recordings, low texture and dark exposures included
([table](RESULTS.md#one-measure-against-the-whole-set-on-every-scene-type)):

| configuration | `adj` | peak plateau | saturation |
| --- | --- | --- | --- |
| **six, as shipped** | **0.822** | **1.00 step** | 0.005 |
| `brenner` alone | 0.747 | 2.00 | 0.090 |
| `wavelet` alone | 0.743 | 2.12 | 0.090 |
| `tenengrad` alone | 0.729 | 2.00 | 0.089 |
| `laplacian` alone | 0.720 | 2.50 | 0.165 |
| `edge_width` alone | 0.705 | 1.75 | 0.179 |
| `fourier` alone | 0.690 | 3.62 | 0.250 |

Every column here is averaged over the same eight recordings.

The gap is **0.075 to 0.132** here against 0.025 on the point source alone. Six
wins on **six recordings of eight**, and by the widest margin exactly where
fusion is supposed to help - on the low-texture sweep, 0.735 against 0.640 for
the best single measure.

A single measure also leaves a wider **peak plateau** - 1.75 to 3.62 steps
against one - and saturates **19 to 54 times** more often.

> An earlier version of this table printed 5 to 7 steps in the plateau column
> and concluded that a single measure "does not localise the maximum at all".
> Those were the `peak_plateau` figures from the *two* point-source recordings,
> lifted into a table labelled as covering eight, and they were not even the
> range of that column - `edge_width` sits at 1 step there. The all-recordings
> statistic is `peak_plateau_steps`, printed above. The corrected difference
> still favours six measures and is about a third of the size.

So a single measure nearly matches on a point source and does not on real
scenes.

### A raw measure is not an offline category

`wavelet` is computed from one frame and needs no future ones. Its absence from
the shipped system is not a matter of it being uncomputable online.

What it lacks is a bounded scale, a confidence, and anything to fuse. For
finding a maximum *within one stream* no scale is needed - only an ordering,
and the raw measure gives a better one than the pipeline does. Whether those
properties are worth 0.023 is a choice, not a consequence.

Cross-scene comparability is **not** on that list, although an earlier version
of this section claimed it. The scale is fitted to one stream's own history and
then frozen to it, so two streams are normalised against two different
histories; [README](../README.md#what-it-does-not-do) says so directly. The
pipeline buys a bounded number and a confidence to go with it, not a unit.

Also worth recording: three published measures - NVAR, GLVA and VOL5 - score
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

Fourteen studies ([STUDY_UA_RESULTS.md](STUDY_UA_RESULTS.md)) closed three of
these and sharpened the rest.

- **Edge sufficiency is circular.** The candidate fix is to measure scene
  structure over a history rather than per frame, judged by a coverage-versus-
  error curve rather than by having fewer zeros. Now with a number attached:
  edge density sits at its minimum on **24-79%** of frames depending on the
  recording, so for much of the corpus the input is not a measurement.
- ~~**`auto_freeze` has four constants**, fitted on the same recordings that
  evaluate them.~~ **Closed, and the conclusion is the opposite of the worry.**
  The full grid of 27 threshold combinations spans 0.591 to 0.971 in rank
  correlation, but every combination that fires in time on every recording
  lands near 0.96, and *not freezing at all* gives 0.50. The mechanism carries
  the effect; the constants are almost free parameters in the harmless sense.
  **This is the largest single effect measured anywhere in this project** -
  larger than fusion, weighting and filtering together.
- **The noise estimate has a structure-dependent floor** and reads about 35%
  low. Now measured against a known value rather than inferred: injecting
  Gaussian noise at sigma 2, 5, 10 and 20 is reported as 0.90, 1.79, 3.51 and
  6.50, so roughly a third of truth and **monotone**. It is a relative signal,
  not a measurement.
- ~~**Any difference smaller than about 0.02 in rank correlation is a
  direction, not a result.**~~ **The threshold was too optimistic**: a bootstrap
  respecting the frame autocorrelation puts it near **0.04**, and the naive
  frame-level resampling that would have suggested 0.02 is exactly the mistake
  this corpus invites.
- **No closed-loop test.** Everything here measures the signal a search would
  consume, never a search. A search *simulated over the recorded profiles*
  lands within one step of the reference on 100% of runs using 8.5 evaluations,
  which says the signal has a climbable shape and says nothing about latency:
  there is no actuator, no backlash and no settling time in a replay.
- **The protocol step is a manual step number**, not a calibrated lens position,
  and steps are not equally spaced in defocus.
- **New: the score is blind to uniform defocus.** Blurring every frame does not
  lower the mean score. The normaliser refits, so the least-blurred of a blurred
  set still reads near the top.
- **New: warm-up dominates.** The same frames scored after a different history
  differ by up to 0.64, with rank agreement falling to 0.56.
- **New: the confidence has no reliable sign** as a per-frame quality
  indicator - useful on four recordings of eight, harmful on the other four.

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
