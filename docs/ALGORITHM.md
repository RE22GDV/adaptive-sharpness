# The algorithm

## 1. Scope: what is established, what is proposed

This section exists so the rest of the document can be read without guessing
which parts are original.

**Established, and used unchanged.** All six focus metrics are classical and
long-published. None of them is a contribution of this project:

| Metric | Origin |
| --- | --- |
| Brenner gradient | Brenner et al. (1976) |
| Tenengrad | Krotkov (1987) |
| Variance of the Laplacian | Pech-Pacheco et al. (2000) |
| Haar wavelet detail energy | Kautsky et al. (2002); Yang & Nelson (2003) |
| Spectral high-frequency ratio | standard; surveyed in Pertuz et al. (2013) |
| Edge width / slope | Marziliano et al. (2002); Ferzli & Karam (2009) |

The idea of *combining* focus measures is also not new, and neither is the idea
that different measures suit different conditions - that is the central finding
of the Pertuz, Puig & Garcia (2013) survey of 36 focus measures.

**Proposed here.** The specific combination scheme:

1. a log-linear reliability model that maps measured image degradations onto
   per-metric weights through documented sensitivity coefficients (§4);
2. a consensus-agreement reweighting stage layered on top of it (§5);
3. a confidence value derived from necessary-condition factors, reported
   separately from the score and never modifying it (§6);
4. an innovation-gated temporal filter that smooths jitter without delaying a
   genuine focus transition (§7).

**Not yet established, and stated as such.** Whether this combination beats the
published alternatives. §9 records what was measured, including the cases where
it does not win, and §10 lists what a proper claim of novelty would still
require. **No claim of scientific originality is made here** - only that the
scheme is specified precisely enough to be tested, and that it was tested.

---

## 2. Notation

For frame `t`:

- `I` - the analysis image: single channel, `float32`, values in `[0, 255]`,
  downscaled to `analysis_width` (default 320 px), cropped to the ROI first.
- `x_i` - the raw value of metric `i`, larger meaning sharper.
- `s_i in [0, 1]` - the normalised value of metric `i`.
- `d_j in [0, 1]` - degradation factor `j`, where 0 is ideal conditions.
- `w_i` - the weight of metric `i`, with `sum_i w_i = 1`.
- `S in [0, 1]` - the ensemble score. `S_hat` is its filtered version.
- `C in [0, 1]` - the confidence.

---

## 3. The metrics

### 3.1 Illumination-scale invariance

The textbook energy metrics are not invariant to exposure: doubling the
illumination quadruples a squared-gradient sum with no change of focus. Since
robustness to exposure change is an explicit requirement, the energy-type
metrics are divided by `mean(I)^2`:

```
x_i = (textbook value) / max(mean(I)^2, 1)
```

This makes them invariant to a multiplicative illumination change while leaving
their focus response untouched. `fourier` and `edge_width` are ratios and are
already invariant, so the correction does not apply to them. The behaviour is
controlled by `metrics.contrast_normalize`, and setting it to `false` restores
the textbook definitions.

Note the limit: an exposure change severe enough to **clip** destroys real
detail, so the score genuinely falls. That is correct - what the system must do
in that case is lower its confidence, which it does through the exposure factor.

### 3.2 Laplacian variance

```
x = Var( L * I ) / mean(I)^2,     L = 3x3 Laplacian kernel
```

A second derivative, so its frequency response rises fastest with focus - and
also amplifies white noise more than any first-derivative operator, which is
why it carries the largest noise sensitivity in §4.

### 3.3 Tenengrad

```
x = mean( Gx^2 + Gy^2 ) / mean(I)^2
```

with `Gx`, `Gy` the 3x3 Sobel derivatives. The Sobel kernel has a smoothing
component perpendicular to the derivative, which suppresses part of the sensor
noise.

### 3.4 Brenner

```
x = 0.5 * ( mean( (I(x+k,y) - I(x,y))^2 ) + mean( (I(x,y+k) - I(x,y))^2 ) ) / mean(I)^2
```

with `k = brenner_shift` (default 2). The textbook operator is horizontal only;
both directions are averaged here so that the metric does not favour one edge
orientation, because the ensemble measures gradient anisotropy separately and
would otherwise double-count it.

### 3.5 Haar wavelet detail energy

One level of the orthonormal 2-D Haar transform splits `I` into `LL, LH, HL,
HH`. The metric accumulates the detail energy over `wavelet_levels` levels:

```
x = sum over levels ( mean(LH^2) + mean(HL^2) + mean(HH^2) ) / 4^(level-1) / mean(I)^2
```

The `4^(level-1)` divisor is **necessary, not cosmetic**. The orthonormal Haar
`LL` band has a DC gain of 2 per level, so its energy grows by 4 per level.
Without dividing that out, deeper levels are inflated and the metric stops being
monotone in the defocus: the level-1 detail collapses under blur while the
4x-amplified level-2 detail does not, and the sum can rise with defocus. This
was observed before it was corrected.

The transform is implemented directly with strided slicing rather than via
PyWavelets - it is a handful of array operations, and it removes a dependency
that is not packaged for this platform.

### 3.6 Spectral high-frequency ratio

```
x = ( sum over |f| >= f_c of |F(f)|^2 ) / ( sum over all f of |F(f)|^2 )
```

where `F` is the 2-D DFT of `(I - mean(I))` multiplied by a 2-D Hann window, and
`f_c = fourier_high_cutoff` as a fraction of Nyquist (default 0.25). The DC term
is excluded from the numerator.

The Hann window is essential. Without it, the implicit discontinuity at the
image border injects broadband energy across the whole spectrum, which swamps
the focus signal and leaves the metric nearly focus-independent.

Because it is a ratio of energies, this metric is inherently contrast-invariant.

### 3.7 Edge width

For a step edge of amplitude `A` convolved with a Gaussian of width `sigma`, the
peak gradient is `A / (sigma * sqrt(2*pi))`. So the local intensity range
divided by the local gradient magnitude estimates the edge width in pixels:

```
width(p) = range_W(p) / |grad I(p)|
x = 1 / median over edge pixels of width(p)
```

`range_W` is the morphological gradient (dilate minus erode) over a `W x W`
window, `W = edge_range_window` (default 11). The Sobel output is divided by 8
to become a true intensity-per-pixel derivative.

Two implementation facts were established by measurement, not assumption:

- **The window must span the whole transition.** With a 3x3 window, the measured
  range shrinks in step with the gradient as the edge blurs, so their ratio is
  constant and the metric is completely insensitive to focus. Measured across a
  defocus ramp it varied only between 3.846 and 3.877 - effectively a constant.
  `W = 11` bounds the largest measurable width at roughly 9 px.
- **The Canny thresholds must adapt to the frame.** With fixed thresholds, fewer
  edges clear the threshold as defocus grows, and the survivors are the
  highest-contrast (apparently sharpest) ones, so the median width can *fall*
  while the image blurs. Measured on a defocus ramp at four different window
  sizes, fixed thresholds were non-monotone at **every** size, while thresholds
  anchored at the 97th percentile of the frame's own gradient magnitude were
  monotone at every size and gave a larger dynamic range (2.78x vs 2.48x).

---

## 4. Stage 1: content-conditioned reliability

Each metric `i` carries a vector of non-negative sensitivity coefficients
`kappa_ij` describing how quickly it degrades under condition `j`:

```
r_i = exp( - sum_j kappa_ij * d_j )
```

This is a log-linear model. It has the properties the weighting needs:
independent degradations multiply, every `r_i` stays in `(0, 1]`, no condition
can drive a weight negative, and each coefficient is interpretable on its own.

The five degradation factors, all measured per frame and all in `[0, 1]`:

| `d_j` | Measured as |
| --- | --- |
| `noise` | `sigma / noise_ref_sigma`, `sigma` from the finest Haar detail band: `sigma = Q_p(|HH|) / F(p)` with `p = analysis.noise_quantile` (default **75**) and `F(p)` the half-normal quantile `Phi^-1((1+p/100)/2)`. This is the Donoho & Johnstone (1994) estimator evaluated away from the median: on an 8-bit stream most `HH` coefficients are exactly zero, which pins the median at zero and reports no noise however much there is. At `p = 50` it reduces to the textbook `median(|HH|) / 0.6745`. |
| `edge` | `1 - min(1, edge_density / edge_ref_density)`, edge density from Canny |
| `clip` | clipped-pixel fraction against `clip_ref_fraction`, plus a penalty for a mean intensity outside `[brightness_low, brightness_high]` |
| `motion` | `|shift| / motion_ref_px`, shift from `cv2.phaseCorrelate` on a decimated copy |
| `contrast` | `1 - min(1, RMS_contrast / contrast_ref)` |

Default coefficients (`ensemble.sensitivity` in the config):

| metric | noise | edge | clip | motion | contrast |
| --- | --- | --- | --- | --- | --- |
| `laplacian` | **2.2** | 0.6 | 0.5 | 0.6 | 0.7 |
| `tenengrad` | 1.2 | 0.7 | 0.6 | 0.7 | 0.8 |
| `brenner` | 1.0 | 0.9 | 0.6 | 0.9 | 0.9 |
| `wavelet` | 1.6 | 0.5 | 0.5 | 0.6 | 0.6 |
| `fourier` | 1.9 | 0.4 | 0.7 | 0.5 | 0.6 |
| `edge_width` | 1.4 | **2.2** | 0.9 | 1.0 | 1.0 |

The two extreme values encode the two clearest facts: a second-derivative
operator amplifies white noise more than any first-derivative one, and the
edge-width metric is undefined when there are no edges to measure.

**These coefficients are reasoned starting points, not measured constants.**
Deriving them from data is listed in §10 as outstanding work.

### 4.1 On measuring conditions independently of focus

The degradation factors must not respond to focus, or the weighting would react
to the very thing it is weighting. This ruled out the obvious motion cue: the
temporal intensity difference rises when focus changes even though nothing has
moved. Global translation from phase correlation has the required property and
is used instead.

Two bugs found while validating this, both by measurement:

- `cv2.phaseCorrelate` carries a constant sub-pixel offset that depends on the
  transform size, not the content - correlating an array with itself returns
  `(0.0, 0.5)` at 44x80 but `(0.0, 0.0)` at 46x82. It is measured once per shape
  and subtracted.
- `cv2.phaseCorrelate` **modifies its input arrays** in place. Storing the same
  array as the previous-frame reference silently corrupted it and produced
  drifting phantom motion on a static scene. The reference is now a copy.

Decimating with strided slicing (`gray[::4, ::4]`) also aliases, and the aliasing
pattern changes with blur, which produced apparent motion when only the focus
moved. Area-averaged resize is used instead.

---

## 5. Stage 2: consensus agreement (off by default)

> **This stage is disabled in the shipped configuration** (`use_agreement =
> false`). It is documented because the code is there and can be switched on,
> not because it is running. Measured against the point-source reference,
> switching it off is better on every criterion, and widening the kernel only
> walks the result back towards "off". See
> [CALIBRATION](CALIBRATION.md#does-the-adaptivity-earn-its-keep). With the
> stage off, `a_i = 1` for every metric and the weights are `p_i * r_i` alone.
> Metric disagreement still feeds the confidence value; that path is separate
> and stays on.

Metrics that disagree with the consensus are down-weighted with a Gaussian
kernel centred on the reliability-weighted median `s_med`:

```
a_i = exp( -0.5 * ( (s_i - s_med) / (c * sigma_r) )^2 )
```

`sigma_r = 1.4826 * median(|s_i - s_med|)` is a robust spread, floored at 0.02
so that unanimous agreement does not divide by zero, and `c =
agreement_scale`. This is one step of iteratively reweighted robust estimation;
it protects the score when a single metric is fooled by, say, a specular
highlight.

Set `ensemble.use_agreement = true` to enable it; the default is `false`.

> **The stage that carries the result is not any of these three.** Freezing the
> normalisation scale is worth +0.46 in rank correlation with the physical
> reference - 0.50 without, 0.97 with - while the reliability model is not
> distinguishable from fixed weights and the agreement kernel is negative. See
> [STUDY_UA_RESULTS.md](STUDY_UA_RESULTS.md).

---

## 6. Stage 3: weights and the score

```
w_i = normalise( max( p_i * r_i * a_i , floor ) ),    S = sum_i w_i * s_i
```

`p_i` are the static prior weights, and `floor = weight_floor` (default 0.02)
keeps every metric marginally alive so the ensemble can recover when conditions
improve. The weights sum to exactly 1 by construction, which is asserted in the
test suite across every degradation combination.

### 6.1 Normalisation

Raw metric values are incommensurable, so each is mapped to `[0, 1]` against
robust percentile anchors `q_low`, `q_high` of a history (`window`, default 960
frames = 38 s at 25 fps):

```
mid = (q_low + q_high) / 2          span = q_high - q_low
s_i = 1 / ( 1 + exp( -gain * ( log1p(x_i) - mid ) / span ) )
```

`log1p` is a monotone transform, so it cannot change which frame is judged
sharpest; it makes the heavy-tailed energy metrics roughly symmetric, and
percentile anchors on a symmetric distribution are far more stable.

**The map is logistic, not clipped-linear, and the anchors are frozen once the
observed range settles.** Both of those are corrections, and both were forced by
measurement rather than preference — see [CALIBRATION.md](CALIBRATION.md) §1.

A clipped linear map gives every frame beyond an anchor exactly the same score,
so wherever the anchors do not span the working range the ordering a focus
search depends on is destroyed outright; a frozen linear map pinned 41% of
frames at 0 or 1 on the point-source recordings, against 0.3% for the logistic
map. And a scale that keeps moving rescores the same frame differently
depending on what preceded it: with a 120-frame window the score reached rank
correlation **−0.24 and −0.38** against the physical size of a defocused point,
while the raw metric reached **0.97 and 0.98**. Freezing plus the logistic map
recovers **0.94 and 0.97**.

`auto_freeze` performs the freeze once the range stops growing.
`NormalizerBank.freeze()` and `fit()` do it explicitly, which is what a host
running its own coarse sweep should use; an explicit freeze is never overridden
by the automatic rule. Thawing (`auto_thaw`) is **off**: no threshold tested
could distinguish a scene change from a focus change, and the rule fired
mid-sweep. A host that knows the scene changed should call `unfreeze()`.

### 6.2 Confidence

The confidence does **not** enter the ensemble sum, so it never changes
``instantaneous_score``. It *does* reach the temporal filter when
``confidence_coupling`` is enabled, scaling its baseline gain, and therefore
affects ``filtered_score``. It is
built from six factors in `[0, 1]`, each a necessary condition: edge
sufficiency, SNR, exposure, contrast, inter-metric concordance, and motion. A
warm-up factor is added while the normalisers lack a scale.

```
C = min( geometric_mean(factors), sqrt(min(factors)) )
```

The geometric mean alone is not enough: with six factors and a `1e-3` floor it
bottoms out at 0.32 even when one factor is exactly zero. But with no edges in
the frame there is nothing whose sharpness could be measured, however good the
exposure and SNR are. The weakest-link cap expresses that; the square root stops
a merely mediocre factor from dominating.

---

## 7. Temporal filtering without hiding focus changes

A plain EMA is the wrong tool: the smoothing that suppresses jitter also delays
and attenuates the one event the loop is looking for. The gain is therefore made
to depend on how surprising the sample is:

```
z          = |S_t - S_hat_{t-1}| / sigma_r
g          = smoothstep(z; gate_sigma, gate_full_sigma)          in [0, 1]
alpha_0    = alpha_base * confidence_gain
alpha_t    = alpha_0 + (1 - alpha_0) * g
S_hat_t    = S_hat_{t-1} + alpha_t * (S_t - S_hat_{t-1})
```

`sigma_r` is a robust MAD estimate of the recent frame-to-frame variation, so
the gate measures surprise in units of the score's own noise rather than in
absolute units that would need per-scene tuning. When `g` reaches 1 the filter
is fully transparent and a real transition passes with no lag; when the score is
merely jittering, `g` stays near 0 and the full smoothing applies. An absolute
threshold `gate_abs_jump` covers a very quiet history, where `sigma_r` would sit
at its floor.

Confidence scales only `alpha_0`, never the gated term, so an unreliable frame
is smoothed harder while a genuine large jump is still never suppressed.

Measured on a synthetic focus step (§9): the gated filter follows the step in
**1 frame**, a plain EMA with the same `alpha_base` needs **6**.

---

## 8. Complexity and cost

Per frame, at analysis width `W` and height `H`:

| Stage | Complexity | Measured (320x180, Pi 5) |
| --- | --- | --- |
| preprocess | `O(WH)` | included below |
| five spatial metrics | `O(WH)` | 3.4 ms total |
| `fourier` | `O(WH log WH)` | 1.2 ms |
| analysis (noise, edges, motion, anisotropy) | `O(WH)` | ~3.5 ms |
| ensemble + filter | `O(M)`, M = 6 metrics | 0.12 ms |
| **total** | | **8.1 ms** |

The ensemble arithmetic is negligible; the cost is entirely in the image
operations. The known headroom, if it is ever needed, is that `tenengrad`,
`edge_width` and the anisotropy estimate each compute their own Sobel
derivatives; sharing one would remove roughly 1 ms. It was not done because the
pipeline already has about 5x of margin against the camera's frame interval.

---

## 9. What the measurements showed

Full tables in the [README](../README.md) and `data/comparison.json`. In summary:

**The temporal gate works as designed.** 1 frame to follow a focus step versus 6
for a plain EMA at the same baseline gain, with no loss of smoothing.

**Combining metrics matters more than how you combine them.** On a clean sweep,
a greedy search on any single metric ends 20 steps from the peak; every
combination scheme does far better.

**The adaptive weighting wins where it was designed to.** In the `low_texture`
condition, a plain and a fixed-weighted mean each pick up 2 false peaks and end
3 steps from the peak, while the adaptive scheme has 0 false peaks, ends 1 step
away, and raises discrimination from 1.70 to 2.74. The same pattern, less
strongly, in `clipped_highlights`.

**It does not win everywhere.** Across nine conditions, `adaptive` wins or ties
`plain_mean` in eight and loses one (`exposure_drift`). Outside the two
conditions above, its margin over a plain mean is modest.

**It costs repeatability.** Over five independent noise realisations of the same
sweep, every fixed method selects the same peak every time (std 0.000) while
`adaptive` moves between two adjacent frames (std 0.980). Mean peak error is
identical, so this is variance, not bias - but it is a real cost, caused by the
weights themselves being computed from noisy measurements.

**The noise-compensation ablation is not clearly beneficial.** In several
conditions `adaptive_no_noise_comp` scores *higher* discrimination than the full
scheme. This is partly because of the next point.

**On this camera the noise adaptation never engages.** The GH6 live-view JPEG is
denoised in-camera: the measured noise sigma is 0.0000 across 90 frames. The
noise term of the weighting is therefore inert on this source, and the noise
results above come only from synthetic data.

---

## 10. What would be needed to claim novelty

The scheme is specified and tested, but the following are **not** done and no
claim is made in their absence:

1. **A literature comparison.** No published combination scheme was
   re-implemented and run on the same data. The relevant baselines are the
   Pertuz et al. (2013) survey's recommended measures, and later learned or
   ensemble autofocus methods.
2. **Coefficients derived from data.** The `kappa_ij` are reasoned, not fitted.
   They should be estimated by regressing each metric's focus-response
   degradation against measured conditions on a labelled sweep corpus.
3. **Real focus sweeps.** Every comparison above uses simulated defocus. A disc
   PSF is a reasonable model but a real lens adds aberration, vignetting,
   focus breathing and a shifting subject. `tools/collect_dataset.py` and
   `docs/EXPERIMENTS.md` exist to gather that corpus; it has not been gathered.
4. **More than one scene and one camera.** All the reported numbers come from
   one synthetic scene family and one GH6 in one lighting setup.
5. **Statistical treatment.** Five noise seeds is enough to notice the
   repeatability difference, not to put a confidence interval on it.

Until at least 1 to 3 are done, the correct description of this work is: a
precisely specified combination scheme with a measured benefit in specific,
identified conditions, and a measured cost in repeatability.

---

## References

- Brenner, J. F. et al. (1976). An automated microscope for cytologic research.
  *Journal of Histochemistry & Cytochemistry* 24(1).
- Donoho, D. L. & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet
  shrinkage. *Biometrika* 81(3).
- Ferzli, R. & Karam, L. J. (2009). A no-reference objective image sharpness
  metric based on the notion of just noticeable blur. *IEEE TIP* 18(4).
- Kautsky, J. et al. (2002). A new wavelet-based measure of image focus.
  *Pattern Recognition Letters* 23(14).
- Krotkov, E. (1987). Focusing. *International Journal of Computer Vision* 1(3).
- Marziliano, P. et al. (2002). A no-reference perceptual blur metric. *ICIP*.
- Pech-Pacheco, J. L. et al. (2000). Diatom autofocusing in brightfield
  microscopy: a comparative study. *ICPR*.
- Pertuz, S., Puig, D. & Garcia, M. A. (2013). Analysis of focus measure
  operators for shape-from-focus. *Pattern Recognition* 46(5).
- Yang, G. & Nelson, B. J. (2003). Wavelet-based autofocusing and unsupervised
  segmentation of microscopic images. *IROS*.
