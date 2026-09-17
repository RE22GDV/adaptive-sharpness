# Limitations

What this system does not do, and where its measured numbers stop applying.

## Capture

**640x360 at 25 fps is the ceiling on this camera.** The GH6's PTP live-view
stream is 640x360 and paced at 25 fps by the camera. libgphoto2 cannot enlarge
it. Fine focus detail present only above that resolution is not measurable
through this path. An external HDMI capture device is the supported way around
it (see [GH6_SETUP.md](GH6_SETUP.md)); it is not currently installed.

**Capture latency is ~40 ms and is not ours to improve.** The grab call spends
38.6 ms waiting on the camera. An autofocus loop closing on this signal has at
least one frame interval of dead time regardless of how fast the metrics are.

**One camera at a time.** libgphoto2 claims the device exclusively, and any
other process holding it (typically gvfs) prevents initialisation.

**The camera must not sleep.** There is no automatic reconnection: if the camera
powers down, the source raises after `max_consecutive_errors` failures and the
program must restart the capture.

## Measurement of image conditions

**The noise estimate reads low, and it used to read nothing at all.** An earlier
version of this section said the noise adaptation was inert on the GH6
live-view stream - sigma 0.0000 across 90 frames - and blamed the camera's
preview denoising. That diagnosis was wrong. The estimator took the *median* of
the finest Haar detail band, and on an 8-bit stream most of those coefficients
are exactly zero, so the median was pinned at zero however much noise was
present. Reading the same distribution at the 75th percentile instead moved the
share of frames reporting zero noise from **97.59% to 0.17%** across the eight
recordings ([table](RESULTS.md#each-repair-on-its-own)).

What remains true is that the estimate is not a measurement. It has a
structure-dependent floor, reads about 35% low against a known injected sigma,
and responds to image content as well as to sensor noise. Treat it as a
relative signal within one stream.

**Only global translation is detected as motion.** Motion is estimated by phase
correlation, which measures a rigid shift of the whole frame. A subject moving
*inside* a static frame, rotation, zoom and non-rigid deformation are not
detected, so the motion degradation stays near zero for them and the weighting
does not react.

**Edge density conflates structure with contrast, and there is now evidence it
costs something.** It is a Canny edge fraction at fixed thresholds, so a dim
low-contrast scene reads as having little structure even when it is
geometrically detailed. That means `edge_sufficiency` - and therefore the
confidence - is partly an exposure measurement.

The two recordings where adaptive weights do worst
([CALIBRATION](CALIBRATION.md#does-the-adaptivity-earn-its-keep)) are the two
darkest, with a measured edge density of 0.0000 against an `edge_ref_density`
of 0.006. Both the edge and the exposure degradation factors are pinned at
their extremes there, so the reliability model is weighting from inputs that
have saturated. That is an association across two recordings rather than a
demonstrated cause, but it is the predicted failure of a per-frame edge count
that bottoms out, and it is what the redesign would have to fix.

**The reference constants are scene-dependent.** `edge_ref_density`,
`contrast_ref`, `noise_ref_sigma` and `motion_ref_px` set what counts as "enough
structure" or "too noisy". The shipped values were measured on one GH6 scene and
one synthetic scene family. Use `tools/calibrate_stats.py` on your own footage
before trusting the confidence numbers quantitatively.

## The metrics

**Sharpness is not focus.** Every metric here measures high-frequency content.
Motion blur, a dirty lens, atmospheric haze, heavy compression and a
low-contrast subject all reduce it without any focus error. The system reports
what it measures; interpreting the cause is the caller's job.

**Contrast normalisation does not survive clipping.** Dividing by `mean(I)^2`
makes the energy metrics invariant to a multiplicative illumination change, but
an exposure change severe enough to clip destroys real detail and the score
genuinely falls. The confidence drops in that case, which is the intended
behaviour, but the score is not exposure-invariant across clipping.

**`edge_width` saturates.** The morphological window bounds the largest
measurable edge width at roughly `edge_range_window - 2` pixels, so about 9 px
by default. Beyond that the metric stops discriminating; the other five carry
the signal in the heavily defocused range.

**`fourier` measures the whole frame globally.** It has no spatial localisation,
so a sharp region and a blurred region average together. Use a ROI when that
matters.

**A pixel-aligned step edge is a degenerate case.** An ideally sharp edge that
lands exactly on a pixel boundary has zero transition width and therefore zero
fine-scale detail energy; blurring it *creates* detail before destroying it, and
several metrics rise with defocus. Real optics and real sensors never produce
this, but synthetic test scenes easily do - which is why `tools/synthetic.py`
renders at 4x and area-averages down. Anyone building their own test scenes
should do the same.

## The ensemble

**The sensitivity coefficients are reasoned, not fitted.** The `kappa_ij` matrix
encodes qualitative facts (a second derivative amplifies noise more than a first
derivative; the edge-width metric is undefined without edges). It has not been
estimated from data. See §10 of [ALGORITHM.md](ALGORITHM.md).

**The adaptive weighting reduces repeatability.** Measured over five independent
noise realisations of the same sweep, every fixed method selected the same peak
every time (std 0.000 steps) while the adaptive scheme moved between two
adjacent frames (std 0.980). Mean accuracy was unchanged. The weights are
computed from noisy measurements and therefore add variance of their own. If
deterministic repeatability matters more than adaptivity, use
`fixed_weighted` - the priors alone.

**Its advantage over a plain mean is conditional.** Across nine test conditions
it wins clearly in two (`low_texture`, `clipped_highlights`), ties in six, and
loses one (`exposure_drift`). It is not uniformly better, and the honest
headline is that *combining* metrics matters much more than the combination
rule.

**No comparison against published combination schemes has been made.** See §10
of [ALGORITHM.md](ALGORITHM.md) for what a novelty claim would require.

## Temporal filtering

**The gate can be tripped by a fast-moving subject.** It responds to a large
change in the score, not to a focus change specifically. A subject entering the
frame produces the same signature and passes through unfiltered.

**On an offline sweep the filter mildly hurts.** Every frame of a sweep is a
genuine change, so any smoothing lags. The ablation shows
`adaptive_no_temporal` sometimes locating the peak slightly better. The filter
is for live jitter, and the step-response test is the right measurement of it
(1 frame versus 6 for a plain EMA).

## Validation

**Every reported comparison uses simulated defocus.** A disc PSF is the
geometric-optics circle of confusion and a reasonable model, but a real lens
adds spherical aberration, vignetting, focus breathing and chromatic effects,
and a real sweep adds subject and actuator motion. No real recorded focus sweep
has been analysed. The tools to record one exist
(`tools/collect_dataset.py`, [EXPERIMENTS.md](EXPERIMENTS.md)); the corpus does
not.

**One scene family, one camera, one lighting setup.** All numbers come from a
single synthetic scene generator and a single GH6 pointed at one indoor subject
(measured brightness 0.17, RMS contrast 0.067 - a dim, low-contrast target).

**Five seeds is not a statistical treatment.** Enough to notice the
repeatability difference; not enough to bound it.

## Platform

Measured on: Raspberry Pi 5 Model B Rev 1.0, 8 GB, Raspberry Pi OS bookworm,
kernel 6.12.75, Python 3.11.2, numpy 1.26.4, OpenCV 4.13.0. Timings will differ
elsewhere; re-run `tools/benchmark.py` rather than assuming them.

The library requires Python 3.11 or newer, because configuration loading uses
the standard library's `tomllib`.
