# Experiment protocol

Every comparison reported so far uses **simulated** defocus. This document is
the protocol for collecting a real focus-sweep corpus and re-running the same
analysis on it, so that the conclusions can be checked against optics rather
than a model.

## What has already been run

Reproduce with:

```bash
python3 tools/compare_methods.py --config config/default.toml --json data/comparison.json
```

Nine synthetic conditions (clean, heavy noise, dark and noisy, low texture, low
texture with noise, exposure drift, clipped highlights, 4 px motion, noise plus
motion), 41 steps each, disc PSF, true focus at the centre. Twelve methods: six
single metrics, a plain mean, a fixed weighted mean, the adaptive ensemble, and
three ablations. Plus a five-seed repeatability run and a step-response test.
Results and their honest reading are in the [README](../README.md) and §9 of
[ALGORITHM.md](ALGORITHM.md).

## Recording a real focus sweep

### Setup

1. **Camera**: GH6 in tethering USB mode, manual focus, fixed aperture, fixed
   shutter and ISO. Auto-exposure must be **off** - otherwise exposure varies
   with focus and confounds the measurement.
2. **Scene**: static, well lit, with structure at several scales. Record at
   least three scenes of different character:
   - rich texture (a resolution chart, fabric, printed text);
   - moderate (a normal indoor subject);
   - sparse and low contrast (a plain wall with one object).
   The third is the case where the adaptive weighting is supposed to earn its
   place, so it is the important one.
3. **Actuator**: the servo on the focus ring. Record its position with every
   frame. The loosest coupling is a file the controller rewrites:

   ```bash
   # in the servo controller
   echo "$position" > /run/focus_position
   ```

4. **Stability**: tripod, no vibration. Wait for the lens to stop before
   grabbing - `settle_ms` below.

### Sweep procedure

Sweep from clearly defocused, through best focus, to clearly defocused on the
other side. Use enough steps that the peak is sampled by several frames.

- **Range**: the full usable travel of the focus ring for the chosen subject
  distance, ideally with the subject sharp near the middle.
- **Steps**: 41 or more. Fewer than about 25 makes the peak-error criterion
  meaningless.
- **Settle**: at least 200 ms after each move before grabbing, and discard the
  first frame after a move - at 40 ms of capture latency it may still show the
  previous position.
- **Frames per step**: 5. Averaging is not the point; the repeat frames measure
  the score's own noise, which is what the temporal and repeatability analyses
  need.
- **Repeats**: sweep the full range three times, alternating direction. Two
  forward and one reverse exposes actuator backlash.

### Recording

```bash
python3 tools/collect_dataset.py \
    --config config/default.toml \
    --backend gphoto2 \
    --out data/sweep_scene1.csv \
    --save-frames data/sweep_scene1_frames \
    --motor-file /run/focus_position \
    --note "scene1 rich texture, f/4, 1/125, ISO400, sweep 1 of 3"
```

Saving the frames matters: it makes the recording re-analysable with a changed
configuration without going back to the hardware.

### Labelling the ground truth

The analysis needs the index of the truly in-focus frame. Determine it
*independently of the metrics under test*, or the comparison is circular. Two
acceptable methods:

- **Visual**: inspect the saved frames at 100% and pick the sharpest by eye.
  Record who judged it and when.
- **Optical**: photograph a slanted-edge or resolution chart at each step and
  take the peak of an MTF50 measurement made by separate software.

Record the chosen index in a sidecar file next to the frames.

## Analysing a recording

```bash
python3 tools/compare_methods.py \
    --frames-dir data/sweep_scene1_frames \
    --best-index 20 \
    --json data/comparison_scene1.json
```

This runs every method on the recorded frames with the same frozen
normalisation and prints the same table as the synthetic run.

To re-derive the reference constants for your camera and scenes:

```bash
python3 tools/calibrate_stats.py --backend file --path data/sweep_scene1_frames \
    --json data/stats_scene1.json
```

It prints suggested values for `edge_ref_density`, `contrast_ref`,
`noise_ref_sigma` and `motion_ref_px`. Do not adopt them from a single scene -
collect several and take a value that is reasonable across them.

## Criteria and how to read them

| Criterion | Meaning | Good value |
| --- | --- | --- |
| `monotonicity` | fraction of steps moving the right way relative to true focus | near 1.0 |
| `peak_error` | steps between the score maximum and true focus | 0 or 1 |
| `hill_climb` | mean distance from the peak after a greedy search from each end | near 0 |
| `discrimination` | `(peak - median) / robust spread` | well above 1 |
| `false_peaks` | distinct local maxima above 70% of the peak | 0 |
| `peak_std` (repeatability) | spread of the chosen peak over repeats | near 0 |

`hill_climb` is the criterion that actually separates the methods, because it
models what an autofocus loop does: it never sees the whole curve, it takes
steps and follows the gradient. A score can have good monotonicity and still
trap a search at a local bump.

Note that `peak_error` saturates near focus. With 41 steps over an 8 px defocus
range, the three frames nearest focus differ by less than a sub-pixel blur
radius and are optically almost identical, so a peak error of 1 is not a
meaningful failure.

## Robustness runs

Repeat the recording with a single factor changed, keeping everything else
fixed:

| Factor | How to vary it |
| --- | --- |
| Noise | raise ISO by 3 to 4 stops and compensate with shutter |
| Exposure | change shutter by +/- 1 stop between sweeps |
| Motion | pan slowly on a fluid head, or move the subject |
| Texture | swap to the sparse scene |
| Clipping | include a specular highlight or a window in frame |

Compare each against the clean run of the same scene. Report the *change* in
each criterion, not the absolute values - absolute values differ between scenes.

## Statistical treatment

The current results use five seeds and report a raw standard deviation, which is
enough to notice a difference and not enough to bound it. For a defensible
claim:

- at least 10 independent repeats per condition;
- report medians and interquartile ranges, not means, since the criteria are
  bounded and skewed;
- for a method-versus-method claim use a paired test across conditions (the
  same sweeps are scored by every method, so the pairing is exact);
- state the effect size, not only a p-value.

## What a complete study would still need

Listed in §10 of [ALGORITHM.md](ALGORITHM.md): a comparison against published
combination schemes, coefficients fitted to data rather than reasoned, real
sweeps, more than one camera, and a proper statistical treatment. The tooling
here supplies the data collection for all of them; none of them is done.
