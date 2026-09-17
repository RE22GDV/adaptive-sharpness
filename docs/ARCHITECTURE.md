# Architecture

## Layering

Three packages, with dependencies pointing one way only:

```
tools/  demo/          command-line programs and the visualisation
   |                   (may import anything; nothing imports them
   |                    except the synthetic frame source, lazily)
   v
capture/               frame sources: PTP, V4L2, file, synthetic
   |                   depends on: sharpness.types, sharpness.config
   v
sharpness/             the library: numpy + OpenCV only
```

`sharpness` never imports `capture`, `tools` or `demo`. It has no knowledge of
cameras, files, threads or windows - it turns arrays into scores. That is what
makes it testable without hardware and embeddable in another program that
already has its own capture.

The demo's dependency on OpenCV's `highgui` is deliberately confined to
`demo/demo_app.py`, so the GUI is never a requirement of the library.

## Data flow

```
FrameSource.read()
        |
        v
    Frame(data, index, timestamp, capture_latency_s, meta)
        |
        v
Preprocessor.prepare(data, roi)
    - to greyscale (one conversion, reused buffer)
    - crop to ROI *first* (a NumPy view, no copy)
    - downscale to analysis_width
        |
        v
    AnalysisImage(gray: float32 [0,255], scale, roi)
        |
        +--> ImageAnalyzer.analyze()  -> ImageStats
        |         (noise, edges, contrast, exposure, motion, anisotropy)
        |    ImageAnalyzer.degradations() -> d_j in [0,1]
        |
        +--> each SharpnessMetric.compute() -> raw x_i
                  |
                  v
        NormalizerBank.normalize()   -> s_i in [0,1]
                  |
                  v
        AdaptiveEnsemble.combine(s_i, stats, d_j)
            stage 1: reliability   r_i = exp(-sum_j kappa_ij d_j)
            stage 2: agreement     a_i (consensus kernel)
            stage 3: weights       w_i = norm(max(p_i r_i a_i, floor))
                                   S   = sum_i w_i s_i
                                   C   = confidence
                  |
                  v
        TemporalFilter.update(S, C)  -> gated EMA -> S_hat
                  |
                  v
            SharpnessResult
```

`SharpnessResult` carries everything: the filtered and unfiltered scores, the
confidence, and for every metric its raw value, normalised value, weight and
reliability, plus the image statistics and the timings. `as_row()` flattens it
for CSV. Nothing the ensemble decided is hidden from the caller - the data
collector and the demo both read the same object.

## Module responsibilities

| Module | Owns | State |
| --- | --- | --- |
| `types.py` | the data contracts | none (frozen dataclasses) |
| `config.py` | defaults, validation, TOML loading | none (frozen) |
| `preprocess.py` | colour, ROI, scale | scratch buffers |
| `metrics/` | the six metrics behind one interface | FFT window cache only |
| `analysis.py` | image conditions | previous frame, phase-correlation bias |
| `normalize.py` | mapping raw values to `[0,1]` | histories, frozen anchors |
| `ensemble.py` | weighting and confidence | none |
| `temporal.py` | smoothing | filtered value, innovation history |
| `evaluator.py` | wiring | owns the stateful pieces above |

## Threading and state

`SharpnessEvaluator` holds per-stream state - the normaliser histories, the
previous frame used for motion estimation, the temporal filter. It is therefore
**not thread-safe**: construct one per stream, and call `reset()` on a
discontinuity (stream restart, or a ROI change, which alters the analysis scale
and so the raw metric ranges).

`RegionEvaluator` holds two independent evaluators so that a whole frame and a
ROI can be scored side by side without their normalisation histories mixing.

Capture threading lives entirely in `capture/base.py`. `ThreadedSource` runs the
wrapped source in a worker and keeps a single-slot buffer, so a slow consumer
drops intermediate frames rather than falling behind a growing queue.

### Was threading worth it?

Measured on the live GH6, rather than assumed:

| | threaded | serial |
| --- | --- | --- |
| throughput | 25.00 fps | 24.98 fps |
| frame age at result | **7.44 ms** | 9.08 ms |
| with +60 ms consumer load | 14.81 fps, 87.6 ms age, 45 dropped | 12.49 fps, **67.7 ms** age, 0 dropped |

At the real operating point threading buys 1.6 ms of latency and no throughput,
because the camera paces the stream at exactly 25 fps and the USB wait absorbs
the processing time either way. Under overload it buys 19% throughput but costs
latency, because a grabbed frame then waits in the slot while the consumer is
busy.

It is left on by default - it is the better latency where the system actually
runs - but the honest summary is that this backend is camera-bound and threading
is not the reason the target rate is met. `capture.threaded = false` switches it
off.

## Design decisions worth knowing

**Crop before downscale.** The ROI crop happens on the full-resolution
greyscale image, and the downscale to `analysis_width` is applied to the crop.
A small ROI therefore never pays to resize the whole frame, and keeps its full
detail. The consequence is that raw metric values are not comparable across a
mid-run change of ROI mode; the running normaliser re-adapts within a few frames
but a deliberate switch should call `reset()`.

**Copies are avoided where it is safe, and made where it is not.** The colour
conversion and the resize each write into a reused buffer; the ROI crop is a
view. The one deliberate copy per frame is the motion reference, because
`cv2.phaseCorrelate` modifies its inputs in place.

**Failures are loud.** An unknown metric name, an unknown configuration key, a
metric given the wrong dtype, a ROI outside the frame - all raise. A silently
ignored configuration typo would be far more damaging than an error at startup.

**Confidence never modifies `instantaneous_score`.** It does reach the
temporal filter when confidence coupling is enabled, so it affects
`filtered_score`. A caller
that wants to ignore low-confidence frames does so explicitly.

## Extending

**A new metric**: subclass `SharpnessMetric`, implement `_compute(gray)`
returning a larger-is-sharper float, register a builder in
`metrics.METRIC_BUILDERS`, and add a sensitivity row to
`ensemble.sensitivity`. The evaluator, the ensemble, the demo and the CSV
schema pick it up with no further changes. The constructor validates that every
enabled metric has a sensitivity profile, so a forgotten row fails at startup.

**A new capture backend**: subclass `FrameSource`, implement
`open`/`read`/`close`, set `is_live`, and add it to `capture.open_source`.
