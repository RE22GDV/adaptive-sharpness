# Using the library inside an autofocus system

The library is deliberately narrow: it turns frames into scores. It knows
nothing about cameras, motors or search strategies, so it drops into an existing
system without bringing anything with it.

## Minimal use

```python
from adaptive_sharpness import SharpnessEvaluator, load_default_config

evaluator = SharpnessEvaluator(load_default_config())

result = evaluator.evaluate(bgr_frame)
if result.ready:
    print(result.instantaneous_score, result.filtered_score, result.confidence)
```

**There are two scores, and they answer different questions.**

| Field | What it is | Use it for |
| --- | --- | --- |
| `instantaneous_score` | the ensemble output for this frame alone; the confidence never touches it | comparing two positions you have already settled on |
| `filtered_score` | the same after temporal smoothing, whose gain the confidence does modulate | a live readout, or a loop that wants the jitter suppressed |
| `confidence` | a heuristic about the frame's conditions | **diagnosis, not selection** - see below |
| `ready` | warm-up finished *and* at least half the metrics informative | gating on the *scale*, which does work |

`ready` is the gate worth having. Before it is true the normaliser has not seen
enough to place a value on a scale, so the number is not yet a measurement.
`confidence` is not that gate; the section below says why.

For the whole sequence a real caller needs - calibrate, verify, freeze,
compare, reset - see [the worked example](../demo/calibrated_focus_loop.py),
which runs against a recording and is covered by
[tests/test_calibrated_loop.py](../tests/test_calibrated_loop.py).

`evaluate()` accepts a `Frame` or a bare NumPy array (`H x W x 3` BGR,
`H x W x 4` BGRA, or `H x W` greyscale; `uint8`, `uint16` or `float32`).

## With a region of interest

The ROI is in **source frame** pixel coordinates; the library rescales it
internally, so the caller never thinks about the analysis scale.

```python
from adaptive_sharpness import ROI

roi = ROI(x=320, y=180, width=640, height=360)
result = evaluator.evaluate(frame, roi=roi)
```

Feeding a ROI from an existing tracker - a face box, a depth-selected region, a
tap-to-focus point - is the intended use:

```python
box = tracker.current_box()          # (x, y, w, h) in source pixels
roi = ROI(*box).clipped_to(frame.shape)
result = evaluator.evaluate(frame, roi=roi)
```

Whole frame and ROI at once, each with its own normalisation history:

```python
from adaptive_sharpness import RegionEvaluator

evaluator = RegionEvaluator(config)
full, region = evaluator.evaluate(frame, roi=roi)
```

Call `reset()` whenever the stream is discontinuous or the ROI changes size
substantially - the raw metric ranges depend on the analysed region.

**How much the history matters, measured.** Starting the same recording halfway
through and scoring the *same* frames gives scores differing by up to **0.64**,
rank agreement of **0.56** against the full pass, and in the worst single case
the peak moved three protocol steps
([Д10](STUDY_UA_RESULTS.md#д10-залежність-від-передісторії)). This is not a
detail of the design, it is the largest source of variation an integrator will
meet.

Two practical consequences:

- **Let the evaluator watch before you search.** The scale freezes after 240
  observations plus 120 stable checks; a search started before that is climbing
  a scale that is still moving. Freezing is worth **+0.46 in rank correlation**
  against not freezing ([Д11](STUDY_UA_RESULTS.md#д11-параметри-автофіксації)),
  so it is worth waiting for.
- **Do not compare scores across a `reset()`.** After a reset the scale is
  refitted to whatever the camera sees next, so a number from before and a
  number from after are on different scales even for the same lens position.

## Closing a focus loop

A contrast-detection search needs three things from each frame: a score, a
confidence, and to know when the score changed for real. All three are on
`SharpnessResult`.

> **Read this before using the confidence as a gate.** The example below skips
> low-confidence frames, which is the obvious thing to do and is **not
> supported by the measurements**. Study
> [Д13](STUDY_UA_RESULTS.md#д13-якість-показника-впевненості) asked whether a
> confident frame is a more accurate one, using a criterion that is immune to
> the two artefacts a naive coverage curve suffers from. Within a protocol step
> - where the true focus is constant, so the spread of the score is measurement
> error and nothing else - the confident frames were the accurate ones on **four
> recordings of eight** and the inaccurate ones on the other four, with the rank
> correlation between confidence and error running from -0.71 to +0.63.
>
> As a per-frame indicator the confidence has no reliable sign on this corpus.
> It may still be useful in aggregate ("conditions in this stream are poor"),
> which is a different claim and was not tested. Treat `min_confidence` as an
> unvalidated knob, keep it low, and do not let it stop the search.

```python
class FocusSearch:
    """Hill-climb on the sharpness score, skipping untrustworthy frames."""

    def __init__(self, servo, evaluator, min_confidence=0.0, step=0.02):
        self.servo = servo
        self.evaluator = evaluator
        self.min_confidence = min_confidence
        self.step = step
        self.direction = 1
        self.best_score = -1.0
        self.best_position = servo.position()

    def update(self, frame, roi=None):
        position = self.servo.position()
        result = self.evaluator.evaluate(frame, roi=roi, motor_position=position)

        # Historically this skipped low-confidence frames.  The measurements
        # do not support that (see the note above), so the gate defaults to
        # off; it is left in place because it is the hook you would use if you
        # calibrate the confidence on your own footage.
        if self.min_confidence > 0.0 and result.confidence < self.min_confidence:
            return result

        if not result.ready:
            # The scale has not settled, so this number is not comparable with
            # the ones already collected.  This gate is the one that matters.
            return result

        if result.instantaneous_score > self.best_score:
            self.best_score = result.instantaneous_score
            self.best_position = position
        else:
            # Sharpness fell: we passed the peak, so reverse and halve.
            self.direction = -self.direction
            self.step = max(self.step * 0.5, 0.002)

        self.servo.move_to(position + self.direction * self.step)
        return result
```

The `motor_position` argument is carried through to `result.motor_position` and
into the CSV written by `tools/collect_dataset.py`, which is what makes a
recorded run analysable afterwards.

### Reacting to a real focus change

`result.focus_change_detected` is set when the temporal filter's gate opened,
meaning the score moved by much more than its recent noise. It is a useful
trigger for re-running a search after something in the scene changed:

```python
if result.focus_change_detected and not search.running:
    search.restart()
```

## Integrating with a depth-driven system

If focus is already set open-loop from a distance measurement - for example a
depth camera driving a servo on the focus ring - this library closes the loop:
the depth estimate puts the lens near the right place, and the sharpness score
confirms or corrects it over a small range.

```python
target = focus_mapping.distance_to_position(distance_m)
servo.move_to(target)

# Then refine within a narrow band around the open-loop estimate.
for position in fine_scan(around=target, span=0.05, steps=9):
    servo.move_to(position)
    frame = camera.read()
    result = evaluator.evaluate(frame, roi=subject_roi, motor_position=position)
    # No confidence gate: see the note above.  Collect every settled frame.
    candidates.append((result.instantaneous_score, position))

if candidates:
    servo.move_to(max(candidates)[1])
```

Two things to get right:

- **Let the lens settle.** Evaluate only frames captured after the servo has
  stopped, or the score measures motion blur as much as focus.
- **A ternary search over the range beat a hill climb.** On the recorded
  passes, ternary landed within one step of the physical reference on **100%**
  of runs using 8.5 evaluations; the hill climb managed 97% using 5.6
  ([Д19](STUDY_UA_RESULTS.md#д19-пошук-по-записаному-проходу)). That was a
  replay of stored profiles with no actuator, no backlash and no settling time,
  so it says the signal has a climbable shape - not how fast your lens will
  focus.
- **Account for the capture latency.** On the GH6 path a frame is about 40 ms
  old when its result appears. A frame grabbed immediately after a servo command
  may still show the previous position.

## Choosing frames to trust

```python
if result.confidence < 0.3:
    # Not enough structure, too dark, too noisy, or the metrics disagree.
    # Inspect WHICH, rather than discarding the frame on the number alone:
    print(result.stats.edge_sufficiency, result.stats.noise_sigma,
          result.stats.clipped_high, result.stats.motion_px)
```

The confidence is the geometric mean of six necessary-condition factors, capped
by the weakest one. A value near zero means one factor collapsed - most often
`edge_sufficiency`, i.e. there is nothing in the frame whose sharpness could be
measured.

**Use it to diagnose, not to decide.** The components tell you what is wrong
with a frame and that is worth reading. The scalar does not tell you the frame
is less accurate, because on these eight recordings it did not
([Д13](STUDY_UA_RESULTS.md#д13-якість-показника-впевненості)). It is also zero
on 24-79% of frames depending on the recording, so a threshold above zero
discards a large and non-random share of the stream.

## Bringing your own capture

Nothing forces the use of `capture/`. If the host system already has frames, just
pass them in. To reuse the library's threading and latency accounting instead,
implement one interface:

```python
from capture.base import FrameSource
from adaptive_sharpness.types import Frame

class MyCamera(FrameSource):
    def open(self): ...
    def read(self) -> Frame | None:
        started = time.perf_counter()
        image = self._grab()
        delivered = time.perf_counter()
        return Frame(
            data=image,
            index=self._index,
            timestamp=delivered,
            capture_latency_s=delivered - started,
            source="mycamera",
        )
    def close(self): ...

    @property
    def is_live(self) -> bool:
        return True
```

Wrapping it in `ThreadedSource` then gives newest-frame-wins behaviour - though
measure whether it helps before enabling it; on the GH6 path it buys 1.6 ms.

## Configuration from the host application

```python
from adaptive_sharpness import SharpnessConfig, SharpnessEvaluator

config = SharpnessConfig().with_overrides(
    pipeline={"analysis_width": 240},              # cheaper
    metrics={"enabled": ("laplacian", "tenengrad", "brenner")},
    temporal={"alpha_base": 0.5, "gate_abs_jump": 0.08},
)
evaluator = SharpnessEvaluator(config)
```

Unknown sections and unknown keys raise immediately, so a typo cannot silently
change behaviour.

## Threading

`SharpnessEvaluator` holds per-stream state and is **not thread-safe**.
Construct one per stream. Evaluating two streams from two threads with one
evaluator will corrupt the normalisation histories and the motion reference.

## Dependencies this pulls in

`numpy` and `opencv-python`, and nothing else. Importing `sharpness` does not
import `capture`, does not touch a camera driver, and does not require OpenCV's
GUI support.
