# Using the library inside an autofocus system

The library is deliberately narrow: it turns frames into scores. It knows
nothing about cameras, motors or search strategies, so it drops into an existing
system without bringing anything with it.

## Minimal use

```python
from sharpness import SharpnessEvaluator, load_config

evaluator = SharpnessEvaluator(load_config("config/default.toml"))

result = evaluator.evaluate(bgr_frame)
print(result.score, result.confidence)
```

`evaluate()` accepts a `Frame` or a bare NumPy array (`H x W x 3` BGR,
`H x W x 4` BGRA, or `H x W` greyscale; `uint8`, `uint16` or `float32`).

## With a region of interest

The ROI is in **source frame** pixel coordinates; the library rescales it
internally, so the caller never thinks about the analysis scale.

```python
from sharpness import ROI

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
from sharpness import RegionEvaluator

evaluator = RegionEvaluator(config)
full, region = evaluator.evaluate(frame, roi=roi)
```

Call `reset()` whenever the stream is discontinuous or the ROI changes size
substantially - the raw metric ranges depend on the analysed region.

## Closing a focus loop

A contrast-detection search needs three things from each frame: a score, a
confidence, and to know when the score changed for real. All three are on
`SharpnessResult`.

```python
class FocusSearch:
    """Hill-climb on the sharpness score, skipping untrustworthy frames."""

    def __init__(self, servo, evaluator, min_confidence=0.35, step=0.02):
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

        # A low-confidence frame carries no usable information: hold still
        # rather than stepping on noise.
        if result.confidence < self.min_confidence:
            return result

        if result.score > self.best_score:
            self.best_score = result.score
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
    if result.confidence >= 0.35:
        candidates.append((result.score, position))

if candidates:
    servo.move_to(max(candidates)[1])
```

Two things to get right:

- **Let the lens settle.** Evaluate only frames captured after the servo has
  stopped, or the score measures motion blur as much as focus.
- **Account for the capture latency.** On the GH6 path a frame is about 40 ms
  old when its result appears. A frame grabbed immediately after a servo command
  may still show the previous position.

## Choosing frames to trust

```python
if result.confidence < 0.3:
    # Not enough structure, too dark, too noisy, or the metrics disagree.
    # Inspect why:
    print(result.stats.edge_sufficiency, result.stats.noise_sigma,
          result.stats.clipped_high, result.stats.motion_px)
```

The confidence is the geometric mean of six necessary-condition factors, capped
by the weakest one. A value near zero means one factor collapsed - most often
`edge_sufficiency`, i.e. there is nothing in the frame whose sharpness could be
measured.

## Bringing your own capture

Nothing forces the use of `capture/`. If the host system already has frames, just
pass them in. To reuse the library's threading and latency accounting instead,
implement one interface:

```python
from capture.base import FrameSource
from sharpness.types import Frame

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
from sharpness import SharpnessConfig, SharpnessEvaluator

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
