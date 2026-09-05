# Locating what is in focus

The evaluator answers "how sharp is this region". Answering "**which object** is
in focus" needs the frame scored everywhere at once, then compared region
against region. That is the scene stage: `sharpness/focusmap.py`,
`sharpness/regions.py` and `sharpness/scene.py`, with the UI in
`demo/focus_ui.py`.

```bash
python3 demo/focus_ui.py --config config/default.toml --backend gphoto2
python3 demo/focus_ui.py --backend synthetic --headless --frames 60
python3 demo/focus_ui.py --backend gphoto2 --snapshot /tmp/ui.png
```

## The problem that shapes the design

Low gradient energy has **two completely different causes**: the region is
defocused, or the region has nothing in it to be sharp. A blank wall and a badly
defocused wall look nearly identical to any plain gradient measure. A display
that painted both as "out of focus" would be confidently wrong about half the
frame, so every tile carries a validity flag and only tiles with enough contrast
are ranked at all. The panel reports what fraction of the frame was even
measurable.

A second trap: the tile scores are rescaled *within the frame*, which is the
right comparison to make - but on a uniformly sharp frame that rescale stretches
whatever numerical noise exists across the full range and manufactures a winner.
The stretch is therefore scaled by `relative_spread`, how much the raw measure
actually varies relative to its own median. A frame with nothing to choose
between regions produces a map that stays near a neutral 0.5, and the UI says
so.

## Stage 1: the focus map

A tile grid over the analysis image. Every tile statistic is a block mean
computed with one `cv2.resize` in `INTER_AREA` mode over the whole frame, so the
cost is a few full-frame passes regardless of tile count - about 2 ms at 320 px.

To reduce the dependence on *how much* texture a region has, the measures are
all ratios. Three are implemented:

| measure | definition |
| --- | --- |
| `grad_over_var` | `mean(grad^2) / var(I)` - gradient energy per unit contrast |
| `lap_over_var` | the same with a second derivative |
| `lap_over_grad` | `mean(lap^2) / mean(grad^2)` - a contrast-free estimate of the region's characteristic spatial frequency |

### Which one, and why

`tools/compare_focus_measures.py` runs all three on nine controlled split-focus
scenes, including the adversarial case that matters: a **sparsely textured but
sharp** half beside a **densely textured but blurred** half, where a plain
gradient measure picks the blurred side because it has more texture.

On clean scenes all three pass. Noise separates them, because a ratio of two
high-pass measures is dominated by whatever noise puts back at high
frequencies:

| measure | passed | smallest correct margin (noise sigma 20) | largest false margin |
| --- | --- | --- | --- |
| **`grad_over_var`** | **9/9** | **0.257** | **0.158** |
| `lap_over_var` | 9/9 | 0.141 | 0.263 |
| `lap_over_grad` | 8/9 | 0.039 (fails) | 0.224 |

`grad_over_var` wins on both counts and is the default. `lap_over_grad` has the
largest margins on clean scenes, which is why it was the initial choice - the
noise test is what corrected that.

## Stage 2: region proposals

Two detectors, both offline and dependency-free:

- **`faces`** - the Haar cascade bundled with OpenCV. It gives a *named*
  subject, and "the face is in focus" is a far more useful statement than "tile
  (3, 7) is in focus". It is the most expensive step in this stage, so it runs
  every `face_interval` frames (default 3) and its boxes are reused in between.
- **`sharp_blobs`** - connected components of the focus map above a fraction of
  its own peak. Needs no model, finds whatever is actually sharp, and is the
  fallback that always works.

Proposals are merged by intersection-over-union, and a named detection survives
its overlapping blob.

## Stage 3: ranking and full measurement

Regions are ranked by their **focus-map score**, not by the ensemble. This is
deliberate. The ensemble score is normalised against a rolling history of
previous frames, which is right for tracking one region over time and wrong for
comparing several regions inside one frame: each would need its own history,
those histories would be built from different content, and regions come and go
so there is nothing stable to attach one to. The focus map is rescaled within
the frame, so tile-against-tile is exactly the comparison it supports.

The full adaptive ensemble then runs on two things only: the whole frame, and
the winning region - where a persistent history does make sense. The subject is
tracked between frames by IoU, and its normalisation history is reset when the
identity changes, because a different subject has a different raw metric range.

## What the UI shows

| element | meaning |
| --- | --- |
| heat map | per-tile focus, tinted only where the tile had enough texture, and only in proportion to how decisive the reading was |
| green box | the subject judged to be in focus |
| yellow boxes | the other candidates, with their map scores |
| **decision margin** | how far the winner leads the runner-up |
| **focus variation in frame** | whether there is any real focus difference to detect; `stretch < 1` means the map was damped |
| subject sharpness + conf | the full ensemble on the winning region |
| metric rows | each metric's normalised value and its current adaptive weight |
| whole frame | the same ensemble over the entire frame, for context |
| measurable | fraction of tiles that carried usable texture |

Two distinctions the panel keeps separate, because collapsing them would make it
lie:

- **"no texture" is not "out of focus"** - untextured tiles are left untinted
  and excluded from the ranking.
- **confidence in the score is not confidence in the decision** - a subject can
  be measured precisely while barely beating the runner-up. `conf` is the
  measurement; `decision margin` is the ranking.

## Measured cost

Raspberry Pi 5, 320 px analysis width, all six metrics, live GH6:

| | |
| --- | --- |
| scene stage (map + regions) | 1.7-4.7 ms |
| total per frame | 19.3 ms mean, 22.6 ms p95 |
| end-to-end | 22.3 fps |
| subject stability | one subject held for 100% of 90 frames |

The frame interval is 40 ms, so the scene stage roughly doubles the processing
cost and still leaves about half the budget free.

## Controls

```
q / Esc  quit                 h  heat map on/off
space    pause                b  candidate boxes on/off
r        reset state          f  face detector on/off
g        cycle tile size      t  tile grid on/off
s        save a PNG snapshot
```

## Using it as a library

```python
from sharpness import SceneEvaluator, load_config

evaluator = SceneEvaluator(load_config("config/default.toml"))
result = evaluator.evaluate(frame)

if result.has_subject:
    print(result.subject.label, result.subject.center)
    print(result.subject_result.score, result.subject_result.confidence)
    print("lead over runner-up:", result.separation())
else:
    print("nothing in the frame has enough texture to judge")
```

`SceneEvaluator` holds per-stream state (two evaluators, the motion reference,
the face cache), so construct one per stream and do not share it across threads.

## Limitations

- **The subject is whatever is sharpest, not whatever matters.** Without a face
  in frame, "sharp area" is a geometric answer, not a semantic one. A real
  object detector would need a model this machine does not have.
- **The Haar cascade is frontal-face only** and has the false-positive rate its
  age implies. It is a convenience, not a reliable detector.
- **Blobs are unstable frame to frame** on scenes with little focus variation,
  which is exactly when `decision margin` is small - read the two together.
- **Everything is relative to the current frame.** The map cannot say the whole
  frame is out of focus, only which part of it is sharpest. The whole-frame
  ensemble score, which is relative to recent history, is the complement to
  that.
