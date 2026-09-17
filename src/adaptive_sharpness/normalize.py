"""Running normalisation of heterogeneous raw metric values.

The six metrics have incommensurable units and dynamic ranges: the Laplacian
variance of a textured 320 px frame is in the hundreds, the Fourier ratio lives
in ``[0, 1]``, and the reciprocal edge width is a fraction of an inverse pixel.
They cannot be summed until they share a scale.

The normaliser maps each metric onto ``[0, 1]`` using robust percentiles of a
rolling history, so the scale follows the scene instead of assuming one.  Two
details matter:

* **Monotone compression.**  ``log1p`` is applied first (configurable).  It does
  not change the ordering of values — so it cannot change which frame is judged
  sharpest — but it makes the heavy-tailed energy metrics roughly symmetric, and
  percentile anchors on a symmetric distribution are far more stable.
* **Frozen mode.**  A rolling scale is right for a live autofocus loop, but it
  destroys comparability when analysing a recorded focus sweep offline.
  :meth:`NormalizerBank.freeze` locks the current anchors so that a whole
  dataset is scaled identically.
"""
from __future__ import annotations

import logging
import math
from collections import deque
from typing import Iterable, Mapping, Sequence

import numpy as np

from .config import NormalizationConfig

logger = logging.getLogger(__name__)

__all__ = ["RunningNormalizer", "NormalizerBank"]


class RunningNormalizer:
    """Maps one metric's raw values onto [0, 1] using rolling percentiles."""

    def __init__(self, name: str, config: NormalizationConfig) -> None:
        self.name = name
        self.config = config
        if config.window < 2:
            raise ValueError("normalization window must be at least 2")
        if not 0.0 <= config.low_percentile < config.high_percentile <= 100.0:
            raise ValueError("require 0 <= low_percentile < high_percentile <= 100")
        if config.mapping not in ("linear", "logistic"):
            raise ValueError(
                f"unknown normalization mapping {config.mapping!r}; "
                "expected 'linear' or 'logistic'"
            )
        if config.logistic_gain <= 0.0:
            raise ValueError("logistic_gain must be positive")
        self._history: deque[float] = deque(maxlen=config.window)
        # A longer history, used only when the rolling one has no range left.
        # A focus sweep pauses at each position, so the rolling window can
        # easily contain nothing but one held position; the sweep it belongs to
        # still has range, and scaling against that is far better than
        # refusing to answer.
        self._long_history: deque[float] | None = (
            deque(maxlen=config.window * config.long_window_multiple)
            if config.long_window_multiple > 0
            else None
        )
        self._frozen_anchors: tuple[float, float] | None = None
        self._auto_frozen = False
        self._stability_reference = 0.0
        self._stable_for = 0
        #: Why and when the scale froze, for diagnostics; None while rolling.
        self._freeze_record: dict[str, object] | None = None
        # Total samples ever seen.  Not len(self._history): that deque is
        # capped at `window`, so comparing it against a larger threshold can
        # never be satisfied.
        self._seen = 0

    # -- helpers ---------------------------------------------------------------

    def _consider_freezing(self) -> None:
        """Freeze once the observed range has stopped growing, and thaw if the
        scene later leaves it.

        Freezing is what makes a higher score mean a sharper frame: with a
        rolling scale the same frame is scored differently depending on what
        came before it, which is the defect measured in docs/CALIBRATION.md.
        The wait for a stable range matters - freezing on the first 120 frames
        of a sweep that starts fully defocused would fix the anchors on a
        sliver of the range and clip everything afterwards.
        """
        config = self.config
        if not config.auto_freeze:
            return
        if self._frozen_anchors is not None and not self._auto_frozen:
            # Frozen deliberately by the caller.  That is a statement about the
            # scale, not a guess, so nothing here may overrule it.
            return

        if self._auto_frozen:
            if not config.auto_thaw:
                return
            low, high = self._frozen_anchors  # type: ignore[misc]
            span = high - low
            recent = np.fromiter(self._history, dtype=np.float64, count=len(self._history))
            margin = span * config.auto_thaw_margin
            outside = np.mean((recent < low - margin) | (recent > high + margin))
            if float(outside) >= config.auto_thaw_outside:
                logger.debug("normalizer %s thawed: %.0f%% outside", self.name, 100 * outside)
                self._frozen_anchors = None
                self._auto_frozen = False
                self._stability_reference = 0.0
                self._stable_for = 0
                self._freeze_record = None
            return

        if self._seen < config.auto_freeze_min_samples:
            return
        history = self._long_history if self._long_history is not None else self._history
        anchors = self._percentiles(history)
        if anchors is None or self._too_narrow(anchors):
            return
        low, high = anchors
        span = high - low

        # The stability test compares against a reference fixed at the start of
        # the interval, not against a maximum that is updated every sample.
        # With a running maximum, growth slower than the threshold is never
        # caught: at 1% per sample against a 5% threshold the scale froze after
        # 119 samples while the span had grown 3.3x.
        if self._stable_for == 0:
            self._stability_reference = span
        if span > self._stability_reference * (1.0 + config.auto_freeze_stability):
            self._stability_reference = span
            self._stable_for = 0
            return
        self._stable_for += 1
        if self._stable_for >= config.auto_freeze_patience:
            self._frozen_anchors = anchors
            self._auto_frozen = True
            self._freeze_record = {
                "sample": self._seen,
                "anchors": (float(low), float(high)),
                "span": float(span),
                "stable_for": self._stable_for,
                "reference_span": float(self._stability_reference),
            }
            logger.info(
                "normalizer %s froze after %d samples at [%.6g, %.6g]",
                self.name, self._seen, low, high,
            )

    def _compress(self, value: float) -> float:
        if not self.config.log_compress:
            return value
        # log1p is only defined for value > -1; metrics are non-negative.
        return math.log1p(max(0.0, value))

    def _percentiles(self, history: deque[float]) -> tuple[float, float] | None:
        if len(history) < 2:
            return None
        data = np.fromiter(history, dtype=np.float64, count=len(history))
        low = float(np.percentile(data, self.config.low_percentile))
        high = float(np.percentile(data, self.config.high_percentile))
        return low, high

    def _anchors(self) -> tuple[float, float] | None:
        if self._frozen_anchors is not None:
            return self._frozen_anchors
        return self._percentiles(self._history)

    def _too_narrow(self, anchors: tuple[float, float]) -> bool:
        low, high = anchors
        floor = self.config.min_range_ratio * max(
            abs(high), abs(low), self.config.min_range_absolute
        )
        return (high - low) <= floor

    def _usable_anchors(self) -> tuple[float, float] | None:
        """The anchors to scale against, widening the horizon if need be.

        Returns ``None`` only when neither horizon carries any range, which is
        a genuinely static scene or a stuck sensor rather than a held focus
        position.
        """
        anchors = self._anchors()
        if anchors is not None and not self._too_narrow(anchors):
            return anchors
        if self._frozen_anchors is not None or self._long_history is None:
            return None
        extended = self._percentiles(self._long_history)
        if extended is not None and not self._too_narrow(extended):
            return extended
        return None

    # -- public API ------------------------------------------------------------

    @property
    def warmed_up(self) -> bool:
        return len(self._history) >= self.config.warmup

    @property
    def sample_count(self) -> int:
        return len(self._history)

    @property
    def is_degenerate(self) -> bool:
        """True when the observed range is too small to normalise against.

        In that state :meth:`normalize` returns a neutral 0.5 whatever it is
        given, so the value carries no information.  This has to be visible to
        the caller: on a completely static scene *every* metric collapses to
        0.5, which looks like perfect inter-metric agreement and would otherwise
        be reported as high confidence in a meaningless score.
        """
        return self._usable_anchors() is None

    def observe(self, raw: float) -> None:
        """Add a raw value to the history without producing an output."""
        compressed = self._compress(float(raw))
        self._history.append(compressed)
        if self._long_history is not None:
            self._long_history.append(compressed)
        self._seen += 1
        self._consider_freezing()

    def normalize(self, raw: float, observe: bool = True) -> float:
        """Return the normalised value of ``raw``, optionally recording it."""
        compressed = self._compress(float(raw))
        if observe:
            self._history.append(compressed)
            if self._long_history is not None:
                self._long_history.append(compressed)
            self._seen += 1
            self._consider_freezing()

        anchors = self._usable_anchors()
        if anchors is None:
            # Nothing to scale against: a neutral 0.5 keeps the ensemble well
            # defined without pretending to know where this frame sits.  The
            # caller is told through is_degenerate and informative_fraction,
            # because 0.5 outranks a genuine low score and would otherwise
            # steer a search away from focus.
            return 0.5
        return self._map(compressed, anchors)

    def _map(self, compressed: float, anchors: tuple[float, float]) -> float:
        """Place a compressed value between the anchors, on [0, 1]."""
        low, high = anchors
        span = high - low
        if self.config.mapping == "linear":
            return float(min(1.0, max(0.0, (compressed - low) / span)))
        # Logistic: monotone everywhere, so two frames of different sharpness
        # never collide the way they do past a clipped linear anchor.
        z = self.config.logistic_gain * (compressed - 0.5 * (low + high)) / span
        return float(1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z)))))

    def freeze(self) -> None:
        """Lock the anchors of the current history so later calls use them.

        Recomputed from the history rather than read back through
        :meth:`_anchors`, which would return anchors the automatic rule had
        already installed and so make an explicit freeze a silent no-op - and
        make :meth:`fit` ignore the batch it was handed.
        """
        anchors = self._percentiles(self._history)
        if anchors is None:
            raise RuntimeError(
                f"normalizer {self.name!r} has too few samples ({len(self._history)}) to freeze"
            )
        self._frozen_anchors = anchors
        # An explicit freeze is not the automatic one and must not be thawed.
        self._auto_frozen = False
        logger.debug("froze normalizer %s at %s", self.name, anchors)

    def unfreeze(self) -> None:
        self._frozen_anchors = None
        self._auto_frozen = False
        self._stability_reference = 0.0
        self._stable_for = 0
        self._freeze_record = None

    @property
    def frozen(self) -> bool:
        """True while a fixed scale is in use rather than a rolling one."""
        return self._frozen_anchors is not None

    def fit(self, values: Iterable[float]) -> None:
        """Fit the scale to a whole batch of raw values, then freeze.

        The anchors come from every value given, not from the tail that happens
        to fit in the rolling history: a caller fitting a recorded sweep means
        the sweep, and silently dropping its first third would scale the
        dataset against a fragment of itself.
        """
        self.reset()
        compressed = [self._compress(float(value)) for value in values]
        if len(compressed) < 2:
            raise RuntimeError(
                f"normalizer {self.name!r} needs at least 2 samples to fit"
            )
        for value in compressed:
            self._history.append(value)
            if self._long_history is not None:
                self._long_history.append(value)
        self._seen = len(compressed)
        data = np.asarray(compressed, dtype=np.float64)
        self._frozen_anchors = (
            float(np.percentile(data, self.config.low_percentile)),
            float(np.percentile(data, self.config.high_percentile)),
        )
        self._auto_frozen = False
        logger.debug(
            "fitted normalizer %s on %d samples: %s",
            self.name, len(compressed), self._frozen_anchors,
        )

    def reset(self) -> None:
        self._history.clear()
        if self._long_history is not None:
            self._long_history.clear()
        self._frozen_anchors = None
        self._auto_frozen = False
        self._stability_reference = 0.0
        self._stable_for = 0
        self._freeze_record = None
        self._seen = 0


class NormalizerBank:
    """One :class:`RunningNormalizer` per metric, addressed by name."""

    def __init__(self, names: Sequence[str], config: NormalizationConfig) -> None:
        self.config = config
        self._normalizers: dict[str, RunningNormalizer] = {
            name: RunningNormalizer(name, config) for name in names
        }

    def __contains__(self, name: object) -> bool:
        return name in self._normalizers

    def __getitem__(self, name: str) -> RunningNormalizer:
        return self._normalizers[name]

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._normalizers)

    @property
    def warmed_up(self) -> bool:
        """True once every metric has seen at least ``warmup`` samples."""
        return all(n.warmed_up for n in self._normalizers.values())

    @property
    def sample_count(self) -> int:
        """Frames observed by the least-populated normaliser."""
        if not self._normalizers:
            return 0
        return min(n.sample_count for n in self._normalizers.values())

    @property
    def scale_frozen(self) -> bool:
        """True once every metric is on a fixed scale.

        Until then the score is still relative to a history that keeps moving,
        so two readings taken minutes apart are not strictly comparable.  A
        host that waits for this before trusting an absolute comparison is
        doing the right thing.
        """
        if not self._normalizers:
            return False
        return all(n.frozen for n in self._normalizers.values())

    @property
    def informative_fraction(self) -> float:
        """Fraction of metrics whose normalisation is currently meaningful.

        0.0 means every metric is pinned at the neutral 0.5 because nothing in
        the scene has changed, and the resulting score says nothing.
        """
        if not self._normalizers:
            return 0.0
        live = sum(not n.is_degenerate for n in self._normalizers.values())
        return live / len(self._normalizers)

    def normalize(self, raw: Mapping[str, float], observe: bool = True) -> dict[str, float]:
        """Normalise a whole set of raw metric values."""
        out: dict[str, float] = {}
        for name, value in raw.items():
            normalizer = self._normalizers.get(name)
            if normalizer is None:
                raise KeyError(f"no normalizer registered for metric {name!r}")
            out[name] = normalizer.normalize(value, observe=observe)
        return out

    def fit(self, batch: Mapping[str, Sequence[float]]) -> None:
        """Fit every normaliser from recorded values, then freeze them all."""
        for name, values in batch.items():
            if name in self._normalizers:
                self._normalizers[name].fit(values)

    def freeze(self) -> None:
        for normalizer in self._normalizers.values():
            normalizer.freeze()

    def unfreeze(self) -> None:
        for normalizer in self._normalizers.values():
            normalizer.unfreeze()

    def reset(self) -> None:
        for normalizer in self._normalizers.values():
            normalizer.reset()
