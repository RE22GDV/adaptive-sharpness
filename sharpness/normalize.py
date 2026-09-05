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
        self._history: deque[float] = deque(maxlen=config.window)
        self._frozen_anchors: tuple[float, float] | None = None

    # -- helpers ---------------------------------------------------------------

    def _compress(self, value: float) -> float:
        if not self.config.log_compress:
            return value
        # log1p is only defined for value > -1; metrics are non-negative.
        return math.log1p(max(0.0, value))

    def _anchors(self) -> tuple[float, float] | None:
        if self._frozen_anchors is not None:
            return self._frozen_anchors
        if len(self._history) < 2:
            return None
        data = np.fromiter(self._history, dtype=np.float64, count=len(self._history))
        low = float(np.percentile(data, self.config.low_percentile))
        high = float(np.percentile(data, self.config.high_percentile))
        return low, high

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
        anchors = self._anchors()
        if anchors is None:
            return True
        low, high = anchors
        floor = self.config.min_range_ratio * max(abs(high), abs(low), 1.0)
        return (high - low) <= floor

    def observe(self, raw: float) -> None:
        """Add a raw value to the history without producing an output."""
        self._history.append(self._compress(float(raw)))

    def normalize(self, raw: float, observe: bool = True) -> float:
        """Return the normalised value of ``raw``, optionally recording it."""
        compressed = self._compress(float(raw))
        if observe:
            self._history.append(compressed)

        anchors = self._anchors()
        if anchors is None:
            # Nothing to scale against yet: a neutral 0.5 keeps the ensemble
            # well defined without pretending to know where this frame sits.
            return 0.5
        low, high = anchors
        span = high - low
        # Guard against a degenerate range (a static scene, or a stuck sensor).
        floor = self.config.min_range_ratio * max(abs(high), abs(low), 1.0)
        if span <= floor:
            return 0.5
        return float(min(1.0, max(0.0, (compressed - low) / span)))

    def freeze(self) -> None:
        """Lock the current anchors so later calls use a fixed scale."""
        anchors = self._anchors()
        if anchors is None:
            raise RuntimeError(
                f"normalizer {self.name!r} has too few samples ({len(self._history)}) to freeze"
            )
        self._frozen_anchors = anchors
        logger.debug("froze normalizer %s at %s", self.name, anchors)

    def unfreeze(self) -> None:
        self._frozen_anchors = None

    def fit(self, values: Iterable[float]) -> None:
        """Populate the history from a batch of raw values, then freeze."""
        self._history.clear()
        for value in values:
            self._history.append(self._compress(float(value)))
        self.freeze()

    def reset(self) -> None:
        self._history.clear()
        self._frozen_anchors = None


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
