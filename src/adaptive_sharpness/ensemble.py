"""Adaptive ensemble combination of the individual sharpness metrics.

The combination is a weighted sum ``S = sum_i w_i * s_i`` with ``sum_i w_i = 1``
over the normalised metrics ``s_i``.  What makes it adaptive is how ``w_i`` is
built, in three multiplicative stages.

1. Content-conditioned reliability
----------------------------------
Each metric carries a vector of sensitivity coefficients ``kappa_ij >= 0``
saying how fast it degrades under condition ``j`` (noise, missing edges,
clipping, motion, low contrast).  Given the measured degradation factors
``d_j in [0, 1]`` from :mod:`sharpness.analysis`,

    r_i = exp( - sum_j kappa_ij * d_j )

This is a log-linear model: independent degradations multiply, every term stays
in ``(0, 1]``, and no condition can drive a weight negative.  The coefficients
are interpretable one at a time — ``kappa`` for the Laplacian under noise is the
largest because a second-derivative operator amplifies white noise more than any
first-derivative one, and ``kappa`` for the edge-width metric under missing
edges is the largest overall because that metric is undefined without edges.

2. Consensus agreement
----------------------
Metrics that disagree with the consensus of the others are down-weighted with a
Gaussian kernel around the reliability-weighted median ``s_med``:

    a_i = exp( -0.5 * ( (s_i - s_med) / (c * sigma_r) )^2 )

with ``sigma_r`` a robust (MAD) spread of the normalised scores, floored so that
unanimous agreement does not blow the expression up.  This is one step of
iteratively reweighted robust estimation, and it is what protects the score when
a single metric is fooled by, say, a specular highlight.

3. Prior and renormalisation
----------------------------
    w_i = normalise( max(p_i * r_i * a_i, floor) )

The floor keeps every metric marginally alive, so the ensemble can recover if
conditions improve, and guarantees a well-defined score even if every
reliability collapses.

Confidence
----------
The confidence is reported separately from the score and never modifies it.  It
is the geometric mean of six factors in ``[0, 1]`` — edge sufficiency, SNR,
exposure, contrast, inter-metric concordance and motion — so that any single
collapsed factor pulls the confidence down proportionally.  A low confidence
means "this score is a weak measurement", which is exactly what an autofocus
search loop needs in order to decide whether to trust a step.

Scope note
----------
The individual metrics are classical and long-established.  The contribution
here is the *combination* — the reliability model, the agreement stage, the
confidence and the gated temporal filter.  See docs/ALGORITHM.md for what is
known, what is proposed, and what still has to be validated against the
literature.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from .analysis import DEGRADATION_KEYS
from .config import EnsembleConfig
from .types import ImageStats

logger = logging.getLogger(__name__)

__all__ = ["EnsembleOutput", "AdaptiveEnsemble", "NECESSARY_FACTORS"]

#: Confidence factors without which the score means nothing at all, as opposed
#: to merely meaning less.  Only these are subject to the weakest-link cap.
NECESSARY_FACTORS: frozenset[str] = frozenset(
    {"edges", "contrast", "warmup", "resolution"}
)


@dataclass(frozen=True)
class EnsembleOutput:
    """Everything the ensemble produced for one frame."""

    score: float
    confidence: float
    weights: Mapping[str, float]
    reliabilities: Mapping[str, float]
    agreements: Mapping[str, float]
    confidence_components: Mapping[str, float] = field(default_factory=dict)
    # Robust spread of the normalised metrics; large = the metrics disagree.
    dispersion: float = 0.0


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median of ``values`` under ``weights`` (both 1-D, weights non-negative)."""
    if values.size == 1:
        return float(values[0])
    total = float(weights.sum())
    if total <= 0.0:
        return float(np.median(values))
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order]) / total
    index = int(np.searchsorted(cumulative, 0.5, side="left"))
    index = min(index, sorted_values.size - 1)
    return float(sorted_values[index])


class AdaptiveEnsemble:
    """Combines normalised metrics into a score plus a confidence."""

    def __init__(self, names: Sequence[str], config: EnsembleConfig, priors: Mapping[str, float]):
        if not names:
            raise ValueError("the ensemble needs at least one metric")
        self.names = tuple(names)
        self.config = config

        missing = [n for n in self.names if n not in config.sensitivity]
        if missing:
            raise KeyError(
                f"no sensitivity profile configured for metric(s): {', '.join(missing)}"
            )
        for name in self.names:
            unknown = set(config.sensitivity[name]) - set(DEGRADATION_KEYS)
            if unknown:
                raise KeyError(
                    f"metric {name!r} has unknown degradation key(s): {', '.join(sorted(unknown))}"
                )

        self._priors = np.array(
            [max(0.0, float(priors.get(name, 1.0))) for name in self.names],
            dtype=np.float64,
        )
        if not np.any(self._priors > 0.0):
            raise ValueError("at least one metric must have a positive prior weight")

        # Pre-materialise the kappa matrix: metrics x degradations.
        self._kappa = np.array(
            [
                [float(config.sensitivity[name].get(key, 0.0)) for key in DEGRADATION_KEYS]
                for name in self.names
            ],
            dtype=np.float64,
        )
        if np.any(self._kappa < 0.0):
            raise ValueError("sensitivity coefficients must be non-negative")

        # Indices of the degradation channels that the ablation switches control.
        self._noise_idx = DEGRADATION_KEYS.index("noise")
        self._motion_idx = DEGRADATION_KEYS.index("motion")

    # -- stages ----------------------------------------------------------------

    def reliabilities(self, degradations: Mapping[str, float]) -> np.ndarray:
        """Stage 1: ``r_i = exp(-sum_j kappa_ij d_j)``."""
        d = np.array(
            [float(degradations.get(key, 0.0)) for key in DEGRADATION_KEYS],
            dtype=np.float64,
        )
        np.clip(d, 0.0, 1.0, out=d)
        if not self.config.use_noise_compensation:
            d[self._noise_idx] = 0.0
        if not self.config.use_motion_compensation:
            d[self._motion_idx] = 0.0
        return np.exp(-(self._kappa @ d))

    def agreements(self, scores: np.ndarray, reliabilities: np.ndarray) -> tuple[np.ndarray, float]:
        """Stage 2: consensus kernel. Returns ``(a_i, dispersion)``.

        The dispersion is measured whatever ``use_agreement`` says, because it
        is two different mechanisms sharing one calculation: the kernel
        reweights the metrics, while the dispersion feeds the *concordance*
        factor of the confidence.  Switching off the reweighting must not also
        blind the confidence to metrics that disagree.
        """
        if scores.size == 1:
            return np.ones_like(scores), 0.0
        median = _weighted_median(scores, reliabilities)
        deviations = np.abs(scores - median)
        # 1.4826 * MAD is the consistent Gaussian sigma estimate.
        sigma = 1.4826 * float(np.median(deviations))
        dispersion = sigma
        # Floor: with perfect agreement sigma -> 0 and the kernel would divide
        # by zero; the floor also stops trivial noise from looking like dissent.
        if not self.config.use_agreement:
            return np.ones_like(scores), dispersion
        sigma = max(sigma, 0.02)
        scale = max(self.config.agreement_scale, 1e-6) * sigma
        return np.exp(-0.5 * np.square(deviations / scale)), dispersion

    def weights(self, reliabilities: np.ndarray, agreements: np.ndarray) -> np.ndarray:
        """Stage 3: prior, floor and renormalisation to a unit sum."""
        raw = self._priors * reliabilities * agreements
        floor = max(0.0, self.config.weight_floor)
        if floor > 0.0:
            raw = np.maximum(raw, floor)
        total = float(raw.sum())
        if total <= 0.0:
            # Degenerate but recoverable: fall back to a uniform ensemble.
            logger.warning("all ensemble weights collapsed to zero; using uniform weights")
            return np.full_like(raw, 1.0 / raw.size)
        return raw / total

    def confidence(
        self,
        stats: ImageStats,
        degradations: Mapping[str, float],
        dispersion: float,
        warmed_up: bool,
        informative: float = 1.0,
    ) -> tuple[float, dict[str, float]]:
        """Confidence in the reported score, and its individual factors.

        ``informative`` is the fraction of metrics whose normalisation is not
        pinned at its neutral value.  It matters because a completely static
        scene drives every metric to 0.5, which the concordance factor would
        otherwise read as perfect agreement and reward with high confidence.
        """
        cfg = self.config
        components = {
            "edges": float(stats.edge_sufficiency),
            "snr": float(min(1.0, stats.snr / max(cfg.confidence_snr_ref, 1e-9))),
            "exposure": float(1.0 - min(1.0, degradations.get("clip", 0.0))),
            "contrast": float(1.0 - min(1.0, degradations.get("contrast", 0.0))),
            "concordance": float(
                1.0 - min(1.0, dispersion / max(cfg.confidence_dispersion_ref, 1e-9))
            ),
            "motion": float(1.0 - min(1.0, degradations.get("motion", 0.0))),
        }
        if not warmed_up:
            # Before the normalisers have a scale, the score is not meaningful.
            components["warmup"] = cfg.warmup_confidence
        if informative < 1.0:
            components["resolution"] = max(0.0, min(1.0, informative))

        # Geometric mean, with a small floor so that one weak factor does not
        # erase the information carried by the others.
        floor = 1e-3
        log_sum = sum(math.log(max(v, floor)) for v in components.values())
        geometric = math.exp(log_sum / len(components))

        # The weakest-link cap applies only to the *necessary* factors.  With no
        # edges in the frame there is nothing whose sharpness could be measured,
        # however good the exposure is, and the geometric mean alone does not
        # express that - it bottoms out around 0.32 even when a factor is zero.
        #
        # The soft factors must NOT be capped this way.  Motion, exposure, SNR
        # and concordance degrade a measurement without making it impossible,
        # and a hard cap on them turns a saturated factor into exactly zero
        # confidence.  Measured on a 1288-frame handheld recording, that fired
        # on 39% of frames - 97% of them purely because the motion factor had
        # saturated - while the edges, SNR, exposure and contrast were all fine.
        necessary = [
            value for name, value in components.items() if name in NECESSARY_FACTORS
        ]
        weakest = min(necessary) if necessary else 1.0
        confidence = min(geometric, math.sqrt(max(0.0, weakest)))
        return float(min(1.0, max(0.0, confidence))), components

    # -- public API ------------------------------------------------------------

    def combine(
        self,
        normalized: Mapping[str, float],
        stats: ImageStats,
        degradations: Mapping[str, float],
        warmed_up: bool = True,
        informative: float = 1.0,
    ) -> EnsembleOutput:
        """Run all three stages and produce the score and confidence."""
        missing = [n for n in self.names if n not in normalized]
        if missing:
            raise KeyError(f"missing normalised value(s) for: {', '.join(missing)}")

        scores = np.array([float(normalized[n]) for n in self.names], dtype=np.float64)
        np.clip(scores, 0.0, 1.0, out=scores)

        reliabilities = self.reliabilities(degradations)
        agreements, dispersion = self.agreements(scores, reliabilities)
        weights = self.weights(reliabilities, agreements)
        score = float(np.dot(weights, scores))
        confidence, components = self.confidence(
            stats, degradations, dispersion, warmed_up, informative
        )

        return EnsembleOutput(
            score=min(1.0, max(0.0, score)),
            confidence=confidence,
            weights=dict(zip(self.names, weights.tolist())),
            reliabilities=dict(zip(self.names, reliabilities.tolist())),
            agreements=dict(zip(self.names, agreements.tolist())),
            confidence_components=components,
            dispersion=float(dispersion),
        )
