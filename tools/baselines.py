"""Published focus measures and combination schemes, for comparison.

These exist so the project's own scheme can be measured against something other
than itself.  They are deliberately kept out of the library: they are baselines
for the study, not part of the product.

Focus measures
--------------
Implemented in their textbook form, without the contrast normalisation the
library applies to its own six.  That is the point - a baseline that has been
quietly modified is not a baseline.

    GLVA    gray-level variance                       Krotkov (1987)
    NVAR    normalised variance                       Groen et al. (1985)
    LAPE    energy of the Laplacian                   Subbarao et al. (1993)
    SML     sum-modified Laplacian                    Nayar & Nakagawa (1990)
    SMD     sum of modulus differences                Jarvis (1976)
    TENV    variance of the Sobel magnitude           Pertuz et al. (2013)
    VOL4    Vollath's F4 autocorrelation              Vollath (1987)
    VOL5    Vollath's F5 autocorrelation              Vollath (1988)
    DCTR    DCT high-to-low energy ratio              Shen & Chen (2006)

Combination schemes
-------------------
Standard ways of fusing several normalised measures.  Three of them are
**offline**: they need the whole sequence before they can weight anything, which
gives them an advantage no real-time system has.  They are included precisely to
bound what is achievable, and every report marks them as offline.

    mean              arithmetic mean
    median            robust aggregation
    max               most optimistic measure wins
    fixed_weighted    fixed prior weights
    inverse_variance  weight by 1 / variance                  (offline)
    entropy           CRITIC-style objective weighting        (offline)
    pca1              first principal component               (offline)
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

__all__ = [
    "FOCUS_MEASURES",
    "COMBINERS",
    "OFFLINE_COMBINERS",
    "measure_names",
    "combiner_names",
]

_EPS = 1e-12


# ---------------------------------------------------------------------------
# Focus measures. Each takes a float32 greyscale image in [0, 255].
# ---------------------------------------------------------------------------

def glva(gray: np.ndarray) -> float:
    """Gray-level variance (Krotkov 1987)."""
    return float(gray.var())


def nvar(gray: np.ndarray) -> float:
    """Normalised variance (Groen et al. 1985): var / mean."""
    mean = float(gray.mean())
    return float(gray.var()) / max(mean, _EPS)


def lape(gray: np.ndarray) -> float:
    """Energy of the Laplacian (Subbarao et al. 1993)."""
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return float(np.mean(lap * lap))


def sml(gray: np.ndarray, threshold: float = 7.0) -> float:
    """Sum-modified Laplacian (Nayar & Nakagawa 1990).

    Takes absolute values of the two 1-D second derivatives separately, so that
    horizontal and vertical curvature cannot cancel - which is the failure the
    ordinary Laplacian has on a saddle.
    """
    kx = np.array([[-1.0, 2.0, -1.0]], dtype=np.float32)
    ky = kx.T
    mx = np.abs(cv2.filter2D(gray, cv2.CV_32F, kx))
    my = np.abs(cv2.filter2D(gray, cv2.CV_32F, ky))
    modified = mx + my
    return float(np.sum(np.where(modified >= threshold, modified, 0.0)) / modified.size)


def smd(gray: np.ndarray) -> float:
    """Sum of modulus differences (Jarvis 1976)."""
    dh = np.abs(gray[:, 1:] - gray[:, :-1])
    dv = np.abs(gray[1:, :] - gray[:-1, :])
    return float(dh.mean() + dv.mean())


def tenv(gray: np.ndarray) -> float:
    """Variance of the Sobel gradient magnitude (Pertuz et al. 2013)."""
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    return float(magnitude.var())


def vollath4(gray: np.ndarray) -> float:
    """Vollath's F4 autocorrelation measure (1987)."""
    a = gray[:, :-1] * gray[:, 1:]
    b = gray[:, :-2] * gray[:, 2:]
    return float(a[:, :b.shape[1]].mean() - b.mean())


def vollath5(gray: np.ndarray) -> float:
    """Vollath's F5 autocorrelation measure (1988)."""
    product = float((gray[:, :-1] * gray[:, 1:]).mean())
    return product - float(gray.mean()) ** 2


def dct_ratio(gray: np.ndarray, block: int = 8, cutoff: int = 3) -> float:
    """High-to-low DCT energy ratio (after Shen & Chen 2006).

    Block DCT over the image, summing the energy above a diagonal cut-off
    against the total.  Like the project's Fourier metric it is a ratio, so it
    is inherently contrast-invariant.
    """
    h = gray.shape[0] - gray.shape[0] % block
    w = gray.shape[1] - gray.shape[1] % block
    if h < block or w < block:
        return 0.0
    view = gray[:h, :w]
    # Vectorised block DCT: reshape into blocks, transform along both axes.
    blocks = view.reshape(h // block, block, w // block, block).swapaxes(1, 2)
    flat = np.ascontiguousarray(blocks.reshape(-1, block, block))
    total_high = 0.0
    total_all = 0.0
    for index in range(flat.shape[0]):
        coefficients = cv2.dct(flat[index])
        power = coefficients * coefficients
        mask = np.add.outer(np.arange(block), np.arange(block)) >= cutoff
        total_high += float(power[mask].sum())
        total_all += float(power.sum()) - float(power[0, 0])
    return total_high / max(total_all, _EPS)


FOCUS_MEASURES: dict[str, Callable[[np.ndarray], float]] = {
    "GLVA": glva,
    "NVAR": nvar,
    "LAPE": lape,
    "SML": sml,
    "SMD": smd,
    "TENV": tenv,
    "VOL4": vollath4,
    "VOL5": vollath5,
    "DCTR": dct_ratio,
}


def measure_names() -> tuple[str, ...]:
    return tuple(FOCUS_MEASURES)


# ---------------------------------------------------------------------------
# Combination schemes. Each takes an (n_frames, n_metrics) array of normalised
# values in [0, 1] and returns one score per frame.
# ---------------------------------------------------------------------------

def combine_mean(matrix: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    return matrix.mean(axis=1)


def combine_median(matrix: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    return np.median(matrix, axis=1)


def combine_max(matrix: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    return matrix.max(axis=1)


def combine_fixed(matrix: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    if priors is None:
        return combine_mean(matrix)
    weights = priors / max(priors.sum(), _EPS)
    return matrix @ weights


def combine_inverse_variance(
    matrix: np.ndarray, priors: np.ndarray | None = None
) -> np.ndarray:
    """Weight each measure by the inverse of its variance over the sequence.

    OFFLINE: needs the whole recording before it can weight anything.
    """
    variance = matrix.var(axis=0)
    weights = 1.0 / np.maximum(variance, _EPS)
    weights = weights / weights.sum()
    return matrix @ weights


def combine_entropy(matrix: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    """CRITIC-style objective weighting by information content.

    A measure that varies little carries little information and is
    down-weighted.  OFFLINE.
    """
    columns = matrix - matrix.min(axis=0, keepdims=True)
    span = np.maximum(matrix.max(axis=0) - matrix.min(axis=0), _EPS)
    scaled = columns / span
    probability = scaled / np.maximum(scaled.sum(axis=0, keepdims=True), _EPS)
    with np.errstate(divide="ignore", invalid="ignore"):
        entropy = -np.nansum(
            probability * np.log(np.maximum(probability, _EPS)), axis=0
        ) / np.log(max(matrix.shape[0], 2))
    diversity = 1.0 - entropy
    weights = diversity / max(diversity.sum(), _EPS)
    return matrix @ weights


def combine_pca1(matrix: np.ndarray, priors: np.ndarray | None = None) -> np.ndarray:
    """Projection onto the first principal component, rescaled to [0, 1].

    The standard unsupervised way of fusing correlated measurements.  OFFLINE.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return combine_mean(matrix)
    component = vt[0]
    # Orient it so that "more" means "sharper" rather than the reverse.
    if np.sum(component) < 0:
        component = -component
    projected = centred @ component
    span = projected.max() - projected.min()
    if span < _EPS:
        return np.full(matrix.shape[0], 0.5)
    return (projected - projected.min()) / span


COMBINERS: dict[str, Callable[[np.ndarray, np.ndarray | None], np.ndarray]] = {
    "mean": combine_mean,
    "median": combine_median,
    "max": combine_max,
    "fixed_weighted": combine_fixed,
    "inverse_variance": combine_inverse_variance,
    "entropy": combine_entropy,
    "pca1": combine_pca1,
}

#: Schemes that need the whole sequence in advance. Marked in every report,
#: because a real-time system cannot use them.
OFFLINE_COMBINERS: frozenset[str] = frozenset(
    {"inverse_variance", "entropy", "pca1"}
)


def combiner_names() -> tuple[str, ...]:
    return tuple(COMBINERS)
