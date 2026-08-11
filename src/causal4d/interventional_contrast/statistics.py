"""One-dimensional mixture summaries for contrast queries."""

from __future__ import annotations

import numpy as np
from scipy.special import ndtr


def mixture_probability_positive(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> float:
    positive_variance = variances > 0.0
    probabilities = np.zeros_like(means, dtype=float)
    probabilities[~positive_variance] = means[~positive_variance] > 0.0
    if np.any(positive_variance):
        probabilities[positive_variance] = ndtr(
            means[positive_variance] / np.sqrt(variances[positive_variance])
        )
    return float(np.dot(weights, probabilities))


def mixture_cdf(
    value: float,
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
) -> float:
    positive_variance = variances > 0.0
    probabilities = np.empty_like(means, dtype=float)
    probabilities[~positive_variance] = means[~positive_variance] <= value
    if np.any(positive_variance):
        probabilities[positive_variance] = ndtr(
            (value - means[positive_variance])
            / np.sqrt(variances[positive_variance])
        )
    return float(np.dot(weights, probabilities))


def mixture_quantile(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    if not 0.0 < probability < 1.0:
        raise ValueError("mixture quantile probability must lie in (0, 1)")
    if not np.any(variances > 0.0):
        order = np.argsort(means, kind="stable")
        cumulative = np.cumsum(weights[order])
        index = int(np.searchsorted(cumulative, probability, side="left"))
        return float(means[order[min(index, len(order) - 1)]])
    standard_deviation = np.sqrt(variances)
    lower = float(np.min(means - 12.0 * standard_deviation))
    upper = float(np.max(means + 12.0 * standard_deviation))
    span = max(upper - lower, 1.0)
    while mixture_cdf(lower, means, variances, weights) >= probability:
        lower -= span
        span *= 2.0
    span = max(upper - lower, 1.0)
    while mixture_cdf(upper, means, variances, weights) < probability:
        upper += span
        span *= 2.0
    for _ in range(100):
        midpoint = 0.5 * (lower + upper)
        if mixture_cdf(midpoint, means, variances, weights) >= probability:
            upper = midpoint
        else:
            lower = midpoint
    return float(upper)
