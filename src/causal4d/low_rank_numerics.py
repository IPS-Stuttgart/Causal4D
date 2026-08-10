"""Numerical guards shared by exact low-rank covariance updates."""

from __future__ import annotations

from typing import Final

import numpy as np


_ROUNDOFF_MULTIPLIER: Final = 256.0


def nonnegative_woodbury_quadratic(
    base_quadratic: np.ndarray,
    correction_quadratic: np.ndarray,
    *,
    dimension: int,
    name: str = "Woodbury quadratic",
) -> np.ndarray:
    """Subtract a Woodbury correction while rejecting material cancellation.

    Analytically, ``base_quadratic - correction_quadratic`` is nonnegative.
    Tiny negative values can arise from floating-point roundoff and are clipped
    to zero. A negative value beyond a dimension- and scale-aware tolerance
    indicates numerical breakdown and fails closed instead of silently becoming
    a valid likelihood. The tolerance scales with the observation dimension,
    because that controls accumulated triangular-solve and dot-product error.
    """

    if type(dimension) is not int or dimension < 1:
        raise ValueError("dimension must be a positive integer")
    if type(name) is not str or not name:
        raise ValueError("name must be a nonempty string")

    base = np.asarray(base_quadratic, dtype=float)
    correction = np.asarray(correction_quadratic, dtype=float)
    try:
        base, correction = np.broadcast_arrays(base, correction)
    except ValueError as error:
        raise ValueError(
            "Woodbury quadratic terms must be broadcast-compatible"
        ) from error
    if not np.all(np.isfinite(base)) or not np.all(np.isfinite(correction)):
        raise ValueError("Woodbury quadratic terms must be finite")
    if np.any(base < 0.0) or np.any(correction < 0.0):
        raise ValueError("Woodbury quadratic terms must be nonnegative")

    difference = base - correction
    scale = np.maximum(1.0, np.maximum(np.abs(base), np.abs(correction)))
    tolerance = _ROUNDOFF_MULTIPLIER * np.finfo(float).eps * dimension * scale
    invalid = difference < -tolerance
    if np.any(invalid):
        normalized = np.where(invalid, difference / tolerance, np.inf)
        worst = int(np.argmin(normalized))
        worst_difference = float(difference.flat[worst])
        worst_tolerance = float(tolerance.flat[worst])
        raise FloatingPointError(
            f"{name} became negative beyond roundoff: "
            f"difference={worst_difference:.6e}, "
            f"tolerance={worst_tolerance:.6e}, dimension={dimension}"
        )
    return np.maximum(difference, 0.0)


__all__ = ["nonnegative_woodbury_quadratic"]
