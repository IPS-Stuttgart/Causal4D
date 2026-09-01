"""Reversal-equivariant geometry helpers for the DEFORM decision study."""

from __future__ import annotations

import math

import numpy as np


def middle_directed_tangent(points: np.ndarray) -> np.ndarray:
    """Return a central tangent that changes sign exactly under reversal."""

    value = np.asarray(points, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 3 or value.shape[0] < 4:
        raise ValueError("frame must contain at least four 3-D points")
    left = (value.shape[0] - 1) // 2
    right = value.shape[0] // 2
    tangent = value[right] - value[left]
    planar = tangent[:2]
    norm = float(np.linalg.norm(planar))
    if norm <= 1.0e-9:
        raise ValueError("middle tangent has negligible horizontal projection")
    return planar / norm


def tangent_angle(points: np.ndarray) -> float:
    tangent = middle_directed_tangent(points)
    return float(math.atan2(float(tangent[1]), float(tangent[0])) % (2.0 * math.pi))
