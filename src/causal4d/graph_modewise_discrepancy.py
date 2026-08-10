"""Conservative mode-wise graph-discrepancy dynamics.

This module is intentionally additive.  It provides a post-freeze diagnostic
alternative to the dense learned graph transition in
:mod:`causal4d.graph_temporal_discrepancy`; it does not alter the registered
real-experiment method or the historical graph-persistence baseline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causal4d.graph_temporal_discrepancy import (
    GraphTemporalDiscrepancyModel,
    project_graph_coefficients,
)
from causal4d.immutable_array import readonly_array


@dataclass(frozen=True)
class ModewiseGraphDynamics:
    """Independent stable AR(1) dynamics for each retained graph mode.

    A single retention coefficient is shared by the three spatial coordinates
    of a graph mode, while innovation variance is estimated separately for each
    coordinate.  ``persistence_prior_weight`` shrinks retention toward one
    without coupling different graph modes.
    """

    retention: np.ndarray
    innovation_variance_m2: np.ndarray
    persistence_prior_weight: float
    minimum_retention: float
    maximum_retention: float
    fit_transition_count: int
    innovation_variance_floor_m2: float

    def __post_init__(self) -> None:
        retention = readonly_array(self.retention, dtype=float)
        innovation = readonly_array(self.innovation_variance_m2, dtype=float)
        if retention.ndim != 1 or len(retention) == 0:
            raise ValueError("retention must be a nonempty one-dimensional array")
        if innovation.shape != (len(retention), 3):
            raise ValueError("innovation_variance_m2 must have shape (rank, 3)")
        if not np.all(np.isfinite(retention)) or not np.all(np.isfinite(innovation)):
            raise ValueError("mode-wise dynamics arrays must be finite")
        if np.any(retention < self.minimum_retention - 1e-12) or np.any(
            retention > self.maximum_retention + 1e-12
        ):
            raise ValueError("retention lies outside the declared bounds")
        if np.any(innovation < 0.0):
            raise ValueError("innovation variance must be nonnegative")
        if not 0.0 <= self.persistence_prior_weight < 1.0:
            raise ValueError("persistence_prior_weight must lie in [0, 1)")
        if (
            not np.isfinite(self.minimum_retention)
            or not np.isfinite(self.maximum_retention)
            or self.minimum_retention < 0.0
            or self.maximum_retention > 1.0
            or self.minimum_retention > self.maximum_retention
        ):
            raise ValueError("retention bounds must satisfy 0 <= min <= max <= 1")
        if self.fit_transition_count < 2:
            raise ValueError("fit_transition_count must be at least two")
        if (
            not np.isfinite(self.innovation_variance_floor_m2)
            or self.innovation_variance_floor_m2 <= 0.0
        ):
            raise ValueError("innovation_variance_floor_m2 must be positive")
        object.__setattr__(self, "retention", retention)
        object.__setattr__(self, "innovation_variance_m2", innovation)

    @property
    def rank(self) -> int:
        return int(len(self.retention))

    def as_dict(self) -> dict[str, object]:
        return {
            "rank": self.rank,
            "retention": self.retention.tolist(),
            "innovation_variance_m2": self.innovation_variance_m2.tolist(),
            "persistence_prior_weight": float(self.persistence_prior_weight),
            "minimum_retention": float(self.minimum_retention),
            "maximum_retention": float(self.maximum_retention),
            "fit_transition_count": int(self.fit_transition_count),
            "innovation_variance_floor_m2": float(
                self.innovation_variance_floor_m2
            ),
        }


def fit_modewise_graph_dynamics(
    coefficients: np.ndarray,
    *,
    persistence_prior_weight: float = 0.25,
    minimum_retention: float = 0.0,
    maximum_retention: float = 1.0,
    innovation_variance_floor_m2: float = 1e-12,
) -> ModewiseGraphDynamics:
    """Fit stable mode-wise AR(1) dynamics with shrinkage toward persistence.

    ``coefficients`` must have shape ``(T, rank, 3)``.  For each graph mode,
    the retention parameter minimizes a pooled three-coordinate squared error
    with a quadratic pseudo-observation at retention one.  The pseudo-observation
    energy is a fixed fraction of the observed source energy, making the
    shrinkage invariant to coefficient scale.
    """

    values = np.asarray(coefficients, dtype=float)
    if values.ndim != 3 or values.shape[2] != 3 or len(values) < 3:
        raise ValueError("coefficients must have shape (T>=3, rank, 3)")
    if values.shape[1] < 1 or not np.all(np.isfinite(values)):
        raise ValueError("coefficients must contain finite graph modes")
    if not 0.0 <= persistence_prior_weight < 1.0:
        raise ValueError("persistence_prior_weight must lie in [0, 1)")
    if (
        not np.isfinite(minimum_retention)
        or not np.isfinite(maximum_retention)
        or minimum_retention < 0.0
        or maximum_retention > 1.0
        or minimum_retention > maximum_retention
    ):
        raise ValueError("retention bounds must satisfy 0 <= min <= max <= 1")
    if (
        not np.isfinite(innovation_variance_floor_m2)
        or innovation_variance_floor_m2 <= 0.0
    ):
        raise ValueError("innovation_variance_floor_m2 must be positive")

    source = values[:-1]
    target = values[1:]
    rank = values.shape[1]
    retention = np.empty(rank, dtype=float)
    innovation = np.empty((rank, 3), dtype=float)
    shrinkage_ratio = (
        persistence_prior_weight / (1.0 - persistence_prior_weight)
        if persistence_prior_weight
        else 0.0
    )

    for mode in range(rank):
        x = source[:, mode, :].reshape(-1)
        y = target[:, mode, :].reshape(-1)
        source_energy = float(np.dot(x, x))
        cross_energy = float(np.dot(x, y))
        pseudo_energy = shrinkage_ratio * source_energy
        denominator = source_energy + pseudo_energy
        if denominator <= np.finfo(float).tiny:
            raw_retention = 1.0
        else:
            raw_retention = (cross_energy + pseudo_energy) / denominator
        rho = float(np.clip(raw_retention, minimum_retention, maximum_retention))
        retention[mode] = rho
        residual = target[:, mode, :] - rho * source[:, mode, :]
        innovation[mode] = (
            np.mean(np.square(residual), axis=0) + innovation_variance_floor_m2
        )

    return ModewiseGraphDynamics(
        retention=retention,
        innovation_variance_m2=innovation,
        persistence_prior_weight=float(persistence_prior_weight),
        minimum_retention=float(minimum_retention),
        maximum_retention=float(maximum_retention),
        fit_transition_count=len(values) - 1,
        innovation_variance_floor_m2=float(innovation_variance_floor_m2),
    )


def fit_modewise_graph_discrepancy(
    model: GraphTemporalDiscrepancyModel,
    residual_m: np.ndarray,
    valid: np.ndarray,
    *,
    node_indices: np.ndarray | Sequence[int] | None = None,
    persistence_prior_weight: float = 0.25,
    minimum_retention: float = 0.0,
    maximum_retention: float = 1.0,
    innovation_variance_floor_m2: float = 1e-12,
) -> ModewiseGraphDynamics:
    """Project source residuals into ``model.basis`` and fit mode-wise dynamics."""

    coefficients = project_graph_coefficients(
        residual_m,
        valid,
        model.basis,
        ridge=model.projection_ridge,
        node_indices=node_indices,
    )
    return fit_modewise_graph_dynamics(
        coefficients,
        persistence_prior_weight=persistence_prior_weight,
        minimum_retention=minimum_retention,
        maximum_retention=maximum_retention,
        innovation_variance_floor_m2=innovation_variance_floor_m2,
    )


def forecast_modewise_graph_discrepancy(
    model: GraphTemporalDiscrepancyModel,
    dynamics: ModewiseGraphDynamics,
    prefix_residual_m: np.ndarray,
    prefix_valid: np.ndarray,
    *,
    total_frame_count: int,
    node_indices: np.ndarray | Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast a graph discrepancy with stable independent mode dynamics.

    The coefficient covariance starts at zero at the last observed prefix frame
    and accumulates the fitted per-mode innovation variance.  Projection
    variance is retained as the irreducible node-coordinate floor.
    """

    if dynamics.rank != model.selected_rank:
        raise ValueError("mode-wise dynamics rank must match the graph model")
    prefix = np.asarray(prefix_residual_m, dtype=float)
    valid = np.asarray(prefix_valid, dtype=bool)
    if prefix.ndim != 3 or prefix.shape[2] != 3:
        raise ValueError("prefix_residual_m must have shape (T, observed_node, 3)")
    if valid.shape != prefix.shape[:2]:
        raise ValueError("prefix_valid must have shape (T, observed_node)")
    if not 2 <= len(prefix) < total_frame_count:
        raise ValueError("prefix must reveal evidence and leave future frames")

    coefficients = project_graph_coefficients(
        prefix,
        valid,
        model.basis,
        ridge=model.projection_ridge,
        node_indices=node_indices,
    )
    node_count = model.basis.shape[0]
    mean = np.zeros((total_frame_count, node_count, 3), dtype=float)
    variance = np.zeros_like(mean)
    mean[: len(prefix)] = np.einsum(
        "nr,trc->tnc",
        model.basis,
        coefficients,
    )
    variance[: len(prefix)] = model.projection_variance_m2[None, None, :]

    current = coefficients[-1].copy()
    coefficient_variance = np.zeros((model.selected_rank, 3), dtype=float)
    squared_basis = np.square(model.basis)
    for frame in range(len(prefix), total_frame_count):
        current = dynamics.retention[:, None] * current
        coefficient_variance = (
            np.square(dynamics.retention)[:, None] * coefficient_variance
            + dynamics.innovation_variance_m2
        )
        mean[frame] = model.basis @ current
        marginal = squared_basis @ coefficient_variance
        variance[frame] = marginal + model.projection_variance_m2[None, :]

    return mean, np.maximum(variance, 0.0)
