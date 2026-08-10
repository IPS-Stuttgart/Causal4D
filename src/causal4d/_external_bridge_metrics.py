"""Numerical helpers for the external forecast/rollout bridge."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal4d.external_bridge import _fractional_frame_indices
from causal4d.external_reference import ExternalReferenceTrajectory


def _node_indices(
    node_ids: np.ndarray,
    requested_ids: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    lookup = {int(node_id): index for index, node_id in enumerate(node_ids)}
    missing = [int(node_id) for node_id in requested_ids if int(node_id) not in lookup]
    if missing:
        raise ValueError(f"{name} is missing node IDs: {missing!r}")
    return np.asarray(
        [lookup[int(node_id)] for node_id in requested_ids],
        dtype=np.int64,
    )


def _interpolate_reference(
    reference: ExternalReferenceTrajectory,
    query_times_s: np.ndarray,
    node_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame_indices = _fractional_frame_indices(reference.frame_times_s, query_times_s)
    lower = np.floor(frame_indices).astype(np.int64)
    upper = np.ceil(frame_indices).astype(np.int64)
    alpha = (frame_indices - lower).reshape(-1, 1, 1)
    lower_positions = reference.positions_m[lower][:, node_indices]
    upper_positions = reference.positions_m[upper][:, node_indices]
    positions = (1.0 - alpha) * lower_positions + alpha * upper_positions
    lower_valid = reference.validity[lower][:, node_indices]
    upper_valid = reference.validity[upper][:, node_indices]
    valid = lower_valid & upper_valid
    positions[~valid] = np.nan
    return positions, valid, frame_indices


def _weighted_query_mean(components: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.einsum("hp,hpfqc->fqc", weights, components, optimize=True)


def _weighted_quantile(
    components: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> np.ndarray:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("quantile probability must lie in [0, 1]")
    values = components.reshape(-1, *components.shape[2:]).astype(np.float64)
    flat_weights = weights.reshape(-1).astype(np.float64)
    order = np.argsort(values, axis=0, kind="mergesort")
    sorted_values = np.take_along_axis(values, order, axis=0)
    broadcast_weights = np.broadcast_to(
        flat_weights.reshape(-1, 1, 1, 1), values.shape
    )
    sorted_weights = np.take_along_axis(broadcast_weights, order, axis=0)
    cumulative = np.cumsum(sorted_weights, axis=0)
    cumulative[-1] = 1.0
    indices = np.argmax(cumulative >= probability, axis=0)
    return np.take_along_axis(sorted_values, indices[None], axis=0)[0]


def _trajectory_metrics(
    prediction: np.ndarray,
    truth: np.ndarray,
    truth_valid: np.ndarray,
    *,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> dict[str, Any]:
    predicted = np.asarray(prediction, dtype=np.float64)
    reference = np.asarray(truth, dtype=np.float64)
    coordinate_valid = np.asarray(truth_valid, dtype=bool) & np.isfinite(predicted)
    point_valid = np.all(coordinate_valid, axis=2)
    if not np.any(point_valid):
        raise ValueError("trajectory metric has no valid point-time pairs")
    point_error = np.linalg.norm(predicted - reference, axis=2)
    frame_ade = np.full(len(predicted), np.nan, dtype=np.float64)
    for frame in range(len(predicted)):
        if np.any(point_valid[frame]):
            frame_ade[frame] = float(np.mean(point_error[frame, point_valid[frame]]))
    final_frame = int(np.flatnonzero(np.any(point_valid, axis=1))[-1])
    valid_coordinates = coordinate_valid
    metrics: dict[str, Any] = {
        "ade_m": float(np.mean(point_error[point_valid])),
        "fde_m": float(np.mean(point_error[final_frame, point_valid[final_frame]])),
        "coordinate_rmse_m": float(
            np.sqrt(np.mean(np.square((predicted - reference)[valid_coordinates])))
        ),
        "valid_point_time_count": int(np.sum(point_valid)),
        "valid_coordinate_count": int(np.sum(valid_coordinates)),
        "final_valid_frame_index": final_frame,
        "coordinate_coverage": None,
        "frame_ade_m": [
            float(value) if np.isfinite(value) else None for value in frame_ade
        ],
    }
    if lower is not None or upper is not None:
        if lower is None or upper is None:
            raise ValueError("both interval bounds must be supplied")
        lower_values = np.asarray(lower, dtype=np.float64)
        upper_values = np.asarray(upper, dtype=np.float64)
        if (
            lower_values.shape != reference.shape
            or upper_values.shape != reference.shape
        ):
            raise ValueError("interval bounds must match reference shape")
        covered = (reference >= lower_values) & (reference <= upper_values)
        metrics["coordinate_coverage"] = float(np.mean(covered[valid_coordinates]))
    return metrics


def _weight_diagnostics(
    posterior: np.ndarray,
    prior: np.ndarray,
    hypothesis_ids: tuple[str, ...],
) -> dict[str, Any]:
    flat = posterior.reshape(-1)
    flat_prior = prior.reshape(-1)
    positive = flat > 0.0
    if np.any(positive & (flat_prior <= 0.0)):
        raise RuntimeError("posterior placed mass outside prior support")
    kl = float(
        np.sum(flat[positive] * np.log(flat[positive] / flat_prior[positive]))
    )
    best_flat = int(np.argmax(flat))
    hypothesis_index, particle_index = np.unravel_index(best_flat, posterior.shape)
    return {
        "effective_component_count": float(1.0 / np.sum(np.square(flat))),
        "kl_from_physical_prior": kl,
        "l1_weight_shift": float(np.sum(np.abs(flat - flat_prior))),
        "maximum_component_weight": float(flat[best_flat]),
        "top_hypothesis_id": hypothesis_ids[hypothesis_index],
        "top_hypothesis_index": int(hypothesis_index),
        "top_parameter_particle_index": int(particle_index),
        "weights_bit_identical_to_prior": bool(
            posterior.tobytes() == prior.tobytes()
        ),
    }


def _constant_velocity_prediction(
    reference: ExternalReferenceTrajectory,
    query_times_s: np.ndarray,
    reference_node_indices: np.ndarray,
    anchor_time_s: float,
) -> np.ndarray | None:
    available = np.flatnonzero(reference.frame_times_s <= anchor_time_s + 1e-12)
    if len(available) < 2:
        return None
    last = int(available[-1])
    previous = int(available[-2])
    delta_t = float(reference.frame_times_s[last] - reference.frame_times_s[previous])
    if delta_t <= 0.0:
        return None
    valid = (
        reference.validity[last, reference_node_indices]
        & reference.validity[previous, reference_node_indices]
    )
    previous_positions = reference.positions_m[previous, reference_node_indices]
    last_positions = reference.positions_m[last, reference_node_indices]
    velocity = (last_positions - previous_positions) / delta_t
    prediction = last_positions[None] + (
        query_times_s - reference.frame_times_s[last]
    )[:, None, None] * velocity[None]
    prediction[:, ~np.all(valid, axis=1)] = np.nan
    return prediction


__all__ = [
    "_constant_velocity_prediction",
    "_interpolate_reference",
    "_node_indices",
    "_trajectory_metrics",
    "_weight_diagnostics",
    "_weighted_quantile",
    "_weighted_query_mean",
]
