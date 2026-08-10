"""Preflight validation for sparse external forecasts and rollout banks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from causal4d.external_forecast import ExternalForecastBundle
from causal4d.external_rollout import ExternalRolloutBundle
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.rollout_bank import SparseTrajectoryEvidence

BRIDGE_DOCTOR_SCHEMA = "causal4d.external_forecast_rollout_doctor"
BRIDGE_DOCTOR_SCHEMA_VERSION = 1


def _fractional_frame_indices(
    frame_times_s: np.ndarray,
    query_times_s: np.ndarray,
) -> np.ndarray:
    times = np.asarray(frame_times_s, dtype=np.float64)
    query = np.asarray(query_times_s, dtype=np.float64)
    if times.ndim != 1 or query.ndim != 1 or not len(query):
        raise ValueError("frame and query times must be nonempty vectors")
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(query)):
        raise ValueError("frame and query times must be finite")
    if np.any(np.diff(times) <= 0.0) or np.any(np.diff(query) <= 0.0):
        raise ValueError("frame and query times must be strictly increasing")
    tolerance = 1e-12
    if query[0] < times[0] - tolerance or query[-1] > times[-1] + tolerance:
        raise ValueError("forecast times exceed rollout-bank time support")
    query = np.clip(query, times[0], times[-1])
    return np.interp(query, times, np.arange(len(times), dtype=np.float64))


def _interpolate_components(
    trajectories: np.ndarray,
    frame_indices: np.ndarray,
    node_indices: np.ndarray,
) -> np.ndarray:
    lower = np.floor(frame_indices).astype(np.int64)
    upper = np.ceil(frame_indices).astype(np.int64)
    alpha = (frame_indices - lower).reshape(1, 1, -1, 1, 1)
    selected_lower = trajectories[:, :, lower][:, :, :, node_indices]
    selected_upper = trajectories[:, :, upper][:, :, :, node_indices]
    return (1.0 - alpha) * selected_lower + alpha * selected_upper


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    flattened_weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    return float(np.sum(flattened * flattened_weights))


def _coordinate_motion_rms(
    displacement: np.ndarray,
    valid: np.ndarray,
) -> float:
    values = np.asarray(displacement, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if values.shape != mask.shape or not np.any(mask):
        raise ValueError("motion RMS requires matching nonempty valid coordinates")
    return float(np.sqrt(np.mean(np.square(values[mask]))))


def build_external_bridge_report(
    forecast: ExternalForecastBundle,
    forecast_id: str,
    rollouts: ExternalRolloutBundle,
    *,
    anchor_tolerance_m: float = 0.01,
    motion_ratio_min: float = 0.10,
    motion_ratio_max: float = 10.0,
    scale_m: float = 0.05,
    degrees_of_freedom: float = 3.0,
    metadata: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Validate identity, timing, scale, and exact fallback before reweighting."""

    if not np.isfinite(anchor_tolerance_m) or anchor_tolerance_m <= 0.0:
        raise ValueError("anchor_tolerance_m must be finite and positive")
    if (
        not np.isfinite(motion_ratio_min)
        or not np.isfinite(motion_ratio_max)
        or motion_ratio_min <= 0.0
        or motion_ratio_max <= motion_ratio_min
    ):
        raise ValueError("motion-ratio bounds must be finite, positive, and ordered")
    if forecast.case_id != rollouts.case_id:
        raise ValueError("external forecast and rollout bank case IDs differ")
    if forecast.future_times_s is None:
        raise ValueError(
            "external forecast must retain future_times_s for rollout-bank bridging"
        )
    forecast_index = forecast.forecast_index(forecast_id)
    if rollouts.bank.coordinate_count != 3:
        raise ValueError("external forecast bridge currently requires 3-D rollouts")

    node_lookup = {
        int(node_id): index for index, node_id in enumerate(rollouts.node_ids)
    }
    missing_nodes = [
        int(node_id)
        for node_id in forecast.node_indices
        if int(node_id) not in node_lookup
    ]
    if missing_nodes:
        raise ValueError(
            "forecast node IDs are absent from rollout bank: " + repr(missing_nodes)
        )
    internal_nodes = np.asarray(
        [node_lookup[int(node_id)] for node_id in forecast.node_indices],
        dtype=np.int64,
    )

    query_times = rollouts.anchor_time_s + forecast.future_times_s
    query_frame_indices = _fractional_frame_indices(
        rollouts.frame_times_s,
        query_times,
    )
    target = forecast.future_positions_m[forecast_index]
    valid = forecast.coordinate_validity[forecast_index]
    evidence = SparseTrajectoryEvidence(
        positions_m=target,
        node_indices=internal_nodes,
        rollout_frame_indices=query_frame_indices,
        scale_m=scale_m,
        degrees_of_freedom=degrees_of_freedom,
        likelihood_weight=0.0,
        compare_displacements=True,
        anchor_positions_m=forecast.anchor_positions_m,
        anchor_rollout_frame=rollouts.anchor_frame_index,
        valid=valid,
        source=f"ExternalForecast:{forecast.artifact_id}:{forecast_id}",
    )
    prior = rollouts.bank.prior_joint_weights
    fallback = rollouts.bank.update_from_sparse_evidence(evidence)
    beta_zero_identical = fallback.tobytes() == prior.tobytes()
    if not beta_zero_identical:
        raise RuntimeError("zero semantic weight did not preserve rollout weights")

    anchor_components = rollouts.bank.trajectories[
        :, :, rollouts.anchor_frame_index, internal_nodes
    ].astype(np.float64)
    anchor_error = np.linalg.norm(
        anchor_components - forecast.anchor_positions_m[None, None],
        axis=-1,
    )
    anchor_component_rms = np.sqrt(np.mean(np.square(anchor_error), axis=-1))
    best_anchor_rms = float(np.min(anchor_component_rms))
    maximum_anchor_rms = float(np.max(anchor_component_rms))
    weighted_anchor_rms = _weighted_mean(anchor_component_rms, prior)

    future_components = _interpolate_components(
        rollouts.bank.trajectories.astype(np.float64),
        query_frame_indices,
        internal_nodes,
    )
    rollout_displacement = future_components - anchor_components[:, :, None]
    rollout_component_rms = np.sqrt(
        np.mean(np.square(rollout_displacement), axis=(2, 3, 4))
    )
    weighted_rollout_rms = _weighted_mean(rollout_component_rms, prior)
    forecast_displacement = target - forecast.anchor_positions_m[None]
    forecast_motion_rms = _coordinate_motion_rms(forecast_displacement, valid)
    motion_ratio = (
        forecast_motion_rms / weighted_rollout_rms
        if weighted_rollout_rms > 0.0
        else None
    )

    warnings: list[str] = []
    if best_anchor_rms > anchor_tolerance_m:
        warnings.append(
            "no rollout component matches the forecast anchor within the configured "
            "tolerance"
        )
    elif maximum_anchor_rms > anchor_tolerance_m:
        warnings.append(
            "some rollout components exceed the configured anchor tolerance"
        )
    if motion_ratio is None:
        warnings.append("rollout-bank motion scale is zero")
    elif not motion_ratio_min <= motion_ratio <= motion_ratio_max:
        warnings.append(
            "forecast motion scale lies outside the configured rollout-bank ratio"
        )

    report = {
        "schema": BRIDGE_DOCTOR_SCHEMA,
        "schema_version": BRIDGE_DOCTOR_SCHEMA_VERSION,
        "valid": True,
        "case_id": forecast.case_id,
        "forecast_artifact_id": forecast.artifact_id,
        "forecast_id": forecast_id,
        "rollout_artifact_id": rollouts.artifact_id,
        "rollout_bank_artifact_id": rollouts.bank.artifact_id,
        "matched_node_ids": forecast.node_indices.tolist(),
        "matched_node_count": len(forecast.node_indices),
        "forecast_future_times_s": forecast.future_times_s.tolist(),
        "rollout_query_times_s": query_times.tolist(),
        "rollout_fractional_frame_indices": query_frame_indices.tolist(),
        "valid_coordinate_fraction": float(np.mean(valid)),
        "anchor": {
            "configured_tolerance_m": float(anchor_tolerance_m),
            "best_component_rms_m": best_anchor_rms,
            "prior_weighted_rms_m": weighted_anchor_rms,
            "maximum_component_rms_m": maximum_anchor_rms,
        },
        "motion": {
            "forecast_coordinate_rms_m": forecast_motion_rms,
            "prior_weighted_rollout_coordinate_rms_m": weighted_rollout_rms,
            "forecast_to_rollout_ratio": motion_ratio,
            "configured_ratio_min": float(motion_ratio_min),
            "configured_ratio_max": float(motion_ratio_max),
        },
        "beta_zero_weights_bit_identical": beta_zero_identical,
        "warnings": warnings,
        "metadata": plain_json(
            validated_json_mapping(
                {} if metadata is None else metadata,
                error_message="bridge report metadata must be finite JSON data",
            )
        ),
    }
    return validated_json_mapping(
        report,
        error_message="bridge doctor report must be finite JSON data",
    )


__all__ = [
    "BRIDGE_DOCTOR_SCHEMA",
    "BRIDGE_DOCTOR_SCHEMA_VERSION",
    "build_external_bridge_report",
]
