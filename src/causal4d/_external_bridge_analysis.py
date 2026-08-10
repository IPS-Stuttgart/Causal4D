"""Finite-support analysis for external forecast/rollout bridges."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from causal4d._external_bridge_metrics import (
    _constant_velocity_prediction,
    _interpolate_reference,
    _node_indices,
    _trajectory_metrics,
    _weight_diagnostics,
    _weighted_quantile,
    _weighted_query_mean,
)
from causal4d.external_bridge import (
    _fractional_frame_indices,
    _interpolate_components,
    build_external_bridge_report,
)
from causal4d.external_forecast import ExternalForecastBundle
from causal4d.external_reference import ExternalReferenceTrajectory
from causal4d.external_rollout import ExternalRolloutBundle
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.rollout_bank import SparseTrajectoryEvidence

EXTERNAL_BRIDGE_RUN_SCHEMA = "causal4d.external_forecast_rollout_run"
EXTERNAL_BRIDGE_RUN_SCHEMA_VERSION = 1


def _beta_values(values: Sequence[float]) -> tuple[float, ...]:
    result = tuple(sorted(set(float(value) for value in values)))
    if not result or result[0] != 0.0:
        raise ValueError("beta values must be nonnegative and include zero")
    if any(not np.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("beta values must be finite and nonnegative")
    return result


def analyze_external_bridge(
    forecast: ExternalForecastBundle,
    forecast_id: str,
    rollouts: ExternalRolloutBundle,
    *,
    beta_values: Sequence[float] = (0.0, 1.0, 3.0, 6.0, 12.0),
    scale_m: float = 0.05,
    degrees_of_freedom: float = 3.0,
    anchor_tolerance_m: float = 0.01,
    motion_ratio_min: float = 0.10,
    motion_ratio_max: float = 10.0,
    reference: ExternalReferenceTrajectory | None = None,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray]]:
    """Run a beta sweep while keeping semantic evidence outside physical state."""

    betas = _beta_values(beta_values)
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("scale_m must be finite and positive")
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise ValueError("degrees_of_freedom must be finite and positive")
    doctor = build_external_bridge_report(
        forecast,
        forecast_id,
        rollouts,
        anchor_tolerance_m=anchor_tolerance_m,
        motion_ratio_min=motion_ratio_min,
        motion_ratio_max=motion_ratio_max,
        scale_m=scale_m,
        degrees_of_freedom=degrees_of_freedom,
    )
    forecast_index = forecast.forecast_index(forecast_id)
    if forecast.future_times_s is None:
        raise ValueError("bridge run requires explicit forecast future_times_s")
    internal_nodes = _node_indices(
        rollouts.node_ids,
        forecast.node_indices,
        name="rollout bank",
    )
    query_times = rollouts.anchor_time_s + forecast.future_times_s
    query_frames = _fractional_frame_indices(rollouts.frame_times_s, query_times)
    components = _interpolate_components(
        rollouts.bank.trajectories.astype(np.float64),
        query_frames,
        internal_nodes,
    )
    target = forecast.future_positions_m[forecast_index]
    forecast_valid = forecast.coordinate_validity[forecast_index]
    prior = rollouts.bank.prior_joint_weights
    evidence_base = dict(
        positions_m=target,
        node_indices=internal_nodes,
        rollout_frame_indices=query_frames,
        scale_m=float(scale_m),
        degrees_of_freedom=float(degrees_of_freedom),
        compare_displacements=True,
        anchor_positions_m=forecast.anchor_positions_m,
        anchor_rollout_frame=rollouts.anchor_frame_index,
        valid=forecast_valid,
        source=f"ExternalForecast:{forecast.artifact_id}:{forecast_id}",
    )

    weights_by_beta: list[np.ndarray] = []
    means_by_beta: list[np.ndarray] = []
    lower_by_beta: list[np.ndarray] = []
    upper_by_beta: list[np.ndarray] = []
    beta_entries: list[dict[str, Any]] = []
    tail = 0.5 * (1.0 - rollouts.bank.confidence_level)
    for beta in betas:
        evidence = SparseTrajectoryEvidence(
            **evidence_base,
            likelihood_weight=beta,
        )
        posterior = rollouts.bank.update_from_sparse_evidence(evidence)
        if beta == 0.0 and posterior.tobytes() != prior.tobytes():
            raise RuntimeError("beta=0 did not preserve physical weights exactly")
        mean = _weighted_query_mean(components, posterior)
        lower = _weighted_quantile(components, posterior, tail)
        upper = _weighted_quantile(components, posterior, 1.0 - tail)
        weights_by_beta.append(posterior)
        means_by_beta.append(mean)
        lower_by_beta.append(lower)
        upper_by_beta.append(upper)
        beta_entries.append(
            {
                "beta": beta,
                "weight_diagnostics": _weight_diagnostics(
                    posterior, prior, rollouts.bank.hypothesis_ids
                ),
                "metrics": None,
            }
        )

    method_predictions: list[
        tuple[
            str,
            str,
            float | None,
            np.ndarray,
            np.ndarray | None,
            np.ndarray | None,
        ]
    ] = []
    zero_prediction = np.broadcast_to(
        forecast.anchor_positions_m[None], target.shape
    ).copy()
    method_predictions.append(
        ("zero_motion", "baseline", None, zero_prediction, None, None)
    )
    method_predictions.append(
        ("external_forecast", "forecast", None, target, None, None)
    )
    prior_mean = means_by_beta[betas.index(0.0)]
    prior_lower = lower_by_beta[betas.index(0.0)]
    prior_upper = upper_by_beta[betas.index(0.0)]
    method_predictions.append(
        ("physical_prior", "physical", 0.0, prior_mean, prior_lower, prior_upper)
    )
    for beta, mean, lower, upper in zip(
        betas, means_by_beta, lower_by_beta, upper_by_beta, strict=True
    ):
        if beta == 0.0:
            continue
        method_predictions.append(
            (f"semantic_beta_{beta:g}", "semantic", beta, mean, lower, upper)
        )

    truth_positions = None
    truth_valid = None
    reference_frames = None
    metrics_rows: list[dict[str, Any]] = []
    horizon_rows: list[dict[str, Any]] = []
    oracle: dict[str, Any] | None = None
    evaluation_best_beta: float | None = None
    if reference is not None:
        if reference.case_id != forecast.case_id:
            raise ValueError("reference and forecast case IDs differ")
        reference_nodes = _node_indices(
            reference.node_ids,
            forecast.node_indices,
            name="reference trajectory",
        )
        truth_positions, truth_valid, reference_frames = _interpolate_reference(
            reference,
            query_times,
            reference_nodes,
        )
        constant_velocity = _constant_velocity_prediction(
            reference,
            query_times,
            reference_nodes,
            rollouts.anchor_time_s,
        )
        if constant_velocity is not None:
            method_predictions.insert(
                1,
                (
                    "constant_velocity",
                    "baseline",
                    None,
                    constant_velocity,
                    None,
                    None,
                ),
            )

        for method, kind, beta, prediction, lower, upper in method_predictions:
            metrics = _trajectory_metrics(
                prediction,
                truth_positions,
                truth_valid,
                lower=lower,
                upper=upper,
            )
            row = {
                "method": method,
                "kind": kind,
                "beta": beta,
                **{
                    key: value for key, value in metrics.items() if key != "frame_ade_m"
                },
            }
            metrics_rows.append(row)
            for frame, error in enumerate(metrics["frame_ade_m"]):
                horizon_rows.append(
                    {
                        "method": method,
                        "kind": kind,
                        "beta": beta,
                        "forecast_step": frame + 1,
                        "future_time_s": float(forecast.future_times_s[frame]),
                        "absolute_time_s": float(query_times[frame]),
                        "ade_m": error,
                    }
                )
            if kind == "semantic" and beta is not None:
                beta_entries[betas.index(beta)]["metrics"] = {
                    key: value for key, value in metrics.items() if key != "frame_ade_m"
                }
            if method == "physical_prior":
                beta_entries[betas.index(0.0)]["metrics"] = {
                    key: value for key, value in metrics.items() if key != "frame_ade_m"
                }

        component_metrics: list[tuple[float, int, int, dict[str, Any]]] = []
        for hypothesis in range(components.shape[0]):
            for particle in range(components.shape[1]):
                metrics = _trajectory_metrics(
                    components[hypothesis, particle],
                    truth_positions,
                    truth_valid,
                )
                component_metrics.append(
                    (float(metrics["ade_m"]), hypothesis, particle, metrics)
                )
        _, hypothesis, particle, best_metrics = min(
            component_metrics,
            key=lambda row: row[0],
        )
        oracle = {
            "diagnostic_only": True,
            "hypothesis_id": rollouts.bank.hypothesis_ids[hypothesis],
            "hypothesis_index": hypothesis,
            "parameter_particle_index": particle,
            "metrics": {
                key: value
                for key, value in best_metrics.items()
                if key != "frame_ade_m"
            },
        }
        beta_metric_rows = [
            row
            for row in metrics_rows
            if row["method"] == "physical_prior" or row["kind"] == "semantic"
        ]
        best_row = min(
            beta_metric_rows,
            key=lambda row: (row["ade_m"], row["beta"] or 0.0),
        )
        evaluation_best_beta = float(best_row["beta"] or 0.0)

    report: dict[str, Any] = {
        "schema": EXTERNAL_BRIDGE_RUN_SCHEMA,
        "schema_version": EXTERNAL_BRIDGE_RUN_SCHEMA_VERSION,
        "case_id": forecast.case_id,
        "forecast_artifact_id": forecast.artifact_id,
        "forecast_id": forecast_id,
        "rollout_artifact_id": rollouts.artifact_id,
        "rollout_bank_artifact_id": rollouts.bank.artifact_id,
        "reference_artifact_id": (
            reference.artifact_id if reference is not None else None
        ),
        "settings": {
            "beta_values": list(betas),
            "scale_m": float(scale_m),
            "degrees_of_freedom": float(degrees_of_freedom),
            "anchor_tolerance_m": float(anchor_tolerance_m),
            "motion_ratio_min": float(motion_ratio_min),
            "motion_ratio_max": float(motion_ratio_max),
            "confidence_level": float(rollouts.bank.confidence_level),
        },
        "doctor": plain_json(doctor),
        "beta_results": beta_entries,
        "metrics": metrics_rows,
        "component_oracle": oracle,
        "evaluation_only_best_beta": evaluation_best_beta,
        "claim_boundary": (
            "The beta sweep and any evaluation-only best beta are diagnostic. "
            "Positive semantic trust requires source-only calibration and independent "
            "confirmation; beta=0 preserves the physical support exactly."
        ),
    }
    report = plain_json(
        validated_json_mapping(
            report,
            error_message="external bridge run report must be finite JSON data",
        )
    )
    prediction_names = [method for method, *_ in method_predictions]
    prediction_values = np.stack(
        [prediction for _, _, _, prediction, _, _ in method_predictions], axis=0
    )
    arrays = {
        "query_node_ids": forecast.node_indices,
        "forecast_future_times_s": forecast.future_times_s,
        "absolute_query_times_s": query_times,
        "rollout_fractional_frame_indices": query_frames,
        "beta_values": np.asarray(betas, dtype=np.float64),
        "posterior_weights": np.stack(weights_by_beta, axis=0),
        "posterior_query_means_m": np.stack(means_by_beta, axis=0),
        "posterior_query_lower_m": np.stack(lower_by_beta, axis=0),
        "posterior_query_upper_m": np.stack(upper_by_beta, axis=0),
        "hypothesis_ids": np.asarray(rollouts.bank.hypothesis_ids),
        "method_names": np.asarray(prediction_names),
        "method_query_predictions_m": prediction_values,
        "reference_positions_m": (
            truth_positions
            if truth_positions is not None
            else np.asarray([], dtype=np.float64)
        ),
        "reference_validity": (
            truth_valid if truth_valid is not None else np.asarray([], dtype=bool)
        ),
        "reference_fractional_frame_indices": (
            reference_frames
            if reference_frames is not None
            else np.asarray([], dtype=np.float64)
        ),
    }
    arrays["metrics_rows_json"] = np.asarray(
        json.dumps(metrics_rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    arrays["horizon_rows_json"] = np.asarray(
        json.dumps(horizon_rows, sort_keys=True, separators=(",", ":"), allow_nan=False)
    )
    return report, arrays


__all__ = [
    "EXTERNAL_BRIDGE_RUN_SCHEMA",
    "EXTERNAL_BRIDGE_RUN_SCHEMA_VERSION",
    "analyze_external_bridge",
]
