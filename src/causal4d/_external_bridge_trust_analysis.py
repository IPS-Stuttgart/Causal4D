"""Source-case analysis helpers for external bridge trust."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from causal4d._external_bridge_metrics import _node_indices
from causal4d._external_bridge_trust_contracts import (
    ExternalBridgeTrustCaseSpec,
    ExternalBridgeTrustStudy,
)
from causal4d.external_bridge import (
    _fractional_frame_indices,
    _interpolate_components,
    build_external_bridge_report,
)
from causal4d.external_bridge_run import analyze_external_bridge
from causal4d.external_forecast import ExternalForecastBundle, load_external_forecast
from causal4d.external_reference import (
    ExternalReferenceTrajectory,
    load_external_reference,
)
from causal4d.external_rollout import (
    ExternalRolloutBundle,
    load_external_rollout_bank,
)
from causal4d.immutable_json import plain_json, validated_json_mapping


@dataclass(frozen=True)
class _LoadedCase:
    specification: ExternalBridgeTrustCaseSpec
    forecast: ExternalForecastBundle
    rollouts: ExternalRolloutBundle
    reference: ExternalReferenceTrajectory


def _resolved(root: Path, relative: str) -> Path:
    return root.joinpath(*relative.split("/"))


def _load_case(
    study: ExternalBridgeTrustStudy,
    spec: ExternalBridgeTrustCaseSpec,
) -> _LoadedCase:
    forecast = load_external_forecast(_resolved(study.root, spec.forecast))
    rollouts = load_external_rollout_bank(_resolved(study.root, spec.rollouts))
    reference = load_external_reference(_resolved(study.root, spec.reference))
    identities = {forecast.case_id, rollouts.case_id, reference.case_id, spec.case_id}
    if len(identities) != 1:
        raise ValueError(
            f"trust case {spec.case_id!r} has inconsistent case identities: "
            f"{sorted(identities)!r}"
        )
    forecast.forecast_index(spec.forecast_id)
    for control in spec.control_forecast_ids:
        forecast.forecast_index(control)
    return _LoadedCase(spec, forecast, rollouts, reference)


def _beta_key(beta: float) -> str:
    return format(float(beta), ".17g")


def _metric_by_method(report: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["method"]): row for row in report["metrics"]}


def _beta_metric(report: Mapping[str, Any], beta: float, metric: str) -> float:
    metrics = _metric_by_method(report)
    method = "physical_prior" if beta == 0.0 else f"semantic_beta_{beta:g}"
    if method not in metrics:
        raise ValueError(f"bridge report is missing {method!r}")
    return float(metrics[method][metric])


def _beta_ade(report: Mapping[str, Any], beta: float) -> float:
    return _beta_metric(report, beta, "ade_m")


def _relative_harm(candidate: float, baseline: float) -> float:
    denominator = max(float(baseline), 1e-12)
    return float((float(candidate) - float(baseline)) / denominator)


def _support_diagnostics(
    forecast: ExternalForecastBundle,
    forecast_id: str,
    rollouts: ExternalRolloutBundle,
    *,
    anchor_tolerance_m: float,
    doctor_motion_ratio_min: float,
    doctor_motion_ratio_max: float,
    scale_m: float,
    degrees_of_freedom: float,
) -> Mapping[str, Any]:
    doctor = build_external_bridge_report(
        forecast,
        forecast_id,
        rollouts,
        anchor_tolerance_m=anchor_tolerance_m,
        motion_ratio_min=doctor_motion_ratio_min,
        motion_ratio_max=doctor_motion_ratio_max,
        scale_m=scale_m,
        degrees_of_freedom=degrees_of_freedom,
    )
    if forecast.future_times_s is None:
        raise ValueError("trust calibration requires explicit forecast future_times_s")
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
    anchors = rollouts.bank.trajectories[
        :, :, rollouts.anchor_frame_index, internal_nodes
    ].astype(np.float64)
    component_displacements = components - anchors[:, :, None]
    forecast_index = forecast.forecast_index(forecast_id)
    target = forecast.future_positions_m[forecast_index]
    target_displacement = target - forecast.anchor_positions_m[None]
    valid = forecast.coordinate_validity[forecast_index]
    if not np.any(valid):
        raise ValueError("forecast has no valid coordinates")
    residual = component_displacements - target_displacement[None, None]
    component_distance = np.sqrt(
        np.mean(np.square(residual[:, :, valid]), axis=2)
    )
    prior = rollouts.bank.prior_joint_weights
    raw_motion_ratio = doctor["motion"]["forecast_to_rollout_ratio"]
    rollout_motion_zero = raw_motion_ratio is None
    motion_ratio = 0.0 if rollout_motion_zero else float(raw_motion_ratio)
    return validated_json_mapping(
        {
            "doctor": plain_json(doctor),
            "minimum_physical_support_distance_m": float(
                np.min(component_distance)
            ),
            "prior_weighted_support_distance_m": float(
                np.sum(prior * component_distance)
            ),
            "semantic_motion_rms_m": float(
                doctor["motion"]["forecast_coordinate_rms_m"]
            ),
            "semantic_to_physical_motion_ratio": motion_ratio,
            "rollout_motion_zero": rollout_motion_zero,
            "anchor_error_m": float(doctor["anchor"]["prior_weighted_rms_m"]),
            "valid_coordinate_fraction": float(
                doctor["valid_coordinate_fraction"]
            ),
        },
        error_message="external bridge trust diagnostics must be finite JSON data",
    )


def _control_advantage(
    case: _LoadedCase,
    selected_beta: float,
    correct_ade_m: float,
    *,
    scale_m: float,
    degrees_of_freedom: float,
    anchor_tolerance_m: float,
    doctor_motion_ratio_min: float,
    doctor_motion_ratio_max: float,
) -> tuple[float | None, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for control_id in case.specification.control_forecast_ids:
        report, _ = analyze_external_bridge(
            case.forecast,
            control_id,
            case.rollouts,
            beta_values=(0.0, selected_beta),
            scale_m=scale_m,
            degrees_of_freedom=degrees_of_freedom,
            anchor_tolerance_m=anchor_tolerance_m,
            motion_ratio_min=doctor_motion_ratio_min,
            motion_ratio_max=doctor_motion_ratio_max,
            reference=case.reference,
        )
        control_ade = _beta_ade(report, selected_beta)
        rows.append(
            {
                "forecast_id": control_id,
                "selected_beta_ade_m": control_ade,
            }
        )
    if not rows:
        return None, rows
    advantage = min(float(row["selected_beta_ade_m"]) for row in rows) - float(
        correct_ade_m
    )
    return float(advantage), rows


def _analyze_cases(
    cases: Sequence[_LoadedCase],
    beta_candidates: tuple[float, ...],
    *,
    scale_m: float,
    degrees_of_freedom: float,
    anchor_tolerance_m: float,
    doctor_motion_ratio_min: float,
    doctor_motion_ratio_max: float,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for case in cases:
        report, _ = analyze_external_bridge(
            case.forecast,
            case.specification.forecast_id,
            case.rollouts,
            beta_values=beta_candidates,
            scale_m=scale_m,
            degrees_of_freedom=degrees_of_freedom,
            anchor_tolerance_m=anchor_tolerance_m,
            motion_ratio_min=doctor_motion_ratio_min,
            motion_ratio_max=doctor_motion_ratio_max,
            reference=case.reference,
        )
        ade_by_beta = {
            _beta_key(beta): _beta_ade(report, beta) for beta in beta_candidates
        }
        fde_by_beta = {
            _beta_key(beta): _beta_metric(report, beta, "fde_m")
            for beta in beta_candidates
        }
        diagnostics = _support_diagnostics(
            case.forecast,
            case.specification.forecast_id,
            case.rollouts,
            anchor_tolerance_m=anchor_tolerance_m,
            doctor_motion_ratio_min=doctor_motion_ratio_min,
            doctor_motion_ratio_max=doctor_motion_ratio_max,
            scale_m=scale_m,
            degrees_of_freedom=degrees_of_freedom,
        )
        results.append(
            {
                "case_id": case.specification.case_id,
                "forecast_artifact_id": case.forecast.artifact_id,
                "rollout_artifact_id": case.rollouts.artifact_id,
                "rollout_bank_artifact_id": case.rollouts.bank.artifact_id,
                "reference_artifact_id": case.reference.artifact_id,
                "forecast_id": case.specification.forecast_id,
                "control_forecast_ids": list(
                    case.specification.control_forecast_ids
                ),
                "ade_m_by_beta": ade_by_beta,
                "fde_m_by_beta": fde_by_beta,
                "diagnostics": plain_json(diagnostics),
            }
        )
    return results


def _mean_metric_by_beta(
    results: Sequence[Mapping[str, Any]],
    beta_candidates: tuple[float, ...],
    metric_key: str,
) -> tuple[float, ...]:
    return tuple(
        float(
            np.mean(
                [
                    float(result[metric_key][_beta_key(beta)])
                    for result in results
                ]
            )
        )
        for beta in beta_candidates
    )


def _selected_beta(
    beta_candidates: tuple[float, ...],
    mean_ade: tuple[float, ...],
    mean_fde: tuple[float, ...],
) -> float:
    index = min(
        range(len(beta_candidates)),
        key=lambda item: (mean_ade[item], mean_fde[item], beta_candidates[item]),
    )
    return beta_candidates[index]


def _panel_summary(
    results: list[dict[str, Any]],
    beta_candidates: tuple[float, ...],
    selected_beta: float,
) -> dict[str, Any]:
    mean_ade = _mean_metric_by_beta(results, beta_candidates, "ade_m_by_beta")
    mean_fde = _mean_metric_by_beta(results, beta_candidates, "fde_m_by_beta")
    baseline_ade = mean_ade[beta_candidates.index(0.0)]
    selected_ade = mean_ade[beta_candidates.index(selected_beta)]
    baseline_fde = mean_fde[beta_candidates.index(0.0)]
    selected_fde = mean_fde[beta_candidates.index(selected_beta)]
    ade_case_harm = []
    fde_case_harm = []
    for result in results:
        case_baseline_ade = float(result["ade_m_by_beta"][_beta_key(0.0)])
        case_selected_ade = float(
            result["ade_m_by_beta"][_beta_key(selected_beta)]
        )
        case_baseline_fde = float(result["fde_m_by_beta"][_beta_key(0.0)])
        case_selected_fde = float(
            result["fde_m_by_beta"][_beta_key(selected_beta)]
        )
        ade_harm = _relative_harm(case_selected_ade, case_baseline_ade)
        fde_harm = _relative_harm(case_selected_fde, case_baseline_fde)
        result["selected_beta_ade_m"] = case_selected_ade
        result["physical_prior_ade_m"] = case_baseline_ade
        result["selected_beta_relative_harm"] = ade_harm
        result["selected_beta_fde_m"] = case_selected_fde
        result["physical_prior_fde_m"] = case_baseline_fde
        result["selected_beta_fde_relative_harm"] = fde_harm
        ade_case_harm.append(ade_harm)
        fde_case_harm.append(fde_harm)
    return {
        "case_ids": [str(result["case_id"]) for result in results],
        "mean_ade_m_by_beta": {
            _beta_key(beta): value
            for beta, value in zip(beta_candidates, mean_ade, strict=True)
        },
        "mean_fde_m_by_beta": {
            _beta_key(beta): value
            for beta, value in zip(beta_candidates, mean_fde, strict=True)
        },
        "physical_prior_mean_ade_m": baseline_ade,
        "selected_beta_mean_ade_m": selected_ade,
        "relative_improvement": float(
            1.0 - selected_ade / max(baseline_ade, 1e-12)
        ),
        "physical_prior_mean_fde_m": baseline_fde,
        "selected_beta_mean_fde_m": selected_fde,
        "fde_relative_improvement": float(
            1.0 - selected_fde / max(baseline_fde, 1e-12)
        ),
        "maximum_case_relative_harm": float(max(ade_case_harm)),
        "maximum_case_fde_relative_harm": float(max(fde_case_harm)),
        "case_results": results,
    }


def _derive_thresholds(
    selection_results: Sequence[Mapping[str, Any]],
    *,
    support_margin: float,
    require_clean_doctor: bool,
) -> dict[str, Any]:
    diagnostics = [result["diagnostics"] for result in selection_results]
    support = [
        float(value["minimum_physical_support_distance_m"]) for value in diagnostics
    ]
    anchor = [float(value["anchor_error_m"]) for value in diagnostics]
    motion = [float(value["semantic_motion_rms_m"]) for value in diagnostics]
    ratio = [
        float(value["semantic_to_physical_motion_ratio"]) for value in diagnostics
    ]
    valid_fraction = [
        float(value["valid_coordinate_fraction"]) for value in diagnostics
    ]
    return {
        "maximum_support_distance_m": max(max(support) * support_margin + 0.001, 0.005),
        "maximum_anchor_error_m": max(max(anchor) * support_margin + 0.001, 0.005),
        "minimum_semantic_motion_m": max(min(motion) / support_margin, 1e-4),
        "minimum_motion_ratio": max(min(ratio) / support_margin, 0.05),
        "maximum_motion_ratio": max(max(ratio) * support_margin, 2.0),
        "minimum_valid_coordinate_fraction": max(
            min(valid_fraction) / support_margin,
            0.05,
        ),
        "require_clean_doctor": bool(require_clean_doctor),
    }


def _ood_reasons(
    diagnostics: Mapping[str, Any],
    thresholds: Mapping[str, Any],
) -> tuple[str, ...]:
    reasons: list[str] = []
    doctor = diagnostics["doctor"]
    if bool(thresholds["require_clean_doctor"]) and doctor["warnings"]:
        reasons.append("bridge_doctor_warning")
    if bool(diagnostics.get("rollout_motion_zero", False)):
        reasons.append("physical_rollout_motion_zero")
    if (
        float(diagnostics["minimum_physical_support_distance_m"])
        > float(thresholds["maximum_support_distance_m"])
    ):
        reasons.append("outside_physical_support")
    if float(diagnostics["anchor_error_m"]) > float(
        thresholds["maximum_anchor_error_m"]
    ):
        reasons.append("anchor_misalignment")
    if float(diagnostics["semantic_motion_rms_m"]) < float(
        thresholds["minimum_semantic_motion_m"]
    ):
        reasons.append("static_semantic_forecast")
    ratio = float(diagnostics["semantic_to_physical_motion_ratio"])
    if ratio < float(thresholds["minimum_motion_ratio"]):
        reasons.append("semantic_motion_too_small")
    if ratio > float(thresholds["maximum_motion_ratio"]):
        reasons.append("semantic_motion_too_large")
    if float(diagnostics["valid_coordinate_fraction"]) < float(
        thresholds["minimum_valid_coordinate_fraction"]
    ):
        reasons.append("insufficient_valid_coordinates")
    return tuple(reasons)
