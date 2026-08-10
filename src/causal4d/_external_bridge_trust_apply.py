"""Label-free target admission for a frozen external bridge calibration."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from causal4d._external_bridge_trust_analysis import (
    _ood_reasons,
    _support_diagnostics,
)
from causal4d._external_bridge_trust_contracts import (
    ExternalBridgeTrustCalibration,
    ExternalBridgeTrustDecision,
)
from causal4d.external_bridge_run import analyze_external_bridge
from causal4d.external_forecast import ExternalForecastBundle
from causal4d.external_reference import ExternalReferenceTrajectory
from causal4d.external_rollout import ExternalRolloutBundle
from causal4d.immutable_json import plain_json, validated_json_mapping


def apply_external_bridge_trust(
    forecast: ExternalForecastBundle,
    forecast_id: str,
    rollouts: ExternalRolloutBundle,
    calibration: ExternalBridgeTrustCalibration,
    *,
    reference: ExternalReferenceTrajectory | None = None,
) -> tuple[Mapping[str, Any], dict[str, np.ndarray], ExternalBridgeTrustDecision]:
    """Apply a frozen source/confirmation gate without reading a target future."""

    settings = calibration.settings
    diagnostics = _support_diagnostics(
        forecast,
        forecast_id,
        rollouts,
        anchor_tolerance_m=float(settings["anchor_tolerance_m"]),
        doctor_motion_ratio_min=float(settings["doctor_motion_ratio_min"]),
        doctor_motion_ratio_max=float(settings["doctor_motion_ratio_max"]),
        scale_m=float(settings["scale_m"]),
        degrees_of_freedom=float(settings["degrees_of_freedom"]),
    )
    reasons: list[str] = []
    source_records = [
        *calibration.source_cases["selection"],
        *calibration.source_cases["confirmation"],
    ]
    if forecast.case_id in {str(record["case_id"]) for record in source_records}:
        reasons.append("target_reuses_source_case_id")
    if forecast.artifact_id in {
        str(record["forecast_artifact_id"]) for record in source_records
    }:
        reasons.append("target_reuses_source_forecast")
    if rollouts.artifact_id in {
        str(record["rollout_artifact_id"]) for record in source_records
    }:
        reasons.append("target_reuses_source_rollouts")
    if calibration.admitted_beta == 0.0:
        reasons.append("calibration_not_admitted")
    reasons.extend(_ood_reasons(diagnostics, calibration.thresholds))
    applied_beta = 0.0 if reasons else calibration.admitted_beta
    decision = ExternalBridgeTrustDecision(
        calibration_id=calibration.calibration_id,
        forecast_artifact_id=forecast.artifact_id,
        rollout_artifact_id=rollouts.artifact_id,
        forecast_id=forecast_id,
        admitted_beta=calibration.admitted_beta,
        applied_beta=applied_beta,
        accepted=applied_beta > 0.0,
        reasons=tuple(dict.fromkeys(reasons)),
        diagnostics=plain_json(diagnostics),
    )
    betas = (0.0, applied_beta) if applied_beta > 0.0 else (0.0,)
    report, arrays = analyze_external_bridge(
        forecast,
        forecast_id,
        rollouts,
        beta_values=betas,
        scale_m=float(settings["scale_m"]),
        degrees_of_freedom=float(settings["degrees_of_freedom"]),
        anchor_tolerance_m=float(settings["anchor_tolerance_m"]),
        motion_ratio_min=float(settings["doctor_motion_ratio_min"]),
        motion_ratio_max=float(settings["doctor_motion_ratio_max"]),
        reference=reference,
    )
    mutable = plain_json(report)
    mutable["trust"] = decision.descriptor()
    mutable["deployment_beta"] = applied_beta
    mutable["settings"]["trust_calibration_id"] = calibration.calibration_id
    mutable["claim_boundary"] = (
        "The applied beta is fixed by a disjoint source-selection and independent-"
        "confirmation calibration, then admitted on this target using label-free OOD "
        "diagnostics only. Any evaluation-only best beta uses target reference data "
        "and cannot alter the deployment decision. Rejection preserves the physical "
        "weights exactly at beta=0."
    )
    report = validated_json_mapping(
        mutable,
        error_message="trusted external bridge report must be finite JSON data",
    )
    posterior_weights = np.asarray(arrays["posterior_weights"])
    prior = rollouts.bank.prior_joint_weights
    if applied_beta == 0.0 and posterior_weights[0].tobytes() != prior.tobytes():
        raise RuntimeError("trust rejection failed to preserve physical weights")
    return report, arrays, decision


__all__ = ["apply_external_bridge_trust"]
