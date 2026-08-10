"""Source selection and independent confirmation for bridge trust."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from causal4d._external_bridge_trust_analysis import (
    _LoadedCase,
    _analyze_cases,
    _control_advantage,
    _derive_thresholds,
    _load_case,
    _mean_metric_by_beta,
    _ood_reasons,
    _panel_summary,
    _selected_beta,
)
from causal4d._external_bridge_trust_contracts import (
    ExternalBridgeTrustCalibration,
    ExternalBridgeTrustStudy,
)
from causal4d.immutable_json import plain_json


def fit_external_bridge_trust(
    study: ExternalBridgeTrustStudy,
    *,
    beta_candidates: Sequence[float] = (0.0, 1.0, 3.0, 6.0, 12.0),
    scale_m: float = 0.05,
    degrees_of_freedom: float = 3.0,
    anchor_tolerance_m: float = 0.01,
    doctor_motion_ratio_min: float = 0.10,
    doctor_motion_ratio_max: float = 10.0,
    minimum_selection_relative_improvement: float = 0.0,
    minimum_confirmation_relative_improvement: float = 0.0,
    maximum_case_relative_harm: float = 0.05,
    support_margin: float = 1.5,
    controls_required: bool = False,
    minimum_control_advantage_m: float = 0.0,
    require_clean_doctor: bool = True,
) -> ExternalBridgeTrustCalibration:
    """Select beta on source cases and admit it only after independent confirmation."""

    candidates = tuple(sorted(set(float(value) for value in beta_candidates)))
    if (
        not candidates
        or candidates[0] != 0.0
        or any(not np.isfinite(value) or value < 0.0 for value in candidates)
    ):
        raise ValueError(
            "beta_candidates must be finite, nonnegative, and include zero"
        )
    if any(
        right <= left for left, right in zip(candidates, candidates[1:], strict=False)
    ):
        raise ValueError("beta_candidates must be strictly increasing")
    if not np.isfinite(scale_m) or scale_m <= 0.0:
        raise ValueError("scale_m must be finite and positive")
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise ValueError("degrees_of_freedom must be finite and positive")
    if not np.isfinite(anchor_tolerance_m) or anchor_tolerance_m <= 0.0:
        raise ValueError("anchor_tolerance_m must be finite and positive")
    if (
        not np.isfinite(doctor_motion_ratio_min)
        or not np.isfinite(doctor_motion_ratio_max)
        or doctor_motion_ratio_min <= 0.0
        or doctor_motion_ratio_max <= doctor_motion_ratio_min
    ):
        raise ValueError("doctor motion-ratio bounds are invalid")
    nonnegative = (
        minimum_selection_relative_improvement,
        minimum_confirmation_relative_improvement,
        maximum_case_relative_harm,
        minimum_control_advantage_m,
    )
    if any(not np.isfinite(value) or value < 0.0 for value in nonnegative):
        raise ValueError("trust gates must be finite and nonnegative")
    if not np.isfinite(support_margin) or support_margin < 1.0:
        raise ValueError("support_margin must be finite and at least one")

    selection_cases = tuple(_load_case(study, spec) for spec in study.selection_cases)
    confirmation_cases = tuple(
        _load_case(study, spec) for spec in study.confirmation_cases
    )
    selection_reference_ids = {case.reference.artifact_id for case in selection_cases}
    confirmation_reference_ids = {
        case.reference.artifact_id for case in confirmation_cases
    }
    reused_references = sorted(selection_reference_ids & confirmation_reference_ids)
    if reused_references:
        raise ValueError(
            "selection and confirmation panels reuse reference artifacts: "
            + repr(reused_references)
        )
    selection_results = _analyze_cases(
        selection_cases,
        candidates,
        scale_m=scale_m,
        degrees_of_freedom=degrees_of_freedom,
        anchor_tolerance_m=anchor_tolerance_m,
        doctor_motion_ratio_min=doctor_motion_ratio_min,
        doctor_motion_ratio_max=doctor_motion_ratio_max,
    )
    selection_mean_ade = _mean_metric_by_beta(
        selection_results,
        candidates,
        "ade_m_by_beta",
    )
    selection_mean_fde = _mean_metric_by_beta(
        selection_results,
        candidates,
        "fde_m_by_beta",
    )
    selected_beta = _selected_beta(
        candidates,
        selection_mean_ade,
        selection_mean_fde,
    )
    selection_summary = _panel_summary(
        selection_results,
        candidates,
        selected_beta,
    )
    thresholds = _derive_thresholds(
        selection_results,
        support_margin=support_margin,
        require_clean_doctor=require_clean_doctor,
    )

    selection_control_advantages: list[float] = []
    for case, result in zip(selection_cases, selection_results, strict=True):
        advantage, control_rows = _control_advantage(
            case,
            selected_beta,
            float(result["selected_beta_ade_m"]),
            scale_m=scale_m,
            degrees_of_freedom=degrees_of_freedom,
            anchor_tolerance_m=anchor_tolerance_m,
            doctor_motion_ratio_min=doctor_motion_ratio_min,
            doctor_motion_ratio_max=doctor_motion_ratio_max,
        )
        result["control_results"] = control_rows
        result["instruction_control_advantage_m"] = advantage
        if advantage is not None:
            selection_control_advantages.append(advantage)
    selection_summary["controls_evaluated"] = bool(selection_control_advantages)
    selection_summary["minimum_instruction_control_advantage_m"] = (
        min(selection_control_advantages) if selection_control_advantages else None
    )

    reasons: list[str] = []
    if selected_beta == 0.0:
        reasons.append("no_positive_selection_beta")
    if (
        float(selection_summary["relative_improvement"])
        < minimum_selection_relative_improvement
    ):
        reasons.append("insufficient_selection_improvement")
    if (
        float(selection_summary["maximum_case_relative_harm"])
        > maximum_case_relative_harm
    ):
        reasons.append("selection_case_harm_exceeded")
    if (
        float(selection_summary["maximum_case_fde_relative_harm"])
        > maximum_case_relative_harm
    ):
        reasons.append("selection_fde_case_harm_exceeded")
    if require_clean_doctor and any(
        result["diagnostics"]["doctor"]["warnings"] for result in selection_results
    ):
        reasons.append("selection_doctor_warning")
    if controls_required:
        if len(selection_control_advantages) != len(selection_cases):
            reasons.append("selection_controls_missing")
        elif min(selection_control_advantages) < minimum_control_advantage_m:
            reasons.append("selection_control_gate_failed")

    confirmation_results: list[dict[str, Any]] = []
    confirmation_summary: dict[str, Any] = {
        "case_ids": [case.specification.case_id for case in confirmation_cases],
        "evaluated": False,
        "passed": False,
        "mean_ade_m_by_beta": {},
        "mean_fde_m_by_beta": {},
        "physical_prior_mean_ade_m": None,
        "selected_beta_mean_ade_m": None,
        "relative_improvement": None,
        "physical_prior_mean_fde_m": None,
        "selected_beta_mean_fde_m": None,
        "fde_relative_improvement": None,
        "maximum_case_relative_harm": None,
        "maximum_case_fde_relative_harm": None,
        "controls_evaluated": False,
        "minimum_instruction_control_advantage_m": None,
        "case_results": [],
    }
    selection_passed = not reasons
    if not confirmation_cases:
        reasons.append("missing_independent_confirmation")
    elif selection_passed:
        confirmation_candidates = (
            (0.0, selected_beta) if selected_beta > 0.0 else (0.0,)
        )
        confirmation_results = _analyze_cases(
            confirmation_cases,
            confirmation_candidates,
            scale_m=scale_m,
            degrees_of_freedom=degrees_of_freedom,
            anchor_tolerance_m=anchor_tolerance_m,
            doctor_motion_ratio_min=doctor_motion_ratio_min,
            doctor_motion_ratio_max=doctor_motion_ratio_max,
        )
        confirmation_summary = _panel_summary(
            confirmation_results,
            confirmation_candidates,
            selected_beta,
        )
        confirmation_summary["evaluated"] = True
        confirmation_ood_rejections = []
        confirmation_control_advantages: list[float] = []
        for case, result in zip(
            confirmation_cases,
            confirmation_results,
            strict=True,
        ):
            ood_reasons = _ood_reasons(result["diagnostics"], thresholds)
            result["ood_reasons"] = list(ood_reasons)
            confirmation_ood_rejections.extend(ood_reasons)
            advantage, control_rows = _control_advantage(
                case,
                selected_beta,
                float(result["selected_beta_ade_m"]),
                scale_m=scale_m,
                degrees_of_freedom=degrees_of_freedom,
                anchor_tolerance_m=anchor_tolerance_m,
                doctor_motion_ratio_min=doctor_motion_ratio_min,
                doctor_motion_ratio_max=doctor_motion_ratio_max,
            )
            result["control_results"] = control_rows
            result["instruction_control_advantage_m"] = advantage
            if advantage is not None:
                confirmation_control_advantages.append(advantage)
        confirmation_summary["controls_evaluated"] = bool(
            confirmation_control_advantages
        )
        confirmation_summary["minimum_instruction_control_advantage_m"] = (
            min(confirmation_control_advantages)
            if confirmation_control_advantages
            else None
        )
        confirmation_summary["case_results"] = confirmation_results
        if confirmation_ood_rejections:
            reasons.append("confirmation_ood_rejection")
        if (
            float(confirmation_summary["relative_improvement"])
            < minimum_confirmation_relative_improvement
        ):
            reasons.append("insufficient_confirmation_improvement")
        if (
            float(confirmation_summary["maximum_case_relative_harm"])
            > maximum_case_relative_harm
        ):
            reasons.append("confirmation_case_harm_exceeded")
        if (
            float(confirmation_summary["maximum_case_fde_relative_harm"])
            > maximum_case_relative_harm
        ):
            reasons.append("confirmation_fde_case_harm_exceeded")
        if controls_required:
            if len(confirmation_control_advantages) != len(confirmation_cases):
                reasons.append("confirmation_controls_missing")
            elif min(confirmation_control_advantages) < minimum_control_advantage_m:
                reasons.append("confirmation_control_gate_failed")
        confirmation_summary["passed"] = not reasons

    admitted_beta = selected_beta if not reasons else 0.0

    def source_identity(case: _LoadedCase) -> dict[str, Any]:
        return {
            "case_id": case.specification.case_id,
            "forecast_artifact_id": case.forecast.artifact_id,
            "rollout_artifact_id": case.rollouts.artifact_id,
            "rollout_bank_artifact_id": case.rollouts.bank.artifact_id,
            "reference_artifact_id": case.reference.artifact_id,
            "forecast_id": case.specification.forecast_id,
            "control_forecast_ids": list(case.specification.control_forecast_ids),
        }

    source_cases = {
        "selection": [source_identity(case) for case in selection_cases],
        "confirmation": [source_identity(case) for case in confirmation_cases],
    }
    return ExternalBridgeTrustCalibration(
        study_manifest_sha256=study.manifest_sha256,
        beta_candidates=candidates,
        selected_beta=selected_beta,
        admitted_beta=admitted_beta,
        confirmed=admitted_beta > 0.0,
        selection=selection_summary,
        confirmation=confirmation_summary,
        thresholds=thresholds,
        gates={
            "minimum_selection_relative_improvement": (
                minimum_selection_relative_improvement
            ),
            "minimum_confirmation_relative_improvement": (
                minimum_confirmation_relative_improvement
            ),
            "maximum_case_relative_harm": maximum_case_relative_harm,
            "support_margin": support_margin,
            "controls_required": bool(controls_required),
            "minimum_control_advantage_m": minimum_control_advantage_m,
        },
        settings={
            "scale_m": scale_m,
            "degrees_of_freedom": degrees_of_freedom,
            "anchor_tolerance_m": anchor_tolerance_m,
            "doctor_motion_ratio_min": doctor_motion_ratio_min,
            "doctor_motion_ratio_max": doctor_motion_ratio_max,
        },
        source_cases=source_cases,
        reasons=tuple(dict.fromkeys(reasons)),
        metadata={
            "study": plain_json(study.metadata),
            "selection_and_confirmation_are_disjoint": True,
            "target_future_used_for_admission": False,
        },
    )


__all__ = ["fit_external_bridge_trust"]
