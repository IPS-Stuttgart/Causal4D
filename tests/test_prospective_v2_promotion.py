from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from causal4d.decision_trace import (
    DecisionTraceArtifact,
    DecisionTraceDecision,
    DecisionTraceStage,
)
from causal4d.prospective_v2_profile import (
    PROSPECTIVE_V2_REQUIRED_DECISION_NAMES,
    build_prospective_v2_decision_trace_v1,
)
from causal4d.prospective_v2_promotion import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2MetricContractV1,
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2PromotionPolicyV1,
    ProspectiveV2TargetOpeningV1,
    ProspectiveV2UnitMetricValuesV1,
    build_prospective_v2_target_opening_v1,
    build_prospective_v2_unit_evaluation_v1,
    evaluate_prospective_v2_promotion_v1,
    validate_prospective_v2_promotion_result_v1,
)


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(
    artifact_id: str,
    role: str,
    producer: str,
) -> DecisionTraceArtifact:
    return DecisionTraceArtifact(
        artifact_id=artifact_id,
        artifact_kind=f"test.{role}",
        role=role,
        producer=producer,  # type: ignore[arg-type]
    )


def _decision(
    name: str,
    producer: str,
    *,
    accepted: bool,
    label: str,
) -> DecisionTraceDecision:
    return DecisionTraceDecision(
        name=name,
        decision_id=_id(f"decision:{label}:{name}:{accepted}"),
        decision_kind=f"test.{name}",
        producer=producer,  # type: ignore[arg-type]
        accepted=accepted,
        reasons=() if accepted else ("source_gate_rejected",),
    )


def _candidate_ladder() -> tuple[ProspectiveV2CandidateV1, ...]:
    return tuple(
        ProspectiveV2CandidateV1(
            candidate_id=f"candidate-{index}",
            candidate_kind=kind,
            configuration_artifact_id=_id(f"configuration:{kind}"),
        )
        for index, kind in enumerate(PROSPECTIVE_V2_CANDIDATE_KINDS)
    )


def _units() -> tuple[ProspectiveV2EvaluationUnitV1, ...]:
    endpoints = (
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
    )
    return tuple(
        ProspectiveV2EvaluationUnitV1(
            unit_id=f"unit-{index}",
            endpoint=endpoint,
            protocol_id="protocol-v2",
            case_id=f"case-{index}",
            session_id=f"session-{index}",
            independent_group_id=f"group-{index}",
            target_artifact_id=_id(f"target:{index}"),
            factual_context_artifact_id=_id(f"factual:{index}"),
            counterfactual_query_artifact_id=_id(f"query:{index}"),
            target_access_seal_id=_id("target-access-seal"),
        )
        for index, endpoint in enumerate(endpoints)
    )


def _freeze() -> ProspectiveV2PromotionFreezeV1:
    return ProspectiveV2PromotionFreezeV1(
        experiment_id="prospective-v2-selection-panel",
        stack_lock_id=_id("stack-lock"),
        target_access_seal_id=_id("target-access-seal"),
        candidates=_candidate_ladder(),
        evaluation_units=_units(),
        metric_contract=ProspectiveV2MetricContractV1(
            scoring_implementation_artifact_id=_id("scoring-implementation"),
            nominal_coverage=0.9,
            harmful_regret_threshold_m=0.001,
        ),
        policy=ProspectiveV2PromotionPolicyV1(
            minimum_units_per_endpoint=1,
            minimum_mean_log_score_gain=0.05,
            maximum_mean_brier_change=0.0,
            maximum_mean_trajectory_regret_m=0.0,
            maximum_mean_coverage_error=0.05,
            maximum_mean_interval_width_ratio=1.1,
            minimum_accepted_update_rate=1.0,
            maximum_harmful_accepted_update_rate=0.0,
            maximum_fallback_rate=0.0,
        ),
        source_artifact_ids=(
            _id("stack-lock"),
            _id("target-access-seal"),
            _id("scoring-implementation"),
            *tuple(
                _id(f"configuration:{kind}") for kind in PROSPECTIVE_V2_CANDIDATE_KINDS
            ),
            *tuple(_id(f"factual:{index}") for index in range(3)),
            *tuple(_id(f"query:{index}") for index in range(3)),
        ),
    )


def _trace(
    freeze: ProspectiveV2PromotionFreezeV1,
    unit: ProspectiveV2EvaluationUnitV1,
    candidate: ProspectiveV2CandidateV1,
    *,
    selected: bool = True,
    metadata_overrides: dict[str, str] | None = None,
    wrong_factual_context: bool = False,
):
    label = f"{unit.unit_id}:{candidate.candidate_id}"
    factual = _artifact(
        (
            _id(f"wrong-factual:{unit.unit_id}")
            if wrong_factual_context
            else unit.factual_context_artifact_id
        ),
        "factual_evidence_context",
        "causal4d",
    )
    query = _artifact(
        unit.counterfactual_query_artifact_id,
        "counterfactual_query_context",
        "causal4d",
    )
    observation = _artifact(
        _id(f"observation:{label}"),
        "prob4d_observation",
        "prob4d",
    )
    belief = _artifact(
        _id(f"belief:{label}"),
        "bayesian_phystwin_belief",
        "bayesian-phystwin",
    )
    baseline_prediction = _artifact(
        _id(f"baseline-prediction:{unit.unit_id}"),
        "baseline_prediction",
        "bayesian-phystwin",
    )
    factual_posterior = _artifact(
        _id(f"factual-posterior:{label}"),
        "causal4d_factual_posterior",
        "causal4d",
    )
    candidate_prediction = _artifact(
        _id(f"candidate-prediction:{label}"),
        "candidate_prediction",
        "causal4d",
    )
    decisions = {
        name: _decision(
            name,
            (
                "prob4d"
                if name == "prob4d_provider_acceptance"
                else "bayesian-phystwin"
                if name == "bayesian_phystwin_acceptance"
                else "causal4d"
            ),
            accepted=selected,
            label=label,
        )
        for name in PROSPECTIVE_V2_REQUIRED_DECISION_NAMES
    }
    stages = (
        DecisionTraceStage(
            stage_name=f"prob4d:{label}",
            stage_kind="prob4d_observation",
            producer="prob4d",
            input_artifact_ids=(factual.artifact_id,),
            output_artifacts=(observation,),
            decisions=(decisions["prob4d_provider_acceptance"],),
        ),
        DecisionTraceStage(
            stage_name=f"bpt:{label}",
            stage_kind="bayesian_phystwin_belief",
            producer="bayesian-phystwin",
            input_artifact_ids=(observation.artifact_id,),
            output_artifacts=(belief, baseline_prediction),
            decisions=(decisions["bayesian_phystwin_acceptance"],),
        ),
        DecisionTraceStage(
            stage_name=f"abduction:{label}",
            stage_kind="causal4d_abduction",
            producer="causal4d",
            input_artifact_ids=(observation.artifact_id, belief.artifact_id),
            output_artifacts=(factual_posterior,),
            decisions=tuple(
                decisions[name]
                for name in (
                    "joint_covariance_admission",
                    "functional_support",
                    "intervention_identifiability",
                    "action_support",
                    "contact_v2_support",
                )
            ),
        ),
        DecisionTraceStage(
            stage_name=f"counterfactual:{label}",
            stage_kind="causal4d_counterfactual",
            producer="causal4d",
            input_artifact_ids=(factual_posterior.artifact_id, query.artifact_id),
            output_artifacts=(candidate_prediction,),
            decisions=tuple(
                decisions[name]
                for name in (
                    "conditional_uncertainty_calibration",
                    "query_calibration",
                )
            ),
        ),
        DecisionTraceStage(
            stage_name=f"deployment:{label}",
            stage_kind="deployment",
            producer="causal4d",
            input_artifact_ids=(
                baseline_prediction.artifact_id,
                candidate_prediction.artifact_id,
            ),
            decisions=(decisions["counterfactual_regret"],),
        ),
    )
    metadata = {
        "promotion_candidate_id": candidate.candidate_id,
        "promotion_candidate_configuration_id": (candidate.configuration_artifact_id),
        "promotion_evaluation_unit_id": unit.unit_id,
        "promotion_target_access_seal_id": unit.target_access_seal_id,
    }
    if metadata_overrides:
        metadata.update(metadata_overrides)
    baseline = object()
    candidate_object = object()
    result = build_prospective_v2_decision_trace_v1(
        trace_name=f"promotion trace {label}",
        protocol_id=unit.protocol_id,
        case_id=unit.case_id,
        session_id=unit.session_id,
        endpoint=unit.endpoint,  # type: ignore[arg-type]
        stack_lock_id=freeze.stack_lock_id,
        root_artifacts=(factual, query),
        stages=stages,
        baseline=baseline,
        candidate=candidate_object,
        deployed=candidate_object if selected else baseline,
        baseline_artifact_id=baseline_prediction.artifact_id,
        candidate_artifact_id=candidate_prediction.artifact_id,
        metadata=metadata,
    )
    return result.trace


def _metric_values(
    freeze: ProspectiveV2PromotionFreezeV1,
    opening,
    unit: ProspectiveV2EvaluationUnitV1,
    candidate: ProspectiveV2CandidateV1,
    trace,
    *,
    candidate_log_score: float = 1.2,
    candidate_brier_score: float = 0.15,
    candidate_trajectory_error_m: float = 0.09,
    candidate_coverage: float = 0.9,
    baseline_interval_width_m: float = 1.0,
    candidate_interval_width_m: float = 0.95,
) -> ProspectiveV2UnitMetricValuesV1:
    return ProspectiveV2UnitMetricValuesV1(
        opening_id=opening.opening_id,
        unit_binding_id=unit.unit_binding_id,
        candidate_binding_id=candidate.candidate_binding_id,
        target_artifact_id=unit.target_artifact_id,
        baseline_prediction_artifact_id=(trace.selection.baseline_artifact_id),
        candidate_prediction_artifact_id=(trace.selection.candidate_artifact_id),
        metric_contract_id=freeze.metric_contract.metric_contract_id,
        scoring_run_artifact_id=_id(
            f"scoring-run:{unit.unit_id}:{candidate.candidate_id}"
        ),
        baseline_log_score=1.0,
        candidate_log_score=candidate_log_score,
        baseline_brier_score=0.2,
        candidate_brier_score=candidate_brier_score,
        baseline_trajectory_error_m=0.1,
        candidate_trajectory_error_m=candidate_trajectory_error_m,
        baseline_coverage=0.85,
        candidate_coverage=candidate_coverage,
        baseline_interval_width_m=baseline_interval_width_m,
        candidate_interval_width_m=candidate_interval_width_m,
    )


def _evaluations(
    freeze: ProspectiveV2PromotionFreezeV1,
    opening,
):
    values = []
    for unit in freeze.evaluation_units:
        for candidate in freeze.candidates[1:]:
            trace = _trace(freeze, unit, candidate)
            metrics = _metric_values(
                freeze,
                opening,
                unit,
                candidate,
                trace,
            )
            values.append(
                build_prospective_v2_unit_evaluation_v1(
                    freeze,
                    opening,
                    unit_id=unit.unit_id,
                    candidate_id=candidate.candidate_id,
                    trace=trace,
                    metric_values=metrics,
                )
            )
    return tuple(values)


def test_complete_bound_panel_selects_highest_passing_candidate() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluations = _evaluations(freeze, opening)

    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        opening,
        evaluations,
    )

    assert result.selected_candidate_kind == "sparse_contact_patch"
    assert result.selected_candidate_id == freeze.candidates[-1].candidate_id
    assert result.selection_panel_role == "candidate_selection_only"
    assert result.unbiased_post_selection_performance_claimed is False
    assert result.independent_confirmation_required is True
    assert (
        validate_prospective_v2_promotion_result_v1(
            result,
            freeze,
            opening,
            evaluations,
        )
        is result
    )


def test_acceptance_fallback_and_harm_are_derived_from_bound_sources() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    unit = freeze.evaluation_units[0]
    candidate = freeze.candidates[1]
    rejected_trace = _trace(freeze, unit, candidate, selected=False)
    rejected_metrics = _metric_values(
        freeze,
        opening,
        unit,
        candidate,
        rejected_trace,
        candidate_trajectory_error_m=0.2,
    )
    rejected = build_prospective_v2_unit_evaluation_v1(
        freeze,
        opening,
        unit_id=unit.unit_id,
        candidate_id=candidate.candidate_id,
        trace=rejected_trace,
        metric_values=rejected_metrics,
    )

    assert rejected.candidate_selected is False
    assert rejected.fallback_used is True
    assert rejected.candidate_harmful is True
    assert rejected.harmful_accepted_update is False
    assert "candidate_accepted" not in rejected_metrics.as_dict()
    assert "fallback_used" not in rejected_metrics.as_dict()
    assert "harmful_update" not in rejected_metrics.as_dict()

    selected_trace = _trace(freeze, unit, candidate, selected=True)
    selected_metrics = _metric_values(
        freeze,
        opening,
        unit,
        candidate,
        selected_trace,
        candidate_trajectory_error_m=0.2,
    )
    selected = build_prospective_v2_unit_evaluation_v1(
        freeze,
        opening,
        unit_id=unit.unit_id,
        candidate_id=candidate.candidate_id,
        trace=selected_trace,
        metric_values=selected_metrics,
    )
    assert selected.harmful_accepted_update is True


def test_trace_candidate_and_unit_bindings_fail_closed() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    unit = freeze.evaluation_units[0]
    candidate = freeze.candidates[1]
    trace = _trace(
        freeze,
        unit,
        candidate,
        metadata_overrides={
            "promotion_candidate_configuration_id": _id("other-configuration")
        },
    )
    metrics = _metric_values(freeze, opening, unit, candidate, trace)

    with pytest.raises(ValueError, match="candidate_configuration_id"):
        build_prospective_v2_unit_evaluation_v1(
            freeze,
            opening,
            unit_id=unit.unit_id,
            candidate_id=candidate.candidate_id,
            trace=trace,
            metric_values=metrics,
        )


def test_metric_target_prediction_and_contract_bindings_fail_closed() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    unit = freeze.evaluation_units[0]
    candidate = freeze.candidates[1]
    trace = _trace(freeze, unit, candidate)
    metrics = _metric_values(freeze, opening, unit, candidate, trace)

    for changed, message in (
        (
            replace(metrics, target_artifact_id=_id("different-target")),
            "registered evaluation sources",
        ),
        (
            replace(
                metrics,
                candidate_prediction_artifact_id=_id("different-prediction"),
            ),
            "registered evaluation sources",
        ),
        (
            replace(metrics, metric_contract_id=_id("different-contract")),
            "registered evaluation sources",
        ),
    ):
        with pytest.raises(ValueError, match=message):
            build_prospective_v2_unit_evaluation_v1(
                freeze,
                opening,
                unit_id=unit.unit_id,
                candidate_id=candidate.candidate_id,
                trace=trace,
                metric_values=changed,
            )


def test_target_opening_must_equal_the_complete_frozen_inventory() -> None:
    freeze = _freeze()
    opening = ProspectiveV2TargetOpeningV1(
        freeze_id=freeze.freeze_id,
        target_access_seal_id=freeze.target_access_seal_id,
        target_artifact_ids=(freeze.evaluation_units[0].target_artifact_id,),
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )

    with pytest.raises(ValueError, match="frozen inventory"):
        evaluate_prospective_v2_promotion_v1(
            freeze,
            opening,
            (),
        )


def test_promotion_requires_complete_unit_candidate_product() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluations = _evaluations(freeze, opening)

    with pytest.raises(ValueError, match="unit/candidate product"):
        evaluate_prospective_v2_promotion_v1(
            freeze,
            opening,
            evaluations[:-1],
        )


def test_revalidation_rejects_result_against_changed_source_metrics() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    evaluations = _evaluations(freeze, opening)
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        opening,
        evaluations,
    )
    first = evaluations[0]
    changed_values = replace(
        first.metric_values,
        candidate_log_score=first.metric_values.candidate_log_score + 0.5,
    )
    changed_first = replace(first, metric_values=changed_values)
    changed_evaluations = (changed_first, *evaluations[1:])

    with pytest.raises(ValueError, match="bound source evidence"):
        validate_prospective_v2_promotion_result_v1(
            result,
            freeze,
            opening,
            changed_evaluations,
        )


def test_freeze_rejects_duplicate_independent_groups_and_target_metadata() -> None:
    freeze = _freeze()
    duplicate = replace(
        freeze.evaluation_units[1],
        endpoint=freeze.evaluation_units[0].endpoint,
        independent_group_id=freeze.evaluation_units[0].independent_group_id,
    )
    with pytest.raises(ValueError, match="independent_group_id"):
        replace(
            freeze,
            evaluation_units=(
                freeze.evaluation_units[0],
                duplicate,
                freeze.evaluation_units[2],
            ),
        )
    with pytest.raises(ValueError, match="forbidden before target opening"):
        replace(
            freeze.candidates[1],
            metadata={"target_metric": 0.1},
        )


def test_trace_root_artifacts_must_match_registered_contexts() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    unit = freeze.evaluation_units[0]
    candidate = freeze.candidates[1]
    trace = _trace(
        freeze,
        unit,
        candidate,
        wrong_factual_context=True,
    )
    metrics = _metric_values(freeze, opening, unit, candidate, trace)

    with pytest.raises(ValueError, match="factual-context binding"):
        build_prospective_v2_unit_evaluation_v1(
            freeze,
            opening,
            unit_id=unit.unit_id,
            candidate_id=candidate.candidate_id,
            trace=trace,
            metric_values=metrics,
        )


def test_interval_width_ratio_uses_the_frozen_policy_floor() -> None:
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="independent-evaluator",
    )
    unit = freeze.evaluation_units[0]
    candidate = freeze.candidates[1]
    trace = _trace(freeze, unit, candidate)
    metrics = _metric_values(
        freeze,
        opening,
        unit,
        candidate,
        trace,
        baseline_interval_width_m=0.0,
        candidate_interval_width_m=1e-10,
    )
    evaluation = build_prospective_v2_unit_evaluation_v1(
        freeze,
        opening,
        unit_id=unit.unit_id,
        candidate_id=candidate.candidate_id,
        trace=trace,
        metric_values=metrics,
    )

    assert evaluation.interval_width_ratio == pytest.approx(0.1)
    assert evaluation.as_dict()["policy_id"] == freeze.policy.policy_id


def test_freeze_requires_complete_target_free_source_inventory() -> None:
    freeze = _freeze()
    with pytest.raises(ValueError, match="target-free source"):
        replace(
            freeze,
            source_artifact_ids=tuple(
                value
                for value in freeze.source_artifact_ids
                if value != freeze.stack_lock_id
            ),
        )


def test_target_opening_timestamp_must_use_utc() -> None:
    freeze = _freeze()
    with pytest.raises(ValueError, match="must use UTC"):
        build_prospective_v2_target_opening_v1(
            freeze,
            opened_at_utc="2026-08-09T02:00:00+02:00",
            opened_by="independent-evaluator",
        )
