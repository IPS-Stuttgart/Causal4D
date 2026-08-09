from __future__ import annotations

from dataclasses import replace
import hashlib

import pytest

from causal4d.prospective_v2_confirmation import (
    PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE,
    build_prospective_v2_confirmation_freeze_v1,
    build_prospective_v2_confirmation_opening_v1,
    build_prospective_v2_confirmation_unit_evaluation_v1,
    evaluate_prospective_v2_confirmation_v1,
    validate_prospective_v2_confirmation_freeze_v1,
    validate_prospective_v2_confirmation_opening_v1,
    validate_prospective_v2_confirmation_result_v1,
)
from causal4d.prospective_v2_promotion import (
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2UnitEvaluationV1,
    build_prospective_v2_target_opening_v1,
    build_prospective_v2_unit_evaluation_v1,
    evaluate_prospective_v2_promotion_v1,
)
from tests.test_prospective_v2_promotion import (
    _evaluations,
    _freeze,
    _metric_values,
    _trace,
)


def _id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _selection_evidence():
    freeze = _freeze()
    opening = build_prospective_v2_target_opening_v1(
        freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="selection-evaluator",
    )
    evaluations = _evaluations(freeze, opening)
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        opening,
        evaluations,
    )
    return freeze, opening, evaluations, result


def _confirmation_units(
    *,
    seal_id: str | None = None,
) -> tuple[ProspectiveV2EvaluationUnitV1, ...]:
    seal = _id("confirmation-target-access-seal") if seal_id is None else seal_id
    return tuple(
        ProspectiveV2EvaluationUnitV1(
            unit_id=f"confirmation-unit-{index}",
            endpoint=endpoint,
            protocol_id="protocol-v2-confirmation",
            case_id=f"confirmation-case-{index}",
            session_id=f"confirmation-session-{index}",
            independent_group_id=f"confirmation-group-{index}",
            target_artifact_id=_id(f"confirmation-target:{index}"),
            factual_context_artifact_id=_id(f"confirmation-factual:{index}"),
            counterfactual_query_artifact_id=_id(f"confirmation-query:{index}"),
            target_access_seal_id=seal,
        )
        for index, endpoint in enumerate(
            (
                "factual_continuation",
                "same_grasp_transfer",
                "new_contact_transfer",
            )
        )
    )


def _confirmation_evidence():
    selection_freeze, selection_opening, selection_evaluations, selection_result = (
        _selection_evidence()
    )
    confirmation_freeze = build_prospective_v2_confirmation_freeze_v1(
        selection_result,
        selection_freeze,
        selection_opening,
        selection_evaluations,
        experiment_id="prospective-v2-independent-confirmation",
        target_access_seal_id=_id("confirmation-target-access-seal"),
        evaluation_units=_confirmation_units(),
    )
    confirmation_opening = build_prospective_v2_confirmation_opening_v1(
        confirmation_freeze,
        opened_at_utc="2026-08-10T00:00:00+00:00",
        opened_by="independent-confirmation-evaluator",
    )
    evaluations = []
    candidate = confirmation_freeze.selected_candidate
    for unit in confirmation_freeze.evaluation_units:
        trace = _trace(
            selection_freeze,
            unit,
            candidate,
            metadata_overrides={
                "confirmation_freeze_id": (confirmation_freeze.confirmation_freeze_id),
                "confirmation_selection_result_id": (
                    confirmation_freeze.selection_result_id
                ),
                "confirmation_panel_role": PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE,
            },
        )
        metrics = _metric_values(
            selection_freeze,
            confirmation_opening,
            unit,
            candidate,
            trace,
        )
        evaluations.append(
            build_prospective_v2_confirmation_unit_evaluation_v1(
                confirmation_freeze,
                confirmation_opening,
                unit_id=unit.unit_id,
                trace=trace,
                metric_values=metrics,
            )
        )
    return (
        selection_freeze,
        selection_opening,
        selection_evaluations,
        selection_result,
        confirmation_freeze,
        confirmation_opening,
        tuple(evaluations),
    )


def test_disjoint_confirmation_passes_without_reselecting_candidates() -> None:
    (
        _,
        _,
        _,
        selection_result,
        freeze,
        opening,
        evaluations,
    ) = _confirmation_evidence()

    result = evaluate_prospective_v2_confirmation_v1(
        freeze,
        opening,
        evaluations,
    )

    assert result.selection_result_id == selection_result.result_id
    assert result.confirmation_passed
    assert result.fixed_candidate_confirmation_gate_passed
    assert result.deployed_candidate_id == freeze.selected_candidate.candidate_id
    assert result.as_dict()["candidate_selection_performed"] is False
    assert result.as_dict()["selection_panel_performance_reused"] is False
    assert (
        validate_prospective_v2_confirmation_result_v1(
            result,
            freeze,
            opening,
            evaluations,
        )
        is result
    )


def test_confirmation_freeze_revalidates_against_selection_evidence() -> None:
    (
        selection_freeze,
        selection_opening,
        selection_evaluations,
        _,
        freeze,
        _,
        _,
    ) = _confirmation_evidence()

    assert (
        validate_prospective_v2_confirmation_freeze_v1(
            freeze,
            evaluate_prospective_v2_promotion_v1(
                selection_freeze,
                selection_opening,
                selection_evaluations,
            ),
            selection_freeze,
            selection_opening,
            selection_evaluations,
        )
        is freeze
    )
    changed = replace(
        freeze,
        selection_result_id=_id("different-selection-result"),
        bound_artifact_ids=(
            _id("different-selection-result"),
            *tuple(
                value
                for value in freeze.bound_artifact_ids
                if value != freeze.selection_result_id
            ),
        ),
    )
    selection_result = evaluate_prospective_v2_promotion_v1(
        selection_freeze,
        selection_opening,
        selection_evaluations,
    )
    with pytest.raises(ValueError, match="selection evidence"):
        validate_prospective_v2_confirmation_freeze_v1(
            changed,
            selection_result,
            selection_freeze,
            selection_opening,
            selection_evaluations,
        )


def test_confirmation_freeze_rejects_selection_panel_overlap() -> None:
    selection_freeze, selection_opening, selection_evaluations, selection_result = (
        _selection_evidence()
    )
    units = list(_confirmation_units())
    units[0] = replace(
        units[0],
        target_artifact_id=selection_freeze.evaluation_units[0].target_artifact_id,
    )

    with pytest.raises(ValueError, match="target artifacts"):
        build_prospective_v2_confirmation_freeze_v1(
            selection_result,
            selection_freeze,
            selection_opening,
            selection_evaluations,
            experiment_id="overlapping-confirmation",
            target_access_seal_id=_id("confirmation-target-access-seal"),
            evaluation_units=tuple(units),
        )

    with pytest.raises(ValueError, match="new target-access seal"):
        build_prospective_v2_confirmation_freeze_v1(
            selection_result,
            selection_freeze,
            selection_opening,
            selection_evaluations,
            experiment_id="reused-seal-confirmation",
            target_access_seal_id=selection_freeze.target_access_seal_id,
            evaluation_units=_confirmation_units(
                seal_id=selection_freeze.target_access_seal_id
            ),
        )


def test_confirmation_rejects_relabelled_unit_group_and_session_overlap() -> None:
    selection_freeze, selection_opening, selection_evaluations, selection_result = (
        _selection_evidence()
    )
    selection_unit = selection_freeze.evaluation_units[0]
    confirmation_unit = _confirmation_units()[0]
    changes = (
        (
            replace(confirmation_unit, unit_id=selection_unit.unit_id),
            "unit IDs",
        ),
        (
            replace(
                confirmation_unit,
                independent_group_id=selection_unit.independent_group_id,
            ),
            "endpoint independence groups",
        ),
        (
            replace(
                confirmation_unit,
                protocol_id=selection_unit.protocol_id,
                session_id=selection_unit.session_id,
            ),
            "protocol sessions",
        ),
    )
    for changed, message in changes:
        units = (changed, *_confirmation_units()[1:])
        with pytest.raises(ValueError, match=message):
            build_prospective_v2_confirmation_freeze_v1(
                selection_result,
                selection_freeze,
                selection_opening,
                selection_evaluations,
                experiment_id="relabelled-overlap",
                target_access_seal_id=_id("confirmation-target-access-seal"),
                evaluation_units=units,
            )


def test_confirmation_requires_a_nonbaseline_selected_candidate() -> None:
    selection_freeze = _freeze()
    selection_opening = build_prospective_v2_target_opening_v1(
        selection_freeze,
        opened_at_utc="2026-08-09T00:00:00+00:00",
        opened_by="selection-evaluator",
    )
    evaluations = []
    for unit in selection_freeze.evaluation_units:
        for candidate in selection_freeze.candidates[1:]:
            trace = _trace(selection_freeze, unit, candidate, selected=False)
            metrics = _metric_values(
                selection_freeze,
                selection_opening,
                unit,
                candidate,
                trace,
            )
            evaluations.append(
                build_prospective_v2_unit_evaluation_v1(
                    selection_freeze,
                    selection_opening,
                    unit_id=unit.unit_id,
                    candidate_id=candidate.candidate_id,
                    trace=trace,
                    metric_values=metrics,
                )
            )
    selection_result = evaluate_prospective_v2_promotion_v1(
        selection_freeze,
        selection_opening,
        tuple(evaluations),
    )
    assert selection_result.selected_candidate_kind == "registered_baseline"

    with pytest.raises(ValueError, match="did not select a candidate"):
        build_prospective_v2_confirmation_freeze_v1(
            selection_result,
            selection_freeze,
            selection_opening,
            tuple(evaluations),
            experiment_id="no-candidate-confirmation",
            target_access_seal_id=_id("confirmation-target-access-seal"),
            evaluation_units=_confirmation_units(),
        )


def test_confirmation_opening_is_bound_to_the_complete_inventory() -> None:
    (
        _,
        _,
        _,
        _,
        freeze,
        opening,
        _,
    ) = _confirmation_evidence()

    assert validate_prospective_v2_confirmation_opening_v1(opening, freeze) is opening
    changed = replace(
        opening,
        target_artifact_ids=opening.target_artifact_ids[:-1],
    )
    with pytest.raises(ValueError, match="frozen inventory"):
        validate_prospective_v2_confirmation_opening_v1(changed, freeze)


def test_confirmation_trace_requires_explicit_panel_bindings() -> None:
    (
        selection_freeze,
        _,
        _,
        _,
        freeze,
        opening,
        _,
    ) = _confirmation_evidence()
    unit = freeze.evaluation_units[0]
    candidate = freeze.selected_candidate
    trace = _trace(selection_freeze, unit, candidate)
    metrics = _metric_values(
        selection_freeze,
        opening,
        unit,
        candidate,
        trace,
    )

    with pytest.raises(ValueError, match="confirmation_freeze_id"):
        build_prospective_v2_confirmation_unit_evaluation_v1(
            freeze,
            opening,
            unit_id=unit.unit_id,
            trace=trace,
            metric_values=metrics,
        )


def test_confirmation_evaluator_rejects_direct_trace_metadata_bypass() -> None:
    (
        selection_freeze,
        _,
        _,
        _,
        freeze,
        opening,
        evaluations,
    ) = _confirmation_evidence()
    unit = freeze.evaluation_units[0]
    candidate = freeze.selected_candidate
    trace = _trace(selection_freeze, unit, candidate)
    metrics = _metric_values(
        selection_freeze,
        opening,
        unit,
        candidate,
        trace,
    )
    bypass = ProspectiveV2UnitEvaluationV1(
        freeze_id=freeze.confirmation_freeze_id,
        stack_lock_id=freeze.stack_lock_id,
        unit=unit,
        candidate=candidate,
        opening=opening,
        metric_contract=freeze.metric_contract,
        policy=freeze.policy,
        trace=trace,
        metric_values=metrics,
    )

    with pytest.raises(ValueError, match="confirmation_freeze_id"):
        evaluate_prospective_v2_confirmation_v1(
            freeze,
            opening,
            (bypass, *evaluations[1:]),
        )


def test_confirmation_rejects_selection_scoring_run_reuse() -> None:
    (
        selection_freeze,
        _,
        selection_evaluations,
        _,
        freeze,
        opening,
        _,
    ) = _confirmation_evidence()
    unit = freeze.evaluation_units[0]
    candidate = freeze.selected_candidate
    trace = _trace(
        selection_freeze,
        unit,
        candidate,
        metadata_overrides={
            "confirmation_freeze_id": freeze.confirmation_freeze_id,
            "confirmation_selection_result_id": freeze.selection_result_id,
            "confirmation_panel_role": PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE,
        },
    )
    metrics = _metric_values(
        selection_freeze,
        opening,
        unit,
        candidate,
        trace,
    )
    metrics = replace(
        metrics,
        scoring_run_artifact_id=(
            selection_evaluations[0].metric_values.scoring_run_artifact_id
        ),
    )

    with pytest.raises(ValueError, match="selection-panel scoring run"):
        build_prospective_v2_confirmation_unit_evaluation_v1(
            freeze,
            opening,
            unit_id=unit.unit_id,
            trace=trace,
            metric_values=metrics,
        )


def test_failed_confirmation_preserves_exact_baseline() -> None:
    (
        _,
        _,
        _,
        _,
        freeze,
        opening,
        evaluations,
    ) = _confirmation_evidence()
    first = evaluations[0]
    changed_values = replace(
        first.metric_values,
        candidate_log_score=0.0,
        candidate_trajectory_error_m=0.3,
    )
    changed_first = replace(first, metric_values=changed_values)
    changed = (changed_first, *evaluations[1:])

    result = evaluate_prospective_v2_confirmation_v1(
        freeze,
        opening,
        changed,
    )

    assert not result.confirmation_passed
    assert result.deployed_candidate_id == freeze.baseline_candidate.candidate_id
    assert result.deployed_configuration_artifact_id == (
        freeze.baseline_candidate.configuration_artifact_id
    )
    assert result.exact_baseline_fallback_verified


def test_confirmation_requires_the_complete_frozen_panel() -> None:
    (
        _,
        _,
        _,
        _,
        freeze,
        opening,
        evaluations,
    ) = _confirmation_evidence()

    with pytest.raises(ValueError, match="frozen confirmation units"):
        evaluate_prospective_v2_confirmation_v1(
            freeze,
            opening,
            evaluations[:-1],
        )


def test_confirmation_revalidation_rejects_changed_metrics() -> None:
    (
        _,
        _,
        _,
        _,
        freeze,
        opening,
        evaluations,
    ) = _confirmation_evidence()
    result = evaluate_prospective_v2_confirmation_v1(
        freeze,
        opening,
        evaluations,
    )
    first = evaluations[0]
    changed_first = replace(
        first,
        metric_values=replace(
            first.metric_values,
            candidate_brier_score=first.metric_values.candidate_brier_score + 0.1,
        ),
    )

    with pytest.raises(ValueError, match="bound evidence"):
        validate_prospective_v2_confirmation_result_v1(
            result,
            freeze,
            opening,
            (changed_first, *evaluations[1:]),
        )
