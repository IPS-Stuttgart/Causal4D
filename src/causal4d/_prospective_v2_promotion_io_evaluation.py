"""Strict persistence and replay for V2 unit scores and results."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from causal4d._prospective_v2_promotion_evidence import (
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2TargetOpeningV1,
    ProspectiveV2UnitEvaluationV1,
    ProspectiveV2UnitMetricValuesV1,
    build_prospective_v2_unit_evaluation_v1,
)
from causal4d._prospective_v2_promotion_io_common import (
    load_object as _load_object,
    require_expected_identity as _require_expected_identity,
    require_fields as _require_fields,
    require_mapping as _require_mapping,
    require_schema as _require_schema,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.decision_trace import UnifiedDecisionTrace
from causal4d.prospective_v2_promotion import (
    ProspectiveV2PromotionResultV1,
    evaluate_prospective_v2_promotion_v1,
)

_METRIC_VALUES_FIELDS = frozenset(
    {
        "schema_version", "artifact_kind", "opening_id", "unit_binding_id",
        "candidate_binding_id", "target_artifact_id",
        "baseline_prediction_artifact_id", "candidate_prediction_artifact_id",
        "metric_contract_id", "scoring_run_artifact_id", "baseline_log_score",
        "candidate_log_score", "baseline_brier_score", "candidate_brier_score",
        "baseline_trajectory_error_m", "candidate_trajectory_error_m",
        "baseline_coverage", "candidate_coverage", "baseline_interval_width_m",
        "candidate_interval_width_m", "target_outcomes_used", "metadata",
        "metric_values_id",
    }
)
_UNIT_EVALUATION_FIELDS = frozenset(
    {
        "schema_version", "artifact_kind", "freeze_id", "stack_lock_id",
        "opening_id", "unit_binding_id", "candidate_binding_id",
        "metric_contract_id", "policy_id", "metric_values_id", "trace_id",
        "trace_validation_id", "unit_id", "endpoint", "candidate_id",
        "candidate_kind", "baseline_prediction_artifact_id",
        "candidate_prediction_artifact_id", "deployed_prediction_artifact_id",
        "candidate_selected", "fallback_used", "candidate_harmful",
        "harmful_accepted_update", "log_score_gain", "brier_change",
        "trajectory_regret_m", "candidate_coverage_error",
        "candidate_interval_width_m", "interval_width_ratio",
        "target_outcomes_used", "evaluation_id",
    }
)
_RESULT_FIELDS = frozenset(
    {
        "schema_version", "artifact_kind", "freeze_id", "opening_id",
        "baseline_candidate_id", "baseline_configuration_artifact_id",
        "selected_candidate_id", "selected_candidate_kind",
        "selected_configuration_artifact_id", "candidate_results",
        "evaluation_ids", "selection_panel_role",
        "unbiased_post_selection_performance_claimed",
        "independent_confirmation_required", "target_outcomes_used", "metadata",
        "result_id",
    }
)


def write_prospective_v2_unit_metric_values(
    path: str | Path,
    metric_values: ProspectiveV2UnitMetricValuesV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one raw, artifact-bound unit score record."""

    if type(metric_values) is not ProspectiveV2UnitMetricValuesV1:
        raise ValueError("metric_values has the wrong type")
    atomic_write_json(path, metric_values.as_dict(), overwrite=overwrite)


def load_prospective_v2_unit_metric_values(
    path: str | Path,
    *,
    expected_metric_values_id: str | None = None,
) -> ProspectiveV2UnitMetricValuesV1:
    """Load raw unit scores while enforcing their complete source bindings."""

    fields = _require_fields(
        _load_object(path, name="prospective V2 unit metric values"),
        expected=_METRIC_VALUES_FIELDS,
        name="prospective V2 unit metric values",
    )
    _require_schema(
        fields,
        artifact_kind="Causal4DProspectiveV2UnitMetricValuesV1",
        name="prospective V2 unit metric values",
    )
    result = ProspectiveV2UnitMetricValuesV1(
        opening_id=fields["opening_id"],
        unit_binding_id=fields["unit_binding_id"],
        candidate_binding_id=fields["candidate_binding_id"],
        target_artifact_id=fields["target_artifact_id"],
        baseline_prediction_artifact_id=(
            fields["baseline_prediction_artifact_id"]
        ),
        candidate_prediction_artifact_id=(
            fields["candidate_prediction_artifact_id"]
        ),
        metric_contract_id=fields["metric_contract_id"],
        scoring_run_artifact_id=fields["scoring_run_artifact_id"],
        baseline_log_score=fields["baseline_log_score"],
        candidate_log_score=fields["candidate_log_score"],
        baseline_brier_score=fields["baseline_brier_score"],
        candidate_brier_score=fields["candidate_brier_score"],
        baseline_trajectory_error_m=fields["baseline_trajectory_error_m"],
        candidate_trajectory_error_m=fields["candidate_trajectory_error_m"],
        baseline_coverage=fields["baseline_coverage"],
        candidate_coverage=fields["candidate_coverage"],
        baseline_interval_width_m=fields["baseline_interval_width_m"],
        candidate_interval_width_m=fields["candidate_interval_width_m"],
        target_outcomes_used=fields["target_outcomes_used"],
        metadata=_require_mapping(fields["metadata"], name="metric-value metadata"),
    )
    if fields["metric_values_id"] != result.metric_values_id:
        raise ValueError("prospective V2 metric-values identity changed")
    _require_expected_identity(
        result.metric_values_id,
        expected_metric_values_id,
        name="metric_values_id",
    )
    return result


def write_prospective_v2_unit_evaluation(
    path: str | Path,
    evaluation: ProspectiveV2UnitEvaluationV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish one derived source-bound unit evaluation."""

    if type(evaluation) is not ProspectiveV2UnitEvaluationV1:
        raise ValueError("evaluation has the wrong type")
    atomic_write_json(path, evaluation.as_dict(), overwrite=overwrite)


def load_prospective_v2_unit_evaluation(
    path: str | Path,
    freeze: ProspectiveV2PromotionFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    trace: UnifiedDecisionTrace,
    metric_values: ProspectiveV2UnitMetricValuesV1,
    *,
    expected_evaluation_id: str | None = None,
) -> ProspectiveV2UnitEvaluationV1:
    """Reload and recompute one unit evaluation from its exact source objects."""

    fields = _require_fields(
        _load_object(path, name="prospective V2 unit evaluation"),
        expected=_UNIT_EVALUATION_FIELDS,
        name="prospective V2 unit evaluation",
    )
    _require_schema(
        fields,
        artifact_kind="Causal4DProspectiveV2UnitEvaluationV1",
        name="prospective V2 unit evaluation",
    )
    result = build_prospective_v2_unit_evaluation_v1(
        freeze,
        opening,
        unit_id=fields["unit_id"],
        candidate_id=fields["candidate_id"],
        trace=trace,
        metric_values=metric_values,
    )
    if dict(fields) != result.as_dict():
        raise ValueError(
            "prospective V2 unit evaluation does not match its bound sources"
        )
    _require_expected_identity(
        result.evaluation_id,
        expected_evaluation_id,
        name="evaluation_id",
    )
    return result


def load_prospective_v2_promotion_result(
    path: str | Path,
    freeze: ProspectiveV2PromotionFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
    *,
    expected_result_id: str | None = None,
) -> ProspectiveV2PromotionResultV1:
    """Reload and recompute a selection result from every bound evaluation."""

    fields = _require_fields(
        _load_object(path, name="prospective V2 promotion result"),
        expected=_RESULT_FIELDS,
        name="prospective V2 promotion result",
    )
    _require_schema(
        fields,
        artifact_kind="Causal4DProspectiveV2PromotionResultV1",
        name="prospective V2 promotion result",
    )
    result = evaluate_prospective_v2_promotion_v1(
        freeze,
        opening,
        evaluations,
        metadata=_require_mapping(fields["metadata"], name="result metadata"),
    )
    if dict(fields) != result.as_dict():
        raise ValueError(
            "prospective V2 promotion result does not match its bound sources"
        )
    _require_expected_identity(
        result.result_id,
        expected_result_id,
        name="result_id",
    )
    return result
