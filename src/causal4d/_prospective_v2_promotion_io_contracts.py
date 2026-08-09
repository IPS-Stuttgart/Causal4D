"""Closed-schema parsers for target-free prospective V2 contracts."""

from __future__ import annotations

from typing import Any

from causal4d._prospective_v2_promotion_evidence import (
    PROSPECTIVE_V2_METRIC_SEMANTICS,
    PROSPECTIVE_V2_SELECTION_PANEL_ROLE,
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2MetricContractV1,
    ProspectiveV2PromotionPolicyV1,
)
from causal4d._prospective_v2_promotion_io_common import (
    require_fields,
    require_mapping,
    require_schema,
)

_METRIC_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "scoring_implementation_artifact_id",
        "metric_semantics",
        "nominal_coverage",
        "harmful_regret_threshold_m",
        "selection_panel_role",
        "unbiased_post_selection_performance_claimed",
        "independent_confirmation_required",
        "target_outcomes_used",
        "metadata",
        "metric_contract_id",
    }
)
_POLICY_FIELDS = frozenset(
    {
        "minimum_units_per_endpoint",
        "minimum_mean_log_score_gain",
        "maximum_mean_brier_change",
        "maximum_mean_trajectory_regret_m",
        "maximum_mean_coverage_error",
        "maximum_mean_interval_width_ratio",
        "minimum_accepted_update_rate",
        "maximum_harmful_accepted_update_rate",
        "maximum_fallback_rate",
        "interval_width_floor_m",
        "policy_id",
    }
)
_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "candidate_kind",
        "configuration_artifact_id",
        "target_outcomes_used",
        "metadata",
        "candidate_binding_id",
    }
)
_UNIT_FIELDS = frozenset(
    {
        "unit_id",
        "endpoint",
        "protocol_id",
        "case_id",
        "session_id",
        "independent_group_id",
        "target_artifact_id",
        "factual_context_artifact_id",
        "counterfactual_query_artifact_id",
        "target_access_seal_id",
        "target_outcomes_used",
        "metadata",
        "unit_binding_id",
    }
)


def parse_metric_contract(values: Any) -> ProspectiveV2MetricContractV1:
    fields = require_fields(
        values,
        expected=_METRIC_CONTRACT_FIELDS,
        name="prospective V2 metric contract",
    )
    require_schema(
        fields,
        artifact_kind="Causal4DProspectiveV2MetricContractV1",
        name="prospective V2 metric contract",
    )
    if fields["metric_semantics"] != dict(PROSPECTIVE_V2_METRIC_SEMANTICS):
        raise ValueError("prospective V2 metric semantics changed")
    if fields["selection_panel_role"] != PROSPECTIVE_V2_SELECTION_PANEL_ROLE:
        raise ValueError("prospective V2 metric selection-panel role changed")
    if fields["unbiased_post_selection_performance_claimed"] is not False:
        raise ValueError("prospective V2 metric contract claims post-selection bias")
    if fields["independent_confirmation_required"] is not True:
        raise ValueError("prospective V2 metric contract dropped confirmation")
    if fields["target_outcomes_used"] is not False:
        raise ValueError("prospective V2 metric contract is not target-free")
    result = ProspectiveV2MetricContractV1(
        scoring_implementation_artifact_id=fields["scoring_implementation_artifact_id"],
        nominal_coverage=fields["nominal_coverage"],
        harmful_regret_threshold_m=fields["harmful_regret_threshold_m"],
        target_outcomes_used=fields["target_outcomes_used"],
        metadata=require_mapping(fields["metadata"], name="metric metadata"),
    )
    if fields["metric_contract_id"] != result.metric_contract_id:
        raise ValueError("prospective V2 metric-contract identity changed")
    return result


def parse_policy(values: Any) -> ProspectiveV2PromotionPolicyV1:
    fields = require_fields(
        values,
        expected=_POLICY_FIELDS,
        name="prospective V2 promotion policy",
    )
    result = ProspectiveV2PromotionPolicyV1(
        minimum_units_per_endpoint=fields["minimum_units_per_endpoint"],
        minimum_mean_log_score_gain=fields["minimum_mean_log_score_gain"],
        maximum_mean_brier_change=fields["maximum_mean_brier_change"],
        maximum_mean_trajectory_regret_m=fields["maximum_mean_trajectory_regret_m"],
        maximum_mean_coverage_error=fields["maximum_mean_coverage_error"],
        maximum_mean_interval_width_ratio=fields["maximum_mean_interval_width_ratio"],
        minimum_accepted_update_rate=fields["minimum_accepted_update_rate"],
        maximum_harmful_accepted_update_rate=fields[
            "maximum_harmful_accepted_update_rate"
        ],
        maximum_fallback_rate=fields["maximum_fallback_rate"],
        interval_width_floor_m=fields["interval_width_floor_m"],
    )
    if fields["policy_id"] != result.policy_id:
        raise ValueError("prospective V2 promotion-policy identity changed")
    return result


def parse_candidate(values: Any) -> ProspectiveV2CandidateV1:
    fields = require_fields(
        values,
        expected=_CANDIDATE_FIELDS,
        name="prospective V2 candidate",
    )
    result = ProspectiveV2CandidateV1(
        candidate_id=fields["candidate_id"],
        candidate_kind=fields["candidate_kind"],
        configuration_artifact_id=fields["configuration_artifact_id"],
        target_outcomes_used=fields["target_outcomes_used"],
        metadata=require_mapping(fields["metadata"], name="candidate metadata"),
    )
    if fields["candidate_binding_id"] != result.candidate_binding_id:
        raise ValueError("prospective V2 candidate identity changed")
    return result


def parse_unit(values: Any) -> ProspectiveV2EvaluationUnitV1:
    fields = require_fields(
        values,
        expected=_UNIT_FIELDS,
        name="prospective V2 evaluation unit",
    )
    result = ProspectiveV2EvaluationUnitV1(
        unit_id=fields["unit_id"],
        endpoint=fields["endpoint"],
        protocol_id=fields["protocol_id"],
        case_id=fields["case_id"],
        session_id=fields["session_id"],
        independent_group_id=fields["independent_group_id"],
        target_artifact_id=fields["target_artifact_id"],
        factual_context_artifact_id=fields["factual_context_artifact_id"],
        counterfactual_query_artifact_id=fields["counterfactual_query_artifact_id"],
        target_access_seal_id=fields["target_access_seal_id"],
        target_outcomes_used=fields["target_outcomes_used"],
        metadata=require_mapping(fields["metadata"], name="unit metadata"),
    )
    if fields["unit_binding_id"] != result.unit_binding_id:
        raise ValueError("prospective V2 evaluation-unit identity changed")
    return result
