"""Fail-closed validator for the secondary PokeFlex anytime protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_SCHEMA = "causal4d/pokeflex-adaptive-stopping-protocol"
_POLICIES = (
    "no-probe",
    "source-fixed-probe-order",
    "random-safe-with-stopping",
    "greedy-generic-mutual-information-with-stopping",
    "greedy-task-conditioned-regret-reduction-with-stopping",
    "dependence-destroyed-task-policy",
    "oracle-diagnostic-only",
)
_ENDPOINTS = {
    "object-balanced-downstream-decision-regret",
    "area-under-regret-versus-probe-cost-curve",
    "mean-probe-count",
    "decision-certification-rate",
    "fallback-rate",
    "harmful-nonfallback-rate",
}


def load_adaptive_stopping_protocol(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_adaptive_stopping_protocol(payload)
    return payload


def _true(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"adaptive protocol requires {key}=true")


def _false(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not False:
        raise ValueError(f"adaptive protocol requires {key}=false")


def validate_adaptive_stopping_protocol(protocol: Mapping[str, Any]) -> None:
    if protocol.get("schema") != _SCHEMA or protocol.get("schema_version") != 1:
        raise ValueError("unexpected adaptive-stopping schema")
    if protocol.get("role") != "secondary-after-positive-one-probe-primary":
        raise ValueError("adaptive study cannot replace the one-probe primary")
    if protocol.get("maximum_probe_count") != 2:
        raise ValueError("maximum probe count changed")
    if tuple(protocol.get("policy_panel", ())) != _POLICIES:
        raise ValueError("adaptive policy panel changed")
    if set(protocol.get("primary_endpoints", ())) != _ENDPOINTS:
        raise ValueError("adaptive endpoint panel changed")

    stopping = protocol["stopping_rule"]
    for key in (
        "act_when_worst_case_decision_regret_at_most_epsilon",
        "probe_only_when_expected_regret_reduction_exceeds_cost",
        "fallback_when_no_safe_positive_value_probe_exists",
        "epsilon_selected_on_source_objects_only",
        "probe_cost_weight_selected_on_source_objects_only",
    ):
        _true(stopping, key)

    custody = protocol["custody"]
    for key in (
        "same_joint_prediction_seal_as_one_probe_study",
        "maximum_two_selected_responses_per_policy_query",
        "unselected_response_reads_forbidden",
        "challenge_outcomes_forbidden_before_prediction_seal",
        "target_policy_or_threshold_changes_forbidden",
        "target_retry_forbidden",
    ):
        _true(custody, key)

    source = protocol["source_gate"]
    if float(source.get("minimum_regret_cost_auc_gain_vs_generic_information", 0)) < 0.02:
        raise ValueError("source regret-cost gate weakened")
    if float(source.get("maximum_mean_probe_count_ratio_vs_generic_information", 99)) > 1.0:
        raise ValueError("source probe-efficiency gate weakened")
    if float(source.get("minimum_dependence_placebo_advantage_collapse_fraction", 0)) < 0.5:
        raise ValueError("dependence-collapse gate weakened")
    if int(source.get("minimum_probe_count_diversity", 0)) < 2:
        raise ValueError("adaptive stopping must exercise multiple probe counts")

    target = protocol["target_success"]
    for key in (
        "positive_object_balanced_regret_gain_vs_fixed_order",
        "positive_object_balanced_regret_gain_vs_generic_information",
        "positive_regret_cost_auc_gain_vs_generic_information",
        "mean_probe_count_no_greater_than_generic_information",
    ):
        _true(target, key)
    if int(target.get("minimum_certified_objects_out_of_nine", 0)) < 7:
        raise ValueError("target certification gate weakened")
    if int(target.get("maximum_harmful_nonfallback_objects", 99)) > 1:
        raise ValueError("target harm gate weakened")

    stats = protocol["statistics"]
    if stats.get("primary_unit") != "physical-object":
        raise ValueError("physical object must remain the inferential unit")
    for key in (
        "queries_reported_separately",
        "paired_object_bootstrap",
        "exact_object_sign_test",
        "frame_pseudoreplication_forbidden",
    ):
        _true(stats, key)

    boundary = protocol["claim_boundary"]
    _true(boundary, "offline_reset_matched_multi_observation_acquisition")
    for key in (
        "physical_sequential_execution_claim",
        "identical_state_after_first_probe_claim",
        "deployment_safety_claim",
        "general_state_of_the_art_claim",
    ):
        _false(boundary, key)
