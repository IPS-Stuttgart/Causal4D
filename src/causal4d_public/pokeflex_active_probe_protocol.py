"""Fail-closed validator for the PokeFlex active-probe protocol v2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

_SCHEMA = "causal4d/pokeflex-active-probe-protocol"
_REQUIRED_POLICIES = (
    "no-probe",
    "random-safe-expected",
    "source-fixed-safe",
    "generic-mutual-information",
    "action-only-query-heuristic",
    "task-conditioned-decision-value",
    "within-object-dependence-destroyed",
    "matched-wrong-object-response",
    "double-placebo",
    "oracle-diagnostic-only",
)
_REQUIRED_QUERIES = (
    "held-poke-local-response",
    "held-drop-impact-geometry",
    "held-drop-settled-geometry",
)


def load_active_probe_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate one protocol document."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_active_probe_protocol(payload)
    return payload


def _require_true(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not True:
        raise ValueError(f"protocol must require {key}=true")


def _require_false(mapping: Mapping[str, Any], key: str) -> None:
    if mapping.get(key) is not False:
        raise ValueError(f"protocol must require {key}=false")


def validate_active_probe_protocol(protocol: Mapping[str, Any]) -> None:
    """Reject protocol drift that would weaken the registered confirmation."""

    if protocol.get("schema") != _SCHEMA or protocol.get("schema_version") != 2:
        raise ValueError("unexpected active-probe protocol schema")

    dataset = protocol["dataset"]
    if dataset != {
        "name": "PokeFlex",
        "archive_count": 170,
        "poking_archive_count": 116,
        "dropping_archive_count": 54,
        "physical_object_count": 18,
    }:
        raise ValueError("unexpected public PokeFlex dataset contract")

    split = protocol["split"]
    expected_split = {
        "source_object_count": 9,
        "primary_target_object_count": 6,
        "replication_target_object_count": 3,
        "outcome_calibration_object_count": 0,
        "source_hyperparameter_selection": "nested-leave-one-source-object-out",
    }
    for key, value in expected_split.items():
        if split.get(key) != value:
            raise ValueError(f"unexpected split setting {key}")
    if not isinstance(split.get("salt"), str) or not split["salt"]:
        raise ValueError("split salt must be frozen")

    if tuple(protocol["registered_queries"]) != _REQUIRED_QUERIES:
        raise ValueError("registered query panel changed")
    if protocol.get("candidate_probe_count") != 4:
        raise ValueError("candidate probe count changed")
    if tuple(protocol["policy_panel"]) != _REQUIRED_POLICIES:
        raise ValueError("matched policy panel changed")

    initial = protocol["initial_state_admission"]
    if initial.get("certificate_schema") != (
        "causal4d/pokeflex-initial-state-matching"
    ):
        raise ValueError("reset-matching certificate changed")
    if initial.get("minimum_compatible_probe_count") != 3:
        raise ValueError("reset-matching support weakened")
    _require_true(initial, "source_frozen_caliper")
    _require_true(initial, "target_outcome_free")

    parity = protocol["baseline_parity"]
    for key in (
        "same_candidate_probe_set",
        "same_source_objects",
        "same_initial_state_information",
        "same_probe_costs",
        "same_probe_risk_limits",
        "one_selected_response_budget",
        "oracle_excluded_from_primary_gate",
    ):
        _require_true(parity, key)

    custody = protocol["custody"]
    for key in (
        "historical_exposure_scan_required",
        "target_selection_before_response_reveal",
        "selected_probe_response_only",
        "unselected_target_probe_responses_forbidden",
        "challenge_outcomes_forbidden_before_prediction_seal",
        "one_joint_prediction_seal_for_primary_and_replication",
        "score_primary_before_replication",
        "replication_scoring_requires_primary_pass",
        "prediction_changes_between_panels_forbidden",
        "target_retry_forbidden",
    ):
        _require_true(custody, key)

    source = protocol["source_gate"]
    if min(
        float(source["minimum_gain_vs_fixed"]),
        float(source["minimum_gain_vs_generic_information"]),
        float(source["minimum_gain_vs_dependence_destroyed"]),
    ) < 0.02:
        raise ValueError("source gain gate weakened")
    if float(source["minimum_object_win_fraction"]) < 0.60:
        raise ValueError("source object-win gate weakened")
    if float(source["minimum_query_probe_switch_fraction"]) < 0.25:
        raise ValueError("source task-specificity gate weakened")
    _require_true(source, "cost_adjusted_gain_must_be_positive")

    primary = protocol["primary_gate"]
    if primary.get("minimum_objects") != 6:
        raise ValueError("primary object panel changed")
    if primary.get("minimum_wins_out_of_six_against_main_baselines", 0) < 5:
        raise ValueError("primary object-win gate weakened")
    if primary.get("minimum_distinct_nonfallback_probes", 0) < 2:
        raise ValueError("probe-diversity gate weakened")
    if float(primary.get("minimum_query_probe_switch_fraction", 0.0)) < 1.0 / 3.0:
        raise ValueError("primary task-specificity gate weakened")
    required_primary_controls = {
        "no-probe",
        "source-fixed-safe",
        "generic-mutual-information",
        "within-object-dependence-destroyed",
        "matched-wrong-object-response",
    }
    if set(primary["positive_gain_required_against"]) != required_primary_controls:
        raise ValueError("primary comparator panel changed")
    if set(primary["positive_object_bootstrap_lower_bound_required_against"]) != {
        "source-fixed-safe",
        "within-object-dependence-destroyed",
    }:
        raise ValueError("primary inferential gate changed")
    _require_true(primary, "cost_adjusted_gain_must_be_positive")

    replication = protocol["replication_gate"]
    if replication.get("minimum_objects") != 3:
        raise ValueError("replication object panel changed")
    if replication.get("minimum_wins_out_of_three", 0) < 2:
        raise ValueError("replication win gate weakened")
    if float(replication.get("maximum_worst_object_loss_ratio", 99.0)) > 1.25:
        raise ValueError("replication harm gate weakened")

    combined = protocol["combined_gate"]
    if combined.get("minimum_wins_out_of_nine_against_fixed", 0) < 8:
        raise ValueError("combined fixed-probe win gate weakened")
    if combined.get("minimum_wins_out_of_nine_against_generic_information", 0) < 8:
        raise ValueError("combined information-probe win gate weakened")
    if float(combined.get("maximum_exact_one_sided_sign_pvalue", 1.0)) > 0.01953125:
        raise ValueError("combined sign-test gate weakened")
    _require_true(combined, "primary_gate_required")
    _require_true(combined, "replication_gate_required")

    statistics = protocol["statistics"]
    if statistics.get("primary_unit") != "physical-object":
        raise ValueError("physical object must remain the inferential unit")
    _require_true(statistics, "frame_pseudoreplication_forbidden")
    _require_true(statistics, "queries_reported_separately")

    boundary = protocol["claim_boundary"]
    _require_true(boundary, "offline_logged_active_probing_only")
    for key in (
        "identical_microscopic_reset_claim",
        "individual_counterfactual_claim",
        "online_closed_loop_claim",
        "deployment_safety_claim",
        "unique_material_identification_claim",
        "general_state_of_the_art_claim",
    ):
        _require_false(boundary, key)
