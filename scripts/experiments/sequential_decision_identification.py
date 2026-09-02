#!/usr/bin/env python3
"""Run the exact sequential decision-identification mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causal4d.probe_quotient_audit import audit_decision_quotient_for_probes
from causal4d.sequential_decision_identification import (
    FiniteProbe,
    build_probe_action_quotient,
    minimum_nonadaptive_probe_set,
    select_active_decision,
    select_information_probe,
    solve_sequential_decision,
)


def _content_id(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _problem() -> tuple[
    tuple[tuple[float, float], ...],
    tuple[float, ...],
    tuple[FiniteProbe, ...],
]:
    losses: list[tuple[float, float]] = []
    weights: list[float] = []
    route_rows: list[tuple[float, float]] = []
    local_zero_rows: list[tuple[float, float, float]] = []
    local_one_rows: list[tuple[float, float, float]] = []
    global_rows: list[tuple[float, float]] = []
    nuisance_rows: list[tuple[float, float, float, float]] = []
    for task in range(2):
        for route in range(2):
            for nuisance in range(4):
                for _duplicate in range(2):
                    losses.append((0.0, 1.0) if task == 0 else (1.0, 0.0))
                    weights.append(1.0 / 32.0)
                    route_rows.append((1.0, 0.0) if route == 0 else (0.0, 1.0))
                    if route == 0:
                        local_zero_rows.append(
                            (1.0, 0.0, 0.0) if task == 0 else (0.0, 1.0, 0.0)
                        )
                        local_one_rows.append((0.0, 0.0, 1.0))
                    else:
                        local_zero_rows.append((0.0, 0.0, 1.0))
                        local_one_rows.append(
                            (1.0, 0.0, 0.0) if task == 0 else (0.0, 1.0, 0.0)
                        )
                    global_rows.append((1.0, 0.0) if task == 0 else (0.0, 1.0))
                    nuisance_rows.append(
                        tuple(
                            1.0 if outcome == nuisance else 0.0 for outcome in range(4)
                        )
                    )
    probes = (
        FiniteProbe(
            "route",
            route_rows,
            cost=0.05,
            risk=0.01,
            outcome_names=("route-0", "route-1"),
        ),
        FiniteProbe(
            "local-r0",
            local_zero_rows,
            cost=0.30,
            risk=0.01,
            outcome_names=("task-0", "task-1", "not-route-0"),
        ),
        FiniteProbe(
            "local-r1",
            local_one_rows,
            cost=0.30,
            risk=0.01,
            outcome_names=("task-0", "task-1", "not-route-1"),
        ),
        FiniteProbe(
            "global-task",
            global_rows,
            cost=0.50,
            risk=0.01,
            outcome_names=("task-0", "task-1"),
        ),
        FiniteProbe(
            "nuisance-four-way",
            nuisance_rows,
            cost=0.01,
            risk=0.01,
            outcome_names=("n0", "n1", "n2", "n3"),
        ),
    )
    return tuple(losses), tuple(weights), probes


def _policy_signature(policy: object) -> tuple[object, ...]:
    return (
        policy.mode,
        policy.action_index,
        policy.probe_name,
        tuple(
            (branch.outcome_index, _policy_signature(branch.policy))
            for branch in policy.outcomes
        ),
    )


def _compact_policy(policy: object) -> dict[str, object]:
    return {
        "mode": policy.mode,
        "action_index": policy.action_index,
        "probe_name": policy.probe_name,
        "support_size": len(policy.certificate.support_indices),
        "expected_probe_cost": policy.expected_probe_cost,
        "worst_case_probe_cost": policy.worst_case_probe_cost,
        "worst_case_risk": policy.worst_case_risk,
        "outcomes": [
            {
                "name": branch.outcome_name,
                "probability": branch.probability,
                "policy": _compact_policy(branch.policy),
            }
            for branch in policy.outcomes
        ],
    }


def run() -> dict[str, Any]:
    losses, weights, probes = _problem()
    risk_budget = 0.10
    one_step = select_active_decision(
        losses,
        weights,
        probes,
        risk_cap=risk_budget,
    )
    information_index = select_information_probe(
        weights,
        probes,
        risk_cap=risk_budget,
    )
    if one_step.probe_index is None or information_index is None:
        raise RuntimeError("controlled one-step selectors failed")
    horizon_one = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=1,
        risk_budget=risk_budget,
    )
    horizon_two = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=2,
        risk_budget=risk_budget,
    )
    fixed = minimum_nonadaptive_probe_set(
        losses,
        weights,
        probes,
        risk_budget=risk_budget,
    )
    if fixed is None:
        raise RuntimeError("controlled fixed probe problem is not solvable")

    decision_audit = audit_decision_quotient_for_probes(losses, weights, probes)
    quotient = build_probe_action_quotient(losses, weights, probes)
    quotient_policy = solve_sequential_decision(
        quotient.normalized_losses,
        quotient.class_weights,
        quotient.probes,
        max_probes=2,
        risk_budget=risk_budget,
    )
    route_evaluation = next(
        item for item in one_step.probe_evaluations if item.probe_name == "route"
    )
    terminal_support_sizes = tuple(
        len(terminal.policy.certificate.support_indices)
        for route_branch in horizon_two.outcomes
        for terminal in route_branch.policy.outcomes
    )
    branch_probes = tuple(branch.policy.probe_name for branch in horizon_two.outcomes)
    result: dict[str, Any] = {
        "schema": "causal4d.sequential-decision-identification-mechanism.v1",
        "complete_hypotheses": len(losses),
        "terminal_actions": 2,
        "registered_probes": len(probes),
        "risk_budget": risk_budget,
        "one_step_decision_value": {
            "selected_probe": probes[one_step.probe_index].name,
            "route_expected_regret_reduction": (
                route_evaluation.expected_regret_reduction
            ),
            "route_all_outcomes_certified": (
                route_evaluation.all_possible_outcomes_certified
            ),
        },
        "generic_information": {
            "selected_probe": probes[information_index].name,
            "mutual_information_nats": next(
                item.mutual_information
                for item in one_step.probe_evaluations
                if item.probe_index == information_index
            ),
        },
        "horizon_one": {
            "selected_probe": horizon_one.probe_name,
            "expected_cost": horizon_one.expected_probe_cost,
            "worst_case_cost": horizon_one.worst_case_probe_cost,
        },
        "horizon_two": {
            "selected_probe": horizon_two.probe_name,
            "branch_probes": branch_probes,
            "expected_cost": horizon_two.expected_probe_cost,
            "worst_case_cost": horizon_two.worst_case_probe_cost,
            "worst_case_risk": horizon_two.worst_case_risk,
            "terminal_supported_hypotheses": terminal_support_sizes,
            "complete_state_identified": all(
                size == 1 for size in terminal_support_sizes
            ),
            "policy": _compact_policy(horizon_two),
        },
        "minimum_nonadaptive": fixed.as_dict(),
        "decision_quotient_probe_audit": decision_audit.as_dict(),
        "probe_action_quotient": {
            "complete_hypotheses": quotient.original_hypothesis_count,
            "quotient_classes": quotient.class_count,
            "policy_structure_preserved": (
                _policy_signature(horizon_two) == _policy_signature(quotient_policy)
            ),
            "expected_cost_preserved": (
                horizon_two.expected_probe_cost == quotient_policy.expected_probe_cost
            ),
            "worst_case_cost_preserved": (
                horizon_two.worst_case_probe_cost
                == quotient_policy.worst_case_probe_cost
            ),
            "reduced_policy": _compact_policy(quotient_policy),
        },
    }
    result["registered_checks"] = {
        "generic_information_selects_nuisance": result["generic_information"][
            "selected_probe"
        ]
        == "nuisance-four-way",
        "one_step_selects_direct_global_probe": result["one_step_decision_value"][
            "selected_probe"
        ]
        == "global-task",
        "routing_probe_has_zero_immediate_value": result["one_step_decision_value"][
            "route_expected_regret_reduction"
        ]
        == 0.0,
        "horizon_one_requires_direct_probe": result["horizon_one"]["selected_probe"]
        == "global-task",
        "horizon_two_selects_router": result["horizon_two"]["selected_probe"]
        == "route",
        "router_enables_branch_specific_probes": result["horizon_two"]["branch_probes"]
        == ("local-r0", "local-r1"),
        "adaptive_cost_strictly_lower_than_one_step": result["horizon_two"][
            "expected_cost"
        ]
        < result["horizon_one"]["expected_cost"],
        "adaptive_cost_strictly_lower_than_fixed_set": result["horizon_two"][
            "expected_cost"
        ]
        < result["minimum_nonadaptive"]["total_cost"],
        "decision_certified_without_state_identification": not result["horizon_two"][
            "complete_state_identified"
        ],
        "decision_quotient_is_not_probe_lumpable": not result[
            "decision_quotient_probe_audit"
        ]["sequentially_sufficient"],
        "decision_quotient_has_two_classes": result[
            "decision_quotient_probe_audit"
        ]["decision_class_count"]
        == 2,
        "probe_action_quotient_strictly_refines_decision_quotient": result[
            "probe_action_quotient"
        ]["quotient_classes"]
        > result["decision_quotient_probe_audit"]["decision_class_count"],
        "probe_action_quotient_is_strict": result["probe_action_quotient"][
            "quotient_classes"
        ]
        < result["probe_action_quotient"]["complete_hypotheses"],
        "quotient_preserves_policy": all(
            result["probe_action_quotient"][key]
            for key in (
                "policy_structure_preserved",
                "expected_cost_preserved",
                "worst_case_cost_preserved",
            )
        ),
    }
    if not all(result["registered_checks"].values()):
        raise RuntimeError("sequential decision-identification check failed")
    result["claim_boundary"] = (
        "Controlled finite-interface mechanism only; no real sensor, physical "
        "probe, target transport, calibration, deployment, or safety claim."
    )
    result["result_id"] = _content_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
