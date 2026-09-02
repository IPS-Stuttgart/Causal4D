#!/usr/bin/env python3
"""Run the exact state-changing decision-identification mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.controlled_decision_identification import (
    FiniteControlledIntervention,
    build_controlled_decision_quotient,
    marginal_static_probe,
    minimum_nonadaptive_control_sequence,
    solve_controlled_decision,
)
from causal4d.sequential_decision_identification import (
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
    tuple[FiniteControlledIntervention, ...],
    tuple[tuple[int, int, int, int, int], ...],
]:
    states = tuple(product(range(2), range(2), range(4), range(2), range(2)))
    state_index = {state: index for index, state in enumerate(states)}
    state_count = len(states)
    losses: list[tuple[float, float]] = []
    weights: list[float] = []
    for task, _route, _nuisance, phase, _duplicate in states:
        effective_action = task ^ phase
        losses.append((0.0, 1.0) if effective_action == 0 else (1.0, 0.0))
        weights.append(1.0 / 32.0 if phase == 0 else 0.0)

    def deterministic_intervention(
        name: str,
        outcome_count: int,
        transition: Any,
        *,
        cost: float,
        risk: float,
        outcome_names: tuple[str, ...],
    ) -> FiniteControlledIntervention:
        kernel = np.zeros((state_count, state_count, outcome_count), dtype=float)
        for current_index, state in enumerate(states):
            next_state, outcome = transition(state)
            kernel[current_index, state_index[next_state], outcome] = 1.0
        return FiniteControlledIntervention(
            name,
            kernel,
            cost=cost,
            risk=risk,
            outcome_names=outcome_names,
        )

    interventions = (
        deterministic_intervention(
            "route-toggle",
            2,
            lambda state: (
                (state[0], state[1], state[2], 1 - state[3], state[4]),
                state[1],
            ),
            cost=0.05,
            risk=0.01,
            outcome_names=("route-0", "route-1"),
        ),
        deterministic_intervention(
            "local-r0",
            3,
            lambda state: (state, state[0] if state[1] == 0 else 2),
            cost=0.30,
            risk=0.01,
            outcome_names=("task-0", "task-1", "not-route-0"),
        ),
        deterministic_intervention(
            "local-r1",
            3,
            lambda state: (state, state[0] if state[1] == 1 else 2),
            cost=0.30,
            risk=0.01,
            outcome_names=("task-0", "task-1", "not-route-1"),
        ),
        deterministic_intervention(
            "global-effective",
            2,
            lambda state: (state, state[0] ^ state[3]),
            cost=0.50,
            risk=0.01,
            outcome_names=("action-0", "action-1"),
        ),
        deterministic_intervention(
            "nuisance-four-way",
            4,
            lambda state: (state, state[2]),
            cost=0.01,
            risk=0.01,
            outcome_names=("n0", "n1", "n2", "n3"),
        ),
    )
    return tuple(losses), tuple(weights), interventions, states


def _branch(policy: Any, outcome_index: int) -> Any:
    for branch in policy.outcomes:
        if branch.outcome_index == outcome_index:
            return branch.policy
    raise RuntimeError("policy has no branch for a realized positive-mass outcome")


def _deterministic_step(
    intervention: FiniteControlledIntervention,
    state_index: int,
) -> tuple[int, int]:
    kernel = np.asarray(intervention.kernel, dtype=float)
    locations = np.argwhere(kernel[state_index] > 0.0)
    if locations.shape != (1, 2):
        raise RuntimeError("controlled mechanism is not deterministic")
    next_state, outcome = (int(value) for value in locations[0])
    if kernel[state_index, next_state, outcome] != 1.0:
        raise RuntimeError("controlled mechanism is not deterministic")
    return next_state, outcome


def _execute_controlled_policy(
    policy: Any,
    state_index: int,
    interventions: tuple[FiniteControlledIntervention, ...],
) -> tuple[int, int]:
    if policy.mode == "act":
        if policy.action_index is None:
            raise RuntimeError("act node is missing an action")
        return state_index, int(policy.action_index)
    if policy.mode != "intervene" or policy.intervention_index is None:
        raise RuntimeError("controlled policy reached fallback in registered study")
    next_state, outcome = _deterministic_step(
        interventions[policy.intervention_index],
        state_index,
    )
    return _execute_controlled_policy(
        _branch(policy, outcome),
        next_state,
        interventions,
    )


def _execute_static_policy_on_controlled_system(
    policy: Any,
    state_index: int,
    interventions: tuple[FiniteControlledIntervention, ...],
) -> tuple[int, int]:
    if policy.mode == "act":
        if policy.action_index is None:
            raise RuntimeError("act node is missing an action")
        return state_index, int(policy.action_index)
    if policy.mode != "probe" or policy.probe_index is None:
        raise RuntimeError("static policy reached fallback in registered study")
    next_state, outcome = _deterministic_step(
        interventions[policy.probe_index],
        state_index,
    )
    return _execute_static_policy_on_controlled_system(
        _branch(policy, outcome),
        next_state,
        interventions,
    )


def _policy_signature(policy: Any) -> tuple[Any, ...]:
    return (
        policy.mode,
        policy.action_index,
        policy.intervention_name,
        tuple(
            (branch.outcome_index, _policy_signature(branch.policy))
            for branch in policy.outcomes
        ),
    )


def run() -> dict[str, Any]:
    losses, weights, interventions, _states = _problem()
    risk_budget = 0.10
    one_step = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=1,
        risk_budget=risk_budget,
    )
    adaptive = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=2,
        risk_budget=risk_budget,
    )
    fixed = minimum_nonadaptive_control_sequence(
        losses,
        weights,
        interventions,
        max_interventions=3,
        risk_budget=risk_budget,
    )
    if fixed is None:
        raise RuntimeError("controlled fixed-sequence comparator is not solvable")

    static_probes = tuple(marginal_static_probe(item) for item in interventions)
    static_policy = solve_sequential_decision(
        losses,
        weights,
        static_probes,
        max_probes=2,
        risk_budget=risk_budget,
    )
    information_index = select_information_probe(
        weights,
        static_probes,
        risk_cap=risk_budget,
    )
    if information_index is None:
        raise RuntimeError("generic-information comparator selected no probe")

    quotient = build_controlled_decision_quotient(
        losses,
        weights,
        interventions,
    )
    reduced = solve_controlled_decision(
        quotient.normalized_losses,
        quotient.class_weights,
        quotient.interventions,
        max_interventions=2,
        risk_budget=risk_budget,
    )

    controlled_loss = 0.0
    static_loss = 0.0
    controlled_wrong = 0
    static_wrong = 0
    supported_states = 0
    loss_matrix = np.asarray(losses, dtype=float)
    for state_index, weight in enumerate(weights):
        if weight <= 0.0:
            continue
        supported_states += 1
        controlled_state, controlled_action = _execute_controlled_policy(
            adaptive,
            state_index,
            interventions,
        )
        static_state, static_action = _execute_static_policy_on_controlled_system(
            static_policy,
            state_index,
            interventions,
        )
        controlled_value = float(loss_matrix[controlled_state, controlled_action])
        static_value = float(loss_matrix[static_state, static_action])
        controlled_loss += weight * controlled_value
        static_loss += weight * static_value
        controlled_wrong += int(controlled_value > 0.0)
        static_wrong += int(static_value > 0.0)

    terminal_support_sizes = tuple(
        len(terminal.policy.certificate.support_indices)
        for route_branch in adaptive.outcomes
        for terminal in route_branch.policy.outcomes
    )
    result: dict[str, Any] = {
        "schema": "causal4d.controlled-decision-identification-mechanism.v1",
        "physical_states": len(losses),
        "initial_supported_states": supported_states,
        "terminal_actions": 2,
        "registered_interventions": len(interventions),
        "risk_budget": risk_budget,
        "generic_information": {
            "selected_intervention": interventions[information_index].name,
        },
        "one_step": {
            "selected_intervention": one_step.intervention_name,
            "expected_cost": one_step.expected_intervention_cost,
            "worst_case_cost": one_step.worst_case_intervention_cost,
        },
        "adaptive": {
            "selected_intervention": adaptive.intervention_name,
            "branch_interventions": tuple(
                branch.policy.intervention_name for branch in adaptive.outcomes
            ),
            "expected_cost": adaptive.expected_intervention_cost,
            "worst_case_cost": adaptive.worst_case_intervention_cost,
            "worst_case_risk": adaptive.worst_case_risk,
            "terminal_supported_states": terminal_support_sizes,
            "expected_terminal_loss_on_controlled_system": controlled_loss,
            "wrong_supported_initial_states": controlled_wrong,
        },
        "minimum_nonadaptive": fixed.as_dict(),
        "static_observation_approximation": {
            "selected_probe": static_policy.probe_name,
            "expected_cost": static_policy.expected_probe_cost,
            "expected_terminal_loss_on_controlled_system": static_loss,
            "wrong_supported_initial_states": static_wrong,
        },
        "controlled_quotient": {
            "decision_classes": quotient.decision_class_count,
            "controlled_classes": quotient.class_count,
            "refinement_iterations": quotient.refinement_iterations,
            "passive_decision_quotient_sufficient": (
                quotient.passive_decision_quotient_sufficient
            ),
            "witness_count": len(quotient.witnesses),
            "policy_structure_preserved": (
                _policy_signature(adaptive) == _policy_signature(reduced)
            ),
            "expected_cost_preserved": (
                adaptive.expected_intervention_cost
                == reduced.expected_intervention_cost
            ),
            "worst_case_cost_preserved": (
                adaptive.worst_case_intervention_cost
                == reduced.worst_case_intervention_cost
            ),
        },
    }
    result["registered_checks"] = {
        "generic_information_selects_nuisance": result["generic_information"][
            "selected_intervention"
        ]
        == "nuisance-four-way",
        "one_step_requires_global_intervention": result["one_step"][
            "selected_intervention"
        ]
        == "global-effective",
        "adaptive_selects_state_changing_router": result["adaptive"][
            "selected_intervention"
        ]
        == "route-toggle",
        "adaptive_uses_branch_specific_interventions": result["adaptive"][
            "branch_interventions"
        ]
        == ("local-r0", "local-r1"),
        "adaptive_cost_strictly_lower_than_one_step": result["adaptive"][
            "expected_cost"
        ]
        < result["one_step"]["expected_cost"],
        "adaptive_cost_strictly_lower_than_nonadaptive": result["adaptive"][
            "expected_cost"
        ]
        < result["minimum_nonadaptive"]["total_cost"],
        "controlled_policy_has_zero_terminal_loss": result["adaptive"][
            "expected_terminal_loss_on_controlled_system"
        ]
        == 0.0,
        "static_approximation_is_wrong_on_every_supported_state": result[
            "static_observation_approximation"
        ]["wrong_supported_initial_states"]
        == supported_states,
        "passive_decision_quotient_is_insufficient": not result["controlled_quotient"][
            "passive_decision_quotient_sufficient"
        ],
        "controlled_quotient_strictly_refines_decision_quotient": result[
            "controlled_quotient"
        ]["controlled_classes"]
        > result["controlled_quotient"]["decision_classes"],
        "controlled_quotient_preserves_policy": all(
            result["controlled_quotient"][key]
            for key in (
                "policy_structure_preserved",
                "expected_cost_preserved",
                "worst_case_cost_preserved",
            )
        ),
        "decision_certified_without_complete_state_identification": all(
            size > 1 for size in terminal_support_sizes
        ),
    }
    if not all(result["registered_checks"].values()):
        raise RuntimeError("controlled decision-identification check failed")
    result["claim_boundary"] = (
        "Controlled finite-interface mechanism only; no real physical state "
        "support, transition kernel, sensor model, target transport, deployment, "
        "or safety claim."
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
