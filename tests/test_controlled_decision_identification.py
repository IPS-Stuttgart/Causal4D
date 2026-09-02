from __future__ import annotations

import json
from itertools import product

import numpy as np
import pytest

from causal4d.controlled_decision_identification import (
    FiniteControlledIntervention,
    build_controlled_decision_quotient,
    controlled_intervention_from_probe,
    factorized_controlled_intervention,
    controlled_outcome_probabilities,
    controlled_posterior_weights,
    marginal_static_probe,
    minimum_nonadaptive_control_sequence,
    solve_controlled_decision,
)
from causal4d.dynamic_contact import contact_transition_matrix
from causal4d.sequential_decision_identification import (
    FiniteProbe,
    solve_sequential_decision,
)


def _dual_control_problem() -> tuple[
    tuple[tuple[float, float], ...],
    tuple[float, ...],
    tuple[FiniteControlledIntervention, ...],
]:
    states = list(product(range(2), range(2), range(4), range(2), range(2)))
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
        transition: object,
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

    route_toggle = deterministic_intervention(
        "route-toggle",
        2,
        lambda state: (
            (state[0], state[1], state[2], 1 - state[3], state[4]),
            state[1],
        ),
        cost=0.05,
        risk=0.01,
        outcome_names=("route-0", "route-1"),
    )
    local_zero = deterministic_intervention(
        "local-r0",
        3,
        lambda state: (state, state[0] if state[1] == 0 else 2),
        cost=0.30,
        risk=0.01,
        outcome_names=("task-0", "task-1", "not-route-0"),
    )
    local_one = deterministic_intervention(
        "local-r1",
        3,
        lambda state: (state, state[0] if state[1] == 1 else 2),
        cost=0.30,
        risk=0.01,
        outcome_names=("task-0", "task-1", "not-route-1"),
    )
    global_effective = deterministic_intervention(
        "global-effective",
        2,
        lambda state: (state, state[0] ^ state[3]),
        cost=0.50,
        risk=0.01,
        outcome_names=("action-0", "action-1"),
    )
    nuisance = deterministic_intervention(
        "nuisance-four-way",
        4,
        lambda state: (state, state[2]),
        cost=0.01,
        risk=0.01,
        outcome_names=("n0", "n1", "n2", "n3"),
    )
    return (
        tuple(losses),
        tuple(weights),
        (route_toggle, local_zero, local_one, global_effective, nuisance),
    )


def _policy_signature(policy: object) -> tuple[object, ...]:
    return (
        policy.mode,
        policy.action_index,
        policy.intervention_name,
        tuple(
            (branch.outcome_index, _policy_signature(branch.policy))
            for branch in policy.outcomes
        ),
    )


def _static_policy_terminal_actions(policy: object) -> tuple[int, ...]:
    if policy.mode == "act":
        assert policy.action_index is not None
        return (policy.action_index,)
    return tuple(
        action
        for branch in policy.outcomes
        for action in _static_policy_terminal_actions(branch.policy)
    )


def test_passive_probe_embedding_matches_static_solver() -> None:
    losses = ((0.0, 1.0), (1.0, 0.0))
    weights = (0.5, 0.5)
    probe = FiniteProbe("perfect", ((1.0, 0.0), (0.0, 1.0)), cost=0.2)
    intervention = controlled_intervention_from_probe(probe)
    static = solve_sequential_decision(
        losses,
        weights,
        (probe,),
        max_probes=1,
    )
    controlled = solve_controlled_decision(
        losses,
        weights,
        (intervention,),
        max_interventions=1,
    )
    assert static.probe_name == controlled.intervention_name == "perfect"
    assert static.expected_probe_cost == controlled.expected_intervention_cost
    assert tuple(branch.policy.action_index for branch in static.outcomes) == tuple(
        branch.policy.action_index for branch in controlled.outcomes
    )
    assert marginal_static_probe(intervention) == probe


def test_factorized_intervention_bridges_dynamic_contact_model() -> None:
    transition = contact_transition_matrix(0.9, 0.1)
    observation = np.eye(transition.shape[0], dtype=float)
    intervention = factorized_controlled_intervention(
        "contact-regime-readout",
        transition,
        observation,
        outcome_names=("inactive", "sticking", "slipping", "detached"),
    )
    kernel = np.asarray(intervention.kernel, dtype=float)
    assert np.allclose(np.sum(kernel, axis=2), transition)
    assert np.allclose(np.sum(kernel, axis=(1, 2)), 1.0)


def test_controlled_update_propagates_state_before_conditioning() -> None:
    kernel = np.zeros((2, 2, 2), dtype=float)
    kernel[0, 1, 0] = 1.0
    kernel[1, 0, 1] = 1.0
    intervention = FiniteControlledIntervention("swap", kernel)
    assert controlled_outcome_probabilities((0.75, 0.25), intervention) == (
        0.75,
        0.25,
    )
    assert controlled_posterior_weights((0.75, 0.25), intervention, 0) == (
        0.0,
        1.0,
    )
    assert controlled_posterior_weights((0.75, 0.25), intervention, 1) == (
        1.0,
        0.0,
    )


def test_nonmyopic_state_changing_policy_beats_one_step_and_fixed_sequence() -> None:
    losses, weights, interventions = _dual_control_problem()
    one_step = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=1,
        risk_budget=0.10,
    )
    adaptive = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=2,
        risk_budget=0.10,
    )
    fixed = minimum_nonadaptive_control_sequence(
        losses,
        weights,
        interventions,
        max_interventions=3,
        risk_budget=0.10,
    )
    assert one_step.intervention_name == "global-effective"
    assert one_step.expected_intervention_cost == pytest.approx(0.50)
    assert adaptive.intervention_name == "route-toggle"
    assert adaptive.expected_intervention_cost == pytest.approx(0.35)
    assert adaptive.worst_case_intervention_cost == pytest.approx(0.35)
    assert adaptive.worst_case_risk == pytest.approx(0.02)
    assert tuple(branch.policy.intervention_name for branch in adaptive.outcomes) == (
        "local-r0",
        "local-r1",
    )
    terminal_support_sizes = [
        len(terminal.policy.certificate.support_indices)
        for route_branch in adaptive.outcomes
        for terminal in route_branch.policy.outcomes
    ]
    assert terminal_support_sizes == [8, 8, 8, 8]
    assert fixed is not None
    assert fixed.intervention_names == ("global-effective",)
    assert fixed.total_cost == pytest.approx(0.50)
    assert adaptive.expected_intervention_cost < fixed.total_cost


def test_static_observation_approximation_selects_wrong_terminal_actions() -> None:
    losses, weights, interventions = _dual_control_problem()
    controlled = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=2,
        risk_budget=0.10,
    )
    static = solve_sequential_decision(
        losses,
        weights,
        tuple(marginal_static_probe(item) for item in interventions),
        max_probes=2,
        risk_budget=0.10,
    )
    assert controlled.intervention_name == static.probe_name == "route-toggle"
    controlled_actions = tuple(
        terminal.policy.action_index
        for route_branch in controlled.outcomes
        for terminal in route_branch.policy.outcomes
    )
    static_actions = _static_policy_terminal_actions(static)
    assert controlled_actions == (1, 0, 1, 0)
    assert static_actions == (0, 1, 0, 1)
    assert all(
        controlled_action != static_action
        for controlled_action, static_action in zip(
            controlled_actions,
            static_actions,
            strict=True,
        )
    )


def test_controlled_quotient_is_coarsest_policy_preserving_refinement() -> None:
    losses, weights, interventions = _dual_control_problem()
    full = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=2,
        risk_budget=0.10,
    )
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
        risk_budget=0.10,
    )
    assert quotient.original_state_count == 64
    assert quotient.decision_class_count == 2
    assert quotient.class_count == 32
    assert not quotient.passive_decision_quotient_sufficient
    assert quotient.witnesses
    assert _policy_signature(full) == _policy_signature(reduced)
    assert reduced.expected_intervention_cost == pytest.approx(
        full.expected_intervention_cost
    )
    assert reduced.worst_case_intervention_cost == pytest.approx(
        full.worst_case_intervention_cost
    )


def test_partition_refinement_detects_multi_step_controlled_distinctions() -> None:
    losses = (
        (0.0, 1.0),
        (0.0, 1.0),
        (0.0, 1.0),
        (1.0, 0.0),
    )
    kernel = np.zeros((4, 4, 1), dtype=float)
    kernel[0, 1, 0] = 1.0
    kernel[1, 2, 0] = 1.0
    kernel[2, 3, 0] = 1.0
    kernel[3, 3, 0] = 1.0
    intervention = FiniteControlledIntervention("advance", kernel)
    quotient = build_controlled_decision_quotient(
        losses,
        (0.25, 0.25, 0.25, 0.25),
        (intervention,),
    )
    assert quotient.decision_class_count == 2
    assert quotient.class_count == 4
    assert quotient.refinement_iterations >= 2


def test_controlled_risk_and_fail_closed_limits() -> None:
    losses, weights, interventions = _dual_control_problem()
    policy = solve_controlled_decision(
        losses,
        weights,
        interventions,
        max_interventions=2,
        risk_budget=0.015,
    )
    assert policy.intervention_name == "global-effective"
    assert policy.worst_case_risk == pytest.approx(0.01)
    with pytest.raises(RuntimeError, match="maximum_nodes"):
        solve_controlled_decision(
            losses,
            weights,
            interventions,
            max_interventions=2,
            risk_budget=0.10,
            maximum_nodes=1,
        )
    with pytest.raises(ValueError, match="sum to one"):
        FiniteControlledIntervention("bad", np.zeros((2, 2, 1)))
    encoded = json.dumps(policy.as_dict(), sort_keys=True, allow_nan=False)
    assert "controlled-decision-certificate" in encoded


def test_controlled_mechanism_study_passes_registered_checks(tmp_path) -> None:
    import importlib.util

    script = (
        __import__("pathlib").Path(__file__).parents[1]
        / "scripts"
        / "experiments"
        / "controlled_decision_identification.py"
    )
    spec = importlib.util.spec_from_file_location("controlled_study", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    result = module.run()
    assert all(result["registered_checks"].values())
    assert result["adaptive"]["expected_cost"] == pytest.approx(0.35)
    assert result["one_step"]["expected_cost"] == pytest.approx(0.50)
    assert result["static_observation_approximation"][
        "expected_terminal_loss_on_controlled_system"
    ] == pytest.approx(1.0)
    output = tmp_path / "result.json"
    output.write_text(json.dumps(result, sort_keys=True), encoding="utf-8")
    assert output.stat().st_size > 0
