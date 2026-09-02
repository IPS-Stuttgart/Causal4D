from __future__ import annotations

import json

import pytest

from causal4d.sequential_decision_identification import (
    FiniteProbe,
    build_probe_action_quotient,
    evaluate_probe,
    finite_decision_certificate,
    minimum_nonadaptive_probe_set,
    select_active_decision,
    select_information_probe,
    solve_sequential_decision,
)


def _routing_problem() -> tuple[
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
        FiniteProbe("route", route_rows, cost=0.05, risk=0.01),
        FiniteProbe("local-r0", local_zero_rows, cost=0.30, risk=0.01),
        FiniteProbe("local-r1", local_one_rows, cost=0.30, risk=0.01),
        FiniteProbe("global-task", global_rows, cost=0.50, risk=0.01),
        FiniteProbe("nuisance-four-way", nuisance_rows, cost=0.01, risk=0.01),
    )
    return tuple(losses), tuple(weights), probes


def _policy_signature(policy: object) -> tuple:
    return (
        policy.mode,
        policy.action_index,
        policy.probe_name,
        tuple(
            (branch.outcome_index, _policy_signature(branch.policy))
            for branch in policy.outcomes
        ),
    )


def test_support_wise_certificate_does_not_use_posterior_odds() -> None:
    losses = ((0.0, 1.0), (1.0, 0.0))
    balanced = finite_decision_certificate(losses, (0.5, 0.5))
    concentrated = finite_decision_certificate(losses, (0.999, 0.001))
    assert not balanced.certified
    assert not concentrated.certified
    assert balanced.worst_case_regret == concentrated.worst_case_regret == (1.0, 1.0)


def test_one_step_and_information_policies_miss_nonmyopic_route() -> None:
    losses, weights, probes = _routing_problem()
    one_step = select_active_decision(losses, weights, probes, risk_cap=0.10)
    information = select_information_probe(weights, probes, risk_cap=0.10)
    assert one_step.mode == "probe"
    assert one_step.probe_index is not None
    assert information is not None
    assert probes[one_step.probe_index].name == "global-task"
    assert probes[information].name == "nuisance-four-way"
    route = next(
        item for item in one_step.probe_evaluations if item.probe_name == "route"
    )
    assert route.expected_regret_reduction == pytest.approx(0.0)
    assert not route.all_possible_outcomes_certified


def test_exact_two_step_policy_selects_zero_immediate_value_router() -> None:
    losses, weights, probes = _routing_problem()
    one_step = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=1,
        risk_budget=0.10,
    )
    two_step = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=2,
        risk_budget=0.10,
    )
    assert one_step.probe_name == "global-task"
    assert one_step.expected_probe_cost == pytest.approx(0.50)
    assert two_step.probe_name == "route"
    assert two_step.expected_probe_cost == pytest.approx(0.35)
    assert two_step.worst_case_probe_cost == pytest.approx(0.35)
    assert two_step.worst_case_risk == pytest.approx(0.02)
    assert tuple(branch.policy.probe_name for branch in two_step.outcomes) == (
        "local-r0",
        "local-r1",
    )
    terminal_support_sizes = [
        len(terminal.policy.certificate.support_indices)
        for branch in two_step.outcomes
        for terminal in branch.policy.outcomes
    ]
    assert terminal_support_sizes == [8, 8, 8, 8]


def test_adaptive_policy_beats_every_globally_sufficient_fixed_set() -> None:
    losses, weights, probes = _routing_problem()
    adaptive = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=2,
        risk_budget=0.10,
    )
    fixed = minimum_nonadaptive_probe_set(
        losses,
        weights,
        probes,
        risk_budget=0.10,
    )
    assert fixed is not None
    assert fixed.probe_names == ("global-task",)
    assert fixed.total_cost == pytest.approx(0.50)
    assert adaptive.expected_probe_cost < fixed.total_cost


def test_probe_action_quotient_preserves_complete_adaptive_policy() -> None:
    losses, weights, probes = _routing_problem()
    full = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=2,
        risk_budget=0.10,
    )
    quotient = build_probe_action_quotient(losses, weights, probes)
    reduced = solve_sequential_decision(
        quotient.normalized_losses,
        quotient.class_weights,
        quotient.probes,
        max_probes=2,
        risk_budget=0.10,
    )
    assert quotient.original_hypothesis_count == 32
    assert quotient.class_count == 16
    assert _policy_signature(full) == _policy_signature(reduced)
    assert reduced.expected_probe_cost == pytest.approx(full.expected_probe_cost)
    assert reduced.worst_case_probe_cost == pytest.approx(full.worst_case_probe_cost)


def test_noisy_full_support_probe_cannot_create_robust_certificate() -> None:
    losses = ((0.0, 1.0), (1.0, 0.0))
    probe = FiniteProbe(
        "noisy",
        ((0.8, 0.2), (0.2, 0.8)),
        cost=0.01,
    )
    evaluation = evaluate_probe(losses, (0.5, 0.5), probe)
    policy = solve_sequential_decision(
        losses,
        (0.5, 0.5),
        (probe,),
        max_probes=1,
    )
    assert evaluation.mutual_information > 0.0
    assert not evaluation.all_possible_outcomes_certified
    assert policy.mode == "fallback"
    assert not policy.guaranteed_certification


def test_cumulative_risk_budget_is_enforced() -> None:
    losses, weights, probes = _routing_problem()
    policy = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=2,
        risk_budget=0.015,
    )
    assert policy.probe_name == "global-task"
    assert policy.worst_case_risk == pytest.approx(0.01)


def test_fail_closed_validation_and_json_serialization() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        FiniteProbe("bad", ((0.2, 0.2),))
    with pytest.raises(ValueError, match="wrong hypothesis"):
        solve_sequential_decision(
            ((0.0, 1.0), (1.0, 0.0)),
            (0.5, 0.5),
            (FiniteProbe("bad-size", ((1.0, 0.0),)),),
        )
    losses, weights, probes = _routing_problem()
    policy = solve_sequential_decision(
        losses,
        weights,
        probes,
        max_probes=2,
        risk_budget=0.10,
    )
    encoded = json.dumps(policy.as_dict(), sort_keys=True, allow_nan=False)
    assert "minimum-cost-guaranteed-sequential-certificate" in encoded
    with pytest.raises(RuntimeError, match="maximum_nodes"):
        solve_sequential_decision(
            losses,
            weights,
            probes,
            max_probes=2,
            risk_budget=0.10,
            maximum_nodes=1,
        )
