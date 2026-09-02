#!/usr/bin/env python3
"""Run the finite act--probe--fallback mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from causal4d.active_decision_identification import (
    FiniteProbe,
    build_probe_action_quotient,
    evaluate_probe,
    select_active_decision,
    select_information_probe,
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
    tuple[tuple[float, ...], ...],
    tuple[float, ...],
    tuple[FiniteProbe, ...],
    FiniteProbe,
]:
    losses = tuple(
        (0.0, 1.0) if hypothesis < 4 else (1.0, 0.0)
        for hypothesis in range(8)
    )
    weights = tuple(1.0 / 8.0 for _ in range(8))
    task_probe = FiniteProbe(
        name="task-sign",
        likelihood=tuple(
            (1.0, 0.0) if hypothesis < 4 else (0.0, 1.0)
            for hypothesis in range(8)
        ),
        cost=0.05,
        risk=0.01,
    )
    nuisance_probe = FiniteProbe(
        name="nuisance-four-way",
        likelihood=tuple(
            tuple(
                1.0 if outcome == hypothesis % 4 else 0.0
                for outcome in range(4)
            )
            for hypothesis in range(8)
        ),
        cost=0.05,
        risk=0.01,
    )
    unsafe_task_probe = FiniteProbe(
        name="unsafe-task-sign",
        likelihood=task_probe.likelihood,
        cost=0.0,
        risk=0.20,
    )
    dependence_destroyed = FiniteProbe(
        name="dependence-destroyed",
        likelihood=tuple(
            (1.0, 0.0) if hypothesis % 2 == 0 else (0.0, 1.0)
            for hypothesis in range(8)
        ),
        cost=0.05,
        risk=0.01,
    )
    return (
        losses,
        weights,
        (task_probe, nuisance_probe, unsafe_task_probe),
        dependence_destroyed,
    )


def run() -> dict[str, Any]:
    losses, weights, probes, dependence_destroyed = _problem()
    risk_cap = 0.05
    active = select_active_decision(
        losses,
        weights,
        probes,
        risk_cap=risk_cap,
    )
    information_index = select_information_probe(
        weights,
        probes,
        risk_cap=risk_cap,
    )
    if information_index is None:
        raise RuntimeError("generic information policy found no safe probe")
    information_evaluation = evaluate_probe(
        losses,
        weights,
        probes[information_index],
        probe_index=information_index,
        risk_cap=risk_cap,
    )
    destroyed = select_active_decision(
        losses,
        weights,
        (dependence_destroyed,),
        risk_cap=risk_cap,
    )

    duplicated_losses = tuple(row for row in losses for _ in range(2))
    duplicated_weights = tuple(weight / 2.0 for weight in weights for _ in range(2))
    duplicated_probes = tuple(
        FiniteProbe(
            name=probe.name,
            likelihood=tuple(row for row in probe.likelihood for _ in range(2)),
            cost=probe.cost,
            risk=probe.risk,
        )
        for probe in probes
    )
    quotient = build_probe_action_quotient(
        duplicated_losses,
        duplicated_weights,
        duplicated_probes,
    )
    quotient_decision = select_active_decision(
        quotient.normalized_losses,
        quotient.class_weights,
        quotient.probes,
        risk_cap=risk_cap,
    )

    active_evaluation = next(
        item
        for item in active.probe_evaluations
        if item.probe_index == active.probe_index
    )
    unsafe_evaluation = next(
        item
        for item in active.probe_evaluations
        if item.probe_name == "unsafe-task-sign"
    )
    result: dict[str, Any] = {
        "schema": "causal4d.active-decision-identification-mechanism.v1",
        "hypotheses": 8,
        "terminal_actions": 2,
        "risk_cap": risk_cap,
        "active_policy": {
            "mode": active.mode,
            "selected_probe": probes[active.probe_index].name
            if active.probe_index is not None
            else None,
            "current_minimum_worst_case_regret": (
                active.current_certificate.minimum_worst_case_regret
            ),
            "expected_post_probe_regret": (
                active_evaluation.expected_post_probe_regret
            ),
            "worst_post_probe_regret": active_evaluation.worst_post_probe_regret,
            "all_outcomes_certified": (
                active_evaluation.all_possible_outcomes_certified
            ),
            "mutual_information_nats": active_evaluation.mutual_information,
        },
        "generic_information_policy": {
            "selected_probe": probes[information_index].name,
            "mutual_information_nats": information_evaluation.mutual_information,
            "expected_post_probe_regret": (
                information_evaluation.expected_post_probe_regret
            ),
            "all_outcomes_certified": (
                information_evaluation.all_possible_outcomes_certified
            ),
        },
        "unsafe_probe": {
            "name": unsafe_evaluation.probe_name,
            "risk": probes[unsafe_evaluation.probe_index].risk,
            "safe": unsafe_evaluation.safe,
            "all_outcomes_certified": (
                unsafe_evaluation.all_possible_outcomes_certified
            ),
        },
        "dependence_destroyed_control": {
            "mode": destroyed.mode,
            "selected_probe": destroyed.probe_index,
            "current_minimum_worst_case_regret": (
                destroyed.current_certificate.minimum_worst_case_regret
            ),
            "expected_post_probe_regret": (
                destroyed.probe_evaluations[0].expected_post_probe_regret
            ),
        },
        "registered_quotient": {
            "complete_hypotheses": len(duplicated_losses),
            "quotient_classes": len(quotient.class_members),
            "decision_preserved": (
                active.mode,
                active.action_index,
                active.probe_index,
            )
            == (
                quotient_decision.mode,
                quotient_decision.action_index,
                quotient_decision.probe_index,
            ),
        },
    }
    result["registered_checks"] = {
        "task_probe_selected": result["active_policy"]["selected_probe"]
        == "task-sign",
        "task_probe_identifies_every_outcome": result["active_policy"][
            "all_outcomes_certified"
        ],
        "task_probe_removes_robust_regret": result["active_policy"][
            "worst_post_probe_regret"
        ]
        == 0.0,
        "generic_information_selects_nuisance": result[
            "generic_information_policy"
        ]["selected_probe"]
        == "nuisance-four-way",
        "generic_information_does_not_identify": not result[
            "generic_information_policy"
        ]["all_outcomes_certified"],
        "unsafe_probe_rejected": not result["unsafe_probe"]["safe"],
        "dependence_destroyed_returns_fallback": result[
            "dependence_destroyed_control"
        ]["mode"]
        == "fallback",
        "quotient_preserves_decision": result["registered_quotient"][
            "decision_preserved"
        ],
    }
    if not all(result["registered_checks"].values()):
        raise RuntimeError("active decision-identification mechanism check failed")
    result["claim_boundary"] = (
        "Exact only for the registered finite support, losses, probe channels, "
        "costs, and risk scores; not real-robot, population-calibration, or "
        "deployment-safety evidence."
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
