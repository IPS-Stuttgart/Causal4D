#!/usr/bin/env python3
"""Run the finite certificate-level act--probe--fallback mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.active_decision_identification import (
    CertificateOutcomeV1,
    CertificateProbeV1,
    plan_active_decision,
)
from causal4d.decision_identifiable_intervention import (
    QUERY_DECISION_CERTIFICATE_SEMANTICS,
)
from causal4d.task_conditioned_design import mutual_information_nats


def _content_id(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _certificate(
    pairwise_worst_case_loss_gap: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = 0.0,
) -> dict[str, object]:
    pairwise = np.asarray(pairwise_worst_case_loss_gap, dtype=np.float64)
    if pairwise.ndim != 2 or pairwise.shape[0] != pairwise.shape[1]:
        raise ValueError("pairwise loss gaps must be square")
    regret = np.maximum(np.max(pairwise, axis=1), 0.0)
    robust = np.all(pairwise <= 1e-12, axis=1)
    admissible = regret <= tolerance + 1e-12
    minimum = float(np.min(regret))
    minimax = int(np.flatnonzero(np.isclose(regret, minimum, atol=1e-12))[0])
    return {
        "summary": {
            "version": 1,
            "semantics": QUERY_DECISION_CERTIFICATE_SEMANTICS,
            "action_count": int(pairwise.shape[0]),
            "minimax_action_index": minimax,
            "minimax_worst_case_regret": minimum,
            "regret_tolerance": tolerance,
            "has_tolerance_admissible_action": bool(np.any(admissible)),
            "uniquely_tolerance_identified": bool(np.count_nonzero(admissible) == 1),
            "has_robustly_optimal_action": bool(np.any(robust)),
            "uniquely_robustly_optimal": bool(np.count_nonzero(robust) == 1),
        },
        "pairwise_worst_case_loss_gap": pairwise.tolist(),
        "worst_case_regret": regret.tolist(),
        "minimax_action_index": minimax,
        "minimax_worst_case_regret": minimum,
        "regret_tolerance": tolerance,
        "tolerance_admissible_action_mask": admissible.tolist(),
        "robustly_optimal_action_mask": robust.tolist(),
    }


def _information_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prior = np.full(8, 1.0 / 8.0, dtype=np.float64)
    task_likelihood = np.asarray(
        [
            (1.0, 0.0) if hypothesis < 4 else (0.0, 1.0)
            for hypothesis in range(8)
        ],
        dtype=np.float64,
    )
    nuisance_likelihood = np.asarray(
        [
            tuple(
                1.0 if outcome == hypothesis % 4 else 0.0
                for outcome in range(4)
            )
            for hypothesis in range(8)
        ],
        dtype=np.float64,
    )
    return prior, task_likelihood, nuisance_likelihood


def run() -> dict[str, Any]:
    action_names = ("pull-left", "pull-right")
    fallback = "hold"
    current = _certificate(((0.0, 1.0), (1.0, 0.0)))
    left = _certificate(((0.0, -1.0), (1.0, 0.0)))
    right = _certificate(((0.0, 1.0), (-1.0, 0.0)))

    task_probe = CertificateProbeV1(
        name="task-sign",
        outcomes=(
            CertificateOutcomeV1(0.5, left),
            CertificateOutcomeV1(0.5, right),
        ),
        physical_risk=0.01,
        cost=0.05,
    )
    nuisance_probe = CertificateProbeV1(
        name="nuisance-four-way",
        outcomes=tuple(
            CertificateOutcomeV1(0.25, current) for _ in range(4)
        ),
        physical_risk=0.01,
        cost=0.05,
    )
    unsafe_task_probe = CertificateProbeV1(
        name="unsafe-task-sign",
        outcomes=task_probe.outcomes,
        physical_risk=0.20,
        cost=0.0,
    )
    destroyed_probe = CertificateProbeV1(
        name="dependence-destroyed",
        outcomes=(
            CertificateOutcomeV1(0.5, current),
            CertificateOutcomeV1(0.5, current),
        ),
        physical_risk=0.01,
        cost=0.05,
    )

    active = plan_active_decision(
        current,
        action_names,
        fallback_action_name=fallback,
        probes=(task_probe, nuisance_probe, unsafe_task_probe),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_certification_probability=1.0,
    )
    destroyed = plan_active_decision(
        current,
        action_names,
        fallback_action_name=fallback,
        probes=(destroyed_probe,),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_certification_probability=1.0,
    )

    prior, task_likelihood, nuisance_likelihood = _information_problem()
    task_information = mutual_information_nats(prior, task_likelihood)
    nuisance_information = mutual_information_nats(prior, nuisance_likelihood)
    information_selected = (
        "nuisance-four-way"
        if nuisance_information > task_information
        else "task-sign"
    )

    reports = {report.name: report for report in active.probe_reports}
    result: dict[str, Any] = {
        "schema": "causal4d.active-decision-identification-mechanism.v1",
        "hypotheses_in_information_control": 8,
        "terminal_actions": len(action_names),
        "risk_cap": 0.05,
        "active_policy": active.as_dict(),
        "generic_information_policy": {
            "selected_probe": information_selected,
            "task_probe_mutual_information_nats": task_information,
            "nuisance_probe_mutual_information_nats": nuisance_information,
        },
        "dependence_destroyed_control": destroyed.as_dict(),
        "registered_checks": {
            "task_probe_selected": active.selected_probe_name == "task-sign",
            "task_probe_identifies_every_outcome": reports[
                "task-sign"
            ].certification_probability
            == 1.0,
            "task_probe_removes_expected_minimax_regret": reports[
                "task-sign"
            ].expected_posterior_minimax_worst_case_regret
            == 0.0,
            "generic_information_selects_nuisance": information_selected
            == "nuisance-four-way",
            "generic_information_probe_has_no_decision_value": reports[
                "nuisance-four-way"
            ].expected_regret_reduction
            == 0.0,
            "unsafe_probe_rejected": not reports["unsafe-task-sign"].safe,
            "dependence_destroyed_returns_fallback": destroyed.mode == "fallback",
        },
        "claim_boundary": (
            "Exact only for the supplied finite certificates, outcome masses, "
            "costs, and risk scores. This controlled mechanism does not validate "
            "a physical support, probe likelihood model, real robot benefit, "
            "population calibration, or deployment safety."
        ),
    }
    if not all(result["registered_checks"].values()):
        raise RuntimeError("active decision-identification mechanism check failed")
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
