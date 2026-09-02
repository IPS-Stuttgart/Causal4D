"""Deterministic proof-carrying sensor-reveal mechanism study."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.sensor_reveal_challenge import (
    SensorRevealSubmission,
    build_sensor_reveal_plan,
    execute_sensor_reveal_plan,
    score_sensor_reveal_trace,
    seal_sensor_reveal_case,
)
from causal4d.sensor_reveal_verifier import verify_sensor_reveal_trace
from causal4d.sequential_decision_identification import (
    FiniteProbe,
    minimum_nonadaptive_probe_set,
    select_active_decision,
    select_information_probe,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _content_id(value: dict[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _routing_problem() -> tuple[
    tuple[tuple[float, float, float], ...],
    tuple[float, ...],
    tuple[FiniteProbe, ...],
]:
    losses: list[tuple[float, float, float]] = []
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
                    losses.append(
                        (0.0, 1.0, 0.6) if task == 0 else (1.0, 0.0, 0.6)
                    )
                    weights.append(1.0 / 32.0)
                    route_rows.append(
                        (1.0, 0.0) if route == 0 else (0.0, 1.0)
                    )
                    if route == 0:
                        local_zero_rows.append(
                            (1.0, 0.0, 0.0)
                            if task == 0
                            else (0.0, 1.0, 0.0)
                        )
                        local_one_rows.append((0.0, 0.0, 1.0))
                    else:
                        local_zero_rows.append((0.0, 0.0, 1.0))
                        local_one_rows.append(
                            (1.0, 0.0, 0.0)
                            if task == 0
                            else (0.0, 1.0, 0.0)
                        )
                    global_rows.append(
                        (1.0, 0.0) if task == 0 else (0.0, 1.0)
                    )
                    nuisance_rows.append(
                        tuple(
                            1.0 if outcome == nuisance else 0.0
                            for outcome in range(4)
                        )
                    )
    probes = (
        FiniteProbe(
            "route-camera",
            route_rows,
            cost=0.05,
            risk=0.01,
            outcome_names=("route-0", "route-1"),
        ),
        FiniteProbe(
            "local-tactile-0",
            local_zero_rows,
            cost=0.30,
            risk=0.01,
            outcome_names=("task-0", "task-1", "not-applicable"),
        ),
        FiniteProbe(
            "local-tactile-1",
            local_one_rows,
            cost=0.30,
            risk=0.01,
            outcome_names=("task-0", "task-1", "not-applicable"),
        ),
        FiniteProbe(
            "global-task-camera",
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


def _deterministic_outcome(probe: FiniteProbe, hypothesis: int) -> int:
    row = np.asarray(probe.likelihood[hypothesis], dtype=np.float64)
    if not np.isclose(np.max(row), 1.0) or np.count_nonzero(row) != 1:
        raise ValueError("controlled sensor is not deterministic")
    return int(np.argmax(row))


def _routing_study() -> dict[str, Any]:
    losses, weights, probes = _routing_problem()
    sensor_names = tuple(probe.name for probe in probes)
    sensor_outcomes = tuple(probe.outcome_names for probe in probes)
    sensor_costs = tuple(probe.cost for probe in probes)
    sensor_risks = tuple(probe.risk for probe in probes)
    public_context_id = _digest("sensor-reveal-controlled-public-context-v1")
    provider_id = _digest("sensor-reveal-controlled-provider-v1")
    implementation_id = _digest("sensor-reveal-controlled-implementation-v1")

    one_step = select_active_decision(
        losses,
        weights,
        probes,
        regret_tolerance=0.0,
        risk_cap=0.10,
    )
    information_index = select_information_probe(weights, probes, risk_cap=0.10)
    fixed = minimum_nonadaptive_probe_set(
        losses,
        weights,
        probes,
        regret_tolerance=0.0,
        risk_budget=0.10,
    )
    if one_step.probe_index is None or information_index is None or fixed is None:
        raise RuntimeError("controlled comparator unexpectedly failed")

    traces = []
    scores = []
    verification_ids = []
    unrequested_disclosures = 0
    for hypothesis, realized_losses in enumerate(losses):
        outcomes = tuple(
            _deterministic_outcome(probe, hypothesis) for probe in probes
        )
        payload_ids = tuple(
            _digest(f"controlled-payload/{hypothesis}/{sensor}")
            for sensor in sensor_names
        )
        adapter_ids = tuple(
            _digest(f"controlled-adapter/{sensor}") for sensor in sensor_names
        )
        manifest, truth = seal_sensor_reveal_case(
            case_id=f"controlled/hypothesis-{hypothesis:02d}",
            public_context_id=public_context_id,
            action_names=("choose-task-0", "choose-task-1", "exact-fallback"),
            fallback_action_index=2,
            sensor_names=sensor_names,
            sensor_outcome_names=sensor_outcomes,
            sensor_costs=sensor_costs,
            sensor_risks=sensor_risks,
            sensor_outcome_indices=outcomes,
            sensor_payload_sha256=payload_ids,
            sensor_adapter_ids=adapter_ids,
            realized_action_losses=realized_losses,
            truth_nonce=_digest(f"controlled-truth-nonce/{hypothesis}"),
        )
        submission = SensorRevealSubmission(
            case_id=manifest.case_id,
            manifest_id=manifest.manifest_id,
            provider_id=provider_id,
            implementation_id=implementation_id,
            hypothesis_weights=weights,
            hypothesis_losses=losses,
            probes=probes,
            regret_tolerance=0.0,
            max_probes=2,
            risk_budget=0.10,
            objective="expected_cost",
        )
        plan = build_sensor_reveal_plan(manifest, submission)
        trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
        score = score_sensor_reveal_trace(
            manifest,
            truth,
            submission,
            plan,
            trace,
        )
        verification = verify_sensor_reveal_trace(
            manifest.as_dict(), plan.as_dict(), trace.as_dict()
        )
        serialized_trace = json.dumps(
            trace.as_dict(), sort_keys=True, allow_nan=False
        )
        revealed = set(trace.revealed_sensor_indices)
        for sensor_index, payload_id in enumerate(payload_ids):
            if sensor_index not in revealed and payload_id in serialized_trace:
                unrequested_disclosures += 1
        traces.append(trace)
        scores.append(score)
        verification_ids.append(verification["verification_id"])

    return {
        "case_count": len(traces),
        "sensor_count": len(probes),
        "sequential_first_sensor": traces[0].revealed_sensor_names[0],
        "one_step_sensor": probes[one_step.probe_index].name,
        "generic_information_sensor": probes[information_index].name,
        "minimum_nonadaptive_sensor_set": list(fixed.probe_names),
        "sequential_mean_sensor_cost": float(
            np.mean([trace.total_sensor_cost for trace in traces])
        ),
        "one_step_sensor_cost": probes[one_step.probe_index].cost,
        "minimum_nonadaptive_sensor_cost": fixed.total_cost,
        "sequential_mean_reveal_count": float(
            np.mean([len(trace.events) for trace in traces])
        ),
        "sequential_mean_sensor_fraction": float(
            np.mean([len(trace.events) / len(probes) for trace in traces])
        ),
        "terminal_action_accuracy": float(
            np.mean([score.realized_regret == 0.0 for score in scores])
        ),
        "mean_realized_regret": float(
            np.mean([score.realized_regret for score in scores])
        ),
        "mean_improvement_vs_fallback": float(
            np.mean([score.improvement_vs_fallback for score in scores])
        ),
        "fallback_count": sum(trace.fallback_used for trace in traces),
        "terminal_full_state_identified_count": sum(
            trace.terminal_certificate_support_count == 1 for trace in traces
        ),
        "minimum_terminal_support_count": min(
            trace.terminal_certificate_support_count for trace in traces
        ),
        "maximum_terminal_support_count": max(
            trace.terminal_certificate_support_count for trace in traces
        ),
        "verified_trace_count": len(verification_ids),
        "unique_verification_id_count": len(set(verification_ids)),
        "unrequested_payload_digest_disclosure_count": unrequested_disclosures,
        "sample_manifest": traces[0].manifest_id,
        "sample_trace": traces[0].as_dict(),
        "sample_score": scores[0].as_dict(),
    }


def _fallback_study() -> dict[str, Any]:
    losses = ((0.0, 1.0, 0.6), (1.0, 0.0, 0.6))
    probe = FiniteProbe(
        "ambiguous-camera",
        ((0.8, 0.2), (0.2, 0.8)),
        cost=0.10,
        risk=0.01,
        outcome_names=("left-like", "right-like"),
    )
    manifest, truth = seal_sensor_reveal_case(
        case_id="controlled/noisy-full-support",
        public_context_id=_digest("controlled/noisy-context"),
        action_names=("left", "right", "exact-fallback"),
        fallback_action_index=2,
        sensor_names=(probe.name,),
        sensor_outcome_names=(probe.outcome_names,),
        sensor_costs=(probe.cost,),
        sensor_risks=(probe.risk,),
        sensor_outcome_indices=(0,),
        sensor_payload_sha256=(_digest("controlled/noisy-payload"),),
        sensor_adapter_ids=(_digest("controlled/noisy-adapter"),),
        realized_action_losses=losses[0],
        truth_nonce=_digest("controlled/noisy-nonce"),
    )
    submission = SensorRevealSubmission(
        case_id=manifest.case_id,
        manifest_id=manifest.manifest_id,
        provider_id=_digest("controlled/noisy-provider"),
        implementation_id=_digest("controlled/noisy-implementation"),
        hypothesis_weights=(0.5, 0.5),
        hypothesis_losses=losses,
        probes=(probe,),
        regret_tolerance=0.0,
        max_probes=1,
        risk_budget=0.10,
    )
    plan = build_sensor_reveal_plan(manifest, submission)
    trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
    score = score_sensor_reveal_trace(manifest, truth, submission, plan, trace)
    return {
        "policy_mode": plan.policy["mode"],
        "trace_terminal_mode": trace.terminal_mode,
        "fallback_used": trace.fallback_used,
        "revealed_sensor_count": len(trace.events),
        "action_name": trace.action_name,
        "selected_realized_loss": score.selected_realized_loss,
        "fallback_realized_loss": score.fallback_realized_loss,
        "trace_id": trace.trace_id,
    }


def run_study() -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ProofCarryingSensorRevealControlledStudy",
        "routing": _routing_study(),
        "fail_closed_control": _fallback_study(),
        "claim_boundary": (
            "Deterministic finite-interface mechanism evidence only. The study "
            "does not validate real camera/tactile adapters, learned provider "
            "competence, physical support, target transport, online execution, "
            "deployment authorization, or safety."
        ),
    }
    result["result_id"] = _content_id(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = run_study()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
