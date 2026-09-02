from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from causal4d.sensor_reveal_challenge import (
    SensorRevealSubmission,
    SensorRevealTruth,
    build_sensor_reveal_plan,
    execute_sensor_reveal_plan,
    score_sensor_reveal_trace,
    seal_sensor_reveal_case,
)
from causal4d.sensor_reveal_verifier import verify_sensor_reveal_trace
from causal4d.sequential_decision_identification import FiniteProbe


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _reseal(record: dict[str, object], id_key: str) -> None:
    payload = {key: value for key, value in record.items() if key != id_key}
    record[id_key] = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _simple_case() -> tuple[object, object, SensorRevealSubmission, object]:
    probes = (
        FiniteProbe(
            "task-camera",
            ((1.0, 0.0), (0.0, 1.0)),
            cost=0.25,
            risk=0.01,
            outcome_names=("task-0", "task-1"),
        ),
        FiniteProbe(
            "nuisance-camera",
            ((1.0, 0.0), (1.0, 0.0)),
            cost=0.05,
            risk=0.01,
            outcome_names=("nuisance-0", "nuisance-1"),
        ),
    )
    manifest, truth = seal_sensor_reveal_case(
        case_id="test/case-0",
        public_context_id=_digest("public-context"),
        action_names=("action-0", "action-1", "exact-fallback"),
        fallback_action_index=2,
        sensor_names=tuple(probe.name for probe in probes),
        sensor_outcome_names=tuple(probe.outcome_names for probe in probes),
        sensor_costs=tuple(probe.cost for probe in probes),
        sensor_risks=tuple(probe.risk for probe in probes),
        sensor_outcome_indices=(0, 0),
        sensor_payload_sha256=(
            _digest("requested-payload"),
            _digest("unrequested-payload"),
        ),
        sensor_adapter_ids=(
            _digest("task-adapter"),
            _digest("nuisance-adapter"),
        ),
        realized_action_losses=(0.0, 1.0, 0.6),
        truth_nonce=_digest("truth-nonce"),
    )
    submission = SensorRevealSubmission(
        case_id=manifest.case_id,
        manifest_id=manifest.manifest_id,
        provider_id=_digest("provider"),
        implementation_id=_digest("implementation"),
        hypothesis_weights=(0.5, 0.5),
        hypothesis_losses=((0.0, 1.0, 0.6), (1.0, 0.0, 0.6)),
        probes=probes,
        regret_tolerance=0.0,
        max_probes=1,
        risk_budget=0.10,
    )
    plan = build_sensor_reveal_plan(manifest, submission)
    return manifest, truth, submission, plan


def test_requested_sensor_only_trace_is_sealed_before_scoring() -> None:
    manifest, truth, submission, plan = _simple_case()
    trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
    assert trace.revealed_sensor_names == ("task-camera",)
    assert trace.action_name == "action-0"
    assert trace.terminal_mode == "act"
    encoded = json.dumps(trace.as_dict(), sort_keys=True, allow_nan=False)
    assert _digest("requested-payload") in encoded
    assert _digest("unrequested-payload") not in encoded
    for forbidden in (
        "sensor_outcome_indices",
        "realized_action_losses",
        "truth_nonce",
        "selected_realized_loss",
    ):
        assert forbidden not in encoded
    verification = verify_sensor_reveal_trace(
        manifest.as_dict(), plan.as_dict(), trace.as_dict()
    )
    assert verification["status"] == "verified-trace"
    score = score_sensor_reveal_trace(
        manifest,
        truth,
        submission,
        plan,
        trace,
    )
    assert score.trace_id == trace.trace_id
    assert score.realized_regret == pytest.approx(0.0)
    assert score.improvement_vs_fallback == pytest.approx(0.6)


def test_independent_verifier_rejects_unrequested_sensor_injection() -> None:
    manifest, truth, submission, plan = _simple_case()
    trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
    tampered = trace.as_dict()
    event = tampered["events"][0]
    event["sensor_index"] = 1
    event["sensor_name"] = "nuisance-camera"
    event["payload_sha256"] = _digest("unrequested-payload")
    _reseal(tampered, "trace_id")
    with pytest.raises(ValueError, match="not requested"):
        verify_sensor_reveal_trace(manifest.as_dict(), plan.as_dict(), tampered)


def test_independent_verifier_recomputes_policy_cost_accounting() -> None:
    manifest, truth, submission, plan = _simple_case()
    trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
    tampered = plan.as_dict()
    policy = tampered["policy"]
    assert isinstance(policy, dict)
    policy["expected_probe_cost"] = float(policy["expected_probe_cost"]) - 0.05
    _reseal(tampered, "plan_id")
    with pytest.raises(ValueError, match="expected sensor cost"):
        verify_sensor_reveal_trace(
            manifest.as_dict(),
            tampered,
            trace.as_dict(),
        )


def test_trace_cost_accounting_cannot_be_resealed() -> None:
    manifest, truth, submission, plan = _simple_case()
    trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
    tampered = trace.as_dict()
    tampered["total_sensor_cost"] = float(tampered["total_sensor_cost"]) + 0.1
    _reseal(tampered, "trace_id")
    with pytest.raises(ValueError, match="total sensor cost"):
        verify_sensor_reveal_trace(manifest.as_dict(), plan.as_dict(), tampered)


def test_truth_commitment_detects_outcome_or_loss_mutation() -> None:
    manifest, truth, _submission, _plan = _simple_case()
    with pytest.raises(ValueError, match="public commitment"):
        SensorRevealTruth(
            manifest=manifest,
            sensor_outcome_indices=(1, 0),
            sensor_payload_sha256=truth.sensor_payload_sha256,
            sensor_adapter_ids=truth.sensor_adapter_ids,
            realized_action_losses=truth.realized_action_losses,
            truth_nonce=truth.truth_nonce,
        )
    with pytest.raises(ValueError, match="public commitment"):
        SensorRevealTruth(
            manifest=manifest,
            sensor_outcome_indices=truth.sensor_outcome_indices,
            sensor_payload_sha256=truth.sensor_payload_sha256,
            sensor_adapter_ids=truth.sensor_adapter_ids,
            realized_action_losses=(0.1, 1.0, 0.6),
            truth_nonce=truth.truth_nonce,
        )


def test_no_guaranteed_sensor_path_returns_exact_fallback_without_reveal() -> None:
    probe = FiniteProbe(
        "noisy-camera",
        ((0.8, 0.2), (0.2, 0.8)),
        cost=0.10,
        outcome_names=("left-like", "right-like"),
    )
    manifest, truth = seal_sensor_reveal_case(
        case_id="test/fallback",
        public_context_id=_digest("fallback-context"),
        action_names=("left", "right", "exact-fallback"),
        fallback_action_index=2,
        sensor_names=(probe.name,),
        sensor_outcome_names=(probe.outcome_names,),
        sensor_costs=(probe.cost,),
        sensor_risks=(probe.risk,),
        sensor_outcome_indices=(0,),
        sensor_payload_sha256=(_digest("fallback-payload"),),
        sensor_adapter_ids=(_digest("fallback-adapter"),),
        realized_action_losses=(0.0, 1.0, 0.6),
        truth_nonce=_digest("fallback-nonce"),
    )
    submission = SensorRevealSubmission(
        case_id=manifest.case_id,
        manifest_id=manifest.manifest_id,
        provider_id=_digest("fallback-provider"),
        implementation_id=_digest("fallback-implementation"),
        hypothesis_weights=(0.5, 0.5),
        hypothesis_losses=((0.0, 1.0, 0.6), (1.0, 0.0, 0.6)),
        probes=(probe,),
        max_probes=1,
    )
    plan = build_sensor_reveal_plan(manifest, submission)
    trace = execute_sensor_reveal_plan(manifest, truth, submission, plan)
    assert plan.policy["mode"] == "fallback"
    assert trace.terminal_mode == "fallback"
    assert trace.action_index == manifest.fallback_action_index
    assert trace.action_name == "exact-fallback"
    assert trace.events == ()
    assert trace.total_sensor_cost == pytest.approx(0.0)


def test_verifier_does_not_import_producer_or_planner() -> None:
    source = Path("src/causal4d/sensor_reveal_verifier.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            imported.add(node.module)
    assert "causal4d.sensor_reveal_challenge" not in imported
    assert "causal4d.sequential_decision_identification" not in imported


def test_controlled_study_is_deterministic_and_establishes_strict_separation(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    environment = {**os.environ, "PYTHONPATH": "src"}
    command = [
        sys.executable,
        "scripts/experiments/proof_carrying_sensor_reveal.py",
    ]
    subprocess.run(
        [*command, "--output", str(first)],
        check=True,
        env=environment,
    )
    subprocess.run(
        [*command, "--output", str(second)],
        check=True,
        env=environment,
    )
    assert first.read_bytes() == second.read_bytes()
    result = json.loads(first.read_text(encoding="utf-8"))
    routing = result["routing"]
    assert routing["case_count"] == 32
    assert routing["sequential_first_sensor"] == "route-camera"
    assert routing["one_step_sensor"] == "global-task-camera"
    assert routing["generic_information_sensor"] == "nuisance-four-way"
    assert routing["minimum_nonadaptive_sensor_set"] == ["global-task-camera"]
    assert routing["sequential_mean_sensor_cost"] == pytest.approx(0.35)
    assert routing["one_step_sensor_cost"] == pytest.approx(0.50)
    assert routing["terminal_action_accuracy"] == pytest.approx(1.0)
    assert routing["terminal_full_state_identified_count"] == 0
    assert routing["minimum_terminal_support_count"] == 8
    assert routing["unrequested_payload_digest_disclosure_count"] == 0
    fallback = result["fail_closed_control"]
    assert fallback["fallback_used"] is True
    assert fallback["revealed_sensor_count"] == 0
