from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d.preacquisition_next_action import (
    derive_preacquisition_next_action,
    next_action_evidence_sha256,
    render_preacquisition_next_action_markdown,
)


def _prerequisite(*, present: bool = True, valid: bool = True, template=False):
    return {
        "present": present,
        "valid": valid,
        "template": template,
        "error": None if valid else "invalid",
    }


def _gate(*, present: bool = True, valid: bool = True, template=False):
    return {
        "present": present,
        "valid": valid,
        "template": template,
        "identity_pending": False,
        "error": None if valid else "invalid",
    }


def _readiness() -> dict:
    prerequisites = {
        name: _prerequisite()
        for name in (
            "dataset_protocol",
            "acquisition_schedule",
            "object_registration",
            "slip_pilot",
            "timebase_calibration",
            "contact_registration",
            "operator_registry",
            "operator_identity_bindings",
            "method_freeze",
            "method_freeze_validation",
        )
    }
    prerequisites["operator_registry"]["independent_verifier_available"] = True
    gates = {
        name: _gate()
        for name in (
            "signature_panel_complete",
            "actuator_sync_passed",
            "support_registration_passed",
            "end_to_end_dry_run_passed",
            "software_environment_locked",
        )
    }
    return {
        "schema_version": 2,
        "artifact_kind": "PreacquisitionReadinessStatus",
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "dataset_root": "/data/run",
        "verify_file_hashes": True,
        "prerequisites": prerequisites,
        "operational_gates": gates,
        "missing_prerequisites": [],
        "malformed_prerequisites": [],
        "missing_or_template_gates": [],
        "malformed_gates": [],
        "chronology_blockers": [],
        "blockers": [],
        "confirmatory_collection": {
            "manifest_executions": 0,
            "acquired_executions": 0,
            "validated_executions": 0,
            "not_started": True,
        },
        "evidence_sha256": "c" * 64,
        "status_sha256": "d" * 64,
        "valid": True,
        "ready": False,
    }


def _source_panel(*, complete: bool = True) -> dict:
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DSourcePanelStatus",
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "dataset_root": "/data/run",
        "verify_file_hashes": True,
        "specified_executions": 12,
        "validated_executions": 12 if complete else 0,
        "missing_template_ids": [],
        "blockers": [],
        "next_execution": None,
        "evidence_sha256": "e" * 64,
        "status_sha256": "f" * 64,
        "valid": True,
        "complete": complete,
        "target_outcomes_used": False,
    }


def _derive(readiness: dict, source_panel: dict) -> dict:
    return derive_preacquisition_next_action(
        readiness,
        source_panel,
        repository_root="/repo",
        dataset_root="/data/run",
    )


def test_missing_preacquisition_scaffold_is_the_only_action() -> None:
    readiness = _readiness()
    for gate in readiness["operational_gates"].values():
        gate.update(present=False, valid=False)
    source = _source_panel(complete=False)
    source["valid"] = False
    source["missing_template_ids"] = [f"source-{index}" for index in range(12)]

    decision = _derive(readiness, source)

    assert decision["valid"] is True
    assert decision["action"]["action_id"] == "scaffold_preacquisition_evidence"
    assert decision["action"]["automatable"] is True
    assert "readiness scaffold" in decision["action"]["command_text"]


def test_missing_operator_registry_precedes_physical_work() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["operator_registry"] = _prerequisite(
        present=False, valid=False
    )
    readiness["missing_prerequisites"] = ["operator_registry"]

    decision = _derive(readiness, _source_panel())

    assert decision["action"]["action_id"] == "scaffold_operator_registry"
    assert decision["action"]["physical_acquisition_required"] is False


def test_existing_operator_registry_template_advances_to_human_seal() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["operator_registry"] = {
        **_prerequisite(present=False, valid=False),
        "template_status": {
            "path": "/data/run/preacquisition/operator_registry.template.json",
            "present": True,
            "valid": True,
            "template": True,
            "error": None,
            "operator_count": 0,
        },
    }
    readiness["missing_prerequisites"] = ["operator_registry"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert decision["valid"] is True
    assert action["action_id"] == "seal_operator_registry"
    assert action["automatable"] is False
    assert action["physical_acquisition_required"] is False
    assert action["input_paths"] == [
        "/data/run/preacquisition/operator_registry.template.json"
    ]
    assert action["output_paths"] == ["/data/run/preacquisition/operator_registry.json"]


def test_single_person_registry_blocks_before_manual_or_physical_work() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["operator_registry"][
        "independent_verifier_available"
    ] = False
    readiness["prerequisites"]["object_registration"] = _prerequisite(
        present=False,
        valid=False,
    )
    readiness["missing_prerequisites"] = ["object_registration"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert decision["valid"] is True
    assert action["action_id"] == "stop_independent_verifier_unavailable"
    assert action["category"] == "governance_blocker"
    assert action["automatable"] is False
    assert action["physical_acquisition_required"] is False
    assert action["target_outcomes_permitted"] is False
    assert action["blocking_items"] == [
        "single_operator_project_cannot_satisfy_independent_verification"
    ]


def test_invalid_operator_registry_template_requires_repair() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["operator_registry"] = {
        **_prerequisite(present=False, valid=False),
        "template_status": {
            "path": "/data/run/preacquisition/operator_registry.template.json",
            "present": True,
            "valid": False,
            "template": True,
            "error": "ValueError: protocol digest mismatch",
        },
    }
    readiness["missing_prerequisites"] = ["operator_registry"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert decision["valid"] is False
    assert action["action_id"] == "stop_and_repair_invalid_evidence"
    assert action["automatable"] is False
    assert action["blocking_items"] == [
        "operator_registry_template_invalid",
        "ValueError: protocol digest mismatch",
    ]


def test_malformed_evidence_precedes_missing_operator_registry() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["operator_registry"] = _prerequisite(
        present=False, valid=False
    )
    readiness["missing_prerequisites"] = ["operator_registry"]
    readiness["valid"] = False
    readiness["malformed_gates"] = ["actuator_sync_passed"]
    readiness["blockers"] = ["gate_invalid:actuator_sync_passed"]

    decision = _derive(readiness, _source_panel())

    assert decision["valid"] is False
    assert decision["action"]["action_id"] == "stop_and_repair_invalid_evidence"
    assert decision["action"]["blocking_items"] == ["gate_invalid:actuator_sync_passed"]


def test_manual_prerequisite_names_exact_paths() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["object_registration"] = _prerequisite(
        present=False, valid=False
    )
    readiness["missing_prerequisites"] = ["object_registration"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert action["action_id"] == "complete_object_registration"
    assert action["command_argv"] is None
    assert action["output_paths"] == ["/data/run/object_registration.json"]
    assert "protocol real status" in action["completion_check_text"]


def test_source_panel_action_binds_exact_registered_execution() -> None:
    readiness = _readiness()
    source = _source_panel(complete=False)
    source["next_execution"] = {
        "execution_id": "source-03",
        "session_id": "session-03",
        "command_profile_id": "lift-high",
        "template_path": (
            "preacquisition/source_panel/executions/source-03/manifest.template.json"
        ),
        "manifest_path": (
            "preacquisition/source_panel/executions/source-03/manifest.json"
        ),
        "profile": {"id": "lift-high", "speed": "high"},
    }

    decision = _derive(readiness, source)

    action = decision["action"]
    assert action["action_id"] == "acquire_next_source_panel_execution"
    assert action["physical_acquisition_required"] is True
    assert action["registered_execution"]["execution_id"] == "source-03"
    assert action["after_completion_argv"][-1] == "/data/run/staging/source-03.json"


def test_freeze_attestation_and_software_gate_are_strictly_ordered() -> None:
    readiness = _readiness()
    readiness["prerequisites"]["method_freeze"] = _prerequisite(
        present=False, valid=False
    )
    readiness["missing_prerequisites"] = ["method_freeze"]
    decision = _derive(readiness, _source_panel())
    assert decision["action"]["action_id"] == "seal_method_freeze"

    readiness = _readiness()
    readiness["prerequisites"]["method_freeze_validation"] = _prerequisite(
        present=False, valid=False
    )
    readiness["missing_prerequisites"] = ["method_freeze_validation"]
    decision = _derive(readiness, _source_panel())
    assert decision["action"]["action_id"] == "attest_method_freeze"

    readiness = _readiness()
    readiness["operational_gates"]["software_environment_locked"] = _gate(
        template=True, valid=False
    )
    readiness["missing_or_template_gates"] = ["software_environment_locked"]
    decision = _derive(readiness, _source_panel())
    assert decision["action"]["action_id"] == (
        "complete_and_seal_software_environment_locked"
    )


def test_invalid_evidence_never_suggests_an_automatic_repair() -> None:
    readiness = _readiness()
    readiness["valid"] = False
    readiness["malformed_gates"] = ["actuator_sync_passed"]
    readiness["blockers"] = ["gate_invalid:actuator_sync_passed"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert decision["valid"] is False
    assert action["action_id"] == "stop_and_repair_invalid_evidence"
    assert action["automatable"] is False
    assert action["target_outcomes_permitted"] is False


def test_ready_state_authorizes_only_locked_order() -> None:
    readiness = _readiness()
    readiness["ready"] = True

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert decision["ready"] is True
    assert action["action_id"] == "begin_first_confirmatory_session"
    assert "freeze validate" in action["command_text"]
    assert action["physical_acquisition_required"] is True


def test_portable_evidence_hash_ignores_mount_points() -> None:
    first = _derive(_readiness(), _source_panel())
    second = deepcopy(first)
    second["repository_root"] = "/different/repo"
    second["dataset_root"] = "/different/data"
    for field in ("command_argv", "completion_check_argv", "after_completion_argv"):
        argv = second["action"].get(field)
        if argv:
            second["action"][field] = [
                value.replace("/repo", "/different/repo").replace(
                    "/data/run", "/different/data"
                )
                for value in argv
            ]
    for field in ("command_text", "completion_check_text", "after_completion_text"):
        value = second["action"].get(field)
        if value:
            second["action"][field] = value.replace("/repo", "/different/repo").replace(
                "/data/run", "/different/data"
            )
    second.pop("evidence_sha256")
    second.pop("status_sha256")

    assert next_action_evidence_sha256(first) == next_action_evidence_sha256(second)


def test_markdown_contains_action_and_content_id() -> None:
    decision = _derive(_readiness(), _source_panel())

    markdown = render_preacquisition_next_action_markdown(decision)

    assert "# Causal4D pre-acquisition next action" in markdown
    assert decision["action"]["action_id"] in markdown
    assert decision["evidence_sha256"] in markdown
    assert "Target outcomes permitted: `false`" in markdown


def test_cli_parser_exposes_hash_verified_next_action_by_default() -> None:
    args = readiness_cli.build_parser().parse_args(
        ["next-action", "/repo", "/data/run"]
    )

    assert args.command == "next-action"
    assert args.skip_file_hashes is False


def test_cli_returns_incomplete_exit_code_and_writes_reports(
    monkeypatch, tmp_path: Path
) -> None:
    decision = _derive(_readiness(), _source_panel())
    written: list[tuple[str, Path]] = []
    monkeypatch.setattr(
        readiness_cli,
        "build_preacquisition_next_action",
        lambda *args, **kwargs: decision,
    )
    monkeypatch.setattr(
        readiness_cli,
        "write_preacquisition_next_action",
        lambda path, value: written.append(("json", Path(path))) or Path(path),
    )
    monkeypatch.setattr(
        readiness_cli,
        "write_preacquisition_next_action_markdown",
        lambda path, value: written.append(("markdown", Path(path))) or Path(path),
    )
    output_json = tmp_path / "next-action.json"
    output_markdown = tmp_path / "next-action.md"

    exit_code = readiness_cli.main(
        [
            "next-action",
            "/repo",
            "/data/run",
            "--output-json",
            str(output_json),
            "--output-markdown",
            str(output_markdown),
        ]
    )

    assert exit_code == 3
    assert written == [("json", output_json), ("markdown", output_markdown)]


def _enable_single_operator_v5(readiness: dict) -> None:
    readiness["governance"] = {
        "mode": "single_operator_self_attested",
        "single_operator_allowed": True,
        "independent_verifier_required": False,
        "independent_preacquisition_attestation_claimed": False,
        "self_attestation_required": True,
    }
    readiness["prerequisites"]["operator_registry"][
        "independent_verifier_available"
    ] = False


def test_v5_single_operator_advances_past_independent_verifier_block() -> None:
    readiness = _readiness()
    _enable_single_operator_v5(readiness)
    readiness["prerequisites"]["object_registration"] = _prerequisite(
        present=False,
        valid=False,
    )
    readiness["missing_prerequisites"] = ["object_registration"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert action["action_id"] == "complete_object_registration"
    assert action["operator_role"] == "self_attesting_operator"
    assert action["target_outcomes_permitted"] is False


def test_v5_method_freeze_action_is_disclosed_self_attestation() -> None:
    readiness = _readiness()
    _enable_single_operator_v5(readiness)
    readiness["prerequisites"]["method_freeze_validation"] = _prerequisite(
        present=False,
        valid=False,
    )
    readiness["missing_prerequisites"] = ["method_freeze_validation"]

    decision = _derive(readiness, _source_panel())

    action = decision["action"]
    assert action["action_id"] == "attest_method_freeze"
    assert action["operator_role"] == "self_attesting_operator"
    assert "Self-attest" in action["title"]
    assert "<registered-freezer-id>" in action["command_argv"]
