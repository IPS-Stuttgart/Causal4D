from __future__ import annotations

import io
import json
from pathlib import Path
from zipfile import ZIP_STORED, ZipFile, ZipInfo

import pytest

from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d import preacquisition_next_action_packet as packet_module
from causal4d.preacquisition_next_action_packet import (
    build_preacquisition_next_action_packet_bytes,
    inspect_preacquisition_next_action_packet,
    next_action_packet_id,
    validate_preacquisition_next_action_packet,
    write_preacquisition_next_action_packet,
)
from causal4d.preacquisition_operator_flow import (
    derive_preacquisition_operator_next_action,
    render_preacquisition_operator_next_action_markdown,
)


def _prerequisite() -> dict[str, object]:
    return {
        "present": True,
        "valid": True,
        "template": False,
        "error": None,
    }


def _gate() -> dict[str, object]:
    return {
        "present": True,
        "valid": True,
        "template": False,
        "identity_pending": False,
        "error": None,
    }


def _decision(*, ready: bool = True) -> dict[str, object]:
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
    readiness = {
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
        "ready": ready,
    }
    source = {
        "schema_version": 1,
        "artifact_kind": "Causal4DSourcePanelStatus",
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "dataset_root": "/data/run",
        "verify_file_hashes": True,
        "specified_executions": 12,
        "validated_executions": 12,
        "missing_template_ids": [],
        "blockers": [],
        "next_execution": None,
        "evidence_sha256": "e" * 64,
        "status_sha256": "f" * 64,
        "valid": True,
        "complete": True,
        "target_outcomes_used": False,
    }
    return derive_preacquisition_operator_next_action(
        readiness,
        source,
        repository_root="/repo",
        dataset_root="/data/run",
    )


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    return info


def _tampered_packet(
    packet_bytes: bytes,
    *,
    replace: dict[str, bytes] | None = None,
    extra: tuple[str, bytes] | None = None,
) -> bytes:
    replace = {} if replace is None else replace
    with ZipFile(io.BytesIO(packet_bytes), mode="r") as source:
        members = [(info.filename, source.read(info)) for info in source.infolist()]
    output = io.BytesIO()
    with ZipFile(output, mode="w", compression=ZIP_STORED) as target:
        for name, value in members:
            target.writestr(_zip_info(name), replace.get(name, value))
        if extra is not None:
            target.writestr(_zip_info(extra[0]), extra[1])
    return output.getvalue()


def test_packet_bytes_are_deterministic_and_bind_human_instructions() -> None:
    decision = _decision()

    first_bytes, first_manifest = build_preacquisition_next_action_packet_bytes(
        decision
    )
    second_bytes, second_manifest = build_preacquisition_next_action_packet_bytes(
        decision
    )

    assert first_bytes == second_bytes
    assert first_manifest == second_manifest
    assert first_manifest["packet_id"] == next_action_packet_id(first_manifest)
    assert first_manifest["decision_evidence_sha256"] == decision["evidence_sha256"]
    assert first_manifest["action_identity"]["action_id"] == (
        "stop_independent_verifier_unavailable"
    )

    with ZipFile(io.BytesIO(first_bytes), mode="r") as archive:
        assert archive.namelist() == [
            "decision.json",
            "instructions.md",
            "manifest.json",
        ]
        instructions = archive.read("instructions.md").decode("utf-8")
    assert instructions == render_preacquisition_operator_next_action_markdown(decision)


def test_packet_publication_is_exactly_once_and_self_verifying(tmp_path: Path) -> None:
    decision = _decision()
    packet = tmp_path / "next-action.zip"

    receipt = write_preacquisition_next_action_packet(packet, decision)
    inspected = inspect_preacquisition_next_action_packet(packet)

    assert receipt["packet_file_sha256"] == inspected["packet_file_sha256"]
    assert receipt["packet_id"] == inspected["manifest"]["packet_id"]
    assert receipt["ready"] is True
    assert inspected["decision"] == decision

    with pytest.raises(FileExistsError):
        write_preacquisition_next_action_packet(packet, decision)


def test_packet_rejects_tampered_instructions_and_extra_members(
    tmp_path: Path,
) -> None:
    packet_bytes, _ = build_preacquisition_next_action_packet_bytes(_decision())

    tampered = tmp_path / "tampered.zip"
    tampered.write_bytes(
        _tampered_packet(
            packet_bytes,
            replace={"instructions.md": b"# stale instructions\n"},
        )
    )
    with pytest.raises(ValueError, match="instructions SHA-256 mismatch"):
        inspect_preacquisition_next_action_packet(tampered)

    expanded = tmp_path / "expanded.zip"
    expanded.write_bytes(
        _tampered_packet(packet_bytes, extra=("unexpected.txt", b"unexpected\n"))
    )
    with pytest.raises(ValueError, match="members differ"):
        inspect_preacquisition_next_action_packet(expanded)


def test_packet_validation_requires_the_current_hash_verified_decision(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decision = _decision()
    packet = tmp_path / "next-action.zip"
    write_preacquisition_next_action_packet(packet, decision)

    def validate(repository: Path, dataset: Path, decision_path: Path):
        assert repository == Path("/repo")
        assert dataset == Path("/data/run")
        restored = json.loads(decision_path.read_text(encoding="utf-8"))
        assert restored == decision
        return {
            "decision_evidence_sha256": decision["evidence_sha256"],
            "current_evidence_sha256": decision["evidence_sha256"],
            "current_status_sha256": decision["status_sha256"],
            "safe_to_execute": True,
        }

    monkeypatch.setattr(
        packet_module,
        "validate_preacquisition_next_action_report",
        validate,
    )

    report = validate_preacquisition_next_action_packet(
        "/repo",
        "/data/run",
        packet,
    )

    assert report["decision_current"] is True
    assert report["human_instructions_match_decision"] is True
    assert report["safe_to_execute"] is True
    assert report["target_outcomes_used"] is False
    assert len(report["evidence_sha256"]) == 64
    assert len(report["status_sha256"]) == 64


def test_cli_exposes_packet_publication_and_validation() -> None:
    packet_args = readiness_cli.build_parser().parse_args(
        ["next-action-packet", "/repo", "/data/run", "packet.zip"]
    )
    validation_args = readiness_cli.build_parser().parse_args(
        [
            "next-action-packet-validate",
            "/repo",
            "/data/run",
            "packet.zip",
        ]
    )

    assert packet_args.command == "next-action-packet"
    assert packet_args.output_zip == "packet.zip"
    assert validation_args.command == "next-action-packet-validate"
    assert validation_args.packet_zip == "packet.zip"


def test_cli_packet_preserves_valid_but_incomplete_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    decision = _decision(ready=False)
    packet = tmp_path / "next-action.zip"
    monkeypatch.setattr(
        readiness_cli,
        "build_preacquisition_next_action",
        lambda *args, **kwargs: decision,
    )

    exit_code = readiness_cli.main(
        [
            "next-action-packet",
            "/repo",
            "/data/run",
            str(packet),
        ]
    )

    assert exit_code == 3
    assert packet.is_file()
