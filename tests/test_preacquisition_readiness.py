from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_gate_validation as gate_validation_module
import causal4d.preacquisition_readiness as readiness_module
import causal4d.preacquisition_readiness_contracts as readiness_contracts
from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d.preacquisition_readiness import (
    GATE_PATHS,
    SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    evaluate_preacquisition_readiness,
    gate_evidence_sha256,
    gate_evidence_template,
    readiness_evidence_sha256,
    readiness_status_sha256,
    scaffold_preacquisition_readiness,
    seal_preacquisition_gate,
    source_panel_execution_manifest_template,
)


def _registered_values() -> tuple[dict, dict, dict]:
    protocol = {
        "protocol_id": "test-protocol",
        "design_sha256": "a" * 64,
        "executions": [],
        "quality_gates": {"maximum_rgbd_actuator_sync_error_ms": 10.0},
    }
    executions = [
        {
            "execution_id": f"source-{index:02d}",
            "session_id": f"session-{index:02d}",
            "command_profile_id": "test-profile",
        }
        for index in range(12)
    ]
    v2 = {
        "preacquisition_signature_panel": {
            "executions": executions,
            "profiles": [{"id": "test-profile"}],
        }
    }
    v4 = {
        "plan_id": "test-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, v2, v4


def _prerequisites(*, frozen_at: str = "2026-07-30T12:00:00Z") -> dict:
    names = (
        "dataset_protocol",
        "acquisition_schedule",
        "object_registration",
        "slip_pilot",
        "timebase_calibration",
        "contact_registration",
        "method_freeze",
        "method_freeze_validation",
    )
    values = {name: {"present": True, "valid": True, "error": None} for name in names}
    values["method_freeze"].update(
        {
            "frozen_at_utc": frozen_at,
            "sha256": "c" * 64,
            "causal4d_commit_sha": "d" * 40,
            "bayesian_phystwin_commit_sha": "e" * 40,
        }
    )
    values["method_freeze_validation"].update(
        {
            "verified_at_utc": "2026-07-30T12:05:00Z",
            "sha256": "f" * 64,
        }
    )
    return values


def _real_status(**overrides: int) -> dict:
    status = {
        "prerequisites": _prerequisites(),
        "manifest_executions": 0,
        "acquired_executions": 0,
        "validated_executions": 0,
    }
    status.update(overrides)
    return status


def _gate_result(
    gate_id: str,
    *,
    valid: bool = True,
    template: bool = False,
    approved_at: str = "2026-07-30T11:00:00Z",
) -> dict:
    return {
        "gate_id": gate_id,
        "path": f"/data/{GATE_PATHS[gate_id]}",
        "present": True,
        "template": template,
        "valid": valid,
        "approved_at_utc": approved_at if valid else None,
        "error": None if valid else "incomplete",
    }


def _descriptor(root: Path, relative: str) -> dict:
    path = root / relative
    payload = path.read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _seal_gate_payload(gate: dict) -> dict:
    gate["status"] = "passed"
    gate["completed_at_utc"] = "2026-07-30T10:00:00Z"
    gate["locked_before_confirmatory_collection"] = True
    gate["target_outcomes_used"] = False
    gate["approval"] = {
        "approved": True,
        "approver_id": "reviewer",
        "approved_at_utc": "2026-07-30T10:05:00Z",
    }
    gate["artifact_sha256"] = gate_evidence_sha256(gate)
    return gate


def _patch_gate_results(monkeypatch, overrides: dict[str, dict] | None = None) -> None:
    results = {gate_id: _gate_result(gate_id) for gate_id in GATE_PATHS}
    results["software_environment_locked"] = _gate_result(
        "software_environment_locked",
        approved_at="2026-07-30T12:10:00Z",
    )
    results.update(overrides or {})
    registry = {
        "artifact_sha256": "1" * 64,
        "sealed_at_utc": "2026-07-30T08:00:00Z",
    }

    def validate(gate_id, *args, **kwargs):
        del args, kwargs
        return deepcopy(results[gate_id])

    def load_registry(protocol, v4, root):
        del protocol, v4
        return (
            {
                "path": str(Path(root) / "preacquisition/operator_registry.json"),
                "present": True,
                "valid": True,
                "artifact_sha256": registry["artifact_sha256"],
                "sealed_at_utc": registry["sealed_at_utc"],
                "error": None,
            },
            registry,
        )

    def identity_bindings(root, supplied_registry):
        assert supplied_registry is registry
        return {
            "path": str(Path(root) / "preacquisition/operator_registry.json"),
            "present": True,
            "template": False,
            "valid": True,
            "passed": True,
            "operator_registry_artifact_sha256": registry["artifact_sha256"],
            "source_sha256": {"identity-fixture": "2" * 64},
            "error": None,
        }

    monkeypatch.setattr(readiness_module, "_validate_gate_file", validate)
    monkeypatch.setattr(
        readiness_module,
        "load_operator_registry_prerequisite",
        load_registry,
    )
    monkeypatch.setattr(
        readiness_module,
        "validate_preacquisition_identity_bindings",
        identity_bindings,
    )
    monkeypatch.setattr(
        readiness_module,
        "validate_gate_file_operator_identity",
        lambda gate_id, path, registry, prerequisites: {
            "approver_operator_id": f"approver.{gate_id}",
            "approver_person_identity_sha256": "3" * 64,
        },
    )


def test_gate_template_binds_the_exact_registered_source_panel() -> None:
    protocol, v2, v4 = _registered_values()
    template = gate_evidence_template(
        "signature_panel_complete",
        protocol,
        v2,
        v4,
    )

    assert template["status"] == "template"
    assert template["checks"]["execution_ids"] == [
        f"source-{index:02d}" for index in range(12)
    ]
    assert template["checks"]["independent_session_count"] == 12
    assert template["checks"]["manifest_files"]["source-00"] == (
        SOURCE_PANEL_MANIFEST_PATH.format(execution_id="source-00")
    )
    assert template["artifact_sha256"] is None


def test_source_execution_template_preserves_the_registered_boundary() -> None:
    protocol, _, v4 = _registered_values()
    manifest = source_panel_execution_manifest_template(
        {"execution_id": "source-00", "session_id": "session-00"},
        protocol,
        v4,
    )

    assert manifest["status"] == "template"
    assert manifest["protocol_design_sha256"] == protocol["design_sha256"]
    assert manifest["preacquisition_amendment_sha256"] == v4["amendment_sha256"]
    assert manifest["fresh_reset_and_fresh_grasp"] is True
    assert manifest["confirmatory_fold_member"] is False
    assert manifest["target_outcomes_used"] is False
    assert manifest["included"] is None


def test_scaffold_writes_gates_and_source_manifest_templates(
    monkeypatch, tmp_path: Path
) -> None:
    protocol, v2, v4 = _registered_values()
    monkeypatch.setattr(
        readiness_module,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, v2, {}, v4),
    )

    first = scaffold_preacquisition_readiness(tmp_path, tmp_path)
    second = scaffold_preacquisition_readiness(tmp_path, tmp_path)

    assert len(first["created"]) == len(GATE_PATHS) + 12
    assert first["existing"] == []
    assert second["created"] == []
    assert len(second["existing"]) == len(GATE_PATHS) + 12
    source_template = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id="source-00"
    )
    assert json.loads(source_template.read_text(encoding="utf-8"))["status"] == (
        "template"
    )


def test_gate_digest_changes_when_evidence_changes() -> None:
    protocol, v2, v4 = _registered_values()
    gate = gate_evidence_template("support_registration_passed", protocol, v2, v4)
    gate["status"] = "passed"
    gate["checks"]["world_frame_id"] = "world"
    first = gate_evidence_sha256(gate)
    gate["checks"]["world_frame_id"] = "world-revised"
    second = gate_evidence_sha256(gate)

    assert first != second


def test_signature_gate_validates_all_bound_source_manifests(tmp_path: Path) -> None:
    protocol, v2, v4 = _registered_values()
    gate = gate_evidence_template("signature_panel_complete", protocol, v2, v4)
    evidence = []
    for execution in v2["preacquisition_signature_panel"]["executions"]:
        execution_id = execution["execution_id"]
        manifest_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
        data_relative = (
            f"preacquisition/source_panel/executions/{execution_id}/measurement.bin"
        )
        data_path = tmp_path / data_relative
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_bytes(execution_id.encode("utf-8"))
        template = source_panel_execution_manifest_template(
            execution,
            protocol,
            v4,
        )
        template_relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution_id
        )
        template_path = tmp_path / template_relative
        template_path.parent.mkdir(parents=True, exist_ok=True)
        template_path.write_text(
            json.dumps(template, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        manifest = deepcopy(template)
        manifest.update(
            {
                "status": "complete",
                "included": True,
                "started_at_utc": "2026-07-30T09:00:00Z",
                "ended_at_utc": "2026-07-30T09:01:00Z",
                "artifacts": [_descriptor(tmp_path, data_relative)],
            }
        )
        manifest_path = tmp_path / manifest_relative
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        evidence.append(_descriptor(tmp_path, manifest_relative))
    gate["evidence"] = evidence
    _seal_gate_payload(gate)
    gate_path = tmp_path / GATE_PATHS["signature_panel_complete"]
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(gate, sort_keys=True, allow_nan=False), encoding="utf-8"
    )

    result = readiness_module._validate_gate_file(
        "signature_panel_complete",
        gate_path,
        protocol=protocol,
        v2=v2,
        v4=v4,
        dataset_root=tmp_path,
        prerequisites={},
        verify_file_hashes=True,
    )

    assert result["valid"] is True
    assert result["error"] is None
    assert len(result["source_panel_evidence_sha256"]) == 64


def test_software_gate_binds_freeze_attestation_and_distributions(
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    paths = {
        "causal4d": "software/causal4d-0.4.1.whl",
        "bayesian_phystwin": "software/bayesian_phystwin-0.4.0.whl",
    }
    environment_report = "software/resolved-environment.txt"
    for name, relative in paths.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"{name}-wheel".encode("utf-8"))
    report_path = tmp_path / environment_report
    report_path.write_text(
        "numpy==1.26.4\nscipy==1.12.0\ntorch==2.5.1\nwarp-lang==1.6.0\n",
        encoding="utf-8",
    )
    prerequisites = _prerequisites()
    gate = gate_evidence_template("software_environment_locked", protocol, v2, v4)
    gate["checks"] = {
        "method_freeze_sha256": prerequisites["method_freeze"]["sha256"],
        "method_freeze_validation_sha256": prerequisites["method_freeze_validation"][
            "sha256"
        ],
        "causal4d": {
            "commit_sha": prerequisites["method_freeze"]["causal4d_commit_sha"],
            "version": "0.4.1",
            "distribution": _descriptor(tmp_path, paths["causal4d"]),
        },
        "bayesian_phystwin": {
            "commit_sha": prerequisites["method_freeze"][
                "bayesian_phystwin_commit_sha"
            ],
            "version": "0.4.0",
            "distribution": _descriptor(tmp_path, paths["bayesian_phystwin"]),
        },
        "prob4d": {
            "used": False,
            "reason": "registered physical sensors provide the observations",
        },
        "observation_producer": {
            "name": "registered-rgbd-pipeline",
            "version": "1",
            "artifact_contract": "Causal4DObservationV1",
        },
        "python": {
            "version": "3.11.9",
            "implementation": "CPython",
            "platform": "linux-x86_64",
        },
        "runtime_environment": {
            "resolved_dependency_report": environment_report,
            "execution_backend": "cuda",
            "containerized": True,
            "container_image_digest": "sha256:" + "1" * 64,
            "numpy_version": "1.26.4",
            "scipy_version": "1.12.0",
            "torch_version": "2.5.1",
            "warp_version": "1.6.0",
            "opencv_version": "4.10.0",
            "cuda_runtime_version": "12.4",
            "cuda_driver_version": "550.54",
        },
    }
    gate["evidence"] = [
        _descriptor(tmp_path, paths["causal4d"]),
        _descriptor(tmp_path, paths["bayesian_phystwin"]),
        _descriptor(tmp_path, environment_report),
    ]
    _seal_gate_payload(gate)
    gate["approval"]["approved_at_utc"] = "2026-07-30T12:10:00Z"
    gate["completed_at_utc"] = "2026-07-30T12:06:00Z"
    gate["artifact_sha256"] = gate_evidence_sha256(gate)
    gate_path = tmp_path / GATE_PATHS["software_environment_locked"]
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(
        json.dumps(gate, sort_keys=True, allow_nan=False), encoding="utf-8"
    )

    result = readiness_module._validate_gate_file(
        "software_environment_locked",
        gate_path,
        protocol=protocol,
        v2=v2,
        v4=v4,
        dataset_root=tmp_path,
        prerequisites=prerequisites,
        verify_file_hashes=True,
    )

    assert result["valid"] is True
    assert result["error"] is None


def test_readiness_opens_only_when_every_artifact_gate_passes(
    monkeypatch, tmp_path: Path
) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(monkeypatch)

    status = evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        tmp_path,
        _real_status(),
        verify_file_hashes=True,
    )

    assert status["valid"] is True
    assert status["ready"] is True
    assert status["collection_gate"]["first_confirmatory_execution_allowed"] is True
    assert status["blockers"] == []
    assert status["evidence_sha256"] == readiness_evidence_sha256(status)
    assert status["status_sha256"] == readiness_status_sha256(status)


def test_v5_mode0_crosscheck_is_a_fail_closed_prerequisite(
    monkeypatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    v4["prospective_mode0_reset_crosscheck"] = {}
    _patch_gate_results(monkeypatch)

    with pytest.raises(ValueError, match="missing the reset mode-0 prerequisite"):
        evaluate_preacquisition_readiness(
            protocol,
            v2,
            v4,
            tmp_path,
            _real_status(),
            verify_file_hashes=True,
        )

    real_status = _real_status()
    real_status["prerequisites"]["reset_mode0_crosscheck"] = {
        "path": str(tmp_path / "preacquisition/reset-mode0-crosscheck.json"),
        "present": False,
        "template": False,
        "valid": False,
        "error": "reset-mode0-crosscheck.json is missing",
    }
    status = evaluate_preacquisition_readiness(
        protocol, v2, v4, tmp_path, real_status, verify_file_hashes=True
    )

    assert "reset_mode0_crosscheck" in status["missing_prerequisites"]
    assert status["collection_gate"]["reset_mode0_crosscheck_completed"] is False
    assert status["ready"] is False


def test_template_gate_is_valid_but_not_ready(monkeypatch, tmp_path: Path) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(
        monkeypatch,
        {
            "end_to_end_dry_run_passed": _gate_result(
                "end_to_end_dry_run_passed",
                valid=False,
                template=True,
            )
        },
    )

    status = evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        tmp_path,
        _real_status(),
        verify_file_hashes=True,
    )

    assert status["valid"] is True
    assert status["ready"] is False
    assert status["missing_or_template_gates"] == ["end_to_end_dry_run_passed"]
    assert "gate:end_to_end_dry_run_passed" in status["blockers"]


def test_confirmatory_manifest_before_readiness_is_invalid(
    monkeypatch, tmp_path: Path
) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(monkeypatch)

    status = evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        tmp_path,
        _real_status(manifest_executions=1),
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["ready"] is False
    assert "confirmatory_collection_already_started" in status["blockers"]


def test_readiness_rejects_noninteger_confirmatory_counts(
    monkeypatch, tmp_path: Path
) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(monkeypatch)

    with pytest.raises(ValueError, match="manifest_executions"):
        evaluate_preacquisition_readiness(
            protocol,
            v2,
            v4,
            tmp_path,
            _real_status(manifest_executions=0.0),
            verify_file_hashes=True,
        )


def test_operational_gate_may_not_postdate_method_freeze(
    monkeypatch, tmp_path: Path
) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(
        monkeypatch,
        {
            "signature_panel_complete": _gate_result(
                "signature_panel_complete",
                approved_at="2026-07-30T13:00:00Z",
            )
        },
    )

    status = evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        tmp_path,
        _real_status(),
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["ready"] is False
    assert status["chronology_blockers"] == [
        "method_freeze_precedes_operational_gate:signature_panel_complete"
    ]


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        ({"valid": True, "ready": True, "passed": True}, 0),
        ({"valid": True, "ready": False, "passed": False}, 3),
        ({"valid": False, "ready": False, "passed": False}, 2),
    ],
)
def test_require_ready_exit_codes(monkeypatch, capsys, status, expected) -> None:
    monkeypatch.setattr(
        readiness_cli,
        "build_preacquisition_readiness",
        lambda *args, **kwargs: dict(status),
    )

    result = readiness_cli.main(["status", "repository", "dataset", "--require-ready"])

    assert result == expected
    assert json.loads(capsys.readouterr().out)["ready"] is status["ready"]


def test_seal_refuses_to_rewrite_existing_gate(monkeypatch, tmp_path: Path) -> None:
    protocol, v2, v4 = _registered_values()
    monkeypatch.setattr(
        gate_validation_module,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, v2, {}, v4),
    )
    path = tmp_path / GATE_PATHS["support_registration_passed"]
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "artifact_sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already sealed"):
        seal_preacquisition_gate(
            tmp_path,
            tmp_path,
            "support_registration_passed",
            approved_by="reviewer",
        )


def test_grouped_readiness_route_is_registered() -> None:
    from causal4d.cli.command_registry import find_command

    command = find_command("protocol/readiness")
    assert command.lifecycle == "stable"
    assert command.historical_name is None
    assert command.target.endswith("preacquisition_readiness:main")


def test_evidence_descriptor_rejects_symlink_escape(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"outside")
    linked = root / "linked.bin"
    try:
        linked.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    descriptor = {
        "path": "linked.bin",
        "sha256": hashlib.sha256(outside.read_bytes()).hexdigest(),
        "bytes": outside.stat().st_size,
    }

    with pytest.raises(ValueError, match="symlink"):
        readiness_contracts._validate_descriptor(
            root,
            descriptor,
            name="linked evidence",
            verify_file_hashes=True,
        )


def test_readiness_evidence_digest_is_mount_independent(
    monkeypatch, tmp_path: Path
) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(monkeypatch)
    first = evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        tmp_path / "first-mount",
        _real_status(),
        verify_file_hashes=True,
    )
    relocated = deepcopy(first)
    relocated["dataset_root"] = "/archive/relocated-dataset"
    for section in ("prerequisites", "operational_gates"):
        for record in relocated[section].values():
            record["path"] = "/archive/" + Path(record.get("path", "evidence")).name
    relocated["evidence_sha256"] = readiness_evidence_sha256(relocated)
    relocated["status_sha256"] = readiness_status_sha256(relocated)

    assert relocated["evidence_sha256"] == first["evidence_sha256"]
    assert relocated["status_sha256"] != first["status_sha256"]


def test_missing_operator_registry_is_incomplete_not_malformed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _registered_values()
    _patch_gate_results(monkeypatch)
    monkeypatch.setattr(
        readiness_module,
        "load_operator_registry_prerequisite",
        lambda protocol, v4, root: (
            {
                "path": str(Path(root) / "preacquisition/operator_registry.json"),
                "present": False,
                "valid": False,
                "error": "operator_registry.json is missing",
            },
            None,
        ),
    )
    monkeypatch.setattr(
        readiness_module,
        "validate_preacquisition_identity_bindings",
        lambda root, registry: {
            "path": str(Path(root) / "preacquisition/operator_registry.json"),
            "present": False,
            "template": True,
            "valid": False,
            "passed": False,
            "error": "operator identity bindings are incomplete",
        },
    )

    status = evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        tmp_path,
        _real_status(),
        verify_file_hashes=True,
    )

    assert status["valid"] is True
    assert status["ready"] is False
    assert "operator_registry" in status["missing_prerequisites"]
    assert "operator_identity_bindings" in status["missing_prerequisites"]
    assert status["malformed_prerequisites"] == []
    assert status["malformed_gates"] == []
    assert set(status["missing_or_template_gates"]) == set(GATE_PATHS)
