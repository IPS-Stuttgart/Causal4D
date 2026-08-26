from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import causal4d.preacquisition_source_panel_control as source_control
from causal4d.cli import preacquisition_readiness as readiness_cli
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    source_panel_execution_manifest_template,
)


def _registered_values() -> tuple[dict, dict, dict]:
    protocol = {
        "protocol_id": "test-protocol",
        "design_sha256": "a" * 64,
    }
    profiles = [
        {"id": "lift_high", "amplitude_m": 0.08},
        {"id": "lower_high", "amplitude_m": 0.08},
        {"id": "lift_high_slow", "amplitude_m": 0.08},
        {"id": "lift_high_long_hold", "amplitude_m": 0.08},
    ]
    executions: list[dict] = []
    for profile in profiles:
        for replicate in range(1, 4):
            identifier = f"source-{profile['id']}-r{replicate}"
            executions.append(
                {
                    "execution_id": identifier,
                    "session_id": identifier,
                    "command_profile_id": profile["id"],
                    "contact_region_id": "upper_torso",
                    "realization_condition_id": "nominal",
                    "replicate": replicate,
                    "fresh_reset_and_fresh_grasp": True,
                    "confirmatory_fold_member": False,
                }
            )
    v2 = {
        "preacquisition_signature_panel": {
            "executions": executions,
            "profiles": profiles,
        }
    }
    v4 = {
        "plan_id": "test-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, v2, v4


def _patch_chain(monkeypatch: pytest.MonkeyPatch) -> tuple[dict, dict, dict]:
    protocol, v2, v4 = _registered_values()
    monkeypatch.setattr(
        source_control,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, v2, {}, v4),
    )
    return protocol, v2, v4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _descriptor(root: Path, relative: str) -> dict:
    data = (root / relative).read_bytes()
    return {
        "path": relative,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def _scaffold(root: Path, protocol: dict, v2: dict, v4: dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in executions:
        relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution["execution_id"]
        )
        template = source_panel_execution_manifest_template(
            execution,
            protocol,
            v4,
        )
        _write_json(root / relative, template)


def _completed_manifest(
    root: Path,
    execution: dict,
    protocol: dict,
    v4: dict,
    *,
    bad_digest: bool = False,
) -> dict:
    execution_id = execution["execution_id"]
    artifact_relative = (
        f"preacquisition/source_panel/executions/{execution_id}/raw.bin"
    )
    artifact_path = root / artifact_relative
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_bytes(f"physical-source:{execution_id}".encode())
    descriptor = _descriptor(root, artifact_relative)
    if bad_digest:
        descriptor["sha256"] = "0" * 64
    manifest = source_panel_execution_manifest_template(
        execution,
        protocol,
        v4,
    )
    manifest.update(
        {
            "status": "complete",
            "included": True,
            "quality_gate_failures": [],
            "started_at_utc": "2026-08-04T08:00:00Z",
            "ended_at_utc": "2026-08-04T08:01:00Z",
            "artifacts": [descriptor],
        }
    )
    return manifest


def _write_final_manifest(root: Path, execution: dict, manifest: dict) -> Path:
    relative = SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=execution["execution_id"]
    )
    path = root / relative
    _write_json(path, manifest)
    return path


def test_status_identifies_exact_first_execution_and_profile(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)

    status = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    first = v2["preacquisition_signature_panel"]["executions"][0]
    assert status["valid"] is True
    assert status["complete"] is False
    assert status["validated_executions"] == 0
    assert status["next_execution"]["execution_id"] == first["execution_id"]
    assert status["next_execution"]["profile"]["id"] == "lift_high"
    assert status["registered_prefix_valid"] is True
    assert status["invalid_template_ids"] == []
    assert status["target_outcomes_used"] is False
    assert status["evidence_sha256"] == (
        source_control.source_panel_evidence_sha256(status)
    )
    assert status["status_sha256"] == (
        source_control.source_panel_status_sha256(status)
    )


def test_publish_is_exactly_once_and_advances_registered_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    executions = v2["preacquisition_signature_panel"]["executions"]
    source = tmp_path / "staging" / "source.json"
    _write_json(
        source,
        _completed_manifest(tmp_path, executions[0], protocol, v4),
    )

    published = source_control.publish_source_panel_manifest(
        tmp_path,
        tmp_path,
        source,
    )

    assert published["execution_id"] == executions[0]["execution_id"]
    assert published["source_panel_status"]["validated_executions"] == 1
    assert published["source_panel_status"]["next_execution"]["execution_id"] == (
        executions[1]["execution_id"]
    )
    with pytest.raises(ValueError, match="next registered execution"):
        source_control.publish_source_panel_manifest(tmp_path, tmp_path, source)


def test_v5_publish_rejects_missing_mode0_audit_before_reading_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, v4 = _patch_chain(monkeypatch)
    v4["prospective_mode0_reset_crosscheck"] = {}
    calls = []

    def missing(protocol, preacquisition, root, *, verify_file_hashes):
        calls.append((protocol, preacquisition, root, verify_file_hashes))
        return {
            "present": False,
            "valid": False,
            "error": "reset-mode0-crosscheck.json is missing",
        }

    monkeypatch.setattr(
        source_control,
        "load_reset_mode0_crosscheck_prerequisite",
        missing,
    )
    with pytest.raises(ValueError, match="requires the valid reset mode-0"):
        source_control.publish_source_panel_manifest(
            tmp_path, tmp_path, tmp_path / "source-must-not-be-read.json"
        )
    assert len(calls) == 1


def test_publish_rejects_bad_artifact_hash_without_final_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    source = tmp_path / "staging" / "bad-source.json"
    _write_json(
        source,
        _completed_manifest(
            tmp_path,
            first,
            protocol,
            v4,
            bad_digest=True,
        ),
    )

    with pytest.raises(ValueError, match="checksum mismatch"):
        source_control.publish_source_panel_manifest(tmp_path, tmp_path, source)

    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=first["execution_id"]
    )
    assert not final.exists()
    assert not list(final.parent.glob(f".{final.name}.*.tmp"))


def test_publish_rejects_nested_target_outcome_field(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    manifest = _completed_manifest(tmp_path, first, protocol, v4)
    manifest["artifacts"][0]["target_metrics"] = {"rmse": 0.0}
    source = tmp_path / "staging" / "target-informed.json"
    _write_json(source, manifest)

    with pytest.raises(ValueError, match="target-outcome fields"):
        source_control.publish_source_panel_manifest(tmp_path, tmp_path, source)


def test_status_rejects_out_of_order_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    second = v2["preacquisition_signature_panel"]["executions"][1]
    manifest = _completed_manifest(tmp_path, second, protocol, v4)
    _write_final_manifest(tmp_path, second, manifest)

    status = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["registered_prefix_valid"] is False
    assert "completed_manifests_do_not_form_registered_prefix" in status["blockers"]


def test_status_rejects_modified_template(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    first = v2["preacquisition_signature_panel"]["executions"][0]
    template = tmp_path / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=first["execution_id"]
    )
    payload = json.loads(template.read_text(encoding="utf-8"))
    payload["included"] = False
    _write_json(template, payload)

    status = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["invalid_template_ids"] == [first["execution_id"]]
    assert status["executions"][0]["template_valid"] is False


def test_status_rejects_unregistered_execution_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    root = tmp_path / "preacquisition" / "source_panel" / "executions"
    (root / "unregistered").mkdir()

    status = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    assert status["valid"] is False
    assert status["unexpected_execution_directories"] == ["unregistered"]


def test_status_rejects_symlinked_dataset_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    real_root = tmp_path / "real"
    _scaffold(real_root, protocol, v2, v4)
    alias = tmp_path / "alias"
    alias.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(ValueError, match="symlink component"):
        source_control.build_source_panel_status(tmp_path, alias)


def test_status_completes_only_after_all_hashes_validate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v2, v4 = _patch_chain(monkeypatch)
    _scaffold(tmp_path, protocol, v2, v4)
    executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in executions:
        manifest = _completed_manifest(tmp_path, execution, protocol, v4)
        _write_final_manifest(tmp_path, execution, manifest)

    unchecked = source_control.build_source_panel_status(tmp_path, tmp_path)
    checked = source_control.build_source_panel_status(
        tmp_path,
        tmp_path,
        verify_file_hashes=True,
    )

    assert unchecked["valid"] is True
    assert unchecked["complete"] is False
    assert "file_hashes_not_verified" in unchecked["blockers"]
    assert checked["valid"] is True
    assert checked["complete"] is True
    assert checked["validated_executions"] == 12
    assert checked["next_execution"] is None


def test_source_panel_cli_uses_incomplete_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        readiness_cli,
        "build_source_panel_status",
        lambda *args, **kwargs: {
            "valid": True,
            "complete": False,
            "passed": False,
        },
    )

    code = readiness_cli.main(
        [
            "source-panel-status",
            "repository",
            "dataset",
            "--require-complete",
        ]
    )

    assert code == 3
    assert '"complete": false' in capsys.readouterr().out
