from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_source_panel_builder as builder
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
    execution = {
        "execution_id": "source-lift-high-r1",
        "session_id": "session-lift-high-r1",
        "source_panel_execution_index": 0,
        "command_profile_id": "lift_high",
        "template_present": True,
        "template_valid": True,
    }
    v4 = {
        "plan_id": "test-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, execution, v4


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _setup(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
) -> tuple[dict, dict, dict, Path]:
    protocol, execution, v4 = _registered_values()
    root.mkdir(parents=True, exist_ok=True)
    template = source_panel_execution_manifest_template(execution, protocol, v4)
    template_path = root / SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=execution["execution_id"]
    )
    _write_json(template_path, template)
    artifact_root = (
        root
        / "preacquisition"
        / "source_panel"
        / "executions"
        / execution["execution_id"]
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    status = {
        "valid": True,
        "complete": False,
        "evidence_sha256": "c" * 64,
        "status_sha256": "d" * 64,
        "next_execution": execution,
    }
    monkeypatch.setattr(
        builder,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, {}, {}, v4),
    )
    monkeypatch.setattr(
        builder,
        "build_source_panel_status",
        lambda *args, **kwargs: deepcopy(status),
    )
    return protocol, execution, v4, artifact_root


def test_stage_builds_exact_next_manifest_without_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, execution, v4, artifact_root = _setup(monkeypatch, tmp_path)
    first = artifact_root / "rgbd.npz"
    second = artifact_root / "controller.csv"
    first.write_bytes(b"rgbd")
    second.write_bytes(b"controller")

    result = builder.stage_source_panel_manifest(
        tmp_path,
        tmp_path,
        started_at_utc="2026-08-10T08:00:00Z",
        ended_at_utc="2026-08-10T08:01:00Z",
        artifacts=(first, second.relative_to(tmp_path)),
    )

    staged_path = Path(result["source_json"])
    staged = json.loads(staged_path.read_text(encoding="utf-8"))
    expected = source_panel_execution_manifest_template(execution, protocol, v4)
    expected.update(
        {
            "status": "complete",
            "included": True,
            "quality_gate_failures": [],
            "started_at_utc": "2026-08-10T08:00:00Z",
            "ended_at_utc": "2026-08-10T08:01:00Z",
            "artifacts": result["artifacts"],
        }
    )

    assert staged == expected
    assert result["execution_id"] == execution["execution_id"]
    assert result["session_id"] == execution["session_id"]
    assert result["artifact_count"] == 2
    assert [row["path"] for row in result["artifacts"]] == sorted(
        row["path"] for row in result["artifacts"]
    )
    assert result["source_panel_status_stable"] is True
    assert result["ready_for_staged_verification"] is True
    assert result["published"] is False
    assert result["claim_bearing_evidence_mutated"] is False
    assert result["changes_registered_method"] is False
    assert result["target_outcomes_used"] is False
    assert result["staged_verification_command_argv"][3] == (
        "source-panel-verify-staged"
    )
    final = tmp_path / SOURCE_PANEL_MANIFEST_PATH.format(
        execution_id=execution["execution_id"]
    )
    assert not final.exists()


def test_stage_refuses_to_replace_existing_staging_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, execution, _, artifact_root = _setup(monkeypatch, tmp_path)
    artifact = artifact_root / "raw.bin"
    artifact.write_bytes(b"raw")
    target = tmp_path / "staging" / f"{execution['execution_id']}.json"
    target.parent.mkdir(parents=True)
    target.write_text("existing\n", encoding="utf-8")

    with pytest.raises(ValueError, match="staging manifest already exists"):
        builder.stage_source_panel_manifest(
            tmp_path,
            tmp_path,
            started_at_utc="2026-08-10T08:00:00Z",
            ended_at_utc="2026-08-10T08:01:00Z",
            artifacts=(artifact,),
        )

    assert target.read_text(encoding="utf-8") == "existing\n"


def test_stage_rejects_artifact_outside_registered_execution_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _setup(monkeypatch, tmp_path)
    artifact = tmp_path / "other" / "raw.bin"
    artifact.parent.mkdir()
    artifact.write_bytes(b"raw")

    with pytest.raises(ValueError, match="must be below"):
        builder.stage_source_panel_manifest(
            tmp_path,
            tmp_path,
            started_at_utc="2026-08-10T08:00:00Z",
            ended_at_utc="2026-08-10T08:01:00Z",
            artifacts=(artifact,),
        )


def test_stage_rejects_duplicate_artifact_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, _, _, artifact_root = _setup(monkeypatch, tmp_path)
    artifact = artifact_root / "raw.bin"
    artifact.write_bytes(b"raw")

    with pytest.raises(ValueError, match="duplicate path"):
        builder.stage_source_panel_manifest(
            tmp_path,
            tmp_path,
            started_at_utc="2026-08-10T08:00:00Z",
            ended_at_utc="2026-08-10T08:01:00Z",
            artifacts=(artifact, artifact.relative_to(tmp_path)),
        )


def test_stage_rolls_back_when_status_changes_during_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, execution, _, artifact_root = _setup(monkeypatch, tmp_path)
    artifact = artifact_root / "raw.bin"
    artifact.write_bytes(b"raw")
    calls = 0

    def changing_status(*args, **kwargs):
        nonlocal calls
        del args, kwargs
        calls += 1
        return {
            "valid": True,
            "complete": False,
            "evidence_sha256": "c" * 64,
            "status_sha256": ("d" if calls == 1 else "e") * 64,
            "next_execution": _registered_values()[1],
        }

    monkeypatch.setattr(builder, "build_source_panel_status", changing_status)

    with pytest.raises(ValueError, match="status changed while staging"):
        builder.stage_source_panel_manifest(
            tmp_path,
            tmp_path,
            started_at_utc="2026-08-10T08:00:00Z",
            ended_at_utc="2026-08-10T08:01:00Z",
            artifacts=(artifact,),
        )

    target = tmp_path / "staging" / f"{execution['execution_id']}.json"
    assert not target.exists()


def test_stage_rolls_back_when_artifact_changes_during_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, execution, _, artifact_root = _setup(monkeypatch, tmp_path)
    artifact = artifact_root / "raw.bin"
    artifact.write_bytes(b"raw")
    validate = builder._validate_source_execution_manifest

    def validate_then_mutate(*args, **kwargs) -> None:
        validate(*args, **kwargs)
        artifact.write_bytes(b"changed")

    monkeypatch.setattr(
        builder,
        "_validate_source_execution_manifest",
        validate_then_mutate,
    )

    with pytest.raises(ValueError, match="artifacts changed while staging"):
        builder.stage_source_panel_manifest(
            tmp_path,
            tmp_path,
            started_at_utc="2026-08-10T08:00:00Z",
            ended_at_utc="2026-08-10T08:01:00Z",
            artifacts=(artifact,),
        )

    target = tmp_path / "staging" / f"{execution['execution_id']}.json"
    assert not target.exists()


def test_stage_cli_dispatches_artifacts_and_timestamps(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def stage(repository_root, dataset_root, **kwargs):
        captured.update(
            repository_root=repository_root,
            dataset_root=dataset_root,
            **kwargs,
        )
        return {
            "valid": True,
            "complete": True,
            "passed": True,
            "published": False,
        }

    monkeypatch.setattr(readiness_cli, "stage_source_panel_manifest", stage)
    code = readiness_cli.main(
        [
            "source-panel-stage",
            "/repo",
            "/dataset",
            "--started-at-utc",
            "2026-08-10T08:00:00Z",
            "--ended-at-utc",
            "2026-08-10T08:01:00Z",
            "--artifact",
            "first.bin",
            "--artifact",
            "second.bin",
        ]
    )

    assert code == 0
    assert captured == {
        "repository_root": "/repo",
        "dataset_root": "/dataset",
        "started_at_utc": "2026-08-10T08:00:00Z",
        "ended_at_utc": "2026-08-10T08:01:00Z",
        "artifacts": ["first.bin", "second.bin"],
    }
    assert json.loads(capsys.readouterr().out)["published"] is False
