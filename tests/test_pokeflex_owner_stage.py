from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

import causal4d_public.pokeflex_owner_stage as owner_stage
from causal4d_public.pokeflex_realized_load import (
    PokeFlexRealizedLoadSourceConfig,
)
from causal4d_public.pokeflex_robot_stage import validate_pokeflex_robot_stage


def _robot_bytes(take_id: str) -> bytes:
    return json.dumps(
        [
            {
                "frame": "00001",
                "forces": [0.0, 4.0, 0.0],
                "T_WT": [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                "take_id": take_id,
            }
        ],
        sort_keys=True,
    ).encode("utf-8")


def _source_qa(
    config: PokeFlexRealizedLoadSourceConfig,
    content: dict[str, bytes],
) -> dict[str, object]:
    return {
        "artifact_kind": "PublicPokeFlexSourceQa",
        "schema_version": 1,
        "result_sha256": config.expected_source_qa_result_sha256,
        "source_qa_passed": True,
        "object_id": config.expected_object_id,
        "information_boundary": {
            "opened_take_ids": list(config.expected_development_take_ids),
            "unopened_take_ids": list(config.forbidden_take_ids),
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        },
        "capability_gates": {"pose_wrench_contact_candidate_ready": True},
        "takes": [
            {
                "take_id": take_id,
                "robot_sha256": hashlib.sha256(content[take_id]).hexdigest(),
            }
            for take_id in config.expected_development_take_ids
        ],
    }


def test_owner_fallback_reads_only_exact_development_members(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    commands: list[list[str]] = []

    monkeypatch.setattr(owner_stage.os, "access", lambda *_: False)
    monkeypatch.setattr(owner_stage.shutil, "which", lambda name: f"/usr/bin/{name}")

    def fake_run(command, **kwargs):
        commands.append(list(command))
        member_name = command[-1]
        take_id = member_name.split("/", maxsplit=1)[0]
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=content[take_id],
            stderr=b"",
        )

    monkeypatch.setattr(owner_stage.subprocess, "run", fake_run)
    archive_root = tmp_path / "pokeflex" / "poking"
    destination = tmp_path / "stage"
    result = (
        owner_stage.stage_pokeflex_development_robot_records_with_owner_fallback(
            archive_root,
            _source_qa(config, content),
            destination,
            config,
        )
    )

    assert validate_pokeflex_robot_stage(result)["passed"] is True
    assert len(commands) == len(config.expected_development_take_ids)
    for command, take_id in zip(
        commands,
        config.expected_development_take_ids,
        strict=True,
    ):
        assert command[:5] == [
            "/usr/bin/sudo",
            "-n",
            "-u",
            owner_stage.POKEFLEX_ARCHIVE_OWNER_USER,
            "--",
        ]
        assert command[-2] == str(
            archive_root / config.expected_object_id / f"{take_id}.zip"
        )
        assert command[-1] == f"{take_id}/robot_data.json"
        assert not any(forbidden in " ".join(command) for forbidden in config.forbidden_take_ids)
        staged = destination / config.expected_object_id / take_id / "robot_data.json"
        assert staged.read_bytes() == content[take_id]
    assert result["information_boundary"]["owner_delegated_read"] is True
    assert result["information_boundary"]["dataset_modified"] is False
    assert result["carrier_index"]["nondevelopment_member_payloads_read"] is False


def test_owner_fallback_rejects_source_qa_hash_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    monkeypatch.setattr(owner_stage.os, "access", lambda *_: False)
    monkeypatch.setattr(owner_stage.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        owner_stage.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            0,
            stdout=b"[]",
            stderr=b"",
        ),
    )

    with pytest.raises(ValueError, match="differs from source QA"):
        owner_stage.stage_pokeflex_development_robot_records_with_owner_fallback(
            tmp_path / "poking",
            _source_qa(config, content),
            tmp_path / "stage",
            config,
        )


def test_owner_fallback_reports_noninteractive_sudo_denial(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    monkeypatch.setattr(owner_stage.os, "access", lambda *_: False)
    monkeypatch.setattr(owner_stage.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(
        owner_stage.subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            stdout=b"",
            stderr=b"sudo: a password is required",
        ),
    )

    with pytest.raises(PermissionError, match="password is required"):
        owner_stage.stage_pokeflex_development_robot_records_with_owner_fallback(
            tmp_path / "poking",
            _source_qa(config, content),
            tmp_path / "stage",
            config,
        )
