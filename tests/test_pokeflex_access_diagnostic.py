from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Sequence

from causal4d_public import pokeflex_access_diagnostic as access
from causal4d_public.cli import pokeflex_realized_load_source as cli
from causal4d_public.pokeflex_realized_load import (
    PokeFlexRealizedLoadSourceConfig,
)


def _deterministic_probe(command: Sequence[str]) -> dict[str, Any]:
    return {
        "command": list(command),
        "available": True,
        "returncode": 1 if command[0] in {"sudo", "ssh"} else 0,
        "stdout": "probe-output",
        "stderr": "probe-error" if command[0] in {"sudo", "ssh"} else "",
    }


def test_access_diagnostic_reads_metadata_only(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    archive_root = tmp_path / "poking"
    object_root = archive_root / config.expected_object_id
    object_root.mkdir(parents=True)
    sentinel_by_path = {}
    for take_id in (*config.expected_development_take_ids, *config.forbidden_take_ids):
        path = object_root / f"{take_id}.zip"
        sentinel = f"opaque-{take_id}".encode()
        path.write_bytes(sentinel)
        sentinel_by_path[path] = sentinel

    monkeypatch.setattr(access, "_run_probe", _deterministic_probe)
    diagnostic = access.build_pokeflex_access_diagnostic(
        archive_root,
        config,
        PermissionError("sudo: a password is required"),
    )

    assert diagnostic["diagnostic_sha256"] == access.access_diagnostic_sha256(
        diagnostic
    )
    assert diagnostic["status"] == (
        "source-evaluation-blocked-before-payload-access"
    )
    assert diagnostic["information_boundary"] == {
        "path_metadata_only": True,
        "archive_central_directories_read": False,
        "development_member_payloads_read": False,
        "calibration_take_data_read": False,
        "target_take_data_read": False,
        "dataset_modified": False,
        "new_physical_data_collected": False,
    }
    development_paths = diagnostic["development_archive_paths"]
    assert len(development_paths) == len(config.expected_development_take_ids)
    assert all(take_id in " ".join(development_paths) for take_id in config.expected_development_take_ids)
    assert all(take_id not in " ".join(development_paths) for take_id in config.forbidden_take_ids)
    metadata_text = json.dumps(diagnostic["path_metadata"], sort_keys=True)
    assert all(take_id not in metadata_text for take_id in config.forbidden_take_ids)
    for path, sentinel in sentinel_by_path.items():
        assert path.read_bytes() == sentinel


def test_cli_retains_access_failure_as_non_scientific_result(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    source_qa = tmp_path / "source_qa.json"
    source_qa.write_text("{}\n", encoding="utf-8")
    output = tmp_path / "output"
    policy = tmp_path / "policy.json"
    policy.write_text("{}\n", encoding="utf-8")

    def blocked_stage(*args, **kwargs):
        raise PermissionError("sudo: a password is required")

    diagnostic = {
        "artifact_kind": access.POKEFLEX_ACCESS_DIAGNOSTIC_KIND,
        "schema_version": access.POKEFLEX_ACCESS_DIAGNOSTIC_SCHEMA_VERSION,
        "diagnostic_sha256": "a" * 64,
    }

    monkeypatch.setattr(cli, "load_realized_load_policy", lambda path: config)
    monkeypatch.setattr(
        cli,
        "stage_pokeflex_development_robot_records_with_owner_fallback",
        blocked_stage,
    )
    monkeypatch.setattr(
        cli,
        "build_pokeflex_access_diagnostic",
        lambda archive_root, supplied_config, error: diagnostic,
    )

    def write_diagnostic(path: str | Path, payload) -> Path:
        result = Path(path)
        result.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return result

    monkeypatch.setattr(cli, "write_pokeflex_access_diagnostic", write_diagnostic)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pokeflex-realized-load-source",
            str(tmp_path),
            str(source_qa),
            str(output),
            "--policy",
            str(policy),
        ],
    )

    assert cli.main() == 0
    reported = json.loads(capsys.readouterr().out)
    assert reported == {
        "passed": False,
        "technical_status": "source-evaluation-blocked-before-payload-access",
        "source_gate_executed": False,
        "source_backend_admitted": False,
        "diagnostic_file": "technical_access_boundary.json",
        "diagnostic_sha256": "a" * 64,
        "development_member_payloads_read": False,
        "calibration_take_data_read": False,
        "target_take_data_read": False,
    }
    assert (output / "technical_access_boundary.json").is_file()
