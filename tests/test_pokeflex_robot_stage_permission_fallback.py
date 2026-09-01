from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import causal4d_public.pokeflex_robot_stage as robot_stage
from causal4d_public.pokeflex_realized_load import (
    PokeFlexRealizedLoadSourceConfig,
)


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


def test_permission_denied_direct_files_fall_back_to_verified_archives(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    for take_id, payload in content.items():
        with zipfile.ZipFile(tmp_path / f"{take_id}.zip", "w") as archive:
            archive.writestr(f"{take_id}/robot_data.json", payload)

    def deny_direct_read(path: Path) -> bytes:
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(robot_stage, "_read_bytes_no_follow", deny_direct_read)
    result = robot_stage.stage_pokeflex_development_robot_records(
        tmp_path,
        _source_qa(config, content),
        tmp_path / "stage",
        config,
    )

    assert [record["source_kind"] for record in result["records"]] == [
        "verified-zip-member"
    ] * len(config.expected_development_take_ids)
    assert all(
        record["candidate_failures"][0]["error"] == "PermissionError"
        for record in result["records"]
    )
    assert result["information_boundary"] == {
        "development_robot_records_only": True,
        "development_meshes_read": False,
        "calibration_take_data_read": False,
        "target_take_data_read": False,
        "dataset_modified": False,
    }
