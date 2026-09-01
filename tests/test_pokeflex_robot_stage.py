from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from causal4d_public.pokeflex_realized_load import (
    PokeFlexRealizedLoadSourceConfig,
)
from causal4d_public.pokeflex_robot_stage import (
    stage_pokeflex_development_robot_records,
    validate_pokeflex_robot_stage,
)


def _robot_bytes(take_id: str) -> bytes:
    payload = [
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
    ]
    return json.dumps(payload, sort_keys=True).encode("utf-8")


def _source_qa(
    config: PokeFlexRealizedLoadSourceConfig,
    content_by_take: dict[str, bytes],
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
                "robot_sha256": hashlib.sha256(content_by_take[take_id]).hexdigest(),
            }
            for take_id in config.expected_development_take_ids
        ],
    }


def _write_archives(
    root: Path,
    config: PokeFlexRealizedLoadSourceConfig,
    content_by_take: dict[str, bytes],
) -> dict[str, bytes]:
    archive_bytes = {}
    archive_root = root / "poking" / config.expected_object_id
    archive_root.mkdir(parents=True)
    for take_id in config.expected_development_take_ids:
        archive_path = archive_root / f"{take_id}.zip"
        with zipfile.ZipFile(archive_path, mode="w") as archive:
            archive.writestr(
                f"release/{take_id}/robot_data.json",
                content_by_take[take_id],
            )
            archive.writestr(
                f"release/{take_id}/meshes/mesh-f00001.obj",
                b"v 0 0 0\n",
            )
        archive_bytes[take_id] = archive_path.read_bytes()
    return archive_bytes


def test_stages_only_verified_development_robot_members(tmp_path: Path) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    archive_bytes = _write_archives(tmp_path, config, content)
    for take_id in config.forbidden_take_ids:
        (tmp_path / f"{take_id}.zip").write_bytes(b"forbidden invalid archive")

    destination = tmp_path / "stage"
    result = stage_pokeflex_development_robot_records(
        tmp_path,
        _source_qa(config, content),
        destination,
        config,
    )

    assert validate_pokeflex_robot_stage(result)["passed"] is True
    assert [row["take_id"] for row in result["records"]] == list(
        config.expected_development_take_ids
    )
    assert all(row["source_kind"] == "verified-zip-member" for row in result["records"])
    assert result["carrier_index"]["archive_count"] == 7
    boundary = result["information_boundary"]
    assert boundary["archive_central_directory_metadata_read"] is True
    assert boundary["nondevelopment_member_payloads_read"] is False
    assert boundary["calibration_take_data_read"] is False
    assert boundary["target_take_data_read"] is False
    assert boundary["dataset_modified"] is False
    for take_id in config.expected_development_take_ids:
        staged = destination / config.expected_object_id / take_id / "robot_data.json"
        assert staged.read_bytes() == content[take_id]
        archive = tmp_path / "poking" / config.expected_object_id / f"{take_id}.zip"
        assert archive.read_bytes() == archive_bytes[take_id]


def test_archive_index_does_not_depend_on_archive_filename(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    archive_path = tmp_path / "downloads" / "release-part-042.zip"
    archive_path.parent.mkdir()
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for take_id, payload in content.items():
            archive.writestr(f"poking/{take_id}/robot_data.json", payload)
        for take_id in config.forbidden_take_ids:
            archive.writestr(
                f"poking/{take_id}/robot_data.json",
                b"FORBIDDEN PAYLOAD MUST NOT BE READ",
            )

    original_read = zipfile.ZipFile.read
    read_names: list[str] = []

    def tracked_read(archive, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        read_names.append(filename)
        assert not any(take_id in filename for take_id in config.forbidden_take_ids)
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", tracked_read)
    result = stage_pokeflex_development_robot_records(
        tmp_path,
        _source_qa(config, content),
        tmp_path / "stage",
        config,
    )

    assert set(read_names) == {
        f"poking/{take_id}/robot_data.json"
        for take_id in config.expected_development_take_ids
    }
    assert all(
        Path(record["source_path"]).name == "release-part-042.zip"
        for record in result["records"]
    )
    assert result["carrier_index"]["nondevelopment_member_payloads_read"] is False


def test_prefers_a_readable_source_qa_bound_extracted_file(tmp_path: Path) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    for take_id, payload in content.items():
        path = tmp_path / take_id / "robot_data.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(payload)

    result = stage_pokeflex_development_robot_records(
        tmp_path,
        _source_qa(config, content),
        tmp_path / "stage",
        config,
    )

    assert all(
        row["source_kind"] == "verified-extracted-file" for row in result["records"]
    )
    assert result["carrier_index"]["archive_scan_required"] is False


def test_rejects_a_robot_member_that_differs_from_source_qa(tmp_path: Path) -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    content = {
        take_id: _robot_bytes(take_id)
        for take_id in config.expected_development_take_ids
    }
    _write_archives(tmp_path, config, content)
    first = config.expected_development_take_ids[0]
    archive_path = tmp_path / "poking" / config.expected_object_id / f"{first}.zip"
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        archive.writestr(f"{first}/robot_data.json", b"changed")

    with pytest.raises(ValueError, match="no source-QA-verified archive member"):
        stage_pokeflex_development_robot_records(
            tmp_path,
            _source_qa(config, content),
            tmp_path / "stage",
            config,
        )
