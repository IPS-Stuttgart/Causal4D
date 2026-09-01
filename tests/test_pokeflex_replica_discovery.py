from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from causal4d_public.pokeflex_replica_discovery import (
    DEVELOPMENT_TAKE_IDS,
    EXPECTED_OBJECT_ID,
    EXPECTED_SOURCE_QA_SHA256,
    FORBIDDEN_TAKE_IDS,
    discover_pokeflex_development_replica,
    replica_discovery_sha256,
    validate_pokeflex_replica_discovery,
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


def _source_qa(content_by_take: dict[str, bytes]) -> dict[str, object]:
    return {
        "artifact_kind": "PublicPokeFlexSourceQa",
        "schema_version": 1,
        "result_sha256": EXPECTED_SOURCE_QA_SHA256,
        "source_qa_passed": True,
        "object_id": EXPECTED_OBJECT_ID,
        "information_boundary": {
            "opened_take_ids": list(DEVELOPMENT_TAKE_IDS),
            "unopened_take_ids": list(FORBIDDEN_TAKE_IDS),
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        },
        "takes": [
            {
                "take_id": take_id,
                "robot_sha256": hashlib.sha256(content_by_take[take_id]).hexdigest(),
            }
            for take_id in DEVELOPMENT_TAKE_IDS
        ],
    }


def test_discovers_verified_extracted_and_archive_replicas(
    tmp_path: Path,
    monkeypatch,
) -> None:
    content = {take_id: _robot_bytes(take_id) for take_id in DEVELOPMENT_TAKE_IDS}
    source_root = tmp_path / "source"
    source_root.mkdir()
    for take_id in DEVELOPMENT_TAKE_IDS[:2]:
        path = source_root / EXPECTED_OBJECT_ID / take_id / "robot_data.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(content[take_id])

    archive_path = source_root / "opaque-pokeflex-bunny-stage.zip"
    forbidden_payloads = {
        take_id: f"forbidden-{take_id}".encode() for take_id in FORBIDDEN_TAKE_IDS
    }
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for take_id in DEVELOPMENT_TAKE_IDS[2:]:
            archive.writestr(
                f"stage/{take_id}/robot_data.json",
                content[take_id],
            )
        for take_id, payload in forbidden_payloads.items():
            archive.writestr(f"stage/{take_id}/robot_data.json", payload)
        archive.writestr("stage/unrelated/robot_data.json", b"unrelated")
        archive.writestr("stage/3dPrintedBunny_T4/meshes/mesh.obj", b"v 0 0 0\n")
    archive_before = archive_path.read_bytes()

    original_read = zipfile.ZipFile.read
    read_names: list[str] = []

    def tracked_read(archive, name, *args, **kwargs):
        filename = name.filename if isinstance(name, zipfile.ZipInfo) else str(name)
        read_names.append(filename)
        assert all(take_id not in filename for take_id in FORBIDDEN_TAKE_IDS)
        return original_read(archive, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "read", tracked_read)
    cache_root = tmp_path / "cache"
    result = discover_pokeflex_development_replica(
        source_qa=_source_qa(content),
        search_roots=(source_root,),
        cache_root=cache_root,
    )

    assert validate_pokeflex_replica_discovery(result) == {
        "passed": True,
        "complete": True,
        "cache_verified": True,
        "result_sha256": result["result_sha256"],
    }
    assert result["result_sha256"] == replica_discovery_sha256(result)
    assert result["complete"] is True
    assert result["cache_written"] is True
    assert result["found_take_ids"] == sorted(DEVELOPMENT_TAKE_IDS)
    assert result["missing_take_ids"] == []
    assert result["reads"]["nondevelopment_payload_read_count"] == 0
    assert set(read_names) == {
        f"stage/{take_id}/robot_data.json" for take_id in DEVELOPMENT_TAKE_IDS[2:]
    }
    for take_id in DEVELOPMENT_TAKE_IDS:
        cached = cache_root / EXPECTED_OBJECT_ID / take_id / "robot_data.json"
        assert cached.read_bytes() == content[take_id]
    assert archive_path.read_bytes() == archive_before


def test_incomplete_replica_does_not_create_partial_cache(tmp_path: Path) -> None:
    content = {take_id: _robot_bytes(take_id) for take_id in DEVELOPMENT_TAKE_IDS}
    source_root = tmp_path / "source"
    path = (
        source_root / EXPECTED_OBJECT_ID / DEVELOPMENT_TAKE_IDS[0] / "robot_data.json"
    )
    path.parent.mkdir(parents=True)
    path.write_bytes(content[DEVELOPMENT_TAKE_IDS[0]])
    cache_root = tmp_path / "cache"

    result = discover_pokeflex_development_replica(
        source_qa=_source_qa(content),
        search_roots=(source_root,),
        cache_root=cache_root,
    )

    assert result["complete"] is False
    assert result["cache_written"] is False
    assert result["cache_verified"] is False
    assert result["found_take_ids"] == [DEVELOPMENT_TAKE_IDS[0]]
    assert result["missing_take_ids"] == sorted(DEVELOPMENT_TAKE_IDS[1:])
    assert not cache_root.exists()
    assert validate_pokeflex_replica_discovery(result)["passed"] is True


def test_existing_verified_cache_is_reused_without_replacement(tmp_path: Path) -> None:
    content = {take_id: _robot_bytes(take_id) for take_id in DEVELOPMENT_TAKE_IDS}
    source_root = tmp_path / "source"
    archive_path = source_root / "pokeflex-bunny-development.zip"
    source_root.mkdir()
    with zipfile.ZipFile(archive_path, mode="w") as archive:
        for take_id in DEVELOPMENT_TAKE_IDS:
            archive.writestr(f"{take_id}/robot_data.json", content[take_id])
    cache_root = tmp_path / "cache"

    first = discover_pokeflex_development_replica(
        source_qa=_source_qa(content),
        search_roots=(source_root,),
        cache_root=cache_root,
    )
    manifest_before = (cache_root / "manifest.json").read_bytes()
    second = discover_pokeflex_development_replica(
        source_qa=_source_qa(content),
        search_roots=(source_root,),
        cache_root=cache_root,
    )

    assert first["cache_status"] == "cache-created"
    assert second["cache_status"] == "existing-cache-verified"
    assert second["cache_written"] is False
    assert second["cache_verified"] is True
    assert (cache_root / "manifest.json").read_bytes() == manifest_before
