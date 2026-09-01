"""Read-only staging of authorized PokeFlex development robot records."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from causal4d_public._pokeflex_realized_load_common import (
    PokeFlexRealizedLoadSourceConfig,
    _canonical_bytes,
    _require,
    validate_source_qa_binding,
)


POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION = 1
POKEFLEX_ROBOT_STAGE_KIND = "PublicPokeFlexRobotRecordStage"


def _bytes_sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def robot_stage_manifest_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("stage_manifest_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _source_qa_robot_hashes(
    source_qa: Mapping[str, Any],
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, str]:
    validate_source_qa_binding(source_qa, config)
    rows = source_qa.get("takes")
    _require(isinstance(rows, list), "source QA take inventory is missing")
    expected = set(config.expected_development_take_ids)
    hashes: dict[str, str] = {}
    for raw in rows:
        _require(isinstance(raw, Mapping), "source QA take row is invalid")
        take_id = str(raw.get("take_id", ""))
        if take_id not in expected:
            continue
        digest = str(raw.get("robot_sha256", ""))
        _require(
            len(digest) == 64 and all(value in "0123456789abcdef" for value in digest),
            f"source QA robot digest is invalid for {take_id}",
        )
        _require(take_id not in hashes, f"source QA take repeats: {take_id}")
        hashes[take_id] = digest
    _require(set(hashes) == expected, "source QA robot digest roster changed")
    return hashes


def _inside_root(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    _require(
        resolved == root or root in resolved.parents,
        f"source path escapes the dataset root: {path}",
    )
    return resolved


def _direct_robot_candidates(
    root: Path,
    object_id: str,
    take_id: str,
) -> tuple[Path, ...]:
    candidates = (
        root / take_id / "robot_data.json",
        root / object_id / take_id / "robot_data.json",
    )
    unique: dict[str, Path] = {}
    for candidate in candidates:
        resolved = _inside_root(candidate, root)
        if resolved.is_file() and not candidate.is_symlink():
            unique[str(resolved)] = resolved
    return tuple(unique[key] for key in sorted(unique))


def _archive_candidates(root: Path, take_id: str) -> tuple[Path, ...]:
    unique: dict[str, Path] = {}
    patterns = (f"*{take_id}*.zip", f"*{take_id}*.ZIP")
    for pattern in patterns:
        for candidate in root.rglob(pattern):
            if candidate.is_symlink() or not candidate.is_file():
                continue
            resolved = _inside_root(candidate, root)
            unique[str(resolved)] = resolved
    return tuple(unique[key] for key in sorted(unique))


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    _require(not path.is_absolute(), "archive member is absolute")
    _require(".." not in path.parts, "archive member escapes its root")
    return path


def _robot_member_candidates(
    archive: zipfile.ZipFile,
    take_id: str,
) -> tuple[zipfile.ZipInfo, ...]:
    infos = archive.infolist()
    names = [info.filename for info in infos]
    _require(len(names) == len(set(names)), "archive contains duplicate members")
    selected = []
    for info in infos:
        path = _safe_member_path(info.filename)
        if info.is_dir() or path.name != "robot_data.json":
            continue
        if take_id in path.parts or len(path.parts) == 1:
            selected.append(info)
    return tuple(selected)


def _read_verified_direct(
    candidates: Sequence[Path],
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]] | None:
    failures = []
    for candidate in candidates:
        try:
            content = candidate.read_bytes()
        except OSError as error:
            failures.append({"path": str(candidate), "error": type(error).__name__})
            continue
        observed = _bytes_sha256(content)
        _require(
            observed == expected_sha256,
            f"readable robot record differs from source QA: {candidate}",
        )
        return content, {
            "source_kind": "verified-extracted-file",
            "source_path": str(candidate),
            "archive_member": None,
            "candidate_failures": failures,
        }
    return None


def _read_verified_archive(
    candidates: Sequence[Path],
    take_id: str,
    expected_sha256: str,
) -> tuple[bytes, dict[str, Any]]:
    matches: list[tuple[bytes, Path, str]] = []
    failures = []
    for candidate in candidates:
        try:
            with zipfile.ZipFile(candidate) as archive:
                members = _robot_member_candidates(archive, take_id)
                if len(members) != 1:
                    failures.append(
                        {
                            "path": str(candidate),
                            "error": "robot-member-count",
                            "observed": len(members),
                        }
                    )
                    continue
                member = members[0]
                content = archive.read(member)
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
            failures.append({"path": str(candidate), "error": type(error).__name__})
            continue
        observed = _bytes_sha256(content)
        if observed == expected_sha256:
            matches.append((content, candidate, member.filename))
        else:
            failures.append(
                {
                    "path": str(candidate),
                    "error": "source-qa-digest-mismatch",
                    "observed_sha256": observed,
                }
            )
    _require(matches, f"no verified readable robot record found for {take_id}")
    matches.sort(key=lambda value: (str(value[1]), value[2]))
    content, archive_path, member_name = matches[0]
    return content, {
        "source_kind": "verified-zip-member",
        "source_path": str(archive_path),
        "archive_member": member_name,
        "verified_match_count": len(matches),
        "candidate_failures": failures,
    }


def stage_pokeflex_development_robot_records(
    dataset_root: str | Path,
    source_qa: Mapping[str, Any],
    destination_root: str | Path,
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, Any]:
    """Stage only source-QA-authorized robot logs into an isolated directory."""

    root = Path(dataset_root).resolve()
    _require(root.is_dir(), f"missing PokeFlex dataset root: {root}")
    destination = Path(destination_root).resolve()
    _require(not destination.exists(), "robot-record stage destination exists")
    destination.mkdir(parents=True)
    expected_hashes = _source_qa_robot_hashes(source_qa, config)

    records = []
    for take_id in config.expected_development_take_ids:
        expected_sha256 = expected_hashes[take_id]
        direct = _read_verified_direct(
            _direct_robot_candidates(root, config.expected_object_id, take_id),
            expected_sha256,
        )
        if direct is None:
            content, provenance = _read_verified_archive(
                _archive_candidates(root, take_id),
                take_id,
                expected_sha256,
            )
        else:
            content, provenance = direct
        target = destination / config.expected_object_id / take_id / "robot_data.json"
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_bytes(content)
        os.chmod(target, 0o600)
        records.append(
            {
                "take_id": take_id,
                "robot_sha256": expected_sha256,
                "byte_count": len(content),
                "staged_path": target.relative_to(destination).as_posix(),
                **provenance,
            }
        )

    manifest: dict[str, Any] = {
        "artifact_kind": POKEFLEX_ROBOT_STAGE_KIND,
        "schema_version": POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION,
        "source_qa_result_sha256": source_qa["result_sha256"],
        "object_id": config.expected_object_id,
        "development_take_ids": list(config.expected_development_take_ids),
        "forbidden_take_ids": list(config.forbidden_take_ids),
        "dataset_root": str(root),
        "staging_root": str(destination),
        "records": records,
        "information_boundary": {
            "development_robot_records_only": True,
            "development_meshes_read": False,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "dataset_modified": False,
        },
    }
    manifest["stage_manifest_sha256"] = robot_stage_manifest_sha256(manifest)
    manifest_path = destination / "stage_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def validate_pokeflex_robot_stage(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("artifact_kind") == POKEFLEX_ROBOT_STAGE_KIND,
        "unexpected robot-stage kind",
    )
    _require(
        payload.get("schema_version") == POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION,
        "unsupported robot-stage schema",
    )
    _require(
        payload.get("stage_manifest_sha256") == robot_stage_manifest_sha256(payload),
        "robot-stage checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("development_robot_records_only") is True, "stage widened")
    _require(boundary.get("development_meshes_read") is False, "stage read meshes")
    _require(
        boundary.get("calibration_take_data_read") is False,
        "stage opened calibration",
    )
    _require(boundary.get("target_take_data_read") is False, "stage opened target")
    _require(boundary.get("dataset_modified") is False, "stage modified source")
    return {
        "passed": True,
        "stage_manifest_sha256": payload["stage_manifest_sha256"],
        "record_count": len(payload["records"]),
    }


__all__ = [
    "POKEFLEX_ROBOT_STAGE_KIND",
    "POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION",
    "robot_stage_manifest_sha256",
    "stage_pokeflex_development_robot_records",
    "validate_pokeflex_robot_stage",
]
