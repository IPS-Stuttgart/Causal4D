"""Owner-user fallback for exact PokeFlex development robot members."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from causal4d_public._pokeflex_realized_load_common import (
    PokeFlexRealizedLoadSourceConfig,
    _require,
)
from causal4d_public.pokeflex_robot_stage import (
    POKEFLEX_ROBOT_STAGE_KIND,
    POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION,
    _source_qa_robot_hashes,
    robot_stage_manifest_sha256,
    stage_pokeflex_development_robot_records,
)


POKEFLEX_ARCHIVE_OWNER_USER = "florianpfaff"
_OWNER_PYTHON = "/usr/bin/python3"
_MAX_ROBOT_RECORD_BYTES = 4 * 1024 * 1024
_DELEGATED_READ_TIMEOUT_SECONDS = 120
_DELEGATED_MEMBER_READER = """
import sys
import zipfile

archive_path, member_name = sys.argv[1:3]
with zipfile.ZipFile(archive_path) as archive:
    content = archive.read(member_name)
sys.stdout.buffer.write(content)
""".strip()


def _safe_component(value: str, name: str) -> str:
    path = PurePosixPath(value)
    _require(
        len(path.parts) == 1 and path.parts[0] not in {"", ".", ".."},
        f"{name} is not one path component",
    )
    return value


def _absolute_lexical_path(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _delegated_member_read(
    archive_path: Path,
    member_name: str,
    *,
    owner_user: str,
) -> tuple[bytes, dict[str, Any]]:
    sudo_path = shutil.which("sudo")
    _require(sudo_path is not None, "sudo is unavailable for owner-user staging")
    _require(
        Path(_OWNER_PYTHON).is_file(),
        f"owner-user helper Python is missing: {_OWNER_PYTHON}",
    )
    command = [
        sudo_path,
        "-n",
        "-u",
        owner_user,
        "--",
        _OWNER_PYTHON,
        "-I",
        "-c",
        _DELEGATED_MEMBER_READER,
        str(archive_path),
        member_name,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_DELEGATED_READ_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as error:
        raise PermissionError(
            f"owner-user archive read timed out: {archive_path}"
        ) from error
    stderr = completed.stderr.decode("utf-8", errors="replace").strip()
    if completed.returncode != 0:
        detail = stderr[-1000:] if stderr else "no stderr"
        raise PermissionError(
            "owner-user archive read failed for "
            f"{archive_path}::{member_name}: returncode={completed.returncode}; "
            f"stderr={detail}"
        )
    content = bytes(completed.stdout)
    _require(content, f"owner-user archive member is empty: {member_name}")
    _require(
        len(content) <= _MAX_ROBOT_RECORD_BYTES,
        f"owner-user robot member exceeds byte bound: {member_name}",
    )
    return content, {
        "source_kind": "verified-owner-delegated-zip-member",
        "source_path": str(archive_path),
        "archive_member": member_name,
        "delegated_user": owner_user,
        "delegated_program": _OWNER_PYTHON,
        "delegated_stderr": stderr or None,
    }


def _stage_with_owner_user(
    archive_root: str | Path,
    source_qa: Mapping[str, Any],
    destination_root: str | Path,
    config: PokeFlexRealizedLoadSourceConfig,
    *,
    owner_user: str,
    direct_error: str,
) -> dict[str, Any]:
    root = _absolute_lexical_path(archive_root)
    destination = Path(destination_root).resolve()
    _require(not destination.exists(), "robot-record stage destination exists")
    destination.mkdir(parents=True)
    expected_hashes = _source_qa_robot_hashes(source_qa, config)
    object_id = _safe_component(config.expected_object_id, "object id")
    owner = _safe_component(owner_user, "owner user")

    records = []
    for take_id in config.expected_development_take_ids:
        take = _safe_component(take_id, "take id")
        archive_path = root / object_id / f"{take}.zip"
        member_name = f"{take}/robot_data.json"
        content, provenance = _delegated_member_read(
            archive_path,
            member_name,
            owner_user=owner,
        )
        observed_sha256 = hashlib.sha256(content).hexdigest()
        _require(
            observed_sha256 == expected_hashes[take_id],
            f"owner-user robot member differs from source QA: {take_id}",
        )
        json.loads(content.decode("utf-8"))
        target = destination / object_id / take / "robot_data.json"
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_bytes(content)
        os.chmod(target, 0o600)
        records.append(
            {
                "take_id": take_id,
                "robot_sha256": observed_sha256,
                "byte_count": len(content),
                "staged_path": target.relative_to(destination).as_posix(),
                **provenance,
            }
        )

    manifest: dict[str, Any] = {
        "artifact_kind": POKEFLEX_ROBOT_STAGE_KIND,
        "schema_version": POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION,
        "source_qa_result_sha256": source_qa["result_sha256"],
        "object_id": object_id,
        "development_take_ids": list(config.expected_development_take_ids),
        "forbidden_take_ids": list(config.forbidden_take_ids),
        "dataset_root": str(root),
        "staging_root": str(destination),
        "records": records,
        "carrier_index": {
            "strategy": "exact-owner-delegated-archive-members",
            "archive_count": len(records),
            "archive_central_directories_read_by_caller": False,
            "nondevelopment_member_payloads_read": False,
            "direct_access_error": direct_error,
        },
        "information_boundary": {
            "development_robot_records_only": True,
            "development_meshes_read": False,
            "archive_central_directory_metadata_read": False,
            "nondevelopment_member_payloads_read": False,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "owner_delegated_read": True,
            "dataset_modified": False,
        },
    }
    manifest["stage_manifest_sha256"] = robot_stage_manifest_sha256(manifest)
    (destination / "stage_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def stage_pokeflex_development_robot_records_with_owner_fallback(
    archive_root: str | Path,
    source_qa: Mapping[str, Any],
    destination_root: str | Path,
    config: PokeFlexRealizedLoadSourceConfig,
    *,
    owner_user: str = POKEFLEX_ARCHIVE_OWNER_USER,
) -> dict[str, Any]:
    """Use normal read-only staging, then exact owner-user reads on ACL denial."""

    root = _absolute_lexical_path(archive_root)
    if os.access(root, os.R_OK | os.X_OK):
        try:
            return stage_pokeflex_development_robot_records(
                root,
                source_qa,
                destination_root,
                config,
            )
        except PermissionError as error:
            direct_error = f"{type(error).__name__}: {error}"
    else:
        direct_error = f"current user cannot list and traverse {root}"
    return _stage_with_owner_user(
        root,
        source_qa,
        destination_root,
        config,
        owner_user=owner_user,
        direct_error=direct_error,
    )


__all__ = [
    "POKEFLEX_ARCHIVE_OWNER_USER",
    "stage_pokeflex_development_robot_records_with_owner_fallback",
]
