"""Metadata-only diagnostics for blocked PokeFlex archive access."""

from __future__ import annotations

import grp
import hashlib
import json
import os
import pwd
import shutil
import stat
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from causal4d_public._pokeflex_realized_load_common import (
    PokeFlexRealizedLoadSourceConfig,
    _canonical_bytes,
)
from causal4d_public.pokeflex_owner_stage import POKEFLEX_ARCHIVE_OWNER_USER


POKEFLEX_ACCESS_DIAGNOSTIC_KIND = "PublicPokeFlexAccessBoundaryDiagnostic"
POKEFLEX_ACCESS_DIAGNOSTIC_SCHEMA_VERSION = 1
_PROBE_TIMEOUT_SECONDS = 10
_MAX_TEXT_LENGTH = 4000


def _bounded_text(value: bytes | str) -> str:
    text = (
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
    ).strip()
    if len(text) > _MAX_TEXT_LENGTH:
        return text[:_MAX_TEXT_LENGTH] + "...[truncated]"
    return text


def _group_name(group_id: int) -> str | None:
    try:
        return grp.getgrgid(group_id).gr_name
    except KeyError:
        return None


def _user_record(user_name: str) -> dict[str, Any]:
    try:
        record = pwd.getpwnam(user_name)
    except KeyError:
        return {"exists": False, "name": user_name}
    return {
        "exists": True,
        "name": user_name,
        "uid": record.pw_uid,
        "gid": record.pw_gid,
        "group_name": _group_name(record.pw_gid),
        "home": record.pw_dir,
        "shell": record.pw_shell,
    }


def _identity() -> dict[str, Any]:
    uid = os.geteuid()
    gid = os.getegid()
    try:
        user_name = pwd.getpwuid(uid).pw_name
    except KeyError:
        user_name = None
    groups = sorted(set(os.getgroups()))
    return {
        "effective_uid": uid,
        "effective_gid": gid,
        "effective_user": user_name,
        "effective_group": _group_name(gid),
        "supplementary_groups": [
            {"gid": value, "name": _group_name(value)} for value in groups
        ],
    }


def _path_metadata(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "access_exists": os.access(path, os.F_OK),
        "access_read": os.access(path, os.R_OK),
        "access_execute": os.access(path, os.X_OK),
    }
    try:
        information = os.lstat(path)
    except OSError as error:
        record["lstat_error"] = f"{type(error).__name__}: {error}"
        return record
    record.update(
        {
            "mode_octal": oct(stat.S_IMODE(information.st_mode)),
            "mode_symbolic": stat.filemode(information.st_mode),
            "uid": information.st_uid,
            "gid": information.st_gid,
            "owner_name": (
                pwd.getpwuid(information.st_uid).pw_name
                if information.st_uid in {entry.pw_uid for entry in pwd.getpwall()}
                else None
            ),
            "group_name": _group_name(information.st_gid),
            "size_bytes": information.st_size,
            "is_symlink": stat.S_ISLNK(information.st_mode),
            "is_directory": stat.S_ISDIR(information.st_mode),
            "is_regular_file": stat.S_ISREG(information.st_mode),
        }
    )
    return record


def _path_chain(path: Path) -> list[Path]:
    chain = []
    current = path
    while True:
        chain.append(current)
        if current.parent == current:
            break
        current = current.parent
    return list(reversed(chain))


def _run_probe(command: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {
            "command": list(command),
            "available": False,
            "returncode": None,
            "stdout": "",
            "stderr": "executable unavailable",
        }
    resolved = [executable, *command[1:]]
    try:
        completed = subprocess.run(
            resolved,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=_PROBE_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "command": list(command),
            "available": True,
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(error).__name__}: {error}",
        }
    return {
        "command": list(command),
        "available": True,
        "returncode": completed.returncode,
        "stdout": _bounded_text(completed.stdout),
        "stderr": _bounded_text(completed.stderr),
    }


def _docker_image_probe(image: str) -> dict[str, Any]:
    result = _run_probe(["docker", "image", "inspect", image])
    return {
        "image": image,
        "docker_available": result["available"],
        "present_without_pull": result["returncode"] == 0,
        "returncode": result["returncode"],
        "stderr": result["stderr"],
    }


def access_diagnostic_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("diagnostic_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def build_pokeflex_access_diagnostic(
    archive_root: str | Path,
    config: PokeFlexRealizedLoadSourceConfig,
    error: BaseException,
) -> dict[str, Any]:
    """Describe permissions and delegated-reader availability without payload reads."""

    root = Path(os.path.abspath(os.fspath(archive_root)))
    object_root = root / config.expected_object_id
    development_archives = [
        object_root / f"{take_id}.zip"
        for take_id in config.expected_development_take_ids
    ]
    metadata_paths: dict[str, Path] = {}
    for path in _path_chain(object_root):
        metadata_paths[str(path)] = path
    for path in development_archives:
        metadata_paths[str(path)] = path

    probes = {
        "owner_identity": _run_probe(["id", POKEFLEX_ARCHIVE_OWNER_USER]),
        "noninteractive_owner_switch": _run_probe(
            ["sudo", "-n", "-u", POKEFLEX_ARCHIVE_OWNER_USER, "--", "true"]
        ),
        "owner_localhost_ssh": _run_probe(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=3",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "UserKnownHostsFile=/dev/null",
                f"{POKEFLEX_ARCHIVE_OWNER_USER}@localhost",
                "true",
            ]
        ),
        "mount": _run_probe(
            ["findmnt", "-no", "SOURCE,FSTYPE,OPTIONS", "--target", str(root)]
        ),
        "path_walk": _run_probe(["namei", "-om", str(development_archives[0])]),
        "acl": _run_probe(["getfacl", "-cp", str(development_archives[0])]),
        "docker_server": _run_probe(["docker", "version", "--format", "{{.Server.Version}}"]),
        "podman": _run_probe(["podman", "info", "--format", "json"]),
    }
    images = [
        _docker_image_probe(image)
        for image in (
            "python:3.12-slim",
            "python:3.11-slim",
            "alpine:3.20",
            "ubuntu:24.04",
        )
    ]
    payload: dict[str, Any] = {
        "artifact_kind": POKEFLEX_ACCESS_DIAGNOSTIC_KIND,
        "schema_version": POKEFLEX_ACCESS_DIAGNOSTIC_SCHEMA_VERSION,
        "status": "source-evaluation-blocked-before-payload-access",
        "error": f"{type(error).__name__}: {error}",
        "runner": {
            "runner_name": os.environ.get("RUNNER_NAME"),
            "runner_os": os.environ.get("RUNNER_OS"),
            "runner_arch": os.environ.get("RUNNER_ARCH"),
            "machine_name": os.uname().nodename,
        },
        "identity": _identity(),
        "archive_owner": _user_record(POKEFLEX_ARCHIVE_OWNER_USER),
        "archive_root": str(root),
        "object_root": str(object_root),
        "development_archive_paths": [str(path) for path in development_archives],
        "path_metadata": [
            _path_metadata(metadata_paths[key]) for key in sorted(metadata_paths)
        ],
        "delegated_reader_probes": probes,
        "allowlisted_local_container_images": images,
        "information_boundary": {
            "path_metadata_only": True,
            "archive_central_directories_read": False,
            "development_member_payloads_read": False,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "dataset_modified": False,
            "new_physical_data_collected": False,
        },
        "interpretation": (
            "This is a technical access-boundary result, not a scientific source-gate "
            "outcome. It may authorize no calibration or target access."
        ),
    }
    payload["diagnostic_sha256"] = access_diagnostic_sha256(payload)
    return payload


def write_pokeflex_access_diagnostic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output


__all__ = [
    "POKEFLEX_ACCESS_DIAGNOSTIC_KIND",
    "POKEFLEX_ACCESS_DIAGNOSTIC_SCHEMA_VERSION",
    "access_diagnostic_sha256",
    "build_pokeflex_access_diagnostic",
    "write_pokeflex_access_diagnostic",
]
