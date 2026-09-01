"""Read-only staging of authorized PokeFlex development robot records."""

from __future__ import annotations

import hashlib
import json
import os
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO, Mapping, Sequence

from causal4d_public._pokeflex_realized_load_common import (
    PokeFlexRealizedLoadSourceConfig,
    _canonical_bytes,
    _require,
    validate_source_qa_binding,
)


POKEFLEX_ROBOT_STAGE_SCHEMA_VERSION = 1
POKEFLEX_ROBOT_STAGE_KIND = "PublicPokeFlexRobotRecordStage"
_MAX_ARCHIVE_DEPTH = 4
_MAX_ARCHIVE_COUNT = 512
_MAX_DIRECTORY_COUNT = 4096
_PRUNED_DIRECTORY_NAMES = frozenset(
    {
        "images",
        "kinect",
        "meshes",
        "realsense",
        "triangle_meshes",
        "volucam",
    }
)


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


def _safe_component(value: str, name: str) -> str:
    path = PurePosixPath(value)
    _require(
        len(path.parts) == 1 and path.parts[0] not in {"", ".", ".."},
        f"{name} is not one path component",
    )
    return value


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
    """Return trusted lexical paths without touching an unreadable source file."""

    object_component = _safe_component(object_id, "object id")
    take_component = _safe_component(take_id, "take id")
    return (
        root / take_component / "robot_data.json",
        root / object_component / take_component / "robot_data.json",
    )


def _relative_display(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _bounded_archive_paths(
    root: Path,
) -> tuple[tuple[Path, ...], tuple[dict[str, str], ...]]:
    """Enumerate ZIP carriers without descending into high-volume modality trees."""

    archives: dict[str, Path] = {}
    failures: list[dict[str, str]] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    directory_count = 0
    while stack:
        directory, depth = stack.pop()
        directory_count += 1
        _require(
            directory_count <= _MAX_DIRECTORY_COUNT,
            "PokeFlex archive directory scan exceeded its bound",
        )
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda value: value.name)
        except OSError as error:
            failures.append(
                {
                    "path": _relative_display(directory, root),
                    "error": type(error).__name__,
                }
            )
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                if entry.is_file(follow_symlinks=False):
                    if entry.name.lower().endswith(".zip"):
                        candidate = _inside_root(Path(entry.path), root)
                        archives[str(candidate)] = candidate
                        _require(
                            len(archives) <= _MAX_ARCHIVE_COUNT,
                            "PokeFlex archive count exceeded its bound",
                        )
                    continue
                if (
                    depth < _MAX_ARCHIVE_DEPTH
                    and entry.is_dir(follow_symlinks=False)
                    and entry.name.lower() not in _PRUNED_DIRECTORY_NAMES
                ):
                    stack.append((Path(entry.path), depth + 1))
            except OSError as error:
                failures.append(
                    {
                        "path": _relative_display(Path(entry.path), root),
                        "error": type(error).__name__,
                    }
                )
    ordered = tuple(archives[key] for key in sorted(archives))
    return ordered, tuple(failures)


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    _require(not path.is_absolute(), "archive member is absolute")
    _require(".." not in path.parts, "archive member escapes its root")
    return path


def _normalized_identifier(value: str) -> str:
    return "".join(character.lower() for character in value if character.isalnum())


def _member_take_id(
    member_path: PurePosixPath,
    archive_path: Path,
    expected_take_ids: Sequence[str],
) -> str | None:
    expected = {
        take_id: _normalized_identifier(take_id) for take_id in expected_take_ids
    }
    matches = set()
    for part in member_path.parts[:-1]:
        normalized = _normalized_identifier(part)
        for take_id, expected_normalized in expected.items():
            if normalized == expected_normalized or normalized.endswith(
                expected_normalized
            ):
                matches.add(take_id)
    if len(matches) == 1:
        return next(iter(matches))
    if matches:
        return None
    if len(member_path.parts) != 1:
        return None
    archive_normalized = _normalized_identifier(archive_path.stem)
    filename_matches = [
        take_id
        for take_id, expected_normalized in expected.items()
        if archive_normalized.endswith(expected_normalized)
    ]
    return filename_matches[0] if len(filename_matches) == 1 else None


def _open_binary_no_follow(path: Path) -> BinaryIO:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    return os.fdopen(descriptor, "rb")


def _read_bytes_no_follow(path: Path) -> bytes:
    with _open_binary_no_follow(path) as handle:
        return handle.read()


def _read_verified_direct(
    candidates: Sequence[Path],
    expected_sha256: str,
) -> tuple[tuple[bytes, dict[str, Any]] | None, tuple[dict[str, str], ...]]:
    failures = []
    for candidate in candidates:
        try:
            content = _read_bytes_no_follow(candidate)
        except OSError as error:
            failures.append({"path": str(candidate), "error": type(error).__name__})
            continue
        observed = _bytes_sha256(content)
        _require(
            observed == expected_sha256,
            f"readable robot record differs from source QA: {candidate}",
        )
        return (
            content,
            {
                "source_kind": "verified-extracted-file",
                "source_path": str(candidate),
                "archive_member": None,
                "verified_match_count": 1,
                "direct_candidate_failures": failures,
            },
        ), tuple(failures)
    return None, tuple(failures)


def _archive_failure_summary(
    *,
    root: Path,
    archive_paths: Sequence[Path],
    traversal_failures: Sequence[Mapping[str, str]],
    archive_failures: Sequence[Mapping[str, Any]],
    member_counts: Mapping[str, int],
    payload_read_counts: Mapping[str, int],
) -> dict[str, Any]:
    return {
        "archive_count": len(archive_paths),
        "archive_path_sample": [
            _relative_display(path, root) for path in archive_paths[:20]
        ],
        "traversal_failure_count": len(traversal_failures),
        "traversal_failure_sample": list(traversal_failures[:10]),
        "archive_failure_count": len(archive_failures),
        "archive_failure_sample": list(archive_failures[:20]),
        "development_robot_member_metadata_counts": dict(member_counts),
        "development_robot_member_payload_read_counts": dict(payload_read_counts),
    }


def _read_verified_archive_index(
    root: Path,
    expected_hashes: Mapping[str, str],
) -> tuple[dict[str, tuple[bytes, dict[str, Any]]], dict[str, Any]]:
    """Index ZIP metadata once and read only exact development robot members."""

    expected_take_ids = tuple(expected_hashes)
    archive_paths, traversal_failures = _bounded_archive_paths(root)
    matches: dict[str, list[tuple[bytes, Path, str]]] = {
        take_id: [] for take_id in expected_take_ids
    }
    member_counts = {take_id: 0 for take_id in expected_take_ids}
    payload_read_counts = {take_id: 0 for take_id in expected_take_ids}
    archive_failures: list[dict[str, Any]] = []
    readable_archive_count = 0

    for candidate in archive_paths:
        display_path = _relative_display(candidate, root)
        try:
            with _open_binary_no_follow(candidate) as handle:
                with zipfile.ZipFile(handle) as archive:
                    infos = archive.infolist()
                    names = [info.filename for info in infos]
                    if len(names) != len(set(names)):
                        archive_failures.append(
                            {
                                "path": display_path,
                                "error": "duplicate-members",
                            }
                        )
                        continue
                    indexed: list[tuple[zipfile.ZipInfo, PurePosixPath, str]] = []
                    invalid_member = None
                    for info in infos:
                        try:
                            member_path = _safe_member_path(info.filename)
                        except ValueError as error:
                            invalid_member = str(error)
                            break
                        if info.is_dir() or member_path.name != "robot_data.json":
                            continue
                        take_id = _member_take_id(
                            member_path,
                            candidate,
                            expected_take_ids,
                        )
                        if take_id is not None:
                            indexed.append((info, member_path, take_id))
                    if invalid_member is not None:
                        archive_failures.append(
                            {
                                "path": display_path,
                                "error": "unsafe-member",
                                "detail": invalid_member,
                            }
                        )
                        continue
                    readable_archive_count += 1
                    for info, member_path, take_id in indexed:
                        member_counts[take_id] += 1
                        try:
                            content = archive.read(info)
                        except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                            archive_failures.append(
                                {
                                    "path": display_path,
                                    "error": type(error).__name__,
                                    "take_id": take_id,
                                }
                            )
                            continue
                        payload_read_counts[take_id] += 1
                        observed = _bytes_sha256(content)
                        if observed == expected_hashes[take_id]:
                            matches[take_id].append(
                                (content, candidate, member_path.as_posix())
                            )
                        else:
                            archive_failures.append(
                                {
                                    "path": display_path,
                                    "error": "source-qa-digest-mismatch",
                                    "take_id": take_id,
                                    "observed_sha256": observed,
                                }
                            )
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
            archive_failures.append(
                {
                    "path": display_path,
                    "error": type(error).__name__,
                }
            )

    summary = _archive_failure_summary(
        root=root,
        archive_paths=archive_paths,
        traversal_failures=traversal_failures,
        archive_failures=archive_failures,
        member_counts=member_counts,
        payload_read_counts=payload_read_counts,
    )
    summary["readable_archive_count"] = readable_archive_count
    summary["archive_central_directories_read"] = True
    summary["nondevelopment_member_payloads_read"] = False

    missing = [take_id for take_id, values in matches.items() if not values]
    _require(
        not missing,
        "no source-QA-verified archive member for "
        f"{missing}; carrier diagnostic={json.dumps(summary, sort_keys=True)}",
    )

    selected: dict[str, tuple[bytes, dict[str, Any]]] = {}
    for take_id, values in matches.items():
        values.sort(key=lambda value: (str(value[1]), value[2]))
        content, archive_path, member_name = values[0]
        selected[take_id] = (
            content,
            {
                "source_kind": "verified-zip-member",
                "source_path": str(archive_path),
                "archive_member": member_name,
                "verified_match_count": len(values),
            },
        )
    return selected, summary


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

    resolved: dict[str, tuple[bytes, dict[str, Any]]] = {}
    direct_failures: dict[str, list[dict[str, str]]] = {}
    for take_id in config.expected_development_take_ids:
        direct, failures = _read_verified_direct(
            _direct_robot_candidates(root, config.expected_object_id, take_id),
            expected_hashes[take_id],
        )
        direct_failures[take_id] = list(failures)
        if direct is not None:
            resolved[take_id] = direct

    unresolved_hashes = {
        take_id: expected_hashes[take_id]
        for take_id in config.expected_development_take_ids
        if take_id not in resolved
    }
    archive_summary: dict[str, Any] = {
        "archive_scan_required": bool(unresolved_hashes),
        "archive_central_directories_read": False,
        "nondevelopment_member_payloads_read": False,
    }
    if unresolved_hashes:
        archive_records, archive_summary = _read_verified_archive_index(
            root,
            unresolved_hashes,
        )
        for take_id, value in archive_records.items():
            content, provenance = value
            provenance["direct_candidate_failures"] = direct_failures[take_id]
            resolved[take_id] = content, provenance

    _require(
        set(resolved) == set(config.expected_development_take_ids),
        "development robot-record staging roster is incomplete",
    )
    records = []
    for take_id in config.expected_development_take_ids:
        content, provenance = resolved[take_id]
        target = destination / config.expected_object_id / take_id / "robot_data.json"
        target.parent.mkdir(parents=True, exist_ok=False)
        target.write_bytes(content)
        os.chmod(target, 0o600)
        records.append(
            {
                "take_id": take_id,
                "robot_sha256": expected_hashes[take_id],
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
        "carrier_index": archive_summary,
        "information_boundary": {
            "development_robot_records_only": True,
            "development_meshes_read": False,
            "archive_central_directory_metadata_read": bool(
                archive_summary["archive_central_directories_read"]
            ),
            "nondevelopment_member_payloads_read": False,
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
        boundary.get("nondevelopment_member_payloads_read") is False,
        "stage read nondevelopment payloads",
    )
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
