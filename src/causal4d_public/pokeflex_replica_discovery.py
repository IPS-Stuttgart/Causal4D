"""Discover an authorized runner-readable PokeFlex development replica."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence


POKEFLEX_REPLICA_DISCOVERY_KIND = "PublicPokeFlexDevelopmentReplicaDiscovery"
POKEFLEX_REPLICA_DISCOVERY_SCHEMA_VERSION = 1
EXPECTED_SOURCE_QA_SHA256 = (
    "e09d36db4e1ba8a38c70e112c3af9ab95516ee245302f71a853f36cd2dd0e0e7"
)
EXPECTED_OBJECT_ID = "3dPrintedBunny"
DEVELOPMENT_TAKE_IDS = (
    "3dPrintedBunny_T1",
    "3dPrintedBunny_T3",
    "3dPrintedBunny_T4",
    "3dPrintedBunny_T6",
    "3dPrintedBunny_T7",
)
FORBIDDEN_TAKE_IDS = ("3dPrintedBunny_T2", "3dPrintedBunny_T5")
MAX_DIRECTORY_COUNT = 100_000
MAX_FILE_COUNT = 500_000
MAX_ARCHIVE_COUNT = 1_000
MAX_DEPTH = 10
PRUNED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".tox",
        ".venv",
        "__pycache__",
        "deform",
        "deform360",
        "dot",
        "images",
        "kinect",
        "meshes",
        "node_modules",
        "tracking-cloth-deformation-v1-zenodo-14644526",
        "triangle_meshes",
        "volucam",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def replica_discovery_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _source_qa_hashes(payload: Mapping[str, Any]) -> dict[str, str]:
    _require(payload.get("artifact_kind") == "PublicPokeFlexSourceQa", "bad QA kind")
    _require(payload.get("schema_version") == 1, "bad QA schema")
    _require(
        payload.get("result_sha256") == EXPECTED_SOURCE_QA_SHA256,
        "source QA identity changed",
    )
    _require(payload.get("object_id") == EXPECTED_OBJECT_ID, "source QA object changed")
    boundary = payload.get("information_boundary", {})
    _require(
        tuple(sorted(map(str, boundary.get("opened_take_ids", ()))))
        == tuple(sorted(DEVELOPMENT_TAKE_IDS)),
        "source QA development roster changed",
    )
    _require(boundary.get("calibration_take_data_read") is False, "QA opened T5")
    _require(boundary.get("target_take_data_read") is False, "QA opened T2")
    records = payload.get("takes")
    _require(isinstance(records, list), "source QA take inventory is missing")
    result: dict[str, str] = {}
    for record in records:
        _require(isinstance(record, Mapping), "source QA take row is invalid")
        take_id = str(record.get("take_id", ""))
        if take_id not in DEVELOPMENT_TAKE_IDS:
            continue
        digest = str(record.get("robot_sha256", ""))
        _require(
            len(digest) == 64 and all(value in "0123456789abcdef" for value in digest),
            f"invalid robot digest for {take_id}",
        )
        _require(take_id not in result, f"duplicate QA take: {take_id}")
        result[take_id] = digest
    _require(set(result) == set(DEVELOPMENT_TAKE_IDS), "QA robot roster changed")
    return result


def _safe_member_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name.replace("\\", "/"))
    _require(not path.is_absolute(), "archive member is absolute")
    _require(".." not in path.parts, "archive member escapes its root")
    return path


def _take_from_parts(parts: Sequence[str]) -> str | None:
    normalized = {part.lower() for part in parts}
    matches = [take_id for take_id in DEVELOPMENT_TAKE_IDS if take_id.lower() in normalized]
    return matches[0] if len(matches) == 1 else None


def _should_inspect_archive(path: Path) -> bool:
    lower = path.as_posix().lower()
    exact = {f"{take_id.lower()}.zip" for take_id in DEVELOPMENT_TAKE_IDS}
    return path.name.lower() in exact or any(
        token in lower for token in ("pokeflex", "bunny", "opaque", "stage")
    )


def _scan_candidates(
    roots: Sequence[Path],
) -> tuple[list[Path], list[Path], dict[str, Any]]:
    direct_files: list[Path] = []
    archives: list[Path] = []
    errors: list[dict[str, str]] = []
    seen_paths: set[str] = set()
    directory_count = 0
    file_count = 0
    stack: list[tuple[Path, int]] = []
    for root in roots:
        absolute = Path(os.path.abspath(os.fspath(root)))
        if str(absolute) in seen_paths:
            continue
        seen_paths.add(str(absolute))
        stack.append((absolute, 0))

    while stack:
        directory, depth = stack.pop()
        directory_count += 1
        if directory_count > MAX_DIRECTORY_COUNT:
            errors.append({"path": str(directory), "error": "directory-bound-reached"})
            break
        try:
            with os.scandir(directory) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
        except OSError as error:
            errors.append({"path": str(directory), "error": type(error).__name__})
            continue
        for entry in entries:
            try:
                if entry.is_symlink():
                    continue
                path = Path(entry.path)
                if entry.is_file(follow_symlinks=False):
                    file_count += 1
                    if file_count > MAX_FILE_COUNT:
                        errors.append({"path": str(path), "error": "file-bound-reached"})
                        stack.clear()
                        break
                    if entry.name == "robot_data.json":
                        take_id = _take_from_parts(path.parts[:-1])
                        if take_id is not None:
                            direct_files.append(path)
                    elif entry.name.lower().endswith(".zip") and _should_inspect_archive(path):
                        archives.append(path)
                        if len(archives) >= MAX_ARCHIVE_COUNT:
                            stack.clear()
                            break
                    continue
                if (
                    depth < MAX_DEPTH
                    and entry.is_dir(follow_symlinks=False)
                    and entry.name.lower() not in PRUNED_DIRECTORY_NAMES
                ):
                    stack.append((path, depth + 1))
            except OSError as error:
                errors.append({"path": str(entry.path), "error": type(error).__name__})
    return (
        sorted(set(direct_files), key=str),
        sorted(set(archives), key=str),
        {
            "search_roots": [str(Path(root)) for root in roots],
            "directory_count": directory_count,
            "file_count": file_count,
            "direct_candidate_count": len(set(direct_files)),
            "archive_candidate_count": len(set(archives)),
            "errors": errors[:200],
            "truncated": any(
                row["error"] in {"directory-bound-reached", "file-bound-reached"}
                for row in errors
            ),
        },
    )


def _record_match(
    matches: dict[str, list[dict[str, Any]]],
    take_id: str,
    content: bytes,
    expected_hashes: Mapping[str, str],
    provenance: Mapping[str, Any],
) -> None:
    digest = _sha256_bytes(content)
    if digest != expected_hashes[take_id]:
        return
    matches[take_id].append(
        {
            "content": content,
            "robot_sha256": digest,
            **dict(provenance),
        }
    )


def _read_matches(
    direct_files: Sequence[Path],
    archives: Sequence[Path],
    expected_hashes: Mapping[str, str],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    matches: dict[str, list[dict[str, Any]]] = {
        take_id: [] for take_id in DEVELOPMENT_TAKE_IDS
    }
    errors: list[dict[str, Any]] = []
    development_payload_read_count = 0
    archive_metadata_read_count = 0
    for path in direct_files:
        take_id = _take_from_parts(path.parts[:-1])
        if take_id is None:
            continue
        try:
            content = path.read_bytes()
        except OSError as error:
            errors.append({"path": str(path), "error": type(error).__name__})
            continue
        development_payload_read_count += 1
        _record_match(
            matches,
            take_id,
            content,
            expected_hashes,
            {"source_kind": "extracted-file", "source_path": str(path)},
        )

    for path in archives:
        try:
            with zipfile.ZipFile(path) as archive:
                infos = archive.infolist()
                archive_metadata_read_count += 1
                names = [info.filename for info in infos]
                if len(names) != len(set(names)):
                    errors.append({"path": str(path), "error": "duplicate-members"})
                    continue
                for info in infos:
                    try:
                        member = _safe_member_path(info.filename)
                    except ValueError as error:
                        errors.append(
                            {"path": str(path), "error": "unsafe-member", "detail": str(error)}
                        )
                        break
                    if info.is_dir() or member.name != "robot_data.json":
                        continue
                    take_id = _take_from_parts(member.parts[:-1])
                    if take_id is None:
                        continue
                    try:
                        content = archive.read(info)
                    except (OSError, RuntimeError, zipfile.BadZipFile) as error:
                        errors.append(
                            {
                                "path": str(path),
                                "member": member.as_posix(),
                                "error": type(error).__name__,
                            }
                        )
                        continue
                    development_payload_read_count += 1
                    _record_match(
                        matches,
                        take_id,
                        content,
                        expected_hashes,
                        {
                            "source_kind": "zip-member",
                            "source_path": str(path),
                            "archive_member": member.as_posix(),
                        },
                    )
        except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as error:
            errors.append({"path": str(path), "error": type(error).__name__})
    return matches, {
        "archive_metadata_read_count": archive_metadata_read_count,
        "development_payload_read_count": development_payload_read_count,
        "nondevelopment_payload_read_count": 0,
        "errors": errors[:200],
    }


def _validate_existing_cache(
    cache_root: Path,
    expected_hashes: Mapping[str, str],
) -> bool:
    if not cache_root.is_dir():
        return False
    for take_id, expected in expected_hashes.items():
        path = cache_root / EXPECTED_OBJECT_ID / take_id / "robot_data.json"
        if not path.is_file() or _sha256_file(path) != expected:
            return False
    return True


def _write_cache(
    cache_root: Path,
    selected: Mapping[str, Mapping[str, Any]],
    expected_hashes: Mapping[str, str],
) -> tuple[bool, str]:
    if cache_root.exists():
        if _validate_existing_cache(cache_root, expected_hashes):
            return False, "existing-cache-verified"
        return False, "existing-cache-conflicts"
    cache_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=".pokeflex-realized-load-", dir=cache_root.parent)
    )
    try:
        os.chmod(temporary, 0o700)
        for take_id in DEVELOPMENT_TAKE_IDS:
            destination = temporary / EXPECTED_OBJECT_ID / take_id / "robot_data.json"
            destination.parent.mkdir(parents=True, exist_ok=False)
            os.chmod(destination.parent, 0o700)
            destination.write_bytes(selected[take_id]["content"])
            os.chmod(destination, 0o600)
        manifest = {
            "artifact_kind": "PublicPokeFlexDevelopmentRobotCache",
            "schema_version": 1,
            "source_qa_result_sha256": EXPECTED_SOURCE_QA_SHA256,
            "object_id": EXPECTED_OBJECT_ID,
            "development_take_ids": list(DEVELOPMENT_TAKE_IDS),
            "robot_sha256": dict(expected_hashes),
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.chmod(temporary / "manifest.json", 0o600)
        os.replace(temporary, cache_root)
        return True, "cache-created"
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def discover_pokeflex_development_replica(
    *,
    source_qa: Mapping[str, Any],
    search_roots: Sequence[str | Path],
    cache_root: str | Path,
) -> dict[str, Any]:
    """Find and cache only the five source-QA-authorized robot records."""

    expected_hashes = _source_qa_hashes(source_qa)
    normalized_roots = tuple(Path(root) for root in search_roots)
    direct_files, archives, scan = _scan_candidates(normalized_roots)
    matches, reads = _read_matches(direct_files, archives, expected_hashes)
    selected: dict[str, dict[str, Any]] = {}
    for take_id in DEVELOPMENT_TAKE_IDS:
        candidates = sorted(
            matches[take_id],
            key=lambda record: (
                str(record.get("source_path", "")),
                str(record.get("archive_member", "")),
            ),
        )
        if candidates:
            selected[take_id] = candidates[0]
    complete = set(selected) == set(DEVELOPMENT_TAKE_IDS)
    cache = Path(cache_root)
    cache_written = False
    cache_status = "incomplete-source-roster"
    if complete:
        cache_written, cache_status = _write_cache(cache, selected, expected_hashes)

    public_selected = {
        take_id: {
            key: value
            for key, value in record.items()
            if key != "content"
        }
        for take_id, record in selected.items()
    }
    result: dict[str, Any] = {
        "artifact_kind": POKEFLEX_REPLICA_DISCOVERY_KIND,
        "schema_version": POKEFLEX_REPLICA_DISCOVERY_SCHEMA_VERSION,
        "source_qa_result_sha256": EXPECTED_SOURCE_QA_SHA256,
        "object_id": EXPECTED_OBJECT_ID,
        "development_take_ids": list(DEVELOPMENT_TAKE_IDS),
        "forbidden_take_ids": list(FORBIDDEN_TAKE_IDS),
        "complete": complete,
        "found_take_ids": sorted(selected),
        "missing_take_ids": sorted(set(DEVELOPMENT_TAKE_IDS) - set(selected)),
        "match_counts": {
            take_id: len(matches[take_id]) for take_id in DEVELOPMENT_TAKE_IDS
        },
        "selected_sources": public_selected,
        "cache_root": str(cache),
        "cache_written": cache_written,
        "cache_status": cache_status,
        "cache_verified": _validate_existing_cache(cache, expected_hashes),
        "scan": scan,
        "reads": reads,
        "information_boundary": {
            "filesystem_metadata_scanned": True,
            "archive_central_directories_read": reads["archive_metadata_read_count"] > 0,
            "development_robot_payloads_read": reads[
                "development_payload_read_count"
            ],
            "nondevelopment_payloads_read": 0,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "source_trees_modified": False,
            "new_physical_data_collected": False,
        },
        "claim_boundary": (
            "Technical custody discovery only. This is not a source-gate result and "
            "cannot authorize calibration, target access, or a paper claim."
        ),
    }
    result["result_sha256"] = replica_discovery_sha256(result)
    return result


def validate_pokeflex_replica_discovery(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("artifact_kind") == POKEFLEX_REPLICA_DISCOVERY_KIND,
        "unexpected replica-discovery kind",
    )
    _require(
        payload.get("schema_version") == POKEFLEX_REPLICA_DISCOVERY_SCHEMA_VERSION,
        "unsupported replica-discovery schema",
    )
    _require(
        payload.get("result_sha256") == replica_discovery_sha256(payload),
        "replica-discovery checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("nondevelopment_payloads_read") == 0, "payload scope widened")
    _require(boundary.get("calibration_take_data_read") is False, "T5 was opened")
    _require(boundary.get("target_take_data_read") is False, "T2 was opened")
    return {
        "passed": True,
        "complete": bool(payload["complete"]),
        "cache_verified": bool(payload["cache_verified"]),
        "result_sha256": payload["result_sha256"],
    }


__all__ = [
    "DEVELOPMENT_TAKE_IDS",
    "EXPECTED_OBJECT_ID",
    "EXPECTED_SOURCE_QA_SHA256",
    "FORBIDDEN_TAKE_IDS",
    "POKEFLEX_REPLICA_DISCOVERY_KIND",
    "POKEFLEX_REPLICA_DISCOVERY_SCHEMA_VERSION",
    "discover_pokeflex_development_replica",
    "replica_discovery_sha256",
    "validate_pokeflex_replica_discovery",
]
