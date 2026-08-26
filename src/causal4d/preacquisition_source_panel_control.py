"""Fail-closed progress and publication for the physical source panel."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import json
from pathlib import Path
from typing import Any, BinaryIO

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _assert_ordinary_file_or_missing,
    _reject_target_outcomes,
    _require,
)
from causal4d.atomic_io import atomic_write_binary, atomic_write_json
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    _canonical_sha256,
    _expected_source_panel,
    _read_json_mapping,
    _sha256_file,
    load_registered_preacquisition_chain,
    source_panel_execution_manifest_template,
)
from causal4d.preacquisition_source_validation import (
    _validate_source_execution_manifest,
)
from causal4d.reset_mode0_crosscheck import (
    load_reset_mode0_crosscheck_prerequisite,
)

SOURCE_PANEL_STATUS_SCHEMA_VERSION = 1
SOURCE_PANEL_STATUS_ARTIFACT_KIND = "Causal4DSourcePanelStatus"


def source_panel_evidence_sha256(values: Mapping[str, Any]) -> str:
    """Return a mount-independent digest of one source-panel status snapshot."""

    payload = deepcopy(dict(values))
    for field in ("dataset_root", "evidence_sha256", "status_sha256"):
        payload.pop(field, None)
    return _canonical_sha256(payload, omitted_field="evidence_sha256")


def source_panel_status_sha256(values: Mapping[str, Any]) -> str:
    """Return the digest of the exact host-local source-panel status snapshot."""

    return _canonical_sha256(values, omitted_field="status_sha256")


def _resolved_dataset_root(dataset_root: str | Path) -> Path:
    candidate = Path(dataset_root)
    _assert_no_symlink_components(candidate, name="dataset root")
    _require(candidate.is_dir(), "dataset root must exist")
    return candidate.resolve()


def _registered_source_executions(
    v2: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected_ids, expected_sessions = _expected_source_panel(v2)
    panel = v2.get("preacquisition_signature_panel")
    if not isinstance(panel, Mapping):
        raise ValueError("source-panel registration is missing")
    raw_executions = panel.get("executions")
    if not isinstance(raw_executions, list):
        raise ValueError("source-panel executions are missing")
    executions = [dict(value) for value in raw_executions]
    _require(
        [str(value.get("execution_id")) for value in executions] == expected_ids,
        "source-panel execution order differs from the registered panel",
    )
    _require(
        [str(value.get("session_id")) for value in executions] == expected_sessions,
        "source-panel session order differs from the registered panel",
    )
    return executions


def _registered_profiles(v2: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    panel = v2.get("preacquisition_signature_panel")
    if not isinstance(panel, Mapping):
        raise ValueError("source-panel registration is missing")
    raw_profiles = panel.get("profiles")
    if not isinstance(raw_profiles, list):
        raise ValueError("source-panel profiles are missing")
    profiles: dict[str, dict[str, Any]] = {}
    for index, value in enumerate(raw_profiles):
        if not isinstance(value, Mapping):
            raise ValueError(f"source-panel profile {index} is invalid")
        profile = dict(value)
        identifier = profile.get("id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError(f"source-panel profile {index} lacks an id")
        _require(
            identifier not in profiles,
            f"duplicate source-panel profile: {identifier}",
        )
        profiles[identifier] = profile
    return profiles


def _validate_template(
    path: Path,
    *,
    expected: Mapping[str, Any],
) -> None:
    _assert_no_symlink_components(path, name="source-panel manifest template")
    _require(path.is_file(), "source-panel manifest template is missing")
    actual = _read_json_mapping(path, name="source-panel manifest template")
    _require(
        actual == dict(expected),
        "source-panel manifest template differs from the registered scaffold",
    )


def _execution_status(
    dataset_root: Path,
    execution: Mapping[str, Any],
    *,
    protocol: Mapping[str, Any],
    v4: Mapping[str, Any],
    verify_file_hashes: bool,
) -> dict[str, Any]:
    execution_id = str(execution["execution_id"])
    session_id = str(execution["session_id"])
    manifest_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
    template_relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=execution_id
    )
    manifest_path = dataset_root / manifest_relative
    template_path = dataset_root / template_relative
    result: dict[str, Any] = {
        "execution_id": execution_id,
        "session_id": session_id,
        "manifest_path": manifest_relative,
        "template_path": template_relative,
        "manifest_present": manifest_path.exists() or manifest_path.is_symlink(),
        "template_present": template_path.is_file(),
        "template_valid": None,
        "template_error": None,
        "valid": False,
        "error": None,
    }
    expected_template = source_panel_execution_manifest_template(
        execution,
        protocol,
        v4,
    )
    if template_path.exists() or template_path.is_symlink():
        try:
            _validate_template(template_path, expected=expected_template)
        except (OSError, KeyError, TypeError, ValueError) as error:
            result["template_valid"] = False
            result["template_error"] = str(error)
        else:
            result["template_valid"] = True
    if not result["manifest_present"]:
        return result
    try:
        _assert_no_symlink_components(manifest_path, name="source-panel manifest")
        manifest = _read_json_mapping(manifest_path, name="source-panel manifest")
        _require(
            set(manifest) == set(expected_template),
            "source-panel manifest fields differ from schema version 1",
        )
        _reject_target_outcomes(manifest)
        _validate_source_execution_manifest(
            dataset_root,
            manifest_relative,
            protocol=protocol,
            v4=v4,
            execution_id=execution_id,
            session_id=session_id,
            verify_file_hashes=verify_file_hashes,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = str(error)
        return result
    result["valid"] = True
    return result


def _unexpected_execution_directories(
    dataset_root: Path,
    expected_ids: set[str],
) -> list[str]:
    root = dataset_root / "preacquisition" / "source_panel" / "executions"
    if not root.exists():
        return []
    _assert_no_symlink_components(root, name="source-panel execution root")
    _require(root.is_dir(), "source-panel execution root must be a directory")
    unexpected: list[str] = []
    for entry in root.iterdir():
        if entry.is_symlink() or not entry.is_dir() or entry.name not in expected_ids:
            unexpected.append(entry.name)
    return sorted(unexpected)


def _build_registered_source_panel_status(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Validate source-panel progress against an already loaded registration."""

    root = _resolved_dataset_root(dataset_root)
    executions = _registered_source_executions(v2)
    profiles = _registered_profiles(v2)
    results = [
        _execution_status(
            root,
            execution,
            protocol=protocol,
            v4=v4,
            verify_file_hashes=verify_file_hashes,
        )
        for execution in executions
    ]
    expected_ids = [str(value["execution_id"]) for value in executions]
    valid_ids = [result["execution_id"] for result in results if result["valid"]]
    invalid_ids = [
        result["execution_id"]
        for result in results
        if result["manifest_present"] and not result["valid"]
    ]
    invalid_template_ids = [
        result["execution_id"]
        for result in results
        if result["template_valid"] is False
    ]
    missing_ids = [
        result["execution_id"] for result in results if not result["manifest_present"]
    ]
    missing_template_ids = [
        result["execution_id"] for result in results if not result["template_present"]
    ]
    expected_prefix = expected_ids[: len(valid_ids)]
    out_of_order = valid_ids != expected_prefix
    unexpected_directories = _unexpected_execution_directories(
        root,
        set(expected_ids),
    )

    next_execution: dict[str, Any] | None = None
    for index, (execution, result) in enumerate(zip(executions, results, strict=True)):
        if result["valid"]:
            continue
        profile_id = str(execution["command_profile_id"])
        _require(
            profile_id in profiles,
            f"source execution references an unknown profile: {profile_id}",
        )
        next_execution = {
            "source_panel_execution_index": index,
            **dict(execution),
            "profile": profiles[profile_id],
            "manifest_path": result["manifest_path"],
            "template_path": result["template_path"],
            "template_present": result["template_present"],
            "template_valid": result["template_valid"],
        }
        break

    blockers: list[str] = []
    blockers.extend(f"manifest_missing:{identifier}" for identifier in missing_ids)
    blockers.extend(f"manifest_invalid:{identifier}" for identifier in invalid_ids)
    blockers.extend(
        f"template_invalid:{identifier}" for identifier in invalid_template_ids
    )
    blockers.extend(
        f"template_missing:{identifier}" for identifier in missing_template_ids
    )
    blockers.extend(
        f"unexpected_execution_directory:{identifier}"
        for identifier in unexpected_directories
    )
    if out_of_order:
        blockers.append("completed_manifests_do_not_form_registered_prefix")
    if not verify_file_hashes:
        blockers.append("file_hashes_not_verified")

    valid = not (
        invalid_ids
        or invalid_template_ids
        or missing_template_ids
        or unexpected_directories
        or out_of_order
    )
    complete = bool(
        valid and verify_file_hashes and len(valid_ids) == len(expected_ids)
    )
    status: dict[str, Any] = {
        "schema_version": SOURCE_PANEL_STATUS_SCHEMA_VERSION,
        "artifact_kind": SOURCE_PANEL_STATUS_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "dataset_root": str(root),
        "verify_file_hashes": verify_file_hashes,
        "specified_executions": len(expected_ids),
        "manifest_executions": sum(
            int(result["manifest_present"]) for result in results
        ),
        "validated_executions": len(valid_ids),
        "completed_execution_ids": valid_ids,
        "missing_execution_ids": missing_ids,
        "invalid_execution_ids": invalid_ids,
        "invalid_template_ids": invalid_template_ids,
        "missing_template_ids": missing_template_ids,
        "unexpected_execution_directories": unexpected_directories,
        "registered_prefix_valid": not out_of_order,
        "next_execution": next_execution,
        "executions": results,
        "blockers": blockers,
        "valid": valid,
        "complete": complete,
        "passed": complete,
        "target_outcomes_used": False,
    }
    status["evidence_sha256"] = source_panel_evidence_sha256(status)
    status["status_sha256"] = source_panel_status_sha256(status)
    return status


def build_source_panel_status(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Validate source-panel progress and identify the next physical execution."""

    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository_root)
    return _build_registered_source_panel_status(
        protocol,
        v2,
        v4,
        dataset_root,
        verify_file_hashes=verify_file_hashes,
    )


def write_source_panel_status(
    path: str | Path,
    status: Mapping[str, Any],
) -> Path:
    """Atomically replace one derived source-panel status snapshot."""

    output = Path(path)
    atomic_write_json(output, dict(status))
    return output


def _write_manifest_payload(handle: BinaryIO, payload: Mapping[str, Any]) -> None:
    serialized = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    handle.write(serialized + b"\n")


def publish_source_panel_manifest(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
) -> dict[str, Any]:
    """Validate and publish exactly the next source-panel manifest once."""

    protocol, _, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = _resolved_dataset_root(dataset_root)
    if "prospective_mode0_reset_crosscheck" in v4:
        reset_crosscheck = load_reset_mode0_crosscheck_prerequisite(
            protocol,
            v4,
            root,
            verify_file_hashes=True,
        )
        _require(
            reset_crosscheck.get("valid") is True,
            "source-panel publication requires the valid reset mode-0 cross-check: "
            f"{reset_crosscheck.get('error')}",
        )
    status_before = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(status_before["valid"] is True, "source-panel status is invalid")
    _require(status_before["complete"] is False, "source panel is already complete")
    next_execution = status_before.get("next_execution")
    if not isinstance(next_execution, Mapping):
        raise ValueError("source panel has no next execution")
    _require(
        next_execution.get("template_present") is True
        and next_execution.get("template_valid") is True,
        "next source-panel manifest template is missing or invalid",
    )

    source = Path(source_json)
    _assert_no_symlink_components(source, name="source-panel publication source")
    _require(source.is_file(), "source-panel publication source is missing")
    payload = _read_json_mapping(source, name="source-panel publication source")
    _reject_target_outcomes(payload)
    expected_template = source_panel_execution_manifest_template(
        next_execution,
        protocol,
        v4,
    )
    _require(
        set(payload) == set(expected_template),
        "source-panel manifest fields differ from schema version 1",
    )
    execution_id = str(next_execution["execution_id"])
    session_id = str(next_execution["session_id"])
    _require(
        payload.get("execution_id") == execution_id,
        "source-panel manifest is not the next registered execution",
    )
    _require(
        payload.get("session_id") == session_id,
        "source-panel manifest names the wrong session",
    )
    manifest_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
    final_path = root / manifest_relative
    _assert_ordinary_file_or_missing(final_path, name="source-panel manifest")
    _require(not final_path.exists(), "source-panel manifest already exists")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(
        final_path.parent,
        name="source-panel manifest parent",
    )

    def validate(temporary: Path) -> None:
        relative = temporary.resolve().relative_to(root).as_posix()
        _validate_source_execution_manifest(
            root,
            relative,
            protocol=protocol,
            v4=v4,
            execution_id=execution_id,
            session_id=session_id,
            verify_file_hashes=True,
        )

    atomic_write_binary(
        final_path,
        lambda handle: _write_manifest_payload(handle, payload),
        overwrite=False,
        validate=validate,
    )
    digest, byte_count = _sha256_file(final_path)
    status_after = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(
        status_after["valid"] is True,
        "published source-panel status is invalid",
    )
    _require(
        execution_id in status_after["completed_execution_ids"],
        "published source-panel manifest was not admitted",
    )
    return {
        "passed": True,
        "execution_id": execution_id,
        "session_id": session_id,
        "published_manifest": {
            "path": manifest_relative,
            "sha256": digest,
            "bytes": byte_count,
        },
        "source_panel_status": status_after,
        "target_outcomes_used": False,
    }


__all__ = [
    "SOURCE_PANEL_STATUS_ARTIFACT_KIND",
    "SOURCE_PANEL_STATUS_SCHEMA_VERSION",
    "build_source_panel_status",
    "publish_source_panel_manifest",
    "source_panel_evidence_sha256",
    "source_panel_status_sha256",
    "write_source_panel_status",
]
