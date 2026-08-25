"""Shared validators for the Causal4D real-evidence contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.contact_registration import (
    CONTACT_REGISTRATION_SCHEMA_VERSION,
    SINGLE_OPERATOR_CONTACT_REGISTRATION_SCHEMA_VERSION,
    validate_contact_registration,
)
from causal4d.real_protocol import (
    validate_object_registration,
    validate_protocol,
    validate_slip_pilot,
    write_acquisition_schedule,
)

EVIDENCE_STATUS_SCHEMA_VERSION = 2
SESSION_MANIFEST_SCHEMA_VERSION = 1
TIMEBASE_CALIBRATION_SCHEMA_VERSION = 1
METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _load_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    if not isinstance(payload, Mapping):
        raise ValueError(f"JSON root must be an object: {path}")
    return dict(payload)


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _finite_nonnegative(value: Any, *, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    _require(result >= 0.0, f"{name} must be nonnegative")
    return result


def _nonnegative_integer(value: Any, *, name: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0,
        f"{name} must be a nonnegative integer",
    )
    return int(value)


def _parse_utc_timestamp(value: Any, *, name: str) -> datetime:
    _require(isinstance(value, str) and bool(value), f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not ISO 8601") from error
    _require(parsed.tzinfo is not None, f"{name} must include a timezone")
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed), f"{name} must be UTC"
    )
    return parsed


def _safe_relative_path(value: Any, *, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} path is missing")
    path = Path(value)
    _require(
        not path.is_absolute() and ".." not in path.parts, f"{name} path is unsafe"
    )
    return path


def _validate_file_descriptor(
    dataset_root: Path,
    descriptor: Mapping[str, Any],
    *,
    name: str,
    verify_file_hashes: bool,
) -> None:
    relative = _safe_relative_path(descriptor.get("path"), name=name)
    _require(_is_sha256(descriptor.get("sha256")), f"{name} SHA-256 is invalid")
    expected_bytes = _nonnegative_integer(descriptor.get("bytes"), name=f"{name} bytes")
    if verify_file_hashes:
        path = dataset_root / relative
        _require(path.is_file(), f"{name} file is missing: {path}")
        digest, byte_count = _sha256_file(path)
        _require(digest == descriptor["sha256"], f"{name} checksum mismatch")
        _require(byte_count == expected_bytes, f"{name} byte count mismatch")


def _iter_descriptors(
    value: Any, prefix: str = "artifact"
) -> list[tuple[str, Mapping[str, Any]]]:
    descriptors: list[tuple[str, Mapping[str, Any]]] = []
    if isinstance(value, Mapping):
        if {"path", "sha256", "bytes"} <= set(value):
            descriptors.append((prefix, value))
        else:
            for key, child in value.items():
                descriptors.extend(_iter_descriptors(child, f"{prefix}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            descriptors.extend(_iter_descriptors(child, f"{prefix}[{index}]"))
    return descriptors


def _prerequisite_result(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "present": path.is_file(),
        "valid": False,
        "error": None,
    }


def _finalize_prerequisite(result: dict[str, Any], path: Path) -> dict[str, Any]:
    result["valid"] = True
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result


def _validate_dataset_protocol(
    protocol: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    if not result["present"]:
        result["error"] = "dataset protocol.json is missing"
        return result
    try:
        candidate = _load_json_mapping(path)
        validate_protocol(candidate)
        _require(
            candidate == dict(protocol),
            "dataset protocol differs from the locked protocol",
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    return _finalize_prerequisite(result, path)


def _expected_schedule_rows(protocol: Mapping[str, Any]) -> list[dict[str, str]]:
    with tempfile.TemporaryDirectory(prefix="causal4d-status-v2-") as directory:
        generated = write_acquisition_schedule(
            Path(directory) / "acquisition_schedule.csv",
            protocol,
        )
        with generated.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))


def _validate_acquisition_schedule(
    protocol: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    if not result["present"]:
        result["error"] = "acquisition_schedule.csv is missing"
        return result
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        _require(
            rows == _expected_schedule_rows(protocol),
            "acquisition schedule differs from the locked design",
        )
    except (OSError, csv.Error, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    result["row_count"] = len(rows)
    return _finalize_prerequisite(result, path)


def _validate_registration_files(
    dataset_root: Path,
    registration: Mapping[str, Any],
) -> None:
    for region_id, descriptor in registration["contact_regions"].items():
        relative = _safe_relative_path(
            descriptor["canonical_node_set_path"],
            name=f"contact node set {region_id}",
        )
        node_path = dataset_root / relative
        _require(node_path.is_file(), f"contact node set is missing: {region_id}")
        digest, _ = _sha256_file(node_path)
        _require(
            digest == descriptor["canonical_node_set_sha256"],
            f"contact node-set checksum mismatch: {region_id}",
        )


def _validate_object_registration_prerequisite(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    path: Path,
    *,
    verify_file_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = _prerequisite_result(path)
    result["file_hashes_verified"] = None if not verify_file_hashes else False
    if not result["present"]:
        result["error"] = "object_registration.json is missing"
        return result, None
    try:
        registration = _load_json_mapping(path)
        validate_object_registration(protocol, registration)
        if verify_file_hashes:
            _validate_registration_files(dataset_root, registration)
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result, None
    result["file_hashes_verified"] = True if verify_file_hashes else None
    return _finalize_prerequisite(result, path), registration


def _validate_slip_pilot_prerequisite(
    protocol: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    result = _prerequisite_result(path)
    if not result["present"]:
        result["error"] = "slip_pilot.json is missing"
        return result
    try:
        validate_slip_pilot(protocol, _load_json_mapping(path))
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result
    return _finalize_prerequisite(result, path)


def timebase_calibration_template(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Return an explicitly incomplete shared-clock calibration template."""

    validate_protocol(protocol)
    return {
        "schema_version": TIMEBASE_CALIBRATION_SCHEMA_VERSION,
        "artifact_kind": "TimebaseCalibration",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "status": "template",
        "clock_domain_id": None,
        "calibrated_streams": sorted(
            protocol["recording_contract"]["timestamped_artifacts"]
        ),
        "measured_max_sync_error_ms": None,
        "calibrated_at_utc": None,
        "locked_before_confirmatory_collection": None,
        "calibration_artifact": {"path": None, "sha256": None, "bytes": None},
        "approval": {
            "approved": False,
            "approver_id": None,
            "approved_at_utc": None,
        },
    }


def validate_timebase_calibration(
    protocol: Mapping[str, Any],
    calibration: Mapping[str, Any],
    *,
    dataset_root: str | Path | None = None,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Validate the one clock domain used by every timestamped stream."""

    validate_protocol(protocol)
    _require(
        calibration.get("schema_version") == TIMEBASE_CALIBRATION_SCHEMA_VERSION,
        "unsupported timebase calibration schema",
    )
    _require(
        calibration.get("artifact_kind") == "TimebaseCalibration",
        "unexpected timebase calibration kind",
    )
    _require(
        calibration.get("status") == "approved", "timebase calibration is not approved"
    )
    _require(
        calibration.get("protocol_id") == protocol["protocol_id"],
        "timebase protocol mismatch",
    )
    _require(
        calibration.get("protocol_design_sha256") == protocol["design_sha256"],
        "timebase protocol digest mismatch",
    )
    clock_domain_id = calibration.get("clock_domain_id")
    _require(
        isinstance(clock_domain_id, str) and bool(clock_domain_id),
        "clock domain id is missing",
    )
    expected_streams = set(protocol["recording_contract"]["timestamped_artifacts"])
    streams = calibration.get("calibrated_streams")
    _require(
        isinstance(streams, list)
        and set(streams) == expected_streams
        and len(streams) == len(expected_streams),
        "timebase calibration does not cover the exact timestamped stream set",
    )
    measured = _finite_nonnegative(
        calibration.get("measured_max_sync_error_ms"),
        name="measured_max_sync_error_ms",
    )
    maximum = float(protocol["quality_gates"]["maximum_rgbd_actuator_sync_error_ms"])
    _require(
        measured <= maximum, "timebase synchronization error exceeds the locked gate"
    )
    calibrated_at = _parse_utc_timestamp(
        calibration.get("calibrated_at_utc"), name="timebase calibrated_at_utc"
    )
    _require(
        calibration.get("locked_before_confirmatory_collection") is True,
        "timebase was not locked before confirmatory collection",
    )
    approval = calibration.get("approval", {})
    _require(approval.get("approved") is True, "timebase approval is missing")
    _require(
        isinstance(approval.get("approver_id"), str) and bool(approval["approver_id"]),
        "timebase approver id is missing",
    )
    approved_at = _parse_utc_timestamp(
        approval.get("approved_at_utc"), name="timebase approved_at_utc"
    )
    _require(
        approved_at >= calibrated_at,
        "timebase approval predates calibration",
    )
    root = Path(dataset_root) if dataset_root is not None else Path(".")
    _validate_file_descriptor(
        root,
        calibration.get("calibration_artifact", {}),
        name="timebase calibration artifact",
        verify_file_hashes=verify_file_hashes,
    )
    return {
        "passed": True,
        "clock_domain_id": clock_domain_id,
        "measured_max_sync_error_ms": measured,
        "calibrated_at_utc": calibration["calibrated_at_utc"],
        "approved_at_utc": approval["approved_at_utc"],
        "file_hashes_verified": verify_file_hashes,
    }


def _validate_timebase_prerequisite(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    path: Path,
    *,
    verify_file_hashes: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = _prerequisite_result(path)
    result["file_hashes_verified"] = None if not verify_file_hashes else False
    if not result["present"]:
        result["error"] = "timebase_calibration.json is missing"
        return result, None
    try:
        calibration = _load_json_mapping(path)
        validation = validate_timebase_calibration(
            protocol,
            calibration,
            dataset_root=dataset_root,
            verify_file_hashes=verify_file_hashes,
        )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result, None
    result.update(validation)
    result["valid"] = True
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result, calibration


def _cross_check_contact_registration(
    physical: Mapping[str, Any],
    simple: Mapping[str, Any],
    *,
    simple_sha256: str,
) -> None:
    _require(
        physical.get("schema_version")
        in {
            CONTACT_REGISTRATION_SCHEMA_VERSION,
            SINGLE_OPERATOR_CONTACT_REGISTRATION_SCHEMA_VERSION,
        },
        "physical contact registration must use schema 3 or 4",
    )
    object_record = physical["object"]
    _require(
        object_record["physical_instance_serial"] == simple["object_instance_serial"],
        "physical and simple registration object serials differ",
    )
    _require(
        object_record["twin_geometry_sha256"] == simple["phystwin_model_sha256"],
        "physical and simple registration twin hashes differ",
    )
    checksums = physical.get("source_checksums", {})
    _require(
        checksums.get("object_registration.json") == simple_sha256,
        "physical registration does not bind object_registration.json",
    )
    for region_id, descriptor in simple["contact_regions"].items():
        attachment = physical["contact_regions"][region_id]["attachment"]
        _require(
            descriptor["node_count"] == len(attachment["node_indices"]),
            f"physical and simple registration node counts differ: {region_id}",
        )
        _require(
            checksums.get(f"contact_node_set:{region_id}")
            == descriptor["canonical_node_set_sha256"],
            f"physical registration does not bind the contact node set: {region_id}",
        )


def _validate_contact_registration_prerequisite(
    protocol: Mapping[str, Any],
    dataset_root: Path,
    path: Path,
    *,
    simple_registration: Mapping[str, Any] | None,
    simple_registration_sha256: str | None,
    verify_file_hashes: bool,
    require_single_operator_review: bool = False,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    result = _prerequisite_result(path)
    result["file_hashes_verified"] = None if not verify_file_hashes else False
    if not result["present"]:
        result["error"] = "contact_registration.json is missing"
        return result, None
    try:
        _require(
            simple_registration is not None,
            "object_registration.json must validate first",
        )
        _require(
            _is_sha256(simple_registration_sha256),
            "object registration digest is unavailable",
        )
        physical = _load_json_mapping(path)
        if require_single_operator_review:
            _require(
                physical.get("schema_version")
                == SINGLE_OPERATOR_CONTACT_REGISTRATION_SCHEMA_VERSION,
                "single-operator governance requires contact registration schema 4",
            )
        validation = validate_contact_registration(physical, protocol)
        _cross_check_contact_registration(
            physical,
            simple_registration,
            simple_sha256=str(simple_registration_sha256),
        )
        if verify_file_hashes:
            for name, descriptor in _iter_descriptors(physical, "contact_registration"):
                _validate_file_descriptor(
                    dataset_root,
                    descriptor,
                    name=name,
                    verify_file_hashes=True,
                )
    except (OSError, KeyError, TypeError, ValueError) as error:
        result["error"] = _error_text(error)
        return result, None
    result.update(validation)
    result["file_hashes_verified"] = True if verify_file_hashes else None
    result["valid"] = True
    result["sha256"], result["bytes"] = _sha256_file(path)
    return result, physical


__all__ = [
    "EVIDENCE_STATUS_SCHEMA_VERSION",
    "METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION",
    "SESSION_MANIFEST_SCHEMA_VERSION",
    "TIMEBASE_CALIBRATION_SCHEMA_VERSION",
    "_cross_check_contact_registration",
    "_error_text",
    "_finalize_prerequisite",
    "_finite_nonnegative",
    "_is_sha256",
    "_iter_descriptors",
    "_load_json_mapping",
    "_nonnegative_integer",
    "_parse_utc_timestamp",
    "_prerequisite_result",
    "_require",
    "_sha256_file",
    "_validate_acquisition_schedule",
    "_validate_contact_registration_prerequisite",
    "_validate_dataset_protocol",
    "_validate_file_descriptor",
    "_validate_object_registration_prerequisite",
    "_validate_slip_pilot_prerequisite",
    "_validate_timebase_prerequisite",
    "timebase_calibration_template",
    "validate_timebase_calibration",
]
