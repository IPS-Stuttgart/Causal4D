"""Content-addressed handoff packets for pre-acquisition operator actions."""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
import tempfile
from typing import Any, BinaryIO
from zipfile import ZIP_STORED, BadZipFile, ZipFile, ZipInfo

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _reject_target_outcomes,
    _require,
)
from causal4d.atomic_io import atomic_write_binary, atomic_write_json
from causal4d.preacquisition_next_action import (
    NEXT_ACTION_ARTIFACT_KIND,
    next_action_evidence_sha256,
    next_action_status_sha256,
)
from causal4d.preacquisition_next_action_validation import (
    validate_preacquisition_next_action_report,
)
from causal4d.preacquisition_operator_flow import (
    NEXT_ACTION_SCHEMA_VERSION,
    render_preacquisition_operator_next_action_markdown,
)
from causal4d.preacquisition_readiness_contracts import _sha256_file


NEXT_ACTION_PACKET_SCHEMA_VERSION = 1
NEXT_ACTION_PACKET_ARTIFACT_KIND = "Causal4DPreacquisitionNextActionPacket"
NEXT_ACTION_PACKET_VALIDATION_SCHEMA_VERSION = 1
NEXT_ACTION_PACKET_VALIDATION_ARTIFACT_KIND = (
    "Causal4DPreacquisitionNextActionPacketValidation"
)

_DECISION_MEMBER = "decision.json"
_INSTRUCTIONS_MEMBER = "instructions.md"
_MANIFEST_MEMBER = "manifest.json"
_PACKET_MEMBERS = (_DECISION_MEMBER, _INSTRUCTIONS_MEMBER, _MANIFEST_MEMBER)
_MAX_MEMBER_BYTES = 2 * 1024 * 1024


def _canonical_sha256(value: Mapping[str, Any], *, omitted: str) -> str:
    payload = deepcopy(dict(value))
    payload.pop(omitted, None)
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _serialized_json(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            dict(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _object_without_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require(key not in result, f"duplicate JSON object key is forbidden: {key!r}")
        result[key] = value
    return result


def _json_mapping(data: bytes, *, name: str) -> dict[str, Any]:
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{name} is not UTF-8") from error
    value = json.loads(
        text,
        object_pairs_hook=_object_without_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return dict(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _action_identity(decision: Mapping[str, Any]) -> dict[str, Any]:
    action = decision.get("action")
    _require(isinstance(action, Mapping), "next-action decision has no action object")
    registered = action.get("registered_execution")
    execution_id = None
    session_id = None
    if isinstance(registered, Mapping):
        execution_id = registered.get("execution_id")
        session_id = registered.get("session_id")
    return {
        "action_id": action.get("action_id"),
        "category": action.get("category"),
        "execution_id": execution_id,
        "session_id": session_id,
    }


def _validate_decision(decision: Mapping[str, Any]) -> None:
    _require(
        decision.get("schema_version") == NEXT_ACTION_SCHEMA_VERSION,
        "unsupported next-action decision schema",
    )
    _require(
        decision.get("artifact_kind") == NEXT_ACTION_ARTIFACT_KIND,
        "unexpected next-action artifact kind",
    )
    _require(
        decision.get("target_outcomes_used") is False,
        "target outcomes entered the next-action decision",
    )
    action = decision.get("action")
    _require(isinstance(action, Mapping), "next-action decision has no action object")
    _require(
        action.get("target_outcomes_permitted") is False,
        "next-action decision permits target outcomes",
    )
    _require(
        action.get("changes_registered_method") is False,
        "next-action decision permits a registered-method change",
    )
    _require(
        decision.get("evidence_sha256") == next_action_evidence_sha256(decision),
        "next-action evidence SHA-256 mismatch",
    )
    _require(
        decision.get("status_sha256") == next_action_status_sha256(decision),
        "next-action status SHA-256 mismatch",
    )
    _reject_target_outcomes(decision)


def _instructions_bytes(decision: Mapping[str, Any]) -> bytes:
    text = render_preacquisition_operator_next_action_markdown(decision)
    if not text.endswith("\n"):
        text += "\n"
    return text.encode("utf-8")


def next_action_packet_id(values: Mapping[str, Any]) -> str:
    """Return the logical content ID of one operator packet manifest."""

    return _canonical_sha256(values, omitted="packet_id")


def build_preacquisition_next_action_packet_manifest(
    decision: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes, bytes]:
    """Build the packet manifest and exact human/machine member bytes."""

    _validate_decision(decision)
    decision_bytes = _serialized_json(decision)
    instructions_bytes = _instructions_bytes(decision)
    action = decision["action"]
    assert isinstance(action, Mapping)
    manifest: dict[str, Any] = {
        "schema_version": NEXT_ACTION_PACKET_SCHEMA_VERSION,
        "artifact_kind": NEXT_ACTION_PACKET_ARTIFACT_KIND,
        "protocol_id": decision["protocol_id"],
        "protocol_design_sha256": decision["protocol_design_sha256"],
        "preacquisition_plan_id": decision["preacquisition_plan_id"],
        "preacquisition_amendment_sha256": decision[
            "preacquisition_amendment_sha256"
        ],
        "decision_evidence_sha256": decision["evidence_sha256"],
        "decision_status_sha256": decision["status_sha256"],
        "decision_valid": decision.get("valid") is True,
        "decision_ready": decision.get("ready") is True,
        "action_identity": _action_identity(decision),
        "members": {
            "decision": {
                "path": _DECISION_MEMBER,
                "media_type": "application/json",
                "sha256": _sha256_bytes(decision_bytes),
                "bytes": len(decision_bytes),
            },
            "instructions": {
                "path": _INSTRUCTIONS_MEMBER,
                "media_type": "text/markdown; charset=utf-8",
                "sha256": _sha256_bytes(instructions_bytes),
                "bytes": len(instructions_bytes),
            },
        },
        "target_outcomes_used": False,
        "target_outcomes_permitted": False,
        "changes_registered_method": False,
        "physical_acquisition_required": (
            action.get("physical_acquisition_required") is True
        ),
        "valid": True,
        "complete": True,
        "passed": True,
    }
    manifest["packet_id"] = next_action_packet_id(manifest)
    return manifest, decision_bytes, instructions_bytes


def _zip_info(name: str) -> ZipInfo:
    info = ZipInfo(filename=name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.flag_bits |= 0x800
    return info


def build_preacquisition_next_action_packet_bytes(
    decision: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    """Return deterministic ZIP bytes and their logical manifest."""

    manifest, decision_bytes, instructions_bytes = (
        build_preacquisition_next_action_packet_manifest(decision)
    )
    manifest_bytes = _serialized_json(manifest)
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_STORED) as archive:
        for name, value in (
            (_DECISION_MEMBER, decision_bytes),
            (_INSTRUCTIONS_MEMBER, instructions_bytes),
            (_MANIFEST_MEMBER, manifest_bytes),
        ):
            archive.writestr(_zip_info(name), value)
    return buffer.getvalue(), manifest


def _member_record(
    manifest: Mapping[str, Any],
    *,
    name: str,
    expected_path: str,
    data: bytes,
) -> None:
    members = manifest.get("members")
    _require(isinstance(members, Mapping), "packet manifest members are missing")
    record = members.get(name)
    _require(isinstance(record, Mapping), f"packet {name} member record is missing")
    _require(record.get("path") == expected_path, f"packet {name} path changed")
    _require(
        record.get("sha256") == _sha256_bytes(data),
        f"packet {name} SHA-256 mismatch",
    )
    _require(record.get("bytes") == len(data), f"packet {name} byte count mismatch")


def inspect_preacquisition_next_action_packet(
    packet_path: str | Path,
) -> dict[str, Any]:
    """Validate packet structure and return its exact retained contents."""

    source = Path(packet_path)
    _assert_no_symlink_components(source, name="next-action packet")
    _require(source.is_file(), "next-action packet is missing")
    source = source.resolve(strict=True)
    digest_before, bytes_before = _sha256_file(source)
    try:
        with ZipFile(source, mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            _require(
                tuple(names) == _PACKET_MEMBERS,
                "next-action packet members differ from the registered order",
            )
            _require(len(set(names)) == len(names), "next-action packet has duplicates")
            member_bytes: dict[str, bytes] = {}
            for info in infos:
                _require(not info.is_dir(), "next-action packet contains a directory")
                _require(
                    info.compress_type == ZIP_STORED,
                    "next-action packet member compression changed",
                )
                _require(
                    info.flag_bits & 0x1 == 0,
                    "next-action packet contains an encrypted member",
                )
                _require(
                    0 <= info.file_size <= _MAX_MEMBER_BYTES,
                    "next-action packet member exceeds the size limit",
                )
                value = archive.read(info)
                _require(
                    len(value) == info.file_size,
                    "next-action packet member size changed while reading",
                )
                member_bytes[info.filename] = value
    except BadZipFile as error:
        raise ValueError("next-action packet is not a valid ZIP archive") from error

    digest_after, bytes_after = _sha256_file(source)
    _require(
        (digest_after, bytes_after) == (digest_before, bytes_before),
        "next-action packet changed during validation",
    )

    decision_bytes = member_bytes[_DECISION_MEMBER]
    instructions_bytes = member_bytes[_INSTRUCTIONS_MEMBER]
    manifest = _json_mapping(member_bytes[_MANIFEST_MEMBER], name="packet manifest")
    decision = _json_mapping(decision_bytes, name="packet decision")
    _validate_decision(decision)

    _require(
        manifest.get("schema_version") == NEXT_ACTION_PACKET_SCHEMA_VERSION,
        "unsupported next-action packet schema",
    )
    _require(
        manifest.get("artifact_kind") == NEXT_ACTION_PACKET_ARTIFACT_KIND,
        "unexpected next-action packet artifact kind",
    )
    _require(
        manifest.get("packet_id") == next_action_packet_id(manifest),
        "next-action packet ID mismatch",
    )
    _require(
        manifest.get("decision_evidence_sha256") == decision["evidence_sha256"],
        "packet manifest decision evidence differs",
    )
    _require(
        manifest.get("decision_status_sha256") == decision["status_sha256"],
        "packet manifest decision status differs",
    )
    _require(
        manifest.get("action_identity") == _action_identity(decision),
        "packet manifest action identity differs",
    )
    for field in (
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_plan_id",
        "preacquisition_amendment_sha256",
    ):
        _require(manifest.get(field) == decision.get(field), f"packet {field} differs")
    _require(
        manifest.get("target_outcomes_used") is False,
        "packet manifest admits target outcomes",
    )
    _require(
        manifest.get("target_outcomes_permitted") is False,
        "packet manifest permits target outcomes",
    )
    _require(
        manifest.get("changes_registered_method") is False,
        "packet manifest permits a registered-method change",
    )
    _member_record(
        manifest,
        name="decision",
        expected_path=_DECISION_MEMBER,
        data=decision_bytes,
    )
    _member_record(
        manifest,
        name="instructions",
        expected_path=_INSTRUCTIONS_MEMBER,
        data=instructions_bytes,
    )
    _require(
        instructions_bytes == _instructions_bytes(decision),
        "human instructions differ from the machine decision",
    )

    return {
        "path": str(source),
        "packet_file_sha256": digest_after,
        "packet_file_bytes": bytes_after,
        "manifest": manifest,
        "decision": decision,
        "decision_bytes": decision_bytes,
        "instructions_bytes": instructions_bytes,
    }


def write_preacquisition_next_action_packet(
    packet_path: str | Path,
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish one exactly-once operator handoff packet."""

    packet_bytes, manifest = build_preacquisition_next_action_packet_bytes(decision)
    target = Path(packet_path)

    def writer(handle: BinaryIO) -> None:
        handle.write(packet_bytes)

    atomic_write_binary(
        target,
        writer,
        overwrite=False,
        validate=lambda path: inspect_preacquisition_next_action_packet(path),
    )
    packet_sha256, packet_size = _sha256_file(target)
    return {
        "schema_version": NEXT_ACTION_PACKET_SCHEMA_VERSION,
        "artifact_kind": NEXT_ACTION_PACKET_ARTIFACT_KIND,
        "packet_path": str(target.resolve(strict=True)),
        "packet_file_sha256": packet_sha256,
        "packet_file_bytes": packet_size,
        "packet_id": manifest["packet_id"],
        "decision_evidence_sha256": manifest["decision_evidence_sha256"],
        "action_identity": manifest["action_identity"],
        "valid": decision.get("valid") is True,
        "ready": decision.get("ready") is True,
        "complete": True,
        "passed": decision.get("valid") is True,
        "target_outcomes_used": False,
    }


def next_action_packet_validation_evidence_sha256(
    values: Mapping[str, Any],
) -> str:
    """Return the mount-independent digest of one packet validation."""

    payload = deepcopy(dict(values))
    for field in (
        "repository_root",
        "dataset_root",
        "packet_path",
        "status_sha256",
    ):
        payload.pop(field, None)
    return _canonical_sha256(payload, omitted="evidence_sha256")


def next_action_packet_validation_status_sha256(
    values: Mapping[str, Any],
) -> str:
    """Return the digest of one exact host-local packet validation."""

    return _canonical_sha256(values, omitted="status_sha256")


def validate_preacquisition_next_action_packet(
    repository_root: str | Path,
    dataset_root: str | Path,
    packet_path: str | Path,
) -> dict[str, Any]:
    """Require packet bytes and human instructions to match the current action."""

    repository = Path(repository_root).resolve()
    dataset = Path(dataset_root).resolve()
    inspected = inspect_preacquisition_next_action_packet(packet_path)
    decision = inspected["decision"]
    assert isinstance(decision, Mapping)
    decision_bytes = inspected["decision_bytes"]
    assert isinstance(decision_bytes, bytes)

    with tempfile.TemporaryDirectory(prefix="causal4d-next-action-packet-") as directory:
        decision_path = Path(directory) / _DECISION_MEMBER
        decision_path.write_bytes(decision_bytes)
        freshness = validate_preacquisition_next_action_report(
            repository,
            dataset,
            decision_path,
        )

    manifest = inspected["manifest"]
    assert isinstance(manifest, Mapping)
    _require(
        freshness.get("decision_evidence_sha256")
        == manifest.get("decision_evidence_sha256"),
        "packet freshness decision evidence differs",
    )
    report: dict[str, Any] = {
        "schema_version": NEXT_ACTION_PACKET_VALIDATION_SCHEMA_VERSION,
        "artifact_kind": NEXT_ACTION_PACKET_VALIDATION_ARTIFACT_KIND,
        "protocol_id": decision["protocol_id"],
        "protocol_design_sha256": decision["protocol_design_sha256"],
        "preacquisition_plan_id": decision["preacquisition_plan_id"],
        "preacquisition_amendment_sha256": decision[
            "preacquisition_amendment_sha256"
        ],
        "repository_root": str(repository),
        "dataset_root": str(dataset),
        "packet_path": inspected["path"],
        "packet_file_sha256": inspected["packet_file_sha256"],
        "packet_file_bytes": inspected["packet_file_bytes"],
        "packet_id": manifest["packet_id"],
        "decision_evidence_sha256": decision["evidence_sha256"],
        "decision_status_sha256": decision["status_sha256"],
        "current_evidence_sha256": freshness["current_evidence_sha256"],
        "current_status_sha256": freshness["current_status_sha256"],
        "action_identity": manifest["action_identity"],
        "exact_member_set_verified": True,
        "member_hashes_verified": True,
        "human_instructions_match_decision": True,
        "decision_current": True,
        "file_hashes_verified": True,
        "safe_to_execute": freshness["safe_to_execute"] is True,
        "changes_registered_method": False,
        "target_outcomes_used": False,
        "valid": True,
        "complete": True,
        "passed": True,
    }
    report["evidence_sha256"] = next_action_packet_validation_evidence_sha256(report)
    report["status_sha256"] = next_action_packet_validation_status_sha256(report)
    return report


def write_preacquisition_next_action_packet_validation(
    path: str | Path,
    report: Mapping[str, Any],
) -> Path:
    """Atomically write one packet freshness and integrity report."""

    output = Path(path)
    atomic_write_json(output, dict(report))
    return output


__all__ = [
    "NEXT_ACTION_PACKET_ARTIFACT_KIND",
    "NEXT_ACTION_PACKET_SCHEMA_VERSION",
    "NEXT_ACTION_PACKET_VALIDATION_ARTIFACT_KIND",
    "NEXT_ACTION_PACKET_VALIDATION_SCHEMA_VERSION",
    "build_preacquisition_next_action_packet_bytes",
    "build_preacquisition_next_action_packet_manifest",
    "inspect_preacquisition_next_action_packet",
    "next_action_packet_id",
    "next_action_packet_validation_evidence_sha256",
    "next_action_packet_validation_status_sha256",
    "validate_preacquisition_next_action_packet",
    "write_preacquisition_next_action_packet",
    "write_preacquisition_next_action_packet_validation",
]
