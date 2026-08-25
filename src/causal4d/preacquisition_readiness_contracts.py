"""Schemas, templates, and shared utilities for acquisition readiness."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.preacquisition_protocol_v5 import load_v5_chain

READINESS_SCHEMA_VERSION = 2
GATE_EVIDENCE_SCHEMA_VERSION = 2
READINESS_ARTIFACT_KIND = "PreacquisitionReadinessStatus"
GATE_EVIDENCE_ARTIFACT_KIND = "PreacquisitionGateEvidence"

PROTOCOL_PATH = "configs/causal4d/sloth_multi_action_v1.json"
PREACQUISITION_V2_PATH = "configs/causal4d/sloth_preacquisition_v2.json"
PREACQUISITION_V3_PATH = "configs/causal4d/sloth_preacquisition_v3.json"
PREACQUISITION_V4_PATH = "configs/causal4d/sloth_preacquisition_v4.json"
PREACQUISITION_V5_PATH = "configs/causal4d/sloth_preacquisition_v5.json"
MECHANISM_GATE_EVIDENCE_PATH = (
    "runs/causal4d_preacquisition_v4/mechanism_gate_controls.json"
)

GATE_PATHS = {
    "signature_panel_complete": "preacquisition/signature_panel.json",
    "actuator_sync_passed": "preacquisition/actuator_sync.json",
    "support_registration_passed": "preacquisition/support_registration.json",
    "end_to_end_dry_run_passed": "preacquisition/end_to_end_dry_run.json",
    "software_environment_locked": "preacquisition/software_environment.json",
}

SOURCE_PANEL_MANIFEST_PATH = (
    "preacquisition/source_panel/executions/{execution_id}/manifest.json"
)
SOURCE_PANEL_MANIFEST_TEMPLATE_PATH = (
    "preacquisition/source_panel/executions/{execution_id}/manifest.template.json"
)

REQUIRED_DRY_RUN_STAGES = (
    "synchronized_acquisition",
    "observation_prefix_build",
    "intervention_abduction",
    "held_out_prediction",
    "artifact_hash_validation",
    "status_generation",
)
OPERATIONAL_GATES_BEFORE_FREEZE = (
    "signature_panel_complete",
    "actuator_sync_passed",
    "support_registration_passed",
    "end_to_end_dry_run_passed",
)
_SHA40 = frozenset("0123456789abcdef")
_SHA64 = _SHA40


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_sha256(values: Mapping[str, Any], *, omitted_field: str) -> str:
    payload = deepcopy(dict(values))
    payload.pop(omitted_field, None)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def gate_evidence_sha256(values: Mapping[str, Any]) -> str:
    """Return the digest that seals one operational gate record."""

    return _canonical_sha256(values, omitted_field="artifact_sha256")


def readiness_status_sha256(values: Mapping[str, Any]) -> str:
    """Return the digest that binds one exact, host-local readiness snapshot."""

    return _canonical_sha256(values, omitted_field="status_sha256")


def readiness_evidence_sha256(values: Mapping[str, Any]) -> str:
    """Return a mount-point-independent digest of the logical readiness evidence."""

    payload = deepcopy(dict(values))
    for field in ("dataset_root", "evidence_sha256", "status_sha256"):
        payload.pop(field, None)
    for section_name in ("prerequisites", "operational_gates"):
        section = payload.get(section_name)
        if not isinstance(section, Mapping):
            continue
        normalized: dict[str, Any] = {}
        for key, value in section.items():
            if isinstance(value, Mapping):
                record = deepcopy(dict(value))
                record.pop("path", None)
                normalized[str(key)] = record
            else:
                normalized[str(key)] = deepcopy(value)
        payload[section_name] = normalized
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _is_hex_digest(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in _SHA64 for character in value)
    )


def _safe_relative_path(value: Any, *, name: str) -> Path:
    _require(isinstance(value, str) and bool(value), f"{name} path is missing")
    path = Path(value)
    _require(
        not path.is_absolute() and ".." not in path.parts,
        f"{name} path is unsafe",
    )
    return path


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


def _object_without_duplicate_json_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(
            key not in value,
            f"duplicate JSON object key is forbidden: {key!r}",
        )
        value[key] = item
    return value


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_object_without_duplicate_json_keys,
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


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


def _finite_number(value: Any, *, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _resolved_dataset_file(
    dataset_root: Path,
    relative: Path,
    *,
    name: str,
) -> Path:
    root = dataset_root.resolve(strict=True)
    candidate = dataset_root / relative
    cursor = dataset_root
    for part in relative.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), f"{name} path contains a symlink")
    _require(candidate.is_file(), f"{name} file is missing: {relative.as_posix()}")
    resolved = candidate.resolve(strict=True)
    _require(
        resolved.is_relative_to(root),
        f"{name} path escapes the dataset root",
    )
    return resolved


def _validate_descriptor(
    dataset_root: Path,
    descriptor: Mapping[str, Any],
    *,
    name: str,
    verify_file_hashes: bool,
) -> str:
    relative = _safe_relative_path(descriptor.get("path"), name=name)
    _require(_is_hex_digest(descriptor.get("sha256"), 64), f"{name} SHA-256 is invalid")
    byte_count = descriptor.get("bytes")
    _require(
        isinstance(byte_count, int)
        and not isinstance(byte_count, bool)
        and byte_count >= 0,
        f"{name} byte count is invalid",
    )
    if verify_file_hashes:
        path = _resolved_dataset_file(dataset_root, relative, name=name)
        digest, size = _sha256_file(path)
        _require(digest == descriptor["sha256"], f"{name} checksum mismatch")
        _require(size == byte_count, f"{name} byte count mismatch")
    return relative.as_posix()


def _validate_descriptor_list(
    dataset_root: Path,
    values: Any,
    *,
    name: str,
    verify_file_hashes: bool,
) -> set[str]:
    _require(
        isinstance(values, list) and bool(values), f"{name} must be a nonempty list"
    )
    paths: set[str] = set()
    for index, descriptor in enumerate(values):
        _require(isinstance(descriptor, Mapping), f"{name}[{index}] is invalid")
        path = _validate_descriptor(
            dataset_root,
            descriptor,
            name=f"{name}[{index}]",
            verify_file_hashes=verify_file_hashes,
        )
        _require(path not in paths, f"{name} contains a duplicate path: {path}")
        paths.add(path)
    return paths


def _expected_source_panel(
    v2: Mapping[str, Any],
) -> tuple[list[str], list[str]]:
    panel = v2.get("preacquisition_signature_panel")
    _require(
        isinstance(panel, Mapping),
        "registered source panel is missing",
    )
    raw_executions = panel.get("executions")
    _require(
        isinstance(raw_executions, list),
        "registered source executions are missing",
    )
    execution_ids: list[str] = []
    session_ids: list[str] = []
    for index, execution in enumerate(raw_executions):
        _require(
            isinstance(execution, Mapping),
            f"registered source execution {index} is invalid",
        )
        execution_id = execution.get("execution_id")
        session_id = execution.get("session_id")
        _require(
            isinstance(execution_id, str) and bool(execution_id.strip()),
            f"registered source execution {index} id is invalid",
        )
        _require(
            isinstance(session_id, str) and bool(session_id.strip()),
            f"registered source execution {index} session id is invalid",
        )
        execution_ids.append(execution_id)
        session_ids.append(session_id)
    _require(
        len(execution_ids) == 12,
        "registered source panel must contain 12 executions",
    )
    _require(
        len(set(execution_ids)) == 12,
        "registered source execution ids are not unique",
    )
    _require(
        len(set(session_ids)) == 12,
        "registered source sessions are not independent",
    )
    return execution_ids, session_ids


def source_panel_execution_manifest_template(
    execution: Mapping[str, Any],
    protocol: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one explicitly incomplete source-panel execution record."""

    execution_id = execution.get("execution_id")
    session_id = execution.get("session_id")
    _require(
        isinstance(execution_id, str)
        and bool(execution_id.strip())
        and isinstance(session_id, str)
        and bool(session_id.strip()),
        "source execution ids are missing or invalid",
    )
    return {
        "schema_version": 1,
        "artifact_kind": "SourcePanelExecutionManifest",
        "status": "template",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "execution_id": execution_id,
        "session_id": session_id,
        "fresh_reset_and_fresh_grasp": True,
        "confirmatory_fold_member": False,
        "target_outcomes_used": False,
        "included": None,
        "quality_gate_failures": [],
        "started_at_utc": None,
        "ended_at_utc": None,
        "artifacts": [],
    }


def _template_checks(
    gate_id: str,
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
) -> dict[str, Any]:
    execution_ids, session_ids = _expected_source_panel(v2)
    if gate_id == "signature_panel_complete":
        return {
            "execution_ids": execution_ids,
            "session_ids": session_ids,
            "independent_session_count": 12,
            "source_only": True,
            "manifest_files": {
                identifier: SOURCE_PANEL_MANIFEST_PATH.format(execution_id=identifier)
                for identifier in execution_ids
            },
        }
    if gate_id == "actuator_sync_passed":
        return {
            "commanded_vs_measured_validated": None,
            "hardware_timestamps_authoritative": True,
            "maximum_measured_rgbd_actuator_sync_error_ms": None,
            "maximum_allowed_rgbd_actuator_sync_error_ms": protocol["quality_gates"][
                "maximum_rgbd_actuator_sync_error_ms"
            ],
            "calibration_files": {identifier: None for identifier in execution_ids},
            "calibration_artifact_ids": {},
        }
    if gate_id == "support_registration_passed":
        return {
            "support_geometry_registered": None,
            "gravity_registered": None,
            "quality_gate_passed": None,
            "world_frame_id": None,
            "gravity_vector_mps2": None,
            "registration_closure_error_m": None,
            "maximum_registration_closure_error_m": None,
            "registration_file": None,
        }
    if gate_id == "end_to_end_dry_run_passed":
        return {
            "nonconfirmatory": True,
            "target_outcomes_used": False,
            "frozen_entrypoints_exercised": None,
            "execution_id": None,
            "pipeline_stages": {name: False for name in REQUIRED_DRY_RUN_STAGES},
            "output_manifest": None,
        }
    if gate_id == "software_environment_locked":
        return {
            "method_freeze_sha256": None,
            "method_freeze_validation_sha256": None,
            "causal4d": {
                "commit_sha": None,
                "version": None,
                "distribution": {"path": None, "sha256": None, "bytes": None},
            },
            "bayesian_phystwin": {
                "commit_sha": None,
                "version": None,
                "distribution": {"path": None, "sha256": None, "bytes": None},
            },
            "prob4d": {"used": False, "reason": None},
            "observation_producer": {
                "name": None,
                "version": None,
                "artifact_contract": None,
            },
            "python": {"version": None, "implementation": None, "platform": None},
            "runtime_environment": {
                "resolved_dependency_report": None,
                "execution_backend": None,
                "containerized": None,
                "container_image_digest": None,
                "numpy_version": None,
                "scipy_version": None,
                "torch_version": None,
                "warp_version": None,
                "opencv_version": None,
                "cuda_runtime_version": None,
                "cuda_driver_version": None,
            },
        }
    raise KeyError(gate_id)


def gate_evidence_template(
    gate_id: str,
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicitly incomplete evidence-bound gate template."""

    _require(gate_id in GATE_PATHS, f"unknown pre-acquisition gate: {gate_id}")
    return {
        "schema_version": GATE_EVIDENCE_SCHEMA_VERSION,
        "artifact_kind": GATE_EVIDENCE_ARTIFACT_KIND,
        "gate_id": gate_id,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "status": "template",
        "completed_at_utc": None,
        "locked_before_confirmatory_collection": None,
        "target_outcomes_used": None,
        "checks": _template_checks(gate_id, protocol, v2),
        "evidence": [],
        "approval": {
            "approved": False,
            "approver_id": None,
            "approved_at_utc": None,
        },
        "artifact_sha256": None,
    }


def load_registered_preacquisition_chain(
    repository_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Load and validate the immutable protocol-v2-v3-v4-v5 chain."""

    root = Path(repository_root)
    return load_v5_chain(
        root / PROTOCOL_PATH,
        root / PREACQUISITION_V2_PATH,
        root / PREACQUISITION_V3_PATH,
        root / MECHANISM_GATE_EVIDENCE_PATH,
        root / PREACQUISITION_V4_PATH,
        root / PREACQUISITION_V5_PATH,
    )


__all__ = [
    "GATE_EVIDENCE_ARTIFACT_KIND",
    "GATE_EVIDENCE_SCHEMA_VERSION",
    "GATE_PATHS",
    "MECHANISM_GATE_EVIDENCE_PATH",
    "OPERATIONAL_GATES_BEFORE_FREEZE",
    "PREACQUISITION_V2_PATH",
    "PREACQUISITION_V3_PATH",
    "PREACQUISITION_V4_PATH",
    "PREACQUISITION_V5_PATH",
    "PROTOCOL_PATH",
    "READINESS_ARTIFACT_KIND",
    "READINESS_SCHEMA_VERSION",
    "REQUIRED_DRY_RUN_STAGES",
    "SOURCE_PANEL_MANIFEST_PATH",
    "SOURCE_PANEL_MANIFEST_TEMPLATE_PATH",
    "gate_evidence_sha256",
    "gate_evidence_template",
    "load_registered_preacquisition_chain",
    "readiness_evidence_sha256",
    "readiness_status_sha256",
    "source_panel_execution_manifest_template",
]
