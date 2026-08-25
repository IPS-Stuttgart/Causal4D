"""Fail-closed version-2 evidence contract for the Causal4D real experiment."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.preacquisition_protocol_v5 import governance_allows_single_operator
from causal4d.preacquisition_readiness_contracts import (
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_common import (
    EVIDENCE_STATUS_SCHEMA_VERSION,
    METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION,
    SESSION_MANIFEST_SCHEMA_VERSION,
    TIMEBASE_CALIBRATION_SCHEMA_VERSION,
    _load_json_mapping,
    _parse_utc_timestamp,
    _require,
    _validate_acquisition_schedule,
    _validate_contact_registration_prerequisite,
    _validate_dataset_protocol,
    _validate_object_registration_prerequisite,
    _validate_slip_pilot_prerequisite,
    _validate_timebase_prerequisite,
    timebase_calibration_template,
    validate_timebase_calibration,
)
from causal4d.real_freeze_evidence import (
    _validate_method_freeze_attestation,
    _validate_method_freeze_prerequisites,
    build_method_freeze_validation_attestation,
    method_freeze_validation_attestation_template,
    write_method_freeze_validation_attestation,
)
from causal4d.real_execution_evidence import (
    _execution_status,
    _validate_execution_contract_v2,
    scaffold_real_evidence_v2_templates,
    session_manifest_template,
)
from causal4d.real_protocol import validate_protocol
from causal4d.real_session_evidence import (
    _analysis_readiness,
    _claim_blockers,
    _unexpected_directories,
    _validate_session_manifest,
)


def _preacquisition_chronology(
    prerequisites: Mapping[str, Mapping[str, Any]],
    execution_results: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove that timestamped method prerequisites do not postdate acquisition."""

    starts = [
        result["_started_at"]
        for result in execution_results
        if result.get("validated") and result.get("_started_at") is not None
    ]
    earliest = min(starts) if starts else None
    event_fields = {
        "timebase_calibrated": ("timebase_calibration", "calibrated_at_utc"),
        "timebase_approved": ("timebase_calibration", "approved_at_utc"),
        "contact_registration_approved": (
            "contact_registration",
            "approved_at_utc",
        ),
        "method_frozen": ("method_freeze", "frozen_at_utc"),
        "method_freeze_verified": (
            "method_freeze_validation",
            "verified_at_utc",
        ),
    }
    events: dict[str, str | None] = {}
    blockers: list[str] = []
    for event_name, (prerequisite_name, field_name) in event_fields.items():
        prerequisite = prerequisites[prerequisite_name]
        value = prerequisite.get(field_name) if prerequisite.get("valid") else None
        events[event_name] = str(value) if value is not None else None
        if earliest is not None and value is not None:
            event_time = _parse_utc_timestamp(value, name=event_name)
            if event_time > earliest:
                blockers.append(f"preacquisition_chronology:{event_name}")
    return {
        "passed": not blockers,
        "earliest_execution_started_at_utc": (
            None if earliest is None else earliest.isoformat().replace("+00:00", "Z")
        ),
        "events": events,
        "blockers": blockers,
    }


def build_real_evidence_status(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Summarize acquisition, evidence, analysis, and claim readiness."""

    validate_protocol(protocol)
    root = Path(dataset_root)

    dataset_protocol = _validate_dataset_protocol(protocol, root / "protocol.json")
    acquisition_schedule = _validate_acquisition_schedule(
        protocol, root / "acquisition_schedule.csv"
    )
    object_registration, simple_registration = (
        _validate_object_registration_prerequisite(
            protocol,
            root,
            root / "object_registration.json",
            verify_file_hashes=verify_file_hashes,
        )
    )
    slip_pilot = _validate_slip_pilot_prerequisite(protocol, root / "slip_pilot.json")
    timebase, timebase_payload = _validate_timebase_prerequisite(
        protocol,
        root,
        root / "timebase_calibration.json",
        verify_file_hashes=verify_file_hashes,
    )
    require_single_operator_review = False
    if (
        repository_root is not None
        and (
            Path(repository_root) / "configs/causal4d/sloth_preacquisition_v5.json"
        ).is_file()
    ):
        _, _, _, preacquisition = load_registered_preacquisition_chain(repository_root)
        require_single_operator_review = governance_allows_single_operator(
            preacquisition
        )
    contact_registration, _ = _validate_contact_registration_prerequisite(
        protocol,
        root,
        root / "contact_registration.json",
        simple_registration=simple_registration,
        simple_registration_sha256=object_registration.get("sha256"),
        verify_file_hashes=verify_file_hashes,
        require_single_operator_review=require_single_operator_review,
    )
    method_freeze, method_freeze_attestation, _ = _validate_method_freeze_prerequisites(
        protocol,
        root,
        repository_root=repository_root,
        verify_file_hashes=verify_file_hashes,
    )
    prerequisites = {
        "dataset_protocol": dataset_protocol,
        "acquisition_schedule": acquisition_schedule,
        "object_registration": object_registration,
        "slip_pilot": slip_pilot,
        "timebase_calibration": timebase,
        "contact_registration": contact_registration,
        "method_freeze": method_freeze,
        "method_freeze_validation": method_freeze_attestation,
    }

    executions = sorted(
        protocol["executions"],
        key=lambda value: int(value["acquisition_execution_index"]),
    )
    clock_domain_id = None
    if timebase.get("valid") and timebase_payload is not None:
        clock_domain_id = str(timebase_payload["clock_domain_id"])
    execution_results = [
        _execution_status(
            protocol,
            root,
            execution,
            clock_domain_id=clock_domain_id,
            verify_file_hashes=verify_file_hashes,
        )
        for execution in executions
    ]
    execution_by_id = {
        str(result["execution_id"]): result for result in execution_results
    }

    sessions = sorted(
        protocol["sessions"],
        key=lambda value: int(value["acquisition_session_index"]),
    )
    session_results = [
        _validate_session_manifest(
            protocol,
            session,
            root / "sessions" / str(session["session_id"]) / "session.json",
            execution_results=execution_by_id,
            contact_registration_sha256=contact_registration.get("sha256"),
            timebase_calibration_sha256=timebase.get("sha256"),
            clock_domain_id=clock_domain_id,
        )
        for session in sessions
    ]
    grasp_ids = [
        str(result["grasp_instance_id"])
        for result in session_results
        if result.get("validated")
    ]
    if len(grasp_ids) != len(set(grasp_ids)):
        duplicated = {
            identifier for identifier in grasp_ids if grasp_ids.count(identifier) > 1
        }
        for result in session_results:
            if result.get("grasp_instance_id") in duplicated:
                result["validated"] = False
                result["error"] = (
                    "ValueError: grasp_instance_id is reused across sessions"
                )

    expected_execution_ids = {
        str(execution["execution_id"]) for execution in executions
    }
    expected_session_ids = {str(session["session_id"]) for session in sessions}
    unexpected_executions = _unexpected_directories(
        root / "executions",
        expected_ids=expected_execution_ids,
    )
    unexpected_sessions = _unexpected_directories(
        root / "sessions",
        expected_ids=expected_session_ids,
    )

    specified = len(execution_results)
    manifest_count = sum(
        bool(result["manifest_present"]) for result in execution_results
    )
    acquired = sum(bool(result["acquired"]) for result in execution_results)
    validated = sum(bool(result["validated"]) for result in execution_results)
    included = sum(result["included"] is True for result in execution_results)
    excluded = sum(result["included"] is False for result in execution_results)
    sessions_validated = sum(bool(result["validated"]) for result in session_results)
    accounting_complete = included + excluded == specified

    acquisition_prerequisite_names = {
        "dataset_protocol",
        "acquisition_schedule",
        "object_registration",
        "slip_pilot",
    }
    acquisition_prerequisites_valid = all(
        prerequisites[name]["valid"] for name in acquisition_prerequisite_names
    )
    acquisition_complete = bool(
        acquisition_prerequisites_valid
        and manifest_count == specified
        and acquired == specified
        and validated == specified
        and accounting_complete
        and not unexpected_executions
    )

    chronology = _preacquisition_chronology(prerequisites, execution_results)
    blockers = _claim_blockers(
        prerequisites,
        execution_results,
        session_results,
        unexpected_execution_directories=unexpected_executions,
        unexpected_session_directories=unexpected_sessions,
        verify_file_hashes=verify_file_hashes,
    )
    blockers.extend(chronology["blockers"])
    evidence_complete = not blockers
    analysis = _analysis_readiness(protocol, execution_results)
    claim_ready = evidence_complete
    next_pending = next(
        (
            {
                "execution_id": result["execution_id"],
                "session_id": result["session_id"],
                "acquisition_execution_index": result["acquisition_execution_index"],
            }
            for result in execution_results
            if not result["validated"]
        ),
        None,
    )

    public_execution_results = []
    for result in execution_results:
        public = dict(result)
        public.pop("manifest", None)
        public.pop("_started_at", None)
        public.pop("_ended_at", None)
        public_execution_results.append(public)

    return {
        "schema_version": EVIDENCE_STATUS_SCHEMA_VERSION,
        "protocol_id": str(protocol["protocol_id"]),
        "protocol_design_sha256": str(protocol["design_sha256"]),
        "dataset_root": str(root.resolve()),
        "repository_root": None
        if repository_root is None
        else str(Path(repository_root).resolve()),
        "specified_executions": specified,
        "manifest_executions": manifest_count,
        "acquired_executions": acquired,
        "validated_executions": validated,
        "included_executions": included,
        "excluded_executions": excluded,
        "specified_sessions": len(session_results),
        "validated_sessions": sessions_validated,
        "missing_execution_ids": [
            result["execution_id"]
            for result in execution_results
            if not result["manifest_present"]
        ],
        "incomplete_execution_ids": [
            result["execution_id"]
            for result in execution_results
            if result["manifest_parsed"] and not result["acquired"]
        ],
        "invalid_execution_ids": [
            result["execution_id"]
            for result in execution_results
            if result["manifest_present"]
            and (
                not result["manifest_parsed"]
                or (result["acquired"] and not result["validated"])
            )
        ],
        "missing_session_ids": [
            result["session_id"]
            for result in session_results
            if not result["manifest_present"]
        ],
        "invalid_session_ids": [
            result["session_id"]
            for result in session_results
            if result["manifest_present"] and not result["validated"]
        ],
        "unexpected_execution_directories": unexpected_executions,
        "unexpected_session_directories": unexpected_sessions,
        "next_pending_execution": next_pending,
        "prerequisites": prerequisites,
        "executions": public_execution_results,
        "sessions": session_results,
        "file_hashes_requested": verify_file_hashes,
        "file_hashes_verified": bool(verify_file_hashes and evidence_complete),
        "accounting_complete": accounting_complete,
        "acquisition_complete": acquisition_complete,
        "evidence_complete": evidence_complete,
        "analysis_ready": analysis["analysis_ready"],
        "full_registered_power": analysis["full_registered_power"],
        "analysis_readiness": analysis,
        "preacquisition_chronology": chronology,
        "complete": evidence_complete,
        "claim_ready": claim_ready,
        "passed": claim_ready,
        "blockers": blockers,
    }


def validate_real_dataset_v2(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate the complete version-2 evidence tree and return its status."""

    status = build_real_evidence_status(
        protocol,
        dataset_root,
        repository_root=repository_root,
        verify_file_hashes=verify_files,
    )
    if verify_files:
        _require(
            status["claim_ready"],
            "real evidence is not claim-ready: " + ", ".join(status["blockers"]),
        )
    return status


def write_real_evidence_status(
    path: str | Path,
    status: Mapping[str, Any],
) -> Path:
    """Atomically write one deterministic, human-readable status snapshot."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            dict(status),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    descriptor: int | None = None
    temporary_name: str | None = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            text=True,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output)
        temporary_name = None
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return output


__all__ = [
    "EVIDENCE_STATUS_SCHEMA_VERSION",
    "METHOD_FREEZE_ATTESTATION_SCHEMA_VERSION",
    "SESSION_MANIFEST_SCHEMA_VERSION",
    "TIMEBASE_CALIBRATION_SCHEMA_VERSION",
    "_analysis_readiness",
    "_load_json_mapping",
    "_preacquisition_chronology",
    "_validate_execution_contract_v2",
    "_validate_method_freeze_attestation",
    "_validate_session_manifest",
    "build_method_freeze_validation_attestation",
    "build_real_evidence_status",
    "method_freeze_validation_attestation_template",
    "scaffold_real_evidence_v2_templates",
    "session_manifest_template",
    "timebase_calibration_template",
    "validate_real_dataset_v2",
    "validate_timebase_calibration",
    "write_method_freeze_validation_attestation",
    "write_real_evidence_status",
]
