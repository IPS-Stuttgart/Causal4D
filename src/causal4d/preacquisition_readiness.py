"""Fail-closed readiness decision for the registered physical experiment."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json
from causal4d.operator_identity_integration import (
    seal_registered_preacquisition_gate as seal_preacquisition_gate,
    validate_gate_file_operator_identity,
    validate_preacquisition_identity_bindings,
)
from causal4d.operator_registry import load_operator_registry_prerequisite
from causal4d.preacquisition_gate_validation import _validate_gate_file
from causal4d.preacquisition_protocol_v5 import governance_allows_single_operator
from causal4d.preacquisition_readiness_contracts import (
    GATE_EVIDENCE_ARTIFACT_KIND as GATE_EVIDENCE_ARTIFACT_KIND,
    GATE_EVIDENCE_SCHEMA_VERSION as GATE_EVIDENCE_SCHEMA_VERSION,
    GATE_PATHS,
    OPERATIONAL_GATES_BEFORE_FREEZE,
    READINESS_ARTIFACT_KIND,
    READINESS_SCHEMA_VERSION,
    SOURCE_PANEL_MANIFEST_PATH as SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    _parse_utc_timestamp,
    gate_evidence_sha256 as gate_evidence_sha256,
    gate_evidence_template,
    load_registered_preacquisition_chain,
    readiness_evidence_sha256,
    readiness_status_sha256,
    source_panel_execution_manifest_template,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status
from causal4d.registered_real_analysis_prerequisite import (
    validate_registered_real_analysis_prerequisite,
)


def _publish_template(
    path: Path,
    payload: Mapping[str, Any],
    relative: str,
    *,
    created: list[str],
    existing: list[str],
) -> None:
    try:
        atomic_write_json(path, dict(payload), overwrite=False)
    except FileExistsError:
        existing.append(relative)
    else:
        created.append(relative)


def scaffold_preacquisition_readiness(
    repository_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Write incomplete gate templates without overwriting operator evidence."""

    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = Path(dataset_root)
    created: list[str] = []
    existing: list[str] = []
    for gate_id, relative in GATE_PATHS.items():
        path = root / relative
        _publish_template(
            path,
            gate_evidence_template(gate_id, protocol, v2, v4),
            relative,
            created=created,
            existing=existing,
        )
    source_executions = v2["preacquisition_signature_panel"]["executions"]
    for execution in source_executions:
        relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
            execution_id=execution["execution_id"]
        )
        path = root / relative
        _publish_template(
            path,
            source_panel_execution_manifest_template(execution, protocol, v4),
            relative,
            created=created,
            existing=existing,
        )
    return {
        "passed": True,
        "dataset_root": str(root.resolve()),
        "created": created,
        "existing": existing,
    }


def _identity_bound_gate_results(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
    dataset_root: Path,
    prerequisites: Mapping[str, Mapping[str, Any]],
    *,
    registry: Mapping[str, Any] | None,
    verify_file_hashes: bool,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    for gate_id, relative in GATE_PATHS.items():
        path = dataset_root / relative
        result = _validate_gate_file(
            gate_id,
            path,
            protocol=protocol,
            v2=v2,
            v4=v4,
            dataset_root=dataset_root,
            prerequisites=prerequisites,
            verify_file_hashes=verify_file_hashes,
        )
        result["identity_pending"] = False
        if result["valid"]:
            if registry is None:
                result["identity_pending"] = True
                result["valid"] = False
                result["error"] = "operator registry is unavailable"
            else:
                try:
                    result.update(
                        validate_gate_file_operator_identity(
                            gate_id,
                            path,
                            registry,
                            prerequisites,
                            preacquisition=v4,
                        )
                    )
                except (OSError, KeyError, TypeError, ValueError) as error:
                    message = str(error).strip()
                    result["valid"] = False
                    result["error"] = (
                        f"{type(error).__name__}: {message}"
                        if message
                        else type(error).__name__
                    )
        results[gate_id] = result
    return results


def evaluate_preacquisition_readiness(
    protocol: Mapping[str, Any],
    v2: Mapping[str, Any],
    v4: Mapping[str, Any],
    dataset_root: str | Path,
    real_status: Mapping[str, Any],
    *,
    verify_file_hashes: bool,
    require_registered_analysis: bool = False,
) -> dict[str, Any]:
    """Derive all collection gates from validated artifacts and chronology."""

    root = Path(dataset_root)
    prerequisites = {
        str(name): dict(value) for name, value in real_status["prerequisites"].items()
    }
    if require_registered_analysis:
        prerequisites["registered_analysis"] = (
            validate_registered_real_analysis_prerequisite(
                protocol,
                root,
                prerequisites["method_freeze"],
            )
        )
    operator_registry_result, operator_registry = load_operator_registry_prerequisite(
        protocol, v4, root
    )
    prerequisites["operator_registry"] = operator_registry_result
    if governance_allows_single_operator(v4):
        prerequisites["operator_identity_bindings"] = (
            validate_preacquisition_identity_bindings(
                root,
                operator_registry,
                preacquisition=v4,
            )
        )
    else:
        prerequisites["operator_identity_bindings"] = (
            validate_preacquisition_identity_bindings(
                root,
                operator_registry,
            )
        )
    gate_results = _identity_bound_gate_results(
        protocol,
        v2,
        v4,
        root,
        prerequisites,
        registry=operator_registry,
        verify_file_hashes=verify_file_hashes,
    )

    prerequisite_names = (
        "dataset_protocol",
        "acquisition_schedule",
        "object_registration",
        "slip_pilot",
        "timebase_calibration",
        "contact_registration",
        "operator_registry",
        "operator_identity_bindings",
        "method_freeze",
        "method_freeze_validation",
    )
    if require_registered_analysis:
        prerequisite_names = (*prerequisite_names, "registered_analysis")
    missing_prerequisites = [
        name
        for name in prerequisite_names
        if not prerequisites[name].get("present")
        or prerequisites[name].get("template") is True
    ]
    malformed_prerequisites = [
        name
        for name in prerequisite_names
        if prerequisites[name].get("present")
        and prerequisites[name].get("template") is not True
        and not prerequisites[name].get("valid")
    ]
    missing_or_template_gates = [
        gate_id
        for gate_id, result in gate_results.items()
        if not result["present"]
        or result["template"]
        or result.get("identity_pending") is True
    ]
    malformed_gates = [
        gate_id
        for gate_id, result in gate_results.items()
        if result["present"]
        and not result["template"]
        and result.get("identity_pending") is not True
        and not result["valid"]
    ]

    collection_counts = {
        name: real_status.get(name, 0)
        for name in (
            "manifest_executions",
            "acquired_executions",
            "validated_executions",
        )
    }
    for name, value in collection_counts.items():
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative integer")
    manifest_count = collection_counts["manifest_executions"]
    acquired_count = collection_counts["acquired_executions"]
    validated_count = collection_counts["validated_executions"]
    collection_not_started = manifest_count == acquired_count == validated_count == 0

    chronology_blockers: list[str] = []
    freeze = prerequisites["method_freeze"]
    attestation = prerequisites["method_freeze_validation"]
    registry = prerequisites["operator_registry"]
    registry_sealed_at = None
    if registry.get("valid"):
        registry_sealed_at = _parse_utc_timestamp(
            registry.get("sealed_at_utc"),
            name="operator registry sealed_at_utc",
        )
        for gate_id, result in gate_results.items():
            approved_at = result.get("approved_at_utc")
            if approved_at is not None:
                approved = _parse_utc_timestamp(
                    approved_at,
                    name=f"{gate_id} approved_at_utc",
                )
                if registry_sealed_at > approved:
                    chronology_blockers.append(
                        f"operator_registry_postdates_gate:{gate_id}"
                    )

    if freeze.get("valid"):
        frozen_at = _parse_utc_timestamp(
            freeze.get("frozen_at_utc"), name="method freeze frozen_at_utc"
        )
        if registry_sealed_at is not None and registry_sealed_at > frozen_at:
            chronology_blockers.append("operator_registry_postdates_method_freeze")
        for gate_id in OPERATIONAL_GATES_BEFORE_FREEZE:
            approved_at = gate_results[gate_id].get("approved_at_utc")
            if approved_at is not None:
                approved = _parse_utc_timestamp(
                    approved_at, name=f"{gate_id} approved_at_utc"
                )
                if approved > frozen_at:
                    chronology_blockers.append(
                        f"method_freeze_precedes_operational_gate:{gate_id}"
                    )
        software_approved_at = gate_results["software_environment_locked"].get(
            "approved_at_utc"
        )
        if software_approved_at is not None:
            approved = _parse_utc_timestamp(
                software_approved_at,
                name="software_environment_locked approved_at_utc",
            )
            if approved < frozen_at:
                chronology_blockers.append(
                    "software_environment_predates_method_freeze"
                )
            if attestation.get("valid"):
                verified_at = _parse_utc_timestamp(
                    attestation.get("verified_at_utc"),
                    name="method freeze verified_at_utc",
                )
                if approved < verified_at:
                    chronology_blockers.append(
                        "software_environment_predates_freeze_attestation"
                    )
    if registry_sealed_at is not None and attestation.get("valid"):
        verified_at = _parse_utc_timestamp(
            attestation.get("verified_at_utc"),
            name="method freeze verified_at_utc",
        )
        if registry_sealed_at > verified_at:
            chronology_blockers.append("operator_registry_postdates_freeze_attestation")

    blockers: list[str] = []
    blockers.extend(f"prerequisite:{name}" for name in missing_prerequisites)
    blockers.extend(f"prerequisite_invalid:{name}" for name in malformed_prerequisites)
    blockers.extend(f"gate:{name}" for name in missing_or_template_gates)
    blockers.extend(f"gate_invalid:{name}" for name in malformed_gates)
    blockers.extend(chronology_blockers)
    if not verify_file_hashes:
        blockers.append("file_hashes_not_verified")
    if not collection_not_started:
        blockers.append("confirmatory_collection_already_started")

    flags = {
        "signature_panel_complete": gate_results["signature_panel_complete"]["valid"],
        "contact_registration_approved": prerequisites["contact_registration"].get(
            "valid", False
        ),
        "slip_pilot_passed_or_versioned_out": prerequisites["slip_pilot"].get(
            "valid", False
        ),
        "operator_identities_registered": prerequisites["operator_registry"].get(
            "valid", False
        ),
        "operator_approval_bindings_valid": prerequisites[
            "operator_identity_bindings"
        ].get("valid", False),
        "actuator_sync_passed": gate_results["actuator_sync_passed"]["valid"],
        "support_registration_passed": gate_results["support_registration_passed"][
            "valid"
        ],
        "end_to_end_dry_run_passed": gate_results["end_to_end_dry_run_passed"]["valid"],
        "analysis_code_frozen": bool(
            prerequisites["method_freeze"].get("valid")
            and prerequisites["method_freeze_validation"].get("valid")
            and prerequisites["operator_identity_bindings"].get("valid")
        ),
        "software_environment_locked": gate_results["software_environment_locked"][
            "valid"
        ],
    }
    if require_registered_analysis:
        flags["primary_analysis_registered"] = prerequisites["registered_analysis"].get(
            "valid", False
        )
    ready = not blockers and all(flags.values())
    flags["first_confirmatory_execution_allowed"] = ready
    valid = (
        not malformed_prerequisites and not malformed_gates and not chronology_blockers
    )
    valid = bool(valid and collection_not_started)

    status: dict[str, Any] = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "artifact_kind": READINESS_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "governance": {
            "mode": v4.get("governance", {}).get(
                "mode", "independent_two_person_v4"
            ),
            "single_operator_allowed": v4.get("governance", {}).get(
                "single_operator_allowed", False
            ),
            "independent_verifier_required": v4.get("governance", {}).get(
                "independent_verifier_required", True
            ),
            "independent_preacquisition_attestation_claimed": v4.get(
                "governance", {}
            ).get("independent_preacquisition_attestation_claimed", True),
            "self_attestation_required": v4.get("governance", {}).get(
                "self_attestation_required", False
            ),
        },
        "dataset_root": str(root.resolve()),
        "verify_file_hashes": verify_file_hashes,
        "registered_analysis_required": require_registered_analysis,
        "prerequisites": {
            name: dict(prerequisites[name]) for name in prerequisite_names
        },
        "operational_gates": gate_results,
        "collection_gate": flags,
        "confirmatory_collection": {
            "manifest_executions": manifest_count,
            "acquired_executions": acquired_count,
            "validated_executions": validated_count,
            "not_started": collection_not_started,
        },
        "missing_prerequisites": missing_prerequisites,
        "malformed_prerequisites": malformed_prerequisites,
        "missing_or_template_gates": missing_or_template_gates,
        "malformed_gates": malformed_gates,
        "chronology_blockers": chronology_blockers,
        "blockers": blockers,
        "valid": valid,
        "ready": ready,
        "passed": ready,
    }
    status["evidence_sha256"] = readiness_evidence_sha256(status)
    status["status_sha256"] = readiness_status_sha256(status)
    return status


def build_preacquisition_readiness(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Load the registered chain and derive a final collection decision."""

    protocol, v2, _, v4 = load_registered_preacquisition_chain(repository_root)
    real_status = build_real_evidence_status(
        protocol,
        dataset_root,
        repository_root=repository_root,
        verify_file_hashes=verify_file_hashes,
    )
    return evaluate_preacquisition_readiness(
        protocol,
        v2,
        v4,
        dataset_root,
        real_status,
        verify_file_hashes=verify_file_hashes,
        require_registered_analysis=True,
    )


def write_preacquisition_readiness(
    path: str | Path,
    status: Mapping[str, Any],
) -> Path:
    """Atomically write one deterministic readiness snapshot."""

    output = Path(path)
    atomic_write_json(output, dict(status))
    return output


__all__ = [
    "GATE_EVIDENCE_ARTIFACT_KIND",
    "GATE_EVIDENCE_SCHEMA_VERSION",
    "GATE_PATHS",
    "READINESS_ARTIFACT_KIND",
    "READINESS_SCHEMA_VERSION",
    "SOURCE_PANEL_MANIFEST_PATH",
    "SOURCE_PANEL_MANIFEST_TEMPLATE_PATH",
    "build_preacquisition_readiness",
    "evaluate_preacquisition_readiness",
    "gate_evidence_sha256",
    "gate_evidence_template",
    "load_registered_preacquisition_chain",
    "readiness_evidence_sha256",
    "readiness_status_sha256",
    "scaffold_preacquisition_readiness",
    "seal_preacquisition_gate",
    "source_panel_execution_manifest_template",
    "validate_preacquisition_identity_bindings",
    "write_preacquisition_readiness",
]
