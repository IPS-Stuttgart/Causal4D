"""Identity-policy integration for acquisition approvals and freeze evidence."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.operator_registry import (
    OPERATOR_REGISTRY_PATH,
    load_registered_operator_registry,
    validate_attestation_operator_identities,
    validate_gate_approver_identity,
    validate_method_freeze_operator_identity,
)
from causal4d.preacquisition_protocol_v5 import governance_allows_single_operator


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing")
    _require(not path.is_symlink(), f"{name} must not be a symlink")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity_error(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _approval(value: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    approval = value.get("approval")
    _require(isinstance(approval, Mapping), f"{name} approval is invalid")
    _require(approval.get("approved") is True, f"{name} approval is incomplete")
    return approval


def validate_gate_file_operator_identity(
    gate_id: str,
    path: str | Path,
    registry: Mapping[str, Any],
    prerequisites: Mapping[str, Mapping[str, Any]],
    *,
    preacquisition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate the identity behind one otherwise-valid operational gate."""

    gate_path = Path(path)
    gate = _read_json_mapping(gate_path, name=f"{gate_id} gate")
    approval = _approval(gate, name=f"{gate_id} gate")
    freezer_digest = prerequisites.get("method_freeze", {}).get(
        "freezer_person_identity_sha256"
    )
    if gate_id == "software_environment_locked" and freezer_digest is None:
        method_freeze = _read_json_mapping(
            gate_path.parent.parent / "method_freeze.json",
            name="method freeze",
        )
        freezer = validate_method_freeze_operator_identity(method_freeze, registry)
        freezer_digest = freezer["person_identity_sha256"]
    approver = validate_gate_approver_identity(
        gate_id,
        approval.get("approver_id"),
        approval.get("approved_at_utc"),
        registry,
        freezer_person_identity_sha256=(
            str(freezer_digest) if freezer_digest is not None else None
        ),
        allow_software_environment_self_approval=(
            preacquisition is not None
            and governance_allows_single_operator(preacquisition)
        ),
    )
    return {
        "approver_operator_id": str(approver["operator_id"]),
        "approver_person_identity_sha256": str(approver["person_identity_sha256"]),
    }


def validate_method_freeze_identity_evidence(
    method_freeze: Mapping[str, Any],
    attestation: Mapping[str, Any] | None,
    registry: Mapping[str, Any],
    *,
    preacquisition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the freezer and the policy-compatible registered attester."""

    freezer = validate_method_freeze_operator_identity(method_freeze, registry)
    result = {
        "freezer_operator_id": str(freezer["operator_id"]),
        "freezer_person_identity_sha256": str(freezer["person_identity_sha256"]),
    }
    if attestation is not None:
        _, verifier = validate_attestation_operator_identities(
            method_freeze,
            attestation,
            registry,
            allow_self_attestation=(
                preacquisition is not None
                and governance_allows_single_operator(preacquisition)
            ),
        )
        result.update(
            {
                "verifier_operator_id": str(verifier["operator_id"]),
                "verifier_person_identity_sha256": str(
                    verifier["person_identity_sha256"]
                ),
                "attestation_independent": (
                    verifier["person_identity_sha256"]
                    != freezer["person_identity_sha256"]
                ),
            }
        )
    return result


def validate_preacquisition_identity_bindings(
    dataset_root: str | Path,
    registry: Mapping[str, Any] | None,
    *,
    preacquisition: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate every person identity that governs acquisition evidence.

    This derived prerequisite covers the method freezer, the registered freeze
    attester, timebase and contact-registration approvers, and all operational
    readiness-gate approvers. Missing or still-template evidence is reported as
    incomplete; completed evidence with an unknown, inactive, role-incompatible,
    postdated, or policy-incompatible identity fails closed.
    """

    from causal4d.preacquisition_readiness_contracts import GATE_PATHS

    root = Path(dataset_root)
    source_paths = {
        "method_freeze": root / "method_freeze.json",
        "method_freeze_validation": root / "method_freeze_validation.json",
        "timebase_calibration": root / "timebase_calibration.json",
        "contact_registration": root / "contact_registration.json",
        **{
            f"operational_gate:{gate_id}": root / relative
            for gate_id, relative in GATE_PATHS.items()
        },
    }
    missing = [name for name, path in source_paths.items() if not path.is_file()]
    result: dict[str, Any] = {
        "path": str((root / OPERATOR_REGISTRY_PATH).resolve()),
        "present": not missing,
        "template": bool(missing),
        "valid": False,
        "passed": False,
        "error": None,
        "missing_sources": missing,
    }
    if missing:
        result["error"] = "operator identity bindings are incomplete: " + ", ".join(
            missing
        )
        return result
    if registry is None:
        result["error"] = "operator registry is unavailable"
        return result

    try:
        method_freeze = _read_json_mapping(
            source_paths["method_freeze"],
            name="method freeze",
        )
        attestation = _read_json_mapping(
            source_paths["method_freeze_validation"],
            name="method freeze attestation",
        )
        timebase = _read_json_mapping(
            source_paths["timebase_calibration"],
            name="timebase calibration",
        )
        contact = _read_json_mapping(
            source_paths["contact_registration"],
            name="contact registration",
        )
        gates = {
            gate_id: _read_json_mapping(
                root / relative,
                name=f"{gate_id} gate",
            )
            for gate_id, relative in GATE_PATHS.items()
        }

        complete = bool(
            method_freeze.get("status") == "sealed"
            and attestation.get("validation_passed") is True
            and timebase.get("status") == "approved"
            and isinstance(timebase.get("approval"), Mapping)
            and timebase["approval"].get("approved") is True
            and isinstance(contact.get("approval"), Mapping)
            and contact["approval"].get("approved") is True
            and all(
                gate.get("status") == "passed"
                and isinstance(gate.get("approval"), Mapping)
                and gate["approval"].get("approved") is True
                for gate in gates.values()
            )
        )
        if not complete:
            result["template"] = True
            result["error"] = "operator-governed evidence is not fully sealed"
            return result

        identity = validate_method_freeze_identity_evidence(
            method_freeze,
            attestation,
            registry,
            preacquisition=preacquisition,
        )
        freezer_digest = identity["freezer_person_identity_sha256"]
        approvals: dict[str, dict[str, str]] = {}
        for approval_id, artifact in (
            ("timebase_calibration", timebase),
            ("contact_registration", contact),
        ):
            approval = _approval(artifact, name=approval_id)
            approver = validate_gate_approver_identity(
                approval_id,
                approval.get("approver_id"),
                approval.get("approved_at_utc"),
                registry,
            )
            approvals[approval_id] = {
                "operator_id": str(approver["operator_id"]),
                "person_identity_sha256": str(approver["person_identity_sha256"]),
            }
        for gate_id, gate in gates.items():
            approval = _approval(gate, name=f"{gate_id} gate")
            approver = validate_gate_approver_identity(
                gate_id,
                approval.get("approver_id"),
                approval.get("approved_at_utc"),
                registry,
                freezer_person_identity_sha256=(
                    str(freezer_digest)
                    if gate_id == "software_environment_locked"
                    else None
                ),
                allow_software_environment_self_approval=(
                    gate_id == "software_environment_locked"
                    and preacquisition is not None
                    and governance_allows_single_operator(preacquisition)
                ),
            )
            approvals[f"operational_gate:{gate_id}"] = {
                "operator_id": str(approver["operator_id"]),
                "person_identity_sha256": str(approver["person_identity_sha256"]),
            }

        source_sha256 = {
            name: _sha256_file(path) for name, path in sorted(source_paths.items())
        }
        result.update(
            {
                **identity,
                "operator_registry_artifact_sha256": str(registry["artifact_sha256"]),
                "approval_bindings": approvals,
                "source_sha256": source_sha256,
                "software_environment_approval_policy": (
                    "registered_self_approval_disclosed"
                    if preacquisition is not None
                    and governance_allows_single_operator(preacquisition)
                    else "independent_registered_person_distinct_from_method_freezer"
                ),
                "independent_preacquisition_attestation_claimed": not (
                    preacquisition is not None
                    and governance_allows_single_operator(preacquisition)
                ),
                "template": False,
                "valid": True,
                "passed": True,
                "error": None,
                "missing_sources": [],
            }
        )
    except (
        OSError,
        KeyError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        result["template"] = False
        result["error"] = _identity_error(error)
    return result


def seal_registered_preacquisition_gate(
    repository_root: str | Path,
    dataset_root: str | Path,
    gate_id: str,
    *,
    approved_by: str,
    approved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Require a registered, role-compatible approver before sealing a gate."""

    from causal4d.preacquisition_gate_validation import seal_preacquisition_gate
    from causal4d.preacquisition_readiness_contracts import (
        GATE_PATHS,
        load_registered_preacquisition_chain,
    )

    _require(gate_id in GATE_PATHS, f"unknown pre-acquisition gate: {gate_id}")
    gate_path = Path(dataset_root) / GATE_PATHS[gate_id]
    if gate_path.is_file():
        gate = _read_json_mapping(gate_path, name=f"{gate_id} gate")
        _require(gate.get("status") == "template", "gate evidence is already sealed")

    registry_result, registry = load_registered_operator_registry(
        repository_root,
        dataset_root,
    )
    _require(
        registry_result.get("valid") is True and registry is not None,
        str(registry_result.get("error") or "operator registry is invalid"),
    )
    _, _, _, preacquisition = load_registered_preacquisition_chain(repository_root)
    approved_at = approved_at_utc or datetime.now(timezone.utc).isoformat()
    freezer_digest: str | None = None
    if gate_id == "software_environment_locked":
        method_freeze = _read_json_mapping(
            Path(dataset_root) / "method_freeze.json",
            name="method freeze",
        )
        freezer = validate_method_freeze_operator_identity(method_freeze, registry)
        freezer_digest = str(freezer["person_identity_sha256"])
    validate_gate_approver_identity(
        gate_id,
        approved_by,
        approved_at,
        registry,
        freezer_person_identity_sha256=freezer_digest,
        allow_software_environment_self_approval=(
            governance_allows_single_operator(preacquisition)
        ),
    )
    result = seal_preacquisition_gate(
        repository_root,
        dataset_root,
        gate_id,
        approved_by=approved_by,
        approved_at_utc=approved_at,
    )
    result["operator_registry_artifact_sha256"] = registry_result["artifact_sha256"]
    return result


__all__ = [
    "seal_registered_preacquisition_gate",
    "validate_gate_file_operator_identity",
    "validate_method_freeze_identity_evidence",
    "validate_preacquisition_identity_bindings",
]
