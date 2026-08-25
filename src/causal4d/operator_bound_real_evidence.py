"""Operator-bound wrapper around the version-2 physical evidence contract."""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from causal4d.operator_identity_integration import (
    validate_method_freeze_identity_evidence,
    validate_preacquisition_identity_bindings,
)
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_PATH,
    load_registered_operator_registry,
)
from causal4d.preacquisition_readiness_contracts import (
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_contract_v2 import build_real_evidence_status


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(path.is_file(), f"{name} is missing")
    _require(not path.is_symlink(), f"{name} must not be a symlink")
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


def _parse_timestamp(value: Any, *, name: str) -> datetime:
    _require(isinstance(value, str) and bool(value), f"{name} is missing")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not ISO 8601") from error
    _require(parsed.tzinfo is not None, f"{name} must include a timezone")
    return parsed


def _identity_error(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _mark_invalid(result: dict[str, Any], error: BaseException | str) -> None:
    result["valid"] = False
    result["error"] = error if isinstance(error, str) else _identity_error(error)


def build_operator_bound_real_evidence_status(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_file_hashes: bool = False,
) -> dict[str, Any]:
    """Build evidence status with person-level approval identities enforced."""

    status = build_real_evidence_status(
        protocol,
        dataset_root,
        repository_root=repository_root,
        verify_file_hashes=verify_file_hashes,
    )
    root = Path(dataset_root)
    prerequisites = {
        str(name): dict(result) for name, result in status["prerequisites"].items()
    }

    preacquisition: Mapping[str, Any] | None = None
    if repository_root is None:
        registry_path = root / OPERATOR_REGISTRY_PATH
        registry_result: dict[str, Any] = {
            "path": str(registry_path.resolve()),
            "present": registry_path.is_file(),
            "valid": False,
            "error": "repository_root is required to verify operator_registry.json",
        }
        registry = None
    else:
        if (
            Path(repository_root) / "configs/causal4d/sloth_preacquisition_v5.json"
        ).is_file():
            _, _, _, preacquisition = load_registered_preacquisition_chain(
                repository_root
            )
        registry_result, registry = load_registered_operator_registry(
            repository_root,
            root,
        )
    prerequisites["operator_registry"] = registry_result
    prerequisites["operator_identity_bindings"] = (
        validate_preacquisition_identity_bindings(
            root,
            registry,
            preacquisition=preacquisition,
        )
    )

    freeze_result = prerequisites["method_freeze"]
    attestation_result = prerequisites["method_freeze_validation"]
    method_freeze: dict[str, Any] | None = None
    if freeze_result.get("present"):
        try:
            method_freeze = _read_json_mapping(
                root / "method_freeze.json",
                name="method freeze",
            )
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
            if freeze_result.get("valid"):
                _mark_invalid(freeze_result, error)

    if freeze_result.get("valid"):
        if registry is None:
            _mark_invalid(
                freeze_result,
                registry_result.get("error") or "operator registry is invalid",
            )
        elif method_freeze is not None:
            try:
                identity = validate_method_freeze_identity_evidence(
                    method_freeze,
                    None,
                    registry,
                    preacquisition=preacquisition,
                )
                freeze_result.update(identity)
                freeze_result["operator_registry_artifact_sha256"] = registry_result[
                    "artifact_sha256"
                ]
            except (KeyError, TypeError, ValueError) as error:
                _mark_invalid(freeze_result, error)

    if attestation_result.get("valid"):
        if registry is None or method_freeze is None:
            _mark_invalid(
                attestation_result,
                registry_result.get("error") or "operator registry is invalid",
            )
        else:
            try:
                attestation = _read_json_mapping(
                    root / "method_freeze_validation.json",
                    name="method freeze attestation",
                )
                identity = validate_method_freeze_identity_evidence(
                    method_freeze,
                    attestation,
                    registry,
                    preacquisition=preacquisition,
                )
                freeze_result.update(
                    {
                        "freezer_operator_id": identity["freezer_operator_id"],
                        "freezer_person_identity_sha256": identity[
                            "freezer_person_identity_sha256"
                        ],
                    }
                )
                attestation_result.update(
                    {
                        "verifier_operator_id": identity["verifier_operator_id"],
                        "verifier_person_identity_sha256": identity[
                            "verifier_person_identity_sha256"
                        ],
                        "operator_registry_artifact_sha256": registry_result[
                            "artifact_sha256"
                        ],
                    }
                )
            except (
                OSError,
                KeyError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                _mark_invalid(attestation_result, error)

    status["prerequisites"] = prerequisites
    chronology = dict(status["preacquisition_chronology"])
    events = dict(chronology.get("events", {}))
    events["operator_registry_sealed"] = (
        registry_result.get("sealed_at_utc") if registry_result.get("valid") else None
    )
    chronology["events"] = events
    chronology_blockers = list(chronology.get("blockers", []))
    earliest_text = chronology.get("earliest_execution_started_at_utc")
    if registry_result.get("valid") and earliest_text is not None:
        sealed_at = _parse_timestamp(
            registry_result.get("sealed_at_utc"),
            name="operator registry sealed_at_utc",
        )
        earliest = _parse_timestamp(
            earliest_text,
            name="earliest execution started_at_utc",
        )
        if sealed_at > earliest:
            chronology_blockers.append(
                "preacquisition_chronology:operator_registry_sealed"
            )
    chronology["blockers"] = list(dict.fromkeys(chronology_blockers))
    chronology["passed"] = not chronology["blockers"]
    status["preacquisition_chronology"] = chronology

    non_prerequisite_blockers = [
        str(blocker)
        for blocker in status["blockers"]
        if not str(blocker).startswith("prerequisite:")
        and not str(blocker).startswith("preacquisition_chronology:")
    ]
    prerequisite_blockers = [
        f"prerequisite:{name}"
        for name, result in prerequisites.items()
        if not result.get("valid")
    ]
    blockers = list(
        dict.fromkeys(
            prerequisite_blockers
            + non_prerequisite_blockers
            + list(chronology["blockers"])
        )
    )
    complete = not blockers
    status["blockers"] = blockers
    status["evidence_complete"] = complete
    status["complete"] = complete
    status["claim_ready"] = complete
    status["passed"] = complete
    status["file_hashes_verified"] = bool(verify_file_hashes and complete)
    return status


def validate_operator_bound_real_dataset(
    protocol: Mapping[str, Any],
    dataset_root: str | Path,
    *,
    repository_root: str | Path | None = None,
    verify_files: bool = True,
) -> dict[str, Any]:
    """Validate the complete evidence tree including registered identities."""

    status = build_operator_bound_real_evidence_status(
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


__all__ = [
    "build_operator_bound_real_evidence_status",
    "validate_operator_bound_real_dataset",
]
