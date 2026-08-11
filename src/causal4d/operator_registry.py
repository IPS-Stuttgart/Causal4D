"""Registered, privacy-preserving operator identities for acquisition evidence."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json

OPERATOR_REGISTRY_SCHEMA_VERSION = 1
OPERATOR_REGISTRY_ARTIFACT_KIND = "Causal4DOperatorIdentityRegistry"
OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND = "Causal4DOperatorIdentityRegistryTemplate"
OPERATOR_REGISTRY_PATH = "preacquisition/operator_registry.json"
OPERATOR_REGISTRY_TEMPLATE_PATH = "preacquisition/operator_registry.template.json"
PERSON_IDENTITY_DIGEST_METHOD = "hmac-sha256-domain-separated-v1"

ROLE_FREEZER = "freezer"
ROLE_INDEPENDENT_VERIFIER = "independent_verifier"
ROLE_GATE_APPROVER = "gate_approver"
ROLE_SOFTWARE_ENVIRONMENT_APPROVER = "software_environment_approver"
ALLOWED_OPERATOR_ROLES = frozenset(
    {
        ROLE_FREEZER,
        ROLE_INDEPENDENT_VERIFIER,
        ROLE_GATE_APPROVER,
        ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
    }
)

_OPERATOR_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA64 = re.compile(r"^[0-9a-f]{64}$")
_REGISTRY_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "status",
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_plan_id",
        "preacquisition_amendment_sha256",
        "person_identity_digest_method",
        "sealed_at_utc",
        "sealed_by_operator_id",
        "target_outcomes_used",
        "operators",
        "artifact_sha256",
    }
)
_OPERATOR_FIELDS = frozenset(
    {"operator_id", "person_identity_sha256", "active", "roles"}
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(
        not _contains_symlink_component(path),
        f"{name} contains a symlink component",
    )
    _require(path.is_file(), f"{name} is missing")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
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
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{name} must be UTC",
    )
    return parsed


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
    return digest.hexdigest(), size


def operator_registry_sha256(values: Mapping[str, Any]) -> str:
    """Return the canonical digest that seals one operator registry."""

    payload = deepcopy(dict(values))
    payload.pop("artifact_sha256", None)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def operator_registry_template(
    protocol: Mapping[str, Any],
    preacquisition_v4: Mapping[str, Any],
) -> dict[str, Any]:
    """Return an explicitly incomplete, protocol-bound registry template."""

    return {
        "schema_version": OPERATOR_REGISTRY_SCHEMA_VERSION,
        "artifact_kind": OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND,
        "status": "template",
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": preacquisition_v4["plan_id"],
        "preacquisition_amendment_sha256": preacquisition_v4["amendment_sha256"],
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
        "sealed_at_utc": None,
        "sealed_by_operator_id": None,
        "target_outcomes_used": False,
        "operators": [],
        "artifact_sha256": None,
    }


def _validate_operator_record(value: Any, *, index: int) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"operators[{index}] must be an object")
    record = dict(value)
    _require(
        set(record) == _OPERATOR_FIELDS,
        f"operators[{index}] fields differ from the registered schema",
    )
    operator_id = record.get("operator_id")
    _require(
        isinstance(operator_id, str) and bool(_OPERATOR_ID.fullmatch(operator_id)),
        f"operators[{index}].operator_id is invalid",
    )
    person_digest = record.get("person_identity_sha256")
    _require(
        isinstance(person_digest, str) and bool(_SHA64.fullmatch(person_digest)),
        f"operators[{index}].person_identity_sha256 is invalid",
    )
    _require(
        isinstance(record.get("active"), bool),
        f"operators[{index}].active must be Boolean",
    )
    roles = record.get("roles")
    _require(
        isinstance(roles, list) and bool(roles),
        f"operators[{index}].roles must be a nonempty list",
    )
    _require(
        all(isinstance(role, str) for role in roles),
        f"operators[{index}].roles must contain strings",
    )
    _require(
        len(roles) == len(set(roles)),
        f"operators[{index}].roles contains duplicates",
    )
    _require(
        set(roles).issubset(ALLOWED_OPERATOR_ROLES),
        f"operators[{index}].roles contains an unsupported role",
    )
    _require(
        roles == sorted(roles),
        f"operators[{index}].roles must be sorted canonically",
    )
    return record


def validate_operator_registry_template(
    protocol: Mapping[str, Any],
    preacquisition_v4: Mapping[str, Any],
    template: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one unsealed registry draft without requiring it to be complete."""

    _require(
        set(template) == _REGISTRY_FIELDS,
        "operator registry template fields differ from the registered schema",
    )
    _require(
        template.get("schema_version") == OPERATOR_REGISTRY_SCHEMA_VERSION,
        "unsupported operator registry template schema",
    )
    _require(
        template.get("artifact_kind") == OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND,
        "unexpected operator registry template artifact kind",
    )
    _require(
        template.get("status") == "template",
        "operator registry draft is not an unsealed template",
    )
    _require(
        template.get("protocol_id") == protocol["protocol_id"],
        "operator registry template protocol id mismatch",
    )
    _require(
        template.get("protocol_design_sha256") == protocol["design_sha256"],
        "operator registry template protocol digest mismatch",
    )
    _require(
        template.get("preacquisition_plan_id") == preacquisition_v4["plan_id"],
        "operator registry template pre-acquisition plan mismatch",
    )
    _require(
        template.get("preacquisition_amendment_sha256")
        == preacquisition_v4["amendment_sha256"],
        "operator registry template pre-acquisition amendment mismatch",
    )
    _require(
        template.get("person_identity_digest_method") == PERSON_IDENTITY_DIGEST_METHOD,
        "operator registry template uses an unsupported identity digest method",
    )
    _require(
        template.get("sealed_at_utc") is None
        and template.get("sealed_by_operator_id") is None
        and template.get("artifact_sha256") is None,
        "operator registry template contains seal metadata",
    )
    _require(
        template.get("target_outcomes_used") is False,
        "target outcomes entered operator identity evidence",
    )

    operators_value = template.get("operators")
    _require(
        isinstance(operators_value, list),
        "operator registry template operators must be a list",
    )
    records: list[dict[str, Any]] = []
    for index, value in enumerate(operators_value):
        _require(
            isinstance(value, Mapping),
            f"operators[{index}] must be an object",
        )
        record = dict(value)
        roles = record.get("roles")
        _require(
            isinstance(roles, list),
            f"operators[{index}].roles must be a list",
        )
        record["roles"] = sorted(roles)
        records.append(_validate_operator_record(record, index=index))

    operator_ids = [str(record["operator_id"]) for record in records]
    person_digests = [str(record["person_identity_sha256"]) for record in records]
    _require(
        len(operator_ids) == len(set(operator_ids)),
        "operator registry template contains duplicate operator ids",
    )
    _require(
        len(person_digests) == len(set(person_digests)),
        "operator registry template contains duplicate person identity digests",
    )
    return {
        "passed": True,
        "operator_count": len(records),
        "populated": bool(records),
        "target_outcomes_used": False,
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
    }


def validate_operator_registry(
    protocol: Mapping[str, Any],
    preacquisition_v4: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one sealed registry and return its public provenance summary."""

    _require(
        set(registry) == _REGISTRY_FIELDS,
        "operator registry fields differ from the registered schema",
    )
    _require(
        registry.get("schema_version") == OPERATOR_REGISTRY_SCHEMA_VERSION,
        "unsupported operator registry schema",
    )
    _require(
        registry.get("artifact_kind") == OPERATOR_REGISTRY_ARTIFACT_KIND,
        "unexpected operator registry artifact kind",
    )
    _require(registry.get("status") == "sealed", "operator registry is not sealed")
    _require(
        registry.get("protocol_id") == protocol["protocol_id"],
        "operator registry protocol id mismatch",
    )
    _require(
        registry.get("protocol_design_sha256") == protocol["design_sha256"],
        "operator registry protocol digest mismatch",
    )
    _require(
        registry.get("preacquisition_plan_id") == preacquisition_v4["plan_id"],
        "operator registry pre-acquisition plan mismatch",
    )
    _require(
        registry.get("preacquisition_amendment_sha256")
        == preacquisition_v4["amendment_sha256"],
        "operator registry pre-acquisition amendment mismatch",
    )
    _require(
        registry.get("person_identity_digest_method") == PERSON_IDENTITY_DIGEST_METHOD,
        "operator registry uses an unsupported identity digest method",
    )
    sealed_at = _parse_utc_timestamp(
        registry.get("sealed_at_utc"),
        name="operator registry sealed_at_utc",
    )
    _require(
        registry.get("target_outcomes_used") is False,
        "target outcomes entered operator identity evidence",
    )

    operators_value = registry.get("operators")
    _require(
        isinstance(operators_value, list) and bool(operators_value),
        "operator registry must contain at least one operator",
    )
    operators = [
        _validate_operator_record(value, index=index)
        for index, value in enumerate(operators_value)
    ]
    operator_ids = [str(record["operator_id"]) for record in operators]
    person_digests = [str(record["person_identity_sha256"]) for record in operators]
    _require(
        operator_ids == sorted(operator_ids),
        "operator registry entries must be sorted canonically by operator_id",
    )
    _require(
        len(operator_ids) == len(set(operator_ids)),
        "operator registry contains duplicate operator ids",
    )
    _require(
        len(person_digests) == len(set(person_digests)),
        "operator registry contains duplicate person identity digests",
    )

    active = [record for record in operators if record["active"] is True]
    role_counts = {
        role: sum(role in record["roles"] for record in active)
        for role in sorted(ALLOWED_OPERATOR_ROLES)
    }
    for role in (ROLE_FREEZER, ROLE_GATE_APPROVER):
        _require(
            role_counts[role] > 0,
            f"operator registry has no active operator for role: {role}",
        )
    freezer_digests = {
        str(record["person_identity_sha256"])
        for record in active
        if ROLE_FREEZER in record["roles"]
    }
    verifier_digests = {
        str(record["person_identity_sha256"])
        for record in active
        if ROLE_INDEPENDENT_VERIFIER in record["roles"]
    }
    independent_verifier_available = any(
        freezer_digest != verifier_digest
        for freezer_digest in freezer_digests
        for verifier_digest in verifier_digests
    )

    sealed_by = registry.get("sealed_by_operator_id")
    _require(
        isinstance(sealed_by, str) and bool(sealed_by),
        "operator registry sealer id is missing",
    )
    sealer = next(
        (record for record in active if record["operator_id"] == sealed_by),
        None,
    )
    _require(sealer is not None, "operator registry sealer is unknown or inactive")

    artifact_sha256 = registry.get("artifact_sha256")
    _require(
        isinstance(artifact_sha256, str) and bool(_SHA64.fullmatch(artifact_sha256)),
        "operator registry artifact digest is invalid",
    )
    _require(
        artifact_sha256 == operator_registry_sha256(registry),
        "operator registry artifact digest mismatch",
    )
    return {
        "passed": True,
        "sealed_at_utc": str(registry["sealed_at_utc"]),
        "sealed_by_operator_id": sealed_by,
        "artifact_sha256": artifact_sha256,
        "operator_count": len(operators),
        "active_operator_count": len(active),
        "active_role_counts": role_counts,
        "independent_verifier_available": independent_verifier_available,
        "target_outcomes_used": False,
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
        "_sealed_at": sealed_at,
    }


def resolve_operator(
    registry: Mapping[str, Any],
    operator_id: Any,
    *,
    required_role: str | None = None,
    any_role: frozenset[str] | set[str] | None = None,
    name: str = "operator",
) -> dict[str, Any]:
    """Resolve one active operator and enforce its registered role policy."""

    _require(
        isinstance(operator_id, str) and bool(operator_id),
        f"{name} id is missing",
    )
    operators = registry.get("operators")
    _require(isinstance(operators, list), "operator registry entries are missing")
    record = next(
        (
            dict(value)
            for value in operators
            if isinstance(value, Mapping) and value.get("operator_id") == operator_id
        ),
        None,
    )
    _require(record is not None, f"{name} is not registered: {operator_id}")
    _require(record.get("active") is True, f"{name} is inactive: {operator_id}")
    roles = record.get("roles")
    _require(isinstance(roles, list), f"{name} roles are invalid: {operator_id}")
    if required_role is not None:
        _require(
            required_role in roles,
            f"{name} lacks required role {required_role}: {operator_id}",
        )
    if any_role is not None:
        _require(
            bool(set(roles) & set(any_role)),
            f"{name} lacks every permitted role: {operator_id}",
        )
    return record


def require_distinct_operator_people(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    relationship: str,
) -> None:
    """Reject aliases or duplicate identities where independence is required."""

    _require(
        first.get("person_identity_sha256") != second.get("person_identity_sha256"),
        relationship,
    )


def require_registry_precedes_event(
    registry: Mapping[str, Any],
    event_at_utc: Any,
    *,
    event_name: str,
) -> None:
    """Require the immutable identity roster to predate a governed event."""

    sealed_at = _parse_utc_timestamp(
        registry.get("sealed_at_utc"),
        name="operator registry sealed_at_utc",
    )
    event_at = _parse_utc_timestamp(event_at_utc, name=event_name)
    _require(
        sealed_at <= event_at,
        f"operator registry postdates {event_name}",
    )


def validate_method_freeze_operator_identity(
    method_freeze: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the method freezer and enforce registry chronology."""

    freezer = resolve_operator(
        registry,
        method_freeze.get("frozen_by"),
        required_role=ROLE_FREEZER,
        name="method freezer",
    )
    require_registry_precedes_event(
        registry,
        method_freeze.get("frozen_at_utc"),
        event_name="method freeze",
    )
    return freezer


def validate_attestation_operator_identities(
    method_freeze: Mapping[str, Any],
    attestation: Mapping[str, Any],
    registry: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve freezer and verifier and prove person-level independence."""

    freezer = validate_method_freeze_operator_identity(method_freeze, registry)
    verifier = resolve_operator(
        registry,
        attestation.get("verifier_id"),
        required_role=ROLE_INDEPENDENT_VERIFIER,
        name="method freeze verifier",
    )
    require_registry_precedes_event(
        registry,
        attestation.get("verified_at_utc"),
        event_name="method freeze attestation",
    )
    require_distinct_operator_people(
        freezer,
        verifier,
        relationship="method freeze must be verified by a distinct registered person",
    )
    return freezer, verifier


def validate_gate_approver_identity(
    gate_id: str,
    approver_id: Any,
    approved_at_utc: Any,
    registry: Mapping[str, Any],
    *,
    freezer_person_identity_sha256: str | None = None,
) -> dict[str, Any]:
    """Resolve a gate approver and enforce the software-lock independence policy."""

    approver = resolve_operator(
        registry,
        approver_id,
        required_role=ROLE_GATE_APPROVER,
        name=f"{gate_id} approver",
    )
    require_registry_precedes_event(
        registry,
        approved_at_utc,
        event_name=f"{gate_id} approval",
    )
    if gate_id == "software_environment_locked":
        _require(
            bool(
                set(approver["roles"])
                & {
                    ROLE_INDEPENDENT_VERIFIER,
                    ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
                }
            ),
            "software environment approver lacks an independent approval role",
        )
        _require(
            isinstance(freezer_person_identity_sha256, str)
            and bool(_SHA64.fullmatch(freezer_person_identity_sha256)),
            "software environment approval cannot resolve the method freezer identity",
        )
        _require(
            approver["person_identity_sha256"] != freezer_person_identity_sha256,
            "software environment approval is not independent of the method freezer",
        )
    return approver


def load_operator_registry_template_prerequisite(
    protocol: Mapping[str, Any],
    preacquisition_v4: Mapping[str, Any],
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Inspect the optional unsealed registry draft without exposing identities."""

    path = Path(dataset_root) / OPERATOR_REGISTRY_TEMPLATE_PATH
    result: dict[str, Any] = {
        "path": str(path.absolute()),
        "present": os.path.lexists(path),
        "valid": False,
        "template": True,
        "error": None,
    }
    if not result["present"]:
        result["error"] = "operator_registry.template.json is missing"
        return result
    try:
        template = _read_json_mapping(path, name="operator registry template")
        result.update(
            validate_operator_registry_template(
                protocol,
                preacquisition_v4,
                template,
            )
        )
        result["sha256"], result["bytes"] = _sha256_file(path)
        result["valid"] = True
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        message = str(error).strip()
        result["error"] = (
            f"{type(error).__name__}: {message}" if message else type(error).__name__
        )
    return result


def load_operator_registry_prerequisite(
    protocol: Mapping[str, Any],
    preacquisition_v4: Mapping[str, Any],
    dataset_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load the canonical registry as one fail-closed acquisition prerequisite."""

    path = Path(dataset_root) / OPERATOR_REGISTRY_PATH
    template_status = load_operator_registry_template_prerequisite(
        protocol,
        preacquisition_v4,
        dataset_root,
    )
    result: dict[str, Any] = {
        "path": str(path.absolute()),
        "present": os.path.lexists(path),
        "valid": False,
        "error": None,
        "template_status": template_status,
    }
    if not result["present"]:
        result["error"] = "operator_registry.json is missing"
        return result, None
    try:
        registry = _read_json_mapping(path, name="operator registry")
        summary = validate_operator_registry(protocol, preacquisition_v4, registry)
        summary.pop("_sealed_at", None)
        result.update(summary)
        result["sha256"], result["bytes"] = _sha256_file(path)
        result["valid"] = True
        return result, registry
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        message = str(error).strip()
        result["error"] = (
            f"{type(error).__name__}: {message}" if message else type(error).__name__
        )
        return result, None


def load_registered_operator_registry(
    repository_root: str | Path,
    dataset_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Load the registry against the repository's exact registered v4 chain."""

    from causal4d.preacquisition_readiness_contracts import (
        load_registered_preacquisition_chain,
    )

    protocol, _, _, preacquisition_v4 = load_registered_preacquisition_chain(
        repository_root
    )
    return load_operator_registry_prerequisite(
        protocol,
        preacquisition_v4,
        dataset_root,
    )


def scaffold_operator_registry(
    repository_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Write one incomplete registry template without replacing operator work."""

    from causal4d.preacquisition_readiness_contracts import (
        load_registered_preacquisition_chain,
    )

    protocol, _, _, preacquisition_v4 = load_registered_preacquisition_chain(
        repository_root
    )
    path = Path(dataset_root) / OPERATOR_REGISTRY_TEMPLATE_PATH
    try:
        atomic_write_json(
            path,
            operator_registry_template(protocol, preacquisition_v4),
            overwrite=False,
        )
    except FileExistsError:
        created = False
    else:
        created = True
    return {
        "passed": True,
        "path": str(path.resolve()),
        "created": created,
        "existing": not created,
    }


def _assert_registry_precedes_governed_evidence(dataset_root: Path) -> None:
    from causal4d.preacquisition_readiness_contracts import GATE_PATHS

    for relative in ("method_freeze.json", "method_freeze_validation.json"):
        _require(
            not (dataset_root / relative).exists(),
            "operator registry must be sealed before method-freeze evidence",
        )
    for gate_id, relative in GATE_PATHS.items():
        path = dataset_root / relative
        if not path.is_file():
            continue
        gate = _read_json_mapping(path, name=f"{gate_id} gate")
        approval = gate.get("approval")
        _require(
            gate.get("status") == "template"
            and isinstance(approval, Mapping)
            and approval.get("approved") is False,
            f"operator registry must be sealed before gate approval: {gate_id}",
        )
    for pattern in ("executions/*/manifest.json", "sessions/*/session.json"):
        _require(
            not any(dataset_root.glob(pattern)),
            "operator registry must be sealed before confirmatory collection",
        )


def _normalized_operator_records(values: Any) -> list[dict[str, Any]]:
    _require(isinstance(values, list) and bool(values), "operators must be nonempty")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(values):
        _require(isinstance(value, Mapping), f"operators[{index}] must be an object")
        record = dict(value)
        _require(
            set(record) == _OPERATOR_FIELDS,
            f"operators[{index}] fields differ from the registered schema",
        )
        roles = record.get("roles")
        _require(isinstance(roles, list), f"operators[{index}].roles must be a list")
        record["roles"] = sorted(roles)
        records.append(record)
    return sorted(records, key=lambda record: str(record.get("operator_id", "")))


def seal_operator_registry(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
    *,
    sealed_by: str,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate and atomically publish the canonical, once-only identity roster."""

    from causal4d.preacquisition_readiness_contracts import (
        load_registered_preacquisition_chain,
    )

    protocol, _, _, preacquisition_v4 = load_registered_preacquisition_chain(
        repository_root
    )
    root = Path(dataset_root)
    target = root / OPERATOR_REGISTRY_PATH
    _require(not target.exists(), "operator registry is already sealed")
    _assert_registry_precedes_governed_evidence(root)

    draft = _read_json_mapping(Path(source_json), name="operator registry draft")
    _require(
        set(draft) == _REGISTRY_FIELDS,
        "operator registry draft fields differ from the registered schema",
    )
    _require(
        draft.get("schema_version") == OPERATOR_REGISTRY_SCHEMA_VERSION,
        "unsupported operator registry draft schema",
    )
    _require(
        draft.get("artifact_kind") == OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND
        and draft.get("status") == "template",
        "operator registry draft must remain an unsealed template",
    )
    _require(
        draft.get("protocol_id") == protocol["protocol_id"]
        and draft.get("protocol_design_sha256") == protocol["design_sha256"],
        "operator registry draft binds a different protocol",
    )
    _require(
        draft.get("preacquisition_plan_id") == preacquisition_v4["plan_id"]
        and draft.get("preacquisition_amendment_sha256")
        == preacquisition_v4["amendment_sha256"],
        "operator registry draft binds a different pre-acquisition amendment",
    )
    _require(
        draft.get("person_identity_digest_method") == PERSON_IDENTITY_DIGEST_METHOD,
        "operator registry draft uses an unsupported identity digest method",
    )
    _require(
        draft.get("sealed_at_utc") is None
        and draft.get("sealed_by_operator_id") is None
        and draft.get("artifact_sha256") is None,
        "operator registry draft already contains seal metadata",
    )
    _require(
        draft.get("target_outcomes_used") is False,
        "target outcomes entered operator identity evidence",
    )

    timestamp = sealed_at_utc or datetime.now(timezone.utc).isoformat()
    _parse_utc_timestamp(timestamp, name="operator registry sealed_at_utc")
    registry = {
        **draft,
        "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
        "status": "sealed",
        "sealed_at_utc": timestamp,
        "sealed_by_operator_id": sealed_by,
        "operators": _normalized_operator_records(draft.get("operators")),
    }
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    summary = validate_operator_registry(protocol, preacquisition_v4, registry)
    summary.pop("_sealed_at", None)
    atomic_write_json(target, registry, overwrite=False)
    file_sha256, byte_count = _sha256_file(target)
    return {
        **summary,
        "path": str(target.resolve()),
        "sha256": file_sha256,
        "bytes": byte_count,
        "valid": True,
        "passed": True,
    }


__all__ = [
    "ALLOWED_OPERATOR_ROLES",
    "OPERATOR_REGISTRY_ARTIFACT_KIND",
    "OPERATOR_REGISTRY_PATH",
    "OPERATOR_REGISTRY_SCHEMA_VERSION",
    "OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND",
    "OPERATOR_REGISTRY_TEMPLATE_PATH",
    "PERSON_IDENTITY_DIGEST_METHOD",
    "ROLE_FREEZER",
    "ROLE_GATE_APPROVER",
    "ROLE_INDEPENDENT_VERIFIER",
    "ROLE_SOFTWARE_ENVIRONMENT_APPROVER",
    "load_operator_registry_prerequisite",
    "load_operator_registry_template_prerequisite",
    "load_registered_operator_registry",
    "operator_registry_sha256",
    "operator_registry_template",
    "require_distinct_operator_people",
    "require_registry_precedes_event",
    "resolve_operator",
    "scaffold_operator_registry",
    "seal_operator_registry",
    "validate_attestation_operator_identities",
    "validate_gate_approver_identity",
    "validate_method_freeze_operator_identity",
    "validate_operator_registry",
    "validate_operator_registry_template",
]
