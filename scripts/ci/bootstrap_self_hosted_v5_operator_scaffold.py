#!/usr/bin/env python3
"""Bootstrap the fresh v5 single-operator tree without historical identity data."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import shutil
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from causal4d.atomic_io import atomic_write_binary, atomic_write_json
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    PERSON_IDENTITY_DIGEST_METHOD,
    ROLE_FREEZER,
    ROLE_GATE_APPROVER,
    ROLE_INDEPENDENT_VERIFIER,
    ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
    seal_operator_registry,
    scaffold_operator_registry,
    validate_operator_registry,
)
from causal4d.preacquisition_operator_flow import (
    build_preacquisition_operator_next_action,
)
from causal4d.preacquisition_readiness import scaffold_preacquisition_readiness
from causal4d.preacquisition_readiness_contracts import (
    load_registered_preacquisition_chain,
)
from causal4d.real_evidence_contract_v2 import scaffold_real_evidence_v2_templates
from causal4d.real_protocol import load_protocol, scaffold_dataset


REPORT_SCHEMA_VERSION = 2
REPORT_ARTIFACT_KIND = "Causal4DSingleOperatorV5BootstrapReport"
RECEIPT_SCHEMA_VERSION = 2
RECEIPT_ARTIFACT_KIND = "Causal4DSingleOperatorV5BootstrapReceipt"
RECEIPT_PATH = "preacquisition/single_operator_v5_bootstrap.json"
PRIVATE_ROSTER_SCHEMA_VERSION = 2
PRIVATE_ROSTER_ARTIFACT_KIND = "Causal4DPrivateOperatorPrincipalRoster"
KEY_FILENAME = "operator-identity-hmac-v1.key"
PRIVATE_ROSTER_FILENAME = "operator-principals-v1.json"
PERSON_DOMAIN = b"causal4d-operator-v1\0"
KEY_BYTES = 32
IDENTITY_INITIALIZATION_MODE = "fresh_owner_hmac_v1"
IMPLEMENTATION_PATH = "scripts/ci/bootstrap_self_hosted_v5_operator_scaffold.py"
OPERATOR_ID = "florianpfaff"
CANONICAL_PRINCIPAL = "github-login-v1:FlorianPfaff"
OPERATOR_ROLES = tuple(
    sorted(
        (
            ROLE_FREEZER,
            ROLE_GATE_APPROVER,
            ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
        )
    )
)
_EXPECTED_PRIVATE_FILES = frozenset({KEY_FILENAME, PRIVATE_ROSTER_FILENAME})
_GOVERNED_PATHS = (
    "object_registration.json",
    "slip_pilot.json",
    "timebase_calibration.json",
    "contact_registration.json",
    "method_freeze.json",
    "method_freeze_validation.json",
    "registered-analysis.json",
)
_GOVERNED_PATTERNS = (
    "preacquisition/source_panel/executions/*/manifest.json",
    "executions/*/manifest.json",
    "sessions/*/session.json",
)
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "target_preacquisition_plan_id",
        "target_preacquisition_amendment_sha256",
        "identity_initialization_mode",
        "person_identity_digest_method",
        "private_roster_artifact_sha256",
        "bootstrap_implementation_sha256",
        "bootstrap_implementation_bytes",
        "historical_registry_available",
        "historical_registry_reused",
        "identity_digest_continuity_claimed",
        "target_registry_artifact_sha256",
        "target_registry_file_sha256",
        "target_registry_file_bytes",
        "operator_id",
        "operator_roles",
        "sealed_at_utc",
        "independent_preacquisition_attestation_claimed",
        "target_outcomes_used",
        "physical_command_sent",
        "physical_evidence_increment",
        "artifact_sha256",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _require_ordinary_directory(path: Path, *, name: str) -> Path:
    _require(not _contains_symlink_component(path), f"{name} contains a symlink")
    _require(path.is_dir(), f"{name} must be an ordinary directory")
    return path.resolve()


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(not _contains_symlink_component(path), f"{name} contains a symlink")
    _require(path.is_file(), f"{name} is missing")
    payload = json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=_reject_json_constant,
    )
    _require(isinstance(payload, Mapping), f"{name} must be a JSON object")
    return dict(payload)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _canonical_sha256(payload: Mapping[str, Any], *, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_mode(path: Path, *, name: str, expected: int) -> None:
    actual = stat.S_IMODE(path.stat().st_mode)
    _require(actual == expected, f"{name} mode is {actual:o}, expected {expected:o}")


def _private_roster_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PRIVATE_ROSTER_SCHEMA_VERSION,
        "artifact_kind": PRIVATE_ROSTER_ARTIFACT_KIND,
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
        "assignments": [
            {
                "operator_id": OPERATOR_ID,
                "canonical_principal": CANONICAL_PRINCIPAL,
                "roles": list(OPERATOR_ROLES),
            }
        ],
        "target_outcomes_used": False,
    }
    payload["artifact_sha256"] = _canonical_sha256(
        payload,
        field="artifact_sha256",
    )
    return payload


def _person_digest(secret: bytes) -> str:
    return hmac.new(
        secret,
        PERSON_DOMAIN + CANONICAL_PRINCIPAL.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _operator_record(secret: bytes) -> dict[str, Any]:
    return {
        "operator_id": OPERATOR_ID,
        "person_identity_sha256": _person_digest(secret),
        "active": True,
        "roles": list(OPERATOR_ROLES),
    }


def _read_private_material(private_root: Path) -> tuple[bytes, dict[str, Any]]:
    root = _require_ordinary_directory(private_root, name="private identity root")
    _require_mode(root, name="private identity root", expected=0o700)
    members = list(root.iterdir())
    _require(
        {member.name for member in members} == _EXPECTED_PRIVATE_FILES,
        "private identity root members differ from the registered schema",
    )
    _require(
        all(member.is_file() and not member.is_symlink() for member in members),
        "private identity root contains a non-regular member",
    )
    key_path = root / KEY_FILENAME
    roster_path = root / PRIVATE_ROSTER_FILENAME
    _require_mode(key_path, name="operator identity key", expected=0o600)
    _require_mode(roster_path, name="private operator roster", expected=0o600)
    secret = key_path.read_bytes()
    _require(len(secret) == KEY_BYTES, "operator identity key length is invalid")
    roster = _read_json_mapping(roster_path, name="private operator roster")
    _require(
        roster == _private_roster_payload(),
        "private operator roster differs from the fresh-owner lock",
    )
    return secret, roster


def _create_or_read_private_material(
    private_root: Path,
) -> tuple[bytes, dict[str, Any], bool]:
    root = private_root.absolute()
    _require(
        not _contains_symlink_component(root),
        "private identity root contains a symlink component",
    )
    if os.path.lexists(root):
        secret, roster = _read_private_material(root)
        return secret, roster, False

    parent = root.parent
    _require(
        not _contains_symlink_component(parent),
        "private identity parent contains a symlink component",
    )
    if not os.path.lexists(parent):
        grandparent = _require_ordinary_directory(
            parent.parent,
            name="private identity grandparent",
        )
        parent.mkdir(mode=0o700)
        _fsync_directory(grandparent)
    parent = _require_ordinary_directory(parent, name="private identity parent")
    _require_mode(parent, name="private identity parent", expected=0o700)
    staging = parent / f".{root.name}.bootstrap-v5.tmp"
    _require(
        not os.path.lexists(staging),
        "private identity staging path already exists",
    )
    try:
        staging.mkdir(mode=0o700)
        secret = secrets.token_bytes(KEY_BYTES)

        def write_key(handle: Any) -> None:
            handle.write(secret)

        key_path = staging / KEY_FILENAME
        roster_path = staging / PRIVATE_ROSTER_FILENAME
        atomic_write_binary(key_path, write_key, overwrite=False)
        os.chmod(key_path, 0o600)
        atomic_write_json(roster_path, _private_roster_payload(), overwrite=False)
        os.chmod(roster_path, 0o600)
        _fsync_directory(staging)
        staging.rename(root)
        _fsync_directory(parent)
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    loaded_secret, roster = _read_private_material(root)
    _require(loaded_secret == secret, "private identity publication changed")
    return loaded_secret, roster, True


def _require_no_governed_evidence(dataset: Path) -> None:
    for relative in _GOVERNED_PATHS:
        _require(
            not os.path.lexists(dataset / relative),
            f"v5 bootstrap is too late; governed evidence exists: {relative}",
        )
    for pattern in _GOVERNED_PATTERNS:
        _require(
            not any(dataset.glob(pattern)),
            f"v5 bootstrap is too late; governed evidence matches: {pattern}",
        )


def _receipt(
    *,
    private_roster_artifact_sha256: str,
    bootstrap_implementation_sha256: str,
    bootstrap_implementation_bytes: int,
    target_registry_artifact_sha256: str,
    target_registry_file_sha256: str,
    target_registry_file_bytes: int,
    plan_id: str,
    amendment_sha256: str,
    sealed_at_utc: str,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "artifact_kind": RECEIPT_ARTIFACT_KIND,
        "target_preacquisition_plan_id": plan_id,
        "target_preacquisition_amendment_sha256": amendment_sha256,
        "identity_initialization_mode": IDENTITY_INITIALIZATION_MODE,
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
        "private_roster_artifact_sha256": private_roster_artifact_sha256,
        "bootstrap_implementation_sha256": bootstrap_implementation_sha256,
        "bootstrap_implementation_bytes": bootstrap_implementation_bytes,
        "historical_registry_available": False,
        "historical_registry_reused": False,
        "identity_digest_continuity_claimed": False,
        "target_registry_artifact_sha256": target_registry_artifact_sha256,
        "target_registry_file_sha256": target_registry_file_sha256,
        "target_registry_file_bytes": target_registry_file_bytes,
        "operator_id": OPERATOR_ID,
        "operator_roles": list(OPERATOR_ROLES),
        "sealed_at_utc": sealed_at_utc,
        "independent_preacquisition_attestation_claimed": False,
        "target_outcomes_used": False,
        "physical_command_sent": False,
        "physical_evidence_increment": 0,
    }
    payload["artifact_sha256"] = _canonical_sha256(payload, field="artifact_sha256")
    return payload


def _verify_target(
    repository: Path,
    target_dataset: Path,
    operator: Mapping[str, Any],
    private_roster_artifact_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = _require_ordinary_directory(target_dataset, name="v5 target dataset")
    _require_no_governed_evidence(target)
    protocol, _, _, v5 = load_registered_preacquisition_chain(repository)
    _require(
        load_protocol(target / "protocol.json") == protocol,
        "target protocol changed",
    )
    registry_path = target / OPERATOR_REGISTRY_PATH
    registry = _read_json_mapping(registry_path, name="v5 operator registry")
    summary = validate_operator_registry(protocol, v5, registry)
    operators = registry.get("operators")
    _require(
        isinstance(operators, list) and operators == [dict(operator)],
        "v5 registry differs from the owner-only private identity",
    )
    _require(
        summary.get("independent_verifier_available") is False,
        "v5 registry claims independent verification",
    )
    _require(
        ROLE_INDEPENDENT_VERIFIER
        not in cast(list[dict[str, Any]], operators)[0]["roles"],
        "v5 registry assigns a false independent role",
    )

    receipt = _read_json_mapping(
        target / RECEIPT_PATH,
        name="v5 bootstrap receipt",
    )
    _require(
        set(receipt) == _RECEIPT_FIELDS,
        "v5 bootstrap receipt fields differ from schema v2",
    )
    _require(
        receipt.get("schema_version") == RECEIPT_SCHEMA_VERSION
        and receipt.get("artifact_kind") == RECEIPT_ARTIFACT_KIND,
        "unexpected v5 bootstrap receipt",
    )
    _require(
        receipt.get("artifact_sha256")
        == _canonical_sha256(receipt, field="artifact_sha256"),
        "v5 bootstrap receipt digest mismatch",
    )
    file_sha256, file_bytes = _sha256_file(registry_path)
    implementation_sha256, implementation_bytes = _sha256_file(
        repository / IMPLEMENTATION_PATH
    )
    expected = {
        "target_preacquisition_plan_id": v5["plan_id"],
        "target_preacquisition_amendment_sha256": v5["amendment_sha256"],
        "identity_initialization_mode": IDENTITY_INITIALIZATION_MODE,
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
        "private_roster_artifact_sha256": private_roster_artifact_sha256,
        "bootstrap_implementation_sha256": implementation_sha256,
        "bootstrap_implementation_bytes": implementation_bytes,
        "historical_registry_available": False,
        "historical_registry_reused": False,
        "identity_digest_continuity_claimed": False,
        "target_registry_artifact_sha256": registry["artifact_sha256"],
        "target_registry_file_sha256": file_sha256,
        "target_registry_file_bytes": file_bytes,
        "operator_id": OPERATOR_ID,
        "operator_roles": list(OPERATOR_ROLES),
        "sealed_at_utc": registry["sealed_at_utc"],
        "independent_preacquisition_attestation_claimed": False,
        "target_outcomes_used": False,
        "physical_command_sent": False,
        "physical_evidence_increment": 0,
    }
    for field, expected_value in expected.items():
        _require(
            receipt.get(field) == expected_value,
            f"v5 bootstrap receipt {field} mismatch",
        )

    decision = build_preacquisition_operator_next_action(
        repository,
        target,
        verify_file_hashes=True,
    )
    action_value = decision.get("action")
    _require(isinstance(action_value, Mapping), "v5 next action is missing")
    action = dict(cast(Mapping[str, Any], action_value))
    _require(
        action.get("action_id") == "complete_object_registration",
        "v5 bootstrap did not advance to object registration",
    )
    _require(
        action.get("operator_role") == "self_attesting_operator",
        "v5 next action has the wrong operator role",
    )
    _require(
        action.get("automatable") is False,
        "object registration unexpectedly became automatable",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "v5 next action permits target outcomes",
    )
    return receipt, action


def bootstrap_single_operator_v5(
    *,
    repository_root: Path,
    private_identity_root: Path,
    target_dataset_root: Path,
    sealed_at_utc: str | None = None,
) -> dict[str, Any]:
    repository = _require_ordinary_directory(repository_root, name="repository root")
    target = target_dataset_root.absolute()
    _require(
        not _contains_symlink_component(target),
        "target dataset contains a symlink component",
    )
    implementation_sha256, implementation_bytes = _sha256_file(
        repository / IMPLEMENTATION_PATH
    )
    secret, private_roster, private_created = _create_or_read_private_material(
        private_identity_root,
    )
    operator = _operator_record(secret)
    private_roster_artifact_sha256 = str(private_roster["artifact_sha256"])

    created = False
    if not os.path.lexists(target):
        target.parent.mkdir(parents=True, exist_ok=True)
        _require_ordinary_directory(target.parent, name="target dataset parent")
        staging = target.parent / f".{target.name}.bootstrap-v5.tmp"
        _require(
            not os.path.lexists(staging),
            "v5 bootstrap staging path already exists",
        )
        try:
            protocol, _, _, v5 = load_registered_preacquisition_chain(repository)
            scaffold_dataset(protocol, staging)
            scaffold_real_evidence_v2_templates(protocol, staging)
            scaffold_preacquisition_readiness(repository, staging)
            scaffold_operator_registry(repository, staging)
            template_path = staging / OPERATOR_REGISTRY_TEMPLATE_PATH
            template = _read_json_mapping(
                template_path,
                name="v5 operator registry template",
            )
            template["operators"] = [operator]
            atomic_write_json(template_path, template, overwrite=True)
            timestamp = sealed_at_utc or datetime.now(timezone.utc).isoformat()
            sealed = seal_operator_registry(
                repository,
                staging,
                template_path,
                sealed_by=OPERATOR_ID,
                sealed_at_utc=timestamp,
            )
            receipt = _receipt(
                private_roster_artifact_sha256=private_roster_artifact_sha256,
                bootstrap_implementation_sha256=implementation_sha256,
                bootstrap_implementation_bytes=implementation_bytes,
                target_registry_artifact_sha256=str(sealed["artifact_sha256"]),
                target_registry_file_sha256=str(sealed["sha256"]),
                target_registry_file_bytes=int(sealed["bytes"]),
                plan_id=str(v5["plan_id"]),
                amendment_sha256=str(v5["amendment_sha256"]),
                sealed_at_utc=timestamp,
            )
            atomic_write_json(staging / RECEIPT_PATH, receipt, overwrite=False)
            staging.rename(target)
            _fsync_directory(target.parent)
            created = True
        except BaseException:
            if staging.exists():
                shutil.rmtree(staging)
            raise
    receipt, action = _verify_target(
        repository,
        target,
        operator,
        private_roster_artifact_sha256,
    )
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
        "created": created,
        "dataset_modified": created,
        "private_identity_material_created": private_created,
        "identity_initialization_mode": IDENTITY_INITIALIZATION_MODE,
        "historical_registry_available": False,
        "historical_registry_reused": False,
        "identity_digest_continuity_claimed": False,
        "bootstrap_implementation_sha256": receipt["bootstrap_implementation_sha256"],
        "bootstrap_implementation_bytes": receipt["bootstrap_implementation_bytes"],
        "target_preacquisition_plan_id": receipt["target_preacquisition_plan_id"],
        "target_preacquisition_amendment_sha256": receipt[
            "target_preacquisition_amendment_sha256"
        ],
        "target_registry_artifact_sha256": receipt["target_registry_artifact_sha256"],
        "bootstrap_receipt_artifact_sha256": receipt["artifact_sha256"],
        "operator_ids": [OPERATOR_ID],
        "operator_roles": list(OPERATOR_ROLES),
        "independent_verifier_available": False,
        "independent_preacquisition_attestation_claimed": False,
        "next_action": {
            "action_id": action["action_id"],
            "operator_role": action["operator_role"],
            "automatable": action["automatable"],
            "physical_acquisition_required": action["physical_acquisition_required"],
            "target_outcomes_permitted": action["target_outcomes_permitted"],
        },
        "target_outcomes_used": False,
        "device_nodes_opened": False,
        "physical_command_sent": False,
        "registered_method_changed": False,
        "physical_evidence_increment": 0,
        "claim_boundary": (
            "A fresh owner-only HMAC identity was initialized for v5 because the "
            "historical private identity material was unavailable. No historical "
            "identity-digest continuity or independent attestation is claimed."
        ),
    }
    report["report_sha256"] = _canonical_sha256(report, field="report_sha256")
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--private-identity-root", type=Path, required=True)
    parser.add_argument("--target-dataset-root", type=Path, required=True)
    parser.add_argument("--sealed-at-utc")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = bootstrap_single_operator_v5(
        repository_root=arguments.repository_root,
        private_identity_root=arguments.private_identity_root,
        target_dataset_root=arguments.target_dataset_root,
        sealed_at_utc=arguments.sealed_at_utc,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(arguments.output, report, overwrite=True)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
