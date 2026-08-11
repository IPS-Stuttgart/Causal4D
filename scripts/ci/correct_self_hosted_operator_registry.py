#!/usr/bin/env python3
"""Correct the unsupported workstation2 operator roster without physical work."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from causal4d.atomic_io import atomic_write_binary, atomic_write_json
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_ARTIFACT_KIND,
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    PERSON_IDENTITY_DIGEST_METHOD,
    ROLE_FREEZER,
    ROLE_GATE_APPROVER,
    ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
    operator_registry_sha256,
    operator_registry_template,
    validate_operator_registry,
    validate_operator_registry_template,
)
from causal4d.preacquisition_operator_flow import (
    build_preacquisition_operator_next_action,
)
from causal4d.preacquisition_readiness_contracts import (
    GATE_PATHS,
    load_registered_preacquisition_chain,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_KIND = "Causal4DSelfHostedOperatorRegistryCorrection"
CORRECTION_SCHEMA_VERSION = 1
CORRECTION_ARTIFACT_KIND = "Causal4DOperatorRegistryCorrection"
CORRECTION_PATH = "preacquisition/operator_registry_correction_v1.json"
PRIVATE_ROSTER_SCHEMA_VERSION = 2
PRIVATE_ROSTER_ARTIFACT_KIND = "Causal4DPrivateOperatorPrincipalRoster"
KEY_FILENAME = "operator-identity-hmac-v1.key"
PRIVATE_ROSTER_FILENAME = "operator-principals-v1.json"
PERSON_DOMAIN = b"causal4d-operator-v1\0"
KEY_BYTES = 32
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
EXPECTED_OLD_REGISTRY_ARTIFACT_SHA256 = (
    "12dd49f2e32aaee28502cbeffd30176b1df44fe899bf0a9781e461326f6418b1"
)
EXPECTED_OLD_REGISTRY_FILE_SHA256 = (
    "7b428b75d8c60098f638af9ebf4c64ef40df1e6fba2189b7f0deb58b144046e7"
)
EXPECTED_NEXT_ACTION = "stop_independent_verifier_unavailable"
_MAXIMUM_SNAPSHOT_FILES = 50_000
_MAXIMUM_SNAPSHOT_BYTES = 4 * 1024**3
_FORBIDDEN_PUBLIC_IDENTITIES = (
    "Anna Seel",
    "Markus Rummel",
    "Michael Feurer",
    "environment.approver",
    "freezer.primary",
    "gate.operational",
    "verifier.independent",
)
_GOVERNED_FILES = (
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _require_ordinary_directory(path: Path, *, name: str) -> Path:
    _require(
        not _contains_symlink_component(path),
        f"{name} contains a symlink component",
    )
    _require(path.is_dir(), f"{name} must be an ordinary directory")
    return path.resolve()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_private_mode(path: Path, *, name: str, directory: bool) -> None:
    mode = stat.S_IMODE(path.stat().st_mode)
    _require(mode & 0o077 == 0, f"{name} is accessible outside its owner")
    if directory:
        _require(mode & 0o700 == 0o700, f"{name} lacks owner directory access")
    else:
        _require(mode & 0o400 == 0o400, f"{name} lacks owner read access")


def _prepare_private_root(path: Path) -> Path:
    _require(
        not _contains_symlink_component(path),
        "private identity root contains a symlink component",
    )
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _require(path.is_dir(), "private identity root is not a directory")
    os.chmod(path, 0o700)
    _require_private_mode(path, name="private identity root", directory=True)
    return path.resolve()


def _read_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(not _contains_symlink_component(path), f"{name} contains a symlink")
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return dict(value)


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _snapshot_regular_files(root: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        _require(not path.is_symlink(), f"dataset contains a symlink: {path}")
        if path.is_dir():
            continue
        _require(path.is_file(), f"dataset contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        sha256, byte_count = _sha256_file(path)
        snapshot[relative] = {"sha256": sha256, "bytes": byte_count}
        total_bytes += byte_count
        _require(
            len(snapshot) <= _MAXIMUM_SNAPSHOT_FILES,
            "dataset snapshot exceeds the registered file-count guard",
        )
        _require(
            total_bytes <= _MAXIMUM_SNAPSHOT_BYTES,
            "dataset snapshot exceeds the registered byte-count guard",
        )
    return snapshot


def _snapshot_sha256(snapshot: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_delta(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    before_paths = set(before)
    after_paths = set(after)
    return {
        "added": sorted(after_paths - before_paths),
        "removed": sorted(before_paths - after_paths),
        "modified": sorted(
            path
            for path in before_paths & after_paths
            if dict(before[path]) != dict(after[path])
        ),
    }


def _canonical_sha256(payload: Mapping[str, Any], *, digest_field: str) -> str:
    value = dict(payload)
    value.pop(digest_field, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


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
        digest_field="artifact_sha256",
    )
    return payload


def _replace_private_material(private_root: Path) -> bytes:
    secret = secrets.token_bytes(KEY_BYTES)
    key_path = private_root / KEY_FILENAME
    roster_path = private_root / PRIVATE_ROSTER_FILENAME

    def write_key(handle: Any) -> None:
        handle.write(secret)

    atomic_write_binary(key_path, write_key, overwrite=True)
    os.chmod(key_path, 0o600)
    atomic_write_json(roster_path, _private_roster_payload(), overwrite=True)
    os.chmod(roster_path, 0o600)
    _fsync_directory(private_root)
    _require_private_mode(key_path, name="operator identity key", directory=False)
    _require_private_mode(roster_path, name="private operator roster", directory=False)
    _require(key_path.read_bytes() == secret, "private key publication changed")
    _require(
        _read_json_mapping(roster_path, name="private operator roster")
        == _private_roster_payload(),
        "private operator roster publication changed",
    )
    return secret


def _read_current_private_material(private_root: Path) -> bytes:
    key_path = private_root / KEY_FILENAME
    roster_path = private_root / PRIVATE_ROSTER_FILENAME
    _require_private_mode(key_path, name="operator identity key", directory=False)
    _require_private_mode(roster_path, name="private operator roster", directory=False)
    secret = key_path.read_bytes()
    _require(len(secret) == KEY_BYTES, "operator identity key length is invalid")
    _require(
        _read_json_mapping(roster_path, name="private operator roster")
        == _private_roster_payload(),
        "private operator roster differs from the single-person lock",
    )
    return secret


def _require_no_governed_evidence(dataset: Path) -> None:
    for relative in _GOVERNED_FILES:
        _require(
            not os.path.lexists(dataset / relative),
            f"operator correction is too late; governed evidence exists: {relative}",
        )
    for pattern in _GOVERNED_PATTERNS:
        _require(
            not any(dataset.glob(pattern)),
            f"operator correction is too late; governed evidence matches: {pattern}",
        )
    for gate_id, relative in GATE_PATHS.items():
        path = dataset / relative
        if not path.is_file():
            continue
        gate = _read_json_mapping(path, name=f"{gate_id} gate")
        approval = gate.get("approval")
        _require(
            gate.get("status") == "template"
            and isinstance(approval, Mapping)
            and approval.get("approved") is False,
            f"operator correction is too late; gate was approved: {gate_id}",
        )


def _build_registry_payloads(
    repository: Path,
    secret: bytes,
    *,
    sealed_at_utc: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    protocol, _, _, preacquisition_v4 = load_registered_preacquisition_chain(repository)
    template = operator_registry_template(protocol, preacquisition_v4)
    template["operators"] = [_operator_record(secret)]
    validate_operator_registry_template(protocol, preacquisition_v4, template)
    registry = {
        **template,
        "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
        "status": "sealed",
        "sealed_at_utc": sealed_at_utc,
        "sealed_by_operator_id": OPERATOR_ID,
    }
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    summary = validate_operator_registry(protocol, preacquisition_v4, registry)
    _require(
        summary.get("independent_verifier_available") is False,
        "single-person registry unexpectedly claims independent verification",
    )
    return template, registry, summary


def _correction_payload(
    *,
    corrected_at_utc: str,
    new_registry: Mapping[str, Any],
    new_file_sha256: str,
    new_file_bytes: int,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CORRECTION_SCHEMA_VERSION,
        "artifact_kind": CORRECTION_ARTIFACT_KIND,
        "correction_reason": "unsupported_operator_identities_removed",
        "corrected_at_utc": corrected_at_utc,
        "old_registry_artifact_sha256": EXPECTED_OLD_REGISTRY_ARTIFACT_SHA256,
        "old_registry_file_sha256": EXPECTED_OLD_REGISTRY_FILE_SHA256,
        "old_operator_count": 4,
        "new_registry_artifact_sha256": new_registry["artifact_sha256"],
        "new_registry_file_sha256": new_file_sha256,
        "new_registry_file_bytes": new_file_bytes,
        "new_operator_id": OPERATOR_ID,
        "new_operator_count": 1,
        "independent_verifier_available": False,
        "governed_evidence_present": False,
        "target_outcomes_used": False,
        "physical_evidence_increment": 0,
    }
    payload["artifact_sha256"] = _canonical_sha256(
        payload,
        digest_field="artifact_sha256",
    )
    return payload


def _require_no_forbidden_identity_text(*roots: Path) -> None:
    for root in roots:
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_dir():
                continue
            _require(path.is_file() and not path.is_symlink(), f"unsafe path: {path}")
            data = path.read_bytes()
            for value in _FORBIDDEN_PUBLIC_IDENTITIES:
                _require(
                    value.encode("utf-8") not in data,
                    f"unsupported operator identity remains in {path}",
                )


def _next_action_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    _require(decision.get("valid") is True, "post-correction next action is invalid")
    action_value = decision.get("action")
    _require(isinstance(action_value, Mapping), "post-correction action is missing")
    action = dict(cast(Mapping[str, Any], action_value))
    _require(
        action.get("action_id") == EXPECTED_NEXT_ACTION,
        "post-correction action does not expose the single-person blocker",
    )
    _require(
        action.get("category") == "governance_blocker",
        "post-correction action has the wrong category",
    )
    _require(
        action.get("automatable") is False,
        "single-person governance blocker unexpectedly permits automation",
    )
    _require(
        action.get("physical_acquisition_required") is False,
        "single-person governance blocker unexpectedly permits physical work",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "single-person governance blocker permits target outcomes",
    )
    return {
        "action_id": action["action_id"],
        "category": action["category"],
        "operator_role": action["operator_role"],
        "automatable": False,
        "physical_acquisition_required": False,
        "target_outcomes_permitted": False,
        "blocking_items": action.get("blocking_items"),
        "evidence_sha256": decision.get("evidence_sha256"),
        "status_sha256": decision.get("status_sha256"),
    }


def _report_sha256(report: Mapping[str, Any]) -> str:
    return _canonical_sha256(report, digest_field="report_sha256")


def execute_operator_registry_correction(
    *,
    repository_root: Path,
    dataset_root: Path,
    private_root: Path,
) -> dict[str, Any]:
    repository = _require_ordinary_directory(repository_root, name="repository root")
    dataset = _require_ordinary_directory(dataset_root, name="dataset root")
    private = _prepare_private_root(private_root)
    registry_path = dataset / OPERATOR_REGISTRY_PATH
    template_path = dataset / OPERATOR_REGISTRY_TEMPLATE_PATH
    correction_path = dataset / CORRECTION_PATH
    _require_no_governed_evidence(dataset)
    snapshot_before = _snapshot_regular_files(dataset)

    already_corrected = correction_path.is_file()
    if already_corrected:
        correction = _read_json_mapping(correction_path, name="correction receipt")
        _require(
            correction.get("artifact_kind") == CORRECTION_ARTIFACT_KIND,
            "unexpected correction receipt",
        )
        _require(
            correction.get("artifact_sha256")
            == _canonical_sha256(correction, digest_field="artifact_sha256"),
            "correction receipt digest mismatch",
        )
        secret = _read_current_private_material(private)
        _, expected_registry, summary = _build_registry_payloads(
            repository,
            secret,
            sealed_at_utc=str(correction["corrected_at_utc"]),
        )
        actual_registry = _read_json_mapping(registry_path, name="operator registry")
        _require(actual_registry == expected_registry, "corrected registry changed")
        _require_no_forbidden_identity_text(dataset, private)
        decision = build_preacquisition_operator_next_action(
            repository,
            dataset,
            verify_file_hashes=True,
        )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": REPORT_ARTIFACT_KIND,
            "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
            "already_corrected": True,
            "dataset_modified": False,
            "dataset_delta": {"added": [], "removed": [], "modified": []},
            "snapshot_before_sha256": _snapshot_sha256(snapshot_before),
            "snapshot_after_sha256": _snapshot_sha256(snapshot_before),
            "operator_ids": [OPERATOR_ID],
            "active_role_counts": summary["active_role_counts"],
            "independent_verifier_available": False,
            "registry_artifact_sha256": actual_registry["artifact_sha256"],
            "registry_file_sha256": _sha256_file(registry_path)[0],
            "correction_artifact_sha256": correction["artifact_sha256"],
            "next_action": _next_action_summary(decision),
            "private_material_replaced": False,
            "target_outcomes_used": False,
            "device_nodes_opened": False,
            "physical_command_sent": False,
            "registered_method_changed": False,
            "physical_evidence_increment": 0,
            "claim_boundary": (
                "The corrected single-person registry was verified. Independent "
                "verification remains unavailable and governed acquisition stays "
                "blocked."
            ),
        }
        report["report_sha256"] = _report_sha256(report)
        return report

    old_file_sha256, _ = _sha256_file(registry_path)
    _require(
        old_file_sha256 == EXPECTED_OLD_REGISTRY_FILE_SHA256,
        "existing registry file is not the known unsupported roster",
    )
    old_registry = _read_json_mapping(registry_path, name="operator registry")
    _require(
        old_registry.get("artifact_sha256")
        == EXPECTED_OLD_REGISTRY_ARTIFACT_SHA256,
        "existing registry artifact is not the known unsupported roster",
    )
    _require(
        isinstance(old_registry.get("operators"), list)
        and len(old_registry["operators"]) == 4,
        "known unsupported registry operator count changed",
    )
    _require(
        old_registry.get("target_outcomes_used") is False,
        "target outcomes entered the unsupported registry",
    )
    old_template = _read_json_mapping(template_path, name="operator registry template")
    _require(
        old_template.get("status") == "template",
        "operator registry template is not an unsealed draft",
    )

    corrected_at_utc = datetime.now(timezone.utc).isoformat()
    secret = _replace_private_material(private)
    template, registry, summary = _build_registry_payloads(
        repository,
        secret,
        sealed_at_utc=corrected_at_utc,
    )
    atomic_write_json(template_path, template, overwrite=True)
    atomic_write_json(registry_path, registry, overwrite=True)
    new_file_sha256, new_file_bytes = _sha256_file(registry_path)
    correction = _correction_payload(
        corrected_at_utc=corrected_at_utc,
        new_registry=registry,
        new_file_sha256=new_file_sha256,
        new_file_bytes=new_file_bytes,
    )
    atomic_write_json(correction_path, correction, overwrite=False)

    snapshot_after = _snapshot_regular_files(dataset)
    delta = _snapshot_delta(snapshot_before, snapshot_after)
    _require(
        delta
        == {
            "added": [CORRECTION_PATH],
            "removed": [],
            "modified": [OPERATOR_REGISTRY_PATH, OPERATOR_REGISTRY_TEMPLATE_PATH],
        },
        "operator correction changed an unexpected dataset path",
    )
    _require_no_forbidden_identity_text(dataset, private)
    decision = build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
        "already_corrected": False,
        "dataset_modified": True,
        "dataset_delta": delta,
        "snapshot_before_sha256": _snapshot_sha256(snapshot_before),
        "snapshot_after_sha256": _snapshot_sha256(snapshot_after),
        "operator_ids": [OPERATOR_ID],
        "active_role_counts": summary["active_role_counts"],
        "independent_verifier_available": False,
        "registry_artifact_sha256": registry["artifact_sha256"],
        "registry_file_sha256": new_file_sha256,
        "registry_file_bytes": new_file_bytes,
        "correction_artifact_sha256": correction["artifact_sha256"],
        "next_action": _next_action_summary(decision),
        "private_material_replaced": True,
        "target_outcomes_used": False,
        "device_nodes_opened": False,
        "physical_command_sent": False,
        "registered_method_changed": False,
        "physical_evidence_increment": 0,
        "claim_boundary": (
            "Unsupported operator identities were removed and replaced by the "
            "single real participant. Independent verification is explicitly "
            "unavailable, so governed acquisition remains blocked."
        ),
    }
    report["report_sha256"] = _report_sha256(report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--private-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = execute_operator_registry_correction(
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        private_root=arguments.private_root,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
