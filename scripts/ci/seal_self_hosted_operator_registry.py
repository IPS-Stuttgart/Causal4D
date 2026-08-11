#!/usr/bin/env python3
"""Seal the registered operator roster from workstation-local identity material."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import secrets
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from causal4d.atomic_io import atomic_write_json, atomic_write_text
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    PERSON_IDENTITY_DIGEST_METHOD,
    load_registered_operator_registry,
    seal_operator_registry,
    validate_operator_registry_template,
)
from causal4d.preacquisition_operator_flow import (
    build_preacquisition_operator_next_action,
)
from causal4d.preacquisition_readiness_contracts import (
    load_registered_preacquisition_chain,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_KIND = "Causal4DSelfHostedOperatorRegistrySeal"
PRIVATE_ROSTER_SCHEMA_VERSION = 1
PRIVATE_ROSTER_ARTIFACT_KIND = "Causal4DPrivateOperatorPrincipalRoster"
EXPECTED_ACTION_ID = "seal_operator_registry"
SEALED_BY_OPERATOR_ID = "freezer.primary"
KEY_FILENAME = "operator-identity-hmac-v1.key"
PRIVATE_ROSTER_FILENAME = "operator-principals-v1.json"
PERSON_DOMAIN = b"causal4d-operator-v1\0"
KEY_BYTES = 32
_MAXIMUM_SNAPSHOT_FILES = 50_000
_MAXIMUM_SNAPSHOT_BYTES = 4 * 1024**3

_ASSIGNMENTS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    (
        "environment.approver",
        "person-name-v1:Michael Feurer",
        ("gate_approver", "software_environment_approver"),
    ),
    (
        "freezer.primary",
        "person-name-v1:Florian Pfaff",
        ("freezer",),
    ),
    (
        "gate.operational",
        "person-name-v1:Markus Rummel",
        ("gate_approver",),
    ),
    (
        "verifier.independent",
        "person-name-v1:Anna Seel",
        ("independent_verifier",),
    ),
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
    _require_private_mode(
        path,
        name="private identity root",
        directory=True,
    )
    return path.resolve()


def _read_private_key(path: Path) -> bytes:
    _require(
        not _contains_symlink_component(path),
        "operator identity key contains a symlink component",
    )
    _require(path.is_file(), "operator identity key is not an ordinary file")
    _require_private_mode(path, name="operator identity key", directory=False)
    value = path.read_bytes()
    _require(len(value) == KEY_BYTES, "operator identity key length is invalid")
    return value


def _ensure_private_key(private_root: Path) -> tuple[bytes, bool]:
    path = private_root / KEY_FILENAME
    if os.path.lexists(path):
        return _read_private_key(path), False

    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    value = secrets.token_bytes(KEY_BYTES)
    try:
        descriptor = os.open(path, flags, 0o600)
    except FileExistsError:
        return _read_private_key(path), False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
        _fsync_directory(private_root)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return _read_private_key(path), True


def _private_roster_sha256(payload: Mapping[str, Any]) -> str:
    value = dict(payload)
    value.pop("artifact_sha256", None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _private_roster_payload() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": PRIVATE_ROSTER_SCHEMA_VERSION,
        "artifact_kind": PRIVATE_ROSTER_ARTIFACT_KIND,
        "person_identity_digest_method": PERSON_IDENTITY_DIGEST_METHOD,
        "assignments": [
            {
                "operator_id": operator_id,
                "canonical_principal": principal,
                "roles": list(roles),
            }
            for operator_id, principal, roles in _ASSIGNMENTS
        ],
        "target_outcomes_used": False,
    }
    payload["artifact_sha256"] = _private_roster_sha256(payload)
    return payload


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(
        not _contains_symlink_component(path),
        f"{name} contains a symlink component",
    )
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    return dict(value)


def _ensure_private_roster(private_root: Path) -> tuple[dict[str, Any], bool]:
    path = private_root / PRIVATE_ROSTER_FILENAME
    expected = _private_roster_payload()
    if os.path.lexists(path):
        _require(
            not _contains_symlink_component(path),
            "private operator roster contains a symlink component",
        )
        _require(path.is_file(), "private operator roster is not an ordinary file")
        _require_private_mode(path, name="private operator roster", directory=False)
        actual = _load_json_mapping(path, name="private operator roster")
        _require(actual == expected, "private operator roster differs from the lock")
        return actual, False

    atomic_write_json(path, expected, overwrite=False)
    os.chmod(path, 0o600)
    _fsync_directory(private_root)
    _require_private_mode(path, name="private operator roster", directory=False)
    actual = _load_json_mapping(path, name="private operator roster")
    _require(actual == expected, "private operator roster publication changed")
    return actual, True


def _person_digest(secret: bytes, principal: str) -> str:
    return hmac.new(
        secret,
        PERSON_DOMAIN + principal.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _operator_records(
    secret: bytes,
    private_roster: Mapping[str, Any],
) -> list[dict[str, Any]]:
    assignments = private_roster.get("assignments")
    _require(isinstance(assignments, list), "private operator assignments are missing")
    records: list[dict[str, Any]] = []
    for index, value in enumerate(assignments):
        _require(isinstance(value, Mapping), f"private assignment {index} is invalid")
        assignment = dict(cast(Mapping[str, Any], value))
        operator_id = assignment.get("operator_id")
        principal = assignment.get("canonical_principal")
        roles = assignment.get("roles")
        _require(isinstance(operator_id, str), "private operator id is invalid")
        _require(isinstance(principal, str), "private principal is invalid")
        _require(
            isinstance(roles, list) and all(isinstance(role, str) for role in roles),
            "private operator roles are invalid",
        )
        records.append(
            {
                "operator_id": operator_id,
                "person_identity_sha256": _person_digest(secret, principal),
                "active": True,
                "roles": sorted(roles),
            }
        )

    records.sort(key=lambda record: str(record["operator_id"]))
    operator_ids = [str(record["operator_id"]) for record in records]
    person_digests = [str(record["person_identity_sha256"]) for record in records]
    _require(
        len(operator_ids) == len(set(operator_ids)),
        "private roster contains duplicate operator ids",
    )
    _require(
        len(person_digests) == len(set(person_digests)),
        "private roster contains duplicate person identities",
    )
    return records


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


def _expected_command(
    repository_root: Path,
    dataset_root: Path,
) -> list[str]:
    template = dataset_root / OPERATOR_REGISTRY_TEMPLATE_PATH
    return [
        "causal4d",
        "protocol",
        "readiness",
        "seal-operator-registry",
        str(repository_root),
        str(dataset_root),
        str(template),
        "--sealed-by",
        "<registered-operator-id>",
    ]


def _require_registered_seal_action(
    decision: Mapping[str, Any],
    repository_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    _require(decision.get("valid") is True, "registered next action is invalid")
    _require(
        decision.get("target_outcomes_used") is False,
        "registered next action admits target outcomes",
    )
    value = decision.get("action")
    _require(isinstance(value, Mapping), "registered next action is missing")
    action = dict(cast(Mapping[str, Any], value))
    _require(
        action.get("action_id") == EXPECTED_ACTION_ID,
        "registered next action is not the operator registry seal",
    )
    _require(
        action.get("category") == "manual_evidence",
        "registered operator registry action has the wrong category",
    )
    _require(
        action.get("operator_role") == "principal_investigator",
        "operator registry seal requires an unexpected role",
    )
    _require(
        action.get("automatable") is False,
        "registered operator registry seal unexpectedly permits automation",
    )
    _require(
        action.get("physical_acquisition_required") is False,
        "operator registry seal unexpectedly requires physical acquisition",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "operator registry seal unexpectedly permits target outcomes",
    )
    _require(
        action.get("changes_registered_method") is False,
        "operator registry seal changes the registered method",
    )
    _require(
        action.get("command_argv")
        == _expected_command(repository_root, dataset_root),
        "operator registry seal command differs from the registered command",
    )
    return action


def _next_action_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    _require(decision.get("valid") is True, "post-seal next action is invalid")
    _require(
        decision.get("target_outcomes_used") is False,
        "post-seal next action admits target outcomes",
    )
    value = decision.get("action")
    _require(isinstance(value, Mapping), "post-seal next action is missing")
    action = dict(cast(Mapping[str, Any], value))
    _require(
        action.get("action_id") != EXPECTED_ACTION_ID,
        "operator registry seal did not advance the registered state",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "post-seal next action permits target outcomes",
    )
    return {
        "action_id": action.get("action_id"),
        "category": action.get("category"),
        "operator_role": action.get("operator_role"),
        "automatable": action.get("automatable") is True,
        "physical_acquisition_required": (
            action.get("physical_acquisition_required") is True
        ),
        "target_outcomes_permitted": False,
        "evidence_sha256": decision.get("evidence_sha256"),
        "status_sha256": decision.get("status_sha256"),
    }


def _registry_matches(
    registry: Mapping[str, Any],
    expected_records: Sequence[Mapping[str, Any]],
) -> None:
    actual = registry.get("operators")
    _require(isinstance(actual, list), "sealed operator registry has no operators")
    _require(
        actual == [dict(record) for record in expected_records],
        "sealed operator registry differs from the workstation roster",
    )
    _require(
        registry.get("sealed_by_operator_id") == SEALED_BY_OPERATOR_ID,
        "operator registry was sealed by an unexpected operator",
    )


def _report_sha256(report: Mapping[str, Any]) -> str:
    payload = dict(report)
    payload.pop("report_sha256", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def execute_operator_registry_seal(
    *,
    repository_root: Path,
    dataset_root: Path,
    private_root: Path,
) -> dict[str, Any]:
    repository = _require_ordinary_directory(
        repository_root,
        name="repository root",
    )
    dataset = _require_ordinary_directory(dataset_root, name="dataset root")
    private = _prepare_private_root(private_root)
    secret, key_created = _ensure_private_key(private)
    private_roster, roster_created = _ensure_private_roster(private)
    records = _operator_records(secret, private_roster)

    snapshot_before = _snapshot_regular_files(dataset)
    registry_status, existing_registry = load_registered_operator_registry(
        repository,
        dataset,
    )
    if registry_status.get("present") is True:
        _require(
            registry_status.get("valid") is True
            and isinstance(existing_registry, Mapping),
            "existing operator registry is invalid",
        )
        _registry_matches(
            cast(Mapping[str, Any], existing_registry),
            records,
        )
        decision_after = build_preacquisition_operator_next_action(
            repository,
            dataset,
            verify_file_hashes=True,
        )
        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": REPORT_ARTIFACT_KIND,
            "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
            "already_sealed": True,
            "operator_ids": [record["operator_id"] for record in records],
            "active_role_counts": registry_status.get("active_role_counts"),
            "registry_artifact_sha256": registry_status.get("artifact_sha256"),
            "registry_file_sha256": registry_status.get("sha256"),
            "registry_bytes": registry_status.get("bytes"),
            "dataset_delta": {"added": [], "removed": [], "modified": []},
            "snapshot_before_sha256": _snapshot_sha256(snapshot_before),
            "snapshot_after_sha256": _snapshot_sha256(snapshot_before),
            "private_material": {
                "stored_outside_dataset": True,
                "key_created": key_created,
                "roster_created": roster_created,
            },
            "next_action": _next_action_summary(decision_after),
            "target_outcomes_used": False,
            "device_nodes_opened": False,
            "physical_command_sent": False,
            "dataset_modified": False,
            "registered_method_changed": False,
            "physical_evidence_increment": 0,
            "claim_boundary": (
                "The exact fixed roster was already sealed and was verified "
                "against the workstation-private identity material. No gate, "
                "freeze, source execution, device access, target outcome, or "
                "physical evidence was produced."
            ),
        }
        report["report_sha256"] = _report_sha256(report)
        return report

    decision_before = build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )
    _require_registered_seal_action(
        decision_before,
        repository,
        dataset,
    )

    protocol, _, _, preacquisition_v4 = load_registered_preacquisition_chain(
        repository
    )
    template_path = dataset / OPERATOR_REGISTRY_TEMPLATE_PATH
    registry_path = dataset / OPERATOR_REGISTRY_PATH
    _require(
        not _contains_symlink_component(template_path),
        "operator registry template contains a symlink component",
    )
    _require(not registry_path.exists(), "operator registry is already sealed")
    original_text = template_path.read_text(encoding="utf-8")
    template = _load_json_mapping(
        template_path,
        name="operator registry template",
    )
    validate_operator_registry_template(
        protocol,
        preacquisition_v4,
        template,
    )
    current_records = template.get("operators")
    _require(
        current_records in ([], records),
        "operator registry template contains an unexpected roster",
    )

    template_changed = current_records == []
    if template_changed:
        template["operators"] = records
        atomic_write_json(template_path, template, overwrite=True)

    try:
        seal_result = seal_operator_registry(
            repository,
            dataset,
            template_path,
            sealed_by=SEALED_BY_OPERATOR_ID,
        )
    except BaseException:
        if template_changed and not registry_path.exists():
            atomic_write_text(template_path, original_text, overwrite=True)
        raise

    registry_status, sealed_registry = load_registered_operator_registry(
        repository,
        dataset,
    )
    _require(
        registry_status.get("valid") is True
        and isinstance(sealed_registry, Mapping),
        "published operator registry failed validation",
    )
    _registry_matches(cast(Mapping[str, Any], sealed_registry), records)

    snapshot_after = _snapshot_regular_files(dataset)
    delta = _snapshot_delta(snapshot_before, snapshot_after)
    expected_modified = (
        [OPERATOR_REGISTRY_TEMPLATE_PATH] if template_changed else []
    )
    _require(
        delta
        == {
            "added": [OPERATOR_REGISTRY_PATH],
            "removed": [],
            "modified": expected_modified,
        },
        "operator registry seal changed an unexpected dataset path",
    )

    decision_after = build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
        "already_sealed": False,
        "operator_ids": [record["operator_id"] for record in records],
        "active_role_counts": registry_status.get("active_role_counts"),
        "registry_artifact_sha256": seal_result.get("artifact_sha256"),
        "registry_file_sha256": seal_result.get("sha256"),
        "registry_bytes": seal_result.get("bytes"),
        "dataset_delta": delta,
        "snapshot_before_sha256": _snapshot_sha256(snapshot_before),
        "snapshot_after_sha256": _snapshot_sha256(snapshot_after),
        "private_material": {
            "stored_outside_dataset": True,
            "key_created": key_created,
            "roster_created": roster_created,
        },
        "next_action": _next_action_summary(decision_after),
        "target_outcomes_used": False,
        "device_nodes_opened": False,
        "physical_command_sent": False,
        "dataset_modified": True,
        "registered_method_changed": False,
        "physical_evidence_increment": 0,
        "claim_boundary": (
            "The fixed operator roster was populated and sealed exactly once. "
            "No readiness gate, method freeze, attestation, source execution, "
            "device access, target outcome, or physical evidence was produced."
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
    report = execute_operator_registry_seal(
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
