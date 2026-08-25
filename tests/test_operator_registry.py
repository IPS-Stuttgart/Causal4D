from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_readiness_contracts as readiness_contracts
from causal4d.operator_identity_integration import (
    validate_preacquisition_identity_bindings,
)
from causal4d.operator_registry import (
    OPERATOR_REGISTRY_ARTIFACT_KIND,
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    PERSON_IDENTITY_DIGEST_METHOD,
    ROLE_FREEZER,
    ROLE_GATE_APPROVER,
    ROLE_INDEPENDENT_VERIFIER,
    ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
    load_operator_registry_prerequisite,
    operator_registry_sha256,
    operator_registry_template,
    require_distinct_operator_people,
    resolve_operator,
    scaffold_operator_registry,
    seal_operator_registry,
    validate_attestation_operator_identities,
    validate_gate_approver_identity,
    validate_operator_registry,
)
from causal4d.preacquisition_readiness_contracts import (
    GATE_PATHS,
    readiness_evidence_sha256,
)


def _registered_values() -> tuple[dict, dict]:
    protocol = {
        "protocol_id": "test-protocol",
        "design_sha256": "a" * 64,
    }
    v4 = {
        "plan_id": "test-preacquisition-v4",
        "amendment_sha256": "b" * 64,
    }
    return protocol, v4


def _operators() -> list[dict]:
    return [
        {
            "operator_id": "freezer.primary",
            "person_identity_sha256": "1" * 64,
            "active": True,
            "roles": [ROLE_FREEZER, ROLE_GATE_APPROVER],
        },
        {
            "operator_id": "verifier.independent",
            "person_identity_sha256": "2" * 64,
            "active": True,
            "roles": [
                ROLE_GATE_APPROVER,
                ROLE_INDEPENDENT_VERIFIER,
                ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
            ],
        },
    ]


def _sealed_registry() -> dict:
    protocol, v4 = _registered_values()
    registry = operator_registry_template(protocol, v4)
    registry.update(
        {
            "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
            "status": "sealed",
            "sealed_at_utc": "2026-07-30T08:00:00Z",
            "sealed_by_operator_id": "freezer.primary",
            "operators": _operators(),
        }
    )
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    return registry


def _single_operator_registry() -> dict:
    protocol, v4 = _registered_values()
    registry = operator_registry_template(protocol, v4)
    registry.update(
        {
            "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
            "status": "sealed",
            "sealed_at_utc": "2026-07-30T08:00:00Z",
            "sealed_by_operator_id": "florianpfaff",
            "operators": [
                {
                    "operator_id": "florianpfaff",
                    "person_identity_sha256": "3" * 64,
                    "active": True,
                    "roles": [
                        ROLE_FREEZER,
                        ROLE_GATE_APPROVER,
                        ROLE_SOFTWARE_ENVIRONMENT_APPROVER,
                    ],
                }
            ],
        }
    )
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    return registry


def _draft() -> dict:
    protocol, v4 = _registered_values()
    draft = operator_registry_template(protocol, v4)
    draft["operators"] = _operators()
    return draft


def test_valid_registry_binds_roles_without_raw_identity_fields() -> None:
    protocol, v4 = _registered_values()
    registry = _sealed_registry()

    result = validate_operator_registry(protocol, v4, registry)

    assert result["passed"] is True
    assert result["operator_count"] == 2
    assert result["active_role_counts"][ROLE_FREEZER] == 1
    assert result["person_identity_digest_method"] == PERSON_IDENTITY_DIGEST_METHOD
    assert set(registry["operators"][0]) == {
        "operator_id",
        "person_identity_sha256",
        "active",
        "roles",
    }


def test_single_person_registry_is_truthful_but_not_independent() -> None:
    protocol, v4 = _registered_values()
    registry = _single_operator_registry()

    result = validate_operator_registry(protocol, v4, registry)

    assert result["passed"] is True
    assert result["operator_count"] == 1
    assert result["active_role_counts"][ROLE_INDEPENDENT_VERIFIER] == 0
    assert result["independent_verifier_available"] is False


def test_duplicate_person_digest_rejects_operator_aliases() -> None:
    protocol, v4 = _registered_values()
    registry = _sealed_registry()
    registry["operators"][1]["person_identity_sha256"] = "1" * 64
    registry["artifact_sha256"] = operator_registry_sha256(registry)

    with pytest.raises(ValueError, match="duplicate person identity"):
        validate_operator_registry(protocol, v4, registry)


def test_unknown_inactive_and_wrong_role_operators_fail_closed() -> None:
    registry = _sealed_registry()

    with pytest.raises(ValueError, match="not registered"):
        resolve_operator(registry, "unknown", required_role=ROLE_FREEZER)

    inactive = deepcopy(registry)
    inactive["operators"][0]["active"] = False
    with pytest.raises(ValueError, match="inactive"):
        resolve_operator(inactive, "freezer.primary", required_role=ROLE_FREEZER)

    with pytest.raises(ValueError, match="lacks required role"):
        resolve_operator(
            registry,
            "verifier.independent",
            required_role=ROLE_FREEZER,
        )


def test_person_level_independence_rejects_two_alias_ids() -> None:
    first = {
        "operator_id": "florian-pfaff",
        "person_identity_sha256": "9" * 64,
    }
    alias = {
        "operator_id": "f-pfaff",
        "person_identity_sha256": "9" * 64,
    }

    with pytest.raises(ValueError, match="distinct registered person"):
        require_distinct_operator_people(
            first,
            alias,
            relationship=(
                "method freeze must be verified by a distinct registered person"
            ),
        )


def test_target_outcomes_are_neither_accepted_nor_schema_fields() -> None:
    protocol, v4 = _registered_values()
    registry = _sealed_registry()
    registry["target_outcomes_used"] = True
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    with pytest.raises(ValueError, match="target outcomes"):
        validate_operator_registry(protocol, v4, registry)

    registry = _sealed_registry()
    registry["target_outcomes"] = {"execution-01": 0.0}
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    with pytest.raises(ValueError, match="fields differ"):
        validate_operator_registry(protocol, v4, registry)


def test_freezer_and_verifier_must_resolve_to_distinct_people() -> None:
    registry = _sealed_registry()
    method_freeze = {
        "frozen_by": "freezer.primary",
        "frozen_at_utc": "2026-07-30T09:00:00Z",
    }
    attestation = {
        "verifier_id": "verifier.independent",
        "verified_at_utc": "2026-07-30T09:05:00Z",
    }

    freezer, verifier = validate_attestation_operator_identities(
        method_freeze,
        attestation,
        registry,
    )

    assert freezer["person_identity_sha256"] != verifier["person_identity_sha256"]


def test_software_environment_approval_has_an_independent_role() -> None:
    registry = _sealed_registry()

    approver = validate_gate_approver_identity(
        "software_environment_locked",
        "verifier.independent",
        "2026-07-30T09:10:00Z",
        registry,
        freezer_person_identity_sha256="1" * 64,
    )
    assert approver["operator_id"] == "verifier.independent"

    with pytest.raises(ValueError, match="independent approval role"):
        validate_gate_approver_identity(
            "software_environment_locked",
            "freezer.primary",
            "2026-07-30T09:10:00Z",
            registry,
            freezer_person_identity_sha256="1" * 64,
        )


def test_registry_prerequisite_reports_a_valid_existing_template(
    tmp_path: Path,
) -> None:
    protocol, v4 = _registered_values()
    template_path = tmp_path / OPERATOR_REGISTRY_TEMPLATE_PATH
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        json.dumps(operator_registry_template(protocol, v4)),
        encoding="utf-8",
    )

    result, registry = load_operator_registry_prerequisite(
        protocol,
        v4,
        tmp_path,
    )

    assert registry is None
    assert result["present"] is False
    assert result["template_status"]["present"] is True
    assert result["template_status"]["valid"] is True
    assert result["template_status"]["operator_count"] == 0
    assert result["template_status"]["target_outcomes_used"] is False


def test_registry_prerequisite_reports_an_invalid_existing_template(
    tmp_path: Path,
) -> None:
    protocol, v4 = _registered_values()
    template = operator_registry_template(protocol, v4)
    template["target_outcomes_used"] = True
    template_path = tmp_path / OPERATOR_REGISTRY_TEMPLATE_PATH
    template_path.parent.mkdir(parents=True)
    template_path.write_text(json.dumps(template), encoding="utf-8")

    result, registry = load_operator_registry_prerequisite(
        protocol,
        v4,
        tmp_path,
    )

    assert registry is None
    assert result["present"] is False
    assert result["template_status"]["present"] is True
    assert result["template_status"]["valid"] is False
    assert "target outcomes" in result["template_status"]["error"]


def test_registry_scaffold_and_seal_are_non_overwriting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v4 = _registered_values()
    monkeypatch.setattr(
        readiness_contracts,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, {}, {}, v4),
    )

    first = scaffold_operator_registry(tmp_path, tmp_path)
    second = scaffold_operator_registry(tmp_path, tmp_path)
    assert first["created"] is True
    assert second["existing"] is True

    template_path = tmp_path / OPERATOR_REGISTRY_TEMPLATE_PATH
    draft = json.loads(template_path.read_text(encoding="utf-8"))
    draft["operators"] = _operators()
    template_path.write_text(json.dumps(draft), encoding="utf-8")

    result = seal_operator_registry(
        tmp_path,
        tmp_path,
        template_path,
        sealed_by="freezer.primary",
        sealed_at_utc="2026-07-30T08:00:00Z",
    )
    assert result["valid"] is True
    assert (tmp_path / OPERATOR_REGISTRY_PATH).is_file()

    with pytest.raises(ValueError, match="already sealed"):
        seal_operator_registry(
            tmp_path,
            tmp_path,
            template_path,
            sealed_by="freezer.primary",
        )


def test_registry_must_precede_freeze_and_gate_approvals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    protocol, v4 = _registered_values()
    monkeypatch.setattr(
        readiness_contracts,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, {}, {}, v4),
    )
    draft_path = tmp_path / "draft.json"
    draft_path.write_text(json.dumps(_draft()), encoding="utf-8")
    (tmp_path / "method_freeze.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="before method-freeze evidence"):
        seal_operator_registry(
            tmp_path,
            tmp_path,
            draft_path,
            sealed_by="freezer.primary",
        )


def test_portable_readiness_identity_binds_registry_not_mount_path() -> None:
    status = {
        "dataset_root": "/mnt/acquisition",
        "prerequisites": {
            "operator_registry": {
                "path": "/mnt/acquisition/preacquisition/operator_registry.json",
                "present": True,
                "valid": True,
                "artifact_sha256": "a" * 64,
            }
        },
        "operational_gates": {},
        "evidence_sha256": None,
        "status_sha256": None,
    }
    first = readiness_evidence_sha256(status)
    relocated = deepcopy(status)
    relocated["dataset_root"] = "/archive/acquisition"
    relocated["prerequisites"]["operator_registry"]["path"] = (
        "/archive/acquisition/preacquisition/operator_registry.json"
    )
    assert readiness_evidence_sha256(relocated) == first

    changed = deepcopy(relocated)
    changed["prerequisites"]["operator_registry"]["artifact_sha256"] = "b" * 64
    assert readiness_evidence_sha256(changed) != first


def _write_identity_governed_sources(
    root: Path,
    *,
    contact_approver: str = "verifier.independent",
    verifier_id: str = "verifier.independent",
) -> None:
    sources = {
        "method_freeze.json": {
            "status": "sealed",
            "frozen_by": "freezer.primary",
            "frozen_at_utc": "2026-07-30T09:00:00Z",
        },
        "method_freeze_validation.json": {
            "validation_passed": True,
            "verifier_id": verifier_id,
            "verified_at_utc": "2026-07-30T09:05:00Z",
        },
        "timebase_calibration.json": {
            "status": "approved",
            "approval": {
                "approved": True,
                "approver_id": "verifier.independent",
                "approved_at_utc": "2026-07-30T09:10:00Z",
            },
        },
        "contact_registration.json": {
            "approval": {
                "approved": True,
                "approver_id": contact_approver,
                "approved_at_utc": "2026-07-30T09:15:00Z",
            }
        },
    }
    for relative, payload in sources.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")
    for gate_id, relative in GATE_PATHS.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "status": "passed",
                    "approval": {
                        "approved": True,
                        "approver_id": "verifier.independent",
                        "approved_at_utc": "2026-07-30T09:20:00Z",
                    },
                    "gate_id": gate_id,
                }
            ),
            encoding="utf-8",
        )


def test_identity_bindings_cover_all_governed_approvals(tmp_path: Path) -> None:
    registry = _sealed_registry()
    _write_identity_governed_sources(tmp_path)

    result = validate_preacquisition_identity_bindings(tmp_path, registry)

    assert result["valid"] is True
    assert result["freezer_operator_id"] == "freezer.primary"
    assert result["verifier_operator_id"] == "verifier.independent"
    assert result["approval_bindings"]["timebase_calibration"]["operator_id"] == (
        "verifier.independent"
    )
    assert result["approval_bindings"]["contact_registration"]["operator_id"] == (
        "verifier.independent"
    )
    assert len(result["source_sha256"]) == 4 + len(GATE_PATHS)


def test_identity_bindings_reject_unknown_contact_approver(tmp_path: Path) -> None:
    registry = _sealed_registry()
    _write_identity_governed_sources(tmp_path, contact_approver="unknown.alias")

    result = validate_preacquisition_identity_bindings(tmp_path, registry)

    assert result["valid"] is False
    assert result["template"] is False
    assert "not registered" in result["error"]


def test_identity_bindings_reject_nonindependent_freeze_verifier(
    tmp_path: Path,
) -> None:
    registry = _sealed_registry()
    _write_identity_governed_sources(tmp_path, verifier_id="freezer.primary")

    result = validate_preacquisition_identity_bindings(tmp_path, registry)

    assert result["valid"] is False
    assert "lacks required role" in result["error"]


def test_v5_allows_truthful_freezer_self_attestation() -> None:
    registry = _single_operator_registry()
    method_freeze = {
        "frozen_by": "florianpfaff",
        "frozen_at_utc": "2026-07-30T09:00:00Z",
    }
    attestation = {
        "verifier_id": "florianpfaff",
        "verified_at_utc": "2026-07-30T09:05:00Z",
    }

    freezer, attester = validate_attestation_operator_identities(
        method_freeze,
        attestation,
        registry,
        allow_self_attestation=True,
    )

    assert freezer["operator_id"] == "florianpfaff"
    assert attester["operator_id"] == freezer["operator_id"]


def test_v5_allows_registered_software_environment_self_approval() -> None:
    registry = _single_operator_registry()

    approver = validate_gate_approver_identity(
        "software_environment_locked",
        "florianpfaff",
        "2026-07-30T09:10:00Z",
        registry,
        freezer_person_identity_sha256="3" * 64,
        allow_software_environment_self_approval=True,
    )

    assert approver["operator_id"] == "florianpfaff"
