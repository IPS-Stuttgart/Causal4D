from __future__ import annotations

import importlib.util
import json
import stat
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "seal_self_hosted_operator_registry.py"
)


def _load_sealer() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "causal4d_self_hosted_operator_registry_seal",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


sealer = _load_sealer()


def _identity() -> dict[str, str]:
    return {
        "protocol_id": "protocol-v1",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan-v1",
        "preacquisition_amendment_sha256": "b" * 64,
    }


def _template() -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DOperatorIdentityRegistryTemplate",
        "status": "template",
        **_identity(),
        "person_identity_digest_method": "hmac-sha256-domain-separated-v1",
        "sealed_at_utc": None,
        "sealed_by_operator_id": None,
        "target_outcomes_used": False,
        "operators": [],
        "artifact_sha256": None,
    }


def _decision(
    repository: Path,
    dataset: Path,
    *,
    before: bool,
) -> dict[str, object]:
    if before:
        action = {
            "action_id": "seal_operator_registry",
            "category": "manual_evidence",
            "operator_role": "principal_investigator",
            "physical_acquisition_required": False,
            "automatable": False,
            "changes_registered_method": False,
            "target_outcomes_permitted": False,
            "command_argv": sealer._expected_command(repository, dataset),
        }
    else:
        action = {
            "action_id": "complete_object_registration",
            "category": "manual_evidence",
            "operator_role": "acquisition_operator_and_independent_reviewer",
            "physical_acquisition_required": False,
            "automatable": False,
            "changes_registered_method": False,
            "target_outcomes_permitted": False,
            "command_argv": None,
        }
    return {
        "valid": True,
        "target_outcomes_used": False,
        "evidence_sha256": "c" * 64,
        "status_sha256": "d" * 64,
        "action": action,
    }


def test_fixed_roster_has_distinct_people_and_required_roles() -> None:
    private_roster = sealer._private_roster_payload()
    records = sealer._operator_records(b"k" * 32, private_roster)

    assert [record["operator_id"] for record in records] == [
        "environment.approver",
        "freezer.primary",
        "gate.operational",
        "verifier.independent",
    ]
    assert records[0]["roles"] == [
        "gate_approver",
        "software_environment_approver",
    ]
    assert records[1]["roles"] == ["freezer"]
    assert records[2]["roles"] == ["gate_approver"]
    assert records[3]["roles"] == ["independent_verifier"]
    assert len({record["person_identity_sha256"] for record in records}) == 4
    assert all(record["active"] is True for record in records)


def test_person_digest_is_domain_separated_and_deterministic() -> None:
    secret = b"s" * 32
    first = sealer._person_digest(secret, "person-name-v1:Alpha")
    repeated = sealer._person_digest(secret, "person-name-v1:Alpha")
    other = sealer._person_digest(secret, "person-name-v1:Beta")

    assert first == repeated
    assert first != other
    assert len(first) == 64


def test_private_key_is_created_once_with_owner_only_mode(
    tmp_path: Path,
) -> None:
    private = sealer._prepare_private_root(tmp_path / "private")
    first, created = sealer._ensure_private_key(private)
    second, repeated_created = sealer._ensure_private_key(private)
    key_path = private / sealer.KEY_FILENAME

    assert created is True
    assert repeated_created is False
    assert first == second
    assert len(first) == 32
    assert stat.S_IMODE(key_path.stat().st_mode) & 0o077 == 0


def test_private_key_rejects_group_read_access(tmp_path: Path) -> None:
    private = sealer._prepare_private_root(tmp_path / "private")
    key_path = private / sealer.KEY_FILENAME
    key_path.write_bytes(b"k" * 32)
    key_path.chmod(0o640)

    with pytest.raises(ValueError, match="accessible outside its owner"):
        sealer._ensure_private_key(private)


def test_private_roster_is_exactly_reused_and_tamper_rejected(
    tmp_path: Path,
) -> None:
    private = sealer._prepare_private_root(tmp_path / "private")
    first, created = sealer._ensure_private_roster(private)
    second, repeated_created = sealer._ensure_private_roster(private)

    assert created is True
    assert repeated_created is False
    assert first == second

    path = private / sealer.PRIVATE_ROSTER_FILENAME
    changed = dict(first)
    changed["target_outcomes_used"] = True
    path.write_text(json.dumps(changed), encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(ValueError, match="differs from the lock"):
        sealer._ensure_private_roster(private)


def test_execute_seals_only_the_registered_roster(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    private = tmp_path / "private"
    repository.mkdir()
    dataset.mkdir()
    template_path = dataset / sealer.OPERATOR_REGISTRY_TEMPLATE_PATH
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        json.dumps(_template(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dataset / "existing.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(
        sealer,
        "build_preacquisition_operator_next_action",
        lambda repo, root, verify_file_hashes: _decision(
            Path(repo),
            Path(root),
            before=not (Path(root) / sealer.OPERATOR_REGISTRY_PATH).exists(),
        ),
    )
    monkeypatch.setattr(
        sealer,
        "load_registered_preacquisition_chain",
        lambda repository_root: (
            {
                "protocol_id": "protocol-v1",
                "design_sha256": "a" * 64,
            },
            {},
            {},
            {
                "plan_id": "plan-v1",
                "amendment_sha256": "b" * 64,
            },
        ),
    )

    def seal_stub(
        repository_root: Path,
        dataset_root: Path,
        source_json: Path,
        *,
        sealed_by: str,
    ) -> dict[str, object]:
        source = json.loads(Path(source_json).read_text(encoding="utf-8"))
        target = Path(dataset_root) / sealer.OPERATOR_REGISTRY_PATH
        target.write_text(
            json.dumps(
                {
                    "operators": source["operators"],
                    "sealed_by_operator_id": sealed_by,
                },
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "artifact_sha256": "e" * 64,
            "sha256": "f" * 64,
            "bytes": target.stat().st_size,
        }

    monkeypatch.setattr(sealer, "seal_operator_registry", seal_stub)

    def load_stub(
        repository_root: Path,
        dataset_root: Path,
    ) -> tuple[dict[str, object], dict[str, object] | None]:
        target = Path(dataset_root) / sealer.OPERATOR_REGISTRY_PATH
        if not target.exists():
            return {"present": False, "valid": False}, None
        registry = json.loads(target.read_text(encoding="utf-8"))
        return (
            {
                "present": True,
                "valid": True,
                "artifact_sha256": "e" * 64,
                "sha256": "f" * 64,
                "bytes": target.stat().st_size,
                "active_role_counts": {
                    "freezer": 1,
                    "gate_approver": 2,
                    "independent_verifier": 1,
                    "software_environment_approver": 1,
                },
            },
            registry,
        )

    monkeypatch.setattr(sealer, "load_registered_operator_registry", load_stub)

    report = sealer.execute_operator_registry_seal(
        repository_root=repository,
        dataset_root=dataset,
        private_root=private,
    )

    assert report["already_sealed"] is False
    assert report["dataset_delta"] == {
        "added": ["preacquisition/operator_registry.json"],
        "removed": [],
        "modified": ["preacquisition/operator_registry.template.json"],
    }
    assert report["next_action"]["action_id"] == "complete_object_registration"
    assert report["target_outcomes_used"] is False
    assert report["physical_command_sent"] is False
    assert report["physical_evidence_increment"] == 0

    public_report = json.dumps(report, sort_keys=True)
    for private_value in (
        "Florian Pfaff",
        "Anna Seel",
        "Markus Rummel",
        "Michael Feurer",
        "person_identity_sha256",
    ):
        assert private_value not in public_report
