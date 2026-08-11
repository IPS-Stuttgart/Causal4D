from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from causal4d.operator_registry import (
    OPERATOR_REGISTRY_ARTIFACT_KIND,
    OPERATOR_REGISTRY_PATH,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    operator_registry_sha256,
    operator_registry_template,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "correct_self_hosted_operator_registry.py"
)


def _load_corrector() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "causal4d_self_hosted_operator_registry_correction",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


corrector = _load_corrector()


def _identity() -> tuple[dict[str, str], dict[str, str]]:
    return (
        {
            "protocol_id": "protocol-v1",
            "design_sha256": "a" * 64,
        },
        {
            "plan_id": "plan-v1",
            "amendment_sha256": "b" * 64,
        },
    )


def _unsupported_registry() -> tuple[dict, dict]:
    protocol, v4 = _identity()
    template = operator_registry_template(protocol, v4)
    template["operators"] = [
        {
            "operator_id": f"unsupported-{index}",
            "person_identity_sha256": str(index + 1) * 64,
            "active": True,
            "roles": ["gate_approver"],
        }
        for index in range(4)
    ]
    registry = {
        **template,
        "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
        "status": "sealed",
        "sealed_at_utc": "2026-08-11T10:00:00+00:00",
        "sealed_by_operator_id": "unsupported-0",
    }
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    return template, registry


def _decision() -> dict:
    return {
        "valid": True,
        "evidence_sha256": "c" * 64,
        "status_sha256": "d" * 64,
        "action": {
            "action_id": "stop_independent_verifier_unavailable",
            "category": "governance_blocker",
            "operator_role": "principal_investigator",
            "automatable": False,
            "physical_acquisition_required": False,
            "target_outcomes_permitted": False,
            "blocking_items": [
                "single_operator_project_cannot_satisfy_independent_verification"
            ],
        },
    }


def test_single_operator_record_has_no_independent_role() -> None:
    record = corrector._operator_record(b"k" * 32)

    assert record["operator_id"] == "florianpfaff"
    assert record["roles"] == [
        "freezer",
        "gate_approver",
        "software_environment_approver",
    ]
    assert "independent_verifier" not in record["roles"]
    assert len(record["person_identity_sha256"]) == 64


def test_correction_replaces_only_the_known_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    private = tmp_path / "private"
    repository.mkdir()
    dataset.mkdir()
    private.mkdir(mode=0o700)
    template, registry = _unsupported_registry()
    template_path = dataset / OPERATOR_REGISTRY_TEMPLATE_PATH
    registry_path = dataset / OPERATOR_REGISTRY_PATH
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        json.dumps(template, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    old_file_sha256, _ = corrector._sha256_file(registry_path)
    monkeypatch.setattr(
        corrector,
        "EXPECTED_OLD_REGISTRY_ARTIFACT_SHA256",
        registry["artifact_sha256"],
    )
    monkeypatch.setattr(
        corrector,
        "EXPECTED_OLD_REGISTRY_FILE_SHA256",
        old_file_sha256,
    )
    protocol, v4 = _identity()
    monkeypatch.setattr(
        corrector,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, {}, {}, v4),
    )
    monkeypatch.setattr(
        corrector,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: _decision(),
    )

    report = corrector.execute_operator_registry_correction(
        repository_root=repository,
        dataset_root=dataset,
        private_root=private,
    )

    assert report["already_corrected"] is False
    assert report["operator_ids"] == ["florianpfaff"]
    assert report["independent_verifier_available"] is False
    assert report["next_action"]["action_id"] == (
        "stop_independent_verifier_unavailable"
    )
    assert report["physical_evidence_increment"] == 0
    assert report["dataset_delta"] == {
        "added": ["preacquisition/operator_registry_correction_v1.json"],
        "removed": [],
        "modified": [
            "preacquisition/operator_registry.json",
            "preacquisition/operator_registry.template.json",
        ],
    }

    corrected = json.loads(registry_path.read_text(encoding="utf-8"))
    assert [value["operator_id"] for value in corrected["operators"]] == [
        "florianpfaff"
    ]
    assert corrected["sealed_by_operator_id"] == "florianpfaff"
    assert corrected["operators"][0]["roles"] == [
        "freezer",
        "gate_approver",
        "software_environment_approver",
    ]
    public_bytes = b"".join(
        path.read_bytes() for path in dataset.rglob("*") if path.is_file()
    )
    for unsupported in corrector._FORBIDDEN_PUBLIC_IDENTITIES:
        assert unsupported.encode("utf-8") not in public_bytes


def test_correction_is_idempotent_after_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    private = tmp_path / "private"
    repository.mkdir()
    dataset.mkdir()
    private.mkdir(mode=0o700)
    template, registry = _unsupported_registry()
    template_path = dataset / OPERATOR_REGISTRY_TEMPLATE_PATH
    registry_path = dataset / OPERATOR_REGISTRY_PATH
    template_path.parent.mkdir(parents=True)
    template_path.write_text(json.dumps(template), encoding="utf-8")
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    old_file_sha256, _ = corrector._sha256_file(registry_path)
    monkeypatch.setattr(
        corrector,
        "EXPECTED_OLD_REGISTRY_ARTIFACT_SHA256",
        registry["artifact_sha256"],
    )
    monkeypatch.setattr(
        corrector,
        "EXPECTED_OLD_REGISTRY_FILE_SHA256",
        old_file_sha256,
    )
    protocol, v4 = _identity()
    monkeypatch.setattr(
        corrector,
        "load_registered_preacquisition_chain",
        lambda repository_root: (protocol, {}, {}, v4),
    )
    monkeypatch.setattr(
        corrector,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: _decision(),
    )

    corrector.execute_operator_registry_correction(
        repository_root=repository,
        dataset_root=dataset,
        private_root=private,
    )
    repeated = corrector.execute_operator_registry_correction(
        repository_root=repository,
        dataset_root=dataset,
        private_root=private,
    )

    assert repeated["already_corrected"] is True
    assert repeated["dataset_modified"] is False
    assert repeated["dataset_delta"] == {
        "added": [],
        "removed": [],
        "modified": [],
    }


def test_correction_refuses_after_governed_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "object_registration.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="correction is too late"):
        corrector._require_no_governed_evidence(dataset)
