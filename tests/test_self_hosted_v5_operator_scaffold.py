from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from causal4d.operator_registry import (
    OPERATOR_REGISTRY_ARTIFACT_KIND,
    OPERATOR_REGISTRY_PATH,
    operator_registry_sha256,
    operator_registry_template,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "ci" / "bootstrap_self_hosted_v5_operator_scaffold.py"


def _load_bootstrap() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "causal4d_self_hosted_v5_operator_scaffold",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


bootstrap = _load_bootstrap()


def _source_dataset(tmp_path: Path) -> Path:
    source = tmp_path / "source-v4"
    registry_path = source / OPERATOR_REGISTRY_PATH
    registry_path.parent.mkdir(parents=True)
    protocol, v4 = bootstrap._load_v4(ROOT)
    registry = operator_registry_template(protocol, v4)
    registry.update(
        {
            "artifact_kind": OPERATOR_REGISTRY_ARTIFACT_KIND,
            "status": "sealed",
            "sealed_at_utc": "2026-08-11T10:00:00+00:00",
            "sealed_by_operator_id": bootstrap.OPERATOR_ID,
            "operators": [
                {
                    "operator_id": bootstrap.OPERATOR_ID,
                    "person_identity_sha256": "1" * 64,
                    "active": True,
                    "roles": list(bootstrap.OPERATOR_ROLES),
                }
            ],
        }
    )
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return source


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_bootstrap_creates_fresh_v5_tree_and_advances_to_registration(
    tmp_path: Path,
) -> None:
    source = _source_dataset(tmp_path)
    target = tmp_path / "fresh-v5"

    report = bootstrap.bootstrap_single_operator_v5(
        repository_root=ROOT,
        source_dataset_root=source,
        target_dataset_root=target,
        sealed_at_utc="2026-08-26T01:00:00+00:00",
    )

    assert report["created"] is True
    assert report["dataset_modified"] is True
    assert report["target_preacquisition_plan_id"] == (
        "causal4d-sloth-preacquisition-v5-single-operator"
    )
    assert report["operator_ids"] == [bootstrap.OPERATOR_ID]
    assert report["operator_roles"] == list(bootstrap.OPERATOR_ROLES)
    assert report["independent_verifier_available"] is False
    assert report["next_action"] == {
        "action_id": "complete_object_registration",
        "operator_role": "self_attesting_operator",
        "automatable": False,
        "physical_acquisition_required": False,
        "target_outcomes_permitted": False,
    }
    assert report["target_outcomes_used"] is False
    assert report["device_nodes_opened"] is False
    assert report["physical_command_sent"] is False
    assert report["physical_evidence_increment"] == 0

    registry = json.loads((target / OPERATOR_REGISTRY_PATH).read_text(encoding="utf-8"))
    assert registry["operators"][0]["operator_id"] == bootstrap.OPERATOR_ID
    assert registry["operators"][0]["person_identity_sha256"] == "1" * 64
    assert registry["operators"][0]["roles"] == list(bootstrap.OPERATOR_ROLES)
    assert "independent_verifier" not in registry["operators"][0]["roles"]

    receipt = json.loads((target / bootstrap.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["independent_preacquisition_attestation_claimed"] is False
    assert receipt["physical_evidence_increment"] == 0


def test_bootstrap_is_idempotent_and_does_not_reseal(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path)
    target = tmp_path / "fresh-v5"
    first = bootstrap.bootstrap_single_operator_v5(
        repository_root=ROOT,
        source_dataset_root=source,
        target_dataset_root=target,
        sealed_at_utc="2026-08-26T01:00:00+00:00",
    )
    snapshot = _tree_snapshot(target)

    repeated = bootstrap.bootstrap_single_operator_v5(
        repository_root=ROOT,
        source_dataset_root=source,
        target_dataset_root=target,
        sealed_at_utc="2026-08-26T02:00:00+00:00",
    )

    assert first["created"] is True
    assert repeated["created"] is False
    assert repeated["dataset_modified"] is False
    assert _tree_snapshot(target) == snapshot
    assert (
        repeated["target_registry_artifact_sha256"]
        == (first["target_registry_artifact_sha256"])
    )
    assert (
        repeated["bootstrap_receipt_artifact_sha256"]
        == (first["bootstrap_receipt_artifact_sha256"])
    )


def test_bootstrap_refuses_historical_tree_after_governed_work(
    tmp_path: Path,
) -> None:
    source = _source_dataset(tmp_path)
    (source / "object_registration.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="contains governed evidence"):
        bootstrap.bootstrap_single_operator_v5(
            repository_root=ROOT,
            source_dataset_root=source,
            target_dataset_root=tmp_path / "fresh-v5",
        )


def test_bootstrap_refuses_false_independent_role(tmp_path: Path) -> None:
    source = _source_dataset(tmp_path)
    registry_path = source / OPERATOR_REGISTRY_PATH
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["operators"][0]["roles"].append("independent_verifier")
    registry["operators"][0]["roles"].sort()
    registry["artifact_sha256"] = operator_registry_sha256(registry)
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="roles differ from the v5 one-person lock"):
        bootstrap.bootstrap_single_operator_v5(
            repository_root=ROOT,
            source_dataset_root=source,
            target_dataset_root=tmp_path / "fresh-v5",
        )
