from __future__ import annotations

import importlib.util
import json
import os
import stat
from pathlib import Path
from types import ModuleType

import pytest

from causal4d.operator_registry import OPERATOR_REGISTRY_PATH


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


def _private_root(tmp_path: Path) -> Path:
    return tmp_path / "private" / "operator-registry-v5"


def _tree_snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(tmp_path: Path, *, sealed_at_utc: str = "2026-08-26T01:00:00+00:00"):
    return bootstrap.bootstrap_single_operator_v5(
        repository_root=ROOT,
        private_identity_root=_private_root(tmp_path),
        target_dataset_root=tmp_path / "fresh-v5",
        sealed_at_utc=sealed_at_utc,
    )


def test_bootstrap_creates_fresh_owner_identity_and_advances_to_registration(
    tmp_path: Path,
) -> None:
    target = tmp_path / "fresh-v5"
    private = _private_root(tmp_path)

    report = _run(tmp_path)

    assert report["schema_version"] == 2
    assert report["created"] is True
    assert report["dataset_modified"] is True
    assert report["private_identity_material_created"] is True
    assert report["identity_initialization_mode"] == "fresh_owner_hmac_v1"
    assert report["historical_registry_available"] is False
    assert report["historical_registry_reused"] is False
    assert report["identity_digest_continuity_claimed"] is False
    implementation = ROOT / bootstrap.IMPLEMENTATION_PATH
    assert (
        report["bootstrap_implementation_sha256"]
        == bootstrap._sha256_file(implementation)[0]
    )
    assert report["bootstrap_implementation_bytes"] == implementation.stat().st_size
    assert report["target_preacquisition_plan_id"] == (
        "causal4d-sloth-preacquisition-v5-single-operator"
    )
    assert report["operator_ids"] == [bootstrap.OPERATOR_ID]
    assert report["operator_roles"] == list(bootstrap.OPERATOR_ROLES)
    assert report["independent_verifier_available"] is False
    assert report["independent_preacquisition_attestation_claimed"] is False
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
    operator = registry["operators"][0]
    assert operator["operator_id"] == bootstrap.OPERATOR_ID
    assert len(operator["person_identity_sha256"]) == 64
    assert operator["roles"] == list(bootstrap.OPERATOR_ROLES)
    assert "independent_verifier" not in operator["roles"]

    receipt = json.loads((target / bootstrap.RECEIPT_PATH).read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["identity_initialization_mode"] == "fresh_owner_hmac_v1"
    assert receipt["historical_registry_available"] is False
    assert receipt["historical_registry_reused"] is False
    assert receipt["identity_digest_continuity_claimed"] is False
    assert (
        receipt["bootstrap_implementation_sha256"]
        == report["bootstrap_implementation_sha256"]
    )
    assert receipt["bootstrap_implementation_bytes"] == implementation.stat().st_size
    assert receipt["independent_preacquisition_attestation_claimed"] is False
    assert receipt["physical_evidence_increment"] == 0

    assert stat.S_IMODE(private.stat().st_mode) == 0o700
    assert stat.S_IMODE((private / bootstrap.KEY_FILENAME).stat().st_mode) == 0o600
    assert (
        stat.S_IMODE((private / bootstrap.PRIVATE_ROSTER_FILENAME).stat().st_mode)
        == 0o600
    )
    assert len((private / bootstrap.KEY_FILENAME).read_bytes()) == 32

    serialized_report = json.dumps(report, sort_keys=True)
    assert bootstrap.CANONICAL_PRINCIPAL not in serialized_report
    assert "person_identity_sha256" not in serialized_report
    assert str(private) not in serialized_report


def test_bootstrap_is_idempotent_and_does_not_reseal(tmp_path: Path) -> None:
    private = _private_root(tmp_path)
    target = tmp_path / "fresh-v5"
    first = _run(tmp_path)
    private_snapshot = _tree_snapshot(private)
    target_snapshot = _tree_snapshot(target)

    repeated = _run(tmp_path, sealed_at_utc="2026-08-26T02:00:00+00:00")

    assert first["created"] is True
    assert repeated["created"] is False
    assert repeated["dataset_modified"] is False
    assert repeated["private_identity_material_created"] is False
    assert _tree_snapshot(private) == private_snapshot
    assert _tree_snapshot(target) == target_snapshot
    assert (
        repeated["target_registry_artifact_sha256"]
        == first["target_registry_artifact_sha256"]
    )
    assert (
        repeated["bootstrap_receipt_artifact_sha256"]
        == first["bootstrap_receipt_artifact_sha256"]
    )


def test_bootstrap_refuses_partial_private_identity_root(tmp_path: Path) -> None:
    private = _private_root(tmp_path)
    private.mkdir(parents=True, mode=0o700)
    key = private / bootstrap.KEY_FILENAME
    key.write_bytes(b"1" * 32)
    os.chmod(key, 0o600)

    with pytest.raises(ValueError, match="members differ"):
        _run(tmp_path)


def test_bootstrap_refuses_invalid_private_key_length(tmp_path: Path) -> None:
    _run(tmp_path)
    target = tmp_path / "fresh-v5"
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    target.rmdir()
    key = _private_root(tmp_path) / bootstrap.KEY_FILENAME
    key.write_bytes(b"too-short")
    os.chmod(key, 0o600)

    with pytest.raises(ValueError, match="key length is invalid"):
        _run(tmp_path)


def test_bootstrap_refuses_changed_private_roster(tmp_path: Path) -> None:
    _run(tmp_path)
    target = tmp_path / "fresh-v5"
    for path in sorted(target.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    target.rmdir()
    roster_path = _private_root(tmp_path) / bootstrap.PRIVATE_ROSTER_FILENAME
    roster = json.loads(roster_path.read_text(encoding="utf-8"))
    roster["assignments"][0]["roles"].append("independent_verifier")
    roster_path.write_text(json.dumps(roster), encoding="utf-8")
    os.chmod(roster_path, 0o600)

    with pytest.raises(ValueError, match="differs from the fresh-owner lock"):
        _run(tmp_path)


def test_bootstrap_refuses_public_private_material(tmp_path: Path) -> None:
    _run(tmp_path)
    key = _private_root(tmp_path) / bootstrap.KEY_FILENAME
    owner_mode = stat.S_IMODE(key.stat().st_mode)
    key.chmod(owner_mode | stat.S_IRGRP)

    with pytest.raises(ValueError, match="mode is 640, expected 600"):
        _run(tmp_path)


def test_bootstrap_refuses_private_identity_symlink(tmp_path: Path) -> None:
    private = _private_root(tmp_path)
    private.parent.mkdir(mode=0o700)
    key_target = tmp_path / "key-target"
    key_target.write_bytes(b"1" * 32)
    private.symlink_to(key_target)

    with pytest.raises(ValueError, match="symlink"):
        _run(tmp_path)


def test_bootstrap_refuses_registry_private_identity_mismatch(tmp_path: Path) -> None:
    _run(tmp_path)
    key = _private_root(tmp_path) / bootstrap.KEY_FILENAME
    key.write_bytes(b"2" * 32)
    os.chmod(key, 0o600)

    with pytest.raises(
        ValueError, match="differs from the owner-only private identity"
    ):
        _run(tmp_path)


def test_bootstrap_refuses_rerun_after_governed_work(tmp_path: Path) -> None:
    _run(tmp_path)
    (tmp_path / "fresh-v5" / "object_registration.json").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="bootstrap is too late"):
        _run(tmp_path)
