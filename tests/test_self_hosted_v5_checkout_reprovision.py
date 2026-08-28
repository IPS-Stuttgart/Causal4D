from __future__ import annotations

import importlib.util
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "reprovision_self_hosted_v5_checkout.py"
WORKFLOW = ROOT / ".github" / "workflows" / "reprovision-v5-checkout-self-hosted.yml"
DOCUMENTATION = ROOT / "docs" / "self_hosted_v5_checkout_reprovision.md"


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "reprovision_self_hosted_v5_checkout",
        SCRIPT,
    )
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _run(*arguments: str, cwd: Path) -> str:
    completed = subprocess.run(
        list(arguments),
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _repository(path: Path, files: Mapping[str, str]) -> str:
    path.mkdir()
    _run("git", "init", "--quiet", cwd=path)
    _run("git", "config", "user.name", "Causal4D test", cwd=path)
    _run("git", "config", "user.email", "test@example.invalid", cwd=path)
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    _run("git", "add", ".", cwd=path)
    _run("git", "commit", "--quiet", "-m", "fixture", cwd=path)
    return _run("git", "rev-parse", "HEAD", cwd=path)


def _decision(*, action_id: str = "complete_object_registration"):
    def build(_repository: Path, _dataset: Path) -> dict[str, Any]:
        return {
            "protocol_id": "fixture-protocol",
            "valid": True,
            "ready": False,
            "verify_file_hashes": True,
            "evidence_sha256": "a" * 64,
            "status_sha256": "b" * 64,
            "action": {
                "action_id": action_id,
                "category": "manual_prerequisite",
                "operator_role": "self_attesting_operator",
                "automatable": False,
                "physical_acquisition_required": False,
                "target_outcomes_permitted": False,
                "changes_registered_method": False,
            },
        }

    return build


def _source_files() -> dict[str, str]:
    return {
        "configs/causal4d/sloth_preacquisition_v5.json": "{}\n",
        "configs/causal4d/sloth_multi_action_v1.json": "{}\n",
        "scripts/ci/probe_self_hosted_acquisition.py": "print('probe')\n",
        "README.md": "reviewed source\n",
    }


def test_reprovision_replaces_only_checkout_and_retains_backup(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    target = tmp_path / "causal4d-frozen"
    dataset = tmp_path / "dataset"
    source_commit = _repository(source, _source_files())
    old_commit = _repository(target, {"README.md": "stale checkout\n"})
    dataset.mkdir()
    protocol = dataset / "protocol.json"
    protocol.write_text(json.dumps({"protocol": "fixture"}) + "\n", encoding="utf-8")
    before = protocol.read_bytes()

    report = module.reprovision_checkout(
        source,
        target,
        dataset,
        expected_source_commit=source_commit,
        decision_builder=_decision(),
    )

    assert report["previous_target_commit"] == old_commit
    assert report["deployed_target_commit"] == source_commit
    assert report["dataset_modified"] is False
    assert report["target_outcomes_used"] is False
    assert report["physical_evidence_increment"] == 0
    assert protocol.read_bytes() == before
    assert (target / "configs/causal4d/sloth_preacquisition_v5.json").is_file()
    backup = Path(report["retained_backup_repository"])
    assert backup.is_dir()
    assert _run("git", "rev-parse", "HEAD", cwd=backup) == old_commit
    assert _run("git", "status", "--porcelain=v1", cwd=target) == ""


def test_reprovision_rejects_dirty_target_without_mutation(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    target = tmp_path / "causal4d-frozen"
    dataset = tmp_path / "dataset"
    source_commit = _repository(source, _source_files())
    old_commit = _repository(target, {"README.md": "stale checkout\n"})
    (target / "local-untracked.txt").write_text("do not discard\n", encoding="utf-8")
    dataset.mkdir()
    (dataset / "protocol.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="registered target checkout is not clean"):
        module.reprovision_checkout(
            source,
            target,
            dataset,
            expected_source_commit=source_commit,
            decision_builder=_decision(),
        )

    assert _run("git", "rev-parse", "HEAD", cwd=target) == old_commit
    assert (target / "local-untracked.txt").is_file()
    assert not list(tmp_path.glob("causal4d-frozen.before-*"))


def test_reprovision_rejects_nonregistered_next_action(tmp_path: Path) -> None:
    module = _load_script()
    source = tmp_path / "source"
    target = tmp_path / "causal4d-frozen"
    dataset = tmp_path / "dataset"
    source_commit = _repository(source, _source_files())
    old_commit = _repository(target, {"README.md": "stale checkout\n"})
    dataset.mkdir()
    (dataset / "protocol.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="allowed only at complete_object_registration",
    ):
        module.reprovision_checkout(
            source,
            target,
            dataset,
            expected_source_commit=source_commit,
            decision_builder=_decision(action_id="run_slip_pilot"),
        )

    assert _run("git", "rev-parse", "HEAD", cwd=target) == old_commit
    assert not list(tmp_path.glob("causal4d-frozen.before-*"))


def test_workflow_is_narrowly_issue_authorized_and_nonphysical() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "[self-hosted] reprovision Causal4D v5 acquisition checkout" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert "permissions:\n  contents: read" in text
    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, "
        "data-causal4d-physical-v1]"
        in text
    )
    assert "scripts/ci/reprovision_self_hosted_v5_checkout.py" in text
    assert "/mnt/lexar4tb/causal4d-physical/causal4d-frozen" in text
    assert "/mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1-v5" in text
    assert "physical_evidence_increment" in text
    assert "target_outcomes_used" in text
    assert "secrets." not in text


def test_documentation_preserves_the_pre_freeze_boundary() -> None:
    text = DOCUMENTATION.read_text(encoding="utf-8")

    assert "complete_object_registration" in text
    assert "method_freeze.json" in text
    assert "byte-preserved" in text
    assert "Physical evidence increment: `0`" in text
    assert "retained rollback checkout" in text
