from __future__ import annotations

import hashlib
import importlib.util
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "stage_self_hosted_object_registration_inputs.py"
WORKFLOW = (
    ROOT / ".github" / "workflows" / "stage-object-registration-inputs-self-hosted.yml"
)
DOCUMENTATION = ROOT / "docs" / "self_hosted_object_registration_input_stage.md"


def _load_script():
    specification = importlib.util.spec_from_file_location(
        "stage_self_hosted_object_registration_inputs",
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


def _repository(path: Path, files: Mapping[str, bytes]) -> str:
    path.mkdir()
    _run("git", "init", "--quiet", cwd=path)
    _run("git", "config", "user.name", "Causal4D test", cwd=path)
    _run("git", "config", "user.email", "test@example.invalid", cwd=path)
    for relative, content in files.items():
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    _run("git", "add", ".", cwd=path)
    _run("git", "commit", "--quiet", "-m", "fixture", cwd=path)
    return _run("git", "rev-parse", "HEAD", cwd=path)


def _decision(*, action_id: str = "complete_object_registration"):
    def build(_repository: Path, _dataset: Path) -> dict[str, Any]:
        return {
            "valid": True,
            "ready": False,
            "verify_file_hashes": True,
            "evidence_sha256": "a" * 64,
            "status_sha256": "b" * 64,
            "action": {
                "action_id": action_id,
                "operator_role": "self_attesting_operator",
                "automatable": False,
                "physical_acquisition_required": False,
                "target_outcomes_permitted": False,
                "changes_registered_method": False,
            },
        }

    return build


def _fixture(tmp_path: Path):
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "protocol.json").write_text("{}\n", encoding="utf-8")
    contents = {
        "left_forepaw": b"left-node-set\n",
        "right_forepaw": b"right-node-set\n",
        "upper_torso": b"torso-node-set\n",
    }
    candidates = {
        "left_forepaw": "L1",
        "right_forepaw": "P1",
        "upper_torso": "F2",
    }
    requirements: dict[str, dict[str, Any]] = {}
    files: dict[str, bytes] = {
        "configs/causal4d/sloth_multi_action_v1.json": b'{"protocol_id":"p"}\n'
    }
    for index, (region_id, content) in enumerate(contents.items(), start=1):
        relative = f"contact_node_sets/{region_id}.json"
        files[f"evidence/object-registration-anatomy-v8/{relative}"] = content
        requirements[region_id] = {
            "path": relative,
            "node_count": index,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "selected_candidate_id": candidates[region_id],
        }
    commit = _repository(repository, files)

    def packet(_protocol: Mapping[str, Any], _root: Path) -> dict[str, Any]:
        return {
            "artifact_kind": "Causal4DObjectRegistrationSealPacket",
            "phystwin_model_id": "phystwin-single_lift_sloth-best_199",
            "phystwin_model_sha256": (
                "e7b853f8369ccb5b0d56dee0991fd6e95482a2baa37a913fc7f4b22db93044ad"
            ),
            "missing_operator_inputs": ["physical_instance_serial"],
            "ready_to_seal_object_registration": False,
            "packet_id": "c" * 64,
            "anatomical_approval": {"artifact_id": "d" * 64},
            "contact_regions": requirements,
        }

    return repository, dataset, commit, requirements, packet


def test_stages_only_approved_node_sets_and_is_idempotent(tmp_path: Path) -> None:
    module = _load_script()
    repository, dataset, commit, requirements, packet = _fixture(tmp_path)

    first = module.stage_object_registration_inputs(
        repository,
        dataset,
        expected_commit=commit,
        decision_builder=_decision(),
        packet_builder=packet,
        requirements=requirements,
    )
    second = module.stage_object_registration_inputs(
        repository,
        dataset,
        expected_commit=commit,
        decision_builder=_decision(),
        packet_builder=packet,
        requirements=requirements,
    )

    assert first["added_paths"] == sorted(
        requirement["path"] for requirement in requirements.values()
    )
    assert first["ready_except_physical_serial"] is True
    assert first["object_registration_json_created"] is False
    assert first["physical_evidence_increment"] == 0
    assert second["added_paths"] == []
    assert second["dataset_modified"] is False
    for requirement in requirements.values():
        path = dataset / requirement["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == requirement["sha256"]


def test_rejects_existing_nonapproved_node_set_without_overwrite(
    tmp_path: Path,
) -> None:
    module = _load_script()
    repository, dataset, commit, requirements, packet = _fixture(tmp_path)
    destination = dataset / requirements["left_forepaw"]["path"]
    destination.parent.mkdir(parents=True)
    destination.write_bytes(b"wrong\n")

    with pytest.raises(ValueError, match="left_forepaw.json differs"):
        module.stage_object_registration_inputs(
            repository,
            dataset,
            expected_commit=commit,
            decision_builder=_decision(),
            packet_builder=packet,
            requirements=requirements,
        )

    assert destination.read_bytes() == b"wrong\n"
    assert not (dataset / requirements["right_forepaw"]["path"]).exists()


def test_rejects_staging_after_registered_action_changes(tmp_path: Path) -> None:
    module = _load_script()
    repository, dataset, commit, requirements, packet = _fixture(tmp_path)

    with pytest.raises(
        ValueError,
        match="allowed only at complete_object_registration",
    ):
        module.stage_object_registration_inputs(
            repository,
            dataset,
            expected_commit=commit,
            decision_builder=_decision(action_id="complete_contact_registration"),
            packet_builder=packet,
            requirements=requirements,
        )

    assert not (dataset / "contact_node_sets").exists()


def test_workflow_is_narrowly_authorized_and_keeps_the_serial_manual() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "[self-hosted] stage Causal4D object-registration inputs" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert "permissions:\n  contents: read" in text
    assert (
        "runs-on: [self-hosted, Linux, X64, nvidia-smi, "
        "data-causal4d-physical-v1]" in text
    )
    assert "stage_self_hosted_object_registration_inputs.py" in text
    assert "ready_except_physical_serial" in text
    assert "physical_instance_serial" in text
    assert "object_registration_json_created" in text
    assert "secrets." not in text


def test_documentation_preserves_object_registration_claim_boundary() -> None:
    text = DOCUMENTATION.read_text(encoding="utf-8")

    assert "complete_object_registration" in text
    assert "stable inventory serial" in text
    assert "logical object ID" in text
    assert "Physical evidence increment: `0`" in text
    assert "object_registration.json` created: `false" in text
