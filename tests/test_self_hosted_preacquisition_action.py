from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "execute_self_hosted_preacquisition_action.py"
)


def _load_executor() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "causal4d_self_hosted_preacquisition_action",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


executor = _load_executor()


def _identity() -> dict[str, str]:
    return {
        "protocol_id": "protocol-v1",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan-v1",
        "preacquisition_amendment_sha256": "b" * 64,
    }


def _decision(
    repository: Path,
    dataset: Path,
    *,
    action_id: str,
    automatable: bool,
    physical: bool = False,
    command: list[str] | None = None,
) -> dict[str, object]:
    category = (
        "scaffold" if action_id == "scaffold_operator_registry" else "manual_evidence"
    )
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DPreacquisitionNextAction",
        **_identity(),
        "valid": True,
        "ready": False,
        "target_outcomes_used": False,
        "evidence_sha256": "c" * 64,
        "status_sha256": "d" * 64,
        "action": {
            "action_id": action_id,
            "category": category,
            "title": action_id,
            "operator_role": "principal_investigator",
            "physical_acquisition_required": physical,
            "automatable": automatable,
            "changes_registered_method": False,
            "target_outcomes_permitted": False,
            "command_argv": command,
            "blocking_items": [],
        },
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


def _install_successful_stubs(
    monkeypatch: pytest.MonkeyPatch,
    repository: Path,
    dataset: Path,
    *,
    mutate_extra_path: bool = False,
) -> None:
    expected_command = executor._expected_command(
        repository.resolve(),
        dataset.resolve(),
    )
    decisions = iter(
        (
            _decision(
                repository.resolve(),
                dataset.resolve(),
                action_id="scaffold_operator_registry",
                automatable=True,
                command=expected_command,
            ),
            _decision(
                repository.resolve(),
                dataset.resolve(),
                action_id="seal_operator_registry",
                automatable=False,
                command=None,
            ),
        )
    )
    monkeypatch.setattr(
        executor,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: next(decisions),
    )

    def scaffold(repository_root: Path, dataset_root: Path) -> dict[str, object]:
        target = dataset_root / executor.OPERATOR_REGISTRY_TEMPLATE_PATH
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(_template(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        if mutate_extra_path:
            (dataset_root / "unexpected.txt").write_text("changed\n", encoding="utf-8")
        return {
            "passed": True,
            "path": str(target.resolve()),
            "created": True,
            "existing": False,
        }

    monkeypatch.setattr(executor, "scaffold_operator_registry", scaffold)


def test_execute_allowlisted_action_adds_only_the_empty_template(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    (dataset / "existing.json").write_text("{}\n", encoding="utf-8")
    _install_successful_stubs(monkeypatch, repository, dataset)

    report = executor.execute_allowlisted_action(
        repository_root=repository,
        dataset_root=dataset,
        expected_action_id="scaffold_operator_registry",
    )

    assert report["executed_action"]["action_id"] == "scaffold_operator_registry"
    assert report["dataset_delta"] == {
        "added": ["preacquisition/operator_registry.template.json"],
        "removed": [],
        "modified": [],
    }
    assert report["created_template"]["operator_count"] == 0
    assert report["next_action"]["action_id"] == "seal_operator_registry"
    assert report["next_action"]["automatable"] is False
    assert report["target_outcomes_used"] is False
    assert report["physical_command_sent"] is False
    assert report["physical_evidence_increment"] == 0


def test_execute_allowlisted_action_rejects_physical_registered_action(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    command = executor._expected_command(repository.resolve(), dataset.resolve())
    monkeypatch.setattr(
        executor,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: _decision(
            repository.resolve(),
            dataset.resolve(),
            action_id="scaffold_operator_registry",
            automatable=True,
            physical=True,
            command=command,
        ),
    )

    with pytest.raises(ValueError, match="requires physical acquisition"):
        executor.execute_allowlisted_action(
            repository_root=repository,
            dataset_root=dataset,
            expected_action_id="scaffold_operator_registry",
        )

    assert not (dataset / executor.OPERATOR_REGISTRY_TEMPLATE_PATH).exists()


def test_execute_allowlisted_action_rejects_command_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    monkeypatch.setattr(
        executor,
        "build_preacquisition_operator_next_action",
        lambda *args, **kwargs: _decision(
            repository.resolve(),
            dataset.resolve(),
            action_id="scaffold_operator_registry",
            automatable=True,
            command=["causal4d", "unexpected"],
        ),
    )

    with pytest.raises(ValueError, match="exact allowlisted command"):
        executor.execute_allowlisted_action(
            repository_root=repository,
            dataset_root=dataset,
            expected_action_id="scaffold_operator_registry",
        )


def test_execute_allowlisted_action_rejects_preexisting_template(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    target = dataset / executor.OPERATOR_REGISTRY_TEMPLATE_PATH
    target.parent.mkdir(parents=True)
    target.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="already exists"):
        executor.execute_allowlisted_action(
            repository_root=repository,
            dataset_root=dataset,
            expected_action_id="scaffold_operator_registry",
        )


def test_execute_allowlisted_action_rejects_unexpected_dataset_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path / "repository"
    dataset = tmp_path / "dataset"
    repository.mkdir()
    dataset.mkdir()
    _install_successful_stubs(
        monkeypatch,
        repository,
        dataset,
        mutate_extra_path=True,
    )

    with pytest.raises(ValueError, match="outside the registered template path"):
        executor.execute_allowlisted_action(
            repository_root=repository,
            dataset_root=dataset,
            expected_action_id="scaffold_operator_registry",
        )


def test_execute_allowlisted_action_rejects_symlink_dataset_root(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    real_dataset = tmp_path / "real-dataset"
    real_dataset.mkdir()
    dataset = tmp_path / "dataset"
    try:
        dataset.symlink_to(real_dataset, target_is_directory=True)
    except OSError as error:  # pragma: no cover - symlinks unavailable
        pytest.skip(f"symlinks unavailable: {error}")

    with pytest.raises(ValueError, match="symlink component"):
        executor.execute_allowlisted_action(
            repository_root=repository,
            dataset_root=dataset,
            expected_action_id="scaffold_operator_registry",
        )
