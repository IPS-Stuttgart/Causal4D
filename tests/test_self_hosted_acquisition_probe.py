from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ci"
    / "probe_self_hosted_acquisition.py"
)


def _load_probe() -> ModuleType:
    specification = importlib.util.spec_from_file_location(
        "causal4d_self_hosted_acquisition_probe",
        SCRIPT_PATH,
    )
    assert specification is not None
    assert specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


probe = _load_probe()


def _complete_pair(root: Path, name: str) -> tuple[Path, Path]:
    repository = root / name / "repository"
    dataset = root / name / "dataset"
    repository.mkdir(parents=True)
    dataset.mkdir(parents=True)
    return repository, dataset


def test_root_selection_uses_the_only_complete_pair(tmp_path: Path) -> None:
    persistent_repository, persistent_dataset = _complete_pair(
        tmp_path,
        "persistent",
    )
    selection = probe._select_registered_roots(
        (
            (
                "canonical",
                tmp_path / "canonical" / "repository",
                tmp_path / "canonical" / "dataset",
            ),
            (
                "persistent",
                persistent_repository,
                persistent_dataset,
            ),
        )
    )

    assert selection["selection_status"] == "selected"
    assert selection["selected_candidate_id"] == "persistent"
    assert selection["selected_repository_root"] == str(
        persistent_repository.absolute()
    )
    assert selection["selected_dataset_root"] == str(persistent_dataset.absolute())
    assert [candidate["pair_state"] for candidate in selection["candidates"]] == [
        "absent",
        "complete",
    ]


def test_root_selection_never_mixes_partial_pairs(tmp_path: Path) -> None:
    first_repository = tmp_path / "first" / "repository"
    first_repository.mkdir(parents=True)
    second_dataset = tmp_path / "second" / "dataset"
    second_dataset.mkdir(parents=True)

    selection = probe._select_registered_roots(
        (
            (
                "first",
                first_repository,
                tmp_path / "first" / "dataset",
            ),
            (
                "second",
                tmp_path / "second" / "repository",
                second_dataset,
            ),
        )
    )

    assert selection["selection_status"] == "unavailable"
    assert selection["complete_candidate_count"] == 0
    assert selection["selected_candidate_id"] is None
    assert [candidate["pair_state"] for candidate in selection["candidates"]] == [
        "partial",
        "partial",
    ]


def test_root_selection_fails_closed_when_multiple_pairs_are_complete(
    tmp_path: Path,
) -> None:
    first = _complete_pair(tmp_path, "first")
    second = _complete_pair(tmp_path, "second")

    selection = probe._select_registered_roots(
        (
            ("first", *first),
            ("second", *second),
        )
    )

    assert selection["selection_status"] == "ambiguous"
    assert selection["complete_candidate_count"] == 2
    assert selection["selected_candidate_id"] is None
    assert selection["selected_repository_root"] is None
    assert selection["selected_dataset_root"] is None


def test_root_selection_rejects_symlink_components(tmp_path: Path) -> None:
    real_repository = tmp_path / "real-repository"
    real_repository.mkdir()
    linked_repository = tmp_path / "linked-repository"
    try:
        linked_repository.symlink_to(real_repository, target_is_directory=True)
    except OSError as error:  # pragma: no cover - symlinks unavailable
        pytest.skip(f"symlinks unavailable: {error}")
    dataset = tmp_path / "dataset"
    dataset.mkdir()

    selection = probe._select_registered_roots(
        (("linked", linked_repository, dataset),)
    )

    candidate = selection["candidates"][0]
    assert selection["selection_status"] == "unavailable"
    assert candidate["pair_state"] == "invalid"
    assert candidate["repository"]["contains_symlink_component"] is True
    assert candidate["repository"]["ordinary_directory"] is False


def test_root_selection_rejects_duplicate_candidate_ids(tmp_path: Path) -> None:
    first = _complete_pair(tmp_path, "first")
    second = _complete_pair(tmp_path, "second")

    with pytest.raises(ValueError, match="must be unique"):
        probe._select_registered_roots(
            (
                ("duplicate", *first),
                ("duplicate", *second),
            )
        )


def test_build_report_derives_action_from_selected_pair(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, dataset = _complete_pair(tmp_path, "persistent")
    device_root = tmp_path / "dev"
    device_root.mkdir()
    called: dict[str, Path] = {}

    def next_action(
        repository_root: Path,
        dataset_root: Path,
    ) -> tuple[dict[str, object], None]:
        called["repository"] = repository_root
        called["dataset"] = dataset_root
        return (
            {
                "action_id": "review-source-panel",
                "physical_acquisition_required": False,
                "automatable": False,
            },
            None,
        )

    monkeypatch.setattr(probe, "_next_action_summary", next_action)
    monkeypatch.setattr(
        probe,
        "_command_summary",
        lambda *args, **kwargs: {
            "available": False,
            "return_code": None,
            "line_count": 0,
            "output_sha256": None,
            "timed_out": False,
        },
    )

    report = probe.build_report(
        root_candidates=(
            (
                "canonical",
                tmp_path / "canonical" / "repository",
                tmp_path / "canonical" / "dataset",
            ),
            ("persistent", repository, dataset),
        ),
        device_root=device_root,
    )

    assert called == {
        "repository": repository.absolute(),
        "dataset": dataset.absolute(),
    }
    assert report["registered_roots"]["candidate_id"] == "persistent"
    assert report["next_action"]["action_id"] == "review-source-panel"
    assert report["conclusion"] == "next_action_requires_registered_human_role"
    assert report["dataset_modified"] is False
    assert report["physical_evidence_increment"] == 0


def test_build_report_does_not_derive_action_for_ambiguous_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _complete_pair(tmp_path, "first")
    second = _complete_pair(tmp_path, "second")
    device_root = tmp_path / "dev"
    device_root.mkdir()

    def unexpected(*args, **kwargs):
        raise AssertionError("next action must not be derived for ambiguous roots")

    monkeypatch.setattr(probe, "_next_action_summary", unexpected)
    monkeypatch.setattr(
        probe,
        "_command_summary",
        lambda *args, **kwargs: {
            "available": False,
            "return_code": None,
            "line_count": 0,
            "output_sha256": None,
            "timed_out": False,
        },
    )

    report = probe.build_report(
        root_candidates=(
            ("first", *first),
            ("second", *second),
        ),
        device_root=device_root,
    )

    assert report["registered_root_selection"]["selection_status"] == "ambiguous"
    assert report["next_action"] is None
    assert report["conclusion"] == "registered_root_selection_ambiguous"
    assert report["registered_roots"]["repository"]["path"] is None
