from __future__ import annotations

import importlib.util
import json
import zipfile
from pathlib import Path
from types import ModuleType


def _load_script() -> ModuleType:
    root = Path(__file__).resolve().parents[1]
    path = root / "scripts" / "remote" / "inventory_complete_deformable_datasets.py"
    spec = importlib.util.spec_from_file_location("dataset_admission", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)


def test_build_report_admits_complete_fixture_without_loading_payloads(
    tmp_path: Path,
) -> None:
    module = _load_script()
    dot = tmp_path / "dot"
    tracking = tmp_path / "tracking"
    deform = tmp_path / "deform"
    dot.mkdir()
    tracking.mkdir()
    deform.mkdir()

    _write_zip(dot / "part-01.zip", {"recording/a.bin": b"payload"})
    _write_zip(dot / "part-02.zip", {"recording/b.bin": b"payload"})
    (dot / "README.txt").write_text("DOT fixture\n", encoding="utf-8")

    for index in range(3):
        recording = tracking / f"recording_{index:03d}"
        recording.mkdir()
        (recording / "trajectory.bin").write_bytes(b"not opened")
    (tracking / "metadata.json").write_text(
        json.dumps({"recordings": 3}), encoding="utf-8"
    )

    for name in ("DLO4", "DLO5"):
        unit = deform / name
        unit.mkdir()
        (unit / "a.bin").write_bytes(b"a")
        (unit / "b.bin").write_bytes(b"b")

    report = module.build_report(
        dot_root=dot,
        tracking_root=tracking,
        deform_root=deform,
        expected_dot_archives=2,
        expected_tracking_recordings=3,
        expected_deform_units=("DLO4", "DLO5"),
        expected_deform_files_per_unit=2,
    )

    assert report["decisions"]["all_expected_source_layouts_admitted"] is True
    assert report["future_or_target_outcomes_read"] is False
    assert report["numeric_payloads_read"] is False
    assert report["archives_extracted"] is False
    assert report["datasets"]["dot"]["archives"][0][
        "central_directory_ok"
    ] is True
    assert report["datasets"]["tracking_cloth"][
        "recording_count_estimate"
    ] == 3
    assert set(report["datasets"]["deform"]["units"]) == {"DLO4", "DLO5"}


def test_missing_layout_is_retained_as_negative_admission(tmp_path: Path) -> None:
    module = _load_script()
    dot = tmp_path / "dot"
    tracking = tmp_path / "tracking"
    deform = tmp_path / "deform"
    dot.mkdir()
    tracking.mkdir()
    deform.mkdir()

    report = module.build_report(
        dot_root=dot,
        tracking_root=tracking,
        deform_root=deform,
        expected_dot_archives=21,
        expected_tracking_recordings=120,
        expected_deform_units=("DLO4", "DLO5"),
        expected_deform_files_per_unit=70,
    )

    assert report["decisions"]["all_expected_source_layouts_admitted"] is False
    assert report["decisions"]["task_conditioned_intervention_claim_authorized"] is False
