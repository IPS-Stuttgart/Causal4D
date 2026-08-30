from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/causal4d_public/tracking_cloth_query_observation.py"
MODULE_NAME = "causal4d_tracking_cloth_query_observation_test"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_csv(
    path: Path,
    *,
    markers: int = 12,
    frames: int = 240,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Format Version", "1.0"])
        writer.writerow(["Capture Frame Rate", "120"])
        writer.writerow(["marker ids"])
        writer.writerow(["units", "millimeters"])
        writer.writerow(
            ["Frame", "Time"]
            + [f"coordinate_{index}" for index in range(3 * markers)]
        )
        for frame in range(frames):
            time = frame / 120.0
            values: list[float] = []
            for marker in range(markers):
                x = 100.0 * (marker % 4) + 5.0 * np.sin(time * 3 + marker)
                y = 80.0 * (marker // 4) + 2.0 * np.cos(time * 2 + marker)
                z = (
                    500.0
                    - 30.0 * (marker // 4)
                    + 10.0 * np.sin(time * 5 + marker / 3)
                )
                values.extend([x, y, z])
            writer.writerow([frame, time, *values])


def test_csv_read_and_window_extraction(tmp_path: Path) -> None:
    path = (
        tmp_path
        / "Free-hanging"
        / "cotton_A3_shake_fast_hanger.csv"
    )
    _write_csv(path)
    key = MODULE.classify_recording(path, tmp_path)
    assert key is not None
    assert key.material == "cotton"
    assert key.scenario == "shake"
    recording = MODULE.read_recording(key)
    assert recording.points_m.shape == (240, 12, 3)
    assert recording.unit_scale_to_m == pytest.approx(1e-3)
    batch = MODULE.extract_windows(
        recording,
        horizon_seconds=0.1,
        lag_seconds=0.05,
        stride_seconds=0.1,
    )
    assert batch.y.shape[1] == 4
    assert set(batch.x_by_candidate) == set(MODULE.CANDIDATES)
    assert all(value.shape[1] == 4 for value in batch.x_by_candidate.values())


def test_group_geometry_is_deterministic() -> None:
    older = np.array([[index, 0.0, float(index)] for index in range(12)])
    previous = older.copy()
    previous[:, 0] += np.arange(12) ** 2 * 0.01
    groups_a = MODULE.marker_groups(previous, older)
    groups_b = MODULE.marker_groups(previous, older)
    for name in MODULE.CANDIDATES:
        np.testing.assert_array_equal(groups_a[name], groups_b[name])
        assert groups_a[name].shape == (2,)
    lower = MODULE.lower_query_indices(previous, older)
    assert lower.size == 4
    np.testing.assert_array_equal(lower, [0, 1, 2, 3])


def test_task_and_generic_can_select_different_groups() -> None:
    rng = np.random.default_rng(7)
    batches = []
    for recording_index in range(5):
        count = 60
        signal = rng.normal(size=(count, 1))
        y = np.repeat(signal, 4, axis=1) + 0.05 * rng.normal(
            size=(count, 4)
        )
        x_by_candidate = {}
        for name in MODULE.CANDIDATES:
            if name == "lower":
                x = np.column_stack(
                    (
                        signal[:, 0],
                        rng.normal(scale=0.1, size=(count, 3)),
                    )
                )
            elif name == "fast":
                x = rng.normal(scale=20.0, size=(count, 4))
            else:
                x = rng.normal(size=(count, 4))
            x_by_candidate[name] = x
        key = MODULE.RecordingKey(
            Path(f"recording_{recording_index}.csv"),
            f"recording_{recording_index}.csv",
            "cotton",
            "shake",
            "A3",
        )
        batches.append(
            MODULE.WindowBatch(
                key,
                0.1,
                x_by_candidate,
                y,
                np.zeros_like(y),
                y,
            )
        )
    decisions = MODULE.choose_groups(
        {("shake", 0.1): batches},
        ridge=1e-3,
    )
    row = decisions["shake/0.1"]
    assert row["task_selected"] == "lower"
    assert row["generic_selected"] == "fast"


def test_result_validation_binds_target_to_gate() -> None:
    result = {
        "schema": MODULE.RESULT_SCHEMA,
        "schema_version": 1,
        "source_gate": {"passed": False},
        "target_contents_opened": False,
        "target": None,
    }
    result["result_id"] = MODULE.canonical_sha256(result)
    unhashed = dict(result)
    unhashed.pop("result_id")
    result["result_id"] = MODULE.canonical_sha256(unhashed)
    MODULE.validate_result(result)

    invalid = dict(result)
    invalid["target_contents_opened"] = True
    unhashed = dict(invalid)
    unhashed.pop("result_id")
    invalid["result_id"] = MODULE.canonical_sha256(unhashed)
    with pytest.raises(ValueError, match="target access"):
        MODULE.validate_result(invalid)


def test_request_hash_and_information_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "dataset"
    monkeypatch.setattr(MODULE, "EXPECTED_ROOT", root)
    request = {
        "schema": MODULE.REQUEST_SCHEMA,
        "schema_version": 1,
        "stage": "source-gated-evaluation",
        "dataset_root": str(root),
        "source_fit_materials": ["cotton"],
        "source_gate_materials": ["denim"],
        "target_materials": ["wool", "polyester"],
        "candidate_groups": list(MODULE.CANDIDATES),
        "primary_scenarios": list(MODULE.PRIMARY_SCENARIOS),
        "information_boundary": {
            "target_file_contents_opened_before_source_gate": False,
            "rgb_or_depth_data_used": False,
            "physical_command_sent": False,
            "dataset_modified": False,
            "paper_claim_authorized": False,
        },
    }
    request["request_id"] = MODULE.canonical_sha256(request)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert MODULE.load_request(path)["request_id"] == request["request_id"]
