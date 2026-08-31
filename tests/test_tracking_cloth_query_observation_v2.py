from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src/causal4d_public/tracking_cloth_query_observation_v2.py"
MODULE_NAME = "tracking_cloth_query_observation_v2_test_module"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = MODULE
SPEC.loader.exec_module(MODULE)


def _write_csv(path: Path, *, markers: int = 12, frames: int = 240) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Format Version", "1.0"])
        writer.writerow(["Capture Frame Rate", "120"])
        writer.writerow(["marker ids"])
        writer.writerow(["units", "millimeters"])
        writer.writerow(
            ["Frame", "Time"] + [f"coordinate_{index}" for index in range(3 * markers)]
        )
        for frame in range(frames):
            time = frame / 120.0
            values: list[float] = []
            for marker in range(markers):
                x = 100.0 * (marker % 4) + 5.0 * np.sin(time * 3 + marker)
                y = 80.0 * (marker // 4) + 2.0 * np.cos(time * 2 + marker)
                z = 500.0 - 30.0 * (marker // 4) + 10.0 * np.sin(time * 5 + marker / 3)
                values.extend([x, y, z])
            writer.writerow([frame, time, *values])


def _key(
    index: int,
    *,
    material: str = "cotton",
    scenario: str = "shake",
    size: str = "A3",
):
    return MODULE.RecordingKey(
        Path(f"{material}_{size}_{scenario}_{index}.csv"),
        f"{material}_{size}_{scenario}_{index}.csv",
        material,
        scenario,
        size,
    )


def _batch(
    index: int,
    horizon: float,
    x_by_candidate: dict[str, np.ndarray],
    query: np.ndarray,
    global_state: np.ndarray,
    *,
    material: str = "cotton",
    scenario: str = "shake",
    size: str = "A3",
):
    zeros = np.zeros_like(query)
    return MODULE.WindowBatch(
        _key(index, material=material, scenario=scenario, size=size),
        horizon,
        x_by_candidate,
        query,
        global_state,
        zeros,
        query,
    )


def _score_row(
    recording: str,
    horizon: float,
    base: float,
    *,
    material: str,
    size: str,
    scenario: str = "shake",
) -> dict[str, object]:
    arms = {}
    for offset, arm in enumerate(MODULE.ARMS):
        mse = base + offset * 0.01
        arms[arm] = {
            "mse_m2": mse,
            "rmse_mm": 1000.0 * np.sqrt(mse),
            "gaussian_nll": mse + 1.0,
            "marginal_90_coverage": 0.8,
            "normalized_joint_nees": 1.2,
        }
    return {
        "recording": recording,
        "material": material,
        "scenario": scenario,
        "size": size,
        "horizon_seconds": horizon,
        "arms": arms,
    }


def test_csv_read_and_window_extraction(tmp_path: Path) -> None:
    path = tmp_path / "Free-hanging" / "cotton_A3_shake_fast_hanger.csv"
    _write_csv(path)
    key = MODULE.classify_recording(path, tmp_path)
    assert key is not None
    recording = MODULE.read_recording(key)
    assert recording.points_m.shape == (240, 12, 3)
    assert recording.unit_scale_to_m == pytest.approx(1e-3)
    batch = MODULE.extract_windows(
        recording,
        horizon_seconds=0.1,
        lag_seconds=0.05,
        stride_seconds=0.1,
    )
    assert batch.query_target.shape[1] == 4
    assert batch.global_state_target.shape[1] == 10
    assert set(batch.x_by_candidate) == set(MODULE.CANDIDATES)


def test_task_and_global_state_policies_can_differ() -> None:
    rng = np.random.default_rng(7)
    batches = []
    for recording_index in range(6):
        count = 80
        task_signal = rng.normal(size=(count, 1))
        global_signal = rng.normal(size=(count, 1))
        query = np.repeat(task_signal, 4, axis=1) + 0.03 * rng.normal(size=(count, 4))
        global_state = np.repeat(global_signal, 10, axis=1) + 0.03 * rng.normal(
            size=(count, 10)
        )
        x_by_candidate: dict[str, np.ndarray] = {}
        for name in MODULE.CANDIDATES:
            if name == "lower":
                x = np.column_stack(
                    (task_signal[:, 0], rng.normal(scale=0.1, size=(count, 3)))
                )
            elif name == "fast":
                x = np.column_stack(
                    (global_signal[:, 0], rng.normal(scale=0.1, size=(count, 3)))
                )
            else:
                x = rng.normal(size=(count, 4))
            x_by_candidate[name] = x
        batches.append(
            _batch(
                recording_index,
                0.1,
                x_by_candidate,
                query,
                global_state,
            )
        )
    decisions = MODULE.choose_policies({("shake", 0.1): batches}, ridge=1e-3)
    row = decisions["shake/0.1"]
    assert row["task_selected"] == "lower"
    assert row["global_state_selected"] == "fast"


def test_aggregation_uses_complete_recordings_not_horizon_rows() -> None:
    rows = [
        _score_row("a.csv", 0.1, 1.0, material="denim", size="A3"),
        _score_row("a.csv", 0.5, 3.0, material="denim", size="A3"),
        _score_row("b.csv", 0.1, 5.0, material="denim", size="A3"),
    ]
    aggregate = MODULE.aggregate_rows(rows)
    assert aggregate["recordings"] == 2
    assert aggregate["recording_horizon_rows"] == 3
    # Recording a contributes mean 2, recording b contributes 5: equal-recording mean 3.5.
    assert aggregate["arms"]["constant_velocity"][
        "equal_recording_mse_m2"
    ] == pytest.approx(3.5)


def test_target_aggregation_uses_four_physical_specimens() -> None:
    rows = [
        _score_row("wool_a2_1.csv", 0.1, 1.0, material="wool", size="A2"),
        _score_row("wool_a2_1.csv", 0.5, 3.0, material="wool", size="A2"),
        _score_row("wool_a2_2.csv", 0.1, 5.0, material="wool", size="A2"),
        _score_row("wool_a3_1.csv", 0.1, 7.0, material="wool", size="A3"),
        _score_row(
            "polyester_a2_1.csv",
            0.1,
            9.0,
            material="polyester",
            size="A2",
        ),
        _score_row(
            "polyester_a3_1.csv",
            0.1,
            11.0,
            material="polyester",
            size="A3",
        ),
    ]
    aggregate = MODULE.aggregate_specimens(rows)
    assert aggregate["unit"] == "material_size_physical_specimen"
    assert aggregate["specimens"] == 4
    assert aggregate["recordings"] == 5
    # wool/A2: mean of recording means ((1+3)/2 and 5) = 3.5.
    # The remaining specimens contribute 7, 9, and 11, so the equal-specimen mean is 7.625.
    assert aggregate["arms"]["constant_velocity"][
        "equal_specimen_mse_m2"
    ] == pytest.approx(7.625)
    contrast = MODULE._bootstrap_specimen_difference(
        rows,
        "constant_velocity",
        "source_mean_residual",
        seed=11,
        draws=100,
    )
    assert contrast["analysis_unit"] == "material_size_physical_specimen"
    assert contrast["specimens"] == 4
    assert set(contrast["observed_by_specimen_m2"]) == {
        "polyester/A2",
        "polyester/A3",
        "wool/A2",
        "wool/A3",
    }


def test_constant_velocity_is_exact_zero_residual() -> None:
    count = 20
    query = np.full((count, 4), 2.0)
    global_state = np.zeros((count, 10))
    x = {name: np.zeros((count, 4)) for name in MODULE.CANDIDATES}
    batch = _batch(0, 0.1, x, query, global_state, material="denim")
    covariance = np.eye(4)
    model = MODULE.LinearGaussianModel(
        np.zeros(4), np.ones(4), np.full(4, 3.0), np.zeros((4, 4)), covariance
    )
    models = {
        "shake/0.1": MODULE.PredictionModelSet(
            candidate_models={name: model for name in MODULE.CANDIDATES},
            source_mean=np.full(4, 1.0),
            source_mean_covariance=covariance,
            constant_velocity_covariance=covariance,
            dependence_destroyed=model,
        )
    }
    decisions = {
        "shake/0.1": {
            "task_selected": "lower",
            "global_state_selected": "fast",
        }
    }
    rows = MODULE.score_batches({("shake", 0.1): [batch]}, decisions, models)
    arms = rows[0]["arms"]
    assert arms["constant_velocity"]["mse_m2"] == pytest.approx(4.0)
    assert arms["source_mean_residual"]["mse_m2"] == pytest.approx(1.0)


def test_result_validation_binds_target_to_gate() -> None:
    result = {
        "schema": MODULE.RESULT_SCHEMA,
        "schema_version": 2,
        "pilot_kind": MODULE.PILOT_KIND,
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
        "schema_version": 2,
        "stage": "source-gated-evaluation-v2",
        "dataset_root": str(root),
        "source_fit_materials": ["cotton"],
        "source_gate_materials": ["denim"],
        "target_materials": ["wool", "polyester"],
        "target_model_training_materials": ["cotton"],
        "candidate_groups": list(MODULE.CANDIDATES),
        "primary_scenarios": list(MODULE.PRIMARY_SCENARIOS),
        "primary_statistical_unit": "material_size_physical_specimen",
        "source_gate_statistical_unit": ("complete_recording_after_averaging_horizons"),
        "target_nesting": ("horizons_within_recordings_within_material_size_specimens"),
        "target_specimen_count_if_opened": 4,
        "generic_policy_target": "current_global_affine_residual_field",
        "constant_velocity_residual_prediction": "exact_zero",
        "information_boundary": {
            "target_file_contents_opened_before_source_gate": False,
            "rgb_or_depth_data_used": False,
            "physical_command_sent": False,
            "dataset_modified": False,
            "paper_claim_authorized": False,
            "population_generalization_claim_authorized": False,
            "target_models_refit_after_gate": False,
        },
    }
    request["request_id"] = MODULE.canonical_sha256(request)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(request), encoding="utf-8")
    assert MODULE.load_request(path)["request_id"] == request["request_id"]
