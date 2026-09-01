from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from causal4d_public.pokeflex_realized_load import (
    CANONICAL_POKEFLEX_REALIZED_LOAD_POLICY_SHA256,
    PokeFlexRealizedLoadSourceConfig,
    TargetKinematicConditioning,
    build_forecast_bundle,
    realized_load_policy_sha256,
    run_pokeflex_realized_load_source_gate,
    validate_realized_load_artifact,
    validate_realized_load_policy,
)


def _source_qa(config: PokeFlexRealizedLoadSourceConfig) -> dict[str, object]:
    return {
        "artifact_kind": "PublicPokeFlexSourceQa",
        "schema_version": 1,
        "result_sha256": config.expected_source_qa_result_sha256,
        "source_qa_passed": True,
        "object_id": config.expected_object_id,
        "information_boundary": {
            "opened_take_ids": list(config.expected_development_take_ids),
            "unopened_take_ids": list(config.forbidden_take_ids),
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        },
        "capability_gates": {"pose_wrench_contact_candidate_ready": True},
    }


def _write_take(
    root: Path,
    take_id: str,
    *,
    shape: float,
    prefix: int,
    horizon: int,
) -> None:
    take_root = root / "3dPrintedBunny" / take_id
    take_root.mkdir(parents=True)
    onset = 10
    total = onset + prefix + horizon + 8
    window_time = np.arange(prefix + horizon, dtype=float)
    early = 4.0 + (0.20 + 0.35 * shape) * window_time
    peak = 8.0 + 8.0 * shape
    future_shape = peak + 2.5 * np.sin((window_time - prefix) / 5.0)
    blend = 1.0 / (1.0 + np.exp(-(window_time - prefix + 1.0)))
    window_force = (1.0 - blend) * early + blend * future_shape
    window_force += 0.05 * np.sin(0.7 * window_time + shape)
    force = np.zeros(total, dtype=float)
    force[onset : onset + len(window_force)] = window_force
    position = np.zeros((total, 3), dtype=float)
    position[:, 0] = 0.0015 * np.arange(total)
    position[:, 1] = 0.0002 * shape * np.arange(total)
    records = []
    for frame in range(total):
        transform = np.eye(4)
        transform[:3, 3] = position[frame]
        records.append(
            {
                "frame": f"{frame:05d}",
                "forces": [0.0, float(force[frame]), 0.0],
                "T_WT": transform.tolist(),
            }
        )
    (take_root / "robot_data.json").write_text(json.dumps(records), encoding="utf-8")


def _synthetic_config() -> PokeFlexRealizedLoadSourceConfig:
    return PokeFlexRealizedLoadSourceConfig(
        forecast_horizon_frames=24,
        gain_grid=(0.8, 0.9, 1.0, 1.1, 1.2),
        delay_grid_frames=(-1, 0, 1),
        minimum_mean_rmse_improvement_fraction_vs_persistence=0.02,
        minimum_take_win_fraction_vs_persistence=0.60,
        minimum_mean_rmse_improvement_fraction_vs_dependence_control=0.0,
        maximum_worst_take_rmse_ratio_vs_persistence=1.10,
        maximum_mean_rmse_ratio_vs_kinematic_ridge=1.20,
        bootstrap_replicates=2000,
    )


def test_canonical_policy_validates() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "causal4d_public"
        / "pokeflex_realized_load_source_v1.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    validation = validate_realized_load_policy(payload)
    assert validation["passed"] is True
    assert payload["config_sha256"] == CANONICAL_POKEFLEX_REALIZED_LOAD_POLICY_SHA256
    assert payload["config_sha256"] == realized_load_policy_sha256(payload)


def test_forecast_constructor_has_no_target_suffix_argument() -> None:
    parameters = inspect.signature(build_forecast_bundle).parameters
    assert "target_suffix" not in parameters
    assert tuple(parameters) == ("target_prefix", "target", "sources", "config")
    conditioning_fields = set(TargetKinematicConditioning.__dataclass_fields__)
    assert "window_force_n" not in conditioning_fields
    assert conditioning_fields == {
        "take_id",
        "window_phase",
        "window_speed_m_per_frame",
    }


def test_source_gate_is_deterministic_and_does_not_open_forbidden_takes(
    tmp_path: Path,
) -> None:
    config = _synthetic_config()
    shapes = (0.05, 0.20, 0.40, 0.65, 0.90)
    for take_id, shape in zip(
        config.expected_development_take_ids, shapes, strict=True
    ):
        _write_take(
            tmp_path,
            take_id,
            shape=shape,
            prefix=config.prefix_frame_count,
            horizon=config.forecast_horizon_frames,
        )
    forbidden_contents = {}
    for take_id in config.forbidden_take_ids:
        take_root = tmp_path / "3dPrintedBunny" / take_id
        take_root.mkdir(parents=True)
        path = take_root / "robot_data.json"
        path.write_text("FORBIDDEN PAYLOAD MUST NOT BE PARSED", encoding="utf-8")
        forbidden_contents[take_id] = path.read_bytes()

    first = run_pokeflex_realized_load_source_gate(
        tmp_path,
        _source_qa(config),
        tmp_path / "first",
        config,
    )
    second = run_pokeflex_realized_load_source_gate(
        tmp_path,
        _source_qa(config),
        tmp_path / "second",
        config,
    )

    assert first["result_sha256"] == second["result_sha256"]
    assert first["dataset"]["opened_take_ids"] == list(
        config.expected_development_take_ids
    )
    assert first["information_boundary"]["calibration_take_data_read"] is False
    assert first["information_boundary"]["target_take_data_read"] is False
    assert first["information_boundary"]["development_meshes_read"] is False
    assert len(first["per_take_results"]) == 5
    assert validate_realized_load_artifact(first)["passed"] is True
    for take_id, expected in forbidden_contents.items():
        path = tmp_path / "3dPrintedBunny" / take_id / "robot_data.json"
        assert path.read_bytes() == expected

    proposed = first["aggregate_equal_take_results"]["posterior_mean"]["force_rmse_n"][
        "mean"
    ]
    persistence = first["aggregate_equal_take_results"]["persistence"]["force_rmse_n"][
        "mean"
    ]
    assert proposed < persistence
    for take_id in config.expected_development_take_ids:
        seal_path = tmp_path / "first" / f"prediction_seal_{take_id}.json"
        assert seal_path.is_file()
        seal = json.loads(seal_path.read_text(encoding="utf-8"))
        metadata = seal["metadata"]
        assert (
            metadata["dependence_control_component_prediction_marginal_preserved"]
            is True
        )
        assert (
            metadata["posterior"]["component_prediction_multiset_sha256"]
            == metadata["dependence_destroyed"]["component_prediction_multiset_sha256"]
        )


def test_policy_rejects_target_access() -> None:
    config = PokeFlexRealizedLoadSourceConfig()
    payload = {
        "schema_version": 1,
        "artifact_kind": "PublicPokeFlexRealizedLoadSourcePolicy",
        "config": config.as_dict(),
        "information_boundary": {
            "development_robot_records_only": True,
            "development_force_outcomes_allowed_for_source_gate": True,
            "development_meshes_allowed": False,
            "calibration_take_data_allowed": False,
            "target_take_data_allowed": True,
            "automatic_calibration_or_target_dispatch_allowed": False,
            "new_physical_data_collected": False,
        },
    }
    payload["config_sha256"] = realized_load_policy_sha256(payload)
    try:
        validate_realized_load_policy(payload)
    except ValueError as error:
        assert "canonical lock" in str(error) or "information boundary" in str(error)
    else:
        raise AssertionError("target-enabled policy was accepted")
