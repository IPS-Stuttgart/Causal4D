from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.external_bridge_doctor import main as doctor_main
from causal4d.cli.external_rollout_import import main as rollout_import_main
from causal4d.external_bridge import build_external_bridge_report
from causal4d.external_forecast import (
    ExternalForecastBundle,
    save_external_forecast,
)
from causal4d.external_rollout import (
    EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
    import_external_rollouts,
    load_external_rollout_bank,
    save_external_rollout_bank,
)


def _write_rollout_source(path: Path, *, offset_m: float = 0.0) -> None:
    trajectories = np.zeros((2, 4, 2, 3), dtype=np.float64)
    trajectories[0, :, 0, 0] = np.asarray([0.0, 0.01, 0.02, 0.03]) + offset_m
    trajectories[1, :, 0, 0] = np.asarray([0.0, 0.02, 0.04, 0.06]) + offset_m
    trajectories[:, :, 1, 1] = 0.25
    np.savez_compressed(
        path,
        nodes=np.asarray([10, 20], dtype=np.int64),
        trajectories=trajectories,
        times=np.asarray([0.0, 0.5, 1.0, 1.5]),
        weights=np.asarray([0.6, 0.4]),
        ids=np.asarray(["slow", "fast"]),
        parameters=np.asarray([[100.0], [200.0]]),
    )


def _write_manifest(path: Path, **overrides: object) -> None:
    payload: dict[str, object] = {
        "schema": EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
        "schema_version": 1,
        "case_id": "cloth",
        "source": {
            "simulator": "example-mpm",
            "revision": "abc123",
            "artifact_id": "run-1",
        },
        "arrays": {
            "node_ids": "nodes",
            "trajectories": "trajectories",
            "frame_times_s": "times",
            "rollout_weights": "weights",
            "rollout_ids": "ids",
            "parameter_values": "parameters",
        },
        "layout": "RTNC",
        "coordinate_frame": "world",
        "position_unit": "m",
        "anchor_time_s": 0.0,
        "parameter_names": ["young_modulus_pa"],
        "metadata": {"producer": "unit-test"},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _forecast() -> ExternalForecastBundle:
    future = np.zeros((1, 3, 1, 3), dtype=np.float64)
    future[0, :, 0, 0] = [0.01, 0.02, 0.03]
    return ExternalForecastBundle(
        case_id="cloth",
        source_model="MolmoMotion",
        source_revision="checkpoint",
        forecast_ids=("instruction",),
        node_indices=np.asarray([10], dtype=np.int64),
        anchor_positions_m=np.zeros((1, 3)),
        future_positions_m=future,
        physical_frame_indices=np.asarray([1.0, 2.0, 3.0]),
        future_times_s=np.asarray([0.5, 1.0, 1.5]),
    )


def test_external_rollout_import_roundtrip_and_doctor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "rollouts.npz"
    manifest = tmp_path / "rollouts.json"
    canonical = tmp_path / "rollout_bank.npz"
    forecast_path = tmp_path / "forecast.npz"
    report_path = tmp_path / "doctor.json"
    _write_rollout_source(source)
    _write_manifest(manifest)

    imported = import_external_rollouts(source, manifest)
    assert imported.bank.trajectories.shape == (2, 1, 4, 2, 3)
    assert imported.bank.hypothesis_ids == ("slow", "fast")
    assert imported.parameter_names == ("young_modulus_pa",)
    assert imported.bank.hypothesis_metadata[1]["parameters"] == {
        "young_modulus_pa": 200.0
    }
    save_external_rollout_bank(canonical, imported)
    loaded = load_external_rollout_bank(canonical)
    assert loaded.artifact_id == imported.artifact_id
    assert loaded.node_ids.tolist() == [10, 20]

    forecast = _forecast()
    save_external_forecast(forecast_path, forecast)
    report = build_external_bridge_report(
        forecast,
        "instruction",
        loaded,
        anchor_tolerance_m=1e-6,
    )
    assert report["beta_zero_weights_bit_identical"] is True
    assert report["matched_node_ids"] == [10]
    assert report["rollout_fractional_frame_indices"] == [1.0, 2.0, 3.0]
    assert report["warnings"] == []

    assert (
        rollout_import_main([str(source), str(manifest), str(canonical)]) == 0
    )
    import_summary = json.loads(capsys.readouterr().out)
    assert import_summary["rollout_count"] == 2
    assert (
        doctor_main(
            [
                str(forecast_path),
                str(canonical),
                "instruction",
                str(report_path),
                "--anchor-tolerance-m",
                "0.000001",
            ]
        )
        == 0
    )
    cli_report = json.loads(capsys.readouterr().out)
    assert cli_report["valid"] is True
    assert json.loads(report_path.read_text(encoding="utf-8")) == cli_report


def test_camera_mm_rollouts_are_converted_to_world_metres(tmp_path: Path) -> None:
    source = tmp_path / "camera.npz"
    transform = np.eye(4)
    transform[:3, 3] = [1.0, 2.0, 3.0]
    np.savez_compressed(
        source,
        nodes=np.asarray([10]),
        trajectories=np.asarray([[[[1000.0, 0.0, 0.0]], [[2000.0, 0.0, 0.0]]]]),
        times=np.asarray([0.0, 1.0]),
        weights=np.asarray([1.0]),
        c2w=transform,
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        arrays={
            "node_ids": "nodes",
            "trajectories": "trajectories",
            "frame_times_s": "times",
            "rollout_weights": "weights",
            "camera_to_world": "c2w",
        },
        coordinate_frame="camera",
        position_unit="mm",
        parameter_names=[],
    )
    imported = import_external_rollouts(source, manifest)
    assert np.allclose(imported.bank.trajectories[0, 0, 0, 0], [2.0, 2.0, 3.0])
    assert np.allclose(imported.bank.trajectories[0, 0, 1, 0], [3.0, 2.0, 3.0])


def test_import_rejects_duplicate_node_ids(tmp_path: Path) -> None:
    source = tmp_path / "bad.npz"
    np.savez_compressed(
        source,
        nodes=np.asarray([10, 10]),
        trajectories=np.zeros((1, 2, 2, 3)),
        times=np.asarray([0.0, 1.0]),
        weights=np.asarray([1.0]),
    )
    manifest = tmp_path / "manifest.json"
    _write_manifest(
        manifest,
        arrays={
            "node_ids": "nodes",
            "trajectories": "trajectories",
            "frame_times_s": "times",
            "rollout_weights": "weights",
        },
        parameter_names=[],
    )
    with pytest.raises(ValueError, match="unique"):
        import_external_rollouts(source, manifest)


def test_doctor_rejects_missing_node_and_time_support(tmp_path: Path) -> None:
    source = tmp_path / "rollouts.npz"
    manifest = tmp_path / "rollouts.json"
    _write_rollout_source(source)
    _write_manifest(manifest)
    rollouts = import_external_rollouts(source, manifest)

    missing_node = ExternalForecastBundle(
        case_id="cloth",
        source_model="model",
        forecast_ids=("f",),
        node_indices=np.asarray([999]),
        anchor_positions_m=np.zeros((1, 3)),
        future_positions_m=np.zeros((1, 1, 1, 3)),
        physical_frame_indices=np.asarray([1.0]),
        future_times_s=np.asarray([0.5]),
    )
    with pytest.raises(ValueError, match="absent"):
        build_external_bridge_report(missing_node, "f", rollouts)

    outside = ExternalForecastBundle(
        case_id="cloth",
        source_model="model",
        forecast_ids=("f",),
        node_indices=np.asarray([10]),
        anchor_positions_m=np.zeros((1, 3)),
        future_positions_m=np.zeros((1, 1, 1, 3)),
        physical_frame_indices=np.asarray([1.0]),
        future_times_s=np.asarray([2.0]),
    )
    with pytest.raises(ValueError, match="time support"):
        build_external_bridge_report(outside, "f", rollouts)


def test_doctor_reports_anchor_warning_and_strict_exit(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "rollouts.npz"
    manifest = tmp_path / "rollouts.json"
    canonical = tmp_path / "rollout_bank.npz"
    forecast_path = tmp_path / "forecast.npz"
    report_path = tmp_path / "doctor.json"
    _write_rollout_source(source, offset_m=0.1)
    _write_manifest(manifest)
    rollouts = import_external_rollouts(source, manifest)
    save_external_rollout_bank(canonical, rollouts)
    save_external_forecast(forecast_path, _forecast())

    result = doctor_main(
        [
            str(forecast_path),
            str(canonical),
            "instruction",
            str(report_path),
            "--anchor-tolerance-m",
            "0.01",
            "--strict-warnings",
        ]
    )
    assert result == 3
    report = json.loads(capsys.readouterr().out)
    assert report["warnings"]
