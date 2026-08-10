from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.external_bridge_run import main as bridge_run_main
from causal4d.cli.external_bridge_workflow import main as workflow_main
from causal4d.cli.external_node_mapping import main as mapping_main
from causal4d.external_bridge_run import (
    analyze_external_bridge,
    publish_external_bridge_run,
)
from causal4d.external_forecast import ExternalForecastBundle, save_external_forecast
from causal4d.external_node_mapping import build_external_node_mapping
from causal4d.external_reference import load_external_reference
from causal4d.external_rollout import (
    EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
    import_external_rollouts,
    save_external_rollout_bank,
)


def _write_rollout_source(path: Path) -> None:
    trajectories = np.zeros((2, 4, 2, 3), dtype=np.float64)
    trajectories[0, :, 0, 0] = [0.0, 0.01, 0.02, 0.03]
    trajectories[1, :, 0, 0] = [0.0, 0.02, 0.04, 0.06]
    trajectories[:, :, 1, 1] = 0.25
    np.savez_compressed(
        path,
        nodes=np.asarray([10, 20], dtype=np.int64),
        trajectories=trajectories,
        times=np.asarray([0.0, 0.5, 1.0, 1.5]),
        weights=np.asarray([0.5, 0.5]),
        ids=np.asarray(["slow", "fast"]),
    )


def _write_rollout_manifest(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
                "schema_version": 1,
                "case_id": "cloth",
                "source": {"simulator": "unit-test", "revision": "abc"},
                "arrays": {
                    "node_ids": "nodes",
                    "trajectories": "trajectories",
                    "frame_times_s": "times",
                    "rollout_weights": "weights",
                    "rollout_ids": "ids",
                },
                "layout": "RTNC",
                "coordinate_frame": "world",
                "position_unit": "m",
                "anchor_time_s": 0.0,
            }
        ),
        encoding="utf-8",
    )


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


def _write_reference(path: Path) -> None:
    positions = np.zeros((5, 2, 3), dtype=np.float64)
    positions[:, 0, 0] = [-0.01, 0.0, 0.01, 0.02, 0.03]
    positions[:, 1, 1] = 0.25
    np.savez_compressed(
        path,
        case_id=np.asarray("cloth"),
        node_ids=np.asarray([10, 20], dtype=np.int64),
        positions_world_m=positions,
        frame_times_s=np.asarray([-0.5, 0.0, 0.5, 1.0, 1.5]),
        validity_mask=np.ones((5, 2), dtype=bool),
    )


def test_bridge_run_publishes_metrics_and_exact_fallback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = tmp_path / "rollouts.npz"
    manifest = tmp_path / "rollouts.json"
    rollout_bank_path = tmp_path / "canonical_rollouts.npz"
    forecast_path = tmp_path / "forecast.npz"
    reference_path = tmp_path / "reference.npz"
    output = tmp_path / "result"
    _write_rollout_source(source)
    _write_rollout_manifest(manifest)
    rollouts = import_external_rollouts(source, manifest)
    save_external_rollout_bank(rollout_bank_path, rollouts)
    forecast = _forecast()
    save_external_forecast(forecast_path, forecast)
    _write_reference(reference_path)
    reference = load_external_reference(reference_path)

    report, arrays = analyze_external_bridge(
        forecast,
        "instruction",
        rollouts,
        beta_values=(0.0, 3.0, 12.0),
        scale_m=0.005,
        reference=reference,
    )
    assert report["doctor"]["beta_zero_weights_bit_identical"] is True
    assert arrays["posterior_weights"][0].tobytes() == (
        rollouts.bank.prior_joint_weights.tobytes()
    )
    assert arrays["posterior_weights"][1, 0, 0] > 0.5
    metrics = {row["method"]: row for row in report["metrics"]}
    assert metrics["constant_velocity"]["ade_m"] == pytest.approx(0.0)
    assert metrics["external_forecast"]["ade_m"] == pytest.approx(0.0)
    assert metrics["semantic_beta_3"]["ade_m"] < metrics["physical_prior"]["ade_m"]
    assert report["evaluation_only_best_beta"] in {3.0, 12.0}

    published = publish_external_bridge_run(output, report, arrays)
    assert len(published["manifest_id"]) == 64
    for name in (
        "doctor.json",
        "summary.json",
        "summary.md",
        "metrics.csv",
        "weights.csv",
        "error_vs_horizon.csv",
        "error_vs_horizon.svg",
        "predictions.npz",
        "manifest.json",
    ):
        assert (output / name).is_file()
    assert "constant_velocity" in (output / "metrics.csv").read_text(encoding="utf-8")

    cli_output = tmp_path / "cli-result"
    assert (
        bridge_run_main(
            [
                str(forecast_path),
                str(rollout_bank_path),
                "instruction",
                str(cli_output),
                "--reference",
                str(reference_path),
                "--beta",
                "0",
                "--beta",
                "3",
                "--scale-m",
                "0.005",
            ]
        )
        == 0
    )
    summary = json.loads(capsys.readouterr().out)
    assert summary["exact_beta_zero_fallback"] is True
    assert summary["reference_evaluated"] is True


def test_reference_loader_rejects_unknown_members(tmp_path: Path) -> None:
    path = tmp_path / "reference.npz"
    np.savez_compressed(
        path,
        case_id=np.asarray("cloth"),
        node_ids=np.asarray([10]),
        positions_world_m=np.zeros((2, 1, 3)),
        frame_times_s=np.asarray([0.0, 1.0]),
        unexpected=np.asarray([1]),
    )
    with pytest.raises(ValueError, match="unexpected"):
        load_external_reference(path)


def test_geometric_node_mapping_is_unique_and_fail_closed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    queries = np.asarray([[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]])
    nodes = np.asarray([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [1.0, 0.0, 0.0]])
    node_ids = np.asarray([100, 200, 300])
    report = build_external_node_mapping(
        queries,
        nodes,
        node_ids,
        query_ids=("left", "right"),
        maximum_distance_m=0.09,
    )
    assert report["accepted"] is True
    assigned = [entry["node_id"] for entry in report["entries"]]
    assert len(set(assigned)) == 2
    assert assigned == [100, 200]

    query_path = tmp_path / "query.npz"
    node_path = tmp_path / "nodes.npz"
    output_json = tmp_path / "mapping.json"
    output_npz = tmp_path / "mapping.npz"
    output_svg = tmp_path / "mapping.svg"
    np.savez_compressed(
        query_path,
        anchors=queries,
        labels=np.asarray(["left", "right"]),
    )
    np.savez_compressed(node_path, positions=nodes, ids=node_ids)
    result = mapping_main(
        [
            str(query_path),
            str(node_path),
            str(output_json),
            "--output-npz",
            str(output_npz),
            "--output-svg",
            str(output_svg),
            "--query-position-key",
            "anchors",
            "--query-id-key",
            "labels",
            "--node-position-key",
            "positions",
            "--node-id-key",
            "ids",
            "--maximum-distance-m",
            "0.015",
        ]
    )
    assert result == 3
    cli_summary = json.loads(capsys.readouterr().out)
    assert cli_summary["accepted"] is False
    assert output_json.is_file()
    assert output_npz.is_file()
    assert output_svg.is_file()


def test_single_module_bridge_workflow_dispatches_help() -> None:
    assert workflow_main(["--help"]) == 0
