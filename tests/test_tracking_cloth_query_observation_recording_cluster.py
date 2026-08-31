"""Tests for complete-recording clustering in the Tracking Cloth evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from causal4d_public import (
    tracking_cloth_query_observation_recording_cluster as cluster,
)


def _row(
    recording: str,
    horizon: float,
    *,
    task: float,
    generic: float,
    baseline: float,
    material: str = "denim",
    scenario: str = "shake",
) -> dict[str, Any]:
    def metrics(mse: float) -> dict[str, float]:
        return {
            "mse_m2": mse,
            "rmse_mm": 1000.0 * mse**0.5,
            "gaussian_nll": mse + 1.0,
            "marginal_90_coverage": 0.8,
        }

    return {
        "recording": recording,
        "material": material,
        "scenario": scenario,
        "size": "A3",
        "horizon_seconds": horizon,
        "windows": 20,
        "task_selected": "lower",
        "generic_selected": "upper",
        "random_selected": "central",
        "arms": {
            "task_conditioned": metrics(task),
            "generic_information": metrics(generic),
            "constant_velocity": metrics(baseline),
        },
    }


def test_aggregate_weights_recordings_not_recording_horizon_rows() -> None:
    rows = [
        _row("a.csv", 0.1, task=1.0, generic=2.0, baseline=3.0),
        _row("b.csv", 0.1, task=9.0, generic=10.0, baseline=11.0),
        _row("b.csv", 0.25, task=9.0, generic=10.0, baseline=11.0),
        _row("b.csv", 0.5, task=9.0, generic=10.0, baseline=11.0),
    ]
    aggregate = cluster.aggregate_rows(rows)
    assert aggregate["recordings"] == 2
    assert aggregate["recording_horizon_rows"] == 4
    assert aggregate["arms"]["task_conditioned"]["equal_recording_mse_m2"] == 5.0


def test_source_win_fraction_has_one_vote_per_complete_recording() -> None:
    rows = [
        _row("winner.csv", 0.1, task=0.0, generic=10.0, baseline=10.0),
        _row("loser.csv", 0.1, task=2.0, generic=1.0, baseline=3.0),
        _row("loser.csv", 0.25, task=2.0, generic=1.0, baseline=3.0),
        _row("loser.csv", 0.5, task=2.0, generic=1.0, baseline=3.0),
    ]
    request = {
        "source_gate": {
            "minimum_improvement_vs_generic": 0.0,
            "minimum_improvement_vs_constant_velocity": 0.0,
            "minimum_recording_win_fraction": 0.4,
            "maximum_worst_scenario_ratio": 1.15,
            "minimum_distinct_selection_rows": 1,
        }
    }
    gate = cluster.source_gate(rows, request)
    assert gate["passed"] is True
    assert gate["task_vs_generic_recording_win_fraction"] == 0.5
    assert gate["distinct_registered_task_selections"] == 3


def test_bootstrap_clusters_horizons_before_resampling() -> None:
    rows = [
        _row("a.csv", 0.1, task=10.0, generic=0.0, baseline=11.0),
        _row("b.csv", 0.1, task=-1.0, generic=0.0, baseline=11.0),
        _row("b.csv", 0.25, task=-1.0, generic=0.0, baseline=11.0),
        _row("b.csv", 0.5, task=-1.0, generic=0.0, baseline=11.0),
    ]
    result = cluster._bootstrap_difference(
        rows,
        "task_conditioned",
        "generic_information",
        seed=7,
        draws=1000,
    )
    assert result["mean_m2"] == 4.5


def test_duplicate_horizon_and_inconsistent_selection_fail_closed() -> None:
    duplicate = [
        _row("a.csv", 0.1, task=1.0, generic=2.0, baseline=3.0),
        _row("a.csv", 0.1, task=1.0, generic=2.0, baseline=3.0),
    ]
    with pytest.raises(ValueError, match="duplicate horizon"):
        cluster.aggregate_rows(duplicate)

    inconsistent = [
        _row("a.csv", 0.1, task=1.0, generic=2.0, baseline=3.0),
        _row("b.csv", 0.1, task=1.0, generic=2.0, baseline=3.0),
    ]
    inconsistent[1]["task_selected"] = "central"
    request = {
        "source_gate": {
            "minimum_improvement_vs_generic": 0.0,
            "minimum_improvement_vs_constant_velocity": 0.0,
            "minimum_recording_win_fraction": 0.0,
            "maximum_worst_scenario_ratio": 2.0,
            "minimum_distinct_selection_rows": 0,
        }
    }
    with pytest.raises(ValueError, match="selection changed"):
        cluster.source_gate(inconsistent, request)


def test_run_evaluation_refreshes_result_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cluster,
        "_ORIGINAL_RUN_EVALUATION",
        lambda _root, _request: {"result_id": "old", "decision": "synthetic"},
    )
    result = cluster.run_evaluation(Path("."), {})
    assert result["analysis_unit_contract"]["unit"] == "complete recording"
    unhashed = dict(result)
    supplied = unhashed.pop("result_id")
    assert supplied == cluster.canonical_sha256(unhashed)


def test_frozen_base_is_patched_before_main_execution() -> None:
    assert cluster._BASE.aggregate_rows is cluster.aggregate_rows
    assert cluster._BASE.source_gate is cluster.source_gate
    assert cluster._BASE._bootstrap_difference is cluster._bootstrap_difference
    assert cluster._BASE.run_evaluation is cluster.run_evaluation
