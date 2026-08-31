"""Recording-cluster correction for the frozen Tracking Cloth experiment.

The original evaluator declared complete recordings as the analysis units but
aggregated and bootstrapped recording--horizon rows. This module changes only
that bookkeeping: registered horizons are averaged inside each recording before
source-win counting, aggregate metrics, and material-stratified bootstrapping.
All data splits, marker choices, fitted models, thresholds, and target-access
rules remain those of ``tracking_cloth_query_observation``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from typing import Any

import numpy as np

_BASE_PATH = Path(__file__).with_name("tracking_cloth_query_observation.py")
_BASE_NAME = "causal4d_tracking_cloth_query_observation_frozen_v1"
_SPEC = importlib.util.spec_from_file_location(_BASE_NAME, _BASE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"cannot load frozen evaluator from {_BASE_PATH}")
_BASE: Any = importlib.util.module_from_spec(_SPEC)
sys.modules[_BASE_NAME] = _BASE
_SPEC.loader.exec_module(_BASE)
_ORIGINAL_RUN_EVALUATION = _BASE.run_evaluation

PILOT_KIND = _BASE.PILOT_KIND
EXPECTED_ROOT = _BASE.EXPECTED_ROOT
load_request = _BASE.load_request
canonical_sha256 = _BASE.canonical_sha256


def _group_recordings(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        recording = str(row["recording"])
        grouped.setdefault(recording, []).append(row)
    if not grouped:
        raise ValueError("no scored recording rows")
    for recording, group in grouped.items():
        materials = {str(row["material"]) for row in group}
        scenarios = {str(row["scenario"]) for row in group}
        horizons = [float(row["horizon_seconds"]) for row in group]
        if len(materials) != 1 or len(scenarios) != 1:
            raise ValueError(f"recording metadata changed within {recording}")
        if len(horizons) != len(set(horizons)):
            raise ValueError(f"duplicate horizon row for {recording}")
    return grouped


def _cluster_metric(
    rows: list[dict[str, Any]],
    arm: str,
    metric: str,
) -> float:
    return float(np.mean([float(row["arms"][arm][metric]) for row in rows]))


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = _group_recordings(rows)
    arms = sorted(rows[0]["arms"])
    for row in rows:
        if sorted(row["arms"]) != arms:
            raise ValueError("arm roster differs across scored rows")
    aggregate: dict[str, Any] = {
        "recordings": len(grouped),
        "recording_horizon_rows": len(rows),
        "horizon_clustering": "mean-within-complete-recording-v1",
        "arms": {},
    }
    for arm in arms:
        mse = np.asarray(
            [_cluster_metric(group, arm, "mse_m2") for group in grouped.values()],
            dtype=np.float64,
        )
        nll = np.asarray(
            [_cluster_metric(group, arm, "gaussian_nll") for group in grouped.values()],
            dtype=np.float64,
        )
        coverage = np.asarray(
            [
                _cluster_metric(group, arm, "marginal_90_coverage")
                for group in grouped.values()
            ],
            dtype=np.float64,
        )
        aggregate["arms"][arm] = {
            "equal_recording_mse_m2": float(np.mean(mse)),
            "equal_recording_rmse_mm": 1000.0 * float(np.sqrt(np.mean(mse))),
            "equal_recording_gaussian_nll": float(np.mean(nll)),
            "equal_recording_marginal_90_coverage": float(np.mean(coverage)),
        }
    baseline = aggregate["arms"]["constant_velocity"]["equal_recording_mse_m2"]
    for arm in arms:
        value = aggregate["arms"][arm]["equal_recording_mse_m2"]
        aggregate["arms"][arm]["relative_mse_improvement_vs_constant_velocity"] = (
            float((baseline - value) / baseline) if baseline > 0.0 else 0.0
        )
    return aggregate


def source_gate(rows: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    grouped = _group_recordings(rows)
    aggregate = aggregate_rows(rows)
    task = aggregate["arms"]["task_conditioned"]["equal_recording_mse_m2"]
    generic = aggregate["arms"]["generic_information"]["equal_recording_mse_m2"]
    baseline = aggregate["arms"]["constant_velocity"]["equal_recording_mse_m2"]
    task_vs_generic = (generic - task) / generic
    task_vs_baseline = (baseline - task) / baseline
    recording_differences = np.asarray(
        [
            float(
                np.mean(
                    [
                        row["arms"]["generic_information"]["mse_m2"]
                        - row["arms"]["task_conditioned"]["mse_m2"]
                        for row in group
                    ]
                )
            )
            for group in grouped.values()
        ],
        dtype=np.float64,
    )
    win_fraction = float(np.mean(recording_differences > 0.0))

    ratios: dict[str, float] = {}
    for scenario in _BASE.PRIMARY_SCENARIOS:
        scenario_groups = [
            group for group in grouped.values() if str(group[0]["scenario"]) == scenario
        ]
        if not scenario_groups:
            continue
        task_s = float(
            np.mean(
                [
                    _cluster_metric(group, "task_conditioned", "mse_m2")
                    for group in scenario_groups
                ]
            )
        )
        generic_s = float(
            np.mean(
                [
                    _cluster_metric(group, "generic_information", "mse_m2")
                    for group in scenario_groups
                ]
            )
        )
        ratios[scenario] = task_s / generic_s if generic_s > 0.0 else float("inf")

    selections: dict[tuple[str, float], tuple[str, str]] = {}
    for row in rows:
        key = (str(row["scenario"]), float(row["horizon_seconds"]))
        pair = (str(row["task_selected"]), str(row["generic_selected"]))
        previous = selections.setdefault(key, pair)
        if previous != pair:
            raise ValueError(f"selection changed across recordings for {key}")
    distinct = sum(
        task_group != generic_group for task_group, generic_group in selections.values()
    )

    thresholds = request["source_gate"]
    checks = {
        "minimum_improvement_vs_generic": task_vs_generic
        >= float(thresholds["minimum_improvement_vs_generic"]),
        "minimum_improvement_vs_constant_velocity": task_vs_baseline
        >= float(thresholds["minimum_improvement_vs_constant_velocity"]),
        "minimum_recording_win_fraction": win_fraction
        >= float(thresholds["minimum_recording_win_fraction"]),
        "maximum_worst_scenario_ratio": max(ratios.values(), default=float("inf"))
        <= float(thresholds["maximum_worst_scenario_ratio"]),
        "minimum_distinct_selection_rows": distinct
        >= int(thresholds["minimum_distinct_selection_rows"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_vs_generic_relative_mse_improvement": float(task_vs_generic),
        "task_vs_constant_velocity_relative_mse_improvement": float(task_vs_baseline),
        "task_vs_generic_recording_win_fraction": win_fraction,
        "task_to_generic_scenario_mse_ratios": ratios,
        "distinct_task_vs_generic_selection_rows": int(distinct),
        "distinct_registered_task_selections": int(distinct),
        "win_fraction_unit": "complete recording averaged across horizons",
        "aggregate": aggregate,
    }


def _bootstrap_difference(
    rows: list[dict[str, Any]],
    arm_a: str,
    arm_b: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    grouped = _group_recordings(rows)
    by_material: dict[str, list[float]] = {}
    for group in grouped.values():
        material = str(group[0]["material"])
        difference = float(
            np.mean(
                [
                    row["arms"][arm_a]["mse_m2"] - row["arms"][arm_b]["mse_m2"]
                    for row in group
                ]
            )
        )
        by_material.setdefault(material, []).append(difference)
    rng = np.random.default_rng(seed)
    simulated = np.empty(draws, dtype=np.float64)
    materials = sorted(by_material)
    for index in range(draws):
        material_means: list[float] = []
        for material in materials:
            values = np.asarray(by_material[material], dtype=np.float64)
            sample = rng.choice(values, size=values.size, replace=True)
            material_means.append(float(np.mean(sample)))
        simulated[index] = float(np.mean(material_means))
    point = float(np.mean([np.mean(by_material[name]) for name in materials]))
    lower, upper = np.quantile(simulated, [0.025, 0.975])
    return {
        "mean_m2": point,
        "lower_m2": float(lower),
        "upper_m2": float(upper),
    }


def run_evaluation(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    result = _ORIGINAL_RUN_EVALUATION(root, request)
    result["analysis_unit_contract"] = {
        "schema_version": 1,
        "unit": "complete recording",
        "registered_horizons": "nested and averaged within recording",
        "source_win_fraction": "one indicator per complete recording",
        "target_bootstrap": "material-stratified resampling of complete recordings",
        "correction_stage": "before first source-payload evaluation",
    }
    unhashed = dict(result)
    unhashed.pop("result_id", None)
    result["result_id"] = canonical_sha256(unhashed)
    return result


_BASE.aggregate_rows = aggregate_rows
_BASE.source_gate = source_gate
_BASE._bootstrap_difference = _bootstrap_difference
_BASE.run_evaluation = run_evaluation
validate_result = _BASE.validate_result
write_summary = _BASE.write_summary


def main() -> int:
    return int(_BASE.main())


if __name__ == "__main__":
    raise SystemExit(main())
