"""Cluster the frozen Tracking Cloth evaluation at its valid analysis units.

Registered forecast horizons are nested within complete recordings. Repeated
recordings are in turn nested within the physical material--size specimen. The
frozen denim gate remains an operational repeated-recording rule, while target
uncertainty and the primary held-out summary use material--size specimens. All
data splits, marker choices, models, thresholds, and target-access rules remain
those of ``tracking_cloth_query_observation``.
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
_ORIGINAL_WRITE_SUMMARY = _BASE.write_summary

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
        sizes = {str(row["size"]) for row in group}
        horizons = [float(row["horizon_seconds"]) for row in group]
        if len(materials) != 1 or len(scenarios) != 1 or len(sizes) != 1:
            raise ValueError(f"recording metadata changed within {recording}")
        if len(horizons) != len(set(horizons)):
            raise ValueError(f"duplicate horizon row for {recording}")
    return grouped


def _group_specimens(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        material = str(row["material"])
        size = str(row["size"])
        if size not in {"A2", "A3"}:
            raise ValueError(f"unknown physical specimen size: {size}")
        grouped.setdefault(f"{material}/{size}", []).append(row)
    if not grouped:
        raise ValueError("no physical specimen rows")
    return grouped


def _cluster_metric(
    rows: list[dict[str, Any]],
    arm: str,
    metric: str,
) -> float:
    return float(np.mean([float(row["arms"][arm][metric]) for row in rows]))


def _specimen_metric(
    rows: list[dict[str, Any]],
    arm: str,
    metric: str,
) -> float:
    recordings = _group_recordings(rows)
    return float(
        np.mean(
            [_cluster_metric(group, arm, metric) for group in recordings.values()]
        )
    )


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Equal-weight complete recordings after averaging registered horizons."""
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


def aggregate_specimens(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Primary target summary over material--size physical specimens."""
    specimens = _group_specimens(rows)
    arms = sorted(rows[0]["arms"])
    by_specimen: dict[str, Any] = {}
    for specimen, group in sorted(specimens.items()):
        recordings = _group_recordings(group)
        arm_rows: dict[str, Any] = {}
        for arm in arms:
            mse = _specimen_metric(group, arm, "mse_m2")
            arm_rows[arm] = {
                "mse_m2": mse,
                "rmse_mm": 1000.0 * float(np.sqrt(mse)),
                "gaussian_nll": _specimen_metric(group, arm, "gaussian_nll"),
                "marginal_90_coverage": _specimen_metric(
                    group,
                    arm,
                    "marginal_90_coverage",
                ),
            }
        by_specimen[specimen] = {
            "material": str(group[0]["material"]),
            "size": str(group[0]["size"]),
            "recordings": len(recordings),
            "recording_horizon_rows": len(group),
            "arms": arm_rows,
        }

    aggregate: dict[str, Any] = {
        "unit": "material-size physical specimen",
        "specimens": len(specimens),
        "recordings": len(_group_recordings(rows)),
        "recording_horizon_rows": len(rows),
        "by_specimen": by_specimen,
        "arms": {},
    }
    for arm in arms:
        mse = np.asarray(
            [row["arms"][arm]["mse_m2"] for row in by_specimen.values()],
            dtype=np.float64,
        )
        aggregate["arms"][arm] = {
            "equal_specimen_mse_m2": float(np.mean(mse)),
            "equal_specimen_rmse_mm": 1000.0 * float(np.sqrt(np.mean(mse))),
            "equal_specimen_gaussian_nll": float(
                np.mean(
                    [
                        row["arms"][arm]["gaussian_nll"]
                        for row in by_specimen.values()
                    ]
                )
            ),
            "equal_specimen_marginal_90_coverage": float(
                np.mean(
                    [
                        row["arms"][arm]["marginal_90_coverage"]
                        for row in by_specimen.values()
                    ]
                )
            ),
        }
    baseline = aggregate["arms"]["constant_velocity"]["equal_specimen_mse_m2"]
    for arm in arms:
        value = aggregate["arms"][arm]["equal_specimen_mse_m2"]
        aggregate["arms"][arm]["relative_mse_improvement_vs_constant_velocity"] = (
            float((baseline - value) / baseline) if baseline > 0.0 else 0.0
        )
    return aggregate


def source_gate(rows: list[dict[str, Any]], request: dict[str, Any]) -> dict[str, Any]:
    """Apply the frozen operational denim gate with one vote per recording."""
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
        "inferential_boundary": (
            "Operational repeated-recording gate only; it is not a population "
            "confidence statement over physical cloth specimens."
        ),
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
    """Material-stratified bootstrap of material--size specimen contrasts."""
    specimens = _group_specimens(rows)
    by_material: dict[str, list[float]] = {}
    for group in specimens.values():
        recordings = _group_recordings(group)
        difference = float(
            np.mean(
                [
                    float(
                        np.mean(
                            [
                                row["arms"][arm_a]["mse_m2"]
                                - row["arms"][arm_b]["mse_m2"]
                                for row in recording_rows
                            ]
                        )
                    )
                    for recording_rows in recordings.values()
                ]
            )
        )
        material = str(group[0]["material"])
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
        "schema_version": 2,
        "source_selection": "cotton recordings; leave-one-recording-out",
        "source_gate": "denim complete recordings; registered horizons nested",
        "primary_target_unit": "material-size physical specimen",
        "target_nesting": "horizons within recordings within specimens",
        "target_bootstrap": "material-stratified resampling of specimens",
        "target_specimen_count_if_opened": 4,
        "population_generalization_claim_authorized": False,
        "correction_stage": "before first source-payload evaluation",
    }
    if result["target"] is not None:
        result["target"]["primary_statistical_unit"] = "material-size physical specimen"
        result["target"]["primary_specimen_aggregate"] = aggregate_specimens(
            result["target"]["rows"]
        )
    result["claim_boundary"].append(
        "Recordings and horizons are nested within material-size specimens; the "
        "held-out target contains four such specimens and supports no broad "
        "cloth-population inference."
    )
    unhashed = dict(result)
    unhashed.pop("result_id", None)
    result["result_id"] = canonical_sha256(unhashed)
    return result


def write_summary(result: dict[str, Any], path: Path) -> None:
    _ORIGINAL_WRITE_SUMMARY(result, path)
    if result["target"] is None:
        return
    primary = result["target"]["primary_specimen_aggregate"]
    lines = [
        "",
        "## Primary specimen-clustered target analysis",
        "",
        f"- Physical material-size specimens: `{primary['specimens']}`",
        "- Registered horizons are nested within recordings and specimens.",
    ]
    for arm in (
        "task_conditioned",
        "generic_information",
        "constant_velocity",
        "fixed_upper",
        "random_cost_matched",
        "dependence_destroyed",
    ):
        row = primary["arms"][arm]
        lines.append(
            f"- **{arm}:** equal-specimen RMSE="
            f"{row['equal_specimen_rmse_mm']:.6f} mm, "
            f"NLL={row['equal_specimen_gaussian_nll']:.6f}, "
            f"coverage={row['equal_specimen_marginal_90_coverage']:.3%}"
        )
    lines.extend(
        [
            "- The four-specimen target is a bounded diagnostic, not a cloth-population CI.",
            "",
        ]
    )
    with path.open("a", encoding="utf-8") as stream:
        stream.write("\n".join(lines))


_BASE.aggregate_rows = aggregate_rows
_BASE.source_gate = source_gate
_BASE._bootstrap_difference = _bootstrap_difference
_BASE.run_evaluation = run_evaluation
_BASE.write_summary = write_summary
validate_result = _BASE.validate_result


def main() -> int:
    return int(_BASE.main())


if __name__ == "__main__":
    raise SystemExit(main())
