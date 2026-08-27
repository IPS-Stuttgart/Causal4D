#!/usr/bin/env python3
"""Audit controlled latent-contact results with seed-clustered inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

METHODS = ("latent_contact", "nominal_physics")
WORLDS = ("matched_contact", "shifted_contact")
OBJECTS = ("cloth", "rope", "soft_block")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interventions", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-markdown", required=True, type=Path)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser


def _load(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        source = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    strings = {
        "object",
        "source_objects",
        "action",
        "world_condition",
        "setting",
        "method",
    }
    booleans = {"held_out_topology", "gross_failure"}
    integers = {"seed", "forecast_start_frame"}
    for item in source:
        row: dict[str, Any] = {}
        for key, value in item.items():
            if key in strings:
                row[key] = value
            elif key in booleans:
                row[key] = value.lower() == "true"
            elif key in integers:
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    return rows


def _selected(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["setting"] == "online_adaptation" and row["method"] in METHODS
    ]


def _validate(
    rows: list[dict[str, Any]],
) -> tuple[list[int], dict[tuple[Any, ...], dict[str, Any]]]:
    selected = _selected(rows)
    seeds = sorted({int(row["seed"]) for row in selected})
    objects = sorted({str(row["object"]) for row in selected})
    worlds = sorted({str(row["world_condition"]) for row in selected})
    if objects != sorted(OBJECTS):
        raise ValueError(f"unexpected object roster: {objects}")
    if worlds != sorted(WORLDS):
        raise ValueError(f"unexpected world roster: {worlds}")
    index = {
        (
            int(row["seed"]),
            str(row["object"]),
            str(row["world_condition"]),
            str(row["method"]),
        ): row
        for row in selected
    }
    expected = {
        (seed, object_name, world, method)
        for seed in seeds
        for object_name in OBJECTS
        for world in WORLDS
        for method in METHODS
    }
    if set(index) != expected or len(selected) != len(expected):
        raise ValueError("online comparison matrix is incomplete or duplicated")
    if not all(bool(row["held_out_topology"]) for row in selected):
        raise ValueError("all comparison rows must retain held-out-topology status")
    return seeds, index


def _sign_p(positive: int, negative: int) -> float:
    count = positive + negative
    if count == 0:
        return 1.0
    tail_count = min(positive, negative)
    tail = sum(math.comb(count, value) for value in range(tail_count + 1))
    return min(1.0, 2.0 * tail / (2**count))


def _holm(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: item[1])
    adjusted: dict[str, float] = {}
    running = 0.0
    count = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[name] = running
    return adjusted


def _bootstrap(
    latent: np.ndarray,
    nominal: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence: float,
) -> dict[str, Any]:
    if latent.shape != nominal.shape or latent.ndim != 1 or latent.size < 2:
        raise ValueError("paired seed vectors are malformed")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, latent.size, size=(replicates, latent.size))
    latent_boot = np.mean(latent[indices], axis=1)
    nominal_boot = np.mean(nominal[indices], axis=1)
    gain_boot = nominal_boot - latent_boot
    reduction_boot = 1.0 - latent_boot / nominal_boot
    alpha = 0.5 * (1.0 - confidence)
    gain_interval = np.quantile(
        gain_boot,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    reduction_interval = np.quantile(
        reduction_boot,
        [alpha, 1.0 - alpha],
        method="linear",
    )
    gain = nominal - latent
    positive = int(np.count_nonzero(gain > 0.0))
    negative = int(np.count_nonzero(gain < 0.0))
    return {
        "seed_count": int(latent.size),
        "latent_mean_rmse_m": float(np.mean(latent)),
        "nominal_mean_rmse_m": float(np.mean(nominal)),
        "mean_absolute_gain_m": float(np.mean(gain)),
        "absolute_gain_interval_m": [
            float(gain_interval[0]),
            float(gain_interval[1]),
        ],
        "relative_reduction": float(1.0 - np.mean(latent) / np.mean(nominal)),
        "relative_reduction_interval": [
            float(reduction_interval[0]),
            float(reduction_interval[1]),
        ],
        "positive_seed_count": positive,
        "negative_seed_count": negative,
        "tied_seed_count": int(latent.size - positive - negative),
        "exact_two_sided_sign_p": _sign_p(positive, negative),
        "worst_seed_gain_m": float(np.min(gain)),
        "median_seed_gain_m": float(np.median(gain)),
    }


def _comparison(
    index: dict[tuple[Any, ...], dict[str, Any]],
    seeds: list[int],
    world: str,
    object_name: str,
    *,
    replicates: int,
    bootstrap_seed: int,
    confidence: float,
) -> dict[str, Any]:
    nested_objects = OBJECTS if object_name == "ALL" else (object_name,)
    latent = np.asarray(
        [
            np.mean(
                [
                    index[(seed, item, world, "latent_contact")]["trajectory_rmse_m"]
                    for item in nested_objects
                ]
            )
            for seed in seeds
        ],
        dtype=float,
    )
    nominal = np.asarray(
        [
            np.mean(
                [
                    index[(seed, item, world, "nominal_physics")]["trajectory_rmse_m"]
                    for item in nested_objects
                ]
            )
            for seed in seeds
        ],
        dtype=float,
    )
    result = _bootstrap(
        latent,
        nominal,
        replicates=replicates,
        seed=bootstrap_seed,
        confidence=confidence,
    )
    case_pairs = [
        (
            float(index[(seed, item, world, "latent_contact")]["trajectory_rmse_m"]),
            float(index[(seed, item, world, "nominal_physics")]["trajectory_rmse_m"]),
        )
        for seed in seeds
        for item in nested_objects
    ]
    result.update(
        {
            "world_condition": world,
            "object": object_name,
            "case_count": len(case_pairs),
            "case_win_count": int(
                sum(
                    latent_value < nominal_value
                    for latent_value, nominal_value in case_pairs
                )
            ),
            "worst_case_gain_m": float(
                min(
                    nominal_value - latent_value
                    for latent_value, nominal_value in case_pairs
                )
            ),
            "worst_case_ratio": float(
                max(
                    latent_value / nominal_value
                    for latent_value, nominal_value in case_pairs
                )
            ),
        }
    )
    return result


def _adjusted_nll(nll: float, nees: float, scale: float) -> float:
    return float(nll + 0.5 * (math.log(scale) + nees / scale - nees))


def _calibration_rows(
    index: dict[tuple[Any, ...], dict[str, Any]],
    seeds: list[int],
    *,
    exclude_target_topology: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for target_seed in seeds:
        for target_object in OBJECTS:
            calibration = [
                float(index[(seed, source_object, world, "latent_contact")]["nees"])
                for seed in seeds
                if seed != target_seed
                for source_object in OBJECTS
                if not exclude_target_topology or source_object != target_object
                for world in WORLDS
                if exclude_target_topology or source_object == target_object
            ]
            scale = float(np.mean(calibration))
            for world in WORLDS:
                latent = index[(target_seed, target_object, world, "latent_contact")]
                nominal = index[(target_seed, target_object, world, "nominal_physics")]
                adjusted = _adjusted_nll(
                    float(latent["gaussian_nll"]),
                    float(latent["nees"]),
                    scale,
                )
                output.append(
                    {
                        "seed": target_seed,
                        "object": target_object,
                        "world_condition": world,
                        "variance_scale": scale,
                        "adjusted_nees": float(latent["nees"]) / scale,
                        "adjusted_gaussian_nll": adjusted,
                        "nominal_gaussian_nll": float(nominal["gaussian_nll"]),
                        "adjusted_nll_gain_over_nominal": (
                            float(nominal["gaussian_nll"]) - adjusted
                        ),
                        "interval_width_multiplier": math.sqrt(scale),
                    }
                )
    return output


def _calibration_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (str(row["world_condition"]), str(row["object"]))
        groups[key].append(row)
        groups[(str(row["world_condition"]), "ALL")].append(row)
    output: list[dict[str, Any]] = []
    for (world, object_name), selected in sorted(groups.items()):
        output.append(
            {
                "world_condition": world,
                "object": object_name,
                "case_count": len(selected),
                "mean_adjusted_nees": float(
                    np.mean([row["adjusted_nees"] for row in selected])
                ),
                "mean_adjusted_gaussian_nll": float(
                    np.mean([row["adjusted_gaussian_nll"] for row in selected])
                ),
                "mean_nominal_gaussian_nll": float(
                    np.mean([row["nominal_gaussian_nll"] for row in selected])
                ),
                "mean_adjusted_nll_gain_over_nominal": float(
                    np.mean([row["adjusted_nll_gain_over_nominal"] for row in selected])
                ),
                "mean_variance_scale": float(
                    np.mean([row["variance_scale"] for row in selected])
                ),
                "mean_interval_width_multiplier": float(
                    np.mean([row["interval_width_multiplier"] for row in selected])
                ),
                "case_win_count": int(
                    sum(row["adjusted_nll_gain_over_nominal"] > 0.0 for row in selected)
                ),
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    if not 0.0 < arguments.confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    seeds, index = _validate(_load(arguments.interventions))

    comparisons: list[dict[str, Any]] = []
    stream = 0
    for world in WORLDS:
        for object_name in ("ALL", *OBJECTS):
            comparisons.append(
                _comparison(
                    index,
                    seeds,
                    world,
                    object_name,
                    replicates=arguments.bootstrap_replicates,
                    bootstrap_seed=arguments.bootstrap_seed + stream,
                    confidence=arguments.confidence_level,
                )
            )
            stream += 1

    for world in WORLDS:
        family = {
            str(row["object"]): float(row["exact_two_sided_sign_p"])
            for row in comparisons
            if row["world_condition"] == world
        }
        adjusted = _holm(family)
        for row in comparisons:
            if row["world_condition"] == world:
                row["holm_adjusted_sign_p"] = adjusted[str(row["object"])]

    same_topology_rows = _calibration_rows(
        index,
        seeds,
        exclude_target_topology=False,
    )
    topology_excluded_rows = _calibration_rows(
        index,
        seeds,
        exclude_target_topology=True,
    )
    same_topology_summary = _calibration_summary(same_topology_rows)
    topology_excluded_summary = _calibration_summary(topology_excluded_rows)

    shifted = [
        row for row in comparisons if row["world_condition"] == "shifted_contact"
    ]
    robust_shifted = bool(
        all(row["absolute_gain_interval_m"][0] > 0.0 for row in shifted)
        and all(row["holm_adjusted_sign_p"] < 0.05 for row in shifted)
    )
    claim_boundary = (
        "A positive point decision establishes controlled held-out-topology "
        "improvement for the exact graph simulator, action, source split, "
        "online prefix, seeds, and comparison. It does not establish physical "
        "evidence, arbitrary topology or action generalization, calibrated "
        "unseen-topology uncertainty, joint intervention-effect calibration, "
        "or deployment safety. Same-topology variance recalibration is "
        "retrospective mechanism evidence and cannot rescue failure of the "
        "topology-excluded uncertainty stress test."
    )
    result = {
        "schema_version": 1,
        "analysis": "causal4d-controlled-latent-contact-robustness-v1",
        "source": str(arguments.interventions),
        "seed_count": len(seeds),
        "seeds": seeds,
        "objects": list(OBJECTS),
        "independent_unit": (
            "simulation seed; object topologies are nested within seed for the "
            "overall comparison"
        ),
        "bootstrap": {
            "replicates": arguments.bootstrap_replicates,
            "seed": arguments.bootstrap_seed,
            "confidence_level": arguments.confidence_level,
            "method": "percentile bootstrap over complete seed clusters",
        },
        "point_comparisons": comparisons,
        "robust_shifted_point_supported": robust_shifted,
        "same_topology_leave_one_seed_out_calibration": {
            "status": "retrospective cross-fitted mechanism evidence only",
            "summary": same_topology_summary,
            "rows": same_topology_rows,
        },
        "topology_excluded_calibration": {
            "status": "strict unseen-topology stress test",
            "summary": topology_excluded_summary,
            "rows": topology_excluded_rows,
        },
        "claim_boundary": claim_boundary,
    }

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_csv(arguments.output_csv, comparisons)

    shifted_all = next(
        row
        for row in comparisons
        if row["world_condition"] == "shifted_contact" and row["object"] == "ALL"
    )
    matched_all = next(
        row
        for row in comparisons
        if row["world_condition"] == "matched_contact" and row["object"] == "ALL"
    )
    same_shifted = next(
        row
        for row in same_topology_summary
        if row["world_condition"] == "shifted_contact" and row["object"] == "ALL"
    )
    excluded_shifted = next(
        row
        for row in topology_excluded_summary
        if row["world_condition"] == "shifted_contact" and row["object"] == "ALL"
    )
    decision = "SUPPORTED" if robust_shifted else "NOT SUPPORTED"
    markdown = f"""# Controlled latent-contact robustness audit

**Decision:** {decision} for the bounded shifted-contact point claim.

- Independent seeds: {len(seeds)}
- Nested topologies per seed: {len(OBJECTS)}
- Bootstrap replicates: {arguments.bootstrap_replicates}

## Point prediction

| World | Latent RMSE | Nominal RMSE | Relative reduction | 95% interval | Seed wins | Exact sign p |
|---|---:|---:|---:|---:|---:|---:|
| Shifted | {1000 * shifted_all["latent_mean_rmse_m"]:.3f} mm | {1000 * shifted_all["nominal_mean_rmse_m"]:.3f} mm | {100 * shifted_all["relative_reduction"]:.2f}% | [{100 * shifted_all["relative_reduction_interval"][0]:.2f}%, {100 * shifted_all["relative_reduction_interval"][1]:.2f}%] | {shifted_all["positive_seed_count"]}/{shifted_all["seed_count"]} | {shifted_all["exact_two_sided_sign_p"]:.3g} |
| Matched | {1000 * matched_all["latent_mean_rmse_m"]:.3f} mm | {1000 * matched_all["nominal_mean_rmse_m"]:.3f} mm | {100 * matched_all["relative_reduction"]:.2f}% | [{100 * matched_all["relative_reduction_interval"][0]:.2f}%, {100 * matched_all["relative_reduction_interval"][1]:.2f}%] | {matched_all["positive_seed_count"]}/{matched_all["seed_count"]} | {matched_all["exact_two_sided_sign_p"]:.3g} |

The overall inference resamples complete seeds. Object coordinates remain nested.

## Uncertainty boundary

Same-topology leave-one-seed-out scalar recalibration changes no mean. Its shifted-contact mean NLL gain over nominal is **{same_shifted["mean_adjusted_nll_gain_over_nominal"]:.3f}**, with a mean width multiplier of **{same_shifted["mean_interval_width_multiplier"]:.2f}x**. This is retrospective mechanism evidence.

The stricter topology-excluded correction gives a shifted-contact mean NLL gain over nominal of **{excluded_shifted["mean_adjusted_nll_gain_over_nominal"]:.3f}**. This is the relevant unseen-topology uncertainty stress test.

## Claim boundary

{claim_boundary}
"""
    arguments.output_markdown.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_markdown.write_text(markdown, encoding="utf-8")
    print(
        json.dumps(
            {
                "robust_shifted_point_supported": robust_shifted,
                "seed_count": len(seeds),
                "shifted_relative_reduction": shifted_all["relative_reduction"],
                "shifted_relative_interval": shifted_all["relative_reduction_interval"],
                "same_topology_shifted_nll_gain": same_shifted[
                    "mean_adjusted_nll_gain_over_nominal"
                ],
                "topology_excluded_shifted_nll_gain": excluded_shifted[
                    "mean_adjusted_nll_gain_over_nominal"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
