#!/usr/bin/env python3
"""Extract the frozen posterior-averaging versus MAP source comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from pathlib import Path
from typing import Any

import numpy as np

METRICS = (
    "force_rmse_n",
    "force_mae_n",
    "force_gaussian_nll",
    "contact_brier",
    "contact_log_score",
    "peak_force_error_n",
    "mean_force_error_n",
)
PRIMARY_METRICS = (
    "force_rmse_n",
    "force_gaussian_nll",
    "contact_brier",
    "contact_log_score",
)
BMA_METHOD = "posterior_mean"
MAP_METHOD = "posterior_map"


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def paired_comparisons(rows: list[dict[str, Any]]) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for metric in METRICS:
        differences = np.asarray(
            [
                row["metrics"][MAP_METHOD][metric]
                - row["metrics"][BMA_METHOD][metric]
                for row in rows
            ],
            dtype=np.float64,
        )
        tolerance = 1e-12
        wins = int(np.sum(differences > tolerance))
        losses = int(np.sum(differences < -tolerance))
        non_ties = wins + losses
        if non_ties:
            sign_p = sum(
                math.comb(non_ties, value)
                for value in range(wins, non_ties + 1)
            ) / (2**non_ties)
        else:
            sign_p = 1.0
        seed_bytes = hashlib.sha256(metric.encode("utf-8")).digest()[:8]
        seed = (20260903 + int.from_bytes(seed_bytes, "big")) % (2**63 - 1)
        generator = np.random.default_rng(seed)
        indices = generator.integers(
            0,
            len(differences),
            size=(20000, len(differences)),
        )
        bootstrap = np.mean(differences[indices], axis=1)
        comparisons[metric] = {
            "positive_difference_means_bma_is_better": True,
            "take_differences_map_minus_bma": {
                row["take_id"]: float(value)
                for row, value in zip(rows, differences, strict=True)
            },
            "mean_difference": float(np.mean(differences)),
            "median_difference": float(np.median(differences)),
            "bootstrap_lower95": float(np.quantile(bootstrap, 0.025)),
            "bootstrap_upper95": float(np.quantile(bootstrap, 0.975)),
            "wins": wins,
            "ties": len(differences) - wins - losses,
            "losses": losses,
            "one_sided_sign_pvalue": float(sign_p),
        }
    return comparisons


def posterior_seals(output: Path) -> list[dict[str, Any]]:
    seals = []
    for path in sorted(output.glob("prediction_seal_*.json")):
        seal = read_json(path)
        posterior = seal["metadata"]["posterior"]
        seals.append(
            {
                "target_take_id": seal["target_take_id"],
                "candidate_count": int(posterior["candidate_count"]),
                "effective_candidate_count": float(
                    posterior["effective_candidate_count"]
                ),
                "posterior_entropy": float(posterior["posterior_entropy"]),
                "map_candidate": posterior["map_candidate"],
                "target_force_suffix_used": bool(seal["target_force_suffix_used"]),
                "prediction_seal_sha256": seal["seal_sha256"],
            }
        )
    if len(seals) != 5:
        raise ValueError("expected five prediction seals")
    if any(item["target_force_suffix_used"] for item in seals):
        raise ValueError("target suffix leaked into inference")
    return seals


def build_summary(output: Path) -> dict[str, Any]:
    result = read_json(output / "result.json")
    rows = result["per_take_results"]
    if not isinstance(rows, list) or len(rows) != 5:
        raise ValueError("expected exactly five complete-take units")
    comparisons = paired_comparisons(rows)
    selected_aggregate = {
        method: {
            metric: result["aggregate_equal_take_results"][method][metric]
            for metric in METRICS
        }
        for method in (BMA_METHOD, MAP_METHOD)
    }
    return {
        "artifact_kind": "PublicPokeFlexBayesianValueSummary",
        "schema_version": 1,
        "source_result_sha256": result["result_sha256"],
        "github_repository": os.environ.get("GITHUB_REPOSITORY"),
        "github_sha": os.environ.get("GITHUB_SHA"),
        "github_run_id": os.environ.get("GITHUB_RUN_ID"),
        "github_run_attempt": os.environ.get("GITHUB_RUN_ATTEMPT"),
        "request_id": os.environ.get("REQUEST_ID"),
        "superseded_run_id": 33749709370,
        "superseded_failure": "no-compatible-preinstalled-numpy-before-data-access",
        "runner_name": os.environ.get("RUNNER_NAME"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "dataset_root": os.environ.get("DATASET_ROOT"),
        "independent_unit": "complete development take",
        "unit_count": len(rows),
        "factual_prefix_frames": result["policy_config"]["prefix_frame_count"],
        "untouched_suffix_frames": result["policy_config"][
            "forecast_horizon_frames"
        ],
        "contrast": {
            "bma_method": BMA_METHOD,
            "map_method": MAP_METHOD,
            "same_component_bank": True,
            "same_prefix_evidence": True,
            "same_future_tool_kinematics": True,
            "only_posterior_averaging_vs_map_selection_changes": True,
        },
        "aggregate_equal_take_results": selected_aggregate,
        "paired_map_minus_bma_comparisons": comparisons,
        "posterior_ambiguity_by_take": posterior_seals(output),
        "primary_mean_differences_all_favor_bma": all(
            comparisons[metric]["mean_difference"] > 0.0
            for metric in PRIMARY_METRICS
        ),
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "target_force_suffix_passed_to_inference": False,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        },
        "claim_boundary": [
            (
                "This is a five-development-take mechanism test, not an "
                "independent held-out target result."
            ),
            (
                "The force Gaussian NLL scores the moment-matched predictive "
                "mean and variance, not an exact mixture log density."
            ),
            (
                "A favorable result supports posterior averaging over the fixed "
                "template-gain-delay bank relative to MAP selection on this "
                "protocol."
            ),
        ],
    }


def render_markdown(summary: dict[str, Any]) -> str:
    comparisons = summary["paired_map_minus_bma_comparisons"]
    lines = [
        "# PokeFlex Bayesian-value source result",
        "",
        (
            "Positive paired differences below mean that posterior averaging "
            "outperformed MAP selection."
        ),
        "",
        "| Metric | MAP - BMA mean | 95% take-bootstrap CI | Wins / 5 |",
        "|---|---:|---:|---:|",
    ]
    for metric in PRIMARY_METRICS:
        item = comparisons[metric]
        lines.append(
            f"| `{metric}` | {item['mean_difference']:.8g} | "
            f"[{item['bootstrap_lower95']:.8g}, "
            f"{item['bootstrap_upper95']:.8g}] | {item['wins']} |"
        )
    lines.extend(
        [
            "",
            (
                "This result uses five development takes. Calibration T5 and "
                "target T2 remained unopened."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    summary = build_summary(output)
    (output / "bayesian_value_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output / "bayesian_value_summary.md").write_text(
        render_markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
