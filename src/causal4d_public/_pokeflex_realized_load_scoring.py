"""Scoring and source-gate execution for PokeFlex realized-load forecasts."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d_public._pokeflex_realized_load_common import (
    BoolArray,
    FloatArray,
    POKEFLEX_REALIZED_LOAD_ARTIFACT_SCHEMA_VERSION,
    PokeFlexRealizedLoadSourceConfig,
    _array_digest,
    _canonical_bytes,
    _require,
    validate_source_qa_binding,
)
from causal4d_public._pokeflex_realized_load_model import (
    ForecastBundle,
    RealizedLoadTake,
    TargetKinematicConditioning,
    build_forecast_bundle,
    load_realized_load_take,
)


def _method_metrics(
    truth: FloatArray,
    contact_truth: BoolArray,
    mean: FloatArray,
    variance: FloatArray,
    contact_probability: FloatArray,
) -> dict[str, float]:
    safe_variance = np.maximum(variance, 1e-12)
    error = truth - mean
    half_width = 1.6448536269514722 * np.sqrt(safe_variance)
    probability = np.clip(contact_probability, 1e-6, 1.0 - 1e-6)
    contact_float = contact_truth.astype(np.float64)
    log_score = -np.mean(
        contact_float * np.log(probability)
        + (1.0 - contact_float) * np.log(1.0 - probability)
    )
    return {
        "force_rmse_n": float(np.sqrt(np.mean(error**2))),
        "force_mae_n": float(np.mean(np.abs(error))),
        "force_gaussian_nll": float(
            np.mean(
                0.5
                * (
                    np.log(2.0 * math.pi * safe_variance)
                    + error**2 / safe_variance
                )
            )
        ),
        "force_90pct_coverage": float(np.mean(np.abs(error) <= half_width)),
        "mean_90pct_interval_width_n": float(np.mean(2.0 * half_width)),
        "contact_brier": float(np.mean((probability - contact_float) ** 2)),
        "contact_log_score": float(log_score),
        "peak_force_error_n": float(abs(np.max(mean) - np.max(truth))),
        "mean_force_error_n": float(abs(np.mean(mean) - np.mean(truth))),
    }


def _bootstrap_mean_ci(
    values: Sequence[float],
    config: PokeFlexRealizedLoadSourceConfig,
    key: str,
) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(array.ndim == 1 and array.size >= 2, "bootstrap input is invalid")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    seed = (config.bootstrap_seed + int.from_bytes(digest[:8], "big")) % (2**63 - 1)
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(array),
        size=(config.bootstrap_replicates, len(array)),
    )
    means = np.mean(array[indices], axis=1)
    return {
        "mean": float(np.mean(array)),
        "lower95": float(np.quantile(means, 0.025)),
        "upper95": float(np.quantile(means, 0.975)),
    }


def _one_sided_sign_pvalue(wins: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    numerator = sum(
        math.comb(non_ties, value) for value in range(wins, non_ties + 1)
    )
    return float(numerator / (2**non_ties))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_prediction_bundle(
    output_dir: Path,
    target: RealizedLoadTake,
    bundle: ForecastBundle,
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, Any]:
    arrays: dict[str, FloatArray] = {}
    for method in sorted(bundle.means):
        arrays[f"{method}__mean"] = np.asarray(
            bundle.means[method], dtype=np.float64
        )
        arrays[f"{method}__variance"] = np.asarray(
            bundle.variances[method], dtype=np.float64
        )
        arrays[f"{method}__contact_probability"] = np.asarray(
            bundle.contact_probabilities[method],
            dtype=np.float64,
        )
    prediction_sha256 = _array_digest(arrays)
    npz_path = output_dir / f"predictions_{target.take_id}.npz"
    np.savez_compressed(npz_path, **arrays)
    prefix = config.prefix_frame_count
    receipt = {
        "artifact_kind": "PublicPokeFlexRealizedLoadPredictionSeal",
        "schema_version": 1,
        "target_take_id": target.take_id,
        "source_take_ids": list(bundle.metadata["source_take_ids"]),
        "target_robot_sha256": target.robot_sha256,
        "contact_onset_frame_id": int(target.frame_ids[target.onset_index]),
        "prefix_frame_ids": target.frame_ids[
            target.onset_index : target.onset_index + prefix
        ]
        .astype(int)
        .tolist(),
        "forecast_frame_ids": target.frame_ids[
            target.onset_index
            + prefix : target.onset_index
            + prefix
            + config.forecast_horizon_frames
        ]
        .astype(int)
        .tolist(),
        "prediction_sha256": prediction_sha256,
        "prediction_file": npz_path.name,
        "target_force_suffix_used": False,
        "known_future_tool_kinematics_used": True,
        "metadata": bundle.metadata,
    }
    receipt["seal_sha256"] = hashlib.sha256(_canonical_bytes(receipt)).hexdigest()
    _write_json(output_dir / f"prediction_seal_{target.take_id}.json", receipt)
    return receipt


def realized_load_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def run_pokeflex_realized_load_source_gate(
    dataset_root: str | Path,
    source_qa: Mapping[str, Any],
    output_dir: str | Path,
    config: PokeFlexRealizedLoadSourceConfig,
) -> dict[str, Any]:
    """Run the frozen leave-one-development-take-out source gate."""

    binding = validate_source_qa_binding(source_qa, config)
    root = Path(dataset_root).resolve()
    _require(root.is_dir(), f"missing PokeFlex root: {root}")
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    takes = {
        take_id: load_realized_load_take(root, take_id, config)
        for take_id in config.expected_development_take_ids
    }
    _require(
        set(takes).isdisjoint(config.forbidden_take_ids),
        "forbidden take entered the loaded roster",
    )

    bundles: dict[str, ForecastBundle] = {}
    seals: dict[str, dict[str, Any]] = {}
    for target_id in config.expected_development_take_ids:
        target = takes[target_id]
        sources = [
            takes[source_id]
            for source_id in config.expected_development_take_ids
            if source_id != target_id
        ]
        target_prefix = target.window_force_n[: config.prefix_frame_count].copy()
        conditioning = TargetKinematicConditioning(
            take_id=target.take_id,
            window_phase=target.window_phase.copy(),
            window_speed_m_per_frame=target.window_speed_m_per_frame.copy(),
        )
        bundle = build_forecast_bundle(
            target_prefix,
            conditioning,
            sources,
            config,
        )
        bundles[target_id] = bundle
        seals[target_id] = _write_prediction_bundle(output, target, bundle, config)

    rows: list[dict[str, Any]] = []
    prefix = config.prefix_frame_count
    suffix = slice(prefix, prefix + config.forecast_horizon_frames)
    for target_id in config.expected_development_take_ids:
        target = takes[target_id]
        bundle = bundles[target_id]
        truth = target.window_force_n[suffix]
        contact_truth = truth > config.contact_threshold_n
        metrics = {
            method: _method_metrics(
                truth,
                contact_truth,
                np.asarray(bundle.means[method])[suffix],
                np.asarray(bundle.variances[method])[suffix],
                np.asarray(bundle.contact_probabilities[method])[suffix],
            )
            for method in sorted(bundle.means)
        }
        rows.append(
            {
                "take_id": target_id,
                "robot_sha256": target.robot_sha256,
                "contact_onset_index": target.onset_index,
                "contact_onset_frame_id": int(
                    target.frame_ids[target.onset_index]
                ),
                "prediction_seal_sha256": seals[target_id]["seal_sha256"],
                "metrics": metrics,
            }
        )

    methods = sorted(rows[0]["metrics"])
    aggregate = {
        method: {
            metric: {
                "mean": float(
                    np.mean([row["metrics"][method][metric] for row in rows])
                ),
                "median": float(
                    np.median([row["metrics"][method][metric] for row in rows])
                ),
            }
            for metric in rows[0]["metrics"][method]
        }
        for method in methods
    }

    comparisons: dict[str, Any] = {}
    for comparator in (
        "persistence",
        "linear_extrapolation",
        "mean_prefix_offset",
        "kinematic_ridge",
        "posterior_map",
        "dependence_destroyed",
    ):
        differences = [
            row["metrics"][comparator]["force_rmse_n"]
            - row["metrics"]["posterior_mean"]["force_rmse_n"]
            for row in rows
        ]
        wins = sum(value > 1e-12 for value in differences)
        losses = sum(value < -1e-12 for value in differences)
        comparisons[f"posterior_mean_vs_{comparator}"] = {
            "positive_difference_means_posterior_is_better": True,
            "take_differences_n": {
                row["take_id"]: float(value)
                for row, value in zip(rows, differences, strict=True)
            },
            "wins": wins,
            "ties": len(rows) - wins - losses,
            "losses": losses,
            "win_fraction": wins / len(rows),
            "one_sided_sign_pvalue": _one_sided_sign_pvalue(
                wins, wins + losses
            ),
            "mean_difference_bootstrap": _bootstrap_mean_ci(
                differences,
                config,
                f"posterior_mean_vs_{comparator}",
            ),
        }

    proposed_rmse = aggregate["posterior_mean"]["force_rmse_n"]["mean"]
    persistence_rmse = aggregate["persistence"]["force_rmse_n"]["mean"]
    destroyed_rmse = aggregate["dependence_destroyed"]["force_rmse_n"]["mean"]
    ridge_rmse = aggregate["kinematic_ridge"]["force_rmse_n"]["mean"]
    persistence_win_fraction = comparisons["posterior_mean_vs_persistence"][
        "win_fraction"
    ]
    worst_ratio = max(
        row["metrics"]["posterior_mean"]["force_rmse_n"]
        / max(row["metrics"]["persistence"]["force_rmse_n"], 1e-12)
        for row in rows
    )
    criteria = {
        "five_locked_development_takes_evaluated": len(rows) == 5,
        "all_predictions_sealed_before_scoring": all(
            bool(row["prediction_seal_sha256"]) for row in rows
        ),
        "mean_rmse_improvement_vs_persistence": (
            proposed_rmse
            <= (
                1.0
                - config.minimum_mean_rmse_improvement_fraction_vs_persistence
            )
            * persistence_rmse
        ),
        "take_win_fraction_vs_persistence": (
            persistence_win_fraction
            >= config.minimum_take_win_fraction_vs_persistence
        ),
        "mean_rmse_improvement_vs_dependence_destroyed": (
            proposed_rmse
            <= (
                1.0
                - config.minimum_mean_rmse_improvement_fraction_vs_dependence_control
            )
            * destroyed_rmse
        ),
        "worst_take_ratio_vs_persistence": (
            worst_ratio <= config.maximum_worst_take_rmse_ratio_vs_persistence
        ),
        "mean_rmse_noninferior_to_kinematic_ridge": (
            proposed_rmse
            <= config.maximum_mean_rmse_ratio_vs_kinematic_ridge * ridge_rmse
        ),
        "forbidden_takes_remained_outside_loaded_roster": set(takes).isdisjoint(
            config.forbidden_take_ids
        ),
    }
    admitted = all(criteria.values())
    result: dict[str, Any] = {
        "artifact_kind": "PublicPokeFlexRealizedLoadSourceGate",
        "schema_version": POKEFLEX_REALIZED_LOAD_ARTIFACT_SCHEMA_VERSION,
        "policy_id": config.policy_id,
        "policy_config": config.as_dict(),
        "source_qa_binding": binding,
        "dataset": {
            "name": "PokeFlex",
            "object_id": config.expected_object_id,
            "opened_take_ids": list(config.expected_development_take_ids),
            "forbidden_unopened_take_ids": list(config.forbidden_take_ids),
            "independent_unit": "complete development take",
            "robot_records_only": True,
            "mesh_payloads_opened": False,
        },
        "design": {
            "factual_prefix": (
                f"first {config.prefix_frame_count} frames from the source-locked "
                "persistent force-threshold contact onset"
            ),
            "forecast_horizon_frames": config.forecast_horizon_frames,
            "future_conditioning": "released measured tool trajectory only",
            "latent_intervention_bank": "source take, gain, and response delay",
            "dependence_control": (
                "cyclic source-prefix/source-suffix reassignment preserving both "
                "marginals"
            ),
            "target_force_suffix_passed_to_inference": False,
        },
        "per_take_results": rows,
        "aggregate_equal_take_results": aggregate,
        "comparisons": comparisons,
        "source_gate_criteria": criteria,
        "source_backend_admitted": admitted,
        "decision": "source-positive"
        if admitted
        else "source-negative-or-bounded",
        "next_stage_authorization": {
            "automatic_calibration_open": False,
            "automatic_target_open": False,
            "separate_reviewed_protocol_required": True,
        },
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "development_robot_records_read": True,
            "development_meshes_read": False,
            "calibration_take_data_read": False,
            "target_take_data_read": False,
            "forbidden_take_discovery_required": False,
            "predictions_sealed_before_suffix_scoring": True,
        },
        "claim_boundary": [
            "This is a development-only source gate, not a held-out target result.",
            "It tests realized-load and contact-state forecasting from public measured wrench records.",
            "The released future measured tool trajectory is conditioning evidence, not a commanded control stream.",
            "No material-point identity, mesh forecast, calibrated population uncertainty, individual counterfactual, control, or safety claim is authorized.",
            "A positive source gate cannot open calibration or target data without a separate reviewed protocol.",
        ],
    }
    result["result_sha256"] = realized_load_artifact_sha256(result)
    _write_json(output / "result.json", result)
    (output / "summary.md").write_text(render_summary(result), encoding="utf-8")
    return result


def validate_realized_load_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    _require(
        payload.get("artifact_kind") == "PublicPokeFlexRealizedLoadSourceGate",
        "unexpected realized-load artifact kind",
    )
    _require(
        payload.get("schema_version")
        == POKEFLEX_REALIZED_LOAD_ARTIFACT_SCHEMA_VERSION,
        "unsupported realized-load artifact schema",
    )
    _require(
        payload.get("result_sha256") == realized_load_artifact_sha256(payload),
        "realized-load artifact checksum mismatch",
    )
    boundary = payload.get("information_boundary", {})
    _require(boundary.get("public_data_only") is True, "public-data boundary changed")
    _require(
        boundary.get("calibration_take_data_read") is False,
        "calibration was opened",
    )
    _require(boundary.get("target_take_data_read") is False, "target was opened")
    _require(
        boundary.get("development_meshes_read") is False,
        "mesh payload was opened",
    )
    _require(
        payload.get("next_stage_authorization", {}).get("automatic_target_open")
        is False,
        "artifact automatically authorizes target access",
    )
    return {
        "passed": True,
        "result_sha256": payload["result_sha256"],
        "source_backend_admitted": bool(payload["source_backend_admitted"]),
        "decision": payload["decision"],
    }


def render_summary(result: Mapping[str, Any]) -> str:
    lines = [
        "# PokeFlex realized-load source gate",
        "",
        f"Decision: **{result['decision']}**",
        "",
        "This source-only study uses the five metadata-locked development takes. "
        "The calibration take T5 and target take T2 remain unopened.",
        "",
        "## Equal-take force forecast",
        "",
        "| Method | RMSE [N] | Gaussian NLL | 90% coverage | Contact Brier |",
        "|---|---:|---:|---:|---:|",
    ]
    aggregate = result["aggregate_equal_take_results"]
    for method in sorted(aggregate):
        row = aggregate[method]
        lines.append(
            f"| `{method}` | {row['force_rmse_n']['mean']:.4f} | "
            f"{row['force_gaussian_nll']['mean']:.4f} | "
            f"{row['force_90pct_coverage']['mean']:.3f} | "
            f"{row['contact_brier']['mean']:.4f} |"
        )
    lines.extend(["", "## Source-gate criteria", ""])
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in result["source_gate_criteria"].items()
    )
    lines.extend(["", "## Boundary", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    return "\n".join(lines) + "\n"
