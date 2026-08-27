#!/usr/bin/env python3
"""Evaluate a physical-variance floor for latent-contact point predictions.

The successful latent-contact mean is retained exactly.  Coordinate-wise
predictive variance is replaced by the maximum of the latent-contact and
unchanged nominal-physics variances.  The rule is target-independent and uses
only two predictions already available before the held-out suffix is scored.
"""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

import causal4d.contact_evaluation as contact_evaluation
from causal4d.baselines import PredictiveDistribution
from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_inference import LatentContactConfig

ENVELOPE_METHOD = "latent_contact_physical_envelope"
POINT_METRICS = (
    "trajectory_rmse_m",
    "relative_intervention_rmse",
    "ade_m",
    "fde_m",
    "early_rmse_m",
    "middle_rmse_m",
    "late_rmse_m",
    "direction_error_deg",
)


def _parse_seeds(value: str) -> list[int]:
    try:
        if ":" in value:
            values = [int(item) for item in value.split(":")]
            if len(values) not in {2, 3}:
                raise ValueError
            seeds = list(range(*values))
        else:
            seeds = [int(item) for item in value.split(",") if item]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "seeds must be comma-separated or start:stop[:step]"
        ) from error
    if not seeds or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("seeds must be nonempty and unique")
    return seeds


def physical_uncertainty_envelope(
    latent: PredictiveDistribution,
    nominal: PredictiveDistribution,
) -> PredictiveDistribution:
    """Retain the latent mean and impose both variances as lower bounds."""

    if latent.mean.shape != nominal.mean.shape:
        raise ValueError("latent and nominal prediction shapes differ")
    variance = np.maximum(latent.variance, nominal.variance)
    return PredictiveDistribution(
        method=ENVELOPE_METHOD,
        mean=latent.mean,
        variance=variance,
    )


def _run_with_envelope(
    *,
    seeds: Sequence[int],
    benchmark_config: CounterfactualBenchmarkConfig,
    contact_config: LatentContactConfig,
) -> dict[str, Any]:
    original_append = contact_evaluation._append_intervention_row
    cache: dict[tuple[object, ...], dict[str, PredictiveDistribution]] = {}
    emitted: set[tuple[object, ...]] = set()

    def append_with_envelope(
        rows: list[dict[str, Any]],
        *,
        seed: int,
        target: Any,
        episode: Any,
        setting: str,
        prediction: PredictiveDistribution,
        start_frame: int,
        source_objects: tuple[str, ...],
        calibration: Any,
        contact_config: LatentContactConfig,
        benchmark_config: CounterfactualBenchmarkConfig,
    ) -> None:
        original_append(
            rows,
            seed=seed,
            target=target,
            episode=episode,
            setting=setting,
            prediction=prediction,
            start_frame=start_frame,
            source_objects=source_objects,
            calibration=calibration,
            contact_config=contact_config,
            benchmark_config=benchmark_config,
        )
        if prediction.method not in {"latent_contact", "nominal_physics"}:
            return
        key = (
            id(rows),
            seed,
            target.protocol.graph_object.name,
            episode.episode_id,
            setting,
            start_frame,
        )
        pair = cache.setdefault(key, {})
        pair[prediction.method] = prediction
        if key in emitted or set(pair) != {"latent_contact", "nominal_physics"}:
            return
        emitted.add(key)
        envelope = physical_uncertainty_envelope(
            pair["latent_contact"],
            pair["nominal_physics"],
        )
        original_append(
            rows,
            seed=seed,
            target=target,
            episode=episode,
            setting=setting,
            prediction=envelope,
            start_frame=start_frame,
            source_objects=source_objects,
            calibration=calibration,
            contact_config=contact_config,
            benchmark_config=benchmark_config,
        )

    contact_evaluation._append_intervention_row = append_with_envelope
    try:
        result = contact_evaluation.run_latent_contact_benchmark(
            seeds=seeds,
            benchmark_config=benchmark_config,
            contact_config=contact_config,
        )
    finally:
        contact_evaluation._append_intervention_row = original_append

    result["benchmark"] = "causal4d-latent-contact-physical-envelope-v1"
    controls = result["protocol"]["contact_model"]["controls"]
    controls[ENVELOPE_METHOD] = (
        "retain the latent-contact predictive mean exactly and use the "
        "coordinate-wise maximum of latent-contact and nominal-physics "
        "predictive variance; no target suffix or target outcome selects the rule"
    )
    result["protocol"]["physical_uncertainty_envelope"] = {
        "method": ENVELOPE_METHOD,
        "point_mean": "byte-identical latent_contact mean",
        "variance": "maximum(latent_contact variance, nominal_physics variance)",
        "interval": "Gaussian interval derived from the envelope variance",
        "target_outcomes_used_for_design": False,
        "target_suffix_used_for_prediction": False,
        "selection_effect": "none",
        "claim_boundary": (
            "A positive result establishes proper-score value only for the exact "
            "controlled held-out-topology benchmark and fresh seed block. It "
            "does not establish physical evidence, arbitrary-topology transfer, "
            "joint intervention-effect calibration, or deployment safety."
        ),
    }
    return result


def _index(rows: list[dict[str, Any]]) -> tuple[list[int], dict[tuple[Any, ...], dict[str, Any]]]:
    selected = [
        row
        for row in rows
        if row["setting"] == "online_adaptation"
        and row["method"]
        in {"latent_contact", "nominal_physics", ENVELOPE_METHOD}
    ]
    seeds = sorted({int(row["seed"]) for row in selected})
    objects = sorted({str(row["object"]) for row in selected})
    worlds = sorted({str(row["world_condition"]) for row in selected})
    methods = ("latent_contact", "nominal_physics", ENVELOPE_METHOD)
    expected = {
        (seed, object_name, world, method)
        for seed in seeds
        for object_name in objects
        for world in worlds
        for method in methods
    }
    index = {
        (
            int(row["seed"]),
            str(row["object"]),
            str(row["world_condition"]),
            str(row["method"]),
        ): row
        for row in selected
    }
    if set(index) != expected or len(selected) != len(expected):
        raise ValueError("envelope comparison matrix is incomplete or duplicated")
    return seeds, index


def _bootstrap_gain(
    candidate: np.ndarray,
    reference: np.ndarray,
    *,
    replicates: int,
    seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    if candidate.shape != reference.shape or candidate.ndim != 1:
        raise ValueError("bootstrap vectors differ")
    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        candidate.size,
        size=(replicates, candidate.size),
    )
    gains = np.mean(reference[indices] - candidate[indices], axis=1)
    alpha = 0.5 * (1.0 - confidence_level)
    interval = np.quantile(gains, [alpha, 1.0 - alpha], method="linear")
    observed = reference - candidate
    return {
        "candidate_mean": float(np.mean(candidate)),
        "reference_mean": float(np.mean(reference)),
        "mean_gain": float(np.mean(observed)),
        "gain_interval": [float(interval[0]), float(interval[1])],
        "positive_seed_count": int(np.count_nonzero(observed > 0.0)),
        "negative_seed_count": int(np.count_nonzero(observed < 0.0)),
        "worst_seed_gain": float(np.min(observed)),
    }


def _seed_vectors(
    index: dict[tuple[Any, ...], dict[str, Any]],
    seeds: list[int],
    *,
    world: str,
    method: str,
    metric: str,
) -> np.ndarray:
    objects = sorted(
        {
            key[1]
            for key in index
            if key[2] == world and key[3] == method
        }
    )
    return np.asarray(
        [
            np.mean(
                [float(index[(seed, object_name, world, method)][metric]) for object_name in objects]
            )
            for seed in seeds
        ],
        dtype=float,
    )


def _analyse(
    result: dict[str, Any],
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    seeds, index = _index(result["interventions"])
    worlds = sorted({key[2] for key in index})
    comparisons: list[dict[str, Any]] = []
    point_identity_max_abs = 0.0
    for key, envelope in index.items():
        if key[3] != ENVELOPE_METHOD:
            continue
        latent = index[(key[0], key[1], key[2], "latent_contact")]
        for metric in POINT_METRICS:
            point_identity_max_abs = max(
                point_identity_max_abs,
                abs(float(envelope[metric]) - float(latent[metric])),
            )

    stream = 0
    for world in worlds:
        envelope_nll = _seed_vectors(
            index,
            seeds,
            world=world,
            method=ENVELOPE_METHOD,
            metric="gaussian_nll",
        )
        for reference_method in ("nominal_physics", "latent_contact"):
            reference_nll = _seed_vectors(
                index,
                seeds,
                world=world,
                method=reference_method,
                metric="gaussian_nll",
            )
            comparison = _bootstrap_gain(
                envelope_nll,
                reference_nll,
                replicates=bootstrap_replicates,
                seed=bootstrap_seed + stream,
                confidence_level=confidence_level,
            )
            comparison.update(
                {
                    "world_condition": world,
                    "metric": "gaussian_nll",
                    "candidate_method": ENVELOPE_METHOD,
                    "reference_method": reference_method,
                }
            )
            comparisons.append(comparison)
            stream += 1

    summaries: list[dict[str, Any]] = []
    for world in worlds:
        for method in ("latent_contact", "nominal_physics", ENVELOPE_METHOD):
            selected = [
                row
                for key, row in index.items()
                if key[2] == world and key[3] == method
            ]
            summaries.append(
                {
                    "world_condition": world,
                    "method": method,
                    "case_count": len(selected),
                    "mean_rmse_m": float(
                        np.mean([float(row["trajectory_rmse_m"]) for row in selected])
                    ),
                    "mean_gaussian_nll": float(
                        np.mean([float(row["gaussian_nll"]) for row in selected])
                    ),
                    "mean_nees": float(
                        np.mean([float(row["nees"]) for row in selected])
                    ),
                    "mean_coverage": float(
                        np.mean([float(row["coverage"]) for row in selected])
                    ),
                    "mean_coverage_error": float(
                        np.mean([float(row["coverage_error"]) for row in selected])
                    ),
                    "mean_interval_width_m": float(
                        np.mean(
                            [float(row["mean_interval_width_m"]) for row in selected]
                        )
                    ),
                }
            )

    shifted = [
        comparison
        for comparison in comparisons
        if comparison["world_condition"] == "shifted_contact"
    ]
    shifted_nll_supported = bool(
        all(comparison["gain_interval"][0] > 0.0 for comparison in shifted)
    )
    return {
        "schema_version": 1,
        "analysis": "causal4d-latent-contact-physical-envelope-v1",
        "fresh_seed_block": seeds,
        "seed_count": len(seeds),
        "independent_unit": (
            "simulation seed; three held-out graph topologies are nested within seed"
        ),
        "point_mean_identity_max_abs_metric_difference": point_identity_max_abs,
        "point_mean_identity_passed": point_identity_max_abs <= 1e-15,
        "comparisons": comparisons,
        "summaries": summaries,
        "shifted_nll_supported_against_both_references": shifted_nll_supported,
        "original_success_gates": result["success_gates"],
        "claim_boundary": result["protocol"]["physical_uncertainty_envelope"][
            "claim_boundary"
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seeds", type=_parse_seeds, default=_parse_seeds("64:128"))
    parser.add_argument("--frames", type=int, default=56)
    parser.add_argument("--training-repeats", type=int, default=2)
    parser.add_argument("--parameter-grid-count", type=int, default=5)
    parser.add_argument("--contact-parameter-particles", type=int, default=12)
    parser.add_argument("--observation-fraction", type=float, default=0.20)
    parser.add_argument("--observation-noise-mm", type=float, default=1.5)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260828)
    parser.add_argument("--confidence-level", type=float, default=0.95)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    if arguments.bootstrap_replicates < 1_000:
        raise ValueError("bootstrap_replicates must be at least 1000")
    benchmark_config = CounterfactualBenchmarkConfig(
        frame_count=arguments.frames,
        training_repeats=arguments.training_repeats,
        parameter_grid_count=arguments.parameter_grid_count,
        observation_noise_std_m=arguments.observation_noise_mm / 1000.0,
    )
    contact_config = LatentContactConfig(
        parameter_particle_count=arguments.contact_parameter_particles,
        observation_fraction=arguments.observation_fraction,
        observation_noise_std_m=arguments.observation_noise_mm / 1000.0,
        confidence_level=benchmark_config.confidence_level,
    )
    result = _run_with_envelope(
        seeds=arguments.seeds,
        benchmark_config=benchmark_config,
        contact_config=contact_config,
    )
    paths = contact_evaluation.write_latent_contact_artifacts(
        result,
        arguments.output_dir / "result",
    )
    analysis = _analyse(
        result,
        bootstrap_replicates=arguments.bootstrap_replicates,
        bootstrap_seed=arguments.bootstrap_seed,
        confidence_level=arguments.confidence_level,
    )
    analysis_path = arguments.output_dir / "physical-envelope-analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = arguments.output_dir / "physical-envelope-analysis.md"
    lines = [
        "# Latent-contact physical uncertainty envelope",
        "",
        "The latent-contact mean is unchanged. Predictive variance is the",
        "coordinate-wise maximum of latent-contact and nominal-physics variance.",
        "",
        f"- Fresh seeds: {analysis['seed_count']}",
        f"- Point identity passed: {analysis['point_mean_identity_passed']}",
        "- Statistical unit: complete simulation seed",
        "",
        "## Proper-score comparisons",
        "",
        "| World | Reference | Envelope NLL | Reference NLL | Gain | 95% interval | Seed wins |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for comparison in analysis["comparisons"]:
        lines.append(
            "| {world_condition} | {reference_method} | {candidate_mean:.4f} | "
            "{reference_mean:.4f} | {mean_gain:.4f} | [{low:.4f}, {high:.4f}] | "
            "{positive_seed_count}/{seed_count} |".format(
                low=comparison["gain_interval"][0],
                high=comparison["gain_interval"][1],
                seed_count=analysis["seed_count"],
                **comparison,
            )
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            "Shifted-contact NLL superiority over both raw latent-contact and "
            "nominal physics: **{}**.".format(
                "SUPPORTED"
                if analysis["shifted_nll_supported_against_both_references"]
                else "NOT SUPPORTED"
            ),
            "",
            "## Claim boundary",
            "",
            str(analysis["claim_boundary"]),
            "",
        ]
    )
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        json.dumps(
            {
                "analysis": str(analysis_path),
                "markdown": str(markdown_path),
                "result_manifest": paths["manifest"],
                "shifted_nll_supported": analysis[
                    "shifted_nll_supported_against_both_references"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
