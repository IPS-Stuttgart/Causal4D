"""Controlled simulation-based calibration for latent contact inference.

This module deliberately reuses the held-out-topology benchmark construction but
samples truths from each finite rollout bank's own prior. It is therefore an
inference self-consistency diagnostic, not a real-data calibration result and not
part of the frozen physical-acquisition method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from causal4d.baselines import FittedBaselines, fit_baselines
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig,
    ObjectProtocol,
    build_protocol,
    generate_episodes,
    make_parameter_grid,
)
from causal4d.contact_inference import (
    GraphContactHypothesisModel,
    LatentContactConfig,
    build_rollout_bank,
    fit_contact_prior,
)
from causal4d.simulation_calibration import (
    SimulationCalibrationResult,
    run_contact_rollout_sbc,
)


@dataclass(frozen=True)
class _FittedObject:
    protocol: ObjectProtocol
    baselines: FittedBaselines


def _histogram_uniformity(counts: np.ndarray) -> dict[str, float]:
    values = np.asarray(counts, dtype=float)
    if values.ndim != 1 or len(values) < 2 or np.sum(values) <= 0.0:
        raise ValueError("rank histogram must contain at least two nonempty bins")
    frequencies = values / np.sum(values)
    expected = 1.0 / len(values)
    return {
        "max_abs_frequency_error": float(np.max(np.abs(frequencies - expected))),
        "rms_frequency_error": float(
            np.sqrt(np.mean(np.square(frequencies - expected)))
        ),
    }


def _weighted_mean(
    results: Sequence[SimulationCalibrationResult],
    attribute: str,
) -> float:
    weights = np.asarray([result.trial_count for result in results], dtype=float)
    values = np.asarray([float(getattr(result, attribute)) for result in results])
    return float(np.sum(weights * values) / np.sum(weights))


def _weighted_vector_mean(
    results: Sequence[SimulationCalibrationResult],
    attribute: str,
) -> list[float]:
    weights = np.asarray([result.trial_count for result in results], dtype=float)
    values = np.asarray([getattr(result, attribute) for result in results], dtype=float)
    return [
        float(value)
        for value in np.sum(weights[:, None] * values, axis=0) / np.sum(weights)
    ]


def aggregate_controlled_sbc(
    results: Sequence[SimulationCalibrationResult],
) -> dict[str, Any]:
    """Aggregate independent SBC folds without pretending folds are new trials."""

    selected = tuple(results)
    if not selected:
        raise ValueError("at least one SBC fold result is required")
    bin_counts = {result.bin_count for result in selected}
    if len(bin_counts) != 1:
        raise ValueError("all SBC folds must use the same bin count")
    bin_count = selected[0].bin_count
    parameter_count = len(selected[0].parameter_rank_histograms)
    if any(
        len(result.parameter_rank_histograms) != parameter_count for result in selected
    ):
        raise ValueError("all SBC folds must expose the same parameter dimensions")

    joint = np.sum(
        np.asarray([result.joint_rank_histogram for result in selected], dtype=int),
        axis=0,
    )
    contact = np.sum(
        np.asarray([result.contact_rank_histogram for result in selected], dtype=int),
        axis=0,
    )
    parameters = np.sum(
        np.asarray(
            [result.parameter_rank_histograms for result in selected], dtype=int
        ),
        axis=0,
    )
    total_trials = int(sum(result.trial_count for result in selected))
    if int(np.sum(joint)) != total_trials or int(np.sum(contact)) != total_trials:
        raise ValueError("aggregate rank histograms lost SBC trials")

    return {
        "fold_count": len(selected),
        "trial_count": total_trials,
        "bin_count": bin_count,
        "rank_histograms": {
            "joint": joint.tolist(),
            "contact": contact.tolist(),
            "parameters": parameters.tolist(),
        },
        "uniformity": {
            "joint": _histogram_uniformity(joint),
            "contact": _histogram_uniformity(contact),
            "parameters": [
                _histogram_uniformity(parameters[index])
                for index in range(parameter_count)
            ],
        },
        "posterior": {
            "mean_true_joint_mass": _weighted_mean(
                selected, "mean_true_joint_posterior_mass"
            ),
            "mean_true_contact_mass": _weighted_mean(
                selected, "mean_true_contact_posterior_mass"
            ),
            "mean_joint_entropy": _weighted_mean(
                selected, "mean_joint_posterior_entropy"
            ),
            "mean_prior_joint_entropy": _weighted_mean(selected, "prior_joint_entropy"),
            "mean_entropy_reduction": _weighted_mean(
                selected, "mean_entropy_reduction"
            ),
            "mean_joint_effective_sample_size": _weighted_mean(
                selected, "mean_joint_effective_sample_size"
            ),
            "mean_parameter_absolute_error": _weighted_vector_mean(
                selected, "mean_parameter_absolute_error"
            ),
        },
    }


def run_controlled_latent_contact_sbc(
    *,
    seeds: Sequence[int],
    trials_per_fold: int = 1000,
    bin_count: int = 10,
    benchmark_config: CounterfactualBenchmarkConfig | None = None,
    contact_config: LatentContactConfig | None = None,
) -> dict[str, Any]:
    """Run exact finite-support SBC on every leave-one-topology-out bank.

    The observation simulator and inference likelihood are intentionally matched:
    Gaussian noise standard deviation equals the likelihood scale,
    ``likelihood_power`` is one, and dynamic derivative terms are disabled.
    Consequently randomized ranks should be uniform up to Monte Carlo error when
    the finite posterior implementation is internally calibrated.
    """

    if type(trials_per_fold) is not int or trials_per_fold < 1:
        raise ValueError("trials_per_fold must be a positive integer")
    if type(bin_count) is not int or bin_count < 2:
        raise ValueError("bin_count must be an integer of at least two")
    normalized_seeds = tuple(int(seed) for seed in seeds)
    if not normalized_seeds or len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("seeds must be nonempty and unique")

    cfg = benchmark_config or CounterfactualBenchmarkConfig()
    latent_cfg = contact_config or LatentContactConfig(
        confidence_level=cfg.confidence_level,
        observation_noise_std_m=cfg.observation_noise_std_m,
    )
    protocols = build_protocol(cfg)
    fold_rows: list[dict[str, Any]] = []
    fold_results: list[SimulationCalibrationResult] = []

    for seed in normalized_seeds:
        fitted: list[_FittedObject] = []
        for object_index, protocol in enumerate(protocols):
            training, validation, _ = generate_episodes(
                protocol,
                cfg,
                seed=seed * 10_000 + object_index * 101,
            )
            baselines = fit_baselines(
                training,
                validation,
                make_parameter_grid(protocol.graph_object, cfg),
                cfg,
            )
            fitted.append(_FittedObject(protocol=protocol, baselines=baselines))

        for target_index, target in enumerate(fitted):
            sources = tuple(
                item for index, item in enumerate(fitted) if index != target_index
            )
            source_protocols = tuple(item.protocol for item in sources)
            source_names = tuple(item.protocol.graph_object.name for item in sources)
            prior = fit_contact_prior(
                source_protocols,
                latent_cfg,
                action_split="test",
            )
            model = GraphContactHypothesisModel(prior=prior, config=latent_cfg)
            bank = build_rollout_bank(
                target.protocol.graph_object,
                target.protocol.test_action,
                target.baselines.physics.posterior,
                model,
                simulator_config=cfg.simulator,
                parameter_particle_count=latent_cfg.parameter_particle_count,
                variance_floor_m2=cfg.predictive_variance_floor_m2,
                confidence_level=latent_cfg.confidence_level,
            )
            prefix = latent_cfg.prefix_frame_count(cfg.frame_count)
            fold_seed = seed * 1_000_003 + target_index * 10_007 + 701
            result = run_contact_rollout_sbc(
                bank,
                trials=trials_per_fold,
                prefix_frame_count=prefix,
                likelihood_scale_m=latent_cfg.observation_noise_std_m,
                likelihood_power=1.0,
                dynamic_likelihood_weight=0.0,
                observation_noise_std_m=latent_cfg.observation_noise_std_m,
                seed=fold_seed,
                bin_count=bin_count,
            )
            fold_results.append(result)
            fold_rows.append(
                {
                    "seed": seed,
                    "held_out_object": target.protocol.graph_object.name,
                    "source_objects": list(source_names),
                    "source_excludes_target": (
                        target.protocol.graph_object.name not in source_names
                    ),
                    "contact_hypothesis_count": len(bank.contact_states),
                    "parameter_particle_count": len(bank.parameter_particles),
                    "prefix_frame_count": prefix,
                    "sbc_seed": fold_seed,
                    "result": result.as_dict(),
                }
            )

    return {
        "schema": "causal4d.controlled_latent_contact_sbc",
        "schema_version": 1,
        "interpretation": (
            "controlled finite-support inference self-consistency; not real-data "
            "calibration, physical evidence, or provider competence"
        ),
        "exact_null": {
            "observation_model": "independent Gaussian coordinates",
            "observation_noise_std_m": latent_cfg.observation_noise_std_m,
            "likelihood_scale_m": latent_cfg.observation_noise_std_m,
            "likelihood_power": 1.0,
            "dynamic_likelihood_weight": 0.0,
        },
        "seeds": list(normalized_seeds),
        "trials_per_fold": trials_per_fold,
        "benchmark_config": cfg.as_dict(),
        "contact_config": latent_cfg.as_dict(),
        "folds": fold_rows,
        "aggregate": aggregate_controlled_sbc(fold_results),
    }
