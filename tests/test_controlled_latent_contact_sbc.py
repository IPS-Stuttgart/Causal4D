from __future__ import annotations

import numpy as np

from causal4d.benchmark import CounterfactualBenchmarkConfig
from causal4d.contact_inference import LatentContactConfig
from causal4d.controlled_latent_contact_sbc import (
    aggregate_controlled_sbc,
    run_controlled_latent_contact_sbc,
)
from causal4d.simulation_calibration import SimulationCalibrationResult


def _result(
    *, trials: int, seed: int, counts: tuple[int, int]
) -> SimulationCalibrationResult:
    parameter_histograms = (counts, counts, counts)
    return SimulationCalibrationResult(
        trial_count=trials,
        bin_count=2,
        seed=seed,
        joint_rank_histogram=counts,
        contact_rank_histogram=counts,
        parameter_rank_histograms=parameter_histograms,
        joint_rank_max_abs_frequency_error=0.1,
        contact_rank_max_abs_frequency_error=0.1,
        parameter_rank_max_abs_frequency_error=(0.1, 0.1, 0.1),
        joint_rank_rms_frequency_error=0.1,
        contact_rank_rms_frequency_error=0.1,
        parameter_rank_rms_frequency_error=(0.1, 0.1, 0.1),
        mean_true_joint_posterior_mass=0.4,
        mean_true_contact_posterior_mass=0.6,
        mean_joint_posterior_entropy=0.8,
        prior_joint_entropy=1.0,
        mean_entropy_reduction=0.2,
        mean_joint_effective_sample_size=2.0,
        mean_parameter_absolute_error=(0.1, 0.2, 0.3),
    )


def test_aggregate_controlled_sbc_sums_trials_and_histograms() -> None:
    first = _result(trials=10, seed=1, counts=(4, 6))
    second = _result(trials=20, seed=2, counts=(11, 9))
    aggregate = aggregate_controlled_sbc((first, second))

    assert aggregate["fold_count"] == 2
    assert aggregate["trial_count"] == 30
    assert aggregate["rank_histograms"]["joint"] == [15, 15]
    assert aggregate["rank_histograms"]["contact"] == [15, 15]
    assert aggregate["uniformity"]["joint"]["max_abs_frequency_error"] == 0.0
    assert np.isclose(aggregate["posterior"]["mean_entropy_reduction"], 0.2)


def test_controlled_sbc_builds_all_held_out_topology_folds() -> None:
    benchmark = CounterfactualBenchmarkConfig(
        frame_count=8,
        training_repeats=2,
        parameter_grid_count=3,
        fit_frame_stride=2,
        observation_noise_std_m=0.0015,
        inference_noise_std_m=0.004,
    )
    contact = LatentContactConfig(
        observation_fraction=0.20,
        observation_noise_std_m=0.0015,
        parameter_particle_count=2,
        gain_values=(0.8, 1.0),
        delay_values=(0, 1),
        slip_values=(0.0,),
        rotation_values_deg=(0.0,),
        confidence_level=benchmark.confidence_level,
    )
    result = run_controlled_latent_contact_sbc(
        seeds=(3,),
        trials_per_fold=12,
        bin_count=4,
        benchmark_config=benchmark,
        contact_config=contact,
    )

    assert result["schema"] == "causal4d.controlled_latent_contact_sbc"
    assert "not real-data calibration" in result["interpretation"]
    assert result["aggregate"]["fold_count"] == 3
    assert result["aggregate"]["trial_count"] == 36
    assert {fold["held_out_object"] for fold in result["folds"]} == {
        "rope",
        "cloth",
        "soft_block",
    }
    assert all(fold["source_excludes_target"] for fold in result["folds"])
    assert all(
        sum(fold["result"]["rank_histograms"]["joint"]) == 12
        for fold in result["folds"]
    )
