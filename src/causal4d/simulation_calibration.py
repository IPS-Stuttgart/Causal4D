"""Simulation-based calibration for finite Causal4D posterior support.

The diagnostics in this module are controlled-model checks.  They test whether a
finite contact/physics posterior recovers draws from its own declared generative
model.  Passing SBC does not establish real-data calibration or provider
competence, and this module is not part of the frozen 36-execution estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from causal4d.contact_inference import ContactRolloutBank, ContactState


_PARAMETER_NAMES = ("stiffness", "damping", "contact_gain")


def _validated_probability_weights(values: np.ndarray, *, name: str) -> np.ndarray:
    weights = np.asarray(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError(f"{name} must have positive total mass")
    return weights / total


def _entropy(weights: np.ndarray) -> float:
    positive = weights[weights > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def _contact_key(state: ContactState) -> tuple[object, ...]:
    return (
        tuple(map(int, state.contact_nodes)),
        float(state.gain_multiplier),
        int(state.delay_steps),
        float(state.slip_fraction),
        float(state.rotation_radians),
    )


def _randomized_pit_ordered(
    weights: np.ndarray,
    truth_index: int,
    order: Sequence[int],
    rng: np.random.Generator,
) -> float:
    probabilities = _validated_probability_weights(weights, name="posterior weights")
    ordered = tuple(map(int, order))
    if len(ordered) != len(probabilities) or set(ordered) != set(
        range(len(probabilities))
    ):
        raise ValueError("order must be a permutation of posterior indices")
    if not 0 <= truth_index < len(probabilities):
        raise ValueError("truth_index lies outside posterior support")
    position = ordered.index(int(truth_index))
    lower = float(np.sum(probabilities[list(ordered[:position])]))
    mass = float(probabilities[truth_index])
    return float(lower + rng.random() * mass)


def _randomized_numeric_pit(
    values: np.ndarray,
    weights: np.ndarray,
    truth_value: float,
    rng: np.random.Generator,
) -> float:
    support = np.asarray(values, dtype=float)
    probabilities = _validated_probability_weights(weights, name="numeric posterior")
    if support.shape != probabilities.shape or not np.all(np.isfinite(support)):
        raise ValueError("numeric support must be finite and match posterior weights")
    equal = support == truth_value
    if not np.any(equal):
        raise ValueError("truth_value must belong to the numeric support")
    lower = float(np.sum(probabilities[support < truth_value]))
    mass = float(np.sum(probabilities[equal]))
    return float(lower + rng.random() * mass)


def _histogram(
    pits: Sequence[float], bin_count: int
) -> tuple[tuple[int, ...], float, float]:
    values = np.asarray(tuple(pits), dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("at least one PIT value is required")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("PIT values must lie in [0, 1]")
    if type(bin_count) is not int or bin_count < 2:
        raise ValueError("bin_count must be an integer of at least two")
    indices = np.minimum((values * bin_count).astype(int), bin_count - 1)
    counts = np.bincount(indices, minlength=bin_count)
    frequencies = counts / len(values)
    expected = 1.0 / bin_count
    maximum_error = float(np.max(np.abs(frequencies - expected)))
    rms_error = float(np.sqrt(np.mean(np.square(frequencies - expected))))
    return tuple(map(int, counts)), maximum_error, rms_error


@dataclass(frozen=True)
class SimulationCalibrationResult:
    """Aggregate randomized-rank diagnostics for a finite posterior."""

    trial_count: int
    bin_count: int
    seed: int
    joint_rank_histogram: tuple[int, ...]
    contact_rank_histogram: tuple[int, ...]
    parameter_rank_histograms: tuple[tuple[int, ...], ...]
    joint_rank_max_abs_frequency_error: float
    contact_rank_max_abs_frequency_error: float
    parameter_rank_max_abs_frequency_error: tuple[float, ...]
    joint_rank_rms_frequency_error: float
    contact_rank_rms_frequency_error: float
    parameter_rank_rms_frequency_error: tuple[float, ...]
    mean_true_joint_posterior_mass: float
    mean_true_contact_posterior_mass: float
    mean_joint_posterior_entropy: float
    prior_joint_entropy: float
    mean_entropy_reduction: float
    mean_joint_effective_sample_size: float
    mean_parameter_absolute_error: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.trial_count < 1:
            raise ValueError("trial_count must be positive")
        if self.bin_count < 2:
            raise ValueError("bin_count must be at least two")
        histograms = (
            self.joint_rank_histogram,
            self.contact_rank_histogram,
            *self.parameter_rank_histograms,
        )
        for histogram in histograms:
            if len(histogram) != self.bin_count or sum(histogram) != self.trial_count:
                raise ValueError("every rank histogram must cover every trial")
        if len(self.parameter_rank_histograms) != len(_PARAMETER_NAMES):
            raise ValueError("parameter rank histograms must match physical parameters")
        if len(self.parameter_rank_max_abs_frequency_error) != len(_PARAMETER_NAMES):
            raise ValueError("parameter maximum errors must match physical parameters")
        if len(self.parameter_rank_rms_frequency_error) != len(_PARAMETER_NAMES):
            raise ValueError("parameter RMS errors must match physical parameters")
        if len(self.mean_parameter_absolute_error) != len(_PARAMETER_NAMES):
            raise ValueError("parameter errors must match physical parameters")
        scalar_values = (
            self.joint_rank_max_abs_frequency_error,
            self.contact_rank_max_abs_frequency_error,
            *self.parameter_rank_max_abs_frequency_error,
            self.joint_rank_rms_frequency_error,
            self.contact_rank_rms_frequency_error,
            *self.parameter_rank_rms_frequency_error,
            self.mean_true_joint_posterior_mass,
            self.mean_true_contact_posterior_mass,
            self.mean_joint_posterior_entropy,
            self.prior_joint_entropy,
            self.mean_entropy_reduction,
            self.mean_joint_effective_sample_size,
            *self.mean_parameter_absolute_error,
        )
        if not all(np.isfinite(value) for value in scalar_values):
            raise ValueError("simulation-calibration summaries must be finite")
        if not 0.0 <= self.mean_true_joint_posterior_mass <= 1.0:
            raise ValueError("mean_true_joint_posterior_mass must lie in [0, 1]")
        if not 0.0 <= self.mean_true_contact_posterior_mass <= 1.0:
            raise ValueError("mean_true_contact_posterior_mass must lie in [0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "causal4d.simulation_calibration",
            "schema_version": 1,
            "interpretation": (
                "controlled finite-support SBC; not evidence of real-data calibration"
            ),
            "trial_count": self.trial_count,
            "bin_count": self.bin_count,
            "seed": self.seed,
            "rank_histograms": {
                "joint": list(self.joint_rank_histogram),
                "contact": list(self.contact_rank_histogram),
                "parameters": {
                    name: list(histogram)
                    for name, histogram in zip(
                        _PARAMETER_NAMES,
                        self.parameter_rank_histograms,
                        strict=True,
                    )
                },
            },
            "uniformity_error": {
                "joint_max_abs_frequency": self.joint_rank_max_abs_frequency_error,
                "contact_max_abs_frequency": self.contact_rank_max_abs_frequency_error,
                "parameter_max_abs_frequency": dict(
                    zip(
                        _PARAMETER_NAMES,
                        self.parameter_rank_max_abs_frequency_error,
                        strict=True,
                    )
                ),
                "joint_rms_frequency": self.joint_rank_rms_frequency_error,
                "contact_rms_frequency": self.contact_rank_rms_frequency_error,
                "parameter_rms_frequency": dict(
                    zip(
                        _PARAMETER_NAMES,
                        self.parameter_rank_rms_frequency_error,
                        strict=True,
                    )
                ),
            },
            "posterior": {
                "mean_true_joint_mass": self.mean_true_joint_posterior_mass,
                "mean_true_contact_mass": self.mean_true_contact_posterior_mass,
                "prior_joint_entropy": self.prior_joint_entropy,
                "mean_joint_entropy": self.mean_joint_posterior_entropy,
                "mean_entropy_reduction": self.mean_entropy_reduction,
                "mean_joint_effective_sample_size": (
                    self.mean_joint_effective_sample_size
                ),
                "mean_parameter_absolute_error": dict(
                    zip(
                        _PARAMETER_NAMES,
                        self.mean_parameter_absolute_error,
                        strict=True,
                    )
                ),
            },
        }


def run_contact_rollout_sbc(
    bank: ContactRolloutBank,
    *,
    trials: int,
    prefix_frame_count: int,
    likelihood_scale_m: float,
    likelihood_power: float = 1.0,
    dynamic_likelihood_weight: float = 0.0,
    observation_noise_std_m: float | None = None,
    observed_nodes: Sequence[int] | None = None,
    seed: int = 0,
    bin_count: int = 10,
) -> SimulationCalibrationResult:
    """Run randomized-rank SBC against a ``ContactRolloutBank``.

    A joint contact/parameter component is sampled from ``bank.prior_joint_weights``.
    Its complete trajectory is treated as the controlled truth, Gaussian
    observation noise is added, and the ordinary prefix-only update is rerun.

    With ``likelihood_power=1``, ``dynamic_likelihood_weight=0`` and
    ``observation_noise_std_m == likelihood_scale_m``, this is an exact
    self-consistency check for the finite Gaussian observation model used by
    ``ContactRolloutBank.update_weights``.  Other settings are useful sensitivity
    diagnostics and need not produce uniform ranks.
    """

    if type(trials) is not int or trials < 1:
        raise ValueError("trials must be a positive integer")
    if type(bin_count) is not int or bin_count < 2:
        raise ValueError("bin_count must be an integer of at least two")
    if not np.isfinite(likelihood_scale_m) or likelihood_scale_m <= 0.0:
        raise ValueError("likelihood_scale_m must be finite and positive")
    if not np.isfinite(likelihood_power) or likelihood_power < 0.0:
        raise ValueError("likelihood_power must be finite and nonnegative")
    if not np.isfinite(dynamic_likelihood_weight) or dynamic_likelihood_weight < 0.0:
        raise ValueError("dynamic_likelihood_weight must be finite and nonnegative")
    noise_std = (
        float(likelihood_scale_m)
        if observation_noise_std_m is None
        else float(observation_noise_std_m)
    )
    if not np.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("observation_noise_std_m must be finite and nonnegative")

    prior_joint = np.asarray(bank.prior_joint_weights, dtype=float)
    flat_prior = _validated_probability_weights(
        prior_joint.reshape(-1),
        name="joint prior",
    )
    contact_count, parameter_count = prior_joint.shape
    contact_order = tuple(
        sorted(
            range(contact_count),
            key=lambda index: _contact_key(bank.contact_states[index]),
        )
    )
    joint_order = tuple(
        sorted(
            range(contact_count * parameter_count),
            key=lambda flat_index: (
                _contact_key(bank.contact_states[flat_index // parameter_count]),
                tuple(
                    map(
                        float,
                        bank.parameter_particles[flat_index % parameter_count],
                    )
                ),
            ),
        )
    )

    rng = np.random.default_rng(seed)
    joint_pits: list[float] = []
    contact_pits: list[float] = []
    parameter_pits: list[list[float]] = [
        [] for _ in range(bank.parameter_particles.shape[1])
    ]
    true_joint_mass: list[float] = []
    true_contact_mass: list[float] = []
    posterior_entropy: list[float] = []
    posterior_ess: list[float] = []
    parameter_absolute_error: list[np.ndarray] = []

    for _ in range(trials):
        flat_truth = int(rng.choice(len(flat_prior), p=flat_prior))
        contact_index = flat_truth // parameter_count
        parameter_index = flat_truth % parameter_count
        truth = bank.trajectories[contact_index, parameter_index]
        observations = truth.copy()
        if noise_std:
            observations += rng.normal(scale=noise_std, size=observations.shape)

        posterior = bank.update_weights(
            observations,
            prefix_frame_count=prefix_frame_count,
            likelihood_scale_m=likelihood_scale_m,
            likelihood_power=likelihood_power,
            dynamic_likelihood_weight=dynamic_likelihood_weight,
            observed_nodes=observed_nodes,
        )
        flat_posterior = _validated_probability_weights(
            posterior.reshape(-1),
            name="joint posterior",
        )
        contact_posterior = bank.contact_marginal(posterior)
        parameter_posterior = bank.parameter_marginal(posterior)

        joint_pits.append(
            _randomized_pit_ordered(
                flat_posterior,
                flat_truth,
                joint_order,
                rng,
            )
        )
        contact_pits.append(
            _randomized_pit_ordered(
                contact_posterior,
                contact_index,
                contact_order,
                rng,
            )
        )
        truth_parameters = bank.parameter_particles[parameter_index]
        for dimension in range(bank.parameter_particles.shape[1]):
            parameter_pits[dimension].append(
                _randomized_numeric_pit(
                    bank.parameter_particles[:, dimension],
                    parameter_posterior,
                    float(truth_parameters[dimension]),
                    rng,
                )
            )

        true_joint_mass.append(float(posterior[contact_index, parameter_index]))
        true_contact_mass.append(float(contact_posterior[contact_index]))
        posterior_entropy.append(_entropy(flat_posterior))
        posterior_ess.append(float(1.0 / np.sum(np.square(flat_posterior))))
        posterior_parameter_mean = np.sum(
            parameter_posterior[:, None] * bank.parameter_particles,
            axis=0,
        )
        parameter_absolute_error.append(
            np.abs(posterior_parameter_mean - truth_parameters)
        )

    joint_hist, joint_max, joint_rms = _histogram(joint_pits, bin_count)
    contact_hist, contact_max, contact_rms = _histogram(contact_pits, bin_count)
    parameter_summaries = [_histogram(values, bin_count) for values in parameter_pits]
    prior_entropy = _entropy(flat_prior)
    mean_posterior_entropy = float(np.mean(posterior_entropy))

    return SimulationCalibrationResult(
        trial_count=trials,
        bin_count=bin_count,
        seed=int(seed),
        joint_rank_histogram=joint_hist,
        contact_rank_histogram=contact_hist,
        parameter_rank_histograms=tuple(summary[0] for summary in parameter_summaries),
        joint_rank_max_abs_frequency_error=joint_max,
        contact_rank_max_abs_frequency_error=contact_max,
        parameter_rank_max_abs_frequency_error=tuple(
            summary[1] for summary in parameter_summaries
        ),
        joint_rank_rms_frequency_error=joint_rms,
        contact_rank_rms_frequency_error=contact_rms,
        parameter_rank_rms_frequency_error=tuple(
            summary[2] for summary in parameter_summaries
        ),
        mean_true_joint_posterior_mass=float(np.mean(true_joint_mass)),
        mean_true_contact_posterior_mass=float(np.mean(true_contact_mass)),
        mean_joint_posterior_entropy=mean_posterior_entropy,
        prior_joint_entropy=prior_entropy,
        mean_entropy_reduction=float(prior_entropy - mean_posterior_entropy),
        mean_joint_effective_sample_size=float(np.mean(posterior_ess)),
        mean_parameter_absolute_error=tuple(
            map(
                float,
                np.mean(np.stack(parameter_absolute_error, axis=0), axis=0),
            )
        ),
    )
