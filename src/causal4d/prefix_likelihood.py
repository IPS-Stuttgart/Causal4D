"""Correlation-aware prefix likelihoods for finite physical rollout banks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causal4d.immutable_array import readonly_integer_array
from causal4d.weighting import log_weights_from_probabilities

from causal4d.rollout_bank import JointRolloutBank


@dataclass(frozen=True)
class PrefixLikelihoodConfig:
    """Robust composite-likelihood settings for a causal response prefix."""

    observation_scale_m: float = 0.01
    likelihood_power: float = 1.0
    position_likelihood_weight: float = 1.0
    dynamic_likelihood_weight: float = 0.25
    degrees_of_freedom: float = 4.0
    difference_correlation: float = 0.0

    def __post_init__(self) -> None:
        positive = (
            self.observation_scale_m,
            self.likelihood_power,
            self.degrees_of_freedom,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError(
                "observation scale, likelihood power, and dof must be positive"
            )
        if not np.isfinite(self.position_likelihood_weight) or (
            self.position_likelihood_weight < 0.0
        ):
            raise ValueError("position_likelihood_weight must be nonnegative")
        if not np.isfinite(self.dynamic_likelihood_weight) or (
            self.dynamic_likelihood_weight < 0.0
        ):
            raise ValueError("dynamic_likelihood_weight must be nonnegative")
        if self.position_likelihood_weight + self.dynamic_likelihood_weight <= 0.0:
            raise ValueError("at least one prefix likelihood block must be active")
        if not np.isfinite(self.difference_correlation) or not (
            -1.0 < self.difference_correlation < 1.0
        ):
            raise ValueError("difference_correlation must lie in (-1, 1)")


def _coordinate_mask(
    observations: np.ndarray,
    mask: np.ndarray | None,
) -> np.ndarray:
    finite = np.isfinite(observations)
    if mask is None:
        return finite
    supplied = np.asarray(mask, dtype=bool)
    if supplied.shape == observations.shape[:2]:
        supplied = np.repeat(supplied[:, :, None], observations.shape[2], axis=2)
    if supplied.shape != observations.shape:
        raise ValueError("observation mask must have shape (T, N) or (T, N, C)")
    return finite & supplied


def _student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    scale_m: np.ndarray | float,
    degrees_of_freedom: float,
    reduction_axes: tuple[int, ...],
) -> np.ndarray:
    """Return a normalized Student-t score, including scale normalization.

    Constants independent of the rollout component are omitted. The ``-log(scale)``
    term is retained because particle-specific discrepancy variances can make the
    scale differ across physical particles.
    """

    values = np.asarray(residual, dtype=float)
    scale = np.asarray(scale_m, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("likelihood scales must be finite and positive")
    try:
        standardized = values / scale
        log_scale = np.broadcast_to(np.log(scale), values.shape)
    except ValueError as error:
        raise ValueError(
            "likelihood scale is not broadcastable to residuals"
        ) from error
    terms = -log_scale - 0.5 * (degrees_of_freedom + 1.0) * np.log1p(
        np.square(standardized) / degrees_of_freedom
    )
    valid_float = np.asarray(valid, dtype=float)
    while valid_float.ndim < terms.ndim:
        valid_float = valid_float[None]
    count = np.sum(valid_float, axis=reduction_axes)
    if np.any(count <= 0.0):
        raise ValueError("likelihood update has no valid coordinates")
    return (
        np.sum(
            np.where(valid_float > 0.0, terms, 0.0),
            axis=reduction_axes,
        )
        / count
    )


def _normalized_joint_weights(log_weights: np.ndarray) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        raise ValueError("joint log weights must contain finite support")
    maximum = float(np.max(values[np.isfinite(values)]))
    weights = np.exp(np.where(np.isfinite(values), values - maximum, -np.inf))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("joint posterior normalization failed")
    return weights / total


def prefix_component_log_likelihood(
    bank: JointRolloutBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    config: PrefixLikelihoodConfig | None = None,
    mask: np.ndarray | None = None,
    observed_nodes: Sequence[int] | None = None,
    particle_discrepancy_m: np.ndarray | None = None,
    particle_discrepancy_variance_m2: np.ndarray | None = None,
) -> np.ndarray:
    """Score every ``(hypothesis, particle)`` using only the allowed prefix.

    The dynamic block differences the complete prefix, including the endpoint to
    first-response increment. A time-persistent discrepancy mean cancels under
    differencing, and its static uncertainty is therefore used only in the
    position block. Observation-difference scale is derived from the declared
    adjacent-frame correlation rather than reusing the position scale.
    """

    settings = config or PrefixLikelihoodConfig()
    observations = np.asarray(observations_m, dtype=float)
    expected = bank.trajectories.shape[2:]
    if observations.shape != expected:
        raise ValueError(f"observations must have shape {expected}")
    if not 2 <= prefix_frame_count < bank.frame_count:
        raise ValueError("prefix_frame_count must leave at least one future frame")
    raw_nodes = (
        tuple(range(bank.node_count))
        if observed_nodes is None
        else tuple(observed_nodes)
    )
    nodes = readonly_integer_array(raw_nodes, name="observed_nodes")
    if (
        nodes.ndim != 1
        or not len(nodes)
        or np.any(nodes < 0)
        or np.any(nodes >= bank.node_count)
        or len(np.unique(nodes)) != len(nodes)
    ):
        raise ValueError(
            "observed_nodes must uniquely identify available rollout nodes"
        )

    coordinate_valid = _coordinate_mask(observations, mask)
    observed_prefix = observations[:prefix_frame_count, nodes]
    valid_prefix = coordinate_valid[:prefix_frame_count, nodes]
    predicted_prefix = bank.trajectories[:, :, :prefix_frame_count, nodes].astype(float)

    position_scale: np.ndarray | float = settings.observation_scale_m
    if particle_discrepancy_m is not None:
        discrepancy = np.asarray(particle_discrepancy_m, dtype=float)
        expected_shape = (
            len(bank.parameter_weights),
            bank.node_count,
            bank.coordinate_count,
        )
        if discrepancy.shape != expected_shape:
            raise ValueError(f"particle_discrepancy_m must have shape {expected_shape}")
        if not np.all(np.isfinite(discrepancy)):
            raise ValueError("particle discrepancy must be finite")
        predicted_prefix = predicted_prefix + discrepancy[None, :, None, nodes]

    if particle_discrepancy_variance_m2 is not None:
        discrepancy_variance = np.asarray(
            particle_discrepancy_variance_m2,
            dtype=float,
        )
        expected_shape = (
            len(bank.parameter_weights),
            bank.node_count,
            bank.coordinate_count,
        )
        if discrepancy_variance.shape != expected_shape:
            raise ValueError(
                f"particle_discrepancy_variance_m2 must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(discrepancy_variance)) or np.any(
            discrepancy_variance < 0.0
        ):
            raise ValueError("particle discrepancy variance must be nonnegative")
        position_scale = np.sqrt(
            settings.observation_scale_m**2 + discrepancy_variance[None, :, None, nodes]
        )

    result = np.zeros(
        (len(bank.hypothesis_ids), len(bank.parameter_weights)),
        dtype=float,
    )
    if settings.position_likelihood_weight > 0.0:
        position_score = _student_t_mean_log_score(
            predicted_prefix[:, :, 1:] - observed_prefix[None, None, 1:],
            valid_prefix[1:],
            scale_m=position_scale,
            degrees_of_freedom=settings.degrees_of_freedom,
            reduction_axes=(2, 3, 4),
        )
        result += settings.position_likelihood_weight * position_score

    if settings.dynamic_likelihood_weight > 0.0:
        predicted_difference = np.diff(predicted_prefix, axis=2)
        observed_difference = np.diff(observed_prefix, axis=0)
        difference_valid = valid_prefix[1:] & valid_prefix[:-1]
        difference_scale = settings.observation_scale_m * np.sqrt(
            2.0 * (1.0 - settings.difference_correlation)
        )
        dynamic_score = _student_t_mean_log_score(
            predicted_difference - observed_difference[None, None],
            difference_valid,
            scale_m=difference_scale,
            degrees_of_freedom=settings.degrees_of_freedom,
            reduction_axes=(2, 3, 4),
        )
        result += settings.dynamic_likelihood_weight * dynamic_score

    return settings.likelihood_power * result


def update_joint_weights_from_prefix(
    bank: JointRolloutBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    config: PrefixLikelihoodConfig | None = None,
    mask: np.ndarray | None = None,
    observed_nodes: Sequence[int] | None = None,
    base_weights: np.ndarray | None = None,
    particle_discrepancy_m: np.ndarray | None = None,
    particle_discrepancy_variance_m2: np.ndarray | None = None,
) -> np.ndarray:
    """Update a rollout bank with the correlation-aware causal prefix score."""

    weights = (
        bank.prior_joint_weights
        if base_weights is None
        else np.asarray(base_weights, dtype=float)
    )
    if weights.shape != bank.prior_joint_weights.shape:
        raise ValueError("base_weights must match the joint rollout support")
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
        or not np.isclose(np.sum(weights), 1.0)
    ):
        raise ValueError("base_weights must be finite, nonnegative, and sum to one")
    log_likelihood = prefix_component_log_likelihood(
        bank,
        observations_m,
        prefix_frame_count=prefix_frame_count,
        config=config,
        mask=mask,
        observed_nodes=observed_nodes,
        particle_discrepancy_m=particle_discrepancy_m,
        particle_discrepancy_variance_m2=particle_discrepancy_variance_m2,
    )
    return _normalized_joint_weights(
        log_weights_from_probabilities(weights, name="base_weights") + log_likelihood
    )
