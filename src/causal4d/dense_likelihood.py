"""Exact bounded-memory evaluation for dense rollout-bank likelihoods."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np

from causal4d.immutable_array import readonly_integer_array
from causal4d.prefix_likelihood import PrefixLikelihoodConfig
from causal4d.rollout_bank import JointRolloutBank
from causal4d.weighting import log_weights_from_probabilities


DenseLikelihoodSemantics = Literal["legacy_v1", "normalized_v2"]


def _validated_component_batch_size(value: int) -> int:
    if type(value) is not int or value < 1:
        raise ValueError("component_batch_size must be a positive integer")
    return value


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


def _legacy_student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    scale_m: np.ndarray | float,
    degrees_of_freedom: float,
    reduction_axes: tuple[int, ...],
) -> np.ndarray:
    if not np.isfinite(degrees_of_freedom) or degrees_of_freedom <= 0.0:
        raise ValueError("degrees_of_freedom must be finite and positive")
    scale = np.asarray(scale_m, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
        raise ValueError("likelihood scales must be finite and positive")
    standardized = residual / scale
    terms = (
        -0.5
        * (degrees_of_freedom + 1.0)
        * np.log1p(np.square(standardized) / degrees_of_freedom)
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


def _normalized_student_t_mean_log_score(
    residual: np.ndarray,
    valid: np.ndarray,
    *,
    scale_m: np.ndarray | float,
    degrees_of_freedom: float,
    reduction_axes: tuple[int, ...],
) -> np.ndarray:
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


def _validated_base_weights(
    bank: JointRolloutBank,
    base_weights: np.ndarray | None,
) -> np.ndarray:
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
    return weights


def _normalize_joint_log_weights(
    log_weights: np.ndarray,
    *,
    reject_invalid_support: bool,
) -> np.ndarray:
    values = np.asarray(log_weights, dtype=float)
    if reject_invalid_support:
        if values.ndim != 2:
            raise ValueError("joint log weights must be a matrix")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError("joint log weights may contain only finite values or -inf")
    if values.ndim != 2 or not np.any(np.isfinite(values)):
        raise ValueError("joint log weights must contain finite support")
    maximum = float(np.max(values[np.isfinite(values)]))
    weights = np.exp(np.where(np.isfinite(values), values - maximum, -np.inf))
    total = float(np.sum(weights))
    if not np.isfinite(total) or total <= 0.0:
        raise RuntimeError("joint posterior normalization failed")
    return weights / total


def _validated_nodes(
    bank: JointRolloutBank,
    observed_nodes: Sequence[int] | None,
) -> np.ndarray:
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
    return nodes


def _validated_discrepancy(
    bank: JointRolloutBank,
    particle_discrepancy_m: np.ndarray | None,
    particle_discrepancy_variance_m2: np.ndarray | None,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    expected_shape = (
        len(bank.parameter_weights),
        bank.node_count,
        bank.coordinate_count,
    )
    discrepancy: np.ndarray | None = None
    if particle_discrepancy_m is not None:
        discrepancy = np.asarray(particle_discrepancy_m, dtype=float)
        if discrepancy.shape != expected_shape:
            raise ValueError(f"particle_discrepancy_m must have shape {expected_shape}")
        if not np.all(np.isfinite(discrepancy)):
            raise ValueError("particle discrepancy must be finite")

    discrepancy_variance: np.ndarray | None = None
    if particle_discrepancy_variance_m2 is not None:
        discrepancy_variance = np.asarray(
            particle_discrepancy_variance_m2,
            dtype=float,
        )
        if discrepancy_variance.shape != expected_shape:
            raise ValueError(
                f"particle_discrepancy_variance_m2 must have shape {expected_shape}"
            )
        if not np.all(np.isfinite(discrepancy_variance)) or np.any(
            discrepancy_variance < 0.0
        ):
            raise ValueError("particle discrepancy variance must be nonnegative")
    return discrepancy, discrepancy_variance


def update_dense_joint_weights_batched(
    bank: JointRolloutBank,
    observations_m: np.ndarray,
    *,
    prefix_frame_count: int,
    component_batch_size: int,
    likelihood_semantics: DenseLikelihoodSemantics,
    observation_scale_m: float,
    likelihood_power: float = 1.0,
    dynamic_likelihood_weight: float = 0.0,
    degrees_of_freedom: float = 4.0,
    difference_correlation: float = 0.0,
    mask: np.ndarray | None = None,
    observed_nodes: Sequence[int] | None = None,
    base_weights: np.ndarray | None = None,
    particle_discrepancy_m: np.ndarray | None = None,
    particle_discrepancy_variance_m2: np.ndarray | None = None,
) -> np.ndarray:
    """Update dense support without materializing all component residuals.

    Components are traversed in the same hypothesis-major order used by the
    rollout bank. The batch size is execution-only: score reductions, posterior
    normalization, and artifact-visible values are identical to the corresponding
    unbatched ``legacy_v1`` or ``normalized_v2`` implementation.
    """

    batch_size = _validated_component_batch_size(component_batch_size)
    if likelihood_semantics not in {"legacy_v1", "normalized_v2"}:
        raise ValueError("unsupported dense likelihood semantics")

    observations = np.asarray(observations_m, dtype=float)
    expected = bank.trajectories.shape[2:]
    if observations.shape != expected:
        raise ValueError(f"observations must have shape {expected}")
    if not 2 <= prefix_frame_count < bank.frame_count:
        raise ValueError("prefix_frame_count must leave at least one future frame")

    if likelihood_semantics == "legacy_v1":
        if (
            not np.isfinite(observation_scale_m)
            or observation_scale_m <= 0.0
            or not np.isfinite(likelihood_power)
            or likelihood_power <= 0.0
        ):
            raise ValueError(
                "observation scale and likelihood power must be finite and positive"
            )
        if (
            not np.isfinite(dynamic_likelihood_weight)
            or dynamic_likelihood_weight < 0.0
        ):
            raise ValueError("dynamic_likelihood_weight must be finite and nonnegative")
        if difference_correlation != 0.0:
            raise ValueError(
                "difference_correlation is available only with normalized_v2"
            )
        normalized_config = None
    else:
        normalized_config = PrefixLikelihoodConfig(
            observation_scale_m=observation_scale_m,
            likelihood_power=likelihood_power,
            position_likelihood_weight=1.0,
            dynamic_likelihood_weight=dynamic_likelihood_weight,
            degrees_of_freedom=degrees_of_freedom,
            difference_correlation=difference_correlation,
        )

    nodes = _validated_nodes(bank, observed_nodes)
    coordinate_valid = _coordinate_mask(observations, mask)
    discrepancy, discrepancy_variance = _validated_discrepancy(
        bank,
        particle_discrepancy_m,
        particle_discrepancy_variance_m2,
    )
    weights = _validated_base_weights(bank, base_weights)

    hypothesis_count = len(bank.hypothesis_ids)
    particle_count = len(bank.parameter_weights)
    component_count = hypothesis_count * particle_count
    component_scores = np.empty(component_count, dtype=float)

    if likelihood_semantics == "legacy_v1":
        observed_prefix = observations[1:prefix_frame_count, nodes]
        valid_prefix = coordinate_valid[1:prefix_frame_count, nodes]
    else:
        observed_prefix = observations[:prefix_frame_count, nodes]
        valid_prefix = coordinate_valid[:prefix_frame_count, nodes]

    for start in range(0, component_count, batch_size):
        stop = min(start + batch_size, component_count)
        flat_indices = np.arange(start, stop, dtype=np.int64)
        hypothesis_indices = flat_indices // particle_count
        particle_indices = flat_indices % particle_count
        predicted = bank.trajectories[hypothesis_indices, particle_indices]

        if likelihood_semantics == "legacy_v1":
            predicted_prefix = predicted[:, 1:prefix_frame_count, nodes]
        else:
            predicted_prefix = predicted[:, :prefix_frame_count, nodes].astype(float)

        if discrepancy is not None:
            predicted_prefix = (
                predicted_prefix + discrepancy[particle_indices][:, None, nodes]
            )

        position_scale: np.ndarray | float = observation_scale_m
        if discrepancy_variance is not None:
            position_scale = np.sqrt(
                observation_scale_m**2
                + discrepancy_variance[particle_indices][:, None, nodes]
            )

        if likelihood_semantics == "legacy_v1":
            if stop - start == 1:
                # Preserve the registered legacy reduction layout even for a
                # singleton execution batch. NumPy 1.24 may choose a different
                # summation kernel for (1, T, N, C) than for the original
                # (H, P, T, N, C) layout, changing the last bit of the posterior.
                # The extra singleton particle axis uses no additional component
                # support and keeps component_batch_size=1 genuinely bounded.
                legacy_predicted = predicted_prefix[:, None]
                legacy_scale = (
                    position_scale[:, None]
                    if isinstance(position_scale, np.ndarray)
                    else position_scale
                )
                score = _legacy_student_t_mean_log_score(
                    legacy_predicted - observed_prefix[None, None],
                    valid_prefix,
                    scale_m=legacy_scale,
                    degrees_of_freedom=degrees_of_freedom,
                    reduction_axes=(2, 3, 4),
                ).reshape(-1)
                if dynamic_likelihood_weight and prefix_frame_count >= 4:
                    score += (
                        dynamic_likelihood_weight
                        * _legacy_student_t_mean_log_score(
                            np.diff(legacy_predicted, axis=2)
                            - np.diff(observed_prefix, axis=0)[None, None],
                            valid_prefix[1:] & valid_prefix[:-1],
                            scale_m=observation_scale_m,
                            degrees_of_freedom=degrees_of_freedom,
                            reduction_axes=(2, 3, 4),
                        ).reshape(-1)
                    )
            else:
                score = _legacy_student_t_mean_log_score(
                    predicted_prefix - observed_prefix[None],
                    valid_prefix,
                    scale_m=position_scale,
                    degrees_of_freedom=degrees_of_freedom,
                    reduction_axes=(1, 2, 3),
                )
                if dynamic_likelihood_weight and prefix_frame_count >= 4:
                    score += (
                        dynamic_likelihood_weight
                        * _legacy_student_t_mean_log_score(
                            np.diff(predicted_prefix, axis=1)
                            - np.diff(observed_prefix, axis=0)[None],
                            valid_prefix[1:] & valid_prefix[:-1],
                            scale_m=observation_scale_m,
                            degrees_of_freedom=degrees_of_freedom,
                            reduction_axes=(1, 2, 3),
                        )
                    )
        else:
            if normalized_config is None:
                raise RuntimeError("normalized dense configuration was not initialized")
            score = _normalized_student_t_mean_log_score(
                predicted_prefix[:, 1:] - observed_prefix[None, 1:],
                valid_prefix[1:],
                scale_m=position_scale,
                degrees_of_freedom=normalized_config.degrees_of_freedom,
                reduction_axes=(1, 2, 3),
            )
            if normalized_config.dynamic_likelihood_weight > 0.0:
                difference_scale = normalized_config.observation_scale_m * np.sqrt(
                    2.0 * (1.0 - normalized_config.difference_correlation)
                )
                score += (
                    normalized_config.dynamic_likelihood_weight
                    * _normalized_student_t_mean_log_score(
                        np.diff(predicted_prefix, axis=1)
                        - np.diff(observed_prefix, axis=0)[None],
                        valid_prefix[1:] & valid_prefix[:-1],
                        scale_m=difference_scale,
                        degrees_of_freedom=normalized_config.degrees_of_freedom,
                        reduction_axes=(1, 2, 3),
                    )
                )
        component_scores[start:stop] = score

    score_matrix = component_scores.reshape(hypothesis_count, particle_count)
    scaled_score = likelihood_power * score_matrix
    return _normalize_joint_log_weights(
        log_weights_from_probabilities(weights, name="base_weights") + scaled_score,
        reject_invalid_support=likelihood_semantics == "legacy_v1",
    )
