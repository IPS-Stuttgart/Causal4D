"""Dependence-aware scoring for weighted Causal4D trajectory posteriors.

The registered physical experiment remains unchanged. This module provides
additive, post-freeze diagnostics that score the joint rollout distribution
instead of replacing it with independent marginal Gaussians.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from causal4d._posterior_scoring_contracts import (
    POSTERIOR_SCORE_CLAIM_BOUNDARY,
    POSTERIOR_SCORE_SCHEMA_VERSION,
    TrajectoryScoreSpecificationV1,
    _array_sha256,
    _coordinate_mask,
    _require_sha256,
    _validated_weights,
    trajectory_coordinate_index,
)
from causal4d._posterior_scoring_results import (
    GaussianQueryScoreV1,
    OrderedVarianceAttributionV1,
    TrajectoryPosteriorScoreV1,
    VarianceContributionV1,
)
from causal4d._posterior_scoring_metrics import (
    gaussian_log_score,
    multivariate_gaussian_query_score,
    ordered_variance_attribution,
    weighted_energy_score,
    weighted_variogram_score,
)

if TYPE_CHECKING:
    from causal4d.contracts import PhysicalPosterior, TaskPosterior


def _conditional_mixture_samples(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    *,
    draws_per_component: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if type(draws_per_component) is not int or draws_per_component < 1:
        raise ValueError("draws_per_component must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    component_count, dimension = means.shape
    rng = np.random.default_rng(seed)
    noise = np.zeros((component_count, draws_per_component, dimension), dtype=float)
    pair_count = draws_per_component // 2
    if pair_count:
        positive = rng.standard_normal((component_count, pair_count, dimension))
        noise[:, :pair_count] = positive
        noise[:, pair_count : 2 * pair_count] = -positive
    samples = means[:, None] + noise * np.sqrt(variances)[:, None]
    sample_weights = np.repeat(weights / draws_per_component, draws_per_component)
    return (
        samples.reshape(component_count * draws_per_component, dimension),
        sample_weights,
    )


def _resolve_weights(
    posterior: PhysicalPosterior,
    task_posterior: TaskPosterior | None,
) -> tuple[np.ndarray, str, str]:
    posterior_id = _require_sha256(posterior.artifact_id, name="posterior artifact_id")
    posterior_weights = _validated_weights(
        posterior.weights,
        expected_count=len(posterior.readout_trajectories_m),
    )
    if task_posterior is None:
        return posterior_weights, "physical", posterior_id
    if task_posterior.physical_posterior_id != posterior_id:
        raise ValueError("task posterior references a different physical posterior")
    if tuple(task_posterior.component_ids) != tuple(posterior.component_ids):
        raise ValueError("task and physical posterior component identities differ")
    if not np.array_equal(task_posterior.physical_weights, posterior_weights):
        raise ValueError("task posterior changed the physical weights")
    task_weights = _validated_weights(
        task_posterior.task_weights,
        expected_count=len(posterior_weights),
    )
    task_id = _require_sha256(task_posterior.artifact_id, name="task artifact_id")
    return task_weights, "task", task_id


def _physical_groupings(
    posterior: PhysicalPosterior,
) -> tuple[tuple[str, Any], ...]:
    groupings: list[tuple[str, Any]] = [
        ("physical_twin_particle", posterior.twin_particle_indices),
    ]
    phi = np.asarray(posterior.phi)
    if phi.ndim != 2:
        raise ValueError("posterior phi must have shape (K, P)")
    if phi.shape[1]:
        groupings.append(("persistent_phi", phi))
    groupings.append(("contact_hypothesis", posterior.hypothesis_indices))
    kappa = np.asarray(posterior.kappa_cf)
    if kappa.ndim != 2:
        raise ValueError("posterior kappa_cf must have shape (K, P)")
    if kappa.shape[1]:
        groupings.append(("execution_kappa", kappa))
    return tuple(groupings)


def score_physical_posterior(
    posterior: PhysicalPosterior,
    truth_m: Any,
    specification: TrajectoryScoreSpecificationV1,
    *,
    task_posterior: TaskPosterior | None = None,
    conditional_draws_per_component: int = 2,
    random_seed: int = 20260810,
    sample_chunk_size: int = 16,
    coordinate_chunk_size: int = 4096,
) -> TrajectoryPosteriorScoreV1:
    """Score one PhysicalPosterior without altering its support or weights."""

    if getattr(posterior, "contract_type", None) != "PhysicalPosterior":
        raise TypeError("posterior must be a PhysicalPosterior")
    if task_posterior is not None and (
        getattr(task_posterior, "contract_type", None) != "TaskPosterior"
    ):
        raise TypeError("task_posterior must be a TaskPosterior")
    if not isinstance(specification, TrajectoryScoreSpecificationV1):
        raise TypeError("specification has the wrong type")
    trajectories = np.asarray(posterior.readout_trajectories_m, dtype=float)
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("posterior readout trajectories must have shape (K, T, N, 3)")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("posterior readout trajectories must be finite")
    truth = np.asarray(truth_m, dtype=float)
    if truth.shape != trajectories.shape[1:] or not np.all(np.isfinite(truth)):
        raise ValueError("truth must be finite and match the posterior trajectory")
    component_variance = np.asarray(posterior.readout_variance_m2, dtype=float)
    if component_variance.shape != (
        len(trajectories),
        trajectories.shape[2],
        trajectories.shape[3],
    ):
        raise ValueError("posterior readout variance has the wrong shape")
    if not np.all(np.isfinite(component_variance)) or np.any(component_variance < 0.0):
        raise ValueError("posterior readout variance must be finite and nonnegative")
    weights, weight_kind, weight_id = _resolve_weights(posterior, task_posterior)

    coordinate_mask = _coordinate_mask(specification.valid_mask, truth.shape)
    selected_indices = np.flatnonzero(coordinate_mask.reshape(-1))
    full_dimension = truth.size
    selected_truth = truth.reshape(-1)[selected_indices]
    component_matrix = trajectories.reshape(len(trajectories), -1)
    selected_components = component_matrix[:, selected_indices]
    full_component_variance = np.broadcast_to(
        component_variance[:, None],
        trajectories.shape,
    ).reshape(len(trajectories), full_dimension)
    selected_variance = full_component_variance[:, selected_indices]

    exact_energy = weighted_energy_score(
        selected_components,
        weights,
        selected_truth,
        sample_chunk_size=sample_chunk_size,
        coordinate_chunk_size=coordinate_chunk_size,
    )
    mixture_samples, mixture_weights = _conditional_mixture_samples(
        selected_components,
        selected_variance,
        weights,
        draws_per_component=conditional_draws_per_component,
        seed=random_seed,
    )
    sampled_energy = weighted_energy_score(
        mixture_samples,
        mixture_weights,
        selected_truth,
        sample_chunk_size=sample_chunk_size,
        coordinate_chunk_size=coordinate_chunk_size,
    )

    exact_variogram = None
    sampled_variogram = None
    if len(specification.variogram_pairs):
        if np.any(specification.variogram_pairs >= full_dimension):
            raise ValueError("registered variogram pair lies outside the trajectory")
        inverse = np.full(full_dimension, -1, dtype=np.int64)
        inverse[selected_indices] = np.arange(len(selected_indices), dtype=np.int64)
        selected_pairs = inverse[specification.variogram_pairs]
        if np.any(selected_pairs < 0):
            raise ValueError("registered variogram pairs must lie inside valid_mask")
        exact_variogram = weighted_variogram_score(
            selected_components,
            weights,
            selected_truth,
            selected_pairs,
            specification.variogram_pair_weights,
            order=specification.variogram_order,
        )
        sampled_variogram = weighted_variogram_score(
            mixture_samples,
            mixture_weights,
            selected_truth,
            selected_pairs,
            specification.variogram_pair_weights,
            order=specification.variogram_order,
        )

    attribution = ordered_variance_attribution(
        selected_components,
        selected_variance,
        weights,
        _physical_groupings(posterior),
    )

    query_score = None
    if specification.query_matrix is not None:
        if specification.query_matrix.shape[1] != full_dimension:
            raise ValueError(
                "registered query matrix has the wrong trajectory dimension"
            )
        query_score = multivariate_gaussian_query_score(
            component_matrix,
            full_component_variance,
            weights,
            truth.reshape(-1),
            specification.query_matrix,
            labels=specification.query_labels,
            units=specification.query_units,
            covariance_floor_m2=specification.query_covariance_floor_m2,
        )

    posterior_id = _require_sha256(posterior.artifact_id, name="posterior artifact_id")
    return TrajectoryPosteriorScoreV1(
        source_posterior_id=posterior_id,
        source_weight_kind=weight_kind,
        source_weight_id=weight_id,
        specification_id=specification.specification_id,
        truth_sha256=_array_sha256(truth),
        exact_component_energy_score_m=exact_energy,
        sampled_mixture_energy_score_m=sampled_energy,
        exact_component_variogram_score=exact_variogram,
        sampled_mixture_variogram_score=sampled_variogram,
        variogram_order=specification.variogram_order,
        component_count=len(weights),
        selected_coordinate_count=len(selected_indices),
        variogram_pair_count=len(specification.variogram_pairs),
        conditional_draws_per_component=conditional_draws_per_component,
        random_seed=random_seed,
        effective_component_count=float(1.0 / np.sum(np.square(weights))),
        variance_attribution=attribution,
        query_score=query_score,
    )


__all__ = [
    "POSTERIOR_SCORE_CLAIM_BOUNDARY",
    "POSTERIOR_SCORE_SCHEMA_VERSION",
    "GaussianQueryScoreV1",
    "OrderedVarianceAttributionV1",
    "TrajectoryPosteriorScoreV1",
    "TrajectoryScoreSpecificationV1",
    "VarianceContributionV1",
    "gaussian_log_score",
    "multivariate_gaussian_query_score",
    "ordered_variance_attribution",
    "score_physical_posterior",
    "trajectory_coordinate_index",
    "weighted_energy_score",
    "weighted_variogram_score",
]
