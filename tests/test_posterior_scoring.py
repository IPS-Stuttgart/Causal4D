from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np
import pytest

from causal4d.posterior_scoring import (
    TrajectoryScoreSpecificationV1,
    gaussian_log_score,
    ordered_variance_attribution,
    score_physical_posterior,
    trajectory_coordinate_index,
    weighted_energy_score,
    weighted_variogram_score,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


@dataclass
class Posterior:
    contract_type = "PhysicalPosterior"
    readout_trajectories_m: np.ndarray
    readout_variance_m2: np.ndarray
    weights: np.ndarray
    component_ids: tuple[str, ...]
    twin_particle_indices: np.ndarray
    phi: np.ndarray
    hypothesis_indices: np.ndarray
    kappa_cf: np.ndarray
    artifact_id: str


def test_energy_and_variogram_detect_dependence() -> None:
    correlated = np.asarray([[-1.0, -1.0], [1.0, 1.0]])
    anticorrelated = np.asarray([[-1.0, 1.0], [1.0, -1.0]])
    weights = np.asarray([0.5, 0.5])
    truth = np.asarray([1.0, 1.0])
    assert weighted_energy_score(correlated, weights, truth) < weighted_energy_score(
        anticorrelated, weights, truth
    )
    assert weighted_variogram_score(
        correlated,
        weights,
        truth,
        np.asarray([[0, 1]]),
        np.asarray([1.0]),
    ) == pytest.approx(0.0)
    assert weighted_variogram_score(
        anticorrelated,
        weights,
        truth,
        np.asarray([[0, 1]]),
        np.asarray([1.0]),
    ) > 0.0


def test_gaussian_log_score_uses_full_covariance() -> None:
    score = gaussian_log_score(
        [0.0, 0.0],
        [[1.0, 0.8], [0.8, 1.0]],
        [1.0, 1.0],
        labels=("x", "y"),
    )
    diagonal = gaussian_log_score(
        [0.0, 0.0],
        [[1.0, 0.0], [0.0, 1.0]],
        [1.0, 1.0],
        labels=("x", "y"),
    )
    assert score.log_score != pytest.approx(diagonal.log_score)
    assert score.posterior_covariance[0, 1] == pytest.approx(0.8)


def test_ordered_attribution_is_nonnegative_and_additive() -> None:
    means = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    variances = np.full_like(means, 0.25)
    weights = np.full(4, 0.25)
    result = ordered_variance_attribution(
        means,
        variances,
        weights,
        (
            ("particle", np.asarray([0, 0, 1, 1])),
            ("event", np.asarray([0, 1, 0, 1])),
        ),
    )
    values = [value.mean_coordinate_variance_m2 for value in result.contributions]
    assert all(value >= 0.0 for value in values)
    assert sum(values) == pytest.approx(result.total_mean_coordinate_variance_m2)
    assert result.between_component_variance_m2 == pytest.approx(1.25)
    assert result.conditional_readout_variance_m2 == pytest.approx(0.25)


def test_score_physical_posterior_binds_joint_scores_and_attribution() -> None:
    trajectories = np.asarray(
        [
            [[[-1.0, -1.0, 0.0]]],
            [[[1.0, 1.0, 0.0]]],
        ],
        dtype=float,
    )
    posterior = Posterior(
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.zeros((2, 1, 3)),
        weights=np.asarray([0.5, 0.5]),
        component_ids=("left", "right"),
        twin_particle_indices=np.asarray([0, 1]),
        phi=np.asarray([[0.0], [0.0]]),
        hypothesis_indices=np.asarray([0, 1]),
        kappa_cf=np.asarray([[0.0], [1.0]]),
        artifact_id=_digest("posterior"),
    )
    truth = np.asarray([[[1.0, 1.0, 0.0]]])
    query = np.zeros((1, truth.size))
    query[0, 0] = 0.5
    query[0, 1] = 0.5
    specification = TrajectoryScoreSpecificationV1(
        name="joint-score",
        valid_mask=np.ones((1, 1), dtype=bool),
        variogram_pairs=np.asarray([[0, 1]], dtype=np.int64),
        variogram_pair_weights=np.asarray([1.0]),
        query_matrix=query,
        query_labels=("mean-xy",),
        query_covariance_floor_m2=1e-8,
    )
    result = score_physical_posterior(
        posterior,
        truth,
        specification,
        conditional_draws_per_component=2,
        random_seed=5,
    )
    assert result.exact_component_variogram_score == pytest.approx(0.0)
    assert result.sampled_mixture_energy_score_m == pytest.approx(
        result.exact_component_energy_score_m
    )
    assert result.query_score is not None
    assert result.source_posterior_id == posterior.artifact_id
    assert len(result.score_id) == 64
    assert not result.variance_attribution.contributions[-1].name.startswith("physical")


def test_variogram_pairs_must_be_inside_registered_mask() -> None:
    trajectories = np.zeros((1, 1, 1, 3))
    posterior = Posterior(
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.zeros((1, 1, 3)),
        weights=np.asarray([1.0]),
        component_ids=("only",),
        twin_particle_indices=np.asarray([0]),
        phi=np.zeros((1, 0)),
        hypothesis_indices=np.asarray([0]),
        kappa_cf=np.zeros((1, 0)),
        artifact_id=_digest("only"),
    )
    specification = TrajectoryScoreSpecificationV1(
        name="masked",
        valid_mask=np.asarray([[[True, False, False]]]),
        variogram_pairs=np.asarray([[0, 1]], dtype=np.int64),
        variogram_pair_weights=np.asarray([1.0]),
    )
    with pytest.raises(ValueError, match="inside valid_mask"):
        score_physical_posterior(posterior, np.zeros((1, 1, 3)), specification)


@dataclass
class TaskPosterior:
    contract_type = "TaskPosterior"
    physical_posterior_id: str
    component_ids: tuple[str, ...]
    physical_weights: np.ndarray
    task_weights: np.ndarray
    artifact_id: str


def test_task_weights_are_bound_without_changing_physical_weights() -> None:
    posterior = Posterior(
        readout_trajectories_m=np.asarray(
            [
                [[[-1.0, 0.0, 0.0]]],
                [[[1.0, 0.0, 0.0]]],
            ]
        ),
        readout_variance_m2=np.zeros((2, 1, 3)),
        weights=np.asarray([0.75, 0.25]),
        component_ids=("left", "right"),
        twin_particle_indices=np.asarray([0, 1]),
        phi=np.zeros((2, 0)),
        hypothesis_indices=np.asarray([0, 1]),
        kappa_cf=np.zeros((2, 0)),
        artifact_id=_digest("physical"),
    )
    task = TaskPosterior(
        physical_posterior_id=posterior.artifact_id,
        component_ids=posterior.component_ids,
        physical_weights=posterior.weights.copy(),
        task_weights=np.asarray([0.1, 0.9]),
        artifact_id=_digest("task"),
    )
    result = score_physical_posterior(
        posterior,
        np.zeros((1, 1, 3)),
        TrajectoryScoreSpecificationV1(
            name="task-score",
            valid_mask=np.ones((1, 1), dtype=bool),
        ),
        task_posterior=task,
    )
    assert result.source_weight_kind == "task"
    assert result.source_weight_id == task.artifact_id
    assert result.effective_component_count == pytest.approx(
        1.0 / np.sum(np.square(task.task_weights))
    )

    task.physical_weights[0] = 0.5
    with pytest.raises(ValueError, match="physical weights"):
        score_physical_posterior(
            posterior,
            np.zeros((1, 1, 3)),
            TrajectoryScoreSpecificationV1(
                name="changed-physical",
                valid_mask=np.ones((1, 1), dtype=bool),
            ),
            task_posterior=task,
        )


def test_registered_specification_rejects_ambiguous_pairs_and_coercion() -> None:
    with pytest.raises(ValueError, match="exact Boolean"):
        TrajectoryScoreSpecificationV1(
            name="coerced-mask",
            valid_mask=np.ones((1, 1), dtype=np.int64),
        )
    with pytest.raises(ValueError, match="left < right"):
        TrajectoryScoreSpecificationV1(
            name="reversed-pair",
            valid_mask=np.ones((1, 1), dtype=bool),
            variogram_pairs=np.asarray([[1, 0]], dtype=np.int64),
            variogram_pair_weights=np.asarray([1.0]),
        )
    with pytest.raises(ValueError, match="unique"):
        TrajectoryScoreSpecificationV1(
            name="duplicate-pair",
            valid_mask=np.ones((1, 1), dtype=bool),
            variogram_pairs=np.asarray([[0, 1], [0, 1]], dtype=np.int64),
            variogram_pair_weights=np.asarray([0.5, 0.5]),
        )


def test_gaussian_query_score_rejects_tampered_derived_statistics() -> None:
    score = gaussian_log_score(
        [0.0],
        [[1.0]],
        [0.5],
        labels=("x",),
        units=("m",),
    )
    with pytest.raises(ValueError, match="disagree"):
        type(score)(
            labels=score.labels,
            units=score.units,
            posterior_mean=score.posterior_mean,
            posterior_covariance=score.posterior_covariance,
            truth_query=score.truth_query,
            covariance_floor_m2=score.covariance_floor_m2,
            mahalanobis_squared=score.mahalanobis_squared,
            log_determinant=score.log_determinant,
            log_score=score.log_score + 1.0,
        )


def test_trajectory_coordinate_index_is_canonical() -> None:
    assert trajectory_coordinate_index(1, 2, 0, (3, 4, 3)) == 18
    with pytest.raises(ValueError, match="coordinate"):
        trajectory_coordinate_index(1, 2, 3, (3, 4, 3))
