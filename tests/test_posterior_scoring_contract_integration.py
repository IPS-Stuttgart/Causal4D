from __future__ import annotations

import hashlib

import numpy as np
import pytest

from causal4d.contracts import (
    PhysicalPosterior,
    TaskPosterior,
    build_causal_context,
)
from causal4d.posterior_scoring import (
    TrajectoryScoreSpecificationV1,
    score_physical_posterior,
    trajectory_coordinate_index,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _posterior_pair() -> tuple[PhysicalPosterior, TaskPosterior]:
    observations = np.zeros((8, 1, 3), dtype=np.float64)
    actions = np.zeros((8, 1, 3), dtype=np.float64)
    context = build_causal_context(
        protocol_id="posterior-score-integration-v1",
        case_id="case-001",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )
    trajectories = np.zeros((2, 4, 1, 3), dtype=np.float64)
    trajectories[0, :, 0, :2] = -1.0
    trajectories[1, :, 0, :2] = 1.0
    physical = PhysicalPosterior(
        context=context,
        component_ids=("negative", "positive"),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((2, 1, 3), 1.0e-6),
        weights=np.asarray([0.6, 0.4]),
        phi=np.asarray([[1.0], [0.8]]),
        kappa_cf=np.asarray([[0.0], [0.1]]),
        hypothesis_indices=np.asarray([0, 1]),
        twin_particle_indices=np.asarray([0, 1]),
        phi_names=("gain",),
        kappa_names=("slip",),
        source_twin_belief_id=_digest("belief"),
        source_factual_intervention_id=_digest("factual"),
        source_query_id=_digest("query"),
    )
    task = TaskPosterior(
        context=context,
        physical_posterior_id=physical.artifact_id,
        component_ids=physical.component_ids,
        physical_weights=physical.weights,
        task_weights=np.asarray([0.2, 0.8]),
        semantic_log_scores=np.asarray([-1.0, 1.0]),
        beta=1.0,
        query_node_indices=np.asarray([0]),
        semantic_source="integration-test",
    )
    return physical, task


def test_real_contracts_preserve_lineage_and_joint_covariance() -> None:
    physical, task = _posterior_pair()
    shape = physical.readout_trajectories_m.shape[1:]
    first = trajectory_coordinate_index(0, 0, 0, shape)
    second = trajectory_coordinate_index(1, 0, 0, shape)
    final_x = trajectory_coordinate_index(3, 0, 0, shape)
    final_y = trajectory_coordinate_index(3, 0, 1, shape)
    query = np.zeros((2, int(np.prod(shape))), dtype=float)
    query[0, final_x] = 1.0
    query[1, final_y] = 1.0
    specification = TrajectoryScoreSpecificationV1(
        name="real-contract-integration-v1",
        valid_mask=np.ones(shape[:-1], dtype=bool),
        variogram_pairs=np.asarray([[first, second]], dtype=np.int64),
        variogram_pair_weights=np.asarray([1.0]),
        query_matrix=query,
        query_labels=("final-x", "final-y"),
        query_units=("m", "m"),
        query_covariance_floor_m2=1.0e-10,
    )
    truth = np.ones(shape, dtype=float)

    physical_score = score_physical_posterior(
        physical,
        truth,
        specification,
        conditional_draws_per_component=4,
        random_seed=17,
    )
    repeated = score_physical_posterior(
        physical,
        truth,
        specification,
        conditional_draws_per_component=4,
        random_seed=17,
    )
    task_score = score_physical_posterior(
        physical,
        truth,
        specification,
        task_posterior=task,
        conditional_draws_per_component=4,
        random_seed=17,
    )

    assert physical_score.score_id == repeated.score_id
    assert physical_score.source_weight_kind == "physical"
    assert physical_score.source_weight_id == physical.artifact_id
    assert task_score.source_weight_kind == "task"
    assert task_score.source_weight_id == task.artifact_id
    assert physical_score.score_id != task_score.score_id
    assert physical_score.query_score is not None
    assert physical_score.query_score.posterior_covariance[0, 1] > 0.0
    contribution_sum = sum(
        value.mean_coordinate_variance_m2
        for value in physical_score.variance_attribution.contributions
    )
    assert contribution_sum == pytest.approx(
        physical_score.variance_attribution.total_mean_coordinate_variance_m2
    )
