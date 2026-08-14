from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pytest

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.interventional_contrast import (
    InterventionalContrastQueryV1,
    build_interventional_contrast,
)
from causal4d.interventional_cross_covariance import (
    InterventionalCrossCovarianceV1,
    build_interventional_cross_covariance,
)


TWIN_ID = "1" * 64
FACTUAL_ID = "2" * 64


def _context(action_id: str, action_scale: float):
    observations = np.arange(18, dtype=float).reshape(6, 1, 3)
    observed_actions = np.zeros((6, 1, 3), dtype=float)
    counterfactual_actions = observed_actions.copy()
    counterfactual_actions[2:, 0, 0] = action_scale
    return build_causal_context(
        protocol_id="cross-covariance-unit-protocol",
        case_id="cross-covariance-unit-case",
        observations=observations,
        observed_actions=observed_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=2,
        counterfactual_action_id=action_id,
    )


def _posterior(
    action_id: str,
    final_x_m: tuple[float, ...],
    *,
    action_scale: float,
    variance_m2: float,
    metadata: dict[str, Any] | None = None,
) -> PhysicalPosterior:
    count = len(final_x_m)
    trajectories = np.zeros((count, 4, 1, 3), dtype=float)
    trajectories[:, -1, 0, 0] = final_x_m
    return PhysicalPosterior(
        context=_context(action_id, action_scale),
        component_ids=tuple(f"component-{index}" for index in range(count)),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((count, 1, 3), variance_m2),
        weights=np.asarray([0.75, 0.25]),
        phi=np.ones((count, 1), dtype=float),
        kappa_cf=np.column_stack((np.arange(count), np.zeros(count))),
        hypothesis_indices=np.arange(count, dtype=np.int64),
        twin_particle_indices=np.zeros(count, dtype=np.int64),
        phi_names=("gain",),
        kappa_names=("contact_patch", "slip"),
        source_twin_belief_id=TWIN_ID,
        source_factual_intervention_id=FACTUAL_ID,
        source_query_id=hashlib.sha256(action_id.encode()).hexdigest(),
        metadata=metadata or {},
    )


def _query() -> InterventionalContrastQueryV1:
    matrix = np.zeros((1, 12), dtype=float)
    matrix[0, 9] = 1.0
    return InterventionalContrastQueryV1(
        name="final-node-0-x",
        matrix=matrix,
        labels=("final-node-0-x",),
        units=("m",),
        metadata={"registered": True},
    )


def _inputs():
    branch_a = _posterior(
        "action-a",
        (2.0, 4.0),
        action_scale=1.0,
        variance_m2=0.04,
    )
    branch_b = _posterior(
        "action-b",
        (1.0, 1.0),
        action_scale=-1.0,
        variance_m2=0.01,
    )
    source = build_interventional_contrast(
        branch_a,
        branch_b,
        _query(),
        branch_a_label="do(action-a)",
        branch_b_label="do(action-b)",
        conditional_variance_policy="independent_readout",
    )
    return branch_a, branch_b, source


def test_omitted_covariance_returns_exact_source_object() -> None:
    branch_a, branch_b, source = _inputs()
    result = build_interventional_cross_covariance(source, branch_a, branch_b)
    assert result is source


def test_positive_cross_covariance_reduces_uncertainty() -> None:
    branch_a, branch_b, source = _inputs()
    result = build_interventional_cross_covariance(
        source,
        branch_a,
        branch_b,
        cross_branch_conditional_covariance=np.full((2, 1, 1), 0.015),
        cross_covariance_model_id=hashlib.sha256(b"source-fit-v1").hexdigest(),
    )
    assert isinstance(result, InterventionalCrossCovarianceV1)
    np.testing.assert_allclose(result.conditional_covariance[:, 0, 0], 0.02)
    np.testing.assert_allclose(result.covariance, [[0.77]])
    assert result.metadata["future_observations_read"] == 0
    assert result.metadata["target_outcomes_used_for_covariance"] is False


def test_invalid_joint_covariance_fails_closed() -> None:
    branch_a, branch_b, source = _inputs()
    with pytest.raises(ValueError, match="joint branch covariance"):
        build_interventional_cross_covariance(
            source,
            branch_a,
            branch_b,
            cross_branch_conditional_covariance=np.full((2, 1, 1), 0.03),
            cross_covariance_model_id=hashlib.sha256(b"invalid-v1").hexdigest(),
        )


def test_covariance_requires_content_addressed_model_and_source_identity() -> None:
    branch_a, branch_b, source = _inputs()
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_interventional_cross_covariance(
            source,
            branch_a,
            branch_b,
            cross_branch_conditional_covariance=np.zeros((2, 1, 1)),
        )
    wrong_branch = _posterior(
        "action-other",
        (2.0, 4.0),
        action_scale=1.0,
        variance_m2=0.04,
    )
    with pytest.raises(ValueError, match="branch_a"):
        build_interventional_cross_covariance(
            source,
            wrong_branch,
            branch_b,
            cross_branch_conditional_covariance=np.zeros((2, 1, 1)),
            cross_covariance_model_id="a" * 64,
        )
