from __future__ import annotations

import numpy as np
import pytest

import causal4d.grouped_likelihood as grouped
import causal4d.joint_observation as joint
from causal4d.low_rank_numerics import nonnegative_woodbury_quadratic


def test_guard_clips_roundoff_but_rejects_material_cancellation() -> None:
    result = nonnegative_woodbury_quadratic(
        np.array([1.0, 1.0e6]),
        np.array([1.0 + 1.0e-15, 1.0e6 - 1.0]),
        dimension=8,
        name="test quadratic",
    )
    np.testing.assert_allclose(result, np.array([0.0, 1.0]))

    with pytest.raises(FloatingPointError, match="negative beyond roundoff"):
        nonnegative_woodbury_quadratic(
            np.array([1.0]),
            np.array([1.0 + 1.0e-6]),
            dimension=8,
            name="test quadratic",
        )


@pytest.mark.parametrize(
    ("base", "correction", "dimension", "match"),
    [
        (np.array([np.nan]), np.array([0.0]), 1, "must be finite"),
        (np.array([-1.0]), np.array([0.0]), 1, "must be nonnegative"),
        (np.array([1.0]), np.array([0.0]), 0, "positive integer"),
    ],
)
def test_guard_rejects_invalid_inputs(
    base: np.ndarray,
    correction: np.ndarray,
    dimension: int,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        nonnegative_woodbury_quadratic(
            base,
            correction,
            dimension=dimension,
        )


def _joint_evidence() -> joint.LinearJointObservationEvidence:
    return joint.LinearJointObservationEvidence(
        evidence_id="low-rank-guard",
        values_m=np.zeros(3),
        row_indices=np.arange(3),
        frame_indices=np.ones(3, dtype=int),
        node_indices=np.arange(3),
        coordinate_indices=np.zeros(3, dtype=int),
        coefficients=np.ones(3),
        base_covariance_m2=np.eye(3) * 0.2,
        shared_covariance_factor_m=np.array([[0.10], [0.04], [-0.03]]),
    )


def test_every_joint_low_rank_path_uses_the_shared_guard(monkeypatch) -> None:
    calls: list[tuple[int, str]] = []
    original = joint.nonnegative_woodbury_quadratic

    def spy(
        base: np.ndarray,
        correction: np.ndarray,
        *,
        dimension: int,
        name: str,
    ) -> np.ndarray:
        calls.append((dimension, name))
        return original(
            base,
            correction,
            dimension=dimension,
            name=name,
        )

    monkeypatch.setattr(joint, "nonnegative_woodbury_quadratic", spy)
    residual = np.array(
        [[0.2, -0.1, 0.05], [-0.03, 0.07, 0.11]],
        dtype=float,
    )
    factor = np.array([[0.10], [0.02], [-0.04]])
    joint._joint_gaussian_log_density_dense(residual, np.eye(3), factor)
    joint._joint_gaussian_log_density_blocks(
        residual,
        np.ones((3, 1, 1)),
        factor,
    )

    evidence = _joint_evidence()
    shared_solver = joint._prepare_joint_gaussian_base_solver(
        evidence,
        precompute_shared_low_rank=True,
    )
    shared_solver.log_density(residual)
    component_solver = joint._prepare_joint_gaussian_base_solver(
        evidence,
        precompute_shared_low_rank=False,
    )
    component_solver.log_density(
        residual,
        component_covariance_factor_m=np.broadcast_to(
            factor,
            (len(residual), *factor.shape),
        ),
    )

    assert len(calls) == 4
    assert all(dimension == 3 for dimension, _ in calls)


def test_grouped_student_t_low_rank_path_uses_the_shared_guard(
    monkeypatch,
) -> None:
    calls: list[tuple[int, str]] = []
    original = grouped.nonnegative_woodbury_quadratic

    def spy(
        base: np.ndarray,
        correction: np.ndarray,
        *,
        dimension: int,
        name: str,
    ) -> np.ndarray:
        calls.append((dimension, name))
        return original(
            base,
            correction,
            dimension=dimension,
            name=name,
        )

    monkeypatch.setattr(grouped, "nonnegative_woodbury_quadratic", spy)
    grouped._multivariate_student_t_log_density_low_rank(
        np.array([[0.1, -0.2, 0.05]]),
        np.eye(3) * 0.3,
        np.array([[0.04], [0.02], [-0.01]]),
        degrees_of_freedom=7.0,
    )
    assert calls == [(3, "grouped Student-t Woodbury quadratic")]
