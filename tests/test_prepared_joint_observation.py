from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    joint_component_log_likelihoods,
    posterior_weights_from_joint_observation,
)
from causal4d.prepared_joint_observation import (
    posterior_weights_from_prepared_joint_observation,
    prepare_joint_observation,
    prepared_joint_component_log_likelihoods,
)


def _dense_evidence(
    *,
    factor: np.ndarray | None = None,
) -> LinearJointObservationEvidence:
    return LinearJointObservationEvidence(
        evidence_id="prepared-dense",
        values_m=np.array([0.1, -0.2, 0.05]),
        row_indices=np.array([0, 1, 2]),
        frame_indices=np.array([1, 1, 2]),
        node_indices=np.array([0, 1, 0]),
        coordinate_indices=np.array([0, 0, 1]),
        coefficients=np.ones(3),
        base_covariance_m2=np.array(
            [
                [0.04, 0.01, 0.0],
                [0.01, 0.09, 0.015],
                [0.0, 0.015, 0.06],
            ]
        ),
        shared_covariance_factor_m=factor,
        source_id="prob4d",
    )


def _components() -> np.ndarray:
    values = np.zeros((4, 3, 2, 2), dtype=float)
    values[0, 1, 0, 0] = 0.12
    values[0, 1, 1, 0] = -0.18
    values[0, 2, 0, 1] = 0.04
    values[1, 1, 0, 0] = -0.05
    values[1, 1, 1, 0] = -0.25
    values[1, 2, 0, 1] = 0.2
    values[2, 1, 0, 0] = 0.3
    values[2, 1, 1, 0] = 0.1
    values[2, 2, 0, 1] = -0.1
    values[3, 1, 0, 0] = 0.1
    values[3, 1, 1, 0] = -0.2
    values[3, 2, 0, 1] = 0.05
    return values


def _direct_score(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, logdet = np.linalg.slogdet(covariance)
    assert np.all(sign > 0.0)
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    quadratic = np.einsum("...i,...i->...", residual, solved)
    dimension = residual.shape[-1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + logdet + quadratic)


def test_prepared_dense_scores_and_posterior_match_legacy() -> None:
    evidence = _dense_evidence(
        factor=np.array([[0.05], [0.02], [-0.01]])
    )
    prepared = prepare_joint_observation(evidence)
    components = _components()
    prior = np.array([0.1, 0.2, 0.3, 0.4])

    legacy_score, _ = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )
    score, diagnostics = prepared_joint_component_log_likelihoods(
        components,
        prepared,
        prefix_frame_count=3,
        component_chunk_size=2,
    )
    legacy_posterior, _ = posterior_weights_from_joint_observation(
        prior,
        components,
        evidence,
        prefix_frame_count=3,
    )
    posterior, _ = posterior_weights_from_prepared_joint_observation(
        prior,
        components,
        prepared,
        prefix_frame_count=3,
        component_chunk_size=2,
    )

    np.testing.assert_allclose(score, legacy_score, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(
        posterior,
        legacy_posterior,
        rtol=1e-12,
        atol=1e-12,
    )
    assert diagnostics.base_factorization_reused is True
    assert diagnostics.joint.used_shared_base_factorization is True
    assert diagnostics.component_chunk_size == 2
    assert diagnostics.chunk_count == 2


def test_prepared_block_scores_match_legacy() -> None:
    dense = _dense_evidence(
        factor=np.array([[0.05], [0.02], [-0.01]])
    )
    blocks = np.stack(
        (
            dense.base_covariance_m2[:1, :1],
            dense.base_covariance_m2[1:2, 1:2],
            dense.base_covariance_m2[2:3, 2:3],
        )
    )
    evidence = replace(dense, base_covariance_m2=blocks)
    prepared = prepare_joint_observation(evidence)

    legacy_score, _ = joint_component_log_likelihoods(
        _components(),
        evidence,
        prefix_frame_count=3,
    )
    score, diagnostics = prepared_joint_component_log_likelihoods(
        _components(),
        prepared,
        prefix_frame_count=3,
        component_chunk_size=3,
    )

    np.testing.assert_allclose(score, legacy_score, rtol=1e-12, atol=1e-12)
    assert diagnostics.joint.base_covariance_representation == "block_diagonal"
    assert diagnostics.base_factorization_reused is True


def test_rank_deficient_additive_covariance_is_valid() -> None:
    evidence = _dense_evidence(
        factor=np.array([[0.05], [0.02], [-0.01]])
    )
    prepared = prepare_joint_observation(evidence)
    direction = np.array([0.02, -0.01, 0.03])
    additive = np.outer(direction, direction)

    score, diagnostics = prepared_joint_component_log_likelihoods(
        _components(),
        prepared,
        prefix_frame_count=3,
        component_joint_covariance_m2=additive,
        component_chunk_size=2,
    )
    residual = evidence.apply(_components()) - evidence.values_m
    factor = evidence.shared_covariance_factor_m
    assert factor is not None
    covariance = evidence.base_covariance_m2 + additive + factor @ factor.T
    expected = _direct_score(
        residual,
        np.broadcast_to(covariance, (len(residual), 3, 3)),
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics.joint.used_component_covariance is True
    assert diagnostics.base_factorization_reused is False


def test_indefinite_additive_covariance_fails_closed() -> None:
    prepared = prepare_joint_observation(_dense_evidence())
    indefinite = np.diag([0.01, -0.001, 0.02])

    with pytest.raises(ValueError, match="positive semidefinite"):
        prepared_joint_component_log_likelihoods(
            _components(),
            prepared,
            prefix_frame_count=3,
            component_joint_covariance_m2=indefinite,
        )


def test_prepared_operator_combines_duplicate_selectors() -> None:
    evidence = LinearJointObservationEvidence(
        evidence_id="prepared-duplicates",
        values_m=np.zeros(2),
        row_indices=np.array([0, 0, 1, 1]),
        frame_indices=np.array([1, 1, 1, 2]),
        node_indices=np.array([0, 0, 0, 0]),
        coordinate_indices=np.array([0, 0, 0, 1]),
        coefficients=np.array([1.0, 2.0, -3.0, 1.0]),
        base_covariance_m2=np.eye(2),
    )
    prepared = prepare_joint_observation(evidence)
    trajectories = np.zeros((3, 3, 1, 2))
    trajectories[:, 1, 0, 0] = np.array([0.2, -0.1, 0.4])
    trajectories[:, 2, 0, 1] = np.array([0.5, 0.3, -0.2])
    variance = np.zeros_like(trajectories)
    variance[:, 1, 0, 0] = np.array([0.1, 0.2, 0.3])
    variance[:, 2, 0, 1] = 0.4

    np.testing.assert_allclose(
        prepared.apply(trajectories),
        evidence.apply(trajectories),
    )
    np.testing.assert_allclose(
        prepared.apply_independent_covariance(variance),
        evidence.apply_independent_covariance(variance),
    )
    assert prepared.unique_selector_count == 2
    assert prepared.operator_nonzero_count == 3


def test_component_low_rank_factor_uses_cached_base() -> None:
    evidence = _dense_evidence(
        factor=np.array([[0.05], [0.02], [-0.01]])
    )
    prepared = prepare_joint_observation(evidence)
    component_factor = np.broadcast_to(
        np.array([[0.01], [-0.02], [0.015]]),
        (4, 3, 1),
    )

    legacy_score, _ = joint_component_log_likelihoods(
        _components(),
        evidence,
        prefix_frame_count=3,
        component_joint_covariance_factor_m=component_factor,
    )
    score, diagnostics = prepared_joint_component_log_likelihoods(
        _components(),
        prepared,
        prefix_frame_count=3,
        component_joint_covariance_factor_m=component_factor,
        component_chunk_size=2,
    )

    np.testing.assert_allclose(score, legacy_score, rtol=1e-12, atol=1e-12)
    assert diagnostics.base_factorization_reused is True
    assert diagnostics.joint.component_shared_rank == 1


def test_static_scoring_does_not_refactor_base(monkeypatch) -> None:
    prepared = prepare_joint_observation(
        _dense_evidence(factor=np.array([[0.05], [0.02], [-0.01]]))
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("static base was factorized again")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    score, diagnostics = prepared_joint_component_log_likelihoods(
        _components(),
        prepared,
        prefix_frame_count=3,
        component_chunk_size=2,
    )

    assert np.all(np.isfinite(score))
    assert diagnostics.base_factorization_reused is True


def test_dense_dynamic_path_respects_preallocation_budget() -> None:
    dimension = 24
    evidence = LinearJointObservationEvidence(
        evidence_id="prepared-budget",
        values_m=np.zeros(dimension),
        row_indices=np.arange(dimension),
        frame_indices=np.ones(dimension, dtype=np.int64),
        node_indices=np.arange(dimension, dtype=np.int64),
        coordinate_indices=np.zeros(dimension, dtype=np.int64),
        coefficients=np.ones(dimension),
        base_covariance_m2=np.eye(dimension),
    )
    prepared = prepare_joint_observation(evidence)
    components = np.zeros((2, 2, dimension, 1))

    with pytest.raises(MemoryError, match="maximum_working_bytes"):
        prepared_joint_component_log_likelihoods(
            components,
            prepared,
            prefix_frame_count=2,
            component_joint_covariance_m2=np.zeros((dimension, dimension)),
            maximum_working_bytes=1024,
        )


def test_prepared_posterior_preserves_zero_support() -> None:
    prepared = prepare_joint_observation(_dense_evidence())
    posterior, _ = posterior_weights_from_prepared_joint_observation(
        np.array([0.5, 0.0, 0.5, 0.0]),
        _components(),
        prepared,
        prefix_frame_count=3,
        component_chunk_size=2,
    )

    assert posterior[1] == 0.0
    assert posterior[3] == 0.0
