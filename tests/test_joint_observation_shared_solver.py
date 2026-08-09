from __future__ import annotations

import numpy as np

from causal4d.joint_observation import (
    LinearJointObservationEvidence,
    joint_component_log_likelihoods,
)


def _components(count: int, *, seed: int = 13) -> np.ndarray:
    generator = np.random.default_rng(seed)
    return generator.normal(scale=0.15, size=(count, 3, 2, 2))


def _dense_evidence() -> LinearJointObservationEvidence:
    return LinearJointObservationEvidence(
        evidence_id="shared-dense",
        values_m=np.array([0.1, -0.2, 0.05]),
        row_indices=np.arange(3),
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
        shared_covariance_factor_m=np.array(
            [
                [0.05, 0.0],
                [0.02, 0.03],
                [0.01, -0.02],
            ]
        ),
    )


def _direct_score(residual: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    sign, log_determinant = np.linalg.slogdet(covariance)
    assert sign > 0.0
    solved = np.linalg.solve(covariance, residual[..., None])[..., 0]
    quadratic = np.einsum("...i,...i->...", residual, solved)
    dimension = residual.shape[-1]
    return -0.5 * (dimension * np.log(2.0 * np.pi) + log_determinant + quadratic)


def test_dense_shared_base_is_factored_once_for_many_components(monkeypatch) -> None:
    evidence = _dense_evidence()
    components = _components(64)
    residual = evidence.apply(components) - evidence.values_m
    factor = evidence.shared_covariance_factor_m
    assert factor is not None
    covariance = evidence.base_covariance_m2 + factor @ factor.T
    expected = _direct_score(residual, covariance)

    original_cholesky = np.linalg.cholesky
    cholesky_shapes: list[tuple[int, ...]] = []

    def counted_cholesky(values: np.ndarray) -> np.ndarray:
        cholesky_shapes.append(np.asarray(values).shape)
        return original_cholesky(values)

    monkeypatch.setattr(np.linalg, "cholesky", counted_cholesky)
    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert cholesky_shapes == [(3, 3), (2, 2)]
    assert diagnostics.used_shared_base_factorization is True
    assert diagnostics.used_low_rank_path is True


def test_component_low_rank_factors_reuse_the_shared_base() -> None:
    evidence = _dense_evidence()
    components = _components(19, seed=17)
    generator = np.random.default_rng(23)
    component_factor = generator.normal(scale=0.012, size=(19, 3, 1))

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
        component_joint_covariance_factor_m=component_factor,
    )
    residual = evidence.apply(components) - evidence.values_m
    shared = evidence.shared_covariance_factor_m
    assert shared is not None
    expected = np.asarray(
        [
            _direct_score(
                residual[index],
                evidence.base_covariance_m2
                + shared @ shared.T
                + component_factor[index] @ component_factor[index].T,
            )
            for index in range(len(components))
        ]
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics.used_shared_base_factorization is True
    assert diagnostics.component_shared_rank == 1


def test_block_diagonal_shared_base_matches_materialized_covariance() -> None:
    blocks = np.array(
        [
            [[0.04, 0.01], [0.01, 0.05]],
            [[0.06, -0.005], [-0.005, 0.07]],
        ]
    )
    factor = np.array(
        [
            [0.03, 0.0],
            [0.01, 0.02],
            [-0.01, 0.015],
            [0.02, -0.01],
        ]
    )
    evidence = LinearJointObservationEvidence(
        evidence_id="shared-block",
        values_m=np.array([0.1, -0.2, 0.05, 0.12]),
        row_indices=np.arange(4),
        frame_indices=np.array([1, 1, 2, 2]),
        node_indices=np.array([0, 0, 1, 1]),
        coordinate_indices=np.array([0, 1, 0, 1]),
        coefficients=np.ones(4),
        base_covariance_m2=blocks,
        shared_covariance_factor_m=factor,
    )
    components = _components(23, seed=29)
    residual = evidence.apply(components) - evidence.values_m
    base = np.zeros((4, 4), dtype=float)
    base[:2, :2] = blocks[0]
    base[2:, 2:] = blocks[1]
    expected = _direct_score(residual, base + factor @ factor.T)

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
    )

    np.testing.assert_allclose(score, expected, rtol=1e-12, atol=1e-12)
    assert diagnostics.base_covariance_representation == "block_diagonal"
    assert diagnostics.used_shared_base_factorization is True


def test_component_specific_base_covariance_keeps_the_general_path() -> None:
    evidence = _dense_evidence()
    components = _components(11, seed=31)
    component_covariance = np.broadcast_to(
        np.eye(3) * 0.002,
        (len(components), 3, 3),
    )

    score, diagnostics = joint_component_log_likelihoods(
        components,
        evidence,
        prefix_frame_count=3,
        component_joint_covariance_m2=component_covariance,
    )

    assert np.all(np.isfinite(score))
    assert diagnostics.used_component_covariance is True
    assert diagnostics.used_shared_base_factorization is False
