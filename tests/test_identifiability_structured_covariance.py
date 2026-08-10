import numpy as np
import pytest

from causal4d.identifiability import (
    IdentifiabilityConfig,
    InterventionIdentifiabilityResult,
    assess_intervention_identifiability,
)


def _comparison_config() -> IdentifiabilityConfig:
    return IdentifiabilityConfig(
        minimum_information_eigenvalue=1.0e-12,
        maximum_condition_number=1.0e12,
        minimum_residualized_response_fraction=0.0,
        maximum_subspace_cosine=1.0,
        maximum_query_null_response_fraction=1.0,
    )


def _assert_equivalent(
    left: InterventionIdentifiabilityResult,
    right: InterventionIdentifiabilityResult,
) -> None:
    assert np.allclose(
        left.conditional_information,
        right.conditional_information,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert np.allclose(left.eigenvalues, right.eigenvalues, atol=1.0e-10, rtol=1.0e-10)
    assert np.allclose(
        left.identified_projection,
        right.identified_projection,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert left.effective_rank == right.effective_rank
    assert left.failure_reasons == right.failure_reasons
    assert left.query_identifiable == right.query_identifiable
    assert left.query_null_response_fraction == pytest.approx(
        right.query_null_response_fraction,
        abs=1.0e-10,
        rel=1.0e-10,
    )
    assert left.residualized_response_fraction == pytest.approx(
        right.residualized_response_fraction,
        abs=1.0e-10,
        rel=1.0e-10,
    )
    assert left.maximum_subspace_cosine == pytest.approx(
        right.maximum_subspace_cosine,
        abs=1.0e-10,
        rel=1.0e-10,
    )


def test_diagonal_low_rank_covariance_matches_dense_total() -> None:
    rng = np.random.default_rng(6114)
    intervention = rng.normal(size=(12, 3))
    nuisance = rng.normal(size=(12, 2))
    query = rng.normal(size=(4, 3))
    diagonal = np.linspace(0.2, 1.3, len(intervention))
    factor = rng.normal(scale=0.15, size=(len(intervention), 3))

    structured = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=diagonal,
        covariance_factor=factor,
        query_sensitivity=query,
        config=_comparison_config(),
    )
    dense = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=np.diag(diagonal) + factor @ factor.T,
        query_sensitivity=query,
        config=_comparison_config(),
    )

    _assert_equivalent(structured, dense)


def test_diagonal_vector_matches_dense_diagonal_covariance() -> None:
    rng = np.random.default_rng(7892)
    intervention = rng.normal(size=(10, 2))
    nuisance = rng.normal(size=(10, 1))
    diagonal = np.linspace(0.05, 0.5, len(intervention))

    vector = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=diagonal,
        config=_comparison_config(),
    )
    dense = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=np.diag(diagonal),
        config=_comparison_config(),
    )

    assert np.allclose(vector.conditional_information, dense.conditional_information)
    assert np.allclose(vector.identified_projection, dense.identified_projection)


def test_low_rank_covariance_is_invariant_to_factor_rotation() -> None:
    rng = np.random.default_rng(8445)
    intervention = rng.normal(size=(16, 3))
    nuisance = rng.normal(size=(16, 2))
    diagonal = np.linspace(0.1, 0.9, len(intervention))
    factor = rng.normal(scale=0.1, size=(len(intervention), 2))
    angle = 0.73
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)],
        ]
    )

    reference = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=diagonal,
        covariance_factor=factor,
        config=_comparison_config(),
    )
    rotated = assess_intervention_identifiability(
        intervention,
        nuisance,
        covariance=diagonal,
        covariance_factor=factor @ rotation,
        config=_comparison_config(),
    )

    assert np.allclose(
        reference.conditional_information,
        rotated.conditional_information,
        atol=1.0e-10,
        rtol=1.0e-10,
    )
    assert np.allclose(
        reference.identified_projection,
        rotated.identified_projection,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_large_diagonal_low_rank_path_does_not_call_dense_cholesky(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(9017)
    response_count = 2048
    intervention = rng.normal(size=(response_count, 2))
    factor = rng.normal(scale=0.02, size=(response_count, 4))

    def forbidden_cholesky(_: np.ndarray) -> np.ndarray:
        raise AssertionError("diagonal structured whitening formed a dense covariance")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden_cholesky)
    result = assess_intervention_identifiability(
        intervention,
        covariance=np.full(response_count, 0.2),
        covariance_factor=factor,
        config=_comparison_config(),
    )

    assert result.conditional_information.shape == (2, 2)
    assert np.all(np.isfinite(result.conditional_information))


def test_covariance_factor_requires_base_covariance() -> None:
    with pytest.raises(ValueError, match="requires a base covariance"):
        assess_intervention_identifiability(
            np.eye(4, 2),
            covariance_factor=np.ones((4, 1)),
        )


@pytest.mark.parametrize(
    "diagonal",
    [
        np.asarray([1.0, 1.0, 0.0, 1.0]),
        np.asarray([1.0, 1.0, -1.0, 1.0]),
        np.asarray([1.0, 1.0, np.nan, 1.0]),
    ],
)
def test_invalid_diagonal_covariance_fails_closed(diagonal: np.ndarray) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        assess_intervention_identifiability(
            np.eye(4, 2),
            covariance=diagonal,
        )


def test_invalid_covariance_factor_fails_closed() -> None:
    with pytest.raises(ValueError, match="positive rank"):
        assess_intervention_identifiability(
            np.eye(4, 2),
            covariance=np.ones(4),
            covariance_factor=np.empty((4, 0)),
        )
    with pytest.raises(ValueError, match="must be finite"):
        assess_intervention_identifiability(
            np.eye(4, 2),
            covariance=np.ones(4),
            covariance_factor=np.asarray([[0.0], [0.0], [np.nan], [0.0]]),
        )
