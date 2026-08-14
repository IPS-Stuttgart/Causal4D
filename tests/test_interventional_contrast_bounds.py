from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pytest

from causal4d.interventional_contrast import (
    InterventionalContrastPosteriorV1,
    build_interventional_contrast_bounds,
    load_interventional_contrast_bounds,
    save_interventional_contrast_bounds,
)


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def _independent_product_posterior(
    *,
    conditional_variance: float = 0.0,
) -> InterventionalContrastPosteriorV1:
    variance_policy = (
        "component_means_only" if conditional_variance == 0.0 else "independent_readout"
    )
    return InterventionalContrastPosteriorV1(
        source_branch_a_posterior_id=_digest("branch-a"),
        source_branch_b_posterior_id=_digest("branch-b"),
        source_branch_a_query_id=_digest("query-a"),
        source_branch_b_query_id=_digest("query-b"),
        branch_a_label="do(a)",
        branch_b_label="do(b)",
        trajectory_shape=(1, 1, 3),
        branch_a_component_count=2,
        branch_b_component_count=2,
        coupling_policy="independent_product",
        shared_kappa_names=(),
        conditional_variance_policy=variance_policy,
        query_name="x",
        query_matrix=np.asarray([[1.0, 0.0, 0.0]]),
        query_labels=("x",),
        query_units=("m",),
        pair_indices=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        weights=np.full(4, 0.25),
        contrast_values=np.asarray([[0.0], [-2.0], [2.0], [0.0]]),
        conditional_covariance=np.full((4, 1, 1), conditional_variance),
        query_metadata={"registered": True},
        metadata={"source": "unit-test"},
    )


def _shared_component_posterior() -> InterventionalContrastPosteriorV1:
    return InterventionalContrastPosteriorV1(
        source_branch_a_posterior_id=_digest("branch-a"),
        source_branch_b_posterior_id=_digest("branch-b"),
        source_branch_a_query_id=_digest("query-a"),
        source_branch_b_query_id=_digest("query-b"),
        branch_a_label="do(a)",
        branch_b_label="do(b)",
        trajectory_shape=(1, 1, 3),
        branch_a_component_count=2,
        branch_b_component_count=2,
        coupling_policy="shared_component",
        shared_kappa_names=(),
        conditional_variance_policy="component_means_only",
        query_name="x",
        query_matrix=np.asarray([[1.0, 0.0, 0.0]]),
        query_labels=("x",),
        query_units=("m",),
        pair_indices=np.asarray([[0, 0], [1, 1]]),
        weights=np.asarray([0.5, 0.5]),
        contrast_values=np.asarray([[1.0], [3.0]]),
        conditional_covariance=np.zeros((2, 1, 1)),
        query_metadata={"registered": True},
    )


def test_full_product_reports_sharp_coupling_sensitivity() -> None:
    result = build_interventional_contrast_bounds(
        _independent_product_posterior(),
        cdf_thresholds=(-1.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(result.mean, [0.0])
    np.testing.assert_allclose(result.variance_lower, [0.0])
    np.testing.assert_allclose(result.variance_upper, [4.0])
    np.testing.assert_allclose(result.probability_positive_lower, [0.0])
    np.testing.assert_allclose(result.probability_positive_upper, [0.5])
    np.testing.assert_allclose(result.source_probability_positive, [0.25])
    np.testing.assert_allclose(result.cdf_lower[:, 0], [0.0, 0.5, 0.5])
    np.testing.assert_allclose(result.cdf_upper[:, 0], [0.5, 1.0, 1.0])
    assert result.metadata["coordinate_extrema_may_use_different_couplings"]
    assert result.metadata["claim_boundary"]["uses_target_truth"] is False


def test_fixed_component_support_collapses_to_source_posterior() -> None:
    posterior = _shared_component_posterior()
    result = build_interventional_contrast_bounds(posterior)

    np.testing.assert_allclose(result.variance_lower, posterior.covariance.diagonal())
    np.testing.assert_allclose(result.variance_upper, posterior.covariance.diagonal())
    np.testing.assert_allclose(
        result.probability_positive_lower,
        posterior.probability_positive,
    )
    np.testing.assert_allclose(
        result.probability_positive_upper,
        posterior.probability_positive,
    )


def test_conditional_variance_is_included_in_transport_objectives() -> None:
    deterministic = build_interventional_contrast_bounds(
        _independent_product_posterior(),
    )
    uncertain = build_interventional_contrast_bounds(
        _independent_product_posterior(conditional_variance=1.0),
    )

    np.testing.assert_allclose(
        uncertain.variance_lower,
        deterministic.variance_lower + 1.0,
    )
    np.testing.assert_allclose(
        uncertain.variance_upper,
        deterministic.variance_upper + 1.0,
    )
    assert 0.0 < uncertain.probability_positive_lower[0]
    assert uncertain.probability_positive_upper[0] < 1.0


def test_bounds_preserve_registered_cross_branch_variance_policy() -> None:
    posterior = replace(
        _independent_product_posterior(conditional_variance=1.0),
        conditional_variance_policy="registered_cross_branch",
    )

    result = build_interventional_contrast_bounds(posterior)

    assert result.conditional_variance_policy == "registered_cross_branch"
    np.testing.assert_allclose(result.source_variance, posterior.covariance.diagonal())


def test_bound_artifact_round_trip_is_content_addressed(tmp_path: Path) -> None:
    result = build_interventional_contrast_bounds(
        _independent_product_posterior(),
        cdf_thresholds=(-1.0, 0.0, 1.0),
        metadata={"registered_before_target_access": True},
    )
    path = tmp_path / "contrast-bounds.npz"
    save_interventional_contrast_bounds(path, result, overwrite=False)
    expected_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    restored = load_interventional_contrast_bounds(
        path,
        expected_sha256=expected_sha256,
    )

    assert restored.artifact_id == result.artifact_id
    assert restored.source_contrast_id == result.source_contrast_id
    np.testing.assert_array_equal(restored.cdf_thresholds, result.cdf_thresholds)
    np.testing.assert_array_equal(restored.cdf_lower, result.cdf_lower)
    np.testing.assert_array_equal(restored.cdf_upper, result.cdf_upper)


def test_inputs_fail_closed() -> None:
    posterior = _independent_product_posterior()
    with pytest.raises(ValueError, match="maximum_pair_count"):
        build_interventional_contrast_bounds(posterior, maximum_pair_count=3)
    with pytest.raises(ValueError, match="cdf_thresholds"):
        build_interventional_contrast_bounds(
            posterior,
            cdf_thresholds=np.ones((2, 2)),
        )
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        build_interventional_contrast_bounds(posterior, metadata=[])
    with pytest.raises(TypeError, match="InterventionalContrastPosteriorV1"):
        build_interventional_contrast_bounds(object())
