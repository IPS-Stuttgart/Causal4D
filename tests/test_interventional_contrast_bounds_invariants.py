from __future__ import annotations

import numpy as np
import pytest

from causal4d.interventional_contrast import (
    InterventionalContrastPosteriorV1,
    build_interventional_contrast_bounds,
)


SOURCE_IDS = (
    "1" * 64,
    "2" * 64,
    "3" * 64,
    "4" * 64,
)


def _source_contrast(
    *,
    pair_indices: np.ndarray | None = None,
    pair_order: np.ndarray | None = None,
    swap_branches: bool = False,
) -> InterventionalContrastPosteriorV1:
    pairs = np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=np.int64,
    )
    values = np.asarray([[-1.0], [-3.0], [1.0], [-1.0]])
    weights = np.full(4, 0.25)
    if swap_branches:
        pairs = pairs[:, ::-1]
        values = -values
    if pair_indices is not None:
        pairs = np.asarray(pair_indices, dtype=np.int64)
    if pair_order is not None:
        order = np.asarray(pair_order, dtype=np.int64)
        pairs = pairs[order]
        values = values[order]
        weights = weights[order]
    return InterventionalContrastPosteriorV1(
        source_branch_a_posterior_id=SOURCE_IDS[0],
        source_branch_b_posterior_id=SOURCE_IDS[1],
        source_branch_a_query_id=SOURCE_IDS[2],
        source_branch_b_query_id=SOURCE_IDS[3],
        branch_a_label="branch-b" if swap_branches else "branch-a",
        branch_b_label="branch-a" if swap_branches else "branch-b",
        trajectory_shape=(1, 1, 3),
        branch_a_component_count=2,
        branch_b_component_count=2,
        coupling_policy="independent_product",
        shared_kappa_names=(),
        conditional_variance_policy="component_means_only",
        query_name="terminal-x",
        query_matrix=np.asarray([[1.0, 0.0, 0.0]]),
        query_labels=("terminal-x",),
        query_units=("m",),
        pair_indices=pairs,
        weights=weights,
        contrast_values=values,
        conditional_covariance=np.zeros((4, 1, 1)),
    )


def test_bounds_are_invariant_to_allowed_pair_order() -> None:
    original = build_interventional_contrast_bounds(
        _source_contrast(),
        cdf_thresholds=(-2.0, 0.0, 2.0),
    )
    permuted = build_interventional_contrast_bounds(
        _source_contrast(pair_order=np.asarray([2, 0, 3, 1])),
        cdf_thresholds=(-2.0, 0.0, 2.0),
    )

    for name in (
        "cdf_lower",
        "cdf_upper",
        "mean",
        "variance_lower",
        "variance_upper",
        "probability_positive_lower",
        "probability_positive_upper",
    ):
        np.testing.assert_allclose(getattr(original, name), getattr(permuted, name))
    assert np.all(np.diff(original.cdf_lower[:, 0]) >= -1.0e-12)
    assert np.all(np.diff(original.cdf_upper[:, 0]) >= -1.0e-12)


def test_branch_swap_reverses_strict_positive_probability_bounds() -> None:
    original = build_interventional_contrast_bounds(_source_contrast())
    swapped = build_interventional_contrast_bounds(
        _source_contrast(swap_branches=True)
    )

    np.testing.assert_allclose(swapped.mean, -original.mean)
    np.testing.assert_allclose(
        swapped.probability_positive_lower,
        1.0 - original.probability_positive_upper,
    )
    np.testing.assert_allclose(
        swapped.probability_positive_upper,
        1.0 - original.probability_positive_lower,
    )


def test_pair_count_guard_precedes_optimization() -> None:
    with pytest.raises(ValueError, match="maximum_pair_count"):
        build_interventional_contrast_bounds(
            _source_contrast(),
            maximum_pair_count=3,
        )
