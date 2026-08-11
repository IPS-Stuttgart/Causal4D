from __future__ import annotations

import hashlib

import numpy as np
import pytest

from causal4d.interventional_contrast import (
    InterventionalContrastPosteriorV1,
    build_interventional_contrast_bounds,
)


def _digest(name: str) -> str:
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def test_pair_values_must_define_a_coupling_invariant_mean() -> None:
    malformed = InterventionalContrastPosteriorV1(
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
        conditional_variance_policy="component_means_only",
        query_name="x",
        query_matrix=np.asarray([[1.0, 0.0, 0.0]]),
        query_labels=("x",),
        query_units=("m",),
        pair_indices=np.asarray([[0, 0], [0, 1], [1, 0], [1, 1]]),
        weights=np.full(4, 0.25),
        contrast_values=np.asarray([[0.0], [0.0], [0.0], [10.0]]),
        conditional_covariance=np.zeros((4, 1, 1)),
        query_metadata={"registered": True},
    )

    with pytest.raises(ValueError, match="coupling-invariant mean"):
        build_interventional_contrast_bounds(malformed)
