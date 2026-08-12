from __future__ import annotations

import copy

import numpy as np
import pytest

from causal4d.query_variance_decomposition import (
    build_query_variance_decomposition,
    validate_query_variance_decomposition,
)


def test_additive_factor_attribution_and_conditional_covariance() -> None:
    weights = np.full(4, 0.25)
    means = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [1.0, 0.0],
            [1.0, 2.0],
        ]
    )
    observation = np.repeat(np.diag([0.1, 0.2])[None, :, :], 4, axis=0)
    result = build_query_variance_decomposition(
        weights,
        means,
        query_id="endpoint-position",
        query_labels=("x", "y"),
        query_units=("m", "m"),
        query_scales=(1.0, 1.0),
        factor_values={
            "contact": ("left", "left", "right", "right"),
            "gain": ("low", "high", "low", "high"),
        },
        conditional_covariances={"observation": observation},
    )

    arrays = result.summary_arrays()
    np.testing.assert_allclose(arrays["factor__contact"], np.diag([0.25, 0.0]))
    np.testing.assert_allclose(arrays["factor__gain"], np.diag([0.0, 1.0]))
    np.testing.assert_allclose(
        arrays["conditional__observation"],
        np.diag([0.1, 0.2]),
    )
    np.testing.assert_allclose(
        arrays["total_covariance"],
        np.diag([0.35, 1.2]),
    )
    np.testing.assert_allclose(arrays["unresolved_component_covariance"], 0.0)
    payload = result.as_dict()
    assert payload["diagnostics"]["max_abs_additivity_error"] < 1.0e-12
    assert validate_query_variance_decomposition(payload) == payload


def test_factor_order_does_not_change_named_shapley_attribution() -> None:
    weights = np.full(4, 0.25)
    means = np.array([[0.0], [2.0], [1.0], [3.0]])
    first = build_query_variance_decomposition(
        weights,
        means,
        query_id="q",
        query_labels=("q",),
        query_units=("m",),
        query_scales=(1.0,),
        factor_values={
            "a": ("0", "0", "1", "1"),
            "b": ("0", "1", "0", "1"),
        },
    )
    second = build_query_variance_decomposition(
        weights,
        means,
        query_id="q",
        query_labels=("q",),
        query_units=("m",),
        query_scales=(1.0,),
        factor_values={
            "b": ("0", "1", "0", "1"),
            "a": ("0", "0", "1", "1"),
        },
    )

    np.testing.assert_allclose(
        first.summary_arrays()["factor__a"],
        second.summary_arrays()["factor__a"],
    )
    np.testing.assert_allclose(
        first.summary_arrays()["factor__b"],
        second.summary_arrays()["factor__b"],
    )
    assert first.decomposition_id == second.decomposition_id


def test_undeclared_component_identity_is_retained_as_unresolved() -> None:
    result = build_query_variance_decomposition(
        (0.5, 0.5),
        ((0.0,), (2.0,)),
        query_id="q",
        query_labels=("q",),
        query_units=("m",),
        query_scales=(1.0,),
        factor_values={"same-label": ("x", "x")},
    )

    np.testing.assert_allclose(result.summary_arrays()["factor__same-label"], 0.0)
    np.testing.assert_allclose(
        result.summary_arrays()["unresolved_component_covariance"],
        [[1.0]],
    )


def test_portable_validator_rejects_tampered_covariance_and_identity() -> None:
    payload = build_query_variance_decomposition(
        (0.5, 0.5),
        ((0.0,), (2.0,)),
        query_id="q",
        query_labels=("q",),
        query_units=("m",),
        query_scales=(1.0,),
        factor_values={"state": ("a", "b")},
    ).as_dict()

    changed = copy.deepcopy(payload)
    changed["covariance"]["factor_shapley"]["state"][0][0] = 0.5
    with pytest.raises(ValueError, match="reconstruct"):
        validate_query_variance_decomposition(changed)

    changed = copy.deepcopy(payload)
    changed["metadata"] = {"edited": True}
    with pytest.raises(ValueError, match="content identity"):
        validate_query_variance_decomposition(changed)


def test_input_validation_and_irreversible_array_immutability() -> None:
    result = build_query_variance_decomposition(
        (0.5, 0.5),
        ((0.0,), (1.0,)),
        query_id="q",
        query_labels=("q",),
        query_units=("m",),
        query_scales=(1.0,),
    )
    with pytest.raises(ValueError, match="positive"):
        build_query_variance_decomposition(
            (0.5, 0.5),
            ((0.0,), (1.0,)),
            query_id="q",
            query_labels=("q",),
            query_units=("m",),
            query_scales=(0.0,),
        )
    with pytest.raises(ValueError, match="positive semidefinite"):
        build_query_variance_decomposition(
            (1.0,),
            ((0.0, 0.0),),
            query_id="q",
            query_labels=("x", "y"),
            query_units=("m", "m"),
            query_scales=(1.0, 1.0),
            conditional_covariances={"bad": np.array([[[1.0, 2.0], [2.0, 1.0]]])},
        )
    with pytest.raises(ValueError, match="without Boolean"):
        build_query_variance_decomposition(
            (True,),
            ((0.0,),),
            query_id="q",
            query_labels=("q",),
            query_units=("m",),
            query_scales=(1.0,),
        )
    mean = result.summary_arrays()["posterior_mean"]
    with pytest.raises(ValueError):
        mean.setflags(write=True)
