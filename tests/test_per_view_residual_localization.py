from __future__ import annotations

import numpy as np
import pytest

from causal4d.per_view_residual_localization import (
    localize_per_view_residuals,
)


def _basis(node_count: int, rank: int, *, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    values = rng.normal(size=(node_count, rank))
    values -= np.mean(values, axis=0, keepdims=True)
    values /= np.sqrt(np.mean(np.square(values), axis=0, keepdims=True))
    return values


def test_per_view_localization_recovers_view_frame_and_graph_terms() -> None:
    rng = np.random.default_rng(11)
    view_count, frame_count, node_count, rank = 3, 5, 9, 2
    predicted = rng.normal(scale=0.01, size=(frame_count, node_count, 3))
    basis = _basis(node_count, rank)
    view_bias = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.02, -0.01, 0.005],
            [-0.015, 0.004, 0.01],
        ]
    )
    frame_offset = rng.normal(scale=0.003, size=(frame_count, 3))
    graph_coefficients = rng.normal(
        scale=0.002,
        size=(frame_count, rank, 3),
    )
    graph_field = np.einsum("nr,trc->tnc", basis, graph_coefficients)
    observed = (
        predicted[None]
        + view_bias[:, None, None]
        + frame_offset[None, :, None]
        + graph_field[None]
    )
    validity = np.ones((view_count, frame_count, node_count), dtype=bool)

    result = localize_per_view_residuals(
        observed,
        predicted,
        validity,
        evidence_artifact_id="a" * 64,
        graph_basis=basis,
        view_ridge=0.0,
        frame_ridge=0.0,
        graph_ridge=0.0,
        minimum_explained_fraction=0.01,
        dominance_margin=0.001,
    )

    np.testing.assert_allclose(result.view_bias_m, view_bias, atol=1e-12)
    np.testing.assert_allclose(
        result.shared_frame_offset_m,
        frame_offset,
        atol=1e-12,
    )
    np.testing.assert_allclose(result.graph_field_m, graph_field, atol=1e-12)
    assert result.full_explained_fraction == pytest.approx(1.0)
    assert result.full_weighted_sse_m2 < 1e-24
    assert result.reference_view_index == 0
    assert result.as_dict()["target_outcomes_used"] is False


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("view", "view_specific"),
        ("frame", "shared_frame"),
        ("graph", "object_coherent"),
    ],
)
def test_per_view_localization_identifies_pure_residual_families(
    source: str,
    expected: str,
) -> None:
    view_count, frame_count, node_count = 3, 4, 12
    predicted = np.zeros((frame_count, node_count, 3))
    observed = np.zeros((view_count, frame_count, node_count, 3))
    basis = _basis(node_count, 1)
    if source == "view":
        observed[1, ..., 0] = 0.03
        observed[2, ..., 1] = -0.02
    elif source == "frame":
        observed[:, :, :, 0] = np.linspace(0.0, 0.03, frame_count)[None, :, None]
    else:
        coefficients = np.linspace(-0.01, 0.02, frame_count)
        observed[..., 2] = coefficients[None, :, None] * basis[None, None, :, 0]
    validity = np.ones((view_count, frame_count, node_count), dtype=bool)

    result = localize_per_view_residuals(
        observed,
        predicted,
        validity,
        evidence_artifact_id="b" * 64,
        graph_basis=basis,
        view_ridge=0.0,
        frame_ridge=0.0,
        graph_ridge=0.0,
        minimum_explained_fraction=0.01,
        dominance_margin=0.01,
    )

    assert result.dominant_source == expected
    assert result.full_explained_fraction > 0.999999


def test_per_view_localization_uses_only_the_declared_prefix() -> None:
    view_count, frame_count, node_count = 2, 6, 5
    predicted = np.zeros((frame_count, node_count, 3))
    observed = np.zeros((view_count, frame_count, node_count, 3))
    observed[1, :3, :, 0] = 0.01
    observed[1, 3:, :, 0] = 10.0
    validity = np.ones((view_count, frame_count, node_count), dtype=bool)

    prefix = localize_per_view_residuals(
        observed,
        predicted,
        validity,
        evidence_artifact_id="c" * 64,
        causal_prefix_frame_stop=3,
        view_ridge=0.0,
        frame_ridge=0.0,
    )
    sliced = localize_per_view_residuals(
        observed[:, :3],
        predicted[:3],
        validity[:, :3],
        evidence_artifact_id="c" * 64,
        view_ridge=0.0,
        frame_ridge=0.0,
    )

    assert prefix.artifact_id == sliced.artifact_id
    np.testing.assert_allclose(prefix.view_bias_m, sliced.view_bias_m)
    assert prefix.view_bias_m[1, 0] == pytest.approx(0.01)


def test_per_view_localization_respects_preallocation_guard() -> None:
    observed = np.zeros((3, 4, 20, 3))
    predicted = np.zeros((4, 20, 3))
    validity = np.ones((3, 4, 20), dtype=bool)

    with pytest.raises(MemoryError, match="maximum_design_bytes"):
        localize_per_view_residuals(
            observed,
            predicted,
            validity,
            evidence_artifact_id="d" * 64,
            graph_basis=_basis(20, 3),
            maximum_design_bytes=128,
        )


def test_per_view_localization_result_arrays_are_irreversibly_read_only() -> None:
    observed = np.zeros((2, 2, 3, 3))
    observed[1, ..., 0] = 0.01
    predicted = np.zeros((2, 3, 3))
    validity = np.ones((2, 2, 3), dtype=bool)

    result = localize_per_view_residuals(
        observed,
        predicted,
        validity,
        evidence_artifact_id="e" * 64,
    )

    with pytest.raises(ValueError):
        result.view_bias_m.setflags(write=True)
    with pytest.raises(ValueError):
        result.valid_observation_counts.setflags(write=True)
