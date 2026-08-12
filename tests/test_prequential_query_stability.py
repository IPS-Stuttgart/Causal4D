from __future__ import annotations

import numpy as np
import pytest

from causal4d.prequential_abduction import PrequentialAbductionPathV1
from causal4d.prequential_query_stability import (
    PREQUENTIAL_QUERY_STABILITY_CLAIM_BOUNDARY,
    build_prequential_query_stability,
)


def _posterior_summaries(
    weights: np.ndarray,
) -> tuple[np.ndarray, ...]:
    positive = weights > 0.0
    entropy_terms = np.zeros_like(weights)
    entropy_terms[positive] = weights[positive] * np.log(weights[positive])
    entropy = -np.sum(entropy_terms, axis=1)
    effective_sample_size = 1.0 / np.sum(np.square(weights), axis=1)
    maximum_weight = np.max(weights, axis=1)
    map_component_indices = np.argmax(weights, axis=1).astype(np.int64)
    previous_step_kl = np.zeros(len(weights), dtype=float)
    previous_step_total_variation = np.zeros(len(weights), dtype=float)
    for index in range(1, len(weights)):
        current = weights[index]
        previous = weights[index - 1]
        selected = current > 0.0
        previous_safe = np.maximum(previous[selected], np.finfo(float).tiny)
        previous_step_kl[index] = float(
            np.sum(current[selected] * np.log(current[selected] / previous_safe))
        )
        previous_step_total_variation[index] = float(
            0.5 * np.sum(np.abs(current - previous))
        )
    return (
        entropy,
        effective_sample_size,
        maximum_weight,
        map_component_indices,
        previous_step_kl,
        previous_step_total_variation,
    )


def _path(
    weights: np.ndarray | None = None,
    *,
    component_ids: tuple[str, ...] = ("component-a", "component-b"),
) -> PrequentialAbductionPathV1:
    posterior = np.asarray(
        weights
        if weights is not None
        else [[0.5, 0.5], [0.75, 0.25], [1.0, 0.0]],
        dtype=float,
    )
    summaries = _posterior_summaries(posterior)
    return PrequentialAbductionPathV1(
        source_rollout_bank_id="a" * 64,
        source_twin_belief_id="b" * 64,
        component_ids=component_ids,
        factual_intervention_ids=("c" * 64, "d" * 64, "e" * 64),
        step_evidence_ids=(
            "dense-prefix:2",
            "dense-prefix:3",
            "dense-prefix:4",
        ),
        prefix_frame_counts=np.asarray([2, 3, 4], dtype=np.int64),
        evidence_frame_stops=np.asarray([10, 11, 12], dtype=np.int64),
        posterior_weights=posterior,
        posterior_entropy=summaries[0],
        posterior_effective_sample_size=summaries[1],
        posterior_maximum_weight=summaries[2],
        map_component_indices=summaries[3],
        previous_step_kl=summaries[4],
        previous_step_total_variation=summaries[5],
        metadata={"future_frames_read": 0},
    )


def _build(
    path: PrequentialAbductionPathV1,
    values: np.ndarray | None = None,
    *,
    scales: tuple[float, ...] = (2.0, 4.0),
):
    return build_prequential_query_stability(
        path,
        np.asarray(values if values is not None else [[0.0, 0.0], [2.0, 4.0]]),
        query_id="registered-track-query-v1",
        query_labels=("track-x", "track-y"),
        query_units=("m", "m"),
        query_scales=scales,
        metadata={"registered_before_target_access": True},
    )


def test_query_stability_reports_native_and_standardized_drift() -> None:
    result = _build(_path())
    summaries = result.summary_arrays()

    np.testing.assert_allclose(
        summaries["posterior_query_mean"],
        [[1.0, 2.0], [0.5, 1.0], [0.0, 0.0]],
    )
    np.testing.assert_allclose(
        summaries["final_mean_shift_standardized_l2"],
        [np.sqrt(0.5), np.sqrt(0.125), 0.0],
    )
    assert summaries["previous_gaussian_wasserstein_standardized"][0] == 0.0
    assert summaries["final_gaussian_wasserstein_standardized"][-1] == 0.0
    assert summaries["previous_interval_overlap_fraction"][0] == 1.0
    assert summaries["final_interval_overlap_fraction"][-1] == 1.0
    assert result.as_dict()["claim_boundary"] == (
        PREQUENTIAL_QUERY_STABILITY_CLAIM_BOUNDARY
    )


def test_query_stability_is_invariant_to_consistent_unit_conversion() -> None:
    metres = _build(_path())
    millimetres = _build(
        _path(),
        values=np.asarray([[0.0, 0.0], [2000.0, 4000.0]]),
        scales=(2000.0, 4000.0),
    )
    metres_summary = metres.summary_arrays()
    millimetres_summary = millimetres.summary_arrays()

    for name in (
        "previous_mean_shift_standardized_l2",
        "final_mean_shift_standardized_l2",
        "previous_gaussian_wasserstein_standardized",
        "final_gaussian_wasserstein_standardized",
        "previous_interval_overlap_fraction",
        "final_interval_overlap_fraction",
    ):
        np.testing.assert_allclose(
            metres_summary[name],
            millimetres_summary[name],
            rtol=1.0e-12,
            atol=1.0e-12,
        )
    np.testing.assert_allclose(
        millimetres_summary["posterior_query_mean"],
        1000.0 * metres_summary["posterior_query_mean"],
    )


def test_query_stability_is_invariant_to_component_permutation() -> None:
    original_path = _path()
    permuted_weights = original_path.posterior_weights[:, ::-1]
    permuted_path = _path(
        permuted_weights,
        component_ids=tuple(reversed(original_path.component_ids)),
    )
    original = _build(original_path)
    permuted = _build(
        permuted_path,
        values=np.asarray([[2.0, 4.0], [0.0, 0.0]]),
    )

    for name, values in original.summary_arrays().items():
        np.testing.assert_allclose(
            values,
            permuted.summary_arrays()[name],
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_query_stability_rejects_shape_and_scale_errors() -> None:
    path = _path()
    with pytest.raises(ValueError, match="one row per path component"):
        _build(path, values=np.asarray([[0.0, 0.0]]))
    with pytest.raises(ValueError, match="finite and positive"):
        _build(path, scales=(2.0, 0.0))
    with pytest.raises(ValueError, match="query_labels"):
        build_prequential_query_stability(
            path,
            np.asarray([[0.0, 0.0], [2.0, 4.0]]),
            query_id="registered-track-query-v1",
            query_labels=("duplicate", "duplicate"),
            query_units=("m", "m"),
            query_scales=(2.0, 4.0),
        )


def test_query_stability_identity_binds_path_and_query_values() -> None:
    path = _path()
    first = _build(path)
    changed_values = _build(
        path,
        values=np.asarray([[0.0, 0.0], [2.0, 4.1]]),
    )
    changed_path = _build(
        _path([[0.4, 0.6], [0.75, 0.25], [1.0, 0.0]]),
    )

    assert first.artifact_id != changed_values.artifact_id
    assert first.artifact_id != changed_path.artifact_id
