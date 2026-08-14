from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from causal4d.grouped_likelihood import posterior_weights_from_grouped_evidence
from causal4d.grouped_likelihood_streaming import (
    GroupedLikelihoodSummaryDiagnostics,
    posterior_weights_from_grouped_evidence_batched,
)
from causal4d.observation_evidence import GroupedObservationEvidence


def _problem() -> tuple[
    np.ndarray,
    np.ndarray,
    GroupedObservationEvidence,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    frame_count = 7
    components = np.zeros((3, 2, frame_count, 1, 3), dtype=float)
    time = np.arange(frame_count, dtype=float)
    components[1, :, :, 0, 0] = 0.01 * time
    components[2, :, :, 0, 0] = -0.01 * time
    components[:, 1, :, 0, 1] = 0.002 * time
    observations = components[1, 0].copy()
    mask = np.ones((frame_count, 1), dtype=bool)
    evidence = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        mask=mask,
        source_id="streaming-unit",
    )
    prior = np.asarray(
        [
            [0.24, 0.36],
            [0.08, 0.12],
            [0.08, 0.12],
        ],
        dtype=float,
    )
    component_variance = np.full((1, 2, 1, 1, 3), 1.0e-8)

    first, second, *_ = evidence.groups
    dense = np.zeros((3, 2, first.coordinate_count, first.coordinate_count))
    dense[..., 0, 0] = 2.0e-8
    factor = np.zeros((3, 2, second.coordinate_count, 1))
    factor[..., 1, 0] = 5.0e-5
    return (
        prior,
        components,
        evidence,
        component_variance,
        {first.group_id: dense},
        {second.group_id: factor},
    )


@pytest.mark.parametrize(
    ("score_semantics", "likelihood_power"),
    [
        ("legacy_sum_v1", 1.0),
        ("normalized_coordinate_mean_v3", 12.0),
    ],
)
@pytest.mark.parametrize("batch_size", [1, 2, 4, 99])
def test_streaming_summary_preserves_grouped_posterior(
    score_semantics: str,
    likelihood_power: float,
    batch_size: int,
) -> None:
    prior, components, evidence, variance, dense, factor = _problem()
    expected, diagnostics = posterior_weights_from_grouped_evidence(
        prior,
        components,
        evidence,
        prefix_frame_count=4,
        component_variance_m2=variance,
        component_group_covariance_m2=dense,
        component_group_covariance_factor_m=factor,
        score_semantics=score_semantics,  # type: ignore[arg-type]
        likelihood_power=likelihood_power,
    )
    actual, summary = posterior_weights_from_grouped_evidence_batched(
        prior,
        components,
        evidence,
        prefix_frame_count=4,
        component_batch_size=batch_size,
        component_variance_m2=variance,
        component_group_covariance_m2=dense,
        component_group_covariance_factor_m=factor,
        score_semantics=score_semantics,  # type: ignore[arg-type]
        likelihood_power=likelihood_power,
    )

    assert np.array_equal(actual, expected)
    responsibilities = diagnostics.nominal_responsibilities.reshape(-1, 3)
    assert np.allclose(
        summary.mean_nominal_responsibility_by_group,
        np.mean(responsibilities, axis=0),
        atol=1e-15,
        rtol=1e-13,
    )
    assert np.array_equal(
        summary.minimum_nominal_responsibility_by_group,
        np.min(responsibilities, axis=0),
    )
    assert summary.component_count == 6
    assert summary.responsibility_storage == "streaming_summary"
    assert summary.full_covariance_group_ids == (evidence.groups[0].group_id,)
    assert summary.low_rank_covariance_group_ids == (evidence.groups[1].group_id,)
    assert summary.as_dict()["responsibility_storage"] == "streaming_summary"


def test_streaming_summary_is_frozen_and_json_compatible() -> None:
    prior, components, evidence, variance, dense, factor = _problem()
    _, summary = posterior_weights_from_grouped_evidence_batched(
        prior,
        components,
        evidence,
        prefix_frame_count=4,
        component_batch_size=2,
        component_variance_m2=variance,
        component_group_covariance_m2=dense,
        component_group_covariance_factor_m=factor,
    )

    assert isinstance(summary, GroupedLikelihoodSummaryDiagnostics)
    assert summary.as_dict()["component_count"] == 6
    with pytest.raises(FrozenInstanceError):
        summary.component_count = 1  # type: ignore[misc]


@pytest.mark.parametrize("invalid", [0, -1, 1.0, True, "2"])
def test_streaming_summary_rejects_invalid_batch_sizes(invalid: object) -> None:
    prior, components, evidence, *_ = _problem()
    with pytest.raises(ValueError, match="component_batch_size"):
        posterior_weights_from_grouped_evidence_batched(
            prior,
            components,
            evidence,
            prefix_frame_count=4,
            component_batch_size=invalid,  # type: ignore[arg-type]
        )


def test_streaming_summary_rejects_unknown_covariance_groups() -> None:
    prior, components, evidence, *_ = _problem()
    with pytest.raises(ValueError, match="unknown groups"):
        posterior_weights_from_grouped_evidence_batched(
            prior,
            components,
            evidence,
            prefix_frame_count=4,
            component_batch_size=2,
            component_group_covariance_m2={"unknown": np.eye(3)},
        )


def test_streaming_summary_never_scores_more_than_one_batch(monkeypatch) -> None:
    import causal4d.grouped_likelihood_streaming as streaming

    prior, components, evidence, *_ = _problem()
    observed_batch_sizes: list[int] = []
    original = streaming.grouped_component_log_likelihoods

    def recording_score(predicted_components_m, *args, **kwargs):
        observed_batch_sizes.append(predicted_components_m.shape[0])
        return original(predicted_components_m, *args, **kwargs)

    monkeypatch.setattr(streaming, "grouped_component_log_likelihoods", recording_score)
    streaming.posterior_weights_from_grouped_evidence_batched(
        prior,
        components,
        evidence,
        prefix_frame_count=4,
        component_batch_size=2,
    )

    assert observed_batch_sizes == [2, 2, 2]
