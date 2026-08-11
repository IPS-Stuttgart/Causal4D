from __future__ import annotations

import numpy as np
import pytest

from causal4d.grouped_likelihood import (
    GroupLikelihoodDiagnostics,
    grouped_component_log_likelihoods,
    posterior_weights_from_grouped_evidence,
)
from causal4d.observation_evidence import GroupedObservationEvidence, ObservationGroup


def _group(
    group_id: str,
    *,
    values: np.ndarray,
    covariance: np.ndarray,
    contributor: str,
    composite_weight: float = 1.0,
) -> ObservationGroup:
    dimension = len(values)
    return ObservationGroup(
        group_id=group_id,
        values_m=np.asarray(values, dtype=float),
        frame_indices=np.ones(dimension, dtype=np.int64),
        node_indices=np.zeros(dimension, dtype=np.int64),
        coordinate_indices=np.arange(dimension, dtype=np.int64),
        covariance_m2=np.asarray(covariance, dtype=float),
        contributor_ids=(contributor,),
        prior_nominal_probability=0.93,
        outlier_scale_multiplier=40.0,
        degrees_of_freedom=5.0,
        composite_weight=composite_weight,
        source_id="unit",
    )


def _posterior(
    evidence: GroupedObservationEvidence,
    components: np.ndarray,
    *,
    likelihood_power: float = 8.0,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics]:
    return posterior_weights_from_grouped_evidence(
        np.asarray([0.5, 0.5]),
        components,
        evidence,
        prefix_frame_count=2,
        score_semantics="normalized_coordinate_mean_v3",
        likelihood_power=likelihood_power,
    )


def test_normalized_v3_duplicate_contributor_is_exactly_power_capped() -> None:
    components = np.zeros((2, 3, 1, 1), dtype=float)
    components[1, 1, 0, 0] = 1.0
    group = _group(
        "g1",
        values=np.asarray([1.0]),
        covariance=np.asarray([[0.01]]),
        contributor="same",
    )
    duplicate = _group(
        "g2",
        values=np.asarray([1.0]),
        covariance=np.asarray([[0.01]]),
        contributor="same",
    )

    single, single_diagnostics = _posterior(
        GroupedObservationEvidence(groups=(group,)),
        components,
    )
    doubled, doubled_diagnostics = _posterior(
        GroupedObservationEvidence(groups=(group, duplicate)),
        components,
    )

    np.testing.assert_allclose(doubled, single, rtol=1e-13, atol=1e-13)
    assert single_diagnostics.contributor_power_caps == (1.0,)
    assert doubled_diagnostics.contributor_power_caps == (0.5, 0.5)
    assert doubled_diagnostics.normalization_coordinate_mass == 1.0


def test_normalized_v3_preserves_composite_weight_as_reliability_temperature() -> None:
    components = np.zeros((2, 3, 1, 1), dtype=float)
    components[1, 1, 0, 0] = 1.0
    full = GroupedObservationEvidence(
        groups=(
            _group(
                "full",
                values=np.asarray([1.0]),
                covariance=np.asarray([[0.01]]),
                contributor="source",
                composite_weight=1.0,
            ),
        )
    )
    tempered = GroupedObservationEvidence(
        groups=(
            _group(
                "tempered",
                values=np.asarray([1.0]),
                covariance=np.asarray([[0.01]]),
                contributor="source",
                composite_weight=0.25,
            ),
        )
    )

    full_posterior, _ = _posterior(full, components)
    tempered_posterior, _ = _posterior(tempered, components)

    assert full_posterior[1] > tempered_posterior[1] > 0.5


def test_normalized_v3_is_invariant_to_coordinate_permutation() -> None:
    covariance = np.asarray(
        [
            [0.040, 0.010, -0.004],
            [0.010, 0.030, 0.006],
            [-0.004, 0.006, 0.020],
        ]
    )
    values = np.asarray([0.4, -0.1, 0.2])
    components = np.zeros((2, 3, 1, 3), dtype=float)
    components[0, 1, 0] = np.asarray([0.35, -0.05, 0.18])
    components[1, 1, 0] = np.asarray([-0.2, 0.2, 0.5])
    group = _group(
        "original",
        values=values,
        covariance=covariance,
        contributor="source",
    )
    original, _ = _posterior(
        GroupedObservationEvidence(groups=(group,)),
        components,
    )

    permutation = np.asarray([2, 0, 1])
    permuted_group = ObservationGroup(
        group_id="permuted",
        values_m=values[permutation],
        frame_indices=group.frame_indices[permutation],
        node_indices=group.node_indices[permutation],
        coordinate_indices=group.coordinate_indices[permutation],
        covariance_m2=covariance[np.ix_(permutation, permutation)],
        contributor_ids=("source",),
        prior_nominal_probability=group.prior_nominal_probability,
        outlier_scale_multiplier=group.outlier_scale_multiplier,
        degrees_of_freedom=group.degrees_of_freedom,
        source_id="unit",
    )
    permuted, _ = _posterior(
        GroupedObservationEvidence(groups=(permuted_group,)),
        components,
    )

    np.testing.assert_allclose(permuted, original, rtol=1e-12, atol=1e-12)


def test_normalized_v3_is_invariant_to_group_order() -> None:
    components = np.zeros((2, 3, 1, 2), dtype=float)
    components[0, 1, 0] = np.asarray([0.3, -0.2])
    components[1, 1, 0] = np.asarray([-0.1, 0.4])
    first = ObservationGroup(
        group_id="first",
        values_m=np.asarray([0.25]),
        frame_indices=np.asarray([1]),
        node_indices=np.asarray([0]),
        coordinate_indices=np.asarray([0]),
        covariance_m2=np.asarray([[0.02]]),
        contributor_ids=("first",),
        source_id="unit",
    )
    second = ObservationGroup(
        group_id="second",
        values_m=np.asarray([-0.15]),
        frame_indices=np.asarray([1]),
        node_indices=np.asarray([0]),
        coordinate_indices=np.asarray([1]),
        covariance_m2=np.asarray([[0.03]]),
        contributor_ids=("second",),
        source_id="unit",
    )

    forward, _ = _posterior(
        GroupedObservationEvidence(groups=(first, second)),
        components,
    )
    reverse, _ = _posterior(
        GroupedObservationEvidence(groups=(second, first)),
        components,
    )

    np.testing.assert_allclose(reverse, forward, rtol=1e-13, atol=1e-13)


def test_normalized_v3_dense_and_low_rank_covariance_updates_agree() -> None:
    generator = np.random.default_rng(17)
    group = _group(
        "structured",
        values=np.asarray([0.2, -0.1, 0.4]),
        covariance=np.asarray(
            [
                [0.4, 0.05, 0.0],
                [0.05, 0.3, 0.02],
                [0.0, 0.02, 0.2],
            ]
        ),
        contributor="structured",
    )
    evidence = GroupedObservationEvidence(groups=(group,))
    components = generator.normal(size=(2, 3, 1, 3))
    factor = generator.normal(scale=0.1, size=(2, 3, 2))
    dense = np.einsum("...ir,...jr->...ij", factor, factor)

    dense_posterior, _ = posterior_weights_from_grouped_evidence(
        np.asarray([0.5, 0.5]),
        components,
        evidence,
        prefix_frame_count=2,
        component_group_covariance_m2={"structured": dense},
        score_semantics="normalized_coordinate_mean_v3",
        likelihood_power=3.0,
    )
    low_rank_posterior, diagnostics = posterior_weights_from_grouped_evidence(
        np.asarray([0.5, 0.5]),
        components,
        evidence,
        prefix_frame_count=2,
        component_group_covariance_factor_m={"structured": factor},
        score_semantics="normalized_coordinate_mean_v3",
        likelihood_power=3.0,
    )

    np.testing.assert_allclose(
        low_rank_posterior,
        dense_posterior,
        rtol=1e-11,
        atol=1e-11,
    )
    assert diagnostics.low_rank_covariance_group_ids == ("structured",)


def test_normalized_v3_reports_information_fractions_and_conditioning() -> None:
    groups = (
        _group(
            "one",
            values=np.asarray([0.0]),
            covariance=np.asarray([[0.2]]),
            contributor="one",
        ),
        _group(
            "two",
            values=np.asarray([0.0, 0.0]),
            covariance=np.asarray([[0.2, 0.05], [0.05, 0.1]]),
            contributor="two",
        ),
    )
    components = np.zeros((2, 3, 1, 2), dtype=float)
    score, diagnostics = grouped_component_log_likelihoods(
        components,
        GroupedObservationEvidence(groups=groups),
        prefix_frame_count=2,
        score_semantics="normalized_coordinate_mean_v3",
        likelihood_power=2.0,
    )

    assert score.shape == (2,)
    assert diagnostics.group_coordinate_counts == (1, 2)
    assert diagnostics.normalization_coordinate_mass == 3.0
    assert np.isclose(sum(diagnostics.normalization_coordinate_fractions), 1.0)
    assert diagnostics.normalization_coordinate_fractions == pytest.approx(
        (1 / 3, 2 / 3)
    )
    assert all(
        value >= 1.0
        for value in diagnostics.source_covariance_condition_numbers
    )


def test_normalized_v3_rejects_ill_conditioned_source_covariance() -> None:
    group = _group(
        "ill-conditioned",
        values=np.asarray([0.0, 0.0]),
        covariance=np.diag([1.0, 1.0e-14]),
        contributor="source",
    )
    components = np.zeros((2, 3, 1, 2), dtype=float)

    with pytest.raises(ValueError, match="condition number"):
        grouped_component_log_likelihoods(
            components,
            GroupedObservationEvidence(groups=(group,)),
            prefix_frame_count=2,
            score_semantics="normalized_coordinate_mean_v3",
            max_source_covariance_condition_number=1.0e10,
        )


def test_grouped_score_settings_fail_closed() -> None:
    group = _group(
        "valid",
        values=np.asarray([0.0]),
        covariance=np.asarray([[1.0]]),
        contributor="source",
    )
    components = np.zeros((2, 3, 1, 1), dtype=float)
    evidence = GroupedObservationEvidence(groups=(group,))

    with pytest.raises(ValueError, match="unsupported grouped score"):
        grouped_component_log_likelihoods(
            components,
            evidence,
            prefix_frame_count=2,
            score_semantics="unknown",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="likelihood_power"):
        grouped_component_log_likelihoods(
            components,
            evidence,
            prefix_frame_count=2,
            likelihood_power=np.nan,
        )
    with pytest.raises(ValueError, match="condition_number"):
        grouped_component_log_likelihoods(
            components,
            evidence,
            prefix_frame_count=2,
            max_source_covariance_condition_number=0.5,
        )
