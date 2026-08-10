from __future__ import annotations

from dataclasses import replace

import numpy as np

import causal4d.joint_observation as joint
from causal4d.prepared_joint_observation import prepare_joint_observation


def _evidence(*, block: bool) -> joint.LinearJointObservationEvidence:
    base = np.stack((np.eye(2), np.eye(2))) if block else np.eye(4)
    return joint.LinearJointObservationEvidence(
        evidence_id=f"selector-grouping-{block}",
        values_m=np.zeros(4),
        row_indices=np.array([0, 0, 1, 1, 2, 3, 3]),
        frame_indices=np.ones(7, dtype=int),
        node_indices=np.array([0, 0, 0, 1, 1, 2, 2]),
        coordinate_indices=np.array([0, 0, 0, 0, 1, 0, 0]),
        coefficients=np.array([1.0, 2.0, -1.5, 0.5, 1.2, 1.0, -0.25]),
        base_covariance_m2=base,
    )


def _explicit_operator(evidence: joint.LinearJointObservationEvidence) -> np.ndarray:
    frame_count, node_count, coordinate_count = 2, 3, 2
    operator = np.zeros(
        (
            evidence.observation_count,
            frame_count * node_count * coordinate_count,
        )
    )
    columns = (
        evidence.frame_indices * node_count + evidence.node_indices
    ) * coordinate_count + evidence.coordinate_indices
    np.add.at(
        operator,
        (evidence.row_indices, columns),
        evidence.coefficients,
    )
    return operator


def _explicit_covariance(
    evidence: joint.LinearJointObservationEvidence,
    variance: np.ndarray,
) -> np.ndarray:
    operator = _explicit_operator(evidence)
    flattened = variance.reshape(*variance.shape[:-3], -1)
    return np.einsum(
        "di,...i,ei->...de",
        operator,
        flattened,
        operator,
    )


def test_selector_groups_aggregate_duplicate_rows_once() -> None:
    evidence = _evidence(block=False)
    groups = joint._group_selector_terms(
        row_indices=evidence.row_indices,
        frame_indices=evidence.frame_indices,
        node_indices=evidence.node_indices,
        coordinate_indices=evidence.coordinate_indices,
        coefficients=evidence.coefficients,
    )
    assert len(groups) == 4
    representative, rows, coefficients = groups[0]
    assert representative == 0
    np.testing.assert_array_equal(rows, np.array([0, 1]))
    np.testing.assert_allclose(coefficients, np.array([3.0, -1.5]))


def test_grouped_dense_propagation_matches_explicit_operator() -> None:
    evidence = _evidence(block=False)
    variance = np.random.default_rng(7).uniform(
        1.0e-5,
        4.0e-3,
        size=(5, 2, 3, 2),
    )
    actual = evidence.apply_independent_covariance(variance)
    expected = _explicit_covariance(evidence, variance)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)

    prepared = prepare_joint_observation(evidence)
    np.testing.assert_allclose(
        actual,
        prepared.apply_independent_covariance(variance),
        rtol=1e-13,
        atol=1e-13,
    )


def test_grouped_block_propagation_matches_explicit_operator() -> None:
    evidence = _evidence(block=True)
    variance = np.random.default_rng(11).uniform(
        1.0e-5,
        4.0e-3,
        size=(3, 2, 3, 2),
    )
    actual = evidence.apply_independent_covariance_blocks(variance)
    dense = _explicit_covariance(evidence, variance)
    expected = np.stack((dense[..., :2, :2], dense[..., 2:, 2:]), axis=-3)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=1e-13)


def test_grouped_compatibility_paths_use_selector_compilation(monkeypatch) -> None:
    dense = _evidence(block=False)
    block = replace(
        dense,
        evidence_id="selector-grouping-spy",
        base_covariance_m2=np.stack((np.eye(2), np.eye(2))),
    )
    variance = np.ones((2, 3, 2)) * 1.0e-3
    calls = 0
    original = joint._group_selector_terms

    def spy(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(joint, "_group_selector_terms", spy)
    dense.apply_independent_covariance(variance)
    block.apply_independent_covariance_blocks(variance)
    assert calls == 2
