from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from causal4d.joint_observation import joint_component_log_likelihoods
from causal4d.prob4d_prepared_observation import (
    prepare_prob4d_joint_observation,
)


def _descriptor() -> dict[str, object]:
    return {
        "case_id": "case-prepared",
        "source_revision": "a" * 40,
        "source_artifact_sha256": "b" * 64,
    }


def _arrays() -> dict[str, np.ndarray]:
    return {
        "mean_xyz_m": np.array(
            [
                [0.1, 0.2, 0.3],
                [-0.1, 0.0, 0.4],
            ]
        ),
        "local_covariance_m2": np.array(
            [
                np.eye(3) * 0.04,
                np.array(
                    [
                        [0.09, 0.01, 0.0],
                        [0.01, 0.07, 0.005],
                        [0.0, 0.005, 0.08],
                    ]
                ),
            ]
        ),
        "low_rank_factor_m": np.array(
            [
                [[0.02], [0.01], [0.0]],
                [[-0.01], [0.015], [0.005]],
            ]
        ),
        "frame_ids": np.array([10, 11], dtype=np.int64),
        "entity_ids": np.array([7, 8], dtype=np.int64),
        "factor_group_ids": np.zeros(2, dtype=np.int64),
        "association_probability": np.ones(2),
        "prior_reliability": np.ones(2),
        "group_prior_nominal_probability": np.ones(2),
        "group_composite_weight": np.ones(2),
    }


def _components() -> np.ndarray:
    components = np.zeros((4, 3, 2, 3))
    components[0, 1, 1] = [0.12, 0.18, 0.31]
    components[0, 2, 0] = [-0.12, 0.01, 0.38]
    components[1, 1, 1] = [0.4, -0.2, 0.1]
    components[1, 2, 0] = [0.0, 0.2, 0.7]
    components[2, 1, 1] = [0.1, 0.2, 0.3]
    components[2, 2, 0] = [-0.1, 0.0, 0.4]
    components[3, 1, 1] = [0.08, 0.23, 0.29]
    components[3, 2, 0] = [-0.08, -0.02, 0.43]
    return components


@pytest.fixture
def prepared(monkeypatch):
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    return prepare_prob4d_joint_observation(
        _descriptor(),
        _arrays(),
        rollout_frame_ids=(9, 10, 11),
        entity_to_node={7: 1, 8: 0},
    )


def test_prepared_prob4d_scores_match_legacy_and_reuse_solver(
    prepared,
    monkeypatch,
) -> None:
    components = _components()
    expected, _ = joint_component_log_likelihoods(
        components,
        prepared.evidence,
        prefix_frame_count=3,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("prepared Prob4D base was factorized again")

    monkeypatch.setattr(np.linalg, "cholesky", forbidden)
    first, first_diagnostics = prepared.log_likelihoods(
        components,
        prefix_frame_count=3,
        component_chunk_size=2,
    )
    second, second_diagnostics = prepared.log_likelihoods(
        components,
        prefix_frame_count=3,
        component_chunk_size=1,
    )

    np.testing.assert_allclose(first, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(second, expected, rtol=1e-12, atol=1e-12)
    assert first_diagnostics.base_factorization_reused is True
    assert second_diagnostics.base_factorization_reused is True
    assert prepared.adapter_diagnostics.factor_rank == 1
    assert prepared.artifact_id == prepared.evidence.artifact_id


def test_prepared_prob4d_posterior_preserves_zero_support(prepared) -> None:
    posterior, diagnostics = prepared.posterior_weights(
        np.array([0.5, 0.0, 0.5, 0.0]),
        _components(),
        prefix_frame_count=3,
        component_chunk_size=2,
    )

    assert posterior[1] == 0.0
    assert posterior[3] == 0.0
    assert np.isclose(np.sum(posterior), 1.0)
    assert diagnostics.base_factorization_reused is True


def test_prepared_prob4d_accepts_portable_metadata_source_identity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    descriptor = {
        "case_id": "case-prepared",
        "metadata": {
            "source_revision": "a" * 40,
            "source_artifact_sha256": "b" * 64,
        },
    }

    result = prepare_prob4d_joint_observation(
        descriptor,
        _arrays(),
        rollout_frame_ids=(9, 10, 11),
        entity_to_node={7: 1, 8: 0},
    )

    assert result.evidence.metadata["source_revision"] == "a" * 40
    assert result.evidence.metadata["source_artifact_sha256"] == "b" * 64


def test_prepared_prob4d_rejects_conflicting_source_identity(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    descriptor = _descriptor()
    descriptor["metadata"] = {
        "source_revision": "c" * 40,
        "source_artifact_sha256": "b" * 64,
    }

    with pytest.raises(ValueError, match="source_revision differs"):
        prepare_prob4d_joint_observation(
            descriptor,
            _arrays(),
            rollout_frame_ids=(9, 10, 11),
            entity_to_node={7: 1, 8: 0},
        )


def test_prepared_prob4d_rejects_a_mismatched_plan(
    prepared,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "causal4d.prob4d_joint_observation.validate_prob4d_causal_observation_metadata",
        lambda descriptor, arrays: {"validated": True},
    )
    other = prepare_prob4d_joint_observation(
        _descriptor(),
        _arrays(),
        rollout_frame_ids=(9, 10, 11),
        entity_to_node={7: 1, 8: 0},
        evidence_id="other-prob4d-evidence",
    )

    with pytest.raises(ValueError, match="same Prob4D evidence artifact"):
        replace(prepared, prepared=other.prepared)
