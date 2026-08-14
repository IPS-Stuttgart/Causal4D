from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.interventional_contrast import (
    InterventionalContrastQueryV1,
    build_interventional_contrast,
    build_interventional_contrast_readout_correlation_sensitivity,
    load_interventional_contrast_readout_correlation_sensitivity,
    save_interventional_contrast_readout_correlation_sensitivity,
)


TWIN_ID = "1" * 64
FACTUAL_ID = "2" * 64


def _context(action_id: str, action_scale: float):
    observations = np.arange(6 * 1 * 3, dtype=float).reshape(6, 1, 3)
    observed_actions = np.zeros((6, 1, 3), dtype=float)
    counterfactual_actions = observed_actions.copy()
    counterfactual_actions[2:, 0, 0] = action_scale
    return build_causal_context(
        protocol_id="readout-correlation-unit-protocol",
        case_id="readout-correlation-unit-case",
        observations=observations,
        observed_actions=observed_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=2,
        counterfactual_action_id=action_id,
    )


def _posterior(
    action_id: str,
    final_x_m: tuple[float, ...],
    *,
    action_scale: float,
    weights: tuple[float, ...] = (0.75, 0.25),
    variance_m2: float = 0.04,
    factual_id: str = FACTUAL_ID,
) -> PhysicalPosterior:
    count = len(final_x_m)
    trajectories = np.zeros((count, 4, 1, 3), dtype=float)
    trajectories[:, -1, 0, 0] = final_x_m
    return PhysicalPosterior(
        context=_context(action_id, action_scale),
        component_ids=tuple(f"component-{index}" for index in range(count)),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((count, 1, 3), variance_m2),
        weights=np.asarray(weights, dtype=float),
        phi=np.ones((count, 1), dtype=float),
        kappa_cf=np.column_stack(
            (
                np.arange(count, dtype=float),
                np.zeros(count, dtype=float),
            )
        ),
        hypothesis_indices=np.arange(count, dtype=np.int64),
        twin_particle_indices=np.zeros(count, dtype=np.int64),
        phi_names=("gain",),
        kappa_names=("contact_patch", "slip"),
        source_twin_belief_id=TWIN_ID,
        source_factual_intervention_id=factual_id,
        source_query_id=hashlib.sha256(action_id.encode("utf-8")).hexdigest(),
        metadata={},
    )


def _query() -> InterventionalContrastQueryV1:
    matrix = np.zeros((1, 4 * 1 * 3), dtype=float)
    matrix[0, 3 * 1 * 3] = 1.0
    return InterventionalContrastQueryV1(
        name="final-node-0-x",
        matrix=matrix,
        labels=("final-node-0-x",),
        units=("m",),
        metadata={"registered": True},
    )


def _source_contrast(
    *,
    branch_a_variance_m2: float = 0.04,
    branch_b_variance_m2: float = 0.04,
    final_a: tuple[float, ...] = (2.0, 4.0),
    final_b: tuple[float, ...] = (1.0, 1.0),
    policy: str = "independent_readout",
):
    branch_a = _posterior(
        "action-a",
        final_a,
        action_scale=1.0,
        variance_m2=branch_a_variance_m2,
    )
    branch_b = _posterior(
        "action-b",
        final_b,
        action_scale=-1.0,
        variance_m2=branch_b_variance_m2,
    )
    contrast = build_interventional_contrast(
        branch_a,
        branch_b,
        _query(),
        branch_a_label="do(action-a)",
        branch_b_label="do(action-b)",
        conditional_variance_policy=policy,
    )
    return branch_a, branch_b, contrast


def _build(**kwargs: Any):
    branch_a, branch_b, contrast = _source_contrast()
    return build_interventional_contrast_readout_correlation_sensitivity(
        branch_a,
        branch_b,
        contrast,
        **kwargs,
    )


def test_zero_correlation_reproduces_independent_source_exactly() -> None:
    branch_a, branch_b, contrast = _source_contrast()
    sensitivity = build_interventional_contrast_readout_correlation_sensitivity(
        branch_a,
        branch_b,
        contrast,
        correlations=(-1.0, 0.0, 1.0),
    )

    zero = sensitivity.zero_correlation_index
    np.testing.assert_allclose(
        sensitivity.total_variance[zero],
        np.diag(contrast.covariance),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.probability_positive[zero],
        contrast.probability_positive,
        atol=1e-12,
    )
    np.testing.assert_allclose(sensitivity.mean, contrast.mean)
    assert sensitivity.source_contrast_id == contrast.artifact_id
    assert sensitivity.metadata["claim_boundary"]["changes_estimator"] is False
    assert (
        sensitivity.metadata["claim_boundary"][
            "cross_branch_conditional_covariance_identified"
        ]
        is False
    )


def test_perfect_correlation_limits_equal_variance_readout_uncertainty() -> None:
    sensitivity = _build(correlations=(-1.0, 0.0, 1.0))

    np.testing.assert_allclose(
        sensitivity.conditional_variance[:, 0],
        [0.16, 0.08, 0.0],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sensitivity.total_variance[:, 0],
        sensitivity.between_component_variance[0] + np.asarray([0.16, 0.08, 0.0]),
        atol=1e-12,
    )
    lower, upper = sensitivity.total_variance_envelope
    np.testing.assert_allclose(lower, sensitivity.total_variance[-1])
    np.testing.assert_allclose(upper, sensitivity.total_variance[0])


def test_unequal_variance_positive_correlation_retains_scale_difference() -> None:
    branch_a, branch_b, contrast = _source_contrast(
        branch_a_variance_m2=0.04,
        branch_b_variance_m2=0.01,
    )
    sensitivity = build_interventional_contrast_readout_correlation_sensitivity(
        branch_a,
        branch_b,
        contrast,
        correlations=(-1.0, 0.0, 1.0),
    )

    np.testing.assert_allclose(
        sensitivity.conditional_variance[:, 0],
        [0.09, 0.05, 0.01],
        atol=1e-12,
    )


def test_probability_curve_uses_declared_conditional_variance() -> None:
    branch_a, branch_b, contrast = _source_contrast(
        final_a=(0.1, 0.1),
        final_b=(0.0, 0.0),
    )
    sensitivity = build_interventional_contrast_readout_correlation_sensitivity(
        branch_a,
        branch_b,
        contrast,
        correlations=(-1.0, 0.0, 1.0),
    )

    assert (
        sensitivity.probability_positive[0, 0]
        < sensitivity.probability_positive[1, 0]
    )
    assert (
        sensitivity.probability_positive[1, 0]
        < sensitivity.probability_positive[2, 0]
    )
    np.testing.assert_allclose(sensitivity.probability_positive[2, 0], 1.0)


def test_builder_rejects_wrong_source_and_missing_independent_variance() -> None:
    branch_a, branch_b, contrast = _source_contrast()
    replacement = _posterior(
        "action-a",
        (2.0, 5.0),
        action_scale=1.0,
        variance_m2=0.04,
    )
    with pytest.raises(ValueError, match="branch A does not match"):
        build_interventional_contrast_readout_correlation_sensitivity(
            replacement,
            branch_b,
            contrast,
        )

    branch_a, branch_b, means_only = _source_contrast(
        policy="component_means_only"
    )
    with pytest.raises(ValueError, match="requires an independent_readout"):
        build_interventional_contrast_readout_correlation_sensitivity(
            branch_a,
            branch_b,
            means_only,
        )


def test_correlation_grid_is_strict_and_contains_independent_baseline() -> None:
    with pytest.raises(ValueError, match="Booleans"):
        _build(correlations=(False, True))
    with pytest.raises(ValueError, match="strictly increasing"):
        _build(correlations=(0.0, -0.5, 1.0))
    with pytest.raises(ValueError, match="contain zero"):
        _build(correlations=(-1.0, 1.0))
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        _build(correlations=(-1.1, 0.0, 1.0))


def test_sensitivity_is_immutable_and_content_addressed() -> None:
    sensitivity = _build(
        correlations=(-1.0, 0.0, 1.0),
        metadata={"registered_before_target_access": True},
    )
    original_id = sensitivity.artifact_id

    with pytest.raises(ValueError, match="read-only"):
        sensitivity.total_variance[0, 0] = 0.0
    assert sensitivity.artifact_id == original_id

    changed = _build(
        correlations=(-1.0, -0.25, 0.0, 1.0),
        metadata={"registered_before_target_access": True},
    )
    assert changed.artifact_id != original_id


def test_archive_round_trip_and_expected_digest(tmp_path: Path) -> None:
    sensitivity = _build(correlations=(-1.0, 0.0, 1.0))
    archive = tmp_path / "readout-correlation.npz"
    save_interventional_contrast_readout_correlation_sensitivity(
        archive,
        sensitivity,
        overwrite=False,
    )
    archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()

    restored = load_interventional_contrast_readout_correlation_sensitivity(
        archive,
        expected_sha256=archive_sha256,
    )
    assert restored.artifact_id == sensitivity.artifact_id
    np.testing.assert_array_equal(
        restored.correlation_grid,
        sensitivity.correlation_grid,
    )
    np.testing.assert_allclose(restored.total_variance, sensitivity.total_variance)

    with pytest.raises(ValueError, match="SHA-256"):
        load_interventional_contrast_readout_correlation_sensitivity(
            archive,
            expected_sha256="f" * 64,
        )


def test_archive_rejects_unexpected_array(tmp_path: Path) -> None:
    sensitivity = _build(correlations=(-1.0, 0.0, 1.0))
    source = tmp_path / "source.npz"
    save_interventional_contrast_readout_correlation_sensitivity(source, sensitivity)
    with np.load(source, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["unexpected"] = np.asarray([1.0])
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **payload)

    with pytest.raises(ValueError, match="array fields do not match"):
        load_interventional_contrast_readout_correlation_sensitivity(tampered)
