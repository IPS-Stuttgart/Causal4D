from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.special import ndtr

from causal4d.artifact_io import ArtifactValidationError
from causal4d.contracts import PhysicalPosterior, build_causal_context
from causal4d.interventional_contrast import (
    InterventionalContrastQueryV1,
    build_interventional_contrast,
    load_interventional_contrast,
    save_interventional_contrast,
)


TWIN_ID = "1" * 64
FACTUAL_ID = "2" * 64


def _context(action_id: str, action_scale: float):
    observations = np.arange(6 * 1 * 3, dtype=float).reshape(6, 1, 3)
    observed_actions = np.zeros((6, 1, 3), dtype=float)
    counterfactual_actions = observed_actions.copy()
    counterfactual_actions[2:, 0, 0] = action_scale
    return build_causal_context(
        protocol_id="contrast-unit-protocol",
        case_id="contrast-unit-case",
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
    weights: tuple[float, ...] | None = None,
    variance_m2: float = 0.0,
    phi: np.ndarray | None = None,
    kappa: np.ndarray | None = None,
    twin_particle_indices: np.ndarray | None = None,
    component_ids: tuple[str, ...] | None = None,
    factual_id: str = FACTUAL_ID,
    metadata: dict[str, Any] | None = None,
) -> PhysicalPosterior:
    count = len(final_x_m)
    trajectories = np.zeros((count, 4, 1, 3), dtype=float)
    trajectories[:, -1, 0, 0] = final_x_m
    if weights is None:
        weights = tuple(np.full(count, 1.0 / count))
    if phi is None:
        phi = np.ones((count, 1), dtype=float)
    if kappa is None:
        kappa = np.column_stack(
            (
                np.arange(count, dtype=float),
                np.zeros(count, dtype=float),
            )
        )
    if twin_particle_indices is None:
        twin_particle_indices = np.zeros(count, dtype=np.int64)
    if component_ids is None:
        component_ids = tuple(f"component-{index}" for index in range(count))
    return PhysicalPosterior(
        context=_context(action_id, action_scale),
        component_ids=component_ids,
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((count, 1, 3), variance_m2),
        weights=np.asarray(weights, dtype=float),
        phi=np.asarray(phi, dtype=float),
        kappa_cf=np.asarray(kappa, dtype=float),
        hypothesis_indices=np.arange(count, dtype=np.int64),
        twin_particle_indices=np.asarray(twin_particle_indices, dtype=np.int64),
        phi_names=("gain",),
        kappa_names=("contact_patch", "slip"),
        source_twin_belief_id=TWIN_ID,
        source_factual_intervention_id=factual_id,
        source_query_id=hashlib.sha256(action_id.encode("utf-8")).hexdigest(),
        metadata=metadata or {},
    )


def _final_x_query() -> InterventionalContrastQueryV1:
    matrix = np.zeros((1, 4 * 1 * 3), dtype=float)
    matrix[0, 3 * 1 * 3] = 1.0
    return InterventionalContrastQueryV1(
        name="final-node-0-x",
        matrix=matrix,
        labels=("final-node-0-x",),
        units=("m",),
        metadata={"registered": True},
    )


def _build_default(**kwargs: Any):
    return build_interventional_contrast(
        _posterior(
            "action-a",
            (2.0, 4.0),
            action_scale=1.0,
            weights=(0.75, 0.25),
        ),
        _posterior(
            "action-b",
            (1.0, 1.0),
            action_scale=-1.0,
            weights=(0.75, 0.25),
        ),
        _final_x_query(),
        branch_a_label="do(action-a)",
        branch_b_label="do(action-b)",
        **kwargs,
    )


def test_shared_component_contrast_has_declared_direction_and_moments() -> None:
    result = _build_default()

    np.testing.assert_array_equal(result.pair_indices, [[0, 0], [1, 1]])
    np.testing.assert_allclose(result.contrast_values[:, 0], [1.0, 3.0])
    np.testing.assert_allclose(result.mean, [1.5])
    np.testing.assert_allclose(result.covariance, [[0.75]])
    np.testing.assert_allclose(result.probability_positive, [1.0])
    assert result.metadata["claim_boundary"]["changes_estimator"] is False
    assert result.metadata["claim_boundary"]["uses_target_truth"] is False

    with pytest.raises(ValueError, match="read-only"):
        result.contrast_values[0, 0] = 0.0


def test_builder_does_not_modify_source_posteriors() -> None:
    branch_a = _posterior(
        "action-a",
        (2.0, 4.0),
        action_scale=1.0,
        weights=(0.75, 0.25),
    )
    branch_b = _posterior(
        "action-b",
        (1.0, 1.0),
        action_scale=-1.0,
        weights=(0.75, 0.25),
    )
    source_ids = (branch_a.artifact_id, branch_b.artifact_id)
    source_bytes = (
        branch_a.readout_trajectories_m.tobytes(),
        branch_b.readout_trajectories_m.tobytes(),
        branch_a.weights.tobytes(),
        branch_b.weights.tobytes(),
    )

    build_interventional_contrast(
        branch_a,
        branch_b,
        _final_x_query(),
        branch_a_label="a",
        branch_b_label="b",
    )

    assert (branch_a.artifact_id, branch_b.artifact_id) == source_ids
    assert (
        branch_a.readout_trajectories_m.tobytes(),
        branch_b.readout_trajectories_m.tobytes(),
        branch_a.weights.tobytes(),
        branch_b.weights.tobytes(),
    ) == source_bytes


def test_independent_readout_adds_only_declared_branch_covariance() -> None:
    result = build_interventional_contrast(
        _posterior(
            "action-a",
            (2.0, 4.0),
            action_scale=1.0,
            weights=(0.75, 0.25),
            variance_m2=0.04,
        ),
        _posterior(
            "action-b",
            (1.0, 1.0),
            action_scale=-1.0,
            weights=(0.75, 0.25),
            variance_m2=0.01,
        ),
        _final_x_query(),
        branch_a_label="a",
        branch_b_label="b",
        conditional_variance_policy="independent_readout",
    )

    np.testing.assert_allclose(
        result.conditional_covariance[:, 0, 0],
        0.05,
        rtol=1e-6,
    )
    np.testing.assert_allclose(result.covariance, [[0.8]], rtol=1e-6)
    expected = 0.75 * ndtr(1.0 / np.sqrt(0.05)) + 0.25 * ndtr(
        3.0 / np.sqrt(0.05)
    )
    np.testing.assert_allclose(result.probability_positive, [expected])
    assert result.metadata["cross_branch_discrepancy_covariance_available"] is False


def test_shared_twin_phi_preserves_marginals_and_can_fix_patch() -> None:
    weights = (0.3, 0.2, 0.1, 0.4)
    phi = np.ones((4, 1), dtype=float)
    kappa = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 0.2],
            [1.0, 0.0],
            [1.0, 0.4],
        ]
    )
    branch_a = _posterior(
        "action-a",
        (1.0, 2.0, 3.0, 4.0),
        action_scale=1.0,
        weights=weights,
        phi=phi,
        kappa=kappa,
    )
    branch_b = _posterior(
        "action-b",
        (0.0, 0.0, 0.0, 0.0),
        action_scale=-1.0,
        weights=weights,
        phi=phi,
        kappa=kappa,
    )

    unrestricted = build_interventional_contrast(
        branch_a,
        branch_b,
        _final_x_query(),
        branch_a_label="a",
        branch_b_label="b",
        coupling_policy="shared_twin_phi",
    )
    patch_fixed = build_interventional_contrast(
        branch_a,
        branch_b,
        _final_x_query(),
        branch_a_label="a",
        branch_b_label="b",
        coupling_policy="shared_twin_phi",
        shared_kappa_names=("contact_patch",),
    )

    assert len(unrestricted.pair_indices) == 16
    assert len(patch_fixed.pair_indices) == 8
    reconstructed_a = np.bincount(
        patch_fixed.pair_indices[:, 0],
        weights=patch_fixed.weights,
        minlength=4,
    )
    reconstructed_b = np.bincount(
        patch_fixed.pair_indices[:, 1],
        weights=patch_fixed.weights,
        minlength=4,
    )
    np.testing.assert_allclose(reconstructed_a, weights, atol=1e-12)
    np.testing.assert_allclose(reconstructed_b, weights, atol=1e-12)
    assert all(
        branch_a.kappa_cf[first, 0] == branch_b.kappa_cf[second, 0]
        for first, second in patch_fixed.pair_indices
    )


def test_independent_product_is_explicit_and_memory_guarded() -> None:
    result = _build_default(coupling_policy="independent_product")
    assert len(result.pair_indices) == 4
    np.testing.assert_allclose(result.weights, [0.5625, 0.1875, 0.1875, 0.0625])

    with pytest.raises(ValueError, match="maximum_pair_count"):
        _build_default(
            coupling_policy="independent_product",
            maximum_pair_count=3,
        )


def test_marginal_quantiles_cover_point_mass_and_gaussian_mixtures() -> None:
    discrete = _build_default()
    np.testing.assert_allclose(
        discrete.marginal_quantiles((0.5, 0.9))[:, 0],
        [1.0, 3.0],
    )
    lower, upper = discrete.central_interval(0.8)
    np.testing.assert_allclose(lower, [1.0])
    np.testing.assert_allclose(upper, [3.0])

    continuous = build_interventional_contrast(
        _posterior(
            "action-a",
            (2.0, 2.0),
            action_scale=1.0,
            weights=(0.75, 0.25),
            variance_m2=0.04,
        ),
        _posterior(
            "action-b",
            (1.0, 1.0),
            action_scale=-1.0,
            weights=(0.75, 0.25),
            variance_m2=0.01,
        ),
        _final_x_query(),
        branch_a_label="a",
        branch_b_label="b",
        conditional_variance_policy="independent_readout",
    )
    np.testing.assert_allclose(
        continuous.marginal_quantiles((0.5,)),
        [[1.0]],
        atol=1e-10,
    )
    with pytest.raises(ValueError, match="Booleans"):
        continuous.marginal_quantiles((True,))


def test_source_ancestry_and_coupling_requirements_fail_closed() -> None:
    branch_a = _posterior(
        "action-a",
        (2.0, 4.0),
        action_scale=1.0,
        weights=(0.75, 0.25),
    )
    different_factual = _posterior(
        "action-b",
        (1.0, 1.0),
        action_scale=-1.0,
        weights=(0.75, 0.25),
        factual_id="f" * 64,
    )
    with pytest.raises(ValueError, match="different factual interventions"):
        build_interventional_contrast(
            branch_a,
            different_factual,
            _final_x_query(),
            branch_a_label="a",
            branch_b_label="b",
        )

    branch_b = _posterior(
        "action-b",
        (1.0, 1.0),
        action_scale=-1.0,
        weights=(0.5, 0.5),
    )
    with pytest.raises(ValueError, match="identical weights"):
        build_interventional_contrast(
            branch_a,
            branch_b,
            _final_x_query(),
            branch_a_label="a",
            branch_b_label="b",
        )
    with pytest.raises(ValueError, match="require shared_twin_phi"):
        _build_default(shared_kappa_names=("contact_patch",))
    with pytest.raises(ValueError, match="unavailable event variables"):
        _build_default(
            coupling_policy="shared_twin_phi",
            shared_kappa_names=("missing",),
        )


def test_query_and_policy_inputs_are_strict() -> None:
    with pytest.raises(ValueError, match="finite"):
        InterventionalContrastQueryV1(
            name="bad",
            matrix=np.asarray([[np.nan]]),
            labels=("x",),
            units=("m",),
        )
    with pytest.raises(ValueError, match="finite JSON"):
        InterventionalContrastQueryV1(
            name="bad",
            matrix=np.ones((1, 1)),
            labels=("x",),
            units=("m",),
            metadata={"invalid": np.nan},
        )
    with pytest.raises(ValueError, match="does not match"):
        build_interventional_contrast(
            _posterior(
                "action-a",
                (2.0, 4.0),
                action_scale=1.0,
                weights=(0.75, 0.25),
            ),
            _posterior(
                "action-b",
                (1.0, 1.0),
                action_scale=-1.0,
                weights=(0.75, 0.25),
            ),
            InterventionalContrastQueryV1(
                name="wrong-size",
                matrix=np.ones((1, 11)),
                labels=("x",),
                units=("m",),
            ),
            branch_a_label="a",
            branch_b_label="b",
        )
    with pytest.raises(ValueError, match="coupling_policy must be a nonempty"):
        _build_default(coupling_policy=[])
    with pytest.raises(ValueError, match="metadata must be a mapping"):
        _build_default(metadata=[])


def test_contrast_loader_rejects_symlinked_archive(tmp_path: Path) -> None:
    result = _build_default()
    target = tmp_path / "contrast.npz"
    save_interventional_contrast(target, result)
    symlink = tmp_path / "contrast-link.npz"
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("symbolic links are unavailable on this platform")

    with pytest.raises(ArtifactValidationError):
        load_interventional_contrast(symlink)


def test_contrast_archive_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    result = _build_default(metadata={"analysis": "unit"})
    target = tmp_path / "contrast.npz"
    save_interventional_contrast(target, result, overwrite=False)
    archive_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    restored = load_interventional_contrast(
        target,
        expected_sha256=archive_sha256,
    )

    assert restored.artifact_id == result.artifact_id
    assert restored.query_id == result.query_id
    np.testing.assert_array_equal(restored.mean, result.mean)
    assert restored.metadata["user"]["analysis"] == "unit"
    with pytest.raises(FileExistsError):
        save_interventional_contrast(target, result, overwrite=False)

    with pytest.raises(ArtifactValidationError, match="SHA-256 mismatch"):
        load_interventional_contrast(target, expected_sha256="0" * 64)

    with np.load(target, allow_pickle=False) as archive:
        descriptor = json.loads(str(archive["descriptor_json"]))
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
            if name != "descriptor_json"
        }
    descriptor["schema_version"] = True
    malformed_schema = tmp_path / "malformed-schema.npz"
    np.savez_compressed(
        malformed_schema,
        descriptor_json=np.asarray(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    )
    with pytest.raises(ValueError, match="schema_version must be a positive"):
        load_interventional_contrast(malformed_schema)

    descriptor["schema_version"] = 1
    arrays["weights"] = arrays["weights"].astype(np.float32)
    malformed_dtype = tmp_path / "malformed-dtype.npz"
    np.savez_compressed(
        malformed_dtype,
        descriptor_json=np.asarray(
            json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
        ),
        **arrays,
    )
    with pytest.raises(ValueError, match="must use dtype"):
        load_interventional_contrast(malformed_dtype)
