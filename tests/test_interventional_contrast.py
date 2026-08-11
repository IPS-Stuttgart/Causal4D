from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.artifact_io import ArtifactValidationError
from causal4d.contracts import (
    ActionWindow,
    CausalContext,
    ObservationWindow,
    PhysicalPosterior,
    array_sha256,
)
from causal4d.interventional_contrast import (
    InterventionalContrastPosteriorV1,
    InterventionalContrastSpecificationV1,
    build_interventional_contrast,
    load_interventional_contrast,
    save_interventional_contrast,
    validate_interventional_contrast_sources,
)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _context(action_id: str, action_value: float) -> CausalContext:
    observed_action = np.zeros((3, 1, 3), dtype=float)
    counterfactual_action = np.full((2, 1, 3), action_value, dtype=float)
    return CausalContext(
        protocol_id="contrast-protocol-v1",
        o_minus=ObservationWindow(
            case_id="case-1",
            stream_id="points",
            frame_start=0,
            frame_stop=1,
            content_sha256=_digest("o-minus"),
        ),
        o_plus=ObservationWindow(
            case_id="case-1",
            stream_id="points",
            frame_start=1,
            frame_stop=3,
            content_sha256=_digest("o-plus"),
        ),
        u_obs=ActionWindow(
            action_id="observed",
            case_id="case-1",
            frame_start=0,
            frame_stop=3,
            trajectory_sha256=array_sha256(observed_action),
            provenance="recorded",
        ),
        u_cf=ActionWindow(
            action_id=action_id,
            case_id="case-1",
            frame_start=1,
            frame_stop=3,
            trajectory_sha256=array_sha256(counterfactual_action),
            provenance="counterfactual",
        ),
    )


def _trajectories(values: list[float]) -> np.ndarray:
    result = np.zeros((len(values), 2, 1, 3), dtype=np.float32)
    result[:, :, 0, 0] = np.asarray(values, dtype=np.float32)[:, None]
    return result


def _posterior(
    action_id: str,
    action_value: float,
    values: list[float],
    *,
    weights: list[float] | None = None,
    particles: list[int] | None = None,
    phi: list[list[float]] | None = None,
    kappa: list[list[float]] | None = None,
    contact_policy: str = "same_grasp",
    same_grasp_semantics: str = "fixed_kappa",
    variance: float = 0.0,
    factual_id: str | None = None,
    component_prefix: str | None = None,
) -> PhysicalPosterior:
    count = len(values)
    resolved_weights = (
        np.full(count, 1.0 / count)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    resolved_particles = (
        np.arange(count, dtype=np.int64)
        if particles is None
        else np.asarray(particles, dtype=np.int64)
    )
    resolved_phi = (
        np.ones((count, 1), dtype=float)
        if phi is None
        else np.asarray(phi, dtype=float)
    )
    resolved_kappa = (
        np.zeros((count, 2), dtype=float)
        if kappa is None
        else np.asarray(kappa, dtype=float)
    )
    prefix = action_id if component_prefix is None else component_prefix
    trajectories = _trajectories(values)
    return PhysicalPosterior(
        context=_context(action_id, action_value),
        component_ids=tuple(f"{prefix}-{index}" for index in range(count)),
        state_trajectories_m=trajectories,
        readout_trajectories_m=trajectories,
        readout_variance_m2=np.full((count, 1, 3), variance, dtype=np.float32),
        weights=resolved_weights,
        phi=resolved_phi,
        kappa_cf=resolved_kappa,
        hypothesis_indices=np.arange(count, dtype=np.int64),
        twin_particle_indices=resolved_particles,
        phi_names=("gain",),
        kappa_names=("contact_patch", "slip_fraction"),
        source_twin_belief_id=_digest("twin"),
        source_factual_intervention_id=(
            _digest("factual") if factual_id is None else factual_id
        ),
        source_query_id=_digest(f"query-{action_id}-{action_value}"),
        metadata={
            "contact_policy": contact_policy,
            "same_grasp_semantics": same_grasp_semantics,
        },
    )


def _final_x_query() -> np.ndarray:
    query = np.zeros((1, 2 * 1 * 3), dtype=float)
    query[0, 3] = 1.0
    return query


def _specification(**changes: object) -> InterventionalContrastSpecificationV1:
    values: dict[str, object] = {
        "name": "final-x-left-minus-right",
        "query_matrix": _final_x_query(),
        "query_labels": ("final-x",),
        "trajectory_source": "state",
    }
    values.update(changes)
    return InterventionalContrastSpecificationV1(**values)


def test_auto_same_grasp_coupling_cancels_shared_uncertainty() -> None:
    left = _posterior(
        "left",
        1.0,
        [11.0, 21.0],
        weights=[0.4, 0.6],
        particles=[0, 1],
    )
    right = _posterior(
        "right",
        2.0,
        [10.0, 20.0],
        weights=[0.4, 0.6],
        particles=[0, 1],
    )
    coupled = build_interventional_contrast(left, right, _specification())
    assert coupled.resolved_coupling_policy == "shared_theta_phi_kappa"
    assert coupled.shared_variables == ("theta", "phi", "kappa")
    assert coupled.pair_indices.tolist() == [[0, 0], [1, 1]]
    assert coupled.pair_weights == pytest.approx([0.4, 0.6])
    assert coupled.contrast_components_m[:, 0] == pytest.approx([1.0, 1.0])
    assert coupled.posterior_mean_m == pytest.approx([1.0])
    np.testing.assert_allclose(coupled.posterior_covariance_m2, [[0.0]])
    assert coupled.probability_positive == pytest.approx([1.0])

    independent = build_interventional_contrast(
        left,
        right,
        _specification(coupling_policy="independent"),
    )
    assert len(independent.pair_weights) == 4
    assert independent.posterior_mean_m == pytest.approx([1.0])
    assert independent.posterior_covariance_m2[0, 0] > 0.0


def test_new_contact_coupling_shares_theta_phi_and_resamples_kappa() -> None:
    left = _posterior(
        "left",
        1.0,
        [0.0, 1.0, 10.0, 11.0],
        weights=[0.1, 0.3, 0.2, 0.4],
        particles=[0, 0, 1, 1],
        kappa=[[0.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
        contact_policy="new_contact",
    )
    right = _posterior(
        "right",
        2.0,
        [2.0, 3.0, 12.0, 13.0],
        weights=[0.25, 0.15, 0.3, 0.3],
        particles=[0, 0, 1, 1],
        kappa=[[2.0, 0.0], [3.0, 0.0], [2.0, 0.0], [3.0, 0.0]],
        contact_policy="new_contact",
    )
    result = build_interventional_contrast(left, right, _specification())
    assert result.resolved_coupling_policy == "shared_theta_phi"
    assert result.shared_variables == ("theta", "phi")
    assert len(result.pair_weights) == 8
    for left_index, right_index in result.pair_indices:
        assert left.twin_particle_indices[left_index] == (
            right.twin_particle_indices[right_index]
        )
    left_marginal = np.zeros(4)
    right_marginal = np.zeros(4)
    np.add.at(left_marginal, result.pair_indices[:, 0], result.pair_weights)
    np.add.at(right_marginal, result.pair_indices[:, 1], result.pair_weights)
    assert left_marginal == pytest.approx(left.weights)
    assert right_marginal == pytest.approx(right.weights)


def test_evolving_slip_coupling_shares_patch_but_not_slip() -> None:
    left = _posterior(
        "left",
        1.0,
        [0.0, 1.0, 2.0, 3.0],
        weights=[0.1, 0.2, 0.3, 0.4],
        particles=[0, 0, 0, 0],
        kappa=[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]],
        same_grasp_semantics="evolve_slip",
    )
    right = _posterior(
        "right",
        2.0,
        [4.0, 5.0, 6.0, 7.0],
        weights=[0.2, 0.1, 0.4, 0.3],
        particles=[0, 0, 0, 0],
        kappa=[[0.0, 0.2], [0.0, 0.8], [1.0, 0.2], [1.0, 0.8]],
        same_grasp_semantics="evolve_slip",
    )
    result = build_interventional_contrast(left, right, _specification())
    assert result.resolved_coupling_policy == "shared_theta_phi_patch"
    assert result.shared_variables == ("theta", "phi", "contact_patch")
    for left_index, right_index in result.pair_indices:
        assert left.kappa_cf[left_index, 0] == right.kappa_cf[right_index, 0]


def test_auto_rejects_mixed_contact_semantics() -> None:
    left = _posterior("left", 1.0, [1.0])
    right = _posterior(
        "right",
        2.0,
        [0.0],
        contact_policy="new_contact",
    )
    with pytest.raises(ValueError, match="automatic contrast coupling"):
        build_interventional_contrast(left, right, _specification())
    sensitivity = build_interventional_contrast(
        left,
        right,
        _specification(coupling_policy="independent"),
    )
    assert sensitivity.resolved_coupling_policy == "independent"


def test_shared_group_mass_mismatch_fails_closed() -> None:
    left = _posterior(
        "left",
        1.0,
        [0.0, 1.0, 2.0, 3.0],
        weights=[0.2, 0.2, 0.3, 0.3],
        particles=[0, 0, 1, 1],
        contact_policy="new_contact",
    )
    right = _posterior(
        "right",
        2.0,
        [0.0, 1.0, 2.0, 3.0],
        weights=[0.25, 0.25, 0.25, 0.25],
        particles=[0, 0, 1, 1],
        contact_policy="new_contact",
    )
    with pytest.raises(ValueError, match="marginal mass differs"):
        build_interventional_contrast(left, right, _specification())


def test_readout_correlation_is_an_explicit_cross_world_assumption() -> None:
    left = _posterior("left", 1.0, [1.0], variance=0.25)
    right = _posterior("right", 2.0, [0.0], variance=0.25)
    independent_noise = build_interventional_contrast(
        left,
        right,
        _specification(
            trajectory_source="readout",
            conditional_readout_correlation=0.0,
        ),
    )
    np.testing.assert_allclose(
        independent_noise.component_conditional_variance_m2, [[0.5]]
    )
    np.testing.assert_allclose(
        independent_noise.posterior_covariance_m2, [[0.5]]
    )
    shared_noise = build_interventional_contrast(
        left,
        right,
        _specification(
            trajectory_source="readout",
            conditional_readout_correlation=1.0,
        ),
    )
    np.testing.assert_allclose(
        shared_noise.component_conditional_variance_m2, [[0.0]]
    )
    np.testing.assert_allclose(shared_noise.posterior_covariance_m2, [[0.0]])
    excluded_noise = build_interventional_contrast(
        left,
        right,
        _specification(trajectory_source="readout"),
    )
    assert excluded_noise.conditional_variance_included is False
    np.testing.assert_allclose(excluded_noise.posterior_covariance_m2, [[0.0]])


def test_probability_and_discrete_interval_are_derived_from_coupled_support() -> None:
    left = _posterior(
        "left",
        1.0,
        [-1.0, 2.0],
        weights=[0.25, 0.75],
        particles=[0, 1],
    )
    right = _posterior("right", 2.0, [0.0], weights=[1.0], particles=[0])
    result = build_interventional_contrast(
        left,
        right,
        _specification(coupling_policy="independent", confidence_level=0.5),
    )
    assert result.probability_positive == pytest.approx([0.75])
    np.testing.assert_allclose(result.credible_interval_m, [[-1.0, 2.0]])


def test_round_trip_is_content_addressed_and_immutable(tmp_path: Path) -> None:
    left = _posterior("left", 1.0, [2.0, 4.0], weights=[0.25, 0.75])
    right = _posterior("right", 2.0, [1.0, 3.0], weights=[0.25, 0.75])
    artifact = build_interventional_contrast(left, right, _specification())
    output = tmp_path / "contrast.npz"
    save_interventional_contrast(output, artifact)
    restored = load_interventional_contrast(output)
    assert restored.artifact_id == artifact.artifact_id
    assert restored.as_dict() == artifact.as_dict()
    assert isinstance(restored, InterventionalContrastPosteriorV1)
    with pytest.raises(ValueError):
        restored.pair_weights.setflags(write=True)
    with pytest.raises(FileExistsError):
        save_interventional_contrast(output, artifact)


def test_archive_inventory_and_artifact_id_fail_closed(tmp_path: Path) -> None:
    artifact = build_interventional_contrast(
        _posterior("left", 1.0, [1.0]),
        _posterior("right", 2.0, [0.0]),
        _specification(),
    )
    valid = tmp_path / "valid.npz"
    save_interventional_contrast(valid, artifact)
    with np.load(valid, allow_pickle=False) as archive:
        arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    arrays["extra"] = np.asarray([1])
    extra = tmp_path / "extra.npz"
    np.savez_compressed(extra, **arrays)
    with pytest.raises(ArtifactValidationError, match="inventory"):
        load_interventional_contrast(extra)

    del arrays["extra"]
    descriptor = json.loads(arrays["descriptor_json"].item())
    descriptor["artifact_id"] = _digest("tampered")
    arrays["descriptor_json"] = np.asarray(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    )
    tampered = tmp_path / "tampered.npz"
    np.savez_compressed(tampered, **arrays)
    with pytest.raises(ValueError, match="digest does not match"):
        load_interventional_contrast(tampered)

    descriptor = json.loads(arrays["descriptor_json"].item())
    descriptor["coupling"]["contrast_direction"] = "right_minus_left"
    arrays["descriptor_json"] = np.asarray(
        json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    )
    wrong_direction = tmp_path / "wrong-direction.npz"
    np.savez_compressed(wrong_direction, **arrays)
    with pytest.raises(ValueError, match="left_minus_right"):
        load_interventional_contrast(wrong_direction)


def test_source_bound_validation_replays_query_and_coupling() -> None:
    left = _posterior(
        "left",
        1.0,
        [2.0, 4.0],
        weights=[0.5, 0.5],
        particles=[0, 1],
    )
    right = _posterior(
        "right",
        2.0,
        [1.0, 3.0],
        weights=[0.5, 0.5],
        particles=[0, 1],
    )
    artifact = build_interventional_contrast(left, right, _specification())
    validate_interventional_contrast_sources(artifact, left, right)

    changed_right = _posterior(
        "right",
        2.0,
        [1.0, 30.0],
        weights=[0.5, 0.5],
        particles=[0, 1],
    )
    with pytest.raises(ValueError, match="bound source posteriors"):
        validate_interventional_contrast_sources(artifact, left, changed_right)


def test_source_lineage_query_dimension_and_pair_guard_are_validated() -> None:
    left = _posterior("left", 1.0, [1.0, 2.0])
    wrong_factual = _posterior(
        "right",
        2.0,
        [0.0, 1.0],
        factual_id=_digest("other-factual"),
    )
    with pytest.raises(ValueError, match="different factual interventions"):
        build_interventional_contrast(left, wrong_factual, _specification())

    right = _posterior("right", 2.0, [0.0, 1.0])
    with pytest.raises(ValueError, match="trajectory dimension"):
        build_interventional_contrast(
            left,
            right,
            InterventionalContrastSpecificationV1(
                name="wrong-dimension",
                query_matrix=np.ones((1, 5)),
                query_labels=("wrong",),
                trajectory_source="state",
            ),
        )
    with pytest.raises(ValueError, match="maximum_pair_count"):
        build_interventional_contrast(
            left,
            right,
            _specification(coupling_policy="independent"),
            maximum_pair_count=3,
        )


def test_specification_rejects_implicit_or_inconsistent_semantics() -> None:
    with pytest.raises(ValueError, match="only defined for readout"):
        _specification(conditional_readout_correlation=0.0)
    with pytest.raises(ValueError, match="metres"):
        _specification(query_units=("cm",))
    with pytest.raises(ValueError, match="uniquely"):
        InterventionalContrastSpecificationV1(
            name="duplicate-labels",
            query_matrix=np.eye(2),
            query_labels=("same", "same"),
        )
    with pytest.raises(ValueError, match=r"\[-1, 1\]"):
        _specification(
            trajectory_source="readout",
            conditional_readout_correlation=1.1,
        )
