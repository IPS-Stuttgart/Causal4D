"""Cross-world coupling and contrast construction."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np

from causal4d.contracts import PhysicalPosterior
from causal4d.immutable_json import plain_json
from causal4d.interventional_contrast.posterior import (
    InterventionalContrastPosteriorV1,
)
from causal4d.interventional_contrast.specification import (
    CouplingPolicy,
    InterventionalContrastSpecificationV1,
    ResolvedCouplingPolicy,
    _BOUNDARY_METADATA,
    _metadata_string,
    _require_sha256,
    _validated_weights,
)


def _validate_source_posterior(posterior: PhysicalPosterior, *, name: str) -> None:
    if getattr(posterior, "contract_type", None) != "PhysicalPosterior":
        raise TypeError(f"{name} must be a PhysicalPosterior")
    _require_sha256(posterior.artifact_id, name=f"{name}.artifact_id")
    _require_sha256(
        posterior.source_twin_belief_id,
        name=f"{name}.source_twin_belief_id",
    )
    _require_sha256(
        posterior.source_factual_intervention_id,
        name=f"{name}.source_factual_intervention_id",
    )
    _require_sha256(posterior.source_query_id, name=f"{name}.source_query_id")
    trajectories = np.asarray(posterior.state_trajectories_m)
    readout = np.asarray(posterior.readout_trajectories_m)
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError(f"{name} state trajectories must have shape (K, T, N, 3)")
    if readout.shape != trajectories.shape:
        raise ValueError(f"{name} readout trajectories must match state trajectories")
    if not np.all(np.isfinite(trajectories)) or not np.all(np.isfinite(readout)):
        raise ValueError(f"{name} trajectories must be finite")
    weights = _validated_weights(posterior.weights, name=f"{name}.weights")
    if len(weights) != len(trajectories):
        raise ValueError(f"{name} weights do not match trajectory support")
    component_ids = tuple(posterior.component_ids)
    if len(component_ids) != len(weights) or len(set(component_ids)) != len(weights):
        raise ValueError(f"{name} component IDs do not match support")
    phi = np.asarray(posterior.phi, dtype=float)
    kappa = np.asarray(posterior.kappa_cf, dtype=float)
    particles = np.asarray(posterior.twin_particle_indices)
    if phi.shape[0] != len(weights) or kappa.shape[0] != len(weights):
        raise ValueError(f"{name} intervention rows do not match support")
    if particles.shape != (len(weights),) or particles.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{name} particle indices do not match support")


def _validate_common_lineage(
    left: PhysicalPosterior,
    right: PhysicalPosterior,
) -> None:
    if left.source_twin_belief_id != right.source_twin_belief_id:
        raise ValueError("contrast posteriors descend from different twin beliefs")
    if left.source_factual_intervention_id != right.source_factual_intervention_id:
        raise ValueError(
            "contrast posteriors descend from different factual interventions"
        )
    for field_name in ("protocol_id", "o_minus", "o_plus", "u_obs"):
        if getattr(left.context, field_name) != getattr(right.context, field_name):
            raise ValueError(
                f"contrast posteriors disagree on factual {field_name}"
            )
    left_shape = np.asarray(left.state_trajectories_m).shape[1:]
    right_shape = np.asarray(right.state_trajectories_m).shape[1:]
    if left_shape != right_shape:
        raise ValueError("contrast posteriors have different trajectory shapes")
    if tuple(left.phi_names) != tuple(right.phi_names):
        raise ValueError("contrast posteriors use different phi schemas")
    if tuple(left.kappa_names) != tuple(right.kappa_names):
        raise ValueError("contrast posteriors use different kappa schemas")


def _resolved_policy(
    left: PhysicalPosterior,
    right: PhysicalPosterior,
    requested: CouplingPolicy,
) -> ResolvedCouplingPolicy:
    if requested != "auto":
        return requested
    left_policy = _metadata_string(left, "contact_policy")
    right_policy = _metadata_string(right, "contact_policy")
    left_semantics = _metadata_string(
        left,
        "same_grasp_semantics",
        default="fixed_kappa",
    )
    right_semantics = _metadata_string(
        right,
        "same_grasp_semantics",
        default="fixed_kappa",
    )
    if left_policy == right_policy == "new_contact":
        return "shared_theta_phi"
    if left_policy == right_policy == "same_grasp":
        if left_semantics == right_semantics == "fixed_kappa":
            return "shared_theta_phi_kappa"
        if left_semantics == right_semantics == "evolve_slip":
            return "shared_theta_phi_patch"
    raise ValueError(
        "automatic contrast coupling requires matching new-contact, fixed-grasp, "
        "or evolving-slip semantics; choose an explicit sensitivity policy"
    )


def _row_key(values: np.ndarray) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _component_keys(
    posterior: PhysicalPosterior,
    policy: ResolvedCouplingPolicy,
) -> tuple[tuple[Hashable, ...], tuple[str, ...]]:
    component_count = len(posterior.weights)
    if policy == "independent":
        return tuple(() for _ in range(component_count)), ()
    if policy == "component_id":
        return tuple((value,) for value in posterior.component_ids), ("component_id",)
    particles = np.asarray(posterior.twin_particle_indices, dtype=np.int64)
    phi = np.asarray(posterior.phi, dtype=float)
    kappa = np.asarray(posterior.kappa_cf, dtype=float)
    keys: list[tuple[Hashable, ...]] = []
    shared_variables = ["theta", "phi"]
    if policy == "shared_theta_phi_patch":
        try:
            slip_index = tuple(posterior.kappa_names).index("slip_fraction")
        except ValueError as error:
            raise ValueError(
                "shared_theta_phi_patch requires a named slip_fraction variable"
            ) from error
        patch_indices = tuple(
            index for index in range(kappa.shape[1]) if index != slip_index
        )
        if not patch_indices:
            raise ValueError(
                "shared_theta_phi_patch requires at least one contact-patch variable"
            )
        shared_variables.append("contact_patch")
    else:
        patch_indices = ()
        if policy == "shared_theta_phi_kappa":
            shared_variables.append("kappa")
    for index in range(component_count):
        key: tuple[Hashable, ...] = (
            int(particles[index]),
            _row_key(phi[index]),
        )
        if policy == "shared_theta_phi_kappa":
            key += (_row_key(kappa[index]),)
        elif policy == "shared_theta_phi_patch":
            key += (
                tuple(float(kappa[index, item]) for item in patch_indices),
            )
        keys.append(key)
    return tuple(keys), tuple(shared_variables)


def _build_coupling(
    left: PhysicalPosterior,
    right: PhysicalPosterior,
    *,
    requested_policy: CouplingPolicy,
    maximum_pair_count: int,
) -> tuple[np.ndarray, np.ndarray, ResolvedCouplingPolicy, tuple[str, ...]]:
    if type(maximum_pair_count) is not int or maximum_pair_count < 1:
        raise ValueError("maximum_pair_count must be a positive integer")
    resolved = _resolved_policy(left, right, requested_policy)
    left_weights = np.asarray(left.weights, dtype=float)
    right_weights = np.asarray(right.weights, dtype=float)
    if resolved == "independent":
        left_active = np.flatnonzero(left_weights > 0.0)
        right_active = np.flatnonzero(right_weights > 0.0)
        pair_count = len(left_active) * len(right_active)
        if pair_count > maximum_pair_count:
            raise ValueError(
                "independent coupling exceeds maximum_pair_count; use shared "
                "cross-world variables or raise the explicit memory guard"
            )
        pair_indices = np.empty((pair_count, 2), dtype=np.int64)
        pair_weights = np.empty(pair_count, dtype=float)
        cursor = 0
        for left_index in left_active:
            count = len(right_active)
            pair_indices[cursor : cursor + count, 0] = left_index
            pair_indices[cursor : cursor + count, 1] = right_active
            pair_weights[cursor : cursor + count] = (
                left_weights[left_index] * right_weights[right_active]
            )
            cursor += count
        return pair_indices, pair_weights, resolved, ()

    left_keys, shared_variables = _component_keys(left, resolved)
    right_keys, right_shared_variables = _component_keys(right, resolved)
    if right_shared_variables != shared_variables:
        raise RuntimeError("internal shared-variable coupling mismatch")
    left_groups: defaultdict[tuple[Hashable, ...], list[int]] = defaultdict(list)
    right_groups: defaultdict[tuple[Hashable, ...], list[int]] = defaultdict(list)
    for index, key in enumerate(left_keys):
        if left_weights[index] > 0.0:
            left_groups[key].append(index)
    for index, key in enumerate(right_keys):
        if right_weights[index] > 0.0:
            right_groups[key].append(index)
    if set(left_groups) != set(right_groups):
        raise ValueError(
            "shared-variable coupling support differs between interventions"
        )
    pair_count = sum(
        len(left_groups[key]) * len(right_groups[key]) for key in left_groups
    )
    if pair_count > maximum_pair_count:
        raise ValueError(
            "shared-variable coupling exceeds maximum_pair_count; reduce support "
            "or raise the explicit memory guard"
        )
    pair_indices = np.empty((pair_count, 2), dtype=np.int64)
    pair_weights = np.empty(pair_count, dtype=float)
    cursor = 0
    for key in sorted(left_groups, key=repr):
        left_indices = np.asarray(left_groups[key], dtype=np.int64)
        right_indices = np.asarray(right_groups[key], dtype=np.int64)
        left_mass = float(np.sum(left_weights[left_indices]))
        right_mass = float(np.sum(right_weights[right_indices]))
        if not np.isclose(left_mass, right_mass, atol=1e-10, rtol=1e-10):
            raise ValueError(
                "shared-variable marginal mass differs between interventions"
            )
        if left_mass <= 0.0:
            raise RuntimeError("active coupling group has no posterior mass")
        group_weights = (
            left_weights[left_indices, None]
            * right_weights[right_indices][None, :]
            / left_mass
        )
        group_pairs = len(left_indices) * len(right_indices)
        pair_indices[cursor : cursor + group_pairs, 0] = np.repeat(
            left_indices,
            len(right_indices),
        )
        pair_indices[cursor : cursor + group_pairs, 1] = np.tile(
            right_indices,
            len(left_indices),
        )
        pair_weights[cursor : cursor + group_pairs] = group_weights.reshape(-1)
        cursor += group_pairs
    pair_weights /= np.sum(pair_weights)
    return pair_indices, pair_weights, resolved, shared_variables


def _project_component_trajectories(
    posterior: PhysicalPosterior,
    specification: InterventionalContrastSpecificationV1,
) -> np.ndarray:
    source = (
        posterior.state_trajectories_m
        if specification.trajectory_source == "state"
        else posterior.readout_trajectories_m
    )
    trajectories = np.asarray(source, dtype=float)
    component_matrix = trajectories.reshape(len(trajectories), -1)
    if component_matrix.shape[1] != specification.query_matrix.shape[1]:
        raise ValueError(
            "query_matrix trajectory dimension does not match physical posteriors"
        )
    return component_matrix @ specification.query_matrix.T


def _expanded_readout_variance(posterior: PhysicalPosterior) -> np.ndarray:
    trajectories = np.asarray(posterior.readout_trajectories_m)
    variance = np.asarray(posterior.readout_variance_m2, dtype=float)
    expected = (len(trajectories), trajectories.shape[2], trajectories.shape[3])
    if variance.shape != expected:
        raise ValueError("readout variance does not match physical posterior support")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("readout variance must be finite and nonnegative")
    return np.broadcast_to(
        variance[:, None],
        trajectories.shape,
    ).reshape(len(trajectories), -1)


def _conditional_query_uncertainty(
    left: PhysicalPosterior,
    right: PhysicalPosterior,
    specification: InterventionalContrastSpecificationV1,
    pair_indices: np.ndarray,
    pair_weights: np.ndarray,
    *,
    maximum_working_bytes: int,
) -> tuple[np.ndarray, np.ndarray]:
    pair_count = len(pair_weights)
    query_count = len(specification.query_labels)
    component_variance = np.zeros((pair_count, query_count), dtype=float)
    expected_covariance = np.zeros((query_count, query_count), dtype=float)
    correlation = specification.conditional_readout_correlation
    if specification.trajectory_source == "state" or correlation is None:
        return component_variance, expected_covariance
    if type(maximum_working_bytes) is not int or maximum_working_bytes < 1:
        raise ValueError("maximum_working_bytes must be a positive integer")
    left_variance = _expanded_readout_variance(left)
    right_variance = _expanded_readout_variance(right)
    dimension = left_variance.shape[1]
    query = np.asarray(specification.query_matrix, dtype=float)
    bytes_per_pair = 8 * max(3 * dimension + query_count * query_count, 1)
    chunk_size = max(1, maximum_working_bytes // bytes_per_pair)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        left_index = pair_indices[start:stop, 0]
        right_index = pair_indices[start:stop, 1]
        left_values = left_variance[left_index]
        right_values = right_variance[right_index]
        cross_scale = np.sqrt(left_values * right_values)
        difference_variance = (
            left_values + right_values - 2.0 * correlation * cross_scale
        )
        numerical_floor = 64.0 * np.finfo(float).eps * np.maximum(
            left_values + right_values,
            1.0,
        )
        if np.any(difference_variance < -numerical_floor):
            raise FloatingPointError(
                "conditional cross-world variance became materially negative"
            )
        difference_variance = np.maximum(difference_variance, 0.0)
        covariance = np.einsum(
            "qd,bd,rd->bqr",
            query,
            difference_variance,
            query,
            optimize=True,
        )
        component_variance[start:stop] = np.diagonal(
            covariance,
            axis1=-2,
            axis2=-1,
        )
        expected_covariance += np.einsum(
            "b,bij->ij",
            pair_weights[start:stop],
            covariance,
            optimize=True,
        )
    expected_covariance = 0.5 * (expected_covariance + expected_covariance.T)
    return component_variance, expected_covariance


def build_interventional_contrast(
    left: PhysicalPosterior,
    right: PhysicalPosterior,
    specification: InterventionalContrastSpecificationV1,
    *,
    maximum_pair_count: int = 1_000_000,
    maximum_working_bytes: int = 256 * 1024**2,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalContrastPosteriorV1:
    """Build a coupled posterior for ``Q(left) - Q(right)``.

    The coupling preserves both source posterior marginals.  Automatic coupling
    shares ``theta`` and ``phi`` across new-contact actions, additionally shares
    the contact patch for evolving-slip same-grasp actions, and shares complete
    ``kappa`` for fixed same-grasp actions.
    """

    _validate_source_posterior(left, name="left")
    _validate_source_posterior(right, name="right")
    if not isinstance(specification, InterventionalContrastSpecificationV1):
        raise TypeError("specification has the wrong type")
    _validate_common_lineage(left, right)
    pair_indices, pair_weights, resolved_policy, shared_variables = _build_coupling(
        left,
        right,
        requested_policy=specification.coupling_policy,
        maximum_pair_count=maximum_pair_count,
    )
    left_query = _project_component_trajectories(left, specification)
    right_query = _project_component_trajectories(right, specification)
    contrasts = (
        left_query[pair_indices[:, 0]] - right_query[pair_indices[:, 1]]
    )
    conditional_variance, expected_conditional_covariance = (
        _conditional_query_uncertainty(
            left,
            right,
            specification,
            pair_indices,
            pair_weights,
            maximum_working_bytes=maximum_working_bytes,
        )
    )
    left_contact_policy = _metadata_string(left, "contact_policy")
    right_contact_policy = _metadata_string(right, "contact_policy")
    left_semantics = _metadata_string(
        left,
        "same_grasp_semantics",
        default="fixed_kappa",
    )
    right_semantics = _metadata_string(
        right,
        "same_grasp_semantics",
        default="fixed_kappa",
    )
    result_metadata: dict[str, Any] = dict(_BOUNDARY_METADATA)
    if metadata is not None:
        supplied_metadata = dict(metadata)
        reserved = set(result_metadata) & set(supplied_metadata)
        if reserved:
            raise ValueError(
                "contrast metadata cannot replace reserved boundary fields: "
                f"{sorted(reserved)}"
            )
        result_metadata.update(supplied_metadata)
    return InterventionalContrastPosteriorV1(
        specification=specification,
        source_twin_belief_id=left.source_twin_belief_id,
        source_factual_intervention_id=left.source_factual_intervention_id,
        left_posterior_id=left.artifact_id,
        right_posterior_id=right.artifact_id,
        left_query_id=left.source_query_id,
        right_query_id=right.source_query_id,
        left_action_id=left.context.u_cf.action_id,
        right_action_id=right.context.u_cf.action_id,
        left_action_trajectory_sha256=left.context.u_cf.trajectory_sha256,
        right_action_trajectory_sha256=right.context.u_cf.trajectory_sha256,
        left_contact_policy=left_contact_policy,
        right_contact_policy=right_contact_policy,
        left_same_grasp_semantics=left_semantics,
        right_same_grasp_semantics=right_semantics,
        requested_coupling_policy=specification.coupling_policy,
        resolved_coupling_policy=resolved_policy,
        shared_variables=shared_variables,
        left_component_ids=tuple(left.component_ids),
        right_component_ids=tuple(right.component_ids),
        pair_indices=pair_indices,
        pair_weights=pair_weights,
        left_weights=np.asarray(left.weights, dtype=float),
        right_weights=np.asarray(right.weights, dtype=float),
        contrast_components_m=contrasts,
        component_conditional_variance_m2=conditional_variance,
        expected_conditional_covariance_m2=expected_conditional_covariance,
        metadata=result_metadata,
    )

