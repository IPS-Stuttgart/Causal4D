"""Abduction-action-prediction operator for Causal4D PhysTwin rollouts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from typing import Any, Mapping

import numpy as np

from causal4d.contracts import (
    CausalContext,
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TwinBelief,
)
from causal4d.intervention_abduction import physical_readout_components
from causal4d.rollout_bank import JointRolloutBank


SAME_GRASP_SEMANTICS = ("fixed_kappa", "evolve_slip")


def _phi_from_metadata(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    contact = metadata["contact"]
    return (
        float(contact["gain_multiplier"]),
        float(contact["delay_steps"]),
        float(contact["rotation_degrees"]),
    )


def _contact_patch_from_metadata(
    metadata: Mapping[str, Any],
) -> tuple[float, ...]:
    return tuple(map(float, metadata["contact"]["attachment_shifts"]))


def _kappa_from_metadata(metadata: Mapping[str, Any]) -> tuple[float, ...]:
    contact = metadata["contact"]
    return _contact_patch_from_metadata(metadata) + (float(contact["slip_fraction"]),)


def _conditional_hypothesis_prior_weights(
    prior_weights: np.ndarray,
    equivalence_keys: Sequence[Hashable],
) -> np.ndarray:
    """Normalize query priors inside each semantic equivalence class.

    Exact-zero prior hypotheses stay outside the finite support. An equivalence
    class with zero total prior receives zero conditional mass and is therefore
    unavailable to the caller.
    """

    priors = np.asarray(prior_weights, dtype=float)
    if priors.shape != (len(equivalence_keys),):
        raise ValueError("equivalence keys must identify every hypothesis")
    if not np.all(np.isfinite(priors)) or np.any(priors < 0.0):
        raise ValueError("hypothesis priors must be finite and nonnegative")

    denominators: defaultdict[Hashable, float] = defaultdict(float)
    for key, prior in zip(equivalence_keys, priors, strict=True):
        denominators[key] += float(prior)

    conditional = np.zeros_like(priors)
    for index, key in enumerate(equivalence_keys):
        denominator = denominators[key]
        if denominator > 0.0:
            conditional[index] = priors[index] / denominator
    return conditional


def _validate_factual_context(
    belief: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
) -> None:
    if factual.source_twin_belief_id != belief.artifact_id:
        raise ValueError("factual intervention does not descend from TwinBelief")
    if query.source_factual_intervention_id != factual.artifact_id:
        raise ValueError("counterfactual query does not descend from factual abduction")
    for name in ("protocol_id", "o_minus", "o_plus", "u_obs"):
        expected = getattr(belief.context, name)
        if (
            getattr(factual.context, name) != expected
            or getattr(query.context, name) != expected
        ):
            raise ValueError(f"counterfactual artifacts disagree on factual {name}")


def _validated_query_node_indices(
    query: CounterfactualQuery,
    *,
    node_count: int,
) -> np.ndarray:
    if query.query_node_indices is None:
        return np.arange(node_count, dtype=np.int64)
    nodes = np.asarray(query.query_node_indices, dtype=np.int64)
    if np.any(nodes >= node_count):
        raise ValueError("query_node_indices exceed the rollout-bank node count")
    if len(np.unique(nodes)) != len(nodes):
        raise ValueError("query_node_indices must not contain duplicates")
    return nodes


def _validate_query_bank(
    bank: JointRolloutBank,
    query: CounterfactualQuery,
    manifest: Mapping[str, Any],
) -> None:
    if "causal_context" not in manifest:
        raise ValueError("rollout manifest has no causal context")
    bank_context = CausalContext.from_dict(manifest["causal_context"])
    if bank_context != query.context:
        raise ValueError("rollout bank context does not match do(u_cf)")
    action_ids = {
        str(metadata["action"]["proposal_id"]) for metadata in bank.hypothesis_metadata
    }
    if action_ids != {query.context.u_cf.action_id}:
        raise ValueError("rollout bank does not contain exactly the queried action")
    expected_frame_count = query.horizon_frames + 1
    if bank.frame_count != expected_frame_count:
        raise ValueError(
            "rollout bank must contain one intervention-endpoint frame plus "
            "query.horizon_frames future frames"
        )
    _validated_query_node_indices(query, node_count=bank.node_count)


def _new_contact_weights(
    bank: JointRolloutBank,
    factual: FactualIntervention,
) -> tuple[np.ndarray, float]:
    """Carry ``p(theta, phi)`` and sample a fresh event ``kappa_cf``."""

    phi_theta: defaultdict[tuple[int, tuple[float, ...]], float] = defaultdict(float)
    for index, weight in enumerate(factual.weights):
        key = (
            int(factual.twin_particle_indices[index]),
            tuple(map(float, factual.phi[index])),
        )
        phi_theta[key] += float(weight)

    query_phi = [_phi_from_metadata(metadata) for metadata in bank.hypothesis_metadata]
    conditional_kappa = _conditional_hypothesis_prior_weights(
        bank.hypothesis_prior_weights,
        query_phi,
    )
    weights = np.zeros_like(bank.prior_joint_weights)
    for hypothesis_index, phi in enumerate(query_phi):
        for particle_index in range(len(bank.parameter_weights)):
            weights[hypothesis_index, particle_index] = (
                phi_theta[(particle_index, phi)] * conditional_kappa[hypothesis_index]
            )
    retained_mass = float(np.sum(weights))
    if retained_mass <= 0.0:
        raise ValueError("query contact beam has no support for factual phi posterior")
    return weights / retained_mass, retained_mass


def _same_grasp_weights(
    bank: JointRolloutBank,
    factual: FactualIntervention,
) -> tuple[np.ndarray, float]:
    """Carry the complete factual ``(theta, phi, kappa_obs)`` joint posterior."""

    query_keys = [
        (_phi_from_metadata(metadata), _kappa_from_metadata(metadata))
        for metadata in bank.hypothesis_metadata
    ]
    conditional_hypothesis = _conditional_hypothesis_prior_weights(
        bank.hypothesis_prior_weights,
        query_keys,
    )
    query_lookup: defaultdict[
        tuple[tuple[float, ...], tuple[float, ...]], list[int]
    ] = defaultdict(list)
    for hypothesis_index, key in enumerate(query_keys):
        query_lookup[key].append(hypothesis_index)

    weights = np.zeros_like(bank.prior_joint_weights)
    for component_index, weight in enumerate(factual.weights):
        key = (
            tuple(map(float, factual.phi[component_index])),
            tuple(map(float, factual.kappa_obs[component_index])),
        )
        matches = query_lookup.get(key, [])
        particle = int(factual.twin_particle_indices[component_index])
        for hypothesis_index in matches:
            weights[hypothesis_index, particle] += (
                float(weight) * conditional_hypothesis[hypothesis_index]
            )
    retained_mass = float(np.sum(weights))
    if retained_mass <= 0.0:
        raise ValueError("query contact beam cannot represent the factual grasp")
    return weights / retained_mass, retained_mass


def _same_patch_weights(
    bank: JointRolloutBank,
    factual: FactualIntervention,
) -> tuple[np.ndarray, float]:
    """Carry ``p(theta, phi, patch)`` while resampling counterfactual slip."""

    hand_count = len(bank.hypothesis_metadata[0]["contact"]["attachment_shifts"])
    patch_theta_phi: defaultdict[
        tuple[int, tuple[float, ...], tuple[float, ...]], float
    ] = defaultdict(float)
    for index, weight in enumerate(factual.weights):
        patch = tuple(map(float, factual.kappa_obs[index, :hand_count]))
        key = (
            int(factual.twin_particle_indices[index]),
            tuple(map(float, factual.phi[index])),
            patch,
        )
        patch_theta_phi[key] += float(weight)

    query_keys = [
        (
            _phi_from_metadata(metadata),
            _contact_patch_from_metadata(metadata),
        )
        for metadata in bank.hypothesis_metadata
    ]
    conditional_slip = _conditional_hypothesis_prior_weights(
        bank.hypothesis_prior_weights,
        query_keys,
    )

    weights = np.zeros_like(bank.prior_joint_weights)
    for hypothesis_index, (phi, patch) in enumerate(query_keys):
        for particle_index in range(len(bank.parameter_weights)):
            weights[hypothesis_index, particle_index] = (
                patch_theta_phi[(particle_index, phi, patch)]
                * conditional_slip[hypothesis_index]
            )
    retained_mass = float(np.sum(weights))
    if retained_mass <= 0.0:
        raise ValueError("query contact beam cannot represent the factual patch")
    return weights / retained_mass, retained_mass


def apply_counterfactual_operator(
    bank: JointRolloutBank,
    manifest: Mapping[str, Any],
    belief: TwinBelief,
    factual: FactualIntervention,
    query: CounterfactualQuery,
) -> PhysicalPosterior:
    """Apply ``do(u_cf)`` while transferring phi and handling kappa explicitly."""

    _validate_factual_context(belief, factual, query)
    _validate_query_bank(bank, query, manifest)
    if not np.array_equal(bank.parameter_particles, belief.theta):
        raise ValueError("counterfactual bank theta differs from TwinBelief")
    expected_phi_names = (
        "gain_multiplier",
        "delay_steps",
        "rotation_degrees",
    )
    hand_count = len(bank.hypothesis_metadata[0]["contact"]["attachment_shifts"])
    expected_kappa_names = tuple(
        f"attachment_shift_hand_{index}" for index in range(hand_count)
    ) + ("slip_fraction",)
    if (
        factual.phi_names != expected_phi_names
        or factual.kappa_names != expected_kappa_names
    ):
        raise ValueError("factual intervention variable schema differs from query bank")

    same_grasp_semantics = str(
        query.metadata.get("same_grasp_semantics", "fixed_kappa")
    )
    if same_grasp_semantics not in SAME_GRASP_SEMANTICS:
        raise ValueError("same_grasp_semantics must be fixed_kappa or evolve_slip")
    if query.contact_policy == "new_contact":
        joint_weights, retained_mass = _new_contact_weights(bank, factual)
        reused_factual_kappa = False
        reused_factual_patch = False
        reused_factual_slip = False
    elif same_grasp_semantics == "evolve_slip":
        joint_weights, retained_mass = _same_patch_weights(bank, factual)
        reused_factual_kappa = False
        reused_factual_patch = True
        reused_factual_slip = False
    else:
        joint_weights, retained_mass = _same_grasp_weights(bank, factual)
        reused_factual_kappa = True
        reused_factual_patch = True
        reused_factual_slip = True

    state = bank.trajectories.reshape(
        -1,
        bank.frame_count,
        bank.node_count,
        bank.coordinate_count,
    )
    readout = physical_readout_components(bank, belief).reshape(state.shape)
    hypothesis_indices = np.repeat(
        np.arange(len(bank.hypothesis_ids), dtype=np.int64),
        len(bank.parameter_weights),
    )
    particle_indices = np.tile(
        np.arange(len(bank.parameter_weights), dtype=np.int64),
        len(bank.hypothesis_ids),
    )
    phi_by_hypothesis = np.asarray(
        [_phi_from_metadata(value) for value in bank.hypothesis_metadata],
        dtype=float,
    )
    kappa_by_hypothesis = np.asarray(
        [_kappa_from_metadata(value) for value in bank.hypothesis_metadata],
        dtype=float,
    )
    phi = phi_by_hypothesis[hypothesis_indices]
    kappa = kappa_by_hypothesis[hypothesis_indices]
    discrepancy_variance = (
        belief.discrepancy_variance_m2[
            particle_indices,
            : bank.node_count,
        ]
        + bank.variance_floor_m2
    )
    component_ids = tuple(
        f"{bank.hypothesis_ids[hypothesis]}::{belief.particle_ids[particle]}"
        for hypothesis, particle in zip(
            hypothesis_indices,
            particle_indices,
            strict=True,
        )
    )
    return PhysicalPosterior(
        context=query.context,
        component_ids=component_ids,
        state_trajectories_m=state,
        readout_trajectories_m=readout,
        readout_variance_m2=discrepancy_variance,
        weights=joint_weights.reshape(-1),
        phi=phi,
        kappa_cf=kappa,
        hypothesis_indices=hypothesis_indices,
        twin_particle_indices=particle_indices,
        phi_names=expected_phi_names,
        kappa_names=expected_kappa_names,
        source_twin_belief_id=belief.artifact_id,
        source_factual_intervention_id=factual.artifact_id,
        source_query_id=query.artifact_id,
        metadata={
            "operator": "abduction-action-prediction",
            "intervention": f"do({query.context.u_cf.action_id})",
            "contact_policy": query.contact_policy,
            "same_grasp_semantics": same_grasp_semantics,
            "persistent_phi_transferred": True,
            "factual_kappa_reused": reused_factual_kappa,
            "factual_contact_patch_reused": reused_factual_patch,
            "factual_slip_reused": reused_factual_slip,
            "fresh_kappa_cf_sampled": not reused_factual_kappa,
            "counterfactual_slip_resampled": (
                query.contact_policy == "same_grasp"
                and same_grasp_semantics == "evolve_slip"
            ),
            "represented_factual_mass_before_renormalization": retained_mass,
            "rollout_includes_pre_intervention_endpoint": True,
            "discrepancy_injected_into_simulator_state": False,
            "discrepancy_applied_to_readout": True,
        },
    )


def project_physical_posterior(
    posterior: PhysicalPosterior,
    query: CounterfactualQuery,
    *,
    include_endpoint: bool = False,
) -> PhysicalPosterior:
    """Project a dense posterior onto the registered query horizon and nodes.

    Counterfactual rollout banks contain the factual intervention endpoint at
    frame zero followed by exactly ``query.horizon_frames`` future frames.  The
    default projection removes that endpoint and preserves the requested node
    order.  The source posterior is not modified, and its identity is recorded
    in the projected artifact metadata.
    """

    if type(include_endpoint) is not bool:
        raise ValueError("include_endpoint must be boolean")
    if posterior.source_query_id != query.artifact_id:
        raise ValueError("physical posterior does not descend from the query")
    if posterior.context != query.context:
        raise ValueError("physical posterior and query contexts differ")
    expected_frame_count = query.horizon_frames + 1
    if posterior.state_trajectories_m.shape[1] != expected_frame_count:
        raise ValueError(
            "physical posterior must contain one endpoint frame plus the query horizon"
        )
    nodes = _validated_query_node_indices(
        query,
        node_count=posterior.state_trajectories_m.shape[2],
    )
    frame_start = 0 if include_endpoint else 1
    state = np.take(
        posterior.state_trajectories_m[:, frame_start:],
        nodes,
        axis=2,
    )
    readout = np.take(
        posterior.readout_trajectories_m[:, frame_start:],
        nodes,
        axis=2,
    )
    variance = np.take(posterior.readout_variance_m2, nodes, axis=1)
    metadata = dict(posterior.metadata)
    metadata.update(
        {
            "operator": "physical-posterior-query-projection",
            "source_physical_posterior_id": posterior.artifact_id,
            "rollout_includes_pre_intervention_endpoint": include_endpoint,
            "projection_includes_intervention_endpoint": include_endpoint,
            "projection_frame_start_relative_to_endpoint": frame_start,
            "projection_frame_stop_relative_to_endpoint": expected_frame_count,
            "projection_node_selection": (
                "all" if query.query_node_indices is None else "explicit"
            ),
            "projection_node_count": len(nodes),
            "projection_node_indices": (
                None
                if query.query_node_indices is None
                else nodes.tolist()
            ),
            "projection_preserves_component_weights": True,
        }
    )
    return PhysicalPosterior(
        context=posterior.context,
        component_ids=posterior.component_ids,
        state_trajectories_m=state,
        readout_trajectories_m=readout,
        readout_variance_m2=variance,
        weights=posterior.weights,
        phi=posterior.phi,
        kappa_cf=posterior.kappa_cf,
        hypothesis_indices=posterior.hypothesis_indices,
        twin_particle_indices=posterior.twin_particle_indices,
        phi_names=posterior.phi_names,
        kappa_names=posterior.kappa_names,
        source_twin_belief_id=posterior.source_twin_belief_id,
        source_factual_intervention_id=posterior.source_factual_intervention_id,
        source_query_id=posterior.source_query_id,
        metadata=metadata,
    )


def physical_posterior_mean(
    posterior: PhysicalPosterior,
    *,
    readout: bool = True,
) -> np.ndarray:
    """Return the weighted state or discrepancy-aware readout trajectory."""

    values = (
        posterior.readout_trajectories_m if readout else posterior.state_trajectories_m
    )
    return np.einsum("k,ktnc->tnc", posterior.weights, values)
