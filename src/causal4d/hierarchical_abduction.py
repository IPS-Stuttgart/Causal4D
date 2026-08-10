"""Hierarchical multi-execution abduction over finite rollout banks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from causal4d.immutable_array import readonly_array as _readonly_array
from causal4d.immutable_json import validated_json_mapping
from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    prefix_component_log_likelihood,
)
from causal4d.rollout_bank import JointRolloutBank
from causal4d.session_hierarchy import (
    SessionHierarchyPosterior,
    infer_session_phi_hierarchy,
)
from causal4d.weighting import log_weights_from_probabilities


DEFAULT_PHI_NAMES = ("gain_multiplier", "delay_steps", "rotation_degrees")


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    summed = np.sum(np.exp(array - safe_maximum), axis=axis, keepdims=True)
    result = safe_maximum + np.log(summed)
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return np.where(np.squeeze(np.isfinite(maximum), axis=axis), result, -np.inf)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    normalizer = float(np.ravel(_logsumexp(np.asarray(log_weights, dtype=float)))[0])
    if not np.isfinite(normalizer):
        raise RuntimeError("hierarchical posterior normalization failed")
    return np.exp(log_weights - normalizer)


def _default_phi_values(bank: JointRolloutBank) -> np.ndarray:
    rows = []
    for metadata in bank.hypothesis_metadata:
        contact = metadata.get("contact")
        if not isinstance(contact, Mapping):
            raise ValueError("hypothesis metadata is missing contact variables")
        rows.append(
            (
                float(contact["gain_multiplier"]),
                float(contact["delay_steps"]),
                float(contact["rotation_degrees"]),
            )
        )
    return np.asarray(rows, dtype=float)


def _ordered_unique_rows(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2 or not np.all(np.isfinite(rows)):
        raise ValueError("phi values must be a finite matrix")
    keys: list[tuple[float, ...]] = []
    indices = np.empty(rows.shape[0], dtype=np.int64)
    lookup: dict[tuple[float, ...], int] = {}
    for row_index, row in enumerate(rows):
        key = tuple(float(value) for value in row)
        group = lookup.get(key)
        if group is None:
            group = len(keys)
            lookup[key] = group
            keys.append(key)
        indices[row_index] = group
    return np.asarray(keys, dtype=float), indices


def _session_evidence_powers(
    execution_count: int,
    session_ids: Sequence[str] | None,
    execution_evidence_powers: Sequence[float] | None,
) -> tuple[tuple[str, ...], np.ndarray, str]:
    if execution_evidence_powers is not None:
        powers = np.asarray(tuple(execution_evidence_powers), dtype=float)
        if powers.shape != (execution_count,) or not np.all(np.isfinite(powers)):
            raise ValueError("execution_evidence_powers must match executions")
        if np.any(powers <= 0.0):
            raise ValueError("execution_evidence_powers must be positive")
        if session_ids is None:
            identifiers = tuple(
                f"execution-{index}" for index in range(execution_count)
            )
        else:
            identifiers = tuple(map(str, session_ids))
            if len(identifiers) != execution_count or any(
                not value for value in identifiers
            ):
                raise ValueError("session_ids must be nonempty and match executions")
        return identifiers, powers, "explicit_execution_powers"

    if session_ids is None:
        return (
            tuple(f"execution-{index}" for index in range(execution_count)),
            np.ones(execution_count, dtype=float),
            "independent_execution_product",
        )

    identifiers = tuple(map(str, session_ids))
    if len(identifiers) != execution_count or any(not value for value in identifiers):
        raise ValueError("session_ids must be nonempty and match executions")
    counts = Counter(identifiers)
    powers = np.asarray([1.0 / counts[value] for value in identifiers], dtype=float)
    return identifiers, powers, "equal_session_composite_likelihood"


@dataclass(frozen=True)
class HierarchicalAbductionResult:
    """Posterior with persistent, session-local, and execution-local variables."""

    phi_names: tuple[str, ...]
    phi_values: np.ndarray
    parameter_particles: np.ndarray
    shared_weights: np.ndarray
    execution_joint_weights: tuple[np.ndarray, ...]
    execution_log_evidence: tuple[np.ndarray, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    session_hierarchy: SessionHierarchyPosterior | None = None

    def __post_init__(self) -> None:
        phi = _readonly_array(self.phi_values)
        particles = _readonly_array(self.parameter_particles)
        shared = _readonly_array(self.shared_weights)
        if phi.ndim != 2 or phi.shape[1] != len(self.phi_names):
            raise ValueError("phi_values must match phi_names")
        if particles.ndim != 2:
            raise ValueError("parameter_particles must be a matrix")
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(particles)):
            raise ValueError("hierarchical support arrays must be finite")
        if shared.shape != (len(phi), len(particles)):
            raise ValueError("shared_weights must have shape (Phi, P)")
        if (
            not np.all(np.isfinite(shared))
            or np.any(shared < 0.0)
            or not np.isclose(np.sum(shared), 1.0)
        ):
            raise ValueError(
                "shared_weights must be finite, nonnegative, and sum to one"
            )
        if len(self.execution_joint_weights) != len(self.execution_log_evidence):
            raise ValueError("execution posterior and evidence counts differ")
        execution_weights = []
        execution_evidence = []
        for weights, evidence in zip(
            self.execution_joint_weights,
            self.execution_log_evidence,
            strict=True,
        ):
            supplied = _readonly_array(weights)
            log_evidence = _readonly_array(evidence)
            if supplied.ndim != 2 or supplied.shape[1] != len(particles):
                raise ValueError("execution weights must have shape (H_e, P)")
            if (
                not np.all(np.isfinite(supplied))
                or np.any(supplied < 0.0)
                or not np.isclose(np.sum(supplied), 1.0)
            ):
                raise ValueError("execution weights must be finite and sum to one")
            if log_evidence.shape != shared.shape:
                raise ValueError("execution log evidence must have shape (Phi, P)")
            if np.any(np.isnan(log_evidence)) or np.any(np.isposinf(log_evidence)):
                raise ValueError("execution log evidence must not contain NaN or +inf")
            execution_weights.append(supplied)
            execution_evidence.append(log_evidence)
        hierarchy = self.session_hierarchy
        if hierarchy is not None:
            if hierarchy.global_weights.shape != shared.shape or not np.array_equal(
                hierarchy.global_weights,
                shared,
            ):
                raise ValueError(
                    "session hierarchy global weights must equal shared_weights"
                )
            if hierarchy.session_phi_transition.shape != (len(phi), len(phi)):
                raise ValueError("session hierarchy must match the phi support")
            if len(hierarchy.execution_session_ids) != len(execution_weights):
                raise ValueError("session hierarchy must cover every execution")
        object.__setattr__(self, "phi_values", phi)
        object.__setattr__(self, "parameter_particles", particles)
        object.__setattr__(self, "shared_weights", shared)
        object.__setattr__(self, "execution_joint_weights", tuple(execution_weights))
        object.__setattr__(self, "execution_log_evidence", tuple(execution_evidence))
        object.__setattr__(self, "metadata", validated_json_mapping(self.metadata))

    @property
    def phi_marginal(self) -> np.ndarray:
        """Marginal of shared ``phi`` or global ``phi_bar`` support."""

        return np.sum(self.shared_weights, axis=1)

    @property
    def parameter_marginal(self) -> np.ndarray:
        return np.sum(self.shared_weights, axis=0)

    @property
    def session_phi_marginals(self) -> tuple[np.ndarray, ...]:
        """Ordered session-local marginals, empty for the shared-``phi`` model."""

        if self.session_hierarchy is None:
            return ()
        return self.session_hierarchy.session_phi_marginals


def abduct_hierarchical_interventions(
    banks: Sequence[JointRolloutBank],
    observations_m: Sequence[np.ndarray],
    *,
    prefix_frame_counts: Sequence[int],
    config: PrefixLikelihoodConfig | None = None,
    masks: Sequence[np.ndarray | None] | None = None,
    phi_values_by_bank: Sequence[np.ndarray] | None = None,
    phi_names: tuple[str, ...] = DEFAULT_PHI_NAMES,
    shared_phi_prior: np.ndarray | None = None,
    particle_discrepancy_m: Sequence[np.ndarray | None] | None = None,
    particle_discrepancy_variance_m2: Sequence[np.ndarray | None] | None = None,
    session_ids: Sequence[str] | None = None,
    execution_evidence_powers: Sequence[float] | None = None,
    session_phi_transition: np.ndarray | None = None,
) -> HierarchicalAbductionResult:
    """Pool persistent intervention and twin variables across executions.

    Physical parameter particles and persistent intervention variables ``phi``
    are shared by default. Each execution retains its own local hypothesis within
    a ``phi`` group, so contact and slip variables remain event-specific.

    When ``session_ids`` are supplied, execution log evidences are weighted by
    the reciprocal number of executions in that session. Each grasp/session then
    contributes one unit of composite evidence to shared variables, while every
    execution retains its full local ``kappa`` posterior. Supplying neither
    session IDs nor explicit powers preserves the original independent-execution
    product exactly.

    ``session_phi_transition[g, f]`` optionally specifies
    ``p(phi_s=f | phi_bar=g)`` on the same finite support. This introduces a
    session-local persistent intervention while retaining a global hardware
    hyperstate and shared physical particles. Exact zeros preserve excluded
    support. An identity transition is the zero-session-variance limit and
    reproduces the original shared-``phi`` weights and execution posteriors.
    """

    bank_list = tuple(banks)
    observation_list = tuple(observations_m)
    prefix_list = tuple(prefix_frame_counts)
    if not bank_list:
        raise ValueError("at least one execution is required")
    if len(observation_list) != len(bank_list) or len(prefix_list) != len(bank_list):
        raise ValueError("banks, observations, and prefix counts must align")
    mask_list = tuple(masks) if masks is not None else (None,) * len(bank_list)
    discrepancy_list = (
        tuple(particle_discrepancy_m)
        if particle_discrepancy_m is not None
        else (None,) * len(bank_list)
    )
    variance_list = (
        tuple(particle_discrepancy_variance_m2)
        if particle_discrepancy_variance_m2 is not None
        else (None,) * len(bank_list)
    )
    if not (
        len(mask_list) == len(discrepancy_list) == len(variance_list) == len(bank_list)
    ):
        raise ValueError("execution-specific optional inputs must align")
    session_identifiers, evidence_powers, evidence_mode = _session_evidence_powers(
        len(bank_list),
        session_ids,
        execution_evidence_powers,
    )

    reference = bank_list[0]
    for bank in bank_list[1:]:
        if not np.array_equal(bank.parameter_particles, reference.parameter_particles):
            raise ValueError("all executions must share parameter particles")
        if not np.allclose(
            bank.parameter_weights,
            reference.parameter_weights,
            rtol=0.0,
            atol=1e-15,
        ):
            raise ValueError("all executions must share parameter prior weights")

    raw_phi = (
        tuple(np.asarray(values, dtype=float) for values in phi_values_by_bank)
        if phi_values_by_bank is not None
        else tuple(_default_phi_values(bank) for bank in bank_list)
    )
    if len(raw_phi) != len(bank_list):
        raise ValueError("phi_values_by_bank must align with executions")

    reference_phi, reference_group = _ordered_unique_rows(raw_phi[0])
    if reference_phi.shape[1] != len(phi_names):
        raise ValueError("phi values do not match phi_names")
    if raw_phi[0].shape[0] != len(reference.hypothesis_ids):
        raise ValueError("each hypothesis needs one phi row")
    reference_lookup = {
        tuple(float(value) for value in row): index
        for index, row in enumerate(reference_phi)
    }
    group_indices = [reference_group]
    for values, bank in zip(raw_phi[1:], bank_list[1:], strict=True):
        if values.shape != (len(bank.hypothesis_ids), len(phi_names)):
            raise ValueError("each hypothesis needs one phi row")
        indices = np.empty(len(values), dtype=np.int64)
        seen: set[int] = set()
        for row_index, row in enumerate(values):
            key = tuple(float(value) for value in row)
            if key not in reference_lookup:
                raise ValueError("executions expose different phi support")
            indices[row_index] = reference_lookup[key]
            seen.add(int(indices[row_index]))
        if seen != set(range(len(reference_phi))):
            raise ValueError("executions expose different phi support")
        group_indices.append(indices)

    phi_count = len(reference_phi)
    if shared_phi_prior is None:
        phi_prior = np.zeros(phi_count, dtype=float)
        np.add.at(phi_prior, reference_group, reference.hypothesis_prior_weights)
    else:
        phi_prior = np.asarray(shared_phi_prior, dtype=float)
        if phi_prior.shape != (phi_count,):
            raise ValueError("shared_phi_prior must match the phi support")
    if np.any(phi_prior < 0.0) or float(np.sum(phi_prior)) <= 0.0:
        raise ValueError("shared_phi_prior must contain nonnegative mass")
    phi_prior = phi_prior / np.sum(phi_prior)

    local_log_priors: list[np.ndarray] = []
    execution_log_evidence: list[np.ndarray] = []
    component_log_likelihoods: list[np.ndarray] = []
    for execution, bank in enumerate(bank_list):
        groups = group_indices[execution]
        group_mass = np.zeros(phi_count, dtype=float)
        np.add.at(group_mass, groups, bank.hypothesis_prior_weights)
        if np.any(group_mass <= 0.0):
            raise ValueError(
                "every execution must assign prior mass to every phi value"
            )
        conditional_prior = bank.hypothesis_prior_weights / group_mass[groups]
        local_log_prior = log_weights_from_probabilities(
            conditional_prior,
            name="conditional hypothesis prior",
        )
        local_log_priors.append(local_log_prior)
        log_likelihood = prefix_component_log_likelihood(
            bank,
            observation_list[execution],
            prefix_frame_count=prefix_list[execution],
            config=config,
            mask=mask_list[execution],
            particle_discrepancy_m=discrepancy_list[execution],
            particle_discrepancy_variance_m2=variance_list[execution],
        )
        component_log_likelihoods.append(log_likelihood)
        evidence = np.empty((phi_count, len(reference.parameter_weights)), dtype=float)
        for group in range(phi_count):
            selected = groups == group
            evidence[group] = _logsumexp(
                local_log_prior[selected, None] + log_likelihood[selected],
                axis=0,
            )
        execution_log_evidence.append(evidence)

    session_hierarchy: SessionHierarchyPosterior | None = None
    if session_phi_transition is None:
        shared_log_weights = (
            log_weights_from_probabilities(phi_prior, name="shared phi prior")[:, None]
            + log_weights_from_probabilities(
                reference.parameter_weights,
                name="shared parameter prior",
            )[None]
        )
        for power, evidence in zip(
            evidence_powers,
            execution_log_evidence,
            strict=True,
        ):
            shared_log_weights += float(power) * evidence
        shared_weights = _normalize_log_weights(shared_log_weights)
        session_weight_lookup: dict[str, np.ndarray] = {}
        hierarchy_mode = "shared_phi"
    else:
        session_hierarchy = infer_session_phi_hierarchy(
            execution_log_evidence,
            phi_prior=phi_prior,
            parameter_prior=reference.parameter_weights,
            session_ids=session_identifiers,
            execution_evidence_powers=evidence_powers,
            session_phi_transition=session_phi_transition,
        )
        shared_weights = session_hierarchy.global_weights
        session_weight_lookup = dict(
            zip(
                session_hierarchy.session_ids,
                session_hierarchy.session_joint_weights,
                strict=True,
            )
        )
        hierarchy_mode = session_hierarchy.mode

    execution_joint_weights = []
    for execution, (groups, local_prior, log_likelihood, evidence) in enumerate(
        zip(
            group_indices,
            local_log_priors,
            component_log_likelihoods,
            execution_log_evidence,
            strict=True,
        )
    ):
        conditional = np.exp(local_prior[:, None] + log_likelihood - evidence[groups])
        persistent_weights = (
            shared_weights
            if session_hierarchy is None
            else session_weight_lookup[session_identifiers[execution]]
        )
        joint = persistent_weights[groups] * conditional
        joint /= np.sum(joint)
        execution_joint_weights.append(joint)

    result_metadata: dict[str, Any] = {
        "execution_count": len(bank_list),
        "session_count": len(set(session_identifiers)),
        "session_ids": list(session_identifiers),
        "execution_evidence_powers": evidence_powers.tolist(),
        "shared_evidence_mode": evidence_mode,
        "shared_variables": ["theta", "phi"],
        "execution_specific_variables": ["kappa"],
        "future_frames_read": 0,
    }
    if session_hierarchy is not None:
        result_metadata.update(
            {
                "session_hierarchy_mode": hierarchy_mode,
                "shared_variables": ["theta", "phi_bar"],
                "session_variables": ["phi_session"],
            }
        )

    return HierarchicalAbductionResult(
        phi_names=phi_names,
        phi_values=reference_phi,
        parameter_particles=reference.parameter_particles,
        shared_weights=shared_weights,
        execution_joint_weights=tuple(execution_joint_weights),
        execution_log_evidence=tuple(execution_log_evidence),
        metadata=result_metadata,
        session_hierarchy=session_hierarchy,
    )
