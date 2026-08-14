"""Construction and coupling of interventional contrast posteriors."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any, cast

import numpy as np

from causal4d.contracts import PhysicalPosterior
from causal4d.cross_branch_query_covariance import (
    RegisteredCrossBranchQueryCovarianceV1,
)
from causal4d.immutable_json import plain_json
from causal4d._interventional_contrast_common import (
    ContrastConditionalVariancePolicy,
    ContrastCouplingPolicy,
    _CLAIM_BOUNDARY,
    _require_mapping,
    _require_nonempty_string,
    _require_positive_integer,
    _validated_string_tuple,
)
from causal4d._interventional_contrast_posterior import (
    InterventionalContrastPosteriorV1,
)
from causal4d._interventional_contrast_query import (
    InterventionalContrastQueryV1,
)


def _validate_source_posteriors(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    query: InterventionalContrastQueryV1,
) -> tuple[tuple[int, int, int], int, int]:
    if not isinstance(branch_a, PhysicalPosterior) or not isinstance(
        branch_b,
        PhysicalPosterior,
    ):
        raise TypeError("both branches must be PhysicalPosterior instances")
    for name in ("protocol_id", "o_minus", "o_plus", "u_obs"):
        if getattr(branch_a.context, name) != getattr(branch_b.context, name):
            raise ValueError(f"contrast branches disagree on factual {name}")
    if branch_a.source_twin_belief_id != branch_b.source_twin_belief_id:
        raise ValueError("contrast branches descend from different TwinBeliefs")
    if (
        branch_a.source_factual_intervention_id
        != branch_b.source_factual_intervention_id
    ):
        raise ValueError(
            "contrast branches descend from different factual interventions"
        )
    if branch_a.phi_names != branch_b.phi_names:
        raise ValueError("contrast branches use different persistent phi schemas")
    if branch_a.kappa_names != branch_b.kappa_names:
        raise ValueError("contrast branches use different event kappa schemas")
    trajectories_a = np.asarray(branch_a.readout_trajectories_m)
    trajectories_b = np.asarray(branch_b.readout_trajectories_m)
    if trajectories_a.ndim != 4 or trajectories_a.shape[-1] != 3:
        raise ValueError("branch A readout trajectories have the wrong shape")
    if trajectories_b.ndim != 4 or trajectories_b.shape[-1] != 3:
        raise ValueError("branch B readout trajectories have the wrong shape")
    if trajectories_a.shape[1:] != trajectories_b.shape[1:]:
        raise ValueError("contrast branches have different trajectory shapes")
    shape = tuple(map(int, trajectories_a.shape[1:]))
    if query.trajectory_dimension != int(np.prod(shape)):
        raise ValueError("contrast query does not match the source trajectory")
    return shape, len(trajectories_a), len(trajectories_b)


def _positive_support(weights: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.asarray(weights, dtype=float) > 0.0)


def _shared_component_pairs(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
) -> tuple[np.ndarray, np.ndarray]:
    exact_fields = (
        "component_ids",
        "weights",
        "phi",
        "kappa_cf",
        "hypothesis_indices",
        "twin_particle_indices",
    )
    for name in exact_fields:
        first = getattr(branch_a, name)
        second = getattr(branch_b, name)
        if isinstance(first, np.ndarray):
            equal = np.array_equal(first, second)
        else:
            equal = first == second
        if not equal:
            raise ValueError(f"shared_component coupling requires identical {name}")
    selected = _positive_support(branch_a.weights)
    return (
        np.column_stack((selected, selected)).astype(np.int64, copy=False),
        np.asarray(branch_a.weights, dtype=float)[selected],
    )


def _twin_phi_keys(
    posterior: PhysicalPosterior,
    *,
    shared_kappa_indices: tuple[int, ...],
) -> tuple[tuple[Any, ...], ...]:
    phi = np.asarray(posterior.phi, dtype=float)
    kappa = np.asarray(posterior.kappa_cf, dtype=float)
    particles = np.asarray(posterior.twin_particle_indices, dtype=np.int64)
    return tuple(
        (
            int(particle),
            *tuple(map(float, phi_row)),
            *tuple(map(float, kappa_row[list(shared_kappa_indices)])),
        )
        for particle, phi_row, kappa_row in zip(
            particles,
            phi,
            kappa,
            strict=True,
        )
    )


def _shared_twin_phi_pairs(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    *,
    tolerance: float,
    maximum_pair_count: int,
    shared_kappa_indices: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    groups_a: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
    groups_b: defaultdict[tuple[Any, ...], list[int]] = defaultdict(list)
    weights_a = np.asarray(branch_a.weights, dtype=float)
    weights_b = np.asarray(branch_b.weights, dtype=float)
    for index, key in enumerate(
        _twin_phi_keys(
            branch_a,
            shared_kappa_indices=shared_kappa_indices,
        )
    ):
        if weights_a[index] > 0.0:
            groups_a[key].append(index)
    for index, key in enumerate(
        _twin_phi_keys(
            branch_b,
            shared_kappa_indices=shared_kappa_indices,
        )
    ):
        if weights_b[index] > 0.0:
            groups_b[key].append(index)
    if set(groups_a) != set(groups_b):
        raise ValueError("shared_twin_phi branches expose different positive strata")

    pair_rows: list[tuple[int, int]] = []
    pair_weights: list[float] = []
    for key in sorted(groups_a):
        indices_a = groups_a[key]
        indices_b = groups_b[key]
        mass_a = float(np.sum(weights_a[indices_a]))
        mass_b = float(np.sum(weights_b[indices_b]))
        if not np.isclose(mass_a, mass_b, atol=tolerance, rtol=tolerance):
            raise ValueError(
                "shared_twin_phi branches have different stratum marginal mass"
            )
        required_pairs = len(indices_a) * len(indices_b)
        if len(pair_rows) + required_pairs > maximum_pair_count:
            raise ValueError(
                "shared_twin_phi coupling exceeds "
                f"maximum_pair_count={maximum_pair_count}"
            )
        shared_mass = 0.5 * (mass_a + mass_b)
        conditional_a = weights_a[indices_a] / mass_a
        conditional_b = weights_b[indices_b] / mass_b
        for local_a, index_a in enumerate(indices_a):
            for local_b, index_b in enumerate(indices_b):
                pair_rows.append((index_a, index_b))
                pair_weights.append(
                    shared_mass
                    * float(conditional_a[local_a])
                    * float(conditional_b[local_b])
                )
    return (
        np.asarray(pair_rows, dtype=np.int64),
        np.asarray(pair_weights, dtype=float),
    )


def _independent_product_pairs(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    *,
    maximum_pair_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices_a = _positive_support(branch_a.weights)
    indices_b = _positive_support(branch_b.weights)
    required_pairs = len(indices_a) * len(indices_b)
    if required_pairs > maximum_pair_count:
        raise ValueError(
            "independent_product coupling requires "
            f"{required_pairs} pairs, exceeding "
            f"maximum_pair_count={maximum_pair_count}"
        )
    pairs = np.asarray(
        [(int(first), int(second)) for first in indices_a for second in indices_b],
        dtype=np.int64,
    )
    weights = np.asarray(
        [
            float(branch_a.weights[first]) * float(branch_b.weights[second])
            for first, second in pairs
        ],
        dtype=float,
    )
    return pairs, weights


def _query_components(
    posterior: PhysicalPosterior,
    query: InterventionalContrastQueryV1,
    *,
    variance_policy: ContrastConditionalVariancePolicy,
) -> tuple[np.ndarray, np.ndarray | None]:
    trajectories = np.asarray(posterior.readout_trajectories_m, dtype=float)
    flattened = trajectories.reshape(len(trajectories), -1)
    values = flattened @ query.matrix.T
    if variance_policy == "component_means_only":
        return values, None
    component_variance = np.asarray(posterior.readout_variance_m2, dtype=float)
    expected = (len(trajectories), trajectories.shape[2], trajectories.shape[3])
    if component_variance.shape != expected:
        raise ValueError("source readout variance has the wrong shape")
    if not np.all(np.isfinite(component_variance)) or np.any(component_variance < 0.0):
        raise ValueError("source readout variance must be finite and nonnegative")
    full_variance = np.broadcast_to(
        component_variance[:, None],
        trajectories.shape,
    ).reshape(len(trajectories), -1)
    covariance = np.einsum(
        "qi,ki,ri->kqr",
        query.matrix,
        full_variance,
        query.matrix,
    )
    return values, covariance


def _marginal_error(
    pairs: np.ndarray,
    weights: np.ndarray,
    source_weights: np.ndarray,
    *,
    column: int,
) -> float:
    reconstructed = np.bincount(
        pairs[:, column],
        weights=weights,
        minlength=len(source_weights),
    )
    return float(np.max(np.abs(reconstructed - source_weights), initial=0.0))


def _minimum_psd_eigenvalue(
    matrix: np.ndarray,
    *,
    name: str,
) -> float:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        float(np.linalg.norm(symmetric, ord=2)),
    )
    minimum = float(np.min(eigenvalues))
    if minimum < -1.0e-10 * scale:
        raise ValueError(f"{name} must be positive semidefinite")
    return minimum


def _registered_cross_branch_covariance(
    artifact: RegisteredCrossBranchQueryCovarianceV1,
    *,
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    query: InterventionalContrastQueryV1,
    pairs: np.ndarray,
    coupling_policy: ContrastCouplingPolicy,
    shared_kappa_names: tuple[str, ...],
    covariance_a: np.ndarray,
    covariance_b: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not isinstance(artifact, RegisteredCrossBranchQueryCovarianceV1):
        raise TypeError(
            "cross_branch_query_covariance must be "
            "RegisteredCrossBranchQueryCovarianceV1"
        )
    exact_bindings = {
        "source_branch_a_posterior_id": branch_a.artifact_id,
        "source_branch_b_posterior_id": branch_b.artifact_id,
        "source_branch_a_query_id": branch_a.source_query_id,
        "source_branch_b_query_id": branch_b.source_query_id,
        "query_id": query.query_id,
        "branch_a_component_count": len(branch_a.weights),
        "branch_b_component_count": len(branch_b.weights),
        "coupling_policy": coupling_policy,
        "shared_kappa_names": shared_kappa_names,
    }
    for name, expected in exact_bindings.items():
        if getattr(artifact, name) != expected:
            raise ValueError(
                f"registered cross-branch covariance {name} does not match "
                "the contrast problem"
            )
    if not np.array_equal(artifact.pair_indices, pairs):
        raise ValueError(
            "registered cross-branch covariance pair_indices do not match "
            "the exact contrast pair order"
        )
    if artifact.query_dimension != query.output_count:
        raise ValueError(
            "registered cross-branch covariance query dimension does not match"
        )

    result = np.empty_like(covariance_a[pairs[:, 0]])
    minimum_block = float("inf")
    minimum_contrast = float("inf")
    for index, (component_a, component_b) in enumerate(pairs):
        marginal_a = covariance_a[int(component_a)]
        marginal_b = covariance_b[int(component_b)]
        cross = artifact.cross_covariance[index]
        block = np.block(
            [
                [marginal_a, cross],
                [cross.T, marginal_b],
            ]
        )
        block_minimum = _minimum_psd_eigenvalue(
            block,
            name=f"registered cross-branch block covariance[{index}]",
        )
        contrast = marginal_a + marginal_b - cross - cross.T
        contrast = 0.5 * (contrast + contrast.T)
        contrast_minimum = _minimum_psd_eigenvalue(
            contrast,
            name=f"registered contrast covariance[{index}]",
        )
        result[index] = contrast
        minimum_block = min(minimum_block, block_minimum)
        minimum_contrast = min(minimum_contrast, contrast_minimum)
    return result, {
        "artifact_id": artifact.artifact_id,
        "source_artifact_ids": list(artifact.source_artifact_ids),
        "source_only": artifact.source_only,
        "registered_before_target_access": artifact.registered_before_target_access,
        "minimum_block_covariance_eigenvalue": minimum_block,
        "minimum_contrast_covariance_eigenvalue": minimum_contrast,
    }


def build_interventional_contrast(
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    query: InterventionalContrastQueryV1,
    *,
    branch_a_label: str,
    branch_b_label: str,
    coupling_policy: ContrastCouplingPolicy = "shared_component",
    shared_kappa_names: Sequence[str] = (),
    conditional_variance_policy: ContrastConditionalVariancePolicy = (
        "component_means_only"
    ),
    cross_branch_query_covariance: (
        RegisteredCrossBranchQueryCovarianceV1 | None
    ) = None,
    maximum_pair_count: int = 1_000_000,
    marginal_tolerance: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalContrastPosteriorV1:
    """Build ``Q(branch_a) - Q(branch_b)`` without changing either branch.

    ``shared_component`` pairs identical complete finite-support components.
    ``shared_twin_phi`` shares the physical particle, persistent realization,
    and any explicitly named event coordinates while drawing all remaining event
    variables conditionally and independently in each branch.
    ``independent_product`` forms an uncoupled product diagnostic and must not be
    interpreted as an individual-level cross-world effect.

    ``component_means_only`` omits conditional uncertainty.
    ``independent_readout`` adds the two branch query covariances under an
    explicit zero-cross-covariance assumption. ``registered_cross_branch``
    requires an exact source-only covariance artifact, verifies the complete
    joint block covariance for every coupled pair, and then applies
    ``Cov(Q_a-Q_b)=C_a+C_b-C_ab-C_ab.T``. Unrecorded cancellation is never
    inferred.
    """

    if not isinstance(query, InterventionalContrastQueryV1):
        raise TypeError("query must be InterventionalContrastQueryV1")
    shape, count_a, count_b = _validate_source_posteriors(
        branch_a,
        branch_b,
        query,
    )
    label_a = _require_nonempty_string(branch_a_label, name="branch_a_label")
    label_b = _require_nonempty_string(branch_b_label, name="branch_b_label")
    if label_a == label_b:
        raise ValueError("branch labels must be distinct")
    pair_limit = _require_positive_integer(
        maximum_pair_count,
        name="maximum_pair_count",
    )
    if (
        isinstance(marginal_tolerance, (bool, np.bool_))
        or not isinstance(marginal_tolerance, (int, float, np.integer, np.floating))
        or not np.isfinite(marginal_tolerance)
        or float(marginal_tolerance) < 0.0
    ):
        raise ValueError("marginal_tolerance must be finite and nonnegative")
    tolerance = float(marginal_tolerance)
    user_metadata: Mapping[str, Any]
    if metadata is None:
        user_metadata = {}
    else:
        user_metadata = _require_mapping(metadata, name="metadata")
    coupling = _require_nonempty_string(
        coupling_policy,
        name="coupling_policy",
    )
    validated_coupling = cast(ContrastCouplingPolicy, coupling)
    supplied_shared_names = _validated_string_tuple(
        shared_kappa_names,
        name="shared_kappa_names",
        unique=True,
        allow_empty=True,
    )
    if validated_coupling != "shared_twin_phi" and supplied_shared_names:
        raise ValueError("shared_kappa_names require shared_twin_phi coupling")
    unknown_shared_names = sorted(
        set(supplied_shared_names) - set(branch_a.kappa_names)
    )
    if unknown_shared_names:
        raise ValueError(
            "shared_kappa_names reference unavailable event variables: "
            f"{unknown_shared_names}"
        )
    supplied_shared_name_set = set(supplied_shared_names)
    shared_names = tuple(
        name for name in branch_a.kappa_names if name in supplied_shared_name_set
    )
    shared_indices = tuple(branch_a.kappa_names.index(name) for name in shared_names)
    if validated_coupling == "shared_component":
        pairs, weights = _shared_component_pairs(branch_a, branch_b)
    elif validated_coupling == "shared_twin_phi":
        pairs, weights = _shared_twin_phi_pairs(
            branch_a,
            branch_b,
            tolerance=tolerance,
            maximum_pair_count=pair_limit,
            shared_kappa_indices=shared_indices,
        )
    elif validated_coupling == "independent_product":
        pairs, weights = _independent_product_pairs(
            branch_a,
            branch_b,
            maximum_pair_count=pair_limit,
        )
    else:
        raise ValueError("unsupported contrast coupling policy")
    if len(pairs) > pair_limit:
        raise ValueError(
            f"contrast coupling requires {len(pairs)} pairs, exceeding "
            f"maximum_pair_count={pair_limit}"
        )
    if len(pairs) == 0:
        raise ValueError("contrast coupling produced no positive-mass support")
    weights = weights / np.sum(weights)
    error_a = _marginal_error(
        pairs,
        weights,
        np.asarray(branch_a.weights, dtype=float),
        column=0,
    )
    error_b = _marginal_error(
        pairs,
        weights,
        np.asarray(branch_b.weights, dtype=float),
        column=1,
    )
    if max(error_a, error_b) > tolerance:
        raise ValueError("contrast coupling does not preserve source marginals")

    variance_policy = _require_nonempty_string(
        conditional_variance_policy,
        name="conditional_variance_policy",
    )
    if variance_policy not in {
        "component_means_only",
        "independent_readout",
        "registered_cross_branch",
    }:
        raise ValueError("unsupported conditional variance policy")
    if variance_policy == "registered_cross_branch":
        if cross_branch_query_covariance is None:
            raise ValueError(
                "registered_cross_branch requires cross_branch_query_covariance"
            )
    elif cross_branch_query_covariance is not None:
        raise ValueError(
            "cross_branch_query_covariance requires the "
            "registered_cross_branch variance policy"
        )
    validated_variance_policy = cast(
        ContrastConditionalVariancePolicy,
        variance_policy,
    )
    values_a, covariance_a = _query_components(
        branch_a,
        query,
        variance_policy=validated_variance_policy,
    )
    values_b, covariance_b = _query_components(
        branch_b,
        query,
        variance_policy=validated_variance_policy,
    )
    contrast_values = values_a[pairs[:, 0]] - values_b[pairs[:, 1]]
    cross_covariance_metadata: dict[str, Any] | None = None
    if covariance_a is None or covariance_b is None:
        conditional_covariance = np.zeros(
            (len(pairs), query.output_count, query.output_count),
            dtype=float,
        )
    elif validated_variance_policy == "registered_cross_branch":
        assert cross_branch_query_covariance is not None
        conditional_covariance, cross_covariance_metadata = (
            _registered_cross_branch_covariance(
                cross_branch_query_covariance,
                branch_a=branch_a,
                branch_b=branch_b,
                query=query,
                pairs=pairs,
                coupling_policy=validated_coupling,
                shared_kappa_names=shared_names,
                covariance_a=covariance_a,
                covariance_b=covariance_b,
            )
        )
    else:
        conditional_covariance = covariance_a[pairs[:, 0]] + covariance_b[pairs[:, 1]]

    result_metadata = {
        "claim_boundary": _CLAIM_BOUNDARY,
        "contrast_definition": "Q(branch_a) - Q(branch_b)",
        "branch_a_intervention": branch_a.context.u_cf.action_id,
        "branch_b_intervention": branch_b.context.u_cf.action_id,
        "source_marginal_max_abs_error": {
            "branch_a": error_a,
            "branch_b": error_b,
        },
        "source_marginal_tolerance": tolerance,
        "shared_kappa_names": list(shared_names),
        "cross_branch_discrepancy_covariance_available": False,
        "user": plain_json(user_metadata),
    }
    if cross_covariance_metadata is not None:
        result_metadata.update(
            {
                "cross_branch_discrepancy_covariance_available": True,
                "cross_branch_query_covariance_available": True,
                "registered_cross_branch_query_covariance": cross_covariance_metadata,
            }
        )
    return InterventionalContrastPosteriorV1(
        source_branch_a_posterior_id=branch_a.artifact_id,
        source_branch_b_posterior_id=branch_b.artifact_id,
        source_branch_a_query_id=branch_a.source_query_id,
        source_branch_b_query_id=branch_b.source_query_id,
        branch_a_label=label_a,
        branch_b_label=label_b,
        trajectory_shape=shape,
        branch_a_component_count=count_a,
        branch_b_component_count=count_b,
        coupling_policy=validated_coupling,
        shared_kappa_names=shared_names,
        conditional_variance_policy=validated_variance_policy,
        query_name=query.name,
        query_matrix=query.matrix,
        query_labels=query.labels,
        query_units=query.units,
        query_metadata=query.metadata,
        pair_indices=pairs,
        weights=weights,
        contrast_values=contrast_values,
        conditional_covariance=conditional_covariance,
        metadata=result_metadata,
    )
