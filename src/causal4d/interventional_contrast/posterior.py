"""Finite-support posterior contract for interventional contrasts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import validated_json_mapping

from causal4d.interventional_contrast.derived import (
    InterventionalContrastDerivedMixin,
)
from causal4d.interventional_contrast.specification import (
    INTERVENTIONAL_CONTRAST_CLAIM_BOUNDARY,
    CouplingPolicy,
    InterventionalContrastSpecificationV1,
    ResolvedCouplingPolicy,
    _ALLOWED_COUPLING_POLICIES,
    _BOUNDARY_METADATA,
    _require_nonempty_string,
    _require_sha256,
    _validated_string_tuple,
    _validated_weights,
)


@dataclass(frozen=True, slots=True)
class InterventionalContrastPosteriorV1(InterventionalContrastDerivedMixin):
    """Finite-support posterior for ``Q(Y_left) - Q(Y_right)``."""

    specification: InterventionalContrastSpecificationV1
    source_twin_belief_id: str
    source_factual_intervention_id: str
    left_posterior_id: str
    right_posterior_id: str
    left_query_id: str
    right_query_id: str
    left_action_id: str
    right_action_id: str
    left_action_trajectory_sha256: str
    right_action_trajectory_sha256: str
    left_contact_policy: str
    right_contact_policy: str
    left_same_grasp_semantics: str
    right_same_grasp_semantics: str
    requested_coupling_policy: CouplingPolicy
    resolved_coupling_policy: ResolvedCouplingPolicy
    shared_variables: tuple[str, ...]
    left_component_ids: tuple[str, ...]
    right_component_ids: tuple[str, ...]
    pair_indices: np.ndarray
    pair_weights: np.ndarray
    left_weights: np.ndarray
    right_weights: np.ndarray
    contrast_components_m: np.ndarray
    component_conditional_variance_m2: np.ndarray
    expected_conditional_covariance_m2: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    claim_boundary: str = INTERVENTIONAL_CONTRAST_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        if not isinstance(
            self.specification,
            InterventionalContrastSpecificationV1,
        ):
            raise TypeError("specification has the wrong type")
        for name in (
            "source_twin_belief_id",
            "source_factual_intervention_id",
            "left_posterior_id",
            "right_posterior_id",
            "left_query_id",
            "right_query_id",
            "left_action_trajectory_sha256",
            "right_action_trajectory_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        for name in (
            "left_action_id",
            "right_action_id",
            "left_contact_policy",
            "right_contact_policy",
            "left_same_grasp_semantics",
            "right_same_grasp_semantics",
        ):
            _require_nonempty_string(getattr(self, name), name=name)
        requested_policy = _require_nonempty_string(
            self.requested_coupling_policy,
            name="requested_coupling_policy",
        )
        if requested_policy not in _ALLOWED_COUPLING_POLICIES:
            raise ValueError("requested coupling policy is unsupported")
        resolved_policy = _require_nonempty_string(
            self.resolved_coupling_policy,
            name="resolved_coupling_policy",
        )
        if resolved_policy not in (_ALLOWED_COUPLING_POLICIES - {"auto"}):
            raise ValueError("resolved coupling policy is unsupported")
        shared_variables = _validated_string_tuple(
            self.shared_variables,
            name="shared_variables",
            allow_empty=True,
        )
        left_component_ids = _validated_string_tuple(
            self.left_component_ids,
            name="left_component_ids",
        )
        right_component_ids = _validated_string_tuple(
            self.right_component_ids,
            name="right_component_ids",
        )
        if len(set(left_component_ids)) != len(left_component_ids):
            raise ValueError("left component IDs must be unique")
        if len(set(right_component_ids)) != len(right_component_ids):
            raise ValueError("right component IDs must be unique")
        pairs = readonly_integer_array(self.pair_indices, name="pair_indices")
        pair_weights = _validated_weights(self.pair_weights, name="pair_weights")
        left_weights = _validated_weights(self.left_weights, name="left_weights")
        right_weights = _validated_weights(self.right_weights, name="right_weights")
        contrasts = readonly_array(self.contrast_components_m, dtype=float)
        conditional_variance = readonly_array(
            self.component_conditional_variance_m2,
            dtype=float,
        )
        expected_conditional_covariance = readonly_array(
            self.expected_conditional_covariance_m2,
            dtype=float,
        )
        pair_count = len(pair_weights)
        query_count = len(self.specification.query_labels)
        if pairs.shape != (pair_count, 2):
            raise ValueError("pair_indices must have shape (pair, 2)")
        if contrasts.shape != (pair_count, query_count):
            raise ValueError(
                "contrast_components_m must have shape (pair, query)"
            )
        if conditional_variance.shape != contrasts.shape:
            raise ValueError(
                "component conditional variances must match contrast components"
            )
        if expected_conditional_covariance.shape != (query_count, query_count):
            raise ValueError(
                "expected conditional covariance must have shape (query, query)"
            )
        if np.any(pairs[:, 0] < 0) or np.any(
            pairs[:, 0] >= len(left_component_ids)
        ):
            raise ValueError("left pair indices are out of range")
        if np.any(pairs[:, 1] < 0) or np.any(
            pairs[:, 1] >= len(right_component_ids)
        ):
            raise ValueError("right pair indices are out of range")
        if len(left_weights) != len(left_component_ids):
            raise ValueError("left weights do not match left components")
        if len(right_weights) != len(right_component_ids):
            raise ValueError("right weights do not match right components")
        if not np.all(np.isfinite(contrasts)):
            raise ValueError("contrast components must be finite")
        if not np.all(np.isfinite(conditional_variance)) or np.any(
            conditional_variance < 0.0
        ):
            raise ValueError(
                "component conditional variances must be finite and nonnegative"
            )
        if not np.all(np.isfinite(expected_conditional_covariance)):
            raise ValueError("expected conditional covariance must be finite")
        if not np.allclose(
            expected_conditional_covariance,
            expected_conditional_covariance.T,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("expected conditional covariance must be symmetric")
        minimum_eigenvalue = float(
            np.min(np.linalg.eigvalsh(expected_conditional_covariance), initial=0.0)
        )
        if minimum_eigenvalue < -1e-10:
            raise ValueError(
                "expected conditional covariance must be positive semidefinite"
            )
        expected_diagonal = np.einsum(
            "p,pq->q",
            pair_weights,
            conditional_variance,
        )
        if not np.allclose(
            np.diag(expected_conditional_covariance),
            expected_diagonal,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(
                "conditional covariance diagonal disagrees with component variances"
            )
        left_marginal = np.zeros(len(left_weights), dtype=float)
        right_marginal = np.zeros(len(right_weights), dtype=float)
        np.add.at(left_marginal, pairs[:, 0], pair_weights)
        np.add.at(right_marginal, pairs[:, 1], pair_weights)
        if not np.allclose(
            left_marginal,
            left_weights,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError("coupling does not preserve the left posterior marginal")
        if not np.allclose(
            right_marginal,
            right_weights,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(
                "coupling does not preserve the right posterior marginal"
            )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="contrast metadata must contain finite JSON",
        )
        for key, expected in _BOUNDARY_METADATA.items():
            if metadata.get(key) is not expected:
                raise ValueError(
                    f"contrast metadata {key!r} must remain {expected}"
                )
        claim_boundary = _require_nonempty_string(
            self.claim_boundary,
            name="claim_boundary",
        )
        if claim_boundary != INTERVENTIONAL_CONTRAST_CLAIM_BOUNDARY:
            raise ValueError("interventional contrast claim boundary changed")
        expected_shared_variables = {
            "independent": (),
            "component_id": ("component_id",),
            "shared_theta_phi": ("theta", "phi"),
            "shared_theta_phi_patch": ("theta", "phi", "contact_patch"),
            "shared_theta_phi_kappa": ("theta", "phi", "kappa"),
        }[resolved_policy]
        if shared_variables != expected_shared_variables:
            raise ValueError(
                "shared_variables disagree with the resolved coupling policy"
            )
        if self.specification.conditional_readout_correlation is None and (
            np.any(conditional_variance != 0.0)
            or np.any(expected_conditional_covariance != 0.0)
        ):
            raise ValueError(
                "conditional uncertainty must be zero when no correlation model "
                "is declared"
            )
        object.__setattr__(self, "requested_coupling_policy", requested_policy)
        object.__setattr__(self, "resolved_coupling_policy", resolved_policy)
        object.__setattr__(self, "shared_variables", shared_variables)
        object.__setattr__(self, "left_component_ids", left_component_ids)
        object.__setattr__(self, "right_component_ids", right_component_ids)
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "pair_weights", pair_weights)
        object.__setattr__(self, "left_weights", left_weights)
        object.__setattr__(self, "right_weights", right_weights)
        object.__setattr__(self, "contrast_components_m", contrasts)
        object.__setattr__(
            self,
            "component_conditional_variance_m2",
            conditional_variance,
        )
        object.__setattr__(
            self,
            "expected_conditional_covariance_m2",
            expected_conditional_covariance,
        )
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "claim_boundary", claim_boundary)
