"""Derived summaries and identities for interventional contrasts."""

from __future__ import annotations

from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json
from causal4d.interventional_contrast.specification import (
    INTERVENTIONAL_CONTRAST_ARTIFACT_KIND,
    INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
    _canonical_id,
)
from causal4d.interventional_contrast.statistics import (
    mixture_probability_positive,
    mixture_quantile,
)


class InterventionalContrastDerivedMixin:
    """Read-only posterior summaries shared by the concrete contract."""

    @property
    def posterior_mean_m(self) -> np.ndarray:
        return readonly_array(
            np.einsum(
                "p,pq->q",
                self.pair_weights,
                self.contrast_components_m,
            ),
            dtype=float,
        )

    @property
    def between_component_covariance_m2(self) -> np.ndarray:
        centered = self.contrast_components_m - self.posterior_mean_m
        covariance = np.einsum(
            "p,pi,pj->ij",
            self.pair_weights,
            centered,
            centered,
        )
        return readonly_array(covariance, dtype=float)

    @property
    def posterior_covariance_m2(self) -> np.ndarray:
        covariance = (
            self.between_component_covariance_m2
            + self.expected_conditional_covariance_m2
        )
        covariance = 0.5 * (covariance + covariance.T)
        return readonly_array(covariance, dtype=float)

    @property
    def probability_positive(self) -> np.ndarray:
        result = np.empty(self.contrast_components_m.shape[1], dtype=float)
        for query_index in range(self.contrast_components_m.shape[1]):
            result[query_index] = mixture_probability_positive(
                self.contrast_components_m[:, query_index],
                self.component_conditional_variance_m2[:, query_index],
                self.pair_weights,
            )
        return readonly_array(result, dtype=float)

    @property
    def credible_interval_m(self) -> np.ndarray:
        alpha = 1.0 - self.specification.confidence_level
        result = np.empty((self.contrast_components_m.shape[1], 2), dtype=float)
        for query_index in range(self.contrast_components_m.shape[1]):
            means = self.contrast_components_m[:, query_index]
            variances = self.component_conditional_variance_m2[:, query_index]
            result[query_index, 0] = mixture_quantile(
                means,
                variances,
                self.pair_weights,
                alpha / 2.0,
            )
            result[query_index, 1] = mixture_quantile(
                means,
                variances,
                self.pair_weights,
                1.0 - alpha / 2.0,
            )
        return readonly_array(result, dtype=float)

    @property
    def effective_pair_count(self) -> float:
        return float(1.0 / np.sum(np.square(self.pair_weights)))

    @property
    def conditional_variance_included(self) -> bool:
        return self.specification.conditional_readout_correlation is not None

    def _source_descriptor(self) -> dict[str, Any]:
        return {
            "source_twin_belief_id": self.source_twin_belief_id,
            "source_factual_intervention_id": self.source_factual_intervention_id,
            "left_posterior_id": self.left_posterior_id,
            "right_posterior_id": self.right_posterior_id,
            "left_query_id": self.left_query_id,
            "right_query_id": self.right_query_id,
            "left_action_id": self.left_action_id,
            "right_action_id": self.right_action_id,
            "left_action_trajectory_sha256": self.left_action_trajectory_sha256,
            "right_action_trajectory_sha256": self.right_action_trajectory_sha256,
            "left_contact_policy": self.left_contact_policy,
            "right_contact_policy": self.right_contact_policy,
            "left_same_grasp_semantics": self.left_same_grasp_semantics,
            "right_same_grasp_semantics": self.right_same_grasp_semantics,
        }

    def _coupling_descriptor(self) -> dict[str, Any]:
        return {
            "requested_policy": self.requested_coupling_policy,
            "resolved_policy": self.resolved_coupling_policy,
            "shared_variables": list(self.shared_variables),
            "contrast_direction": "left_minus_right",
        }

    def _identity_descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
            "artifact_kind": INTERVENTIONAL_CONTRAST_ARTIFACT_KIND,
            "specification": {
                **self.specification.descriptor(),
                "specification_id": self.specification.specification_id,
            },
            "source": self._source_descriptor(),
            "coupling": self._coupling_descriptor(),
            "left_component_ids": list(self.left_component_ids),
            "right_component_ids": list(self.right_component_ids),
            "metadata": plain_json(self.metadata),
            "claim_boundary": self.claim_boundary,
            "array_bindings": {
                "query_matrix": array_sha256(self.specification.query_matrix),
                "pair_indices": array_sha256(self.pair_indices),
                "pair_weights": array_sha256(self.pair_weights),
                "left_weights": array_sha256(self.left_weights),
                "right_weights": array_sha256(self.right_weights),
                "contrast_components_m": array_sha256(
                    self.contrast_components_m
                ),
                "component_conditional_variance_m2": array_sha256(
                    self.component_conditional_variance_m2
                ),
                "expected_conditional_covariance_m2": array_sha256(
                    self.expected_conditional_covariance_m2
                ),
            },
        }

    @property
    def artifact_id(self) -> str:
        return _canonical_id(self._identity_descriptor())

    def as_dict(self) -> dict[str, Any]:
        intervals = self.credible_interval_m
        return {
            "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
            "artifact_kind": INTERVENTIONAL_CONTRAST_ARTIFACT_KIND,
            "artifact_id": self.artifact_id,
            "specification_id": self.specification.specification_id,
            "name": self.specification.name,
            "source": self._source_descriptor(),
            "coupling": self._coupling_descriptor(),
            "trajectory_source": self.specification.trajectory_source,
            "conditional_readout_correlation": (
                self.specification.conditional_readout_correlation
            ),
            "conditional_variance_included": self.conditional_variance_included,
            "query_labels": list(self.specification.query_labels),
            "query_units": list(self.specification.query_units),
            "posterior_mean_m": self.posterior_mean_m.tolist(),
            "posterior_covariance_m2": self.posterior_covariance_m2.tolist(),
            "between_component_covariance_m2": (
                self.between_component_covariance_m2.tolist()
            ),
            "expected_conditional_covariance_m2": (
                self.expected_conditional_covariance_m2.tolist()
            ),
            "probability_positive": self.probability_positive.tolist(),
            "confidence_level": self.specification.confidence_level,
            "credible_interval_m": intervals.tolist(),
            "pair_count": len(self.pair_weights),
            "effective_pair_count": self.effective_pair_count,
            "left_component_count": len(self.left_weights),
            "right_component_count": len(self.right_weights),
            "metadata": plain_json(self.metadata),
            "claim_boundary": self.claim_boundary,
        }
