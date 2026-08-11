"""Typed posterior for an explicit interventional contrast."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
from scipy.special import ndtr

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d._interventional_contrast_common import (
    INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
    ContrastConditionalVariancePolicy,
    ContrastCouplingPolicy,
    _ARTIFACT_KIND,
    _mixture_marginal_quantiles,
    _require_nonempty_string,
    _require_positive_integer,
    _require_sha256,
    _validated_probabilities,
    _validated_string_tuple,
    _validated_weights,
)
from causal4d._interventional_contrast_query import (
    InterventionalContrastQueryV1,
)


@dataclass(frozen=True)
class InterventionalContrastPosteriorV1:
    """Finite posterior of ``Q(X^a) - Q(X^b)`` under an explicit coupling."""

    source_branch_a_posterior_id: str
    source_branch_b_posterior_id: str
    source_branch_a_query_id: str
    source_branch_b_query_id: str
    branch_a_label: str
    branch_b_label: str
    trajectory_shape: tuple[int, int, int]
    branch_a_component_count: int
    branch_b_component_count: int
    coupling_policy: ContrastCouplingPolicy
    shared_kappa_names: tuple[str, ...]
    conditional_variance_policy: ContrastConditionalVariancePolicy
    query_name: str
    query_matrix: np.ndarray
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    pair_indices: np.ndarray
    weights: np.ndarray
    contrast_values: np.ndarray
    conditional_covariance: np.ndarray
    query_metadata: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_branch_a_posterior_id",
            "source_branch_b_posterior_id",
            "source_branch_a_query_id",
            "source_branch_b_query_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        branch_a = _require_nonempty_string(
            self.branch_a_label,
            name="branch_a_label",
        )
        branch_b = _require_nonempty_string(
            self.branch_b_label,
            name="branch_b_label",
        )
        if branch_a == branch_b:
            raise ValueError("branch labels must be distinct")
        if (
            not isinstance(self.trajectory_shape, Sequence)
            or isinstance(self.trajectory_shape, (str, bytes))
            or len(self.trajectory_shape) != 3
        ):
            raise ValueError("trajectory_shape must contain (frame, node, coordinate)")
        shape = tuple(
            _require_positive_integer(value, name=f"trajectory_shape[{index}]")
            for index, value in enumerate(self.trajectory_shape)
        )
        if shape[2] != 3:
            raise ValueError("trajectory_shape coordinate dimension must be three")
        branch_a_count = _require_positive_integer(
            self.branch_a_component_count,
            name="branch_a_component_count",
        )
        branch_b_count = _require_positive_integer(
            self.branch_b_component_count,
            name="branch_b_component_count",
        )
        coupling_policy = _require_nonempty_string(
            self.coupling_policy,
            name="coupling_policy",
        )
        if coupling_policy not in {
            "shared_component",
            "shared_twin_phi",
            "independent_product",
        }:
            raise ValueError("unsupported contrast coupling policy")
        shared_kappa_names = _validated_string_tuple(
            self.shared_kappa_names,
            name="shared_kappa_names",
            unique=True,
            allow_empty=True,
        )
        if coupling_policy != "shared_twin_phi" and shared_kappa_names:
            raise ValueError("shared_kappa_names require shared_twin_phi coupling")
        variance_policy = _require_nonempty_string(
            self.conditional_variance_policy,
            name="conditional_variance_policy",
        )
        if variance_policy not in {
            "component_means_only",
            "independent_readout",
        }:
            raise ValueError("unsupported conditional variance policy")

        query = InterventionalContrastQueryV1(
            name=self.query_name,
            matrix=self.query_matrix,
            labels=self.query_labels,
            units=self.query_units,
            metadata=self.query_metadata,
        )
        if query.trajectory_dimension != int(np.prod(shape)):
            raise ValueError("query matrix does not match trajectory_shape")
        pairs = readonly_integer_array(self.pair_indices, name="pair_indices")
        if pairs.ndim != 2 or pairs.shape[1:] != (2,) or len(pairs) == 0:
            raise ValueError("pair_indices must have nonempty shape (pair, 2)")
        if (
            np.any(pairs[:, 0] < 0)
            or np.any(pairs[:, 0] >= branch_a_count)
            or np.any(pairs[:, 1] < 0)
            or np.any(pairs[:, 1] >= branch_b_count)
        ):
            raise ValueError("pair_indices exceed a source posterior support")
        if len({tuple(map(int, pair)) for pair in pairs}) != len(pairs):
            raise ValueError("pair_indices must be unique")
        weights = _validated_weights(self.weights, expected_count=len(pairs))
        contrast = readonly_array(self.contrast_values, dtype=float)
        covariance = readonly_array(self.conditional_covariance, dtype=float)
        output_count = query.output_count
        if contrast.shape != (len(pairs), output_count):
            raise ValueError("contrast_values must have shape (pair, query)")
        if covariance.shape != (len(pairs), output_count, output_count):
            raise ValueError(
                "conditional_covariance must have shape (pair, query, query)"
            )
        if not np.all(np.isfinite(contrast)) or not np.all(np.isfinite(covariance)):
            raise ValueError("contrast values and covariance must be finite")
        if not np.allclose(
            covariance,
            covariance.swapaxes(-1, -2),
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError("conditional covariance must be symmetric")
        minimum_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance), initial=0.0))
        if minimum_eigenvalue < -1e-10:
            raise ValueError("conditional covariance must be positive semidefinite")
        if variance_policy == "component_means_only" and np.any(covariance != 0.0):
            raise ValueError(
                "component_means_only requires exactly zero conditional covariance"
            )

        object.__setattr__(self, "branch_a_label", branch_a)
        object.__setattr__(self, "branch_b_label", branch_b)
        object.__setattr__(self, "trajectory_shape", shape)
        object.__setattr__(self, "branch_a_component_count", branch_a_count)
        object.__setattr__(self, "branch_b_component_count", branch_b_count)
        object.__setattr__(self, "coupling_policy", coupling_policy)
        object.__setattr__(self, "shared_kappa_names", shared_kappa_names)
        object.__setattr__(self, "conditional_variance_policy", variance_policy)
        object.__setattr__(self, "query_name", query.name)
        object.__setattr__(self, "query_matrix", query.matrix)
        object.__setattr__(self, "query_labels", query.labels)
        object.__setattr__(self, "query_units", query.units)
        object.__setattr__(self, "query_metadata", query.metadata)
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "contrast_values", contrast)
        object.__setattr__(self, "conditional_covariance", covariance)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="contrast metadata must contain finite JSON data",
            ),
        )

    @property
    def query_id(self) -> str:
        return InterventionalContrastQueryV1(
            name=self.query_name,
            matrix=self.query_matrix,
            labels=self.query_labels,
            units=self.query_units,
            metadata=self.query_metadata,
        ).query_id

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "source_branch_a_posterior_id": self.source_branch_a_posterior_id,
            "source_branch_b_posterior_id": self.source_branch_b_posterior_id,
            "source_branch_a_query_id": self.source_branch_a_query_id,
            "source_branch_b_query_id": self.source_branch_b_query_id,
            "branch_a_label": self.branch_a_label,
            "branch_b_label": self.branch_b_label,
            "trajectory_shape": list(self.trajectory_shape),
            "branch_a_component_count": self.branch_a_component_count,
            "branch_b_component_count": self.branch_b_component_count,
            "coupling_policy": self.coupling_policy,
            "shared_kappa_names": list(self.shared_kappa_names),
            "conditional_variance_policy": self.conditional_variance_policy,
            "query_id": self.query_id,
            "query_name": self.query_name,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "query_metadata": plain_json(self.query_metadata),
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "pair_indices": self.pair_indices,
            "weights": self.weights,
            "query_matrix": self.query_matrix,
            "contrast_values": self.contrast_values,
            "conditional_covariance": self.conditional_covariance,
        }

    @property
    def artifact_id(self) -> str:
        digest = hashlib.sha256()
        digest.update(
            json.dumps(
                self._scalar_payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        )
        for name, values in sorted(self._array_payload().items()):
            digest.update(name.encode("utf-8"))
            digest.update(array_sha256(values).encode("ascii"))
        return digest.hexdigest()

    @property
    def mean(self) -> np.ndarray:
        return readonly_array(
            np.einsum("k,kq->q", self.weights, self.contrast_values),
            dtype=float,
        )

    @property
    def covariance(self) -> np.ndarray:
        mean = np.asarray(self.mean)
        centered = self.contrast_values - mean[None]
        between = np.einsum("k,ki,kj->ij", self.weights, centered, centered)
        within = np.einsum("k,kij->ij", self.weights, self.conditional_covariance)
        covariance = 0.5 * (between + within + (between + within).T)
        return readonly_array(covariance, dtype=float)

    @property
    def standard_deviation(self) -> np.ndarray:
        return readonly_array(
            np.sqrt(np.maximum(np.diag(self.covariance), 0.0)),
            dtype=float,
        )

    @property
    def probability_positive(self) -> np.ndarray:
        variances = np.diagonal(
            self.conditional_covariance,
            axis1=-2,
            axis2=-1,
        )
        component_probability = np.zeros_like(self.contrast_values)
        positive_variance = variances > 0.0
        component_probability[positive_variance] = ndtr(
            self.contrast_values[positive_variance]
            / np.sqrt(variances[positive_variance])
        )
        component_probability[~positive_variance] = (
            self.contrast_values[~positive_variance] > 0.0
        )
        return readonly_array(
            np.einsum("k,kq->q", self.weights, component_probability),
            dtype=float,
        )

    def marginal_quantiles(
        self,
        probabilities: Sequence[float] = (0.05, 0.5, 0.95),
    ) -> np.ndarray:
        """Return generalized-inverse marginal mixture quantiles.

        The returned shape is ``(probability, query_output)``. These are posterior
        credible summaries under the declared coupling and conditional-variance
        policy; they are not an empirical calibration guarantee.
        """

        levels = _validated_probabilities(probabilities)
        variances = np.diagonal(
            self.conditional_covariance,
            axis1=-2,
            axis2=-1,
        )
        return readonly_array(
            _mixture_marginal_quantiles(
                self.contrast_values,
                variances,
                self.weights,
                levels,
            ),
            dtype=float,
        )

    def central_interval(
        self,
        confidence_level: float = 0.90,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return an equal-tail marginal posterior interval."""

        if (
            isinstance(confidence_level, (bool, np.bool_))
            or not isinstance(
                confidence_level,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(confidence_level)
            or not 0.0 < float(confidence_level) < 1.0
        ):
            raise ValueError("confidence_level must lie strictly between zero and one")
        tail = 0.5 * (1.0 - float(confidence_level))
        quantiles = self.marginal_quantiles((tail, 1.0 - tail))
        return quantiles[0], quantiles[1]

    @property
    def effective_component_count(self) -> float:
        return float(1.0 / np.sum(np.square(self.weights)))

    def as_dict(self) -> dict[str, Any]:
        probabilities = (0.05, 0.5, 0.95)
        quantiles = self.marginal_quantiles(probabilities)
        return {
            "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
            "artifact_kind": _ARTIFACT_KIND,
            "artifact_id": self.artifact_id,
            "source_branch_a_posterior_id": self.source_branch_a_posterior_id,
            "source_branch_b_posterior_id": self.source_branch_b_posterior_id,
            "source_branch_a_query_id": self.source_branch_a_query_id,
            "source_branch_b_query_id": self.source_branch_b_query_id,
            "query_id": self.query_id,
            "contrast": f"{self.branch_a_label} minus {self.branch_b_label}",
            "coupling_policy": self.coupling_policy,
            "shared_kappa_names": list(self.shared_kappa_names),
            "conditional_variance_policy": self.conditional_variance_policy,
            "pair_count": len(self.pair_indices),
            "effective_component_count": self.effective_component_count,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "mean": self.mean.tolist(),
            "standard_deviation": self.standard_deviation.tolist(),
            "probability_positive": self.probability_positive.tolist(),
            "marginal_quantiles": {
                "probabilities": list(probabilities),
                "values": quantiles.tolist(),
            },
            "metadata": plain_json(self.metadata),
        }
