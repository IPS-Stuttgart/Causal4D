"""Specification and validation primitives for interventional contrasts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping

if TYPE_CHECKING:
    from causal4d.contracts import PhysicalPosterior


INTERVENTIONAL_CONTRAST_SCHEMA_VERSION = 1
INTERVENTIONAL_CONTRAST_ARTIFACT_KIND = "InterventionalContrastPosteriorV1"
INTERVENTIONAL_CONTRAST_CLAIM_BOUNDARY = (
    "This artifact compares two already-produced Causal4D physical posteriors. "
    "It does not alter the frozen estimator, admit evidence, establish an "
    "individual-level real counterfactual, or authorize model selection, "
    "calibration, deployment, or a physical causal claim by itself."
)

TrajectorySource = Literal["state", "readout"]
CouplingPolicy = Literal[
    "auto",
    "shared_theta_phi_kappa",
    "shared_theta_phi_patch",
    "shared_theta_phi",
    "component_id",
    "independent",
]
ResolvedCouplingPolicy = Literal[
    "shared_theta_phi_kappa",
    "shared_theta_phi_patch",
    "shared_theta_phi",
    "component_id",
    "independent",
]

_ALLOWED_COUPLING_POLICIES = frozenset(
    {
        "auto",
        "shared_theta_phi_kappa",
        "shared_theta_phi_patch",
        "shared_theta_phi",
        "component_id",
        "independent",
    }
)
_EXPECTED_ARRAYS = frozenset(
    {
        "descriptor_json",
        "query_matrix",
        "pair_indices",
        "pair_weights",
        "left_weights",
        "right_weights",
        "contrast_components_m",
        "component_conditional_variance_m2",
        "expected_conditional_covariance_m2",
    }
)
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "specification",
        "source",
        "coupling",
        "left_component_ids",
        "right_component_ids",
        "metadata",
        "claim_boundary",
    }
)
_SPECIFICATION_FIELDS = frozenset(
    {
        "name",
        "trajectory_source",
        "coupling_policy",
        "conditional_readout_correlation",
        "confidence_level",
        "query_labels",
        "query_units",
        "query_matrix_sha256",
        "metadata",
        "specification_id",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "source_twin_belief_id",
        "source_factual_intervention_id",
        "left_posterior_id",
        "right_posterior_id",
        "left_query_id",
        "right_query_id",
        "left_action_id",
        "right_action_id",
        "left_action_trajectory_sha256",
        "right_action_trajectory_sha256",
        "left_contact_policy",
        "right_contact_policy",
        "left_same_grasp_semantics",
        "right_same_grasp_semantics",
    }
)
_COUPLING_FIELDS = frozenset(
    {
        "requested_policy",
        "resolved_policy",
        "shared_variables",
        "contrast_direction",
    }
)
_BOUNDARY_METADATA = {
    "analysis_only": True,
    "source_posteriors_unchanged": True,
    "target_outcomes_used": False,
    "cross_world_coupling_is_an_explicit_assumption": True,
}


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_id(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(fields - actual)
    unexpected = sorted(actual - fields)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields changed; missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    result = _require_nonempty_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _require_probability_open(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must be a finite number in (0, 1)")
    return result


def _require_correlation(value: Any, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite number in [-1, 1]")
    result = float(value)
    if not np.isfinite(result) or not -1.0 <= result <= 1.0:
        raise ValueError(f"{name} must be a finite number in [-1, 1]")
    return result


def _validated_string_tuple(
    values: Sequence[str],
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    return result


def _validated_weights(values: Any, *, name: str) -> np.ndarray:
    weights = readonly_array(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError(f"{name} must be a nonempty vector")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return weights


def _metadata_string(
    posterior: PhysicalPosterior,
    name: str,
    *,
    default: str | None = None,
) -> str:
    value = posterior.metadata.get(name, default)
    if type(value) is not str or not value:
        raise ValueError(f"physical posterior metadata {name!r} is required")
    return value


@dataclass(frozen=True, slots=True)
class InterventionalContrastSpecificationV1:
    """Content-addressed query and cross-world coupling specification."""

    name: str
    query_matrix: np.ndarray
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...] = ()
    trajectory_source: TrajectorySource = "readout"
    coupling_policy: CouplingPolicy = "auto"
    conditional_readout_correlation: float | None = None
    confidence_level: float = 0.90
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        name = _require_nonempty_string(self.name, name="name")
        query = readonly_array(self.query_matrix, dtype=float)
        if query.ndim != 2 or query.shape[0] == 0 or query.shape[1] == 0:
            raise ValueError("query_matrix must have nonempty shape (Q, D)")
        if not np.all(np.isfinite(query)):
            raise ValueError("query_matrix must be finite")
        if np.any(np.linalg.norm(query, axis=1) == 0.0):
            raise ValueError("query_matrix rows must be nonzero")
        labels = _validated_string_tuple(self.query_labels, name="query_labels")
        if len(labels) != query.shape[0] or len(set(labels)) != len(labels):
            raise ValueError("query_labels must uniquely identify every query row")
        units = tuple(self.query_units)
        if not units:
            units = ("m",) * query.shape[0]
        units = _validated_string_tuple(units, name="query_units")
        if len(units) != query.shape[0]:
            raise ValueError("query_units must identify every query row")
        if any(unit != "m" for unit in units):
            raise ValueError("interventional contrast query units must be metres")
        trajectory_source = _require_nonempty_string(
            self.trajectory_source,
            name="trajectory_source",
        )
        if trajectory_source not in {"state", "readout"}:
            raise ValueError("trajectory_source must be 'state' or 'readout'")
        coupling_policy = _require_nonempty_string(
            self.coupling_policy,
            name="coupling_policy",
        )
        if coupling_policy not in _ALLOWED_COUPLING_POLICIES:
            raise ValueError("unsupported interventional coupling policy")
        correlation = self.conditional_readout_correlation
        if correlation is not None:
            correlation = _require_correlation(
                correlation,
                name="conditional_readout_correlation",
            )
        if trajectory_source == "state" and correlation is not None:
            raise ValueError(
                "conditional_readout_correlation is only defined for readout queries"
            )
        confidence = _require_probability_open(
            self.confidence_level,
            name="confidence_level",
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="contrast specification metadata must contain finite JSON",
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "query_matrix", query)
        object.__setattr__(self, "query_labels", labels)
        object.__setattr__(self, "query_units", units)
        object.__setattr__(self, "trajectory_source", trajectory_source)
        object.__setattr__(self, "coupling_policy", coupling_policy)
        object.__setattr__(
            self,
            "conditional_readout_correlation",
            correlation,
        )
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "metadata", metadata)

    def descriptor(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
            "artifact_kind": "InterventionalContrastSpecificationV1",
            "name": self.name,
            "trajectory_source": self.trajectory_source,
            "coupling_policy": self.coupling_policy,
            "conditional_readout_correlation": (
                self.conditional_readout_correlation
            ),
            "confidence_level": self.confidence_level,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "query_matrix_shape": list(self.query_matrix.shape),
            "query_matrix_sha256": array_sha256(self.query_matrix),
            "metadata": plain_json(self.metadata),
        }

    @property
    def specification_id(self) -> str:
        return _canonical_id(self.descriptor())
