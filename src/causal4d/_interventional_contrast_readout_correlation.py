"""Marginal sensitivity to cross-branch conditional readout correlation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d._interventional_contrast_common import (
    ContrastCouplingPolicy,
    _require_nonempty_string,
    _require_sha256,
    _validated_string_tuple,
)


INTERVENTIONAL_CONTRAST_READOUT_CORRELATION_SCHEMA_VERSION = 1
_READOUT_CORRELATION_ARTIFACT_KIND = (
    "Causal4DInterventionalContrastReadoutCorrelationSensitivityV1"
)
_READOUT_CORRELATION_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "source_contrast_id",
        "source_branch_a_posterior_id",
        "source_branch_b_posterior_id",
        "source_query_id",
        "branch_a_label",
        "branch_b_label",
        "coupling_policy",
        "shared_kappa_names",
        "query_name",
        "query_labels",
        "query_units",
        "metadata",
    }
)
_READOUT_CORRELATION_ARRAY_FIELDS = frozenset(
    {
        "correlation_grid",
        "mean",
        "between_component_variance",
        "conditional_variance",
        "total_variance",
        "probability_positive",
        "independent_total_variance",
        "independent_probability_positive",
    }
)
_READOUT_CORRELATION_ARRAY_DTYPES = {
    name: np.dtype(np.float64) for name in _READOUT_CORRELATION_ARRAY_FIELDS
}
_READOUT_CORRELATION_CLAIM_BOUNDARY = {
    "analysis_only": True,
    "changes_estimator": False,
    "changes_source_posterior": False,
    "changes_registered_protocol": False,
    "uses_target_truth": False,
    "cross_branch_conditional_covariance_identified": False,
    "simultaneous_cross_output_covariance_claimed": False,
    "empirical_calibration_claimed": False,
}


def _validated_vector(
    values: Any,
    *,
    name: str,
    length: int,
    nonnegative: bool = False,
    probability: bool = False,
) -> np.ndarray:
    supplied = readonly_array(values, dtype=float)
    if supplied.shape != (length,) or not np.all(np.isfinite(supplied)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    tolerance = 1e-10
    if nonnegative and np.any(supplied < -tolerance):
        raise ValueError(f"{name} must be nonnegative")
    if probability and (
        np.any(supplied < -tolerance) or np.any(supplied > 1.0 + tolerance)
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    if nonnegative:
        supplied = readonly_array(np.maximum(supplied, 0.0), dtype=float)
    if probability:
        supplied = readonly_array(np.clip(supplied, 0.0, 1.0), dtype=float)
    return supplied


def _validated_matrix(
    values: Any,
    *,
    name: str,
    shape: tuple[int, int],
    nonnegative: bool = False,
    probability: bool = False,
) -> np.ndarray:
    supplied = readonly_array(values, dtype=float)
    if supplied.shape != shape or not np.all(np.isfinite(supplied)):
        raise ValueError(f"{name} must have finite shape {shape}")
    tolerance = 1e-10
    if nonnegative and np.any(supplied < -tolerance):
        raise ValueError(f"{name} must be nonnegative")
    if probability and (
        np.any(supplied < -tolerance) or np.any(supplied > 1.0 + tolerance)
    ):
        raise ValueError(f"{name} must lie in [0, 1]")
    if nonnegative:
        supplied = readonly_array(np.maximum(supplied, 0.0), dtype=float)
    if probability:
        supplied = readonly_array(np.clip(supplied, 0.0, 1.0), dtype=float)
    return supplied


@dataclass(frozen=True)
class InterventionalContrastReadoutCorrelationSensitivityV1:
    """Marginal contrast sensitivity over a declared correlation grid.

    One scalar correlation value is applied to the conditional readout errors of
    each paired branch component. Summaries are marginal for every query output;
    the artifact does not construct or identify one simultaneous cross-output
    covariance matrix.
    """

    source_contrast_id: str
    source_branch_a_posterior_id: str
    source_branch_b_posterior_id: str
    source_query_id: str
    branch_a_label: str
    branch_b_label: str
    coupling_policy: ContrastCouplingPolicy
    shared_kappa_names: tuple[str, ...]
    query_name: str
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    correlation_grid: np.ndarray
    mean: np.ndarray
    between_component_variance: np.ndarray
    conditional_variance: np.ndarray
    total_variance: np.ndarray
    probability_positive: np.ndarray
    independent_total_variance: np.ndarray
    independent_probability_positive: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_contrast_id",
            "source_branch_a_posterior_id",
            "source_branch_b_posterior_id",
            "source_query_id",
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
        coupling = _require_nonempty_string(
            self.coupling_policy,
            name="coupling_policy",
        )
        if coupling not in {
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
        if coupling != "shared_twin_phi" and shared_kappa_names:
            raise ValueError("shared_kappa_names require shared_twin_phi coupling")
        query_name = _require_nonempty_string(self.query_name, name="query_name")
        query_labels = _validated_string_tuple(
            self.query_labels,
            name="query_labels",
            unique=True,
        )
        query_units = _validated_string_tuple(
            self.query_units,
            name="query_units",
            unique=False,
        )
        if len(query_units) != len(query_labels):
            raise ValueError("query labels and units must align")

        raw_grid = np.asarray(self.correlation_grid)
        if raw_grid.dtype.kind == "b":
            raise ValueError("correlation_grid must contain numbers, not Booleans")
        grid = readonly_array(raw_grid, dtype=float)
        if grid.ndim != 1 or len(grid) == 0 or not np.all(np.isfinite(grid)):
            raise ValueError("correlation_grid must be a nonempty finite vector")
        tolerance = 1e-10
        if np.any(grid < -1.0 - tolerance) or np.any(grid > 1.0 + tolerance):
            raise ValueError("correlation_grid must lie in [-1, 1]")
        if np.any(np.diff(grid) <= 0.0):
            raise ValueError("correlation_grid must be strictly increasing")
        zero_indices = np.flatnonzero(grid == 0.0)
        if len(zero_indices) != 1:
            raise ValueError("correlation_grid must contain zero exactly once")
        grid = readonly_array(np.clip(grid, -1.0, 1.0), dtype=float)

        output_count = len(query_labels)
        row_count = len(grid)
        mean = _validated_vector(
            self.mean,
            name="mean",
            length=output_count,
        )
        between = _validated_vector(
            self.between_component_variance,
            name="between_component_variance",
            length=output_count,
            nonnegative=True,
        )
        conditional = _validated_matrix(
            self.conditional_variance,
            name="conditional_variance",
            shape=(row_count, output_count),
            nonnegative=True,
        )
        total = _validated_matrix(
            self.total_variance,
            name="total_variance",
            shape=(row_count, output_count),
            nonnegative=True,
        )
        positive = _validated_matrix(
            self.probability_positive,
            name="probability_positive",
            shape=(row_count, output_count),
            probability=True,
        )
        independent_total = _validated_vector(
            self.independent_total_variance,
            name="independent_total_variance",
            length=output_count,
            nonnegative=True,
        )
        independent_positive = _validated_vector(
            self.independent_probability_positive,
            name="independent_probability_positive",
            length=output_count,
            probability=True,
        )

        expected_total = between[None] + conditional
        if not np.allclose(total, expected_total, atol=1e-12, rtol=1e-10):
            raise ValueError(
                "total_variance must equal between plus conditional variance"
            )
        if np.any(np.diff(conditional, axis=0) > 1e-10):
            raise ValueError(
                "conditional_variance must not increase with correlation"
            )
        zero_index = int(zero_indices[0])
        if not np.allclose(
            total[zero_index],
            independent_total,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "zero-correlation total variance must match the independent baseline"
            )
        if not np.allclose(
            positive[zero_index],
            independent_positive,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "zero-correlation probability must match the independent baseline"
            )

        object.__setattr__(self, "branch_a_label", branch_a)
        object.__setattr__(self, "branch_b_label", branch_b)
        object.__setattr__(self, "coupling_policy", coupling)
        object.__setattr__(self, "shared_kappa_names", shared_kappa_names)
        object.__setattr__(self, "query_name", query_name)
        object.__setattr__(self, "query_labels", query_labels)
        object.__setattr__(self, "query_units", query_units)
        object.__setattr__(self, "correlation_grid", grid)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "between_component_variance", between)
        object.__setattr__(self, "conditional_variance", conditional)
        object.__setattr__(self, "total_variance", total)
        object.__setattr__(self, "probability_positive", positive)
        object.__setattr__(self, "independent_total_variance", independent_total)
        object.__setattr__(
            self,
            "independent_probability_positive",
            independent_positive,
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message=(
                    "readout-correlation metadata must contain finite JSON data"
                ),
            ),
        )

    @property
    def zero_correlation_index(self) -> int:
        return int(np.flatnonzero(self.correlation_grid == 0.0)[0])

    @property
    def total_variance_envelope(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            readonly_array(np.min(self.total_variance, axis=0), dtype=float),
            readonly_array(np.max(self.total_variance, axis=0), dtype=float),
        )

    @property
    def probability_positive_envelope(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            readonly_array(np.min(self.probability_positive, axis=0), dtype=float),
            readonly_array(np.max(self.probability_positive, axis=0), dtype=float),
        )

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": (
                INTERVENTIONAL_CONTRAST_READOUT_CORRELATION_SCHEMA_VERSION
            ),
            "artifact_kind": _READOUT_CORRELATION_ARTIFACT_KIND,
            "source_contrast_id": self.source_contrast_id,
            "source_branch_a_posterior_id": self.source_branch_a_posterior_id,
            "source_branch_b_posterior_id": self.source_branch_b_posterior_id,
            "source_query_id": self.source_query_id,
            "branch_a_label": self.branch_a_label,
            "branch_b_label": self.branch_b_label,
            "coupling_policy": self.coupling_policy,
            "shared_kappa_names": list(self.shared_kappa_names),
            "query_name": self.query_name,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "correlation_grid": self.correlation_grid,
            "mean": self.mean,
            "between_component_variance": self.between_component_variance,
            "conditional_variance": self.conditional_variance,
            "total_variance": self.total_variance,
            "probability_positive": self.probability_positive,
            "independent_total_variance": self.independent_total_variance,
            "independent_probability_positive": (
                self.independent_probability_positive
            ),
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

    def as_dict(self) -> dict[str, Any]:
        variance_lower, variance_upper = self.total_variance_envelope
        positive_lower, positive_upper = self.probability_positive_envelope
        return {
            **self._scalar_payload(),
            "artifact_id": self.artifact_id,
            "correlation_grid": self.correlation_grid.tolist(),
            "mean": self.mean.tolist(),
            "between_component_variance": (
                self.between_component_variance.tolist()
            ),
            "conditional_variance": self.conditional_variance.tolist(),
            "total_variance": self.total_variance.tolist(),
            "probability_positive": self.probability_positive.tolist(),
            "independent_total_variance": (
                self.independent_total_variance.tolist()
            ),
            "independent_probability_positive": (
                self.independent_probability_positive.tolist()
            ),
            "grid_envelope": {
                "total_variance_lower": variance_lower.tolist(),
                "total_variance_upper": variance_upper.tolist(),
                "probability_positive_lower": positive_lower.tolist(),
                "probability_positive_upper": positive_upper.tolist(),
            },
        }
