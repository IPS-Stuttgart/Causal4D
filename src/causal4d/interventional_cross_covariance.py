"""Explicit query-space cross-branch covariance for contrast posteriors.

This module is additive.  It never changes either source posterior or the
historical interventional-contrast estimator.  Omitting the cross covariance
returns the exact source contrast object.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

import numpy as np
from scipy.special import ndtr

from causal4d.contracts import PhysicalPosterior, array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d._interventional_contrast_common import (
    _mixture_marginal_quantiles,
    _validated_probabilities,
    _validated_weights,
)
from causal4d._interventional_contrast_posterior import (
    InterventionalContrastPosteriorV1,
)


INTERVENTIONAL_CROSS_COVARIANCE_SCHEMA_VERSION = 1
INTERVENTIONAL_CROSS_COVARIANCE_CLAIM_BOUNDARY = (
    "Analysis-only query-space covariance adjustment. It does not change source "
    "posteriors, estimate covariance from target outcomes, establish calibration, "
    "or identify an individual-level real counterfactual effect."
)
_KIND = "Causal4DInterventionalCrossCovarianceV1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _query_component_covariance(
    posterior: PhysicalPosterior,
    source: InterventionalContrastPosteriorV1,
) -> np.ndarray:
    trajectories = np.asarray(posterior.readout_trajectories_m, dtype=float)
    variance = np.asarray(posterior.readout_variance_m2, dtype=float)
    if trajectories.ndim != 4 or trajectories.shape[-1] != 3:
        raise ValueError("source readout trajectories have the wrong shape")
    expected = (len(trajectories), trajectories.shape[2], trajectories.shape[3])
    if variance.shape != expected:
        raise ValueError("source readout variance has the wrong shape")
    if not np.all(np.isfinite(variance)) or np.any(variance < 0.0):
        raise ValueError("source readout variance must be finite and nonnegative")
    diagonal = np.broadcast_to(variance[:, None], trajectories.shape).reshape(
        len(trajectories), -1
    )
    matrix = np.asarray(source.query_matrix, dtype=float)
    return np.einsum("qi,ki,ri->kqr", matrix, diagonal, matrix)


def _validate_joint_blocks(
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    cross: np.ndarray,
) -> None:
    output_count = branch_a.shape[-1]
    for index in range(len(cross)):
        joint = np.block(
            [
                [branch_a[index], cross[index]],
                [cross[index].T, branch_b[index]],
            ]
        )
        scale = max(1.0, float(np.max(np.abs(joint), initial=0.0)))
        if float(np.min(np.linalg.eigvalsh(joint), initial=0.0)) < -1.0e-10 * scale:
            raise ValueError(
                f"joint branch covariance for pair {index} is not positive semidefinite"
            )
        if joint.shape != (2 * output_count, 2 * output_count):
            raise RuntimeError("internal joint covariance shape changed")


def _contrast_covariance(
    branch_a: np.ndarray,
    branch_b: np.ndarray,
    cross: np.ndarray,
) -> np.ndarray:
    result = branch_a + branch_b - cross - cross.swapaxes(-1, -2)
    result = 0.5 * (result + result.swapaxes(-1, -2))
    scale = np.maximum(1.0, np.max(np.abs(result), axis=(-2, -1)))
    minimum = np.min(np.linalg.eigvalsh(result), axis=-1)
    if np.any(minimum < -1.0e-10 * scale):
        raise ValueError("derived contrast covariance is not positive semidefinite")
    return np.asarray(result, dtype=float)


@dataclass(frozen=True)
class InterventionalCrossCovarianceV1:
    """Finite contrast posterior with an explicit conditional branch coupling."""

    source_contrast_id: str
    cross_covariance_model_id: str
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    pair_indices: np.ndarray
    weights: np.ndarray
    contrast_values: np.ndarray
    branch_a_conditional_covariance: np.ndarray
    branch_b_conditional_covariance: np.ndarray
    cross_branch_conditional_covariance: np.ndarray
    conditional_covariance: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_id = _require_sha256(self.source_contrast_id, name="source_contrast_id")
        model_id = _require_sha256(
            self.cross_covariance_model_id,
            name="cross_covariance_model_id",
        )
        labels = tuple(self.query_labels)
        units = tuple(self.query_units)
        if not labels or len(labels) != len(units):
            raise ValueError("query labels and units must be nonempty and aligned")
        if len(set(labels)) != len(labels) or any(
            not value for value in labels + units
        ):
            raise ValueError("query labels must be unique and labels/units nonempty")
        pairs = readonly_integer_array(self.pair_indices, name="pair_indices")
        if pairs.ndim != 2 or pairs.shape[1:] != (2,) or len(pairs) == 0:
            raise ValueError("pair_indices must have nonempty shape (pair, 2)")
        weights = _validated_weights(self.weights, expected_count=len(pairs))
        values = readonly_array(self.contrast_values, dtype=float)
        dimension = len(labels)
        if values.shape != (len(pairs), dimension) or not np.all(np.isfinite(values)):
            raise ValueError("contrast_values must have finite shape (pair, query)")
        expected = (len(pairs), dimension, dimension)
        covariances = []
        for raw, name in (
            (self.branch_a_conditional_covariance, "branch_a_conditional_covariance"),
            (self.branch_b_conditional_covariance, "branch_b_conditional_covariance"),
            (
                self.cross_branch_conditional_covariance,
                "cross_branch_conditional_covariance",
            ),
            (self.conditional_covariance, "conditional_covariance"),
        ):
            covariance = readonly_array(raw, dtype=float)
            if covariance.shape != expected or not np.all(np.isfinite(covariance)):
                raise ValueError(f"{name} must have finite shape {expected}")
            covariances.append(covariance)
        branch_a, branch_b, cross, conditional = covariances
        _validate_joint_blocks(branch_a, branch_b, cross)
        wanted = _contrast_covariance(branch_a, branch_b, cross)
        if not np.allclose(conditional, wanted, atol=1.0e-12, rtol=1.0e-10):
            raise ValueError("conditional_covariance does not match branch coupling")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="cross-covariance metadata must contain finite JSON data",
        )
        object.__setattr__(self, "source_contrast_id", source_id)
        object.__setattr__(self, "cross_covariance_model_id", model_id)
        object.__setattr__(self, "query_labels", labels)
        object.__setattr__(self, "query_units", units)
        object.__setattr__(self, "pair_indices", pairs)
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "contrast_values", values)
        object.__setattr__(self, "branch_a_conditional_covariance", branch_a)
        object.__setattr__(self, "branch_b_conditional_covariance", branch_b)
        object.__setattr__(self, "cross_branch_conditional_covariance", cross)
        object.__setattr__(self, "conditional_covariance", conditional)
        object.__setattr__(self, "metadata", metadata)

    @property
    def mean(self) -> np.ndarray:
        return readonly_array(np.einsum("k,kq->q", self.weights, self.contrast_values))

    @property
    def covariance(self) -> np.ndarray:
        centered = self.contrast_values - np.asarray(self.mean)[None]
        between = np.einsum("k,ki,kj->ij", self.weights, centered, centered)
        within = np.einsum("k,kij->ij", self.weights, self.conditional_covariance)
        return readonly_array(0.5 * (between + within + (between + within).T))

    @property
    def probability_positive(self) -> np.ndarray:
        variance = np.diagonal(self.conditional_covariance, axis1=-2, axis2=-1)
        probabilities = np.zeros_like(self.contrast_values)
        positive = variance > 0.0
        probabilities[positive] = ndtr(
            self.contrast_values[positive] / np.sqrt(variance[positive])
        )
        probabilities[~positive] = self.contrast_values[~positive] > 0.0
        return readonly_array(np.einsum("k,kq->q", self.weights, probabilities))

    def marginal_quantiles(
        self,
        probabilities: Sequence[float] = (0.05, 0.5, 0.95),
    ) -> np.ndarray:
        levels = _validated_probabilities(probabilities)
        variance = np.diagonal(self.conditional_covariance, axis1=-2, axis2=-1)
        return readonly_array(
            _mixture_marginal_quantiles(
                self.contrast_values,
                variance,
                self.weights,
                levels,
            )
        )

    @property
    def artifact_id(self) -> str:
        descriptor = {
            "schema_version": INTERVENTIONAL_CROSS_COVARIANCE_SCHEMA_VERSION,
            "artifact_kind": _KIND,
            "source_contrast_id": self.source_contrast_id,
            "cross_covariance_model_id": self.cross_covariance_model_id,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "metadata": plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CROSS_COVARIANCE_CLAIM_BOUNDARY,
            "arrays": {
                name: array_sha256(getattr(self, name))
                for name in (
                    "pair_indices",
                    "weights",
                    "contrast_values",
                    "branch_a_conditional_covariance",
                    "branch_b_conditional_covariance",
                    "cross_branch_conditional_covariance",
                    "conditional_covariance",
                )
            },
        }
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_CROSS_COVARIANCE_SCHEMA_VERSION,
            "artifact_kind": _KIND,
            "artifact_id": self.artifact_id,
            "source_contrast_id": self.source_contrast_id,
            "cross_covariance_model_id": self.cross_covariance_model_id,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "mean": self.mean.tolist(),
            "covariance": self.covariance.tolist(),
            "probability_positive": self.probability_positive.tolist(),
            "effective_component_count": float(1.0 / np.sum(np.square(self.weights))),
            "metadata": plain_json(self.metadata),
            "claim_boundary": INTERVENTIONAL_CROSS_COVARIANCE_CLAIM_BOUNDARY,
        }


def build_interventional_cross_covariance(
    source: InterventionalContrastPosteriorV1,
    branch_a: PhysicalPosterior,
    branch_b: PhysicalPosterior,
    *,
    cross_branch_conditional_covariance: np.ndarray | None = None,
    cross_covariance_model_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalContrastPosteriorV1 | InterventionalCrossCovarianceV1:
    """Apply a source-frozen cross covariance, or return ``source`` exactly."""

    if cross_branch_conditional_covariance is None:
        if cross_covariance_model_id is not None:
            raise ValueError("cross_covariance_model_id requires a covariance array")
        return source
    if source.conditional_variance_policy != "independent_readout":
        raise ValueError("cross covariance requires an independent_readout source")
    if source.source_branch_a_posterior_id != branch_a.artifact_id:
        raise ValueError("branch_a does not match the source contrast")
    if source.source_branch_b_posterior_id != branch_b.artifact_id:
        raise ValueError("branch_b does not match the source contrast")
    model_id = _require_sha256(
        cross_covariance_model_id,
        name="cross_covariance_model_id",
    )
    branch_a_all = _query_component_covariance(branch_a, source)
    branch_b_all = _query_component_covariance(branch_b, source)
    pairs = np.asarray(source.pair_indices, dtype=np.int64)
    branch_a_covariance = branch_a_all[pairs[:, 0]]
    branch_b_covariance = branch_b_all[pairs[:, 1]]
    cross = np.asarray(cross_branch_conditional_covariance, dtype=float)
    expected = branch_a_covariance.shape
    if cross.shape != expected or not np.all(np.isfinite(cross)):
        raise ValueError(f"cross covariance must have finite shape {expected}")
    _validate_joint_blocks(branch_a_covariance, branch_b_covariance, cross)
    conditional = _contrast_covariance(
        branch_a_covariance,
        branch_b_covariance,
        cross,
    )
    user_metadata = dict(metadata or {})
    user_metadata.update(
        {
            "future_observations_read": 0,
            "source_conditional_variance_policy": "independent_readout",
            "target_outcomes_used_for_covariance": False,
        }
    )
    return InterventionalCrossCovarianceV1(
        source_contrast_id=source.artifact_id,
        cross_covariance_model_id=model_id,
        query_labels=source.query_labels,
        query_units=source.query_units,
        pair_indices=source.pair_indices,
        weights=source.weights,
        contrast_values=source.contrast_values,
        branch_a_conditional_covariance=branch_a_covariance,
        branch_b_conditional_covariance=branch_b_covariance,
        cross_branch_conditional_covariance=cross,
        conditional_covariance=conditional,
        metadata=user_metadata,
    )


__all__ = [
    "INTERVENTIONAL_CROSS_COVARIANCE_CLAIM_BOUNDARY",
    "INTERVENTIONAL_CROSS_COVARIANCE_SCHEMA_VERSION",
    "InterventionalCrossCovarianceV1",
    "build_interventional_cross_covariance",
]
