"""Prefix-only localization of view, frame, and object-coherent residuals.

The decomposition is diagnostic and non-claim-bearing. It never changes the
registered estimator, causal boundary, or physical evidence count.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Literal, cast

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array


PER_VIEW_RESIDUAL_LOCALIZATION_SCHEMA_VERSION = 1
DominantResidualSource = Literal[
    "view_specific",
    "shared_frame",
    "object_coherent",
    "mixed",
    "unresolved",
]


def _finite_nonnegative(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _fraction(value: float, *, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _sha256(value: str, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _graph_basis(values: np.ndarray | None, *, node_count: int) -> np.ndarray:
    if values is None:
        return np.empty((node_count, 0), dtype=float)
    basis = np.asarray(values, dtype=float)
    if basis.ndim != 2 or basis.shape[0] != node_count:
        raise ValueError("graph_basis must have shape (node, mode)")
    if not np.all(np.isfinite(basis)):
        raise ValueError("graph_basis must be finite")
    if basis.shape[1] == 0:
        return np.empty((node_count, 0), dtype=float)
    basis = basis - np.mean(basis, axis=0, keepdims=True)
    scale = np.sqrt(np.mean(np.square(basis), axis=0))
    if np.any(scale <= 1e-12):
        raise ValueError(
            "every graph_basis mode must retain nonzero centered node variation"
        )
    return basis / scale


def _fit(
    design: np.ndarray,
    observations: np.ndarray,
    weights: np.ndarray,
    ridge: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    if not len(columns):
        prediction = np.zeros_like(observations)
        sse = float(np.sum(weights[:, None] * np.square(observations)))
        return np.empty((0, 3), dtype=float), prediction, sse

    selected = design[:, columns]
    root_weight = np.sqrt(weights)
    weighted_design = selected * root_weight[:, None]
    weighted_observations = observations * root_weight[:, None]
    normal = weighted_design.T @ weighted_design
    normal += np.diag(ridge[columns])
    rhs = weighted_design.T @ weighted_observations
    try:
        coefficients = np.linalg.solve(normal, rhs)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(normal, rhs, rcond=None)[0]
    prediction = selected @ coefficients
    residual = observations - prediction
    sse = float(np.sum(weights[:, None] * np.square(residual)))
    if not np.all(np.isfinite(coefficients)) or not np.isfinite(sse):
        raise ValueError("per-view residual fit produced nonfinite values")
    return coefficients, prediction, sse


def _explained(improvement: float, total: float) -> float:
    if total <= np.finfo(float).tiny:
        return 0.0
    return float(np.clip(improvement / total, 0.0, 1.0))


@dataclass(frozen=True, slots=True)
class PerViewResidualLocalizationResult:
    """Content-addressed prefix-only residual decomposition."""

    evidence_artifact_id: str
    causal_prefix_frame_stop: int
    reference_view_index: int
    view_count: int
    frame_count: int
    node_count: int
    graph_rank: int
    observed_points_sha256: str
    predicted_points_sha256: str
    validity_mask_sha256: str
    confidence_sha256: str
    graph_basis_sha256: str | None
    view_ridge: float
    frame_ridge: float
    graph_ridge: float
    maximum_design_bytes: int
    view_bias_m: np.ndarray
    shared_frame_offset_m: np.ndarray
    graph_coefficients_m: np.ndarray
    graph_field_m: np.ndarray
    per_view_rms_before_m: np.ndarray
    per_view_rms_after_m: np.ndarray
    valid_observation_counts: np.ndarray
    total_weighted_sse_m2: float
    full_weighted_sse_m2: float
    frame_only_weighted_sse_m2: float
    no_view_weighted_sse_m2: float
    no_frame_weighted_sse_m2: float
    no_graph_weighted_sse_m2: float
    full_explained_fraction: float
    view_unique_fraction: float
    frame_unique_fraction: float
    graph_unique_fraction: float
    dominant_source: DominantResidualSource
    artifact_id: str = field(init=False)

    def __post_init__(self) -> None:
        _sha256(self.evidence_artifact_id, name="evidence_artifact_id")
        dimensions = (
            self.causal_prefix_frame_stop,
            self.view_count,
            self.frame_count,
            self.node_count,
        )
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("localization dimensions must be positive integers")
        if self.frame_count != self.causal_prefix_frame_stop:
            raise ValueError("frame_count must equal the causal prefix length")
        if not 0 <= self.reference_view_index < self.view_count:
            raise ValueError("reference_view_index lies outside the view inventory")
        if type(self.graph_rank) is not int or self.graph_rank < 0:
            raise ValueError("graph_rank must be a nonnegative integer")
        if type(self.maximum_design_bytes) is not int or self.maximum_design_bytes < 1:
            raise ValueError("maximum_design_bytes must be a positive integer")

        for name in (
            "observed_points_sha256",
            "predicted_points_sha256",
            "validity_mask_sha256",
            "confidence_sha256",
        ):
            _sha256(getattr(self, name), name=name)
        if self.graph_basis_sha256 is not None:
            _sha256(self.graph_basis_sha256, name="graph_basis_sha256")
        for name in ("view_ridge", "frame_ridge", "graph_ridge"):
            _finite_nonnegative(getattr(self, name), name=name)
        for name in (
            "total_weighted_sse_m2",
            "full_weighted_sse_m2",
            "frame_only_weighted_sse_m2",
            "no_view_weighted_sse_m2",
            "no_frame_weighted_sse_m2",
            "no_graph_weighted_sse_m2",
        ):
            _finite_nonnegative(getattr(self, name), name=name)
        for name in (
            "full_explained_fraction",
            "view_unique_fraction",
            "frame_unique_fraction",
            "graph_unique_fraction",
        ):
            _fraction(getattr(self, name), name=name)
        if self.dominant_source not in {
            "view_specific",
            "shared_frame",
            "object_coherent",
            "mixed",
            "unresolved",
        }:
            raise ValueError("unsupported dominant_source")

        array_shapes = {
            "view_bias_m": (self.view_count, 3),
            "shared_frame_offset_m": (self.frame_count, 3),
            "graph_coefficients_m": (self.frame_count, self.graph_rank, 3),
            "graph_field_m": (self.frame_count, self.node_count, 3),
            "per_view_rms_before_m": (self.view_count,),
            "per_view_rms_after_m": (self.view_count,),
        }
        for name, shape in array_shapes.items():
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must be finite with shape {shape}")
            object.__setattr__(self, name, readonly_array(values, dtype=float))

        counts = np.asarray(self.valid_observation_counts)
        if counts.shape != (self.view_count,) or counts.dtype.kind not in "iu":
            raise ValueError("valid_observation_counts must be an integer per view")
        if np.any(counts < 1):
            raise ValueError("every view must contribute at least one observation")
        object.__setattr__(
            self,
            "valid_observation_counts",
            readonly_integer_array(counts, name="valid_observation_counts"),
        )

        payload = self.as_dict(include_artifact_id=False)
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        object.__setattr__(self, "artifact_id", hashlib.sha256(encoded).hexdigest())

    def as_dict(self, *, include_artifact_id: bool = True) -> dict[str, object]:
        """Return a finite JSON summary without residual-array materialization."""

        result: dict[str, object] = {
            "schema": "causal4d.per_view_residual_localization",
            "schema_version": PER_VIEW_RESIDUAL_LOCALIZATION_SCHEMA_VERSION,
            "evidence_artifact_id": self.evidence_artifact_id,
            "causal_prefix_frame_stop": self.causal_prefix_frame_stop,
            "reference_view_index": self.reference_view_index,
            "dimensions": {
                "view_count": self.view_count,
                "frame_count": self.frame_count,
                "node_count": self.node_count,
                "graph_rank": self.graph_rank,
            },
            "input_sha256": {
                "observed_points": self.observed_points_sha256,
                "predicted_points": self.predicted_points_sha256,
                "validity_mask": self.validity_mask_sha256,
                "confidence": self.confidence_sha256,
                "graph_basis": self.graph_basis_sha256,
            },
            "output_sha256": {
                "view_bias": array_sha256(self.view_bias_m),
                "shared_frame_offset": array_sha256(self.shared_frame_offset_m),
                "graph_coefficients": array_sha256(self.graph_coefficients_m),
                "graph_field": array_sha256(self.graph_field_m),
            },
            "fit": {
                "total_weighted_sse_m2": self.total_weighted_sse_m2,
                "full_weighted_sse_m2": self.full_weighted_sse_m2,
                "frame_only_weighted_sse_m2": self.frame_only_weighted_sse_m2,
                "no_view_weighted_sse_m2": self.no_view_weighted_sse_m2,
                "no_frame_weighted_sse_m2": self.no_frame_weighted_sse_m2,
                "no_graph_weighted_sse_m2": self.no_graph_weighted_sse_m2,
                "full_explained_fraction": self.full_explained_fraction,
                "view_unique_fraction": self.view_unique_fraction,
                "frame_unique_fraction": self.frame_unique_fraction,
                "graph_unique_fraction": self.graph_unique_fraction,
                "dominant_source": self.dominant_source,
            },
            "per_view": {
                "valid_observation_counts": self.valid_observation_counts.tolist(),
                "rms_before_m": self.per_view_rms_before_m.tolist(),
                "rms_after_m": self.per_view_rms_after_m.tolist(),
                "bias_norm_m": np.linalg.norm(self.view_bias_m, axis=1).tolist(),
            },
            "regularization": {
                "view_ridge": self.view_ridge,
                "frame_ridge": self.frame_ridge,
                "graph_ridge": self.graph_ridge,
                "maximum_design_bytes": self.maximum_design_bytes,
            },
            "target_outcomes_used": False,
            "physical_evidence_increment": 0,
            "claim_boundary": (
                "Prefix-only diagnostic decomposition; not a physical discrepancy "
                "claim, estimator update, calibration result, or acquisition count."
            ),
        }
        if include_artifact_id:
            result["artifact_id"] = self.artifact_id
        return result


def localize_per_view_residuals(
    observed_points_m: np.ndarray,
    predicted_points_m: np.ndarray,
    validity_mask: np.ndarray,
    *,
    evidence_artifact_id: str,
    confidence: np.ndarray | None = None,
    graph_basis: np.ndarray | None = None,
    causal_prefix_frame_stop: int | None = None,
    view_ridge: float = 1e-8,
    frame_ridge: float = 1e-8,
    graph_ridge: float = 1e-6,
    minimum_explained_fraction: float = 0.05,
    dominance_margin: float = 0.02,
    maximum_design_bytes: int = 512 * 1024**2,
) -> PerViewResidualLocalizationResult:
    """Separate relative view, shared-frame, and graph-coherent prefix residuals."""

    observed = np.asarray(observed_points_m, dtype=float)
    predicted = np.asarray(predicted_points_m, dtype=float)
    validity = np.asarray(validity_mask)
    if observed.ndim != 4 or observed.shape[-1] != 3:
        raise ValueError("observed_points_m must have shape (view, frame, node, 3)")
    view_count, available_frames, node_count, _ = observed.shape
    if predicted.shape != (available_frames, node_count, 3) or not np.all(
        np.isfinite(predicted)
    ):
        raise ValueError(
            "predicted_points_m must be finite with shape (frame, node, 3)"
        )
    if validity.dtype.kind != "b" or validity.shape != observed.shape[:-1]:
        raise ValueError("validity_mask must be boolean with shape (view, frame, node)")
    if view_count < 2 or node_count < 1:
        raise ValueError("residual localization requires two views and one node")
    if causal_prefix_frame_stop is None:
        frame_count = available_frames
    elif (
        type(causal_prefix_frame_stop) is not int
        or not 1 <= causal_prefix_frame_stop <= available_frames
    ):
        raise ValueError(
            "causal_prefix_frame_stop must select a nonempty available prefix"
        )
    else:
        frame_count = causal_prefix_frame_stop
    if type(maximum_design_bytes) is not int or maximum_design_bytes < 1:
        raise ValueError("maximum_design_bytes must be a positive integer")

    used_observed = observed[:, :frame_count]
    used_predicted = predicted[:frame_count]
    used_validity = validity[:, :frame_count]
    if confidence is None:
        used_confidence = np.ones(used_validity.shape, dtype=float)
    else:
        confidence_values = np.asarray(confidence, dtype=float)
        if confidence_values.shape != observed.shape[:-1]:
            raise ValueError("confidence must have shape (view, frame, node)")
        used_confidence = confidence_values[:, :frame_count]
        if (
            not np.all(np.isfinite(used_confidence))
            or np.any(used_confidence < 0.0)
            or np.any(used_confidence > 1.0)
        ):
            raise ValueError("confidence must contain finite probabilities")

    weights_full = used_confidence * used_validity
    positive = weights_full > 0.0
    if np.any(~np.isfinite(used_observed[positive])):
        raise ValueError("valid observed points must be finite")
    view_support = np.sum(weights_full, axis=(1, 2))
    frame_support = np.sum(weights_full, axis=(0, 2))
    if np.any(view_support <= 0.0):
        raise ValueError("every view must contribute positive weighted support")
    if np.any(frame_support <= 0.0):
        raise ValueError("every prefix frame must contribute positive weighted support")
    reference_view = int(np.argmax(view_support))

    basis = _graph_basis(graph_basis, node_count=node_count)
    graph_rank = basis.shape[1]
    view_indices, frame_indices, node_indices = np.nonzero(positive)
    weights = weights_full[view_indices, frame_indices, node_indices]
    observations = (
        used_observed[view_indices, frame_indices, node_indices]
        - used_predicted[frame_indices, node_indices]
    )
    observation_count = len(weights)

    view_count_parameters = view_count - 1
    frame_start = view_count_parameters
    graph_start = frame_start + frame_count
    parameter_count = graph_start + frame_count * graph_rank
    itemsize = np.dtype(float).itemsize
    estimated_bytes = itemsize * (
        observation_count * (parameter_count + 10) + parameter_count * parameter_count
    )
    if estimated_bytes > maximum_design_bytes:
        raise MemoryError(
            "per-view residual design exceeds maximum_design_bytes before allocation"
        )

    design = np.zeros((observation_count, parameter_count), dtype=float)
    view_columns: dict[int, int] = {}
    for view in range(view_count):
        if view != reference_view:
            view_columns[view] = len(view_columns)
    nonreference = view_indices != reference_view
    if np.any(nonreference):
        rows = np.flatnonzero(nonreference)
        columns = np.asarray(
            [view_columns[int(view)] for view in view_indices[nonreference]],
            dtype=np.int64,
        )
        design[rows, columns] = 1.0
    design[
        np.arange(observation_count),
        frame_start + frame_indices,
    ] = 1.0
    if graph_rank:
        rows = np.arange(observation_count)[:, None]
        columns = (
            graph_start
            + frame_indices[:, None] * graph_rank
            + np.arange(graph_rank)[None, :]
        )
        design[rows, columns] = basis[node_indices]

    ridge = np.empty(parameter_count, dtype=float)
    ridge[:frame_start] = _finite_nonnegative(view_ridge, name="view_ridge")
    ridge[frame_start:graph_start] = _finite_nonnegative(
        frame_ridge,
        name="frame_ridge",
    )
    ridge[graph_start:] = _finite_nonnegative(graph_ridge, name="graph_ridge")
    minimum = _fraction(
        minimum_explained_fraction,
        name="minimum_explained_fraction",
    )
    margin = _fraction(dominance_margin, name="dominance_margin")

    view_columns_array = np.arange(frame_start, dtype=np.int64)
    frame_columns = np.arange(frame_start, graph_start, dtype=np.int64)
    graph_columns = np.arange(graph_start, parameter_count, dtype=np.int64)
    all_columns = np.arange(parameter_count, dtype=np.int64)
    no_view = np.concatenate((frame_columns, graph_columns))
    no_frame = np.concatenate((view_columns_array, graph_columns))
    no_graph = np.concatenate((view_columns_array, frame_columns))

    coefficients, prediction, full_sse = _fit(
        design,
        observations,
        weights,
        ridge,
        all_columns,
    )
    _, _, frame_only_sse = _fit(
        design,
        observations,
        weights,
        ridge,
        frame_columns,
    )
    _, _, no_view_sse = _fit(design, observations, weights, ridge, no_view)
    _, _, no_frame_sse = _fit(design, observations, weights, ridge, no_frame)
    _, _, no_graph_sse = _fit(design, observations, weights, ridge, no_graph)
    total_sse = float(np.sum(weights[:, None] * np.square(observations)))
    full_explained = _explained(total_sse - full_sse, total_sse)
    unique = {
        "view_specific": _explained(no_view_sse - full_sse, total_sse),
        "shared_frame": _explained(no_frame_sse - full_sse, total_sse),
        "object_coherent": _explained(no_graph_sse - full_sse, total_sse),
    }
    ordered = sorted(unique.items(), key=lambda item: item[1], reverse=True)
    if full_explained < minimum or ordered[0][1] < minimum:
        dominant: DominantResidualSource = "unresolved"
    elif ordered[0][1] - ordered[1][1] <= margin:
        dominant = "mixed"
    else:
        dominant = cast(DominantResidualSource, ordered[0][0])

    view_bias = np.zeros((view_count, 3), dtype=float)
    for view, column in view_columns.items():
        view_bias[view] = coefficients[column]
    shared_frame = coefficients[frame_start:graph_start]
    if graph_rank:
        graph_coefficients = coefficients[graph_start:].reshape(
            frame_count,
            graph_rank,
            3,
        )
        graph_field = np.einsum("nr,trc->tnc", basis, graph_coefficients)
    else:
        graph_coefficients = np.empty((frame_count, 0, 3), dtype=float)
        graph_field = np.zeros((frame_count, node_count, 3), dtype=float)

    residual_after = observations - prediction
    rms_before = np.empty(view_count, dtype=float)
    rms_after = np.empty(view_count, dtype=float)
    valid_counts = np.bincount(view_indices, minlength=view_count)
    for view in range(view_count):
        selected = view_indices == view
        denominator = 3.0 * float(np.sum(weights[selected]))
        rms_before[view] = np.sqrt(
            np.sum(weights[selected, None] * np.square(observations[selected]))
            / denominator
        )
        rms_after[view] = np.sqrt(
            np.sum(weights[selected, None] * np.square(residual_after[selected]))
            / denominator
        )

    return PerViewResidualLocalizationResult(
        evidence_artifact_id=evidence_artifact_id,
        causal_prefix_frame_stop=frame_count,
        reference_view_index=reference_view,
        view_count=view_count,
        frame_count=frame_count,
        node_count=node_count,
        graph_rank=graph_rank,
        observed_points_sha256=array_sha256(used_observed),
        predicted_points_sha256=array_sha256(used_predicted),
        validity_mask_sha256=array_sha256(used_validity),
        confidence_sha256=array_sha256(used_confidence),
        graph_basis_sha256=None if not graph_rank else array_sha256(basis),
        view_ridge=float(view_ridge),
        frame_ridge=float(frame_ridge),
        graph_ridge=float(graph_ridge),
        maximum_design_bytes=maximum_design_bytes,
        view_bias_m=view_bias,
        shared_frame_offset_m=shared_frame,
        graph_coefficients_m=graph_coefficients,
        graph_field_m=graph_field,
        per_view_rms_before_m=rms_before,
        per_view_rms_after_m=rms_after,
        valid_observation_counts=valid_counts,
        total_weighted_sse_m2=total_sse,
        full_weighted_sse_m2=full_sse,
        frame_only_weighted_sse_m2=frame_only_sse,
        no_view_weighted_sse_m2=no_view_sse,
        no_frame_weighted_sse_m2=no_frame_sse,
        no_graph_weighted_sse_m2=no_graph_sse,
        full_explained_fraction=full_explained,
        view_unique_fraction=unique["view_specific"],
        frame_unique_fraction=unique["shared_frame"],
        graph_unique_fraction=unique["object_coherent"],
        dominant_source=dominant,
    )


__all__ = [
    "DominantResidualSource",
    "PER_VIEW_RESIDUAL_LOCALIZATION_SCHEMA_VERSION",
    "PerViewResidualLocalizationResult",
    "localize_per_view_residuals",
]
