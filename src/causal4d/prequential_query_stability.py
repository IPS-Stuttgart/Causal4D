"""Registered-query diagnostics over leakage-safe prequential posteriors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.prequential_abduction import PrequentialAbductionPathV1


PREQUENTIAL_QUERY_STABILITY_SCHEMA_VERSION = 1
PREQUENTIAL_QUERY_STABILITY_CLAIM_BOUNDARY = (
    "Diagnostic projection of an unchanged prequential factual-abduction path. "
    "It does not select a prefix, change the estimator, establish calibration, "
    "or authorize confirmatory use."
)
_PREQUENTIAL_QUERY_STABILITY_KIND = "Causal4DPrequentialQueryStabilityV1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _string_tuple(
    values: Sequence[str],
    *,
    name: str,
    expected_count: int,
    unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if len(result) != expected_count or any(
        type(value) is not str or not value for value in result
    ):
        raise ValueError(
            f"{name} must contain {expected_count} nonempty strings"
        )
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _positive_vector(values: object, *, length: int, name: str) -> np.ndarray:
    result = readonly_array(values, dtype=float)
    if result.shape != (length,):
        raise ValueError(f"{name} must have shape ({length},)")
    if not np.all(np.isfinite(result)) or np.any(result <= 0.0):
        raise ValueError(f"{name} must be finite and positive")
    return result


def _weighted_quantile(
    values: np.ndarray,
    weights: np.ndarray,
    probability: float,
) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, probability, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _psd_square_root(matrix: np.ndarray, *, name: str) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    scale = max(1.0, float(np.max(np.abs(eigenvalues), initial=0.0)))
    tolerance = 1.0e-10 * scale
    if float(np.min(eigenvalues, initial=0.0)) < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    clipped = np.maximum(eigenvalues, 0.0)
    return (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T


def _gaussian_wasserstein_distance(
    first_mean: np.ndarray,
    first_covariance: np.ndarray,
    second_mean: np.ndarray,
    second_covariance: np.ndarray,
) -> float:
    second_root = _psd_square_root(
        second_covariance,
        name="second Gaussian covariance",
    )
    middle_root = _psd_square_root(
        second_root @ first_covariance @ second_root,
        name="Gaussian covariance transport term",
    )
    mean_difference = first_mean - second_mean
    squared = float(
        mean_difference @ mean_difference
        + np.trace(first_covariance)
        + np.trace(second_covariance)
        - 2.0 * np.trace(middle_root)
    )
    tolerance = 1.0e-10 * max(
        1.0,
        float(np.trace(first_covariance) + np.trace(second_covariance)),
    )
    if squared < -tolerance:
        raise ValueError("Gaussian Wasserstein distance became negative")
    return float(np.sqrt(max(0.0, squared)))


def _interval_overlap_fraction(
    first_lower: np.ndarray,
    first_upper: np.ndarray,
    second_lower: np.ndarray,
    second_upper: np.ndarray,
) -> float:
    overlap = np.maximum(
        0.0,
        np.minimum(first_upper, second_upper)
        - np.maximum(first_lower, second_lower),
    )
    union = np.maximum(first_upper, second_upper) - np.minimum(
        first_lower,
        second_lower,
    )
    fractions = np.empty_like(union)
    nondegenerate = union > 0.0
    fractions[nondegenerate] = overlap[nondegenerate] / union[nondegenerate]
    fractions[~nondegenerate] = np.isclose(
        first_lower[~nondegenerate],
        second_lower[~nondegenerate],
        atol=0.0,
        rtol=0.0,
    ).astype(float)
    return float(np.mean(fractions))


@dataclass(frozen=True)
class PrequentialQueryStabilityV1:
    """Query-space evolution of one immutable prequential posterior path."""

    source_prequential_path_id: str
    query_id: str
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    query_scales: np.ndarray
    confidence_level: float
    prefix_frame_counts: np.ndarray
    posterior_weights: np.ndarray
    component_query_values: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        path_id = _require_sha256(
            self.source_prequential_path_id,
            name="source_prequential_path_id",
        )
        query_id = _require_nonempty_string(self.query_id, name="query_id")
        query_values = readonly_array(self.component_query_values, dtype=float)
        if query_values.ndim != 2 or min(query_values.shape) < 1:
            raise ValueError(
                "component_query_values must have shape (component, query)"
            )
        if not np.all(np.isfinite(query_values)):
            raise ValueError("component_query_values must be finite")
        component_count, query_dimension = query_values.shape
        labels = _string_tuple(
            self.query_labels,
            name="query_labels",
            expected_count=query_dimension,
            unique=True,
        )
        units = _string_tuple(
            self.query_units,
            name="query_units",
            expected_count=query_dimension,
        )
        scales = _positive_vector(
            self.query_scales,
            length=query_dimension,
            name="query_scales",
        )
        confidence = float(self.confidence_level)
        if (
            isinstance(self.confidence_level, (bool, np.bool_))
            or not np.isfinite(confidence)
            or not 0.0 < confidence < 1.0
        ):
            raise ValueError("confidence_level must lie strictly between zero and one")
        raw_prefixes = np.asarray(self.prefix_frame_counts)
        if (
            raw_prefixes.ndim != 1
            or not len(raw_prefixes)
            or not np.issubdtype(raw_prefixes.dtype, np.integer)
        ):
            raise ValueError("prefix_frame_counts must be a nonempty integer vector")
        prefixes = readonly_array(raw_prefixes, dtype=np.int64)
        if prefixes[0] < 2 or np.any(np.diff(prefixes) <= 0):
            raise ValueError(
                "prefix_frame_counts must be strictly increasing and start at two"
            )
        weights = readonly_array(self.posterior_weights, dtype=float)
        if weights.shape != (len(prefixes), component_count):
            raise ValueError(
                "posterior_weights must have shape (prefix, component)"
            )
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.allclose(
                np.sum(weights, axis=1),
                1.0,
                atol=1.0e-12,
                rtol=1.0e-10,
            )
        ):
            raise ValueError(
                "posterior_weights must be finite, nonnegative, and row-normalized"
            )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="query-stability metadata must contain finite JSON data",
        )
        object.__setattr__(self, "source_prequential_path_id", path_id)
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "query_labels", labels)
        object.__setattr__(self, "query_units", units)
        object.__setattr__(self, "query_scales", scales)
        object.__setattr__(self, "confidence_level", confidence)
        object.__setattr__(self, "prefix_frame_counts", prefixes)
        object.__setattr__(self, "posterior_weights", weights)
        object.__setattr__(self, "component_query_values", query_values)
        object.__setattr__(self, "metadata", metadata)

    @property
    def query_dimension(self) -> int:
        return self.component_query_values.shape[1]

    @property
    def step_count(self) -> int:
        return len(self.prefix_frame_counts)

    def summary_arrays(self) -> dict[str, np.ndarray]:
        """Return deterministic native-unit and standardized diagnostics."""

        means = self.posterior_weights @ self.component_query_values
        centered = self.component_query_values[None] - means[:, None]
        covariance = np.einsum(
            "sc,scd,sce->sde",
            self.posterior_weights,
            centered,
            centered,
        )
        lower_probability = 0.5 * (1.0 - self.confidence_level)
        upper_probability = 1.0 - lower_probability
        lower = np.empty_like(means)
        upper = np.empty_like(means)
        for step in range(self.step_count):
            for coordinate in range(self.query_dimension):
                values = self.component_query_values[:, coordinate]
                weights = self.posterior_weights[step]
                lower[step, coordinate] = _weighted_quantile(
                    values,
                    weights,
                    lower_probability,
                )
                upper[step, coordinate] = _weighted_quantile(
                    values,
                    weights,
                    upper_probability,
                )

        previous_native = np.zeros_like(means)
        previous_native[1:] = np.abs(means[1:] - means[:-1])
        final_native = np.abs(means - means[-1])
        standardized_means = means / self.query_scales
        standardized_covariance = covariance / (
            self.query_scales[None, :, None]
            * self.query_scales[None, None, :]
        )
        previous_mean_l2 = np.zeros(self.step_count, dtype=float)
        previous_mean_l2[1:] = np.linalg.norm(
            standardized_means[1:] - standardized_means[:-1],
            axis=1,
        )
        final_mean_l2 = np.linalg.norm(
            standardized_means - standardized_means[-1],
            axis=1,
        )
        previous_wasserstein = np.zeros(self.step_count, dtype=float)
        final_wasserstein = np.zeros(self.step_count, dtype=float)
        previous_overlap = np.ones(self.step_count, dtype=float)
        final_overlap = np.ones(self.step_count, dtype=float)
        for step in range(1, self.step_count):
            previous_wasserstein[step] = _gaussian_wasserstein_distance(
                standardized_means[step - 1],
                standardized_covariance[step - 1],
                standardized_means[step],
                standardized_covariance[step],
            )
            previous_overlap[step] = _interval_overlap_fraction(
                lower[step - 1],
                upper[step - 1],
                lower[step],
                upper[step],
            )
        for step in range(self.step_count - 1):
            final_wasserstein[step] = _gaussian_wasserstein_distance(
                standardized_means[step],
                standardized_covariance[step],
                standardized_means[-1],
                standardized_covariance[-1],
            )
            final_overlap[step] = _interval_overlap_fraction(
                lower[step],
                upper[step],
                lower[-1],
                upper[-1],
            )
        return {
            "posterior_query_mean": readonly_array(means, dtype=float),
            "posterior_query_covariance": readonly_array(covariance, dtype=float),
            "credible_lower": readonly_array(lower, dtype=float),
            "credible_upper": readonly_array(upper, dtype=float),
            "previous_mean_shift_native": readonly_array(
                previous_native,
                dtype=float,
            ),
            "final_mean_shift_native": readonly_array(final_native, dtype=float),
            "previous_mean_shift_standardized_l2": readonly_array(
                previous_mean_l2,
                dtype=float,
            ),
            "final_mean_shift_standardized_l2": readonly_array(
                final_mean_l2,
                dtype=float,
            ),
            "previous_gaussian_wasserstein_standardized": readonly_array(
                previous_wasserstein,
                dtype=float,
            ),
            "final_gaussian_wasserstein_standardized": readonly_array(
                final_wasserstein,
                dtype=float,
            ),
            "previous_interval_overlap_fraction": readonly_array(
                previous_overlap,
                dtype=float,
            ),
            "final_interval_overlap_fraction": readonly_array(
                final_overlap,
                dtype=float,
            ),
        }

    @property
    def artifact_id(self) -> str:
        payload = {
            "schema_version": PREQUENTIAL_QUERY_STABILITY_SCHEMA_VERSION,
            "artifact_kind": _PREQUENTIAL_QUERY_STABILITY_KIND,
            "source_prequential_path_id": self.source_prequential_path_id,
            "query_id": self.query_id,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "confidence_level": self.confidence_level,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PREQUENTIAL_QUERY_STABILITY_CLAIM_BOUNDARY,
            "arrays": {
                "query_scales": array_sha256(self.query_scales),
                "prefix_frame_counts": array_sha256(self.prefix_frame_counts),
                "posterior_weights": array_sha256(self.posterior_weights),
                "component_query_values": array_sha256(
                    self.component_query_values
                ),
            },
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        summaries = self.summary_arrays()
        return {
            "schema_version": PREQUENTIAL_QUERY_STABILITY_SCHEMA_VERSION,
            "artifact_kind": _PREQUENTIAL_QUERY_STABILITY_KIND,
            "artifact_id": self.artifact_id,
            "source_prequential_path_id": self.source_prequential_path_id,
            "query_id": self.query_id,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "query_scales": self.query_scales.tolist(),
            "confidence_level": self.confidence_level,
            "prefix_frame_counts": self.prefix_frame_counts.tolist(),
            "component_query_values_sha256": array_sha256(
                self.component_query_values
            ),
            "posterior_weights_sha256": array_sha256(self.posterior_weights),
            "summaries": {
                name: values.tolist() for name, values in summaries.items()
            },
            "metadata": plain_json(self.metadata),
            "claim_boundary": PREQUENTIAL_QUERY_STABILITY_CLAIM_BOUNDARY,
        }


def build_prequential_query_stability(
    path: PrequentialAbductionPathV1,
    component_query_values: np.ndarray,
    *,
    query_id: str,
    query_labels: Sequence[str],
    query_units: Sequence[str],
    query_scales: Sequence[float],
    confidence_level: float = 0.90,
    metadata: Mapping[str, Any] | None = None,
) -> PrequentialQueryStabilityV1:
    """Project one prequential posterior path into a registered query space."""

    if not isinstance(path, PrequentialAbductionPathV1):
        raise TypeError("path must be PrequentialAbductionPathV1")
    values = np.asarray(component_query_values, dtype=float)
    if values.ndim != 2 or values.shape[0] != len(path.component_ids):
        raise ValueError(
            "component_query_values must contain one row per path component"
        )
    result = PrequentialQueryStabilityV1(
        source_prequential_path_id=path.artifact_id,
        query_id=query_id,
        query_labels=tuple(query_labels),
        query_units=tuple(query_units),
        query_scales=np.asarray(tuple(query_scales), dtype=float),
        confidence_level=confidence_level,
        prefix_frame_counts=path.prefix_frame_counts,
        posterior_weights=path.posterior_weights,
        component_query_values=values,
        metadata={
            "diagnostic_only": True,
            "changes_estimator": False,
            "selects_prefix": False,
            "future_frames_read": 0,
            "user_metadata": plain_json(metadata or {}),
        },
    )
    result.summary_arrays()
    return result


__all__ = [
    "PREQUENTIAL_QUERY_STABILITY_CLAIM_BOUNDARY",
    "PREQUENTIAL_QUERY_STABILITY_SCHEMA_VERSION",
    "PrequentialQueryStabilityV1",
    "build_prequential_query_stability",
]
