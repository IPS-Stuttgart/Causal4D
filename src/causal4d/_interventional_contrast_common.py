"""Shared definitions for interventional contrast artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from typing import Any, Literal

import numpy as np
from scipy.special import ndtr

from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json


INTERVENTIONAL_CONTRAST_SCHEMA_VERSION = 1
ContrastCouplingPolicy = Literal[
    "shared_component",
    "shared_twin_phi",
    "independent_product",
]
ContrastConditionalVariancePolicy = Literal[
    "component_means_only",
    "independent_readout",
]

_ARTIFACT_KIND = "Causal4DInterventionalContrastPosteriorV1"
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "source_branch_a_posterior_id",
        "source_branch_b_posterior_id",
        "source_branch_a_query_id",
        "source_branch_b_query_id",
        "branch_a_label",
        "branch_b_label",
        "trajectory_shape",
        "branch_a_component_count",
        "branch_b_component_count",
        "coupling_policy",
        "shared_kappa_names",
        "conditional_variance_policy",
        "query_id",
        "query_name",
        "query_labels",
        "query_units",
        "query_metadata",
        "metadata",
    }
)
_ARRAY_FIELDS = frozenset(
    {
        "pair_indices",
        "weights",
        "query_matrix",
        "contrast_values",
        "conditional_covariance",
    }
)
_ARRAY_DTYPES = {
    "pair_indices": np.dtype(np.int64),
    "weights": np.dtype(np.float64),
    "query_matrix": np.dtype(np.float64),
    "contrast_values": np.dtype(np.float64),
    "conditional_covariance": np.dtype(np.float64),
}
_CLAIM_BOUNDARY = {
    "analysis_only": True,
    "changes_estimator": False,
    "changes_source_posterior": False,
    "changes_registered_protocol": False,
    "uses_target_truth": False,
    "individual_real_counterfactual_ground_truth_claimed": False,
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _validated_string_tuple(
    values: Any,
    *,
    name: str,
    unique: bool,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must be unique")
    return result


def _validated_weights(values: Any, *, expected_count: int) -> np.ndarray:
    weights = readonly_array(values, dtype=float)
    if weights.shape != (expected_count,):
        raise ValueError(f"weights must have shape ({expected_count},)")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=1e-12):
        raise ValueError("weights must sum to one")
    return weights


def _validated_probabilities(values: Any) -> np.ndarray:
    supplied = np.asarray(values)
    if supplied.dtype.kind == "b":
        raise ValueError("probabilities must contain numbers, not Booleans")
    probabilities = readonly_array(supplied, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) == 0:
        raise ValueError("probabilities must be a nonempty vector")
    if (
        not np.all(np.isfinite(probabilities))
        or np.any(probabilities <= 0.0)
        or np.any(probabilities >= 1.0)
    ):
        raise ValueError("probabilities must lie strictly between zero and one")
    if np.any(np.diff(probabilities) <= 0.0):
        raise ValueError("probabilities must be strictly increasing")
    return probabilities


def _mixture_marginal_quantiles(
    means: np.ndarray,
    variances: np.ndarray,
    weights: np.ndarray,
    probabilities: np.ndarray,
) -> np.ndarray:
    output_count = means.shape[1]
    result = np.empty((len(probabilities), output_count), dtype=float)
    for output_index in range(output_count):
        component_means = means[:, output_index]
        component_variances = np.maximum(variances[:, output_index], 0.0)
        component_scales = np.sqrt(component_variances)
        continuous = component_scales > 0.0
        if not np.any(continuous):
            order = np.argsort(component_means, kind="stable")
            cumulative = np.cumsum(weights[order])
            for probability_index, probability in enumerate(probabilities):
                index = int(np.searchsorted(cumulative, probability, side="left"))
                result[probability_index, output_index] = component_means[
                    order[min(index, len(order) - 1)]
                ]
            continue

        def cdf(value: float) -> float:
            total = float(
                np.sum(
                    weights[continuous]
                    * ndtr(
                        (value - component_means[continuous])
                        / component_scales[continuous]
                    )
                )
            )
            if np.any(~continuous):
                total += float(
                    np.sum(
                        weights[~continuous] * (component_means[~continuous] <= value)
                    )
                )
            return total

        scale = max(
            float(np.ptp(component_means)),
            float(np.max(component_scales)),
            np.finfo(float).eps,
        )
        initial_low = float(np.min(component_means) - 8.0 * scale)
        initial_high = float(np.max(component_means) + 8.0 * scale)
        for probability_index, probability in enumerate(probabilities):
            low = initial_low
            high = initial_high
            expansion = scale
            for _ in range(64):
                if cdf(low) < probability:
                    break
                low -= expansion
                expansion *= 2.0
            else:
                raise FloatingPointError("failed to bracket mixture quantile below")
            expansion = scale
            for _ in range(64):
                if cdf(high) >= probability:
                    break
                high += expansion
                expansion *= 2.0
            else:
                raise FloatingPointError("failed to bracket mixture quantile above")
            for _ in range(128):
                midpoint = 0.5 * (low + high)
                if cdf(midpoint) >= probability:
                    high = midpoint
                else:
                    low = midpoint
            result[probability_index, output_index] = high
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")
