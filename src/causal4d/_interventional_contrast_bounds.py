"""Coupling-robust bounds for an interventional contrast posterior."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix, csr_matrix
from scipy.special import ndtr

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d._interventional_contrast_common import (
    ContrastConditionalVariancePolicy,
    ContrastCouplingPolicy,
    _require_mapping,
    _require_nonempty_string,
    _require_positive_integer,
    _require_sha256,
    _validated_string_tuple,
)
from causal4d._interventional_contrast_posterior import (
    InterventionalContrastPosteriorV1,
)


INTERVENTIONAL_CONTRAST_BOUNDS_SCHEMA_VERSION = 1
_BOUNDS_ARTIFACT_KIND = "Causal4DInterventionalContrastBoundsV1"
_BOUNDS_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "source_contrast_id",
        "source_query_id",
        "branch_a_label",
        "branch_b_label",
        "coupling_policy",
        "shared_kappa_names",
        "conditional_variance_policy",
        "query_name",
        "query_labels",
        "query_units",
        "metadata",
    }
)
_BOUNDS_ARRAY_FIELDS = frozenset(
    {
        "cdf_thresholds",
        "cdf_lower",
        "cdf_upper",
        "mean",
        "variance_lower",
        "variance_upper",
        "probability_positive_lower",
        "probability_positive_upper",
        "source_variance",
        "source_probability_positive",
    }
)
_BOUNDS_ARRAY_DTYPES = {
    name: np.dtype(np.float64) for name in _BOUNDS_ARRAY_FIELDS
}
_BOUNDS_CLAIM_BOUNDARY = {
    "analysis_only": True,
    "changes_estimator": False,
    "changes_source_posterior": False,
    "changes_registered_protocol": False,
    "uses_target_truth": False,
    "individual_real_counterfactual_ground_truth_claimed": False,
    "coupling_identified_from_branch_marginals": False,
}


def _validated_probability_bounds(
    lower_values: Any,
    upper_values: Any,
    *,
    shape: tuple[int, ...],
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    lower = readonly_array(lower_values, dtype=float)
    upper = readonly_array(upper_values, dtype=float)
    if lower.shape != shape or upper.shape != shape:
        raise ValueError(f"{name} bounds must both have shape {shape}")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError(f"{name} bounds must be finite")
    tolerance = 1e-10
    if (
        np.any(lower < -tolerance)
        or np.any(upper > 1.0 + tolerance)
        or np.any(lower > upper + tolerance)
    ):
        raise ValueError(f"{name} bounds must satisfy 0 <= lower <= upper <= 1")
    return (
        readonly_array(np.clip(lower, 0.0, 1.0), dtype=float),
        readonly_array(np.clip(upper, 0.0, 1.0), dtype=float),
    )


def _validated_variance_bounds(
    lower_values: Any,
    upper_values: Any,
    *,
    output_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    lower = readonly_array(lower_values, dtype=float)
    upper = readonly_array(upper_values, dtype=float)
    expected = (output_count,)
    if lower.shape != expected or upper.shape != expected:
        raise ValueError(f"variance bounds must both have shape {expected}")
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
        raise ValueError("variance bounds must be finite")
    tolerance = 1e-10
    if np.any(lower < -tolerance) or np.any(lower > upper + tolerance):
        raise ValueError("variance bounds must satisfy 0 <= lower <= upper")
    return (
        readonly_array(np.maximum(lower, 0.0), dtype=float),
        readonly_array(np.maximum(upper, 0.0), dtype=float),
    )


@dataclass(frozen=True)
class InterventionalContrastBoundsV1:
    """Coordinatewise sharp bounds over the declared allowed pair support.

    Each lower or upper endpoint may be attained by a different coupling. The
    artifact therefore reports marginal sensitivity bounds, not one joint
    extremal posterior over all query outputs and thresholds simultaneously.
    """

    source_contrast_id: str
    source_query_id: str
    branch_a_label: str
    branch_b_label: str
    coupling_policy: ContrastCouplingPolicy
    shared_kappa_names: tuple[str, ...]
    conditional_variance_policy: ContrastConditionalVariancePolicy
    query_name: str
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    cdf_thresholds: np.ndarray
    cdf_lower: np.ndarray
    cdf_upper: np.ndarray
    mean: np.ndarray
    variance_lower: np.ndarray
    variance_upper: np.ndarray
    probability_positive_lower: np.ndarray
    probability_positive_upper: np.ndarray
    source_variance: np.ndarray
    source_probability_positive: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        source_contrast_id = _require_sha256(
            self.source_contrast_id,
            name="source_contrast_id",
        )
        source_query_id = _require_sha256(
            self.source_query_id,
            name="source_query_id",
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
        variance_policy = _require_nonempty_string(
            self.conditional_variance_policy,
            name="conditional_variance_policy",
        )
        if variance_policy not in {
            "component_means_only",
            "independent_readout",
        }:
            raise ValueError("unsupported conditional variance policy")
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
        output_count = len(query_labels)
        thresholds = readonly_array(self.cdf_thresholds, dtype=float)
        if (
            thresholds.ndim != 2
            or thresholds.shape[0] == 0
            or thresholds.shape[1] != output_count
            or not np.all(np.isfinite(thresholds))
        ):
            raise ValueError(
                "cdf_thresholds must have finite shape (threshold, query_output)"
            )
        cdf_lower, cdf_upper = _validated_probability_bounds(
            self.cdf_lower,
            self.cdf_upper,
            shape=thresholds.shape,
            name="CDF",
        )
        mean = readonly_array(self.mean, dtype=float)
        if mean.shape != (output_count,) or not np.all(np.isfinite(mean)):
            raise ValueError("mean must be a finite query-output vector")
        variance_lower, variance_upper = _validated_variance_bounds(
            self.variance_lower,
            self.variance_upper,
            output_count=output_count,
        )
        positive_lower, positive_upper = _validated_probability_bounds(
            self.probability_positive_lower,
            self.probability_positive_upper,
            shape=(output_count,),
            name="probability-positive",
        )
        source_variance = readonly_array(self.source_variance, dtype=float)
        source_positive = readonly_array(
            self.source_probability_positive,
            dtype=float,
        )
        if (
            source_variance.shape != (output_count,)
            or not np.all(np.isfinite(source_variance))
            or np.any(source_variance < -1e-10)
        ):
            raise ValueError("source_variance must be finite and nonnegative")
        if (
            source_positive.shape != (output_count,)
            or not np.all(np.isfinite(source_positive))
            or np.any(source_positive < -1e-10)
            or np.any(source_positive > 1.0 + 1e-10)
        ):
            raise ValueError("source_probability_positive must lie in [0, 1]")
        if np.any(source_variance < variance_lower - 1e-8) or np.any(
            source_variance > variance_upper + 1e-8
        ):
            raise ValueError("source variance lies outside its coupling bounds")
        if np.any(source_positive < positive_lower - 1e-8) or np.any(
            source_positive > positive_upper + 1e-8
        ):
            raise ValueError(
                "source probability-positive lies outside its coupling bounds"
            )

        object.__setattr__(self, "source_contrast_id", source_contrast_id)
        object.__setattr__(self, "source_query_id", source_query_id)
        object.__setattr__(self, "branch_a_label", branch_a)
        object.__setattr__(self, "branch_b_label", branch_b)
        object.__setattr__(self, "coupling_policy", coupling)
        object.__setattr__(self, "shared_kappa_names", shared_kappa_names)
        object.__setattr__(self, "conditional_variance_policy", variance_policy)
        object.__setattr__(self, "query_name", query_name)
        object.__setattr__(self, "query_labels", query_labels)
        object.__setattr__(self, "query_units", query_units)
        object.__setattr__(self, "cdf_thresholds", thresholds)
        object.__setattr__(self, "cdf_lower", cdf_lower)
        object.__setattr__(self, "cdf_upper", cdf_upper)
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "variance_lower", variance_lower)
        object.__setattr__(self, "variance_upper", variance_upper)
        object.__setattr__(self, "probability_positive_lower", positive_lower)
        object.__setattr__(self, "probability_positive_upper", positive_upper)
        object.__setattr__(
            self,
            "source_variance",
            readonly_array(np.maximum(source_variance, 0.0), dtype=float),
        )
        object.__setattr__(
            self,
            "source_probability_positive",
            readonly_array(np.clip(source_positive, 0.0, 1.0), dtype=float),
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="contrast-bound metadata must contain finite JSON data",
            ),
        )

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INTERVENTIONAL_CONTRAST_BOUNDS_SCHEMA_VERSION,
            "artifact_kind": _BOUNDS_ARTIFACT_KIND,
            "source_contrast_id": self.source_contrast_id,
            "source_query_id": self.source_query_id,
            "branch_a_label": self.branch_a_label,
            "branch_b_label": self.branch_b_label,
            "coupling_policy": self.coupling_policy,
            "shared_kappa_names": list(self.shared_kappa_names),
            "conditional_variance_policy": self.conditional_variance_policy,
            "query_name": self.query_name,
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "cdf_thresholds": self.cdf_thresholds,
            "cdf_lower": self.cdf_lower,
            "cdf_upper": self.cdf_upper,
            "mean": self.mean,
            "variance_lower": self.variance_lower,
            "variance_upper": self.variance_upper,
            "probability_positive_lower": self.probability_positive_lower,
            "probability_positive_upper": self.probability_positive_upper,
            "source_variance": self.source_variance,
            "source_probability_positive": self.source_probability_positive,
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
        return {
            **self._scalar_payload(),
            "artifact_id": self.artifact_id,
            "cdf_thresholds": self.cdf_thresholds.tolist(),
            "cdf_lower": self.cdf_lower.tolist(),
            "cdf_upper": self.cdf_upper.tolist(),
            "mean": self.mean.tolist(),
            "variance_lower": self.variance_lower.tolist(),
            "variance_upper": self.variance_upper.tolist(),
            "probability_positive_lower": (
                self.probability_positive_lower.tolist()
            ),
            "probability_positive_upper": (
                self.probability_positive_upper.tolist()
            ),
            "source_variance": self.source_variance.tolist(),
            "source_probability_positive": (
                self.source_probability_positive.tolist()
            ),
        }


def _validated_thresholds(
    values: Any,
    *,
    output_count: int,
) -> np.ndarray:
    if values is None:
        return np.zeros((1, output_count), dtype=float)
    thresholds = np.asarray(values, dtype=float)
    if thresholds.ndim == 0:
        thresholds = np.full((1, output_count), float(thresholds), dtype=float)
    elif thresholds.ndim == 1:
        if output_count == 1:
            thresholds = thresholds[:, None]
        elif thresholds.shape == (output_count,):
            thresholds = thresholds[None]
        else:
            raise ValueError(
                "one-dimensional cdf_thresholds must match query outputs"
            )
    if (
        thresholds.ndim != 2
        or thresholds.shape[0] == 0
        or thresholds.shape[1] != output_count
        or not np.all(np.isfinite(thresholds))
    ):
        raise ValueError(
            "cdf_thresholds must have finite shape (threshold, query_output)"
        )
    return thresholds


def _transport_constraints(
    posterior: InterventionalContrastPosteriorV1,
    *,
    marginal_tolerance: float,
) -> tuple[csr_matrix, np.ndarray, float, float]:
    pairs = np.asarray(posterior.pair_indices, dtype=np.int64)
    weights = np.asarray(posterior.weights, dtype=float)
    row_indices = np.unique(pairs[:, 0])
    column_indices = np.unique(pairs[:, 1])
    row_lookup = {int(value): index for index, value in enumerate(row_indices)}
    column_lookup = {
        int(value): index for index, value in enumerate(column_indices)
    }
    variable_count = len(pairs)
    variable_indices = np.arange(variable_count, dtype=np.int64)
    constraint_rows = np.concatenate(
        (
            np.asarray([row_lookup[int(value)] for value in pairs[:, 0]]),
            len(row_indices)
            + np.asarray(
                [column_lookup[int(value)] for value in pairs[:, 1]],
                dtype=np.int64,
            ),
        )
    )
    constraint_columns = np.concatenate((variable_indices, variable_indices))
    constraints = coo_matrix(
        (
            np.ones(2 * variable_count, dtype=float),
            (constraint_rows, constraint_columns),
        ),
        shape=(len(row_indices) + len(column_indices), variable_count),
    ).tocsr()
    branch_a_marginal = np.bincount(
        pairs[:, 0],
        weights=weights,
        minlength=posterior.branch_a_component_count,
    )
    branch_b_marginal = np.bincount(
        pairs[:, 1],
        weights=weights,
        minlength=posterior.branch_b_component_count,
    )
    right_hand_side = np.concatenate(
        (branch_a_marginal[row_indices], branch_b_marginal[column_indices])
    )
    residual = np.asarray(constraints @ weights - right_hand_side)
    maximum_error = float(np.max(np.abs(residual), initial=0.0))
    if maximum_error > marginal_tolerance:
        raise ValueError("source contrast weights do not preserve their marginals")
    return constraints, right_hand_side, maximum_error, float(np.sum(weights))


def _optimize_transport(
    cost: np.ndarray,
    *,
    constraints: csr_matrix,
    right_hand_side: np.ndarray,
    maximize: bool,
) -> float:
    objective = -np.asarray(cost, dtype=float) if maximize else np.asarray(cost)
    if objective.ndim != 1 or objective.shape[0] != constraints.shape[1]:
        raise ValueError("transport objective must match the allowed pair support")
    if not np.all(np.isfinite(objective)):
        raise ValueError("transport objective must be finite")
    result = linprog(
        objective,
        A_eq=constraints,
        b_eq=right_hand_side,
        bounds=(0.0, None),
        method="highs",
    )
    if not result.success or result.fun is None or not np.isfinite(result.fun):
        raise RuntimeError(
            "coupling-bound transport optimization failed: "
            f"status={result.status}, message={result.message}"
        )
    value = float(-result.fun if maximize else result.fun)
    return value


def _component_cdf(
    means: np.ndarray,
    variances: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probabilities = np.empty_like(means, dtype=float)
    positive_variance = variances > 0.0
    probabilities[positive_variance] = ndtr(
        (threshold - means[positive_variance])
        / np.sqrt(variances[positive_variance])
    )
    probabilities[~positive_variance] = means[~positive_variance] <= threshold
    return probabilities


def _component_probability_positive(
    means: np.ndarray,
    variances: np.ndarray,
) -> np.ndarray:
    probabilities = np.empty_like(means, dtype=float)
    positive_variance = variances > 0.0
    probabilities[positive_variance] = ndtr(
        means[positive_variance] / np.sqrt(variances[positive_variance])
    )
    probabilities[~positive_variance] = means[~positive_variance] > 0.0
    return probabilities


def build_interventional_contrast_bounds(
    posterior: InterventionalContrastPosteriorV1,
    *,
    cdf_thresholds: Any = None,
    maximum_pair_count: int = 1_000_000,
    marginal_tolerance: float = 1e-12,
    metadata: Mapping[str, Any] | None = None,
) -> InterventionalContrastBoundsV1:
    """Bound coupling-dependent summaries over the allowed pair support.

    The optimization changes only pair weights. It preserves both source
    marginals exactly and never introduces a pair that was absent from the
    source contrast artifact. Thus an ``independent_product`` source gives the
    complete marginal Fréchet class, ``shared_twin_phi`` gives stratum-restricted
    bounds, and ``shared_component`` collapses to the source coupling.
    """

    if not isinstance(posterior, InterventionalContrastPosteriorV1):
        raise TypeError("posterior must be InterventionalContrastPosteriorV1")
    pair_limit = _require_positive_integer(
        maximum_pair_count,
        name="maximum_pair_count",
    )
    if len(posterior.pair_indices) > pair_limit:
        raise ValueError(
            "contrast bound requires "
            f"{len(posterior.pair_indices)} pairs, exceeding "
            f"maximum_pair_count={pair_limit}"
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
    output_count = len(posterior.query_labels)
    thresholds = _validated_thresholds(
        cdf_thresholds,
        output_count=output_count,
    )
    constraints, right_hand_side, marginal_error, source_mass = (
        _transport_constraints(
            posterior,
            marginal_tolerance=tolerance,
        )
    )
    if not np.isclose(source_mass, 1.0, atol=tolerance, rtol=tolerance):
        raise ValueError("source contrast pair weights must sum to one")

    component_means = np.asarray(posterior.contrast_values, dtype=float)
    component_variances = np.diagonal(
        np.asarray(posterior.conditional_covariance, dtype=float),
        axis1=-2,
        axis2=-1,
    )
    source_weights = np.asarray(posterior.weights, dtype=float)
    source_mean = np.asarray(posterior.mean, dtype=float)

    cdf_lower = np.empty_like(thresholds)
    cdf_upper = np.empty_like(thresholds)
    for threshold_index in range(len(thresholds)):
        for output_index in range(output_count):
            cost = _component_cdf(
                component_means[:, output_index],
                component_variances[:, output_index],
                float(thresholds[threshold_index, output_index]),
            )
            cdf_lower[threshold_index, output_index] = _optimize_transport(
                cost,
                constraints=constraints,
                right_hand_side=right_hand_side,
                maximize=False,
            )
            cdf_upper[threshold_index, output_index] = _optimize_transport(
                cost,
                constraints=constraints,
                right_hand_side=right_hand_side,
                maximize=True,
            )

    positive_lower = np.empty(output_count, dtype=float)
    positive_upper = np.empty(output_count, dtype=float)
    source_positive = np.empty(output_count, dtype=float)
    variance_lower = np.empty(output_count, dtype=float)
    variance_upper = np.empty(output_count, dtype=float)
    source_variance = np.empty(output_count, dtype=float)
    for output_index in range(output_count):
        positive_cost = _component_probability_positive(
            component_means[:, output_index],
            component_variances[:, output_index],
        )
        positive_lower[output_index] = _optimize_transport(
            positive_cost,
            constraints=constraints,
            right_hand_side=right_hand_side,
            maximize=False,
        )
        positive_upper[output_index] = _optimize_transport(
            positive_cost,
            constraints=constraints,
            right_hand_side=right_hand_side,
            maximize=True,
        )
        source_positive[output_index] = float(source_weights @ positive_cost)

        second_moment_cost = (
            np.square(component_means[:, output_index])
            + component_variances[:, output_index]
        )
        minimum_second_moment = _optimize_transport(
            second_moment_cost,
            constraints=constraints,
            right_hand_side=right_hand_side,
            maximize=False,
        )
        maximum_second_moment = _optimize_transport(
            second_moment_cost,
            constraints=constraints,
            right_hand_side=right_hand_side,
            maximize=True,
        )
        squared_mean = float(np.square(source_mean[output_index]))
        variance_lower[output_index] = max(
            0.0,
            minimum_second_moment - squared_mean,
        )
        variance_upper[output_index] = max(
            0.0,
            maximum_second_moment - squared_mean,
        )
        source_second_moment = float(source_weights @ second_moment_cost)
        source_variance[output_index] = max(
            0.0,
            source_second_moment - squared_mean,
        )

    result_metadata = {
        "claim_boundary": _BOUNDS_CLAIM_BOUNDARY,
        "bound_semantics": (
            "coordinatewise sharp over nonnegative pair weights preserving "
            "both source marginals and the source artifact's allowed pair support"
        ),
        "cdf_event": "contrast <= threshold",
        "probability_positive_event": "contrast > 0",
        "coordinate_extrema_may_use_different_couplings": True,
        "source_pair_count": len(posterior.pair_indices),
        "source_marginal_max_abs_error": marginal_error,
        "source_marginal_tolerance": tolerance,
        "transport_solver": "scipy.optimize.linprog(method='highs')",
        "user": plain_json(user_metadata),
    }
    return InterventionalContrastBoundsV1(
        source_contrast_id=posterior.artifact_id,
        source_query_id=posterior.query_id,
        branch_a_label=posterior.branch_a_label,
        branch_b_label=posterior.branch_b_label,
        coupling_policy=posterior.coupling_policy,
        shared_kappa_names=posterior.shared_kappa_names,
        conditional_variance_policy=posterior.conditional_variance_policy,
        query_name=posterior.query_name,
        query_labels=posterior.query_labels,
        query_units=posterior.query_units,
        cdf_thresholds=thresholds,
        cdf_lower=np.clip(cdf_lower, 0.0, 1.0),
        cdf_upper=np.clip(cdf_upper, 0.0, 1.0),
        mean=source_mean,
        variance_lower=variance_lower,
        variance_upper=variance_upper,
        probability_positive_lower=np.clip(positive_lower, 0.0, 1.0),
        probability_positive_upper=np.clip(positive_upper, 0.0, 1.0),
        source_variance=source_variance,
        source_probability_positive=np.clip(source_positive, 0.0, 1.0),
        metadata=result_metadata,
    )
