"""Query-space uncertainty attribution for finite posterior mixtures."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Final, cast

import numpy as np

from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping

QUERY_VARIANCE_DECOMPOSITION_SCHEMA_VERSION: Final = 1
QUERY_VARIANCE_DECOMPOSITION_ARTIFACT_KIND: Final = (
    "Causal4DQueryVarianceDecompositionV1"
)
QUERY_VARIANCE_DECOMPOSITION_CLAIM_BOUNDARY: Final = (
    "Diagnostic attribution of a fixed finite posterior query. It does not "
    "change the posterior, identify a physical cause, establish empirical "
    "calibration, or authorize confirmatory method selection."
)
MAX_EXACT_SHAPLEY_FACTORS: Final = 8
_FACTOR_ATTRIBUTION = "exact_shapley_over_explained_covariance"
_UNRESOLVED_ROLE = (
    "between-component variation not distinguished by the declared factor labels"
)
_CONDITIONAL_SOURCE_SEMANTICS = (
    "caller-declared additive conditional covariance sources; independence or "
    "non-overlap is not inferred by this artifact"
)
_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "decomposition_id",
        "claim_boundary",
        "query",
        "support",
        "input_identities",
        "posterior_mean",
        "covariance",
        "variance_share_by_query_coordinate",
        "standardized_trace_share",
        "diagnostics",
        "metadata",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    _require(
        all(type(key) is str for key in value),
        f"{name} keys must be strings",
    )
    return cast(Mapping[str, Any], value)


def _nonempty_string(value: object, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be nonempty")
    return cast(str, value)


def _string_tuple(
    values: object,
    *,
    name: str,
    expected_count: int | None = None,
    unique: bool = False,
) -> tuple[str, ...]:
    _require(
        isinstance(values, Sequence) and not isinstance(values, (str, bytes)),
        f"{name} must be a sequence of strings",
    )
    result = tuple(
        _nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if expected_count is not None:
        _require(
            len(result) == expected_count,
            f"{name} must contain {expected_count} values",
        )
    if unique:
        _require(len(set(result)) == len(result), f"{name} must be unique")
    return result


def _float_array(values: object, *, name: str) -> np.ndarray:
    source = np.asarray(values)
    _require(
        source.dtype.kind in {"i", "u", "f"},
        f"{name} must contain real numbers without Boolean or string coercion",
    )
    result = np.asarray(source, dtype=np.float64)
    if source.dtype.kind in {"i", "u"} and source.size:
        round_trip = result.astype(source.dtype, copy=False)
        _require(
            np.array_equal(round_trip, source),
            f"{name} contains integers not exactly representable as float64",
        )
    _require(np.all(np.isfinite(result)), f"{name} must be finite")
    return readonly_array(result, dtype=np.float64)


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    header = json.dumps(
        {"dtype": array.dtype.str, "shape": list(array.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _validated_sha256(value: object, *, name: str) -> str:
    text = _nonempty_string(value, name=name)
    _require(
        len(text) == 64 and all(character in "0123456789abcdef" for character in text),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return text


def _canonical_sha256(payload: Mapping[str, Any], *, omitted: str) -> str:
    values = dict(payload)
    values.pop(omitted, None)
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _symmetrize_psd(
    matrix: np.ndarray,
    *,
    name: str,
    tolerance_scale: float = 1.0e-10,
) -> np.ndarray:
    _require(matrix.ndim == 2, f"{name} must be a matrix")
    _require(matrix.shape[0] == matrix.shape[1], f"{name} must be square")
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    scale = max(
        1.0,
        float(np.max(np.abs(eigenvalues), initial=0.0)),
        float(np.linalg.norm(symmetric, ord=2)),
    )
    tolerance = tolerance_scale * scale
    minimum = float(np.min(eigenvalues, initial=0.0))
    _require(minimum >= -tolerance, f"{name} is not positive semidefinite")
    return readonly_array(symmetric, dtype=np.float64)


def _validate_covariance_stack(
    values: object,
    *,
    name: str,
    component_count: int,
    query_dimension: int,
) -> np.ndarray:
    stack = _float_array(values, name=name)
    expected = (component_count, query_dimension, query_dimension)
    _require(stack.shape == expected, f"{name} must have shape {expected}")
    validated = np.empty_like(stack)
    for index, covariance in enumerate(stack):
        validated[index] = _symmetrize_psd(
            covariance,
            name=f"{name}[{index}]",
        )
    return readonly_array(validated, dtype=np.float64)


def _weighted_covariance(
    weights: np.ndarray,
    values: np.ndarray,
    mean: np.ndarray,
) -> np.ndarray:
    centered = values - mean[None, :]
    covariance = np.einsum(
        "c,ci,cj->ij",
        weights,
        centered,
        centered,
        optimize=True,
    )
    return _symmetrize_psd(covariance, name="weighted query covariance")


def _explained_covariance(
    weights: np.ndarray,
    means: np.ndarray,
    global_mean: np.ndarray,
    factors: Mapping[str, tuple[str, ...]],
    selected: tuple[str, ...],
) -> np.ndarray:
    dimension = means.shape[1]
    if not selected:
        return readonly_array(np.zeros((dimension, dimension), dtype=np.float64))

    grouped: dict[tuple[str, ...], tuple[float, np.ndarray]] = {}
    for component in range(means.shape[0]):
        key = tuple(factors[name][component] for name in selected)
        mass, weighted_sum = grouped.get(
            key,
            (0.0, np.zeros(dimension, dtype=np.float64)),
        )
        component_weight = float(weights[component])
        grouped[key] = (
            mass + component_weight,
            weighted_sum + component_weight * means[component],
        )

    covariance = np.zeros((dimension, dimension), dtype=np.float64)
    for mass, weighted_sum in grouped.values():
        _require(mass > 0.0, "factor group has zero posterior mass")
        group_mean = weighted_sum / mass
        difference = group_mean - global_mean
        covariance += mass * np.outer(difference, difference)
    return _symmetrize_psd(covariance, name="factor-explained covariance")


def _normalized_weights(values: object, *, component_count: int) -> np.ndarray:
    weights = _float_array(values, name="component_weights")
    _require(
        weights.shape == (component_count,),
        f"component_weights must have shape ({component_count},)",
    )
    _require(np.all(weights >= 0.0), "component_weights must be nonnegative")
    total = float(np.sum(weights))
    _require(total > 0.0, "component_weights must have positive total mass")
    return readonly_array(weights / total, dtype=np.float64)


def _factor_mapping(
    values: Mapping[str, Sequence[str]],
    *,
    component_count: int,
) -> Mapping[str, tuple[str, ...]]:
    mapping = _mapping(values, name="factor_values")
    names = tuple(sorted(mapping))
    _require(
        len(names) <= MAX_EXACT_SHAPLEY_FACTORS,
        "too many factors for exact Shapley attribution",
    )
    result: dict[str, tuple[str, ...]] = {}
    for name in names:
        factor_name = _nonempty_string(name, name="factor name")
        result[factor_name] = _string_tuple(
            mapping[name],
            name=f"factor_values[{factor_name!r}]",
            expected_count=component_count,
        )
    return MappingProxyType(result)


def _conditional_mapping(
    values: Mapping[str, object],
    *,
    component_count: int,
    query_dimension: int,
) -> Mapping[str, np.ndarray]:
    mapping = _mapping(values, name="conditional_covariances")
    result: dict[str, np.ndarray] = {}
    for name in sorted(mapping):
        source = _nonempty_string(name, name="conditional covariance source")
        result[source] = _validate_covariance_stack(
            mapping[name],
            name=f"conditional_covariances[{source!r}]",
            component_count=component_count,
            query_dimension=query_dimension,
        )
    return MappingProxyType(result)


def _trace_in_registered_scale(
    covariance: np.ndarray,
    scales: np.ndarray,
) -> float:
    inverse = 1.0 / scales
    standardized = covariance * inverse[:, None] * inverse[None, :]
    return float(np.trace(standardized))


def _safe_variance_share(
    contribution: np.ndarray,
    total: np.ndarray,
) -> list[float]:
    total_variance = np.diag(total)
    contribution_variance = np.diag(contribution)
    scale = max(1.0, float(np.max(np.abs(total_variance), initial=0.0)))
    tolerance = 1.0e-12 * scale
    shares = np.zeros_like(total_variance)
    active = total_variance > tolerance
    shares[active] = contribution_variance[active] / total_variance[active]
    shares[np.abs(shares) < 1.0e-14] = 0.0
    return [float(value) for value in shares]


def _json_float(value: object, *, name: str) -> float:
    _require(
        type(value) in {int, float} and np.isfinite(value),
        f"{name} must be a finite JSON number",
    )
    return float(cast(float, value))


def _json_array(
    value: object,
    *,
    name: str,
    shape: tuple[int, ...],
) -> np.ndarray:
    array = _float_array(value, name=name)
    _require(array.shape == shape, f"{name} must have shape {shape}")
    return array


@dataclass(frozen=True)
class QueryVarianceDecompositionV1:
    """Exact Shapley attribution of a fixed finite-mixture query covariance."""

    query_id: str
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    query_scales: np.ndarray
    component_weights: np.ndarray
    component_query_means: np.ndarray
    factor_values: Mapping[str, Sequence[str]] = field(default_factory=dict)
    conditional_covariances: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    _posterior_mean: np.ndarray = field(init=False, repr=False)
    _between_covariance: np.ndarray = field(init=False, repr=False)
    _factor_covariances: Mapping[str, np.ndarray] = field(init=False, repr=False)
    _unresolved_covariance: np.ndarray = field(init=False, repr=False)
    _conditional_contributions: Mapping[str, np.ndarray] = field(
        init=False,
        repr=False,
    )
    _total_covariance: np.ndarray = field(init=False, repr=False)
    _max_abs_additivity_error: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        query_id = _nonempty_string(self.query_id, name="query_id")
        means = _float_array(
            self.component_query_means,
            name="component_query_means",
        )
        _require(
            means.ndim == 2 and min(means.shape) >= 1,
            "component_query_means must have shape (component, query)",
        )
        component_count, query_dimension = means.shape
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
        scales = _float_array(self.query_scales, name="query_scales")
        _require(
            scales.shape == (query_dimension,),
            f"query_scales must have shape ({query_dimension},)",
        )
        _require(np.all(scales > 0.0), "query_scales must be positive")
        weights = _normalized_weights(
            self.component_weights,
            component_count=component_count,
        )
        factors = _factor_mapping(
            self.factor_values,
            component_count=component_count,
        )
        conditional = _conditional_mapping(
            self.conditional_covariances,
            component_count=component_count,
            query_dimension=query_dimension,
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="metadata must be finite JSON data",
        )

        posterior_mean = readonly_array(weights @ means, dtype=np.float64)
        between = _weighted_covariance(weights, means, posterior_mean)
        factor_names = tuple(factors)
        explained: dict[int, np.ndarray] = {}
        for mask in range(1 << len(factor_names)):
            selected = tuple(
                factor_names[index]
                for index in range(len(factor_names))
                if mask & (1 << index)
            )
            explained[mask] = _explained_covariance(
                weights,
                means,
                posterior_mean,
                factors,
                selected,
            )

        factor_covariances: dict[str, np.ndarray] = {}
        factor_count = len(factor_names)
        if factor_count:
            denominator = math.factorial(factor_count)
            for index, name in enumerate(factor_names):
                contribution = np.zeros_like(between)
                for mask in range(1 << factor_count):
                    if mask & (1 << index):
                        continue
                    subset_size = int(mask.bit_count())
                    coefficient = (
                        math.factorial(subset_size)
                        * math.factorial(factor_count - subset_size - 1)
                        / denominator
                    )
                    contribution += coefficient * (
                        explained[mask | (1 << index)] - explained[mask]
                    )
                factor_covariances[name] = _symmetrize_psd(
                    contribution,
                    name=f"Shapley covariance for factor {name!r}",
                )
            fully_explained = explained[(1 << factor_count) - 1]
        else:
            fully_explained = np.zeros_like(between)

        unresolved = _symmetrize_psd(
            between - fully_explained,
            name="unresolved component covariance",
        )
        conditional_contributions = {
            name: _symmetrize_psd(
                np.einsum("c,cij->ij", weights, stack, optimize=True),
                name=f"posterior conditional covariance {name!r}",
            )
            for name, stack in conditional.items()
        }
        total = np.array(between, copy=True)
        for covariance in conditional_contributions.values():
            total += covariance
        total = _symmetrize_psd(total, name="total query covariance")

        reconstructed = np.array(unresolved, copy=True)
        for covariance in factor_covariances.values():
            reconstructed += covariance
        for covariance in conditional_contributions.values():
            reconstructed += covariance
        max_error = float(np.max(np.abs(total - reconstructed), initial=0.0))
        tolerance = 1.0e-9 * max(
            1.0,
            float(np.max(np.abs(total), initial=0.0)),
        )
        _require(
            max_error <= tolerance,
            "query variance decomposition failed numerical additivity",
        )

        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "query_labels", labels)
        object.__setattr__(self, "query_units", units)
        object.__setattr__(self, "query_scales", readonly_array(scales))
        object.__setattr__(self, "component_weights", readonly_array(weights))
        object.__setattr__(self, "component_query_means", readonly_array(means))
        object.__setattr__(self, "factor_values", factors)
        object.__setattr__(self, "conditional_covariances", conditional)
        object.__setattr__(self, "metadata", metadata)
        object.__setattr__(self, "_posterior_mean", posterior_mean)
        object.__setattr__(self, "_between_covariance", between)
        object.__setattr__(
            self,
            "_factor_covariances",
            MappingProxyType(factor_covariances),
        )
        object.__setattr__(self, "_unresolved_covariance", unresolved)
        object.__setattr__(
            self,
            "_conditional_contributions",
            MappingProxyType(conditional_contributions),
        )
        object.__setattr__(self, "_total_covariance", total)
        object.__setattr__(self, "_max_abs_additivity_error", max_error)

    def summary_arrays(self) -> Mapping[str, np.ndarray]:
        values: dict[str, np.ndarray] = {
            "posterior_mean": self._posterior_mean,
            "total_covariance": self._total_covariance,
            "between_component_covariance": self._between_covariance,
            "unresolved_component_covariance": self._unresolved_covariance,
        }
        values.update(
            {
                f"factor__{name}": covariance
                for name, covariance in self._factor_covariances.items()
            }
        )
        values.update(
            {
                f"conditional__{name}": covariance
                for name, covariance in self._conditional_contributions.items()
            }
        )
        return MappingProxyType(values)

    def _contributions(self) -> dict[str, np.ndarray]:
        values = {
            f"factor:{name}": covariance
            for name, covariance in self._factor_covariances.items()
        }
        values["unresolved_component"] = self._unresolved_covariance
        values.update(
            {
                f"conditional:{name}": covariance
                for name, covariance in self._conditional_contributions.items()
            }
        )
        return values

    def _payload_without_identity(self) -> dict[str, Any]:
        contributions = self._contributions()
        standardized_total_trace = _trace_in_registered_scale(
            self._total_covariance,
            self.query_scales,
        )
        trace_shares = {
            name: (
                _trace_in_registered_scale(covariance, self.query_scales)
                / standardized_total_trace
                if standardized_total_trace > 0.0
                else 0.0
            )
            for name, covariance in contributions.items()
        }
        variance_shares = {
            name: _safe_variance_share(covariance, self._total_covariance)
            for name, covariance in contributions.items()
        }
        return {
            "schema_version": QUERY_VARIANCE_DECOMPOSITION_SCHEMA_VERSION,
            "artifact_kind": QUERY_VARIANCE_DECOMPOSITION_ARTIFACT_KIND,
            "claim_boundary": QUERY_VARIANCE_DECOMPOSITION_CLAIM_BOUNDARY,
            "query": {
                "query_id": self.query_id,
                "labels": list(self.query_labels),
                "units": list(self.query_units),
                "scales": self.query_scales.tolist(),
            },
            "support": {
                "component_count": int(self.component_query_means.shape[0]),
                "effective_component_count": float(
                    1.0 / np.sum(self.component_weights**2)
                ),
                "factor_names": list(self.factor_values),
                "factor_values": {
                    name: list(values) for name, values in self.factor_values.items()
                },
                "conditional_covariance_sources": list(self.conditional_covariances),
            },
            "input_identities": {
                "query_scales_sha256": _array_sha256(self.query_scales),
                "component_weights_sha256": _array_sha256(self.component_weights),
                "component_query_means_sha256": _array_sha256(
                    self.component_query_means
                ),
                "conditional_covariance_sha256": {
                    name: _array_sha256(values)
                    for name, values in self.conditional_covariances.items()
                },
            },
            "posterior_mean": self._posterior_mean.tolist(),
            "covariance": {
                "total": self._total_covariance.tolist(),
                "between_components": self._between_covariance.tolist(),
                "factor_shapley": {
                    name: covariance.tolist()
                    for name, covariance in self._factor_covariances.items()
                },
                "unresolved_component": self._unresolved_covariance.tolist(),
                "conditional_sources": {
                    name: covariance.tolist()
                    for name, covariance in self._conditional_contributions.items()
                },
            },
            "variance_share_by_query_coordinate": variance_shares,
            "standardized_trace_share": {
                name: float(value) for name, value in trace_shares.items()
            },
            "diagnostics": {
                "max_abs_additivity_error": self._max_abs_additivity_error,
                "standardized_total_trace": standardized_total_trace,
                "factor_attribution": _FACTOR_ATTRIBUTION,
                "unresolved_role": _UNRESOLVED_ROLE,
                "conditional_source_semantics": _CONDITIONAL_SOURCE_SEMANTICS,
            },
            "metadata": plain_json(self.metadata),
        }

    @property
    def decomposition_id(self) -> str:
        return _canonical_sha256(
            self._payload_without_identity(),
            omitted="decomposition_id",
        )

    def as_dict(self) -> dict[str, Any]:
        payload = self._payload_without_identity()
        payload["decomposition_id"] = self.decomposition_id
        return payload


def _contribution_mapping(
    covariance: Mapping[str, Any],
    *,
    query_dimension: int,
) -> tuple[
    np.ndarray,
    np.ndarray,
    Mapping[str, np.ndarray],
    np.ndarray,
    Mapping[str, np.ndarray],
]:
    _require(
        set(covariance)
        == {
            "total",
            "between_components",
            "factor_shapley",
            "unresolved_component",
            "conditional_sources",
        },
        "covariance fields changed",
    )
    shape = (query_dimension, query_dimension)
    total = _symmetrize_psd(
        _json_array(covariance["total"], name="total covariance", shape=shape),
        name="total covariance",
    )
    between = _symmetrize_psd(
        _json_array(
            covariance["between_components"],
            name="between-component covariance",
            shape=shape,
        ),
        name="between-component covariance",
    )
    factor_raw = _mapping(
        covariance["factor_shapley"],
        name="factor Shapley covariance",
    )
    factor = MappingProxyType(
        {
            name: _symmetrize_psd(
                _json_array(
                    value,
                    name=f"factor covariance {name!r}",
                    shape=shape,
                ),
                name=f"factor covariance {name!r}",
            )
            for name, value in sorted(factor_raw.items())
        }
    )
    unresolved = _symmetrize_psd(
        _json_array(
            covariance["unresolved_component"],
            name="unresolved component covariance",
            shape=shape,
        ),
        name="unresolved component covariance",
    )
    conditional_raw = _mapping(
        covariance["conditional_sources"],
        name="conditional covariance sources",
    )
    conditional = MappingProxyType(
        {
            name: _symmetrize_psd(
                _json_array(
                    value,
                    name=f"conditional covariance {name!r}",
                    shape=shape,
                ),
                name=f"conditional covariance {name!r}",
            )
            for name, value in sorted(conditional_raw.items())
        }
    )
    return total, between, factor, unresolved, conditional


def validate_query_variance_decomposition(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a portable decomposition summary and its numerical additivity."""

    values = _mapping(payload, name="query variance decomposition")
    _require(set(values) == _TOP_LEVEL_FIELDS, "decomposition fields changed")
    _require(
        values.get("schema_version") == QUERY_VARIANCE_DECOMPOSITION_SCHEMA_VERSION,
        "unsupported query variance decomposition schema",
    )
    _require(
        values.get("artifact_kind") == QUERY_VARIANCE_DECOMPOSITION_ARTIFACT_KIND,
        "unexpected query variance decomposition artifact kind",
    )
    _require(
        values.get("claim_boundary") == QUERY_VARIANCE_DECOMPOSITION_CLAIM_BOUNDARY,
        "query variance decomposition claim boundary changed",
    )

    query = _mapping(values.get("query"), name="query")
    _require(
        set(query) == {"query_id", "labels", "units", "scales"},
        "query fields changed",
    )
    query_id = _nonempty_string(query.get("query_id"), name="query_id")
    labels = _string_tuple(query.get("labels"), name="query labels", unique=True)
    _require(labels, "query labels must be nonempty")
    dimension = len(labels)
    units = _string_tuple(
        query.get("units"),
        name="query units",
        expected_count=dimension,
    )
    scales = _json_array(
        query.get("scales"),
        name="query scales",
        shape=(dimension,),
    )
    _require(np.all(scales > 0.0), "query scales must be positive")

    support = _mapping(values.get("support"), name="support")
    _require(
        set(support)
        == {
            "component_count",
            "effective_component_count",
            "factor_names",
            "factor_values",
            "conditional_covariance_sources",
        },
        "support fields changed",
    )
    component_count_raw = support.get("component_count")
    _require(
        type(component_count_raw) is int and component_count_raw > 0,
        "component_count must be a positive integer",
    )
    component_count = cast(int, component_count_raw)
    effective_count = _json_float(
        support.get("effective_component_count"),
        name="effective_component_count",
    )
    _require(
        1.0 - 1.0e-12 <= effective_count <= component_count + 1.0e-12,
        "effective_component_count is outside the finite support",
    )
    factor_names = _string_tuple(
        support.get("factor_names"),
        name="factor names",
        unique=True,
    )
    _require(
        len(factor_names) <= MAX_EXACT_SHAPLEY_FACTORS,
        "too many factors for exact Shapley attribution",
    )
    factor_values_raw = _mapping(
        support.get("factor_values"),
        name="factor values",
    )
    _require(
        factor_names == tuple(sorted(factor_values_raw)),
        "factor names do not match the factor-value inventory",
    )
    factor_values = {
        name: list(
            _string_tuple(
                factor_values_raw[name],
                name=f"factor values {name!r}",
                expected_count=component_count,
            )
        )
        for name in factor_names
    }
    conditional_names = _string_tuple(
        support.get("conditional_covariance_sources"),
        name="conditional covariance sources",
        unique=True,
    )
    _require(
        conditional_names == tuple(sorted(conditional_names)),
        "conditional covariance sources must be sorted",
    )

    identities = _mapping(values.get("input_identities"), name="input identities")
    _require(
        set(identities)
        == {
            "query_scales_sha256",
            "component_weights_sha256",
            "component_query_means_sha256",
            "conditional_covariance_sha256",
        },
        "input identity fields changed",
    )
    conditional_hashes_raw = _mapping(
        identities.get("conditional_covariance_sha256"),
        name="conditional covariance identities",
    )
    _require(
        tuple(sorted(conditional_hashes_raw)) == conditional_names,
        "conditional covariance identity inventory changed",
    )
    input_identities = {
        "query_scales_sha256": _validated_sha256(
            identities.get("query_scales_sha256"),
            name="query scales SHA-256",
        ),
        "component_weights_sha256": _validated_sha256(
            identities.get("component_weights_sha256"),
            name="component weights SHA-256",
        ),
        "component_query_means_sha256": _validated_sha256(
            identities.get("component_query_means_sha256"),
            name="component query means SHA-256",
        ),
        "conditional_covariance_sha256": {
            name: _validated_sha256(
                conditional_hashes_raw[name],
                name=f"conditional covariance {name!r} SHA-256",
            )
            for name in conditional_names
        },
    }
    _require(
        input_identities["query_scales_sha256"] == _array_sha256(scales),
        "query scales identity changed",
    )

    posterior_mean = _json_array(
        values.get("posterior_mean"),
        name="posterior mean",
        shape=(dimension,),
    )
    covariance = _mapping(values.get("covariance"), name="covariance")
    total, between, factor, unresolved, conditional = _contribution_mapping(
        covariance,
        query_dimension=dimension,
    )
    _require(tuple(factor) == factor_names, "factor covariance inventory changed")
    _require(
        tuple(conditional) == conditional_names,
        "conditional covariance inventory changed",
    )

    reconstructed_between = np.array(unresolved, copy=True)
    for contribution in factor.values():
        reconstructed_between += contribution
    reconstructed_total = np.array(between, copy=True)
    for contribution in conditional.values():
        reconstructed_total += contribution
    scale = max(1.0, float(np.max(np.abs(total), initial=0.0)))
    tolerance = 1.0e-9 * scale
    between_error = float(np.max(np.abs(between - reconstructed_between), initial=0.0))
    total_error = float(np.max(np.abs(total - reconstructed_total), initial=0.0))
    _require(
        between_error <= tolerance,
        "factor and unresolved covariance do not reconstruct between covariance",
    )
    _require(
        total_error <= tolerance,
        "between and conditional covariance do not reconstruct total covariance",
    )

    contributions: dict[str, np.ndarray] = {
        **{f"factor:{name}": value for name, value in factor.items()},
        "unresolved_component": unresolved,
        **{f"conditional:{name}": value for name, value in conditional.items()},
    }
    expected_variance_shares = {
        name: _safe_variance_share(value, total)
        for name, value in contributions.items()
    }
    variance_shares_raw = _mapping(
        values.get("variance_share_by_query_coordinate"),
        name="coordinatewise variance shares",
    )
    _require(
        set(variance_shares_raw) == set(contributions),
        "coordinatewise variance share inventory changed",
    )
    normalized_variance_shares: dict[str, list[float]] = {}
    for name in sorted(contributions):
        share = _json_array(
            variance_shares_raw[name],
            name=f"coordinatewise variance share {name!r}",
            shape=(dimension,),
        )
        _require(
            np.allclose(
                share,
                expected_variance_shares[name],
                rtol=1.0e-12,
                atol=1.0e-12,
            ),
            f"coordinatewise variance share {name!r} changed",
        )
        normalized_variance_shares[name] = share.tolist()

    standardized_total_trace = _trace_in_registered_scale(total, scales)
    expected_trace_shares = {
        name: (
            _trace_in_registered_scale(value, scales) / standardized_total_trace
            if standardized_total_trace > 0.0
            else 0.0
        )
        for name, value in contributions.items()
    }
    trace_shares_raw = _mapping(
        values.get("standardized_trace_share"),
        name="standardized trace shares",
    )
    _require(
        set(trace_shares_raw) == set(contributions),
        "standardized trace share inventory changed",
    )
    normalized_trace_shares: dict[str, float] = {}
    for name in sorted(contributions):
        share = _json_float(
            trace_shares_raw[name],
            name=f"standardized trace share {name!r}",
        )
        _require(
            math.isclose(
                share,
                expected_trace_shares[name],
                rel_tol=1.0e-12,
                abs_tol=1.0e-12,
            ),
            f"standardized trace share {name!r} changed",
        )
        normalized_trace_shares[name] = share

    diagnostics = _mapping(values.get("diagnostics"), name="diagnostics")
    _require(
        set(diagnostics)
        == {
            "max_abs_additivity_error",
            "standardized_total_trace",
            "factor_attribution",
            "unresolved_role",
            "conditional_source_semantics",
        },
        "diagnostic fields changed",
    )
    reported_error = _json_float(
        diagnostics.get("max_abs_additivity_error"),
        name="max_abs_additivity_error",
    )
    expected_error = float(
        np.max(
            np.abs(
                total
                - sum(
                    (value for value in contributions.values()),
                    start=np.zeros_like(total),
                )
            ),
            initial=0.0,
        )
    )
    _require(
        math.isclose(
            reported_error,
            expected_error,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "reported additivity error changed",
    )
    reported_trace = _json_float(
        diagnostics.get("standardized_total_trace"),
        name="standardized_total_trace",
    )
    _require(
        math.isclose(
            reported_trace,
            standardized_total_trace,
            rel_tol=1.0e-12,
            abs_tol=1.0e-12,
        ),
        "reported standardized total trace changed",
    )
    _require(
        diagnostics.get("factor_attribution") == _FACTOR_ATTRIBUTION,
        "factor attribution semantics changed",
    )
    _require(
        diagnostics.get("unresolved_role") == _UNRESOLVED_ROLE,
        "unresolved component semantics changed",
    )
    _require(
        diagnostics.get("conditional_source_semantics")
        == _CONDITIONAL_SOURCE_SEMANTICS,
        "conditional covariance semantics changed",
    )
    metadata = plain_json(
        validated_json_mapping(
            _mapping(values.get("metadata"), name="metadata"),
            error_message="metadata must be finite JSON data",
        )
    )

    normalized: dict[str, Any] = {
        "schema_version": QUERY_VARIANCE_DECOMPOSITION_SCHEMA_VERSION,
        "artifact_kind": QUERY_VARIANCE_DECOMPOSITION_ARTIFACT_KIND,
        "decomposition_id": values.get("decomposition_id"),
        "claim_boundary": QUERY_VARIANCE_DECOMPOSITION_CLAIM_BOUNDARY,
        "query": {
            "query_id": query_id,
            "labels": list(labels),
            "units": list(units),
            "scales": scales.tolist(),
        },
        "support": {
            "component_count": component_count,
            "effective_component_count": effective_count,
            "factor_names": list(factor_names),
            "factor_values": factor_values,
            "conditional_covariance_sources": list(conditional_names),
        },
        "input_identities": input_identities,
        "posterior_mean": posterior_mean.tolist(),
        "covariance": {
            "total": total.tolist(),
            "between_components": between.tolist(),
            "factor_shapley": {name: value.tolist() for name, value in factor.items()},
            "unresolved_component": unresolved.tolist(),
            "conditional_sources": {
                name: value.tolist() for name, value in conditional.items()
            },
        },
        "variance_share_by_query_coordinate": normalized_variance_shares,
        "standardized_trace_share": normalized_trace_shares,
        "diagnostics": {
            "max_abs_additivity_error": reported_error,
            "standardized_total_trace": reported_trace,
            "factor_attribution": _FACTOR_ATTRIBUTION,
            "unresolved_role": _UNRESOLVED_ROLE,
            "conditional_source_semantics": _CONDITIONAL_SOURCE_SEMANTICS,
        },
        "metadata": metadata,
    }
    _require(
        values.get("decomposition_id")
        == _canonical_sha256(normalized, omitted="decomposition_id"),
        "query variance decomposition content identity changed",
    )
    return normalized


def build_query_variance_decomposition(
    component_weights: object,
    component_query_means: object,
    *,
    query_id: str,
    query_labels: Sequence[str],
    query_units: Sequence[str],
    query_scales: object,
    factor_values: Mapping[str, Sequence[str]] | None = None,
    conditional_covariances: Mapping[str, object] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> QueryVarianceDecompositionV1:
    """Build an immutable finite-mixture query variance decomposition."""

    return QueryVarianceDecompositionV1(
        query_id=query_id,
        query_labels=tuple(query_labels),
        query_units=tuple(query_units),
        query_scales=np.asarray(query_scales),
        component_weights=np.asarray(component_weights),
        component_query_means=np.asarray(component_query_means),
        factor_values={} if factor_values is None else factor_values,
        conditional_covariances=(
            {} if conditional_covariances is None else conditional_covariances
        ),
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "MAX_EXACT_SHAPLEY_FACTORS",
    "QUERY_VARIANCE_DECOMPOSITION_ARTIFACT_KIND",
    "QUERY_VARIANCE_DECOMPOSITION_CLAIM_BOUNDARY",
    "QUERY_VARIANCE_DECOMPOSITION_SCHEMA_VERSION",
    "QueryVarianceDecompositionV1",
    "build_query_variance_decomposition",
    "validate_query_variance_decomposition",
]
