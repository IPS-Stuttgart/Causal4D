"""Query-specific adequacy certificates for truncated finite support."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

import numpy as np
from scipy.special import expit, logsumexp

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping


FINITE_SUPPORT_ADEQUACY_SCHEMA_VERSION = 1
_SUPPORT_ADEQUACY_ARTIFACT_KIND = "Causal4DFiniteSupportAdequacyV1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CLAIM_BOUNDARY = {
    "analysis_only": True,
    "changes_estimator": False,
    "changes_source_posterior": False,
    "changes_registered_protocol": False,
    "uses_target_truth": False,
    "support_completeness_claimed_without_bound": False,
}
_ARRAY_FIELDS = frozenset(
    {
        "retained_prior_weights",
        "retained_log_likelihoods",
        "retained_posterior_weights",
        "retained_query_values",
        "omitted_query_lower",
        "omitted_query_upper",
        "retained_query_mean",
        "full_query_mean_lower",
        "full_query_mean_upper",
        "query_mean_max_abs_shift",
    }
)
_ARRAY_DTYPES = {name: np.dtype(np.float64) for name in _ARRAY_FIELDS}
_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "support_artifact_id",
        "evidence_id",
        "query_id",
        "support_name",
        "component_ids",
        "query_labels",
        "query_units",
        "retained_prior_mass",
        "omitted_log_likelihood_upper_bound",
        "omitted_posterior_mass_upper_bound",
        "minimum_retained_prior_mass",
        "maximum_omitted_posterior_mass",
        "admissible",
        "failure_reasons",
        "fallback_artifact_id",
        "metadata",
    }
)


def _require_sha256(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


def _optional_sha256(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_sha256(value, name=name)


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _validated_string_tuple(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
    unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must not be empty")
    if any(type(value) is not str or not value for value in result):
        raise ValueError(f"{name} entries must be nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} entries must be unique")
    return result


def _probability(value: Any, *, name: str, strictly_positive: bool = False) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be a finite probability")
    result = float(value)
    lower_ok = result > 0.0 if strictly_positive else result >= 0.0
    if not lower_ok or result > 1.0:
        interval = "(0, 1]" if strictly_positive else "[0, 1]"
        raise ValueError(f"{name} must lie in {interval}")
    return result


def _normalized_weights(values: Any, *, name: str) -> np.ndarray:
    weights = readonly_array(values, dtype=float)
    if (
        weights.ndim != 1
        or len(weights) == 0
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise ValueError(f"{name} must be a finite nonnegative vector")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must sum to one")
    return weights


def _canonicalize_log_likelihoods(
    retained_log_likelihoods: Any,
    *,
    retained_prior_mass: float,
    omitted_log_likelihood_upper_bound: float | None,
) -> tuple[np.ndarray, float | None]:
    retained = np.asarray(retained_log_likelihoods, dtype=float)
    if retained.ndim != 1 or len(retained) == 0 or not np.all(np.isfinite(retained)):
        raise ValueError("retained_log_likelihoods must be a finite nonempty vector")
    omitted_mass = 1.0 - retained_prior_mass
    if omitted_mass > 0.0:
        if (
            omitted_log_likelihood_upper_bound is None
            or isinstance(
                omitted_log_likelihood_upper_bound,
                (bool, np.bool_),
            )
            or not isinstance(
                omitted_log_likelihood_upper_bound,
                (int, float, np.integer, np.floating),
            )
            or not np.isfinite(omitted_log_likelihood_upper_bound)
        ):
            raise ValueError(
                "omitted_log_likelihood_upper_bound must be finite when "
                "retained_prior_mass is below one"
            )
        omitted = float(omitted_log_likelihood_upper_bound)
        shift = max(float(np.max(retained)), omitted)
        canonical_retained = retained - shift
        canonical_omitted: float | None = omitted - shift
    else:
        if omitted_log_likelihood_upper_bound is not None:
            raise ValueError(
                "omitted_log_likelihood_upper_bound must be None when all "
                "prior support is retained"
            )
        shift = float(np.max(retained))
        canonical_retained = retained - shift
        canonical_omitted = None
    return readonly_array(canonical_retained, dtype=float), canonical_omitted


def _retained_posterior_weights(
    prior_weights: np.ndarray,
    log_likelihoods: np.ndarray,
) -> np.ndarray:
    log_joint = (
        np.log(
            prior_weights,
            where=prior_weights > 0.0,
            out=np.full_like(prior_weights, -np.inf),
        )
        + log_likelihoods
    )
    normalizer = float(logsumexp(log_joint))
    if not np.isfinite(normalizer):
        raise ValueError("retained support has zero or non-finite evidence")
    return readonly_array(np.exp(log_joint - normalizer), dtype=float)


def _omitted_posterior_mass_bound(
    prior_weights: np.ndarray,
    log_likelihoods: np.ndarray,
    *,
    retained_prior_mass: float,
    omitted_log_likelihood_upper_bound: float | None,
) -> float:
    if retained_prior_mass == 1.0:
        return 0.0
    if omitted_log_likelihood_upper_bound is None:
        raise RuntimeError("validated omitted likelihood bound became absent")
    log_prior_weights = np.log(
        prior_weights,
        where=prior_weights > 0.0,
        out=np.full_like(prior_weights, -np.inf),
    )
    log_retained_evidence = np.log(retained_prior_mass) + float(
        logsumexp(log_prior_weights + log_likelihoods)
    )
    omitted_prior_mass = 1.0 - retained_prior_mass
    log_omitted_evidence_upper = (
        np.log(omitted_prior_mass) + omitted_log_likelihood_upper_bound
    )
    result = float(expit(log_omitted_evidence_upper - log_retained_evidence))
    return min(1.0, max(0.0, result))


def _query_envelope(
    retained_query_values: np.ndarray,
    retained_posterior_weights: np.ndarray,
    omitted_query_lower: np.ndarray,
    omitted_query_upper: np.ndarray,
    omitted_posterior_mass_upper_bound: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    retained_mean = retained_posterior_weights @ retained_query_values
    mass = omitted_posterior_mass_upper_bound
    extreme_lower = (1.0 - mass) * retained_mean + mass * omitted_query_lower
    extreme_upper = (1.0 - mass) * retained_mean + mass * omitted_query_upper
    lower = np.minimum(retained_mean, extreme_lower)
    upper = np.maximum(retained_mean, extreme_upper)
    maximum_shift = np.maximum(retained_mean - lower, upper - retained_mean)
    return (
        readonly_array(retained_mean, dtype=float),
        readonly_array(lower, dtype=float),
        readonly_array(upper, dtype=float),
        readonly_array(maximum_shift, dtype=float),
    )


def _expected_failure_reasons(
    *,
    retained_prior_mass: float,
    omitted_posterior_mass_upper_bound: float,
    minimum_retained_prior_mass: float,
    maximum_omitted_posterior_mass: float,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if retained_prior_mass < minimum_retained_prior_mass:
        reasons.append("retained_prior_mass_below_threshold")
    if omitted_posterior_mass_upper_bound > maximum_omitted_posterior_mass:
        reasons.append("omitted_posterior_mass_bound_exceeds_threshold")
    return tuple(reasons)


@dataclass(frozen=True)
class FiniteSupportAdequacyCertificateV1:
    """Bound omitted posterior mass and registered-query mean sensitivity."""

    support_artifact_id: str
    evidence_id: str
    query_id: str
    support_name: str
    component_ids: tuple[str, ...]
    query_labels: tuple[str, ...]
    query_units: tuple[str, ...]
    retained_prior_mass: float
    omitted_log_likelihood_upper_bound: float | None
    omitted_posterior_mass_upper_bound: float
    minimum_retained_prior_mass: float
    maximum_omitted_posterior_mass: float
    admissible: bool
    failure_reasons: tuple[str, ...]
    fallback_artifact_id: str | None
    retained_prior_weights: np.ndarray
    retained_log_likelihoods: np.ndarray
    retained_posterior_weights: np.ndarray
    retained_query_values: np.ndarray
    omitted_query_lower: np.ndarray
    omitted_query_upper: np.ndarray
    retained_query_mean: np.ndarray
    full_query_mean_lower: np.ndarray
    full_query_mean_upper: np.ndarray
    query_mean_max_abs_shift: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        support_artifact_id = _require_sha256(
            self.support_artifact_id,
            name="support_artifact_id",
        )
        evidence_id = _require_sha256(self.evidence_id, name="evidence_id")
        query_id = _require_sha256(self.query_id, name="query_id")
        support_name = _require_nonempty_string(
            self.support_name,
            name="support_name",
        )
        component_ids = _validated_string_tuple(
            self.component_ids,
            name="component_ids",
        )
        query_labels = _validated_string_tuple(
            self.query_labels,
            name="query_labels",
        )
        query_units = _validated_string_tuple(
            self.query_units,
            name="query_units",
            allow_empty=False,
            unique=False,
        )
        if len(query_units) != len(query_labels):
            raise ValueError("query_labels and query_units must align")
        retained_prior_mass = _probability(
            self.retained_prior_mass,
            name="retained_prior_mass",
            strictly_positive=True,
        )
        omitted_log_upper = self.omitted_log_likelihood_upper_bound
        if omitted_log_upper is not None:
            if (
                isinstance(omitted_log_upper, (bool, np.bool_))
                or not isinstance(
                    omitted_log_upper,
                    (int, float, np.integer, np.floating),
                )
                or not np.isfinite(omitted_log_upper)
            ):
                raise ValueError(
                    "omitted_log_likelihood_upper_bound must be finite or None"
                )
            omitted_log_upper = float(omitted_log_upper)
        omitted_posterior_mass_upper = _probability(
            self.omitted_posterior_mass_upper_bound,
            name="omitted_posterior_mass_upper_bound",
        )
        minimum_retained = _probability(
            self.minimum_retained_prior_mass,
            name="minimum_retained_prior_mass",
        )
        maximum_omitted = _probability(
            self.maximum_omitted_posterior_mass,
            name="maximum_omitted_posterior_mass",
        )
        if type(self.admissible) is not bool:
            raise ValueError("admissible must be Boolean")
        failure_reasons = _validated_string_tuple(
            self.failure_reasons,
            name="failure_reasons",
            allow_empty=True,
        )
        fallback_artifact_id = _optional_sha256(
            self.fallback_artifact_id,
            name="fallback_artifact_id",
        )

        prior_weights = _normalized_weights(
            self.retained_prior_weights,
            name="retained_prior_weights",
        )
        log_likelihoods = readonly_array(
            self.retained_log_likelihoods,
            dtype=float,
        )
        posterior_weights = _normalized_weights(
            self.retained_posterior_weights,
            name="retained_posterior_weights",
        )
        component_count = len(component_ids)
        if (
            prior_weights.shape != (component_count,)
            or log_likelihoods.shape != (component_count,)
            or posterior_weights.shape != (component_count,)
        ):
            raise ValueError("retained support arrays must align with component_ids")
        if not np.all(np.isfinite(log_likelihoods)):
            raise ValueError("retained_log_likelihoods must be finite")
        query_values = readonly_array(self.retained_query_values, dtype=float)
        query_count = len(query_labels)
        if query_values.shape != (component_count, query_count):
            raise ValueError(
                "retained_query_values must have shape (component, query_output)"
            )
        if not np.all(np.isfinite(query_values)):
            raise ValueError("retained_query_values must be finite")
        omitted_lower = readonly_array(self.omitted_query_lower, dtype=float)
        omitted_upper = readonly_array(self.omitted_query_upper, dtype=float)
        expected_query_shape = (query_count,)
        if (
            omitted_lower.shape != expected_query_shape
            or omitted_upper.shape != expected_query_shape
            or not np.all(np.isfinite(omitted_lower))
            or not np.all(np.isfinite(omitted_upper))
            or np.any(omitted_lower > omitted_upper)
        ):
            raise ValueError(
                "omitted query bounds must be finite aligned vectors with "
                "lower <= upper"
            )

        if retained_prior_mass == 1.0:
            if omitted_log_upper is not None:
                raise ValueError(
                    "omitted likelihood bound must be absent for complete support"
                )
            if not np.isclose(
                np.max(log_likelihoods),
                0.0,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "complete-support log likelihoods must be canonically shifted"
                )
        else:
            if omitted_log_upper is None:
                raise ValueError(
                    "omitted likelihood bound is required for truncated support"
                )
            maximum_log_value = max(
                float(np.max(log_likelihoods)),
                omitted_log_upper,
            )
            if not np.isclose(
                maximum_log_value,
                0.0,
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    "retained and omitted log likelihoods must share a "
                    "canonical zero reference"
                )

        expected_posterior = _retained_posterior_weights(
            prior_weights,
            log_likelihoods,
        )
        if not np.allclose(
            posterior_weights,
            expected_posterior,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "retained_posterior_weights disagree with prior and likelihood"
            )
        expected_omitted_mass = _omitted_posterior_mass_bound(
            prior_weights,
            log_likelihoods,
            retained_prior_mass=retained_prior_mass,
            omitted_log_likelihood_upper_bound=omitted_log_upper,
        )
        if not np.isclose(
            omitted_posterior_mass_upper,
            expected_omitted_mass,
            atol=1e-12,
            rtol=1e-10,
        ):
            raise ValueError(
                "omitted_posterior_mass_upper_bound disagrees with its evidence"
            )

        expected_mean, expected_lower, expected_upper, expected_shift = _query_envelope(
            query_values,
            posterior_weights,
            omitted_lower,
            omitted_upper,
            omitted_posterior_mass_upper,
        )
        supplied_query_arrays = (
            ("retained_query_mean", self.retained_query_mean, expected_mean),
            (
                "full_query_mean_lower",
                self.full_query_mean_lower,
                expected_lower,
            ),
            (
                "full_query_mean_upper",
                self.full_query_mean_upper,
                expected_upper,
            ),
            (
                "query_mean_max_abs_shift",
                self.query_mean_max_abs_shift,
                expected_shift,
            ),
        )
        validated_query_arrays: dict[str, np.ndarray] = {}
        for name, supplied, expected in supplied_query_arrays:
            values = readonly_array(supplied, dtype=float)
            if (
                values.shape != expected_query_shape
                or not np.all(np.isfinite(values))
                or not np.allclose(values, expected, atol=1e-12, rtol=1e-10)
            ):
                raise ValueError(f"{name} disagrees with the support bound")
            validated_query_arrays[name] = values

        expected_reasons = _expected_failure_reasons(
            retained_prior_mass=retained_prior_mass,
            omitted_posterior_mass_upper_bound=omitted_posterior_mass_upper,
            minimum_retained_prior_mass=minimum_retained,
            maximum_omitted_posterior_mass=maximum_omitted,
        )
        if failure_reasons != expected_reasons:
            raise ValueError("failure_reasons disagree with the admission policy")
        if self.admissible != (not expected_reasons):
            raise ValueError("admissible disagrees with the admission policy")

        object.__setattr__(
            self,
            "support_artifact_id",
            support_artifact_id,
        )
        object.__setattr__(self, "evidence_id", evidence_id)
        object.__setattr__(self, "query_id", query_id)
        object.__setattr__(self, "support_name", support_name)
        object.__setattr__(self, "component_ids", component_ids)
        object.__setattr__(self, "query_labels", query_labels)
        object.__setattr__(self, "query_units", query_units)
        object.__setattr__(self, "retained_prior_mass", retained_prior_mass)
        object.__setattr__(
            self,
            "omitted_log_likelihood_upper_bound",
            omitted_log_upper,
        )
        object.__setattr__(
            self,
            "omitted_posterior_mass_upper_bound",
            omitted_posterior_mass_upper,
        )
        object.__setattr__(
            self,
            "minimum_retained_prior_mass",
            minimum_retained,
        )
        object.__setattr__(
            self,
            "maximum_omitted_posterior_mass",
            maximum_omitted,
        )
        object.__setattr__(self, "failure_reasons", failure_reasons)
        object.__setattr__(
            self,
            "fallback_artifact_id",
            fallback_artifact_id,
        )
        object.__setattr__(self, "retained_prior_weights", prior_weights)
        object.__setattr__(self, "retained_log_likelihoods", log_likelihoods)
        object.__setattr__(
            self,
            "retained_posterior_weights",
            posterior_weights,
        )
        object.__setattr__(self, "retained_query_values", query_values)
        object.__setattr__(self, "omitted_query_lower", omitted_lower)
        object.__setattr__(self, "omitted_query_upper", omitted_upper)
        for name, values in validated_query_arrays.items():
            object.__setattr__(self, name, values)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message=(
                    "support adequacy metadata must contain finite JSON data"
                ),
            ),
        )

    def _scalar_payload(self) -> dict[str, Any]:
        return {
            "schema_version": FINITE_SUPPORT_ADEQUACY_SCHEMA_VERSION,
            "artifact_kind": _SUPPORT_ADEQUACY_ARTIFACT_KIND,
            "support_artifact_id": self.support_artifact_id,
            "evidence_id": self.evidence_id,
            "query_id": self.query_id,
            "support_name": self.support_name,
            "component_ids": list(self.component_ids),
            "query_labels": list(self.query_labels),
            "query_units": list(self.query_units),
            "retained_prior_mass": self.retained_prior_mass,
            "omitted_log_likelihood_upper_bound": (
                self.omitted_log_likelihood_upper_bound
            ),
            "omitted_posterior_mass_upper_bound": (
                self.omitted_posterior_mass_upper_bound
            ),
            "minimum_retained_prior_mass": self.minimum_retained_prior_mass,
            "maximum_omitted_posterior_mass": (self.maximum_omitted_posterior_mass),
            "admissible": self.admissible,
            "failure_reasons": list(self.failure_reasons),
            "fallback_artifact_id": self.fallback_artifact_id,
            "metadata": plain_json(self.metadata),
        }

    def _array_payload(self) -> dict[str, np.ndarray]:
        return {
            "retained_prior_weights": self.retained_prior_weights,
            "retained_log_likelihoods": self.retained_log_likelihoods,
            "retained_posterior_weights": self.retained_posterior_weights,
            "retained_query_values": self.retained_query_values,
            "omitted_query_lower": self.omitted_query_lower,
            "omitted_query_upper": self.omitted_query_upper,
            "retained_query_mean": self.retained_query_mean,
            "full_query_mean_lower": self.full_query_mean_lower,
            "full_query_mean_upper": self.full_query_mean_upper,
            "query_mean_max_abs_shift": self.query_mean_max_abs_shift,
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
            **{name: values.tolist() for name, values in self._array_payload().items()},
        }


def build_finite_support_adequacy_certificate(
    *,
    support_artifact_id: str,
    evidence_id: str,
    query_id: str,
    support_name: str,
    component_ids: Sequence[str],
    query_labels: Sequence[str],
    query_units: Sequence[str],
    retained_prior_mass: float,
    retained_prior_weights: Any,
    retained_log_likelihoods: Any,
    omitted_log_likelihood_upper_bound: float | None,
    retained_query_values: Any,
    omitted_query_lower: Any,
    omitted_query_upper: Any,
    minimum_retained_prior_mass: float = 0.0,
    maximum_omitted_posterior_mass: float = 1.0,
    fallback_artifact_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FiniteSupportAdequacyCertificateV1:
    """Create a scale-invariant certificate for one finite query support."""

    retained_mass = _probability(
        retained_prior_mass,
        name="retained_prior_mass",
        strictly_positive=True,
    )
    prior_weights = _normalized_weights(
        retained_prior_weights,
        name="retained_prior_weights",
    )
    canonical_logs, canonical_omitted_upper = _canonicalize_log_likelihoods(
        retained_log_likelihoods,
        retained_prior_mass=retained_mass,
        omitted_log_likelihood_upper_bound=(omitted_log_likelihood_upper_bound),
    )
    if len(prior_weights) != len(canonical_logs):
        raise ValueError(
            "retained_prior_weights and retained_log_likelihoods must align"
        )
    posterior_weights = _retained_posterior_weights(
        prior_weights,
        canonical_logs,
    )
    omitted_mass_upper = _omitted_posterior_mass_bound(
        prior_weights,
        canonical_logs,
        retained_prior_mass=retained_mass,
        omitted_log_likelihood_upper_bound=canonical_omitted_upper,
    )
    query_values = np.asarray(retained_query_values, dtype=float)
    if query_values.ndim != 2 or query_values.shape[0] != len(prior_weights):
        raise ValueError(
            "retained_query_values must have shape (retained_component, query_output)"
        )
    omitted_lower = np.asarray(omitted_query_lower, dtype=float)
    omitted_upper = np.asarray(omitted_query_upper, dtype=float)
    retained_mean, full_lower, full_upper, maximum_shift = _query_envelope(
        query_values,
        posterior_weights,
        omitted_lower,
        omitted_upper,
        omitted_mass_upper,
    )
    minimum_retained = _probability(
        minimum_retained_prior_mass,
        name="minimum_retained_prior_mass",
    )
    maximum_omitted = _probability(
        maximum_omitted_posterior_mass,
        name="maximum_omitted_posterior_mass",
    )
    reasons = _expected_failure_reasons(
        retained_prior_mass=retained_mass,
        omitted_posterior_mass_upper_bound=omitted_mass_upper,
        minimum_retained_prior_mass=minimum_retained,
        maximum_omitted_posterior_mass=maximum_omitted,
    )
    user_metadata: Mapping[str, Any]
    if metadata is None:
        user_metadata = {}
    elif isinstance(metadata, Mapping):
        user_metadata = metadata
    else:
        raise ValueError("metadata must be a mapping")
    result_metadata = {
        "claim_boundary": _CLAIM_BOUNDARY,
        "likelihood_scale_canonicalized": True,
        "likelihood_bound_semantics": (
            "all retained likelihoods and the omitted upper bound share one "
            "arbitrary multiplicative scale"
        ),
        "query_bound_semantics": (
            "coordinatewise posterior-mean envelope over omitted mass up to "
            "the certified bound and omitted query values inside supplied bounds"
        ),
        "query_coordinates_are_not_independent_statistical_units": True,
        "user": plain_json(user_metadata),
    }
    return FiniteSupportAdequacyCertificateV1(
        support_artifact_id=support_artifact_id,
        evidence_id=evidence_id,
        query_id=query_id,
        support_name=support_name,
        component_ids=tuple(component_ids),
        query_labels=tuple(query_labels),
        query_units=tuple(query_units),
        retained_prior_mass=retained_mass,
        omitted_log_likelihood_upper_bound=canonical_omitted_upper,
        omitted_posterior_mass_upper_bound=omitted_mass_upper,
        minimum_retained_prior_mass=minimum_retained,
        maximum_omitted_posterior_mass=maximum_omitted,
        admissible=not reasons,
        failure_reasons=reasons,
        fallback_artifact_id=fallback_artifact_id,
        retained_prior_weights=prior_weights,
        retained_log_likelihoods=canonical_logs,
        retained_posterior_weights=posterior_weights,
        retained_query_values=query_values,
        omitted_query_lower=omitted_lower,
        omitted_query_upper=omitted_upper,
        retained_query_mean=retained_mean,
        full_query_mean_lower=full_lower,
        full_query_mean_upper=full_upper,
        query_mean_max_abs_shift=maximum_shift,
        metadata=result_metadata,
    )
