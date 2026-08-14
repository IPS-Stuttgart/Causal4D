"""Source-frozen routing from prequential stability or exact fallback."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.prequential_abduction import PrequentialAbductionPathV1
from causal4d.prequential_query_stability import PrequentialQueryStabilityV1


PREQUENTIAL_STABILITY_CERTIFICATE_SCHEMA_VERSION = 1
PREQUENTIAL_STABILITY_CERTIFICATE_CLAIM_BOUNDARY = (
    "Future-protocol routing under source-frozen thresholds. It does not alter the "
    "registered 36-execution estimator, establish calibration, or authorize target-"
    "informed prefix or threshold selection."
)
_KIND = "Causal4DPrequentialStabilityCertificateV1"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise ValueError(f"{name} must be a real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _future_frames_read(metadata: Mapping[str, Any], *, name: str) -> int:
    value = metadata.get("future_frames_read", 0)
    if type(value) is not int or value != 0:
        raise ValueError(f"{name} must declare future_frames_read=0")
    return value


@dataclass(frozen=True)
class PrequentialStabilityRuleV1:
    """Source-frozen thresholds for chronological prefix admission."""

    threshold_source_id: str
    fallback_artifact_id: str
    maximum_previous_mean_shift_standardized_l2: float
    maximum_previous_gaussian_wasserstein_standardized: float
    minimum_previous_interval_overlap_fraction: float
    maximum_previous_posterior_kl: float
    maximum_previous_posterior_total_variation: float
    minimum_effective_sample_size: float
    required_consecutive_steps: int
    maximum_prefix_frame_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "threshold_source_id",
            _sha256(self.threshold_source_id, name="threshold_source_id"),
        )
        object.__setattr__(
            self,
            "fallback_artifact_id",
            _sha256(self.fallback_artifact_id, name="fallback_artifact_id"),
        )
        for name in (
            "maximum_previous_mean_shift_standardized_l2",
            "maximum_previous_gaussian_wasserstein_standardized",
            "maximum_previous_posterior_kl",
            "maximum_previous_posterior_total_variation",
            "minimum_effective_sample_size",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative(getattr(self, name), name=name),
            )
        overlap = _finite_nonnegative(
            self.minimum_previous_interval_overlap_fraction,
            name="minimum_previous_interval_overlap_fraction",
        )
        if overlap > 1.0:
            raise ValueError(
                "minimum_previous_interval_overlap_fraction must not exceed one"
            )
        object.__setattr__(
            self,
            "minimum_previous_interval_overlap_fraction",
            overlap,
        )
        object.__setattr__(
            self,
            "required_consecutive_steps",
            _positive_integer(
                self.required_consecutive_steps,
                name="required_consecutive_steps",
            ),
        )
        object.__setattr__(
            self,
            "maximum_prefix_frame_count",
            _positive_integer(
                self.maximum_prefix_frame_count,
                name="maximum_prefix_frame_count",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="stability-rule metadata must contain finite JSON data",
            ),
        )

    @property
    def artifact_id(self) -> str:
        payload = {
            "threshold_source_id": self.threshold_source_id,
            "fallback_artifact_id": self.fallback_artifact_id,
            "maximum_previous_mean_shift_standardized_l2": (
                self.maximum_previous_mean_shift_standardized_l2
            ),
            "maximum_previous_gaussian_wasserstein_standardized": (
                self.maximum_previous_gaussian_wasserstein_standardized
            ),
            "minimum_previous_interval_overlap_fraction": (
                self.minimum_previous_interval_overlap_fraction
            ),
            "maximum_previous_posterior_kl": self.maximum_previous_posterior_kl,
            "maximum_previous_posterior_total_variation": (
                self.maximum_previous_posterior_total_variation
            ),
            "minimum_effective_sample_size": self.minimum_effective_sample_size,
            "required_consecutive_steps": self.required_consecutive_steps,
            "maximum_prefix_frame_count": self.maximum_prefix_frame_count,
            "metadata": plain_json(self.metadata),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            **json.loads(json.dumps(plain_json(self.__dict__))),
            "artifact_id": self.artifact_id,
        }


@dataclass(frozen=True)
class PrequentialStabilityCertificateV1:
    """Deterministic earliest-prefix acceptance or exact-fallback decision."""

    source_prequential_path_id: str
    source_query_stability_id: str
    rule_id: str
    prefix_frame_counts: np.ndarray
    step_passes: np.ndarray
    accepted_step_index: int | None
    selected_posterior_id: str
    fallback_artifact_id: str
    decision: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_prequential_path_id",
            "source_query_stability_id",
            "rule_id",
            "selected_posterior_id",
            "fallback_artifact_id",
        ):
            object.__setattr__(self, name, _sha256(getattr(self, name), name=name))
        prefixes = readonly_integer_array(
            self.prefix_frame_counts,
            name="prefix_frame_counts",
        )
        passes = np.asarray(self.step_passes)
        if passes.dtype.kind != "b" or passes.shape != prefixes.shape:
            raise ValueError("step_passes must be a Boolean vector aligned to prefixes")
        passes = readonly_array(passes, dtype=bool)
        if self.accepted_step_index is not None and (
            type(self.accepted_step_index) is not int
            or not 0 <= self.accepted_step_index < len(prefixes)
        ):
            raise ValueError("accepted_step_index is invalid")
        if self.decision not in {
            "accept_stable_prefix",
            "exact_fallback_no_stable_prefix",
        }:
            raise ValueError("unsupported stability decision")
        stable = self.decision == "accept_stable_prefix"
        if stable != (self.accepted_step_index is not None):
            raise ValueError("decision and accepted_step_index disagree")
        if stable and self.selected_posterior_id == self.fallback_artifact_id:
            raise ValueError("accepted prefix cannot select the fallback artifact")
        if not stable and self.selected_posterior_id != self.fallback_artifact_id:
            raise ValueError(
                "fallback decision must select fallback_artifact_id exactly"
            )
        object.__setattr__(self, "prefix_frame_counts", prefixes)
        object.__setattr__(self, "step_passes", passes)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message=(
                    "stability-certificate metadata must contain finite JSON data"
                ),
            ),
        )

    @property
    def stable(self) -> bool:
        return self.accepted_step_index is not None

    @property
    def accepted_prefix_frame_count(self) -> int | None:
        if self.accepted_step_index is None:
            return None
        return int(self.prefix_frame_counts[self.accepted_step_index])

    @property
    def exact_fallback_required(self) -> bool:
        return not self.stable

    @property
    def artifact_id(self) -> str:
        payload = {
            "schema_version": PREQUENTIAL_STABILITY_CERTIFICATE_SCHEMA_VERSION,
            "artifact_kind": _KIND,
            "source_prequential_path_id": self.source_prequential_path_id,
            "source_query_stability_id": self.source_query_stability_id,
            "rule_id": self.rule_id,
            "accepted_step_index": self.accepted_step_index,
            "selected_posterior_id": self.selected_posterior_id,
            "fallback_artifact_id": self.fallback_artifact_id,
            "decision": self.decision,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PREQUENTIAL_STABILITY_CERTIFICATE_CLAIM_BOUNDARY,
            "arrays": {
                "prefix_frame_counts": array_sha256(self.prefix_frame_counts),
                "step_passes": array_sha256(self.step_passes),
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
        return {
            "schema_version": PREQUENTIAL_STABILITY_CERTIFICATE_SCHEMA_VERSION,
            "artifact_kind": _KIND,
            "artifact_id": self.artifact_id,
            "source_prequential_path_id": self.source_prequential_path_id,
            "source_query_stability_id": self.source_query_stability_id,
            "rule_id": self.rule_id,
            "prefix_frame_counts": self.prefix_frame_counts.tolist(),
            "step_passes": self.step_passes.tolist(),
            "accepted_step_index": self.accepted_step_index,
            "accepted_prefix_frame_count": self.accepted_prefix_frame_count,
            "selected_posterior_id": self.selected_posterior_id,
            "fallback_artifact_id": self.fallback_artifact_id,
            "decision": self.decision,
            "metadata": plain_json(self.metadata),
            "claim_boundary": PREQUENTIAL_STABILITY_CERTIFICATE_CLAIM_BOUNDARY,
        }


def build_prequential_stability_certificate(
    stability: PrequentialQueryStabilityV1,
    path: PrequentialAbductionPathV1,
    rule: PrequentialStabilityRuleV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PrequentialStabilityCertificateV1:
    """Select the earliest stable causal prefix, otherwise exact fallback."""

    if stability.source_prequential_path_id != path.artifact_id:
        raise ValueError("query stability does not match the supplied path")
    if not np.array_equal(stability.prefix_frame_counts, path.prefix_frame_counts):
        raise ValueError("query stability and path prefix inventories differ")
    if not np.array_equal(stability.posterior_weights, path.posterior_weights):
        raise ValueError("query stability and path posterior weights differ")
    _future_frames_read(path.metadata, name="prequential path")
    _future_frames_read(stability.metadata, name="query-stability source")
    if rule.fallback_artifact_id in path.factual_intervention_ids:
        raise ValueError(
            "fallback artifact must be distinct from every prefix posterior"
        )

    summaries = stability.summary_arrays()
    prefixes = np.asarray(path.prefix_frame_counts, dtype=np.int64)
    passes = np.zeros(len(prefixes), dtype=bool)
    for index in range(1, len(prefixes)):
        passes[index] = bool(
            prefixes[index] <= rule.maximum_prefix_frame_count
            and summaries["previous_mean_shift_standardized_l2"][index]
            <= rule.maximum_previous_mean_shift_standardized_l2
            and summaries["previous_gaussian_wasserstein_standardized"][index]
            <= rule.maximum_previous_gaussian_wasserstein_standardized
            and summaries["previous_interval_overlap_fraction"][index]
            >= rule.minimum_previous_interval_overlap_fraction
            and path.previous_step_kl[index] <= rule.maximum_previous_posterior_kl
            and path.previous_step_total_variation[index]
            <= rule.maximum_previous_posterior_total_variation
            and path.posterior_effective_sample_size[index]
            >= rule.minimum_effective_sample_size
        )

    accepted: int | None = None
    run = 0
    for index, passed in enumerate(passes):
        run = run + 1 if passed else 0
        if run >= rule.required_consecutive_steps:
            accepted = index
            break
    if accepted is None:
        selected = rule.fallback_artifact_id
        decision = "exact_fallback_no_stable_prefix"
    else:
        selected = path.factual_intervention_ids[accepted]
        decision = "accept_stable_prefix"
    user_metadata = dict(metadata or {})
    user_metadata.update(
        {
            "future_frames_read": 0,
            "source_thresholds_frozen": True,
            "target_outcomes_used_for_thresholds": False,
        }
    )
    return PrequentialStabilityCertificateV1(
        source_prequential_path_id=path.artifact_id,
        source_query_stability_id=stability.artifact_id,
        rule_id=rule.artifact_id,
        prefix_frame_counts=prefixes,
        step_passes=passes,
        accepted_step_index=accepted,
        selected_posterior_id=selected,
        fallback_artifact_id=rule.fallback_artifact_id,
        decision=decision,
        metadata=user_metadata,
    )


__all__ = [
    "PREQUENTIAL_STABILITY_CERTIFICATE_CLAIM_BOUNDARY",
    "PREQUENTIAL_STABILITY_CERTIFICATE_SCHEMA_VERSION",
    "PrequentialStabilityCertificateV1",
    "PrequentialStabilityRuleV1",
    "build_prequential_stability_certificate",
]
