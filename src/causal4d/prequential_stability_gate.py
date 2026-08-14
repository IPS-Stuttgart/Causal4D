"""Source-frozen stability admission for leakage-safe prequential abduction.

The existing prequential artifacts are diagnostics and deliberately do not
select a prefix.  This module adds a separate, content-addressed decision layer
for a future or separately registered protocol.  It consumes only comparisons
to the immediately preceding causal prefix; final-path diagnostics and held-out
future observations are never used.

A rejected decision returns an explicitly supplied fallback
:class:`~causal4d.contracts.FactualIntervention` by exact object identity.
Nothing in this module changes the frozen 36-execution estimator.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np

from causal4d.artifact_io import (
    load_strict_json_object,
    read_regular_file_no_symlinks,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.contracts import FactualIntervention, array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.prequential_abduction import (
    PrequentialAbductionPathV1,
    PrequentialAbductionResult,
)
from causal4d.prequential_query_stability import PrequentialQueryStabilityV1


PREQUENTIAL_STABILITY_GATE_CONFIG_SCHEMA_VERSION = 1
PREQUENTIAL_STABILITY_DECISION_SCHEMA_VERSION = 1
PREQUENTIAL_STABILITY_CRITERIA = (
    "has_previous_prefix",
    "minimum_prefix_frame_count",
    "posterior_total_variation",
    "posterior_kl",
    "posterior_effective_sample_size",
    "query_mean_shift",
    "query_wasserstein",
    "query_interval_overlap",
)
_CONFIG_KIND = "Causal4DPrequentialStabilityGateConfigV1"
_DECISION_KIND = "Causal4DPrequentialStabilityDecisionV1"
DecisionStatus = Literal["accepted", "fallback"]
_CLAIM_BOUNDARY = {
    "future_protocol_only": True,
    "changes_frozen_estimator": False,
    "uses_target_truth": False,
    "uses_final_path_as_selector": False,
    "requires_source_frozen_thresholds": True,
    "fallback_is_exact_caller_supplied_object": True,
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            plain_json(payload),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    actual = set(value)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a mapping with string keys")
    return value


def _require_integer(value: Any, *, name: str, minimum: int) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _require_threshold(value: Any, *, name: str, minimum: float = 0.0) -> float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    result = float(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _require_probability(value: Any, *, name: str) -> float:
    result = _require_threshold(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return result


def _validated_sha_tuple(
    values: Any,
    *,
    name: str,
    unique: bool = True,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of SHA-256 digests")
    result = tuple(
        _require_sha256(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must contain at least one SHA-256 digest")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique SHA-256 digests")
    return result


@dataclass(frozen=True)
class PrequentialStabilityGateConfigV1:
    """Source-frozen thresholds for prospective prefix admission."""

    minimum_prefix_frame_count: int
    required_consecutive_passes: int
    maximum_previous_total_variation: float
    maximum_previous_kl: float
    minimum_effective_sample_size: float
    maximum_query_mean_shift_standardized_l2: float
    maximum_query_wasserstein_standardized: float
    minimum_query_interval_overlap_fraction: float
    source_artifact_ids: tuple[str, ...]
    source_only: bool
    registered_before_target_access: bool
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_prefix_frame_count",
            _require_integer(
                self.minimum_prefix_frame_count,
                name="minimum_prefix_frame_count",
                minimum=2,
            ),
        )
        object.__setattr__(
            self,
            "required_consecutive_passes",
            _require_integer(
                self.required_consecutive_passes,
                name="required_consecutive_passes",
                minimum=1,
            ),
        )
        for name in (
            "maximum_previous_total_variation",
            "maximum_previous_kl",
            "minimum_effective_sample_size",
            "maximum_query_mean_shift_standardized_l2",
            "maximum_query_wasserstein_standardized",
        ):
            minimum = 1.0 if name == "minimum_effective_sample_size" else 0.0
            object.__setattr__(
                self,
                name,
                _require_threshold(getattr(self, name), name=name, minimum=minimum),
            )
        object.__setattr__(
            self,
            "minimum_query_interval_overlap_fraction",
            _require_probability(
                self.minimum_query_interval_overlap_fraction,
                name="minimum_query_interval_overlap_fraction",
            ),
        )
        object.__setattr__(
            self,
            "source_artifact_ids",
            _validated_sha_tuple(
                self.source_artifact_ids,
                name="source_artifact_ids",
            ),
        )
        for name in ("source_only", "registered_before_target_access"):
            value = getattr(self, name)
            if type(value) is not bool or not value:
                raise ValueError(f"{name} must be explicitly true")
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="stability-gate metadata must contain finite JSON data",
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PREQUENTIAL_STABILITY_GATE_CONFIG_SCHEMA_VERSION,
            "artifact_kind": _CONFIG_KIND,
            "minimum_prefix_frame_count": self.minimum_prefix_frame_count,
            "required_consecutive_passes": self.required_consecutive_passes,
            "maximum_previous_total_variation": (self.maximum_previous_total_variation),
            "maximum_previous_kl": self.maximum_previous_kl,
            "minimum_effective_sample_size": self.minimum_effective_sample_size,
            "maximum_query_mean_shift_standardized_l2": (
                self.maximum_query_mean_shift_standardized_l2
            ),
            "maximum_query_wasserstein_standardized": (
                self.maximum_query_wasserstein_standardized
            ),
            "minimum_query_interval_overlap_fraction": (
                self.minimum_query_interval_overlap_fraction
            ),
            "source_artifact_ids": list(self.source_artifact_ids),
            "source_only": self.source_only,
            "registered_before_target_access": self.registered_before_target_access,
            "metadata": plain_json(self.metadata),
            "claim_boundary": _CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "artifact_id": self.artifact_id}


@dataclass(frozen=True)
class PrequentialStabilityDecisionV1:
    """First prospectively stable prefix, or an exact-fallback decision."""

    source_prequential_path_id: str
    source_query_stability_id: str
    config_id: str
    status: DecisionStatus
    required_consecutive_passes: int
    prefix_frame_counts: np.ndarray
    evidence_frame_stops: np.ndarray
    factual_intervention_ids: tuple[str, ...]
    criterion_pass: np.ndarray
    step_pass: np.ndarray
    consecutive_pass_counts: np.ndarray
    selected_step_index: int | None
    selected_prefix_frame_count: int | None
    selected_evidence_frame_stop: int | None
    selected_factual_intervention_id: str | None
    fallback_reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "source_prequential_path_id",
            "source_query_stability_id",
            "config_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        status = _require_nonempty_string(self.status, name="status")
        if status not in {"accepted", "fallback"}:
            raise ValueError("status must be 'accepted' or 'fallback'")
        required_passes = _require_integer(
            self.required_consecutive_passes,
            name="required_consecutive_passes",
            minimum=1,
        )
        prefixes = readonly_integer_array(
            self.prefix_frame_counts,
            name="prefix_frame_counts",
        )
        if prefixes.ndim != 1 or not len(prefixes) or np.any(np.diff(prefixes) <= 0):
            raise ValueError("prefix_frame_counts must be a nonempty increasing vector")
        evidence_stops = readonly_integer_array(
            self.evidence_frame_stops,
            name="evidence_frame_stops",
        )
        if (
            evidence_stops.shape != prefixes.shape
            or np.any(evidence_stops < 1)
            or np.any(np.diff(evidence_stops) <= 0)
        ):
            raise ValueError(
                "evidence_frame_stops must be positive and increase with prefixes"
            )
        factual_ids = _validated_sha_tuple(
            self.factual_intervention_ids,
            name="factual_intervention_ids",
            unique=False,
        )
        if len(factual_ids) != len(prefixes):
            raise ValueError("factual_intervention_ids must identify every prefix")
        raw_criteria = np.asarray(self.criterion_pass)
        if raw_criteria.dtype.kind != "b":
            raise ValueError("criterion_pass must contain Booleans")
        criteria = readonly_array(raw_criteria, dtype=bool)
        expected_shape = (len(prefixes), len(PREQUENTIAL_STABILITY_CRITERIA))
        if criteria.shape != expected_shape:
            raise ValueError(f"criterion_pass must have shape {expected_shape}")
        raw_step = np.asarray(self.step_pass)
        if raw_step.dtype.kind != "b":
            raise ValueError("step_pass must contain Booleans")
        step_pass = readonly_array(raw_step, dtype=bool)
        if step_pass.shape != (len(prefixes),) or not np.array_equal(
            step_pass,
            np.all(criteria, axis=1),
        ):
            raise ValueError("step_pass must equal the conjunction of criterion_pass")
        consecutive = readonly_integer_array(
            self.consecutive_pass_counts,
            name="consecutive_pass_counts",
        )
        if consecutive.shape != (len(prefixes),) or np.any(consecutive < 0):
            raise ValueError("consecutive_pass_counts must identify every prefix")
        expected_consecutive = np.zeros(len(prefixes), dtype=np.int64)
        count = 0
        for index, passed in enumerate(step_pass):
            count = count + 1 if bool(passed) else 0
            expected_consecutive[index] = count
        if not np.array_equal(consecutive, expected_consecutive):
            raise ValueError("consecutive_pass_counts do not match step_pass")

        selected_index = self.selected_step_index
        selected_prefix = self.selected_prefix_frame_count
        selected_stop = self.selected_evidence_frame_stop
        selected_factual_id = self.selected_factual_intervention_id
        fallback_reason = self.fallback_reason
        if status == "accepted":
            index = _require_integer(
                selected_index,
                name="selected_step_index",
                minimum=0,
            )
            if index >= len(prefixes):
                raise ValueError("selected_step_index exceeds the path")
            if selected_prefix != int(prefixes[index]):
                raise ValueError("selected_prefix_frame_count does not match the path")
            if consecutive[index] < required_passes:
                raise ValueError(
                    "accepted decision does not satisfy required_consecutive_passes"
                )
            if np.any(consecutive[:index] >= required_passes):
                raise ValueError(
                    "accepted decision must select the first passing prefix"
                )
            if selected_stop != int(evidence_stops[index]):
                raise ValueError("selected_evidence_frame_stop does not match the path")
            selected_factual_id = _require_sha256(
                selected_factual_id,
                name="selected_factual_intervention_id",
            )
            if selected_factual_id != factual_ids[index]:
                raise ValueError(
                    "selected_factual_intervention_id does not match the path"
                )
            if fallback_reason is not None:
                raise ValueError("accepted decisions must not carry a fallback reason")
        else:
            if any(
                value is not None
                for value in (
                    selected_index,
                    selected_prefix,
                    selected_stop,
                    selected_factual_id,
                )
            ):
                raise ValueError("fallback decisions must not select a path step")
            fallback_reason = _require_nonempty_string(
                fallback_reason,
                name="fallback_reason",
            )
            if np.any(consecutive >= required_passes):
                raise ValueError("fallback decision cannot contain a passing prefix")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "required_consecutive_passes", required_passes)
        object.__setattr__(self, "prefix_frame_counts", prefixes)
        object.__setattr__(self, "evidence_frame_stops", evidence_stops)
        object.__setattr__(self, "factual_intervention_ids", factual_ids)
        object.__setattr__(self, "criterion_pass", criteria)
        object.__setattr__(self, "step_pass", step_pass)
        object.__setattr__(self, "consecutive_pass_counts", consecutive)
        object.__setattr__(
            self,
            "selected_factual_intervention_id",
            selected_factual_id,
        )
        object.__setattr__(self, "fallback_reason", fallback_reason)
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message=(
                    "stability-decision metadata must contain finite JSON data"
                ),
            ),
        )

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PREQUENTIAL_STABILITY_DECISION_SCHEMA_VERSION,
            "artifact_kind": _DECISION_KIND,
            "source_prequential_path_id": self.source_prequential_path_id,
            "source_query_stability_id": self.source_query_stability_id,
            "config_id": self.config_id,
            "status": self.status,
            "required_consecutive_passes": self.required_consecutive_passes,
            "criteria": list(PREQUENTIAL_STABILITY_CRITERIA),
            "prefix_frame_counts": self.prefix_frame_counts.tolist(),
            "evidence_frame_stops": self.evidence_frame_stops.tolist(),
            "factual_intervention_ids": list(self.factual_intervention_ids),
            "criterion_pass": self.criterion_pass.tolist(),
            "step_pass": self.step_pass.tolist(),
            "consecutive_pass_counts": self.consecutive_pass_counts.tolist(),
            "selected_step_index": self.selected_step_index,
            "selected_prefix_frame_count": self.selected_prefix_frame_count,
            "selected_evidence_frame_stop": self.selected_evidence_frame_stop,
            "selected_factual_intervention_id": (self.selected_factual_intervention_id),
            "fallback_reason": self.fallback_reason,
            "metadata": plain_json(self.metadata),
            "claim_boundary": _CLAIM_BOUNDARY,
        }

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "artifact_id": self.artifact_id,
            "array_sha256": {
                "prefix_frame_counts": array_sha256(self.prefix_frame_counts),
                "evidence_frame_stops": array_sha256(self.evidence_frame_stops),
                "criterion_pass": array_sha256(self.criterion_pass),
                "step_pass": array_sha256(self.step_pass),
                "consecutive_pass_counts": array_sha256(self.consecutive_pass_counts),
            },
        }


def evaluate_prequential_stability(
    path: PrequentialAbductionPathV1,
    query_stability: PrequentialQueryStabilityV1,
    config: PrequentialStabilityGateConfigV1,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> PrequentialStabilityDecisionV1:
    """Apply source-frozen thresholds using only previous-prefix diagnostics."""

    if not isinstance(path, PrequentialAbductionPathV1):
        raise TypeError("path must be PrequentialAbductionPathV1")
    if not isinstance(query_stability, PrequentialQueryStabilityV1):
        raise TypeError("query_stability must be PrequentialQueryStabilityV1")
    if not isinstance(config, PrequentialStabilityGateConfigV1):
        raise TypeError("config must be PrequentialStabilityGateConfigV1")
    if path.metadata.get("future_frames_read") != 0:
        raise ValueError("prequential path must declare future_frames_read=0")
    if query_stability.source_prequential_path_id != path.artifact_id:
        raise ValueError("query stability does not bind the supplied path")
    if not np.array_equal(
        query_stability.prefix_frame_counts,
        path.prefix_frame_counts,
    ):
        raise ValueError("query stability prefix counts do not match the path")
    if not np.array_equal(query_stability.posterior_weights, path.posterior_weights):
        raise ValueError("query stability posterior weights do not match the path")

    summaries = query_stability.summary_arrays()
    step_count = len(path.prefix_frame_counts)
    criteria = np.column_stack(
        (
            np.arange(step_count) > 0,
            path.prefix_frame_counts >= config.minimum_prefix_frame_count,
            path.previous_step_total_variation
            <= config.maximum_previous_total_variation,
            path.previous_step_kl <= config.maximum_previous_kl,
            path.posterior_effective_sample_size
            >= config.minimum_effective_sample_size,
            summaries["previous_mean_shift_standardized_l2"]
            <= config.maximum_query_mean_shift_standardized_l2,
            summaries["previous_gaussian_wasserstein_standardized"]
            <= config.maximum_query_wasserstein_standardized,
            summaries["previous_interval_overlap_fraction"]
            >= config.minimum_query_interval_overlap_fraction,
        )
    ).astype(bool, copy=False)
    step_pass = np.all(criteria, axis=1)
    consecutive = np.zeros(step_count, dtype=np.int64)
    count = 0
    selected: int | None = None
    for index, passed in enumerate(step_pass):
        count = count + 1 if bool(passed) else 0
        consecutive[index] = count
        if selected is None and count >= config.required_consecutive_passes:
            selected = index

    accepted = selected is not None
    selected_index = selected if accepted else None
    return PrequentialStabilityDecisionV1(
        source_prequential_path_id=path.artifact_id,
        source_query_stability_id=query_stability.artifact_id,
        config_id=config.artifact_id,
        status="accepted" if accepted else "fallback",
        required_consecutive_passes=config.required_consecutive_passes,
        prefix_frame_counts=path.prefix_frame_counts,
        evidence_frame_stops=path.evidence_frame_stops,
        factual_intervention_ids=path.factual_intervention_ids,
        criterion_pass=criteria,
        step_pass=step_pass,
        consecutive_pass_counts=consecutive,
        selected_step_index=selected_index,
        selected_prefix_frame_count=(
            int(path.prefix_frame_counts[selected]) if accepted else None
        ),
        selected_evidence_frame_stop=(
            int(path.evidence_frame_stops[selected]) if accepted else None
        ),
        selected_factual_intervention_id=(
            path.factual_intervention_ids[selected] if accepted else None
        ),
        fallback_reason=(
            None if accepted else "no_prefix_satisfied_source_frozen_stability_gate"
        ),
        metadata={
            "operator": "prequential-stability-gate-v1",
            "uses_previous_prefix_metrics_only": True,
            "uses_final_path_metrics": False,
            "future_frames_read": 0,
            "required_consecutive_passes": config.required_consecutive_passes,
            "user_metadata": plain_json(metadata or {}),
        },
    )


def route_prequential_factual_intervention(
    result: PrequentialAbductionResult,
    decision: PrequentialStabilityDecisionV1,
    *,
    fallback: FactualIntervention,
) -> FactualIntervention:
    """Return the selected step or the caller's exact fallback object."""

    if not isinstance(result, PrequentialAbductionResult):
        raise TypeError("result must be PrequentialAbductionResult")
    if not isinstance(decision, PrequentialStabilityDecisionV1):
        raise TypeError("decision must be PrequentialStabilityDecisionV1")
    if not isinstance(fallback, FactualIntervention):
        raise TypeError("fallback must be FactualIntervention")
    if decision.source_prequential_path_id != result.path.artifact_id:
        raise ValueError("stability decision does not bind the supplied path")
    if not np.array_equal(
        decision.prefix_frame_counts,
        result.path.prefix_frame_counts,
    ):
        raise ValueError("stability decision prefix counts do not match the path")
    if not np.array_equal(
        decision.evidence_frame_stops,
        result.path.evidence_frame_stops,
    ):
        raise ValueError("stability decision evidence stops do not match the path")
    if decision.factual_intervention_ids != result.path.factual_intervention_ids:
        raise ValueError("stability decision factual identities do not match the path")
    if decision.status == "fallback":
        return fallback
    assert decision.selected_step_index is not None
    selected = result.factual_interventions[decision.selected_step_index]
    if selected.artifact_id != decision.selected_factual_intervention_id:
        raise ValueError("stability decision selected factual identity is inconsistent")
    return selected


_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "minimum_prefix_frame_count",
        "required_consecutive_passes",
        "maximum_previous_total_variation",
        "maximum_previous_kl",
        "minimum_effective_sample_size",
        "maximum_query_mean_shift_standardized_l2",
        "maximum_query_wasserstein_standardized",
        "minimum_query_interval_overlap_fraction",
        "source_artifact_ids",
        "source_only",
        "registered_before_target_access",
        "metadata",
        "claim_boundary",
    }
)
_DECISION_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "source_prequential_path_id",
        "source_query_stability_id",
        "config_id",
        "status",
        "required_consecutive_passes",
        "criteria",
        "prefix_frame_counts",
        "evidence_frame_stops",
        "factual_intervention_ids",
        "criterion_pass",
        "step_pass",
        "consecutive_pass_counts",
        "selected_step_index",
        "selected_prefix_frame_count",
        "selected_evidence_frame_stop",
        "selected_factual_intervention_id",
        "fallback_reason",
        "metadata",
        "claim_boundary",
        "array_sha256",
    }
)


def write_prequential_stability_gate_config(
    path: str | Path,
    config: PrequentialStabilityGateConfigV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(config, PrequentialStabilityGateConfigV1):
        raise TypeError("config must be PrequentialStabilityGateConfigV1")
    atomic_write_json(path, config.as_dict(), overwrite=overwrite)


def load_prequential_stability_gate_config(
    path: str | Path,
) -> PrequentialStabilityGateConfigV1:
    snapshot = read_regular_file_no_symlinks(
        path,
        name="prequential stability gate config",
    )
    fields = _require_exact_fields(
        load_strict_json_object(
            snapshot.payload,
            name="prequential stability gate config",
        ),
        name="prequential stability gate config",
        required=_CONFIG_FIELDS,
    )
    if fields["schema_version"] != PREQUENTIAL_STABILITY_GATE_CONFIG_SCHEMA_VERSION:
        raise ValueError("unsupported prequential stability config schema version")
    if fields["artifact_kind"] != _CONFIG_KIND:
        raise ValueError("unexpected prequential stability config artifact kind")
    declared_id = _require_sha256(fields["artifact_id"], name="artifact_id")
    if fields["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("prequential stability config claim boundary is invalid")
    config = PrequentialStabilityGateConfigV1(
        minimum_prefix_frame_count=fields["minimum_prefix_frame_count"],
        required_consecutive_passes=fields["required_consecutive_passes"],
        maximum_previous_total_variation=fields["maximum_previous_total_variation"],
        maximum_previous_kl=fields["maximum_previous_kl"],
        minimum_effective_sample_size=fields["minimum_effective_sample_size"],
        maximum_query_mean_shift_standardized_l2=fields[
            "maximum_query_mean_shift_standardized_l2"
        ],
        maximum_query_wasserstein_standardized=fields[
            "maximum_query_wasserstein_standardized"
        ],
        minimum_query_interval_overlap_fraction=fields[
            "minimum_query_interval_overlap_fraction"
        ],
        source_artifact_ids=_validated_sha_tuple(
            fields["source_artifact_ids"],
            name="source_artifact_ids",
        ),
        source_only=fields["source_only"],
        registered_before_target_access=fields["registered_before_target_access"],
        metadata=_require_mapping(fields["metadata"], name="metadata"),
    )
    if config.artifact_id != declared_id:
        raise ValueError("prequential stability config digest does not match")
    return config


def write_prequential_stability_decision(
    path: str | Path,
    decision: PrequentialStabilityDecisionV1,
    *,
    overwrite: bool = False,
) -> None:
    if not isinstance(decision, PrequentialStabilityDecisionV1):
        raise TypeError("decision must be PrequentialStabilityDecisionV1")
    atomic_write_json(path, decision.as_dict(), overwrite=overwrite)


def load_prequential_stability_decision(
    path: str | Path,
) -> PrequentialStabilityDecisionV1:
    snapshot = read_regular_file_no_symlinks(
        path,
        name="prequential stability decision",
    )
    fields = _require_exact_fields(
        load_strict_json_object(
            snapshot.payload,
            name="prequential stability decision",
        ),
        name="prequential stability decision",
        required=_DECISION_FIELDS,
    )
    if fields["schema_version"] != PREQUENTIAL_STABILITY_DECISION_SCHEMA_VERSION:
        raise ValueError("unsupported prequential stability decision schema version")
    if fields["artifact_kind"] != _DECISION_KIND:
        raise ValueError("unexpected prequential stability decision artifact kind")
    declared_id = _require_sha256(fields["artifact_id"], name="artifact_id")
    if tuple(fields["criteria"]) != PREQUENTIAL_STABILITY_CRITERIA:
        raise ValueError("prequential stability criterion inventory is invalid")
    if fields["claim_boundary"] != _CLAIM_BOUNDARY:
        raise ValueError("prequential stability decision claim boundary is invalid")
    decision = PrequentialStabilityDecisionV1(
        source_prequential_path_id=fields["source_prequential_path_id"],
        source_query_stability_id=fields["source_query_stability_id"],
        config_id=fields["config_id"],
        status=fields["status"],
        required_consecutive_passes=fields["required_consecutive_passes"],
        prefix_frame_counts=np.asarray(fields["prefix_frame_counts"]),
        evidence_frame_stops=np.asarray(fields["evidence_frame_stops"]),
        factual_intervention_ids=_validated_sha_tuple(
            fields["factual_intervention_ids"],
            name="factual_intervention_ids",
            unique=False,
        ),
        criterion_pass=np.asarray(fields["criterion_pass"]),
        step_pass=np.asarray(fields["step_pass"]),
        consecutive_pass_counts=np.asarray(
            fields["consecutive_pass_counts"],
        ),
        selected_step_index=fields["selected_step_index"],
        selected_prefix_frame_count=fields["selected_prefix_frame_count"],
        selected_evidence_frame_stop=fields["selected_evidence_frame_stop"],
        selected_factual_intervention_id=fields["selected_factual_intervention_id"],
        fallback_reason=fields["fallback_reason"],
        metadata=_require_mapping(fields["metadata"], name="metadata"),
    )
    if decision.artifact_id != declared_id:
        raise ValueError("prequential stability decision digest does not match")
    expected_hashes = decision.as_dict()["array_sha256"]
    if fields["array_sha256"] != expected_hashes:
        raise ValueError("prequential stability decision array digests do not match")
    return decision


__all__ = [
    "PREQUENTIAL_STABILITY_CRITERIA",
    "PREQUENTIAL_STABILITY_DECISION_SCHEMA_VERSION",
    "PREQUENTIAL_STABILITY_GATE_CONFIG_SCHEMA_VERSION",
    "DecisionStatus",
    "PrequentialStabilityDecisionV1",
    "PrequentialStabilityGateConfigV1",
    "evaluate_prequential_stability",
    "load_prequential_stability_decision",
    "load_prequential_stability_gate_config",
    "route_prequential_factual_intervention",
    "write_prequential_stability_decision",
    "write_prequential_stability_gate_config",
]
