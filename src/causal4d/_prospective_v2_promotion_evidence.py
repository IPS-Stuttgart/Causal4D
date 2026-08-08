"""Evidence-bound inputs for prospective Causal4D V2 promotion.

The contracts in this module separate target-free registration from target-side
scoring.  Candidate selection, exact fallback, and harmful-update labels are
derived from a validated decision trace and baseline-relative metrics; callers
do not provide those labels.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Real
from typing import Any

from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS, UnifiedDecisionTrace
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.prospective_v2_profile import (
    validate_prospective_v2_decision_trace_v1,
)


PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION = 1
PROSPECTIVE_V2_SELECTION_PANEL_ROLE = "candidate_selection_only"
PROSPECTIVE_V2_CANDIDATE_KINDS = (
    "registered_baseline",
    "normalized_diagonal",
    "normalized_full_covariance",
    "full_covariance_coreset",
    "sparse_contact_patch",
)
PROSPECTIVE_V2_METRIC_SEMANTICS = {
    "log_score_gain": "candidate_minus_baseline_higher_is_better",
    "brier_change": "candidate_minus_baseline_lower_is_better",
    "trajectory_regret_m": "candidate_minus_baseline_lower_is_better",
    "coverage_error": "absolute_empirical_minus_nominal_lower_is_better",
    "interval_width_ratio": "candidate_over_baseline_lower_is_better",
    "accepted_update": "derived_from_validated_v2_decision_trace",
    "fallback": "deployed_prediction_is_exact_baseline_prediction",
    "harmful_accepted_update": (
        "accepted_and_trajectory_regret_exceeds_frozen_threshold"
    ),
}

_FORBIDDEN_SOURCE_METADATA_KEYS = {
    "candidate_metric",
    "evaluation_target",
    "held_out_target",
    "target_future",
    "target_loss",
    "target_metric",
    "target_outcome",
    "target_outcomes",
    "target_value",
}

_TRACE_METADATA_FIELDS = {
    "promotion_candidate_id": "candidate_id",
    "promotion_candidate_configuration_id": "candidate_configuration_id",
    "promotion_evaluation_unit_id": "evaluation_unit_id",
    "promotion_target_access_seal_id": "target_access_seal_id",
}


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    digest = _require_nonempty_string(value, name=name)
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return digest


def _require_bool(value: Any, *, name: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def _finite_float(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def _finite_nonnegative_float(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _rate(value: Any, *, name: str) -> float:
    result = _finite_float(value, name=name)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _positive_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _utc_timestamp(value: Any, *, name: str) -> str:
    text = _require_nonempty_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError(f"{name} must use UTC")
    return text


def _validated_unique_strings(
    values: Any,
    *,
    name: str,
    require_sha256: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    validator = _require_sha256 if require_sha256 else _require_nonempty_string
    result = tuple(
        validator(value, name=f"{name}[{index}]") for index, value in enumerate(values)
    )
    if not result:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique values")
    return result


def _reject_target_metadata(value: Any, *, path: str) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = key.strip().lower().replace("-", "_")
            if normalized in _FORBIDDEN_SOURCE_METADATA_KEYS:
                raise ValueError(f"{path}.{key} is forbidden before target opening")
            _reject_target_metadata(item, path=f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, item in enumerate(value):
            _reject_target_metadata(item, path=f"{path}[{index}]")


def _validated_source_metadata(
    values: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    metadata = validated_json_mapping(
        values,
        error_message=f"{name} must contain finite JSON data",
    )
    _reject_target_metadata(metadata, path=name)
    return metadata


def _validated_target_metadata(
    values: Mapping[str, Any],
    *,
    name: str,
) -> Mapping[str, Any]:
    return validated_json_mapping(
        values,
        error_message=f"{name} must contain finite JSON data",
    )


@dataclass(frozen=True)
class ProspectiveV2MetricContractV1:
    """Freeze score semantics and harmful-update interpretation before access."""

    scoring_implementation_artifact_id: str
    nominal_coverage: float
    harmful_regret_threshold_m: float
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        implementation_id = _require_sha256(
            self.scoring_implementation_artifact_id,
            name="scoring_implementation_artifact_id",
        )
        nominal_coverage = _rate(self.nominal_coverage, name="nominal_coverage")
        if nominal_coverage in {0.0, 1.0}:
            raise ValueError("nominal_coverage must lie strictly between zero and one")
        threshold = _finite_nonnegative_float(
            self.harmful_regret_threshold_m,
            name="harmful_regret_threshold_m",
        )
        if _require_bool(self.target_outcomes_used, name="target_outcomes_used"):
            raise ValueError("metric contract must be frozen before target access")
        metadata = _validated_source_metadata(
            self.metadata,
            name="metric-contract metadata",
        )
        object.__setattr__(
            self,
            "scoring_implementation_artifact_id",
            implementation_id,
        )
        object.__setattr__(self, "nominal_coverage", nominal_coverage)
        object.__setattr__(self, "harmful_regret_threshold_m", threshold)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2MetricContractV1",
            "scoring_implementation_artifact_id": (
                self.scoring_implementation_artifact_id
            ),
            "metric_semantics": dict(PROSPECTIVE_V2_METRIC_SEMANTICS),
            "nominal_coverage": self.nominal_coverage,
            "harmful_regret_threshold_m": self.harmful_regret_threshold_m,
            "selection_panel_role": PROSPECTIVE_V2_SELECTION_PANEL_ROLE,
            "unbiased_post_selection_performance_claimed": False,
            "independent_confirmation_required": True,
            "target_outcomes_used": False,
            "metadata": plain_json(self.metadata),
        }

    @property
    def metric_contract_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "metric_contract_id": self.metric_contract_id}


@dataclass(frozen=True)
class ProspectiveV2PromotionPolicyV1:
    """Endpoint-wise promotion thresholds frozen before target access."""

    minimum_units_per_endpoint: int
    minimum_mean_log_score_gain: float
    maximum_mean_brier_change: float
    maximum_mean_trajectory_regret_m: float
    maximum_mean_coverage_error: float
    maximum_mean_interval_width_ratio: float
    minimum_accepted_update_rate: float
    maximum_harmful_accepted_update_rate: float
    maximum_fallback_rate: float
    interval_width_floor_m: float = 1e-9

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_units_per_endpoint",
            _positive_integer(
                self.minimum_units_per_endpoint,
                name="minimum_units_per_endpoint",
            ),
        )
        for name in (
            "minimum_mean_log_score_gain",
            "maximum_mean_brier_change",
            "maximum_mean_trajectory_regret_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        object.__setattr__(
            self,
            "maximum_mean_coverage_error",
            _rate(
                self.maximum_mean_coverage_error,
                name="maximum_mean_coverage_error",
            ),
        )
        object.__setattr__(
            self,
            "maximum_mean_interval_width_ratio",
            _finite_nonnegative_float(
                self.maximum_mean_interval_width_ratio,
                name="maximum_mean_interval_width_ratio",
            ),
        )
        object.__setattr__(
            self,
            "minimum_accepted_update_rate",
            _rate(
                self.minimum_accepted_update_rate,
                name="minimum_accepted_update_rate",
            ),
        )
        object.__setattr__(
            self,
            "maximum_harmful_accepted_update_rate",
            _rate(
                self.maximum_harmful_accepted_update_rate,
                name="maximum_harmful_accepted_update_rate",
            ),
        )
        object.__setattr__(
            self,
            "maximum_fallback_rate",
            _rate(self.maximum_fallback_rate, name="maximum_fallback_rate"),
        )
        floor = _finite_nonnegative_float(
            self.interval_width_floor_m,
            name="interval_width_floor_m",
        )
        if floor <= 0.0:
            raise ValueError("interval_width_floor_m must be positive")
        object.__setattr__(self, "interval_width_floor_m", floor)

    def _payload(self) -> dict[str, Any]:
        return {
            "minimum_units_per_endpoint": self.minimum_units_per_endpoint,
            "minimum_mean_log_score_gain": self.minimum_mean_log_score_gain,
            "maximum_mean_brier_change": self.maximum_mean_brier_change,
            "maximum_mean_trajectory_regret_m": (self.maximum_mean_trajectory_regret_m),
            "maximum_mean_coverage_error": self.maximum_mean_coverage_error,
            "maximum_mean_interval_width_ratio": (
                self.maximum_mean_interval_width_ratio
            ),
            "minimum_accepted_update_rate": self.minimum_accepted_update_rate,
            "maximum_harmful_accepted_update_rate": (
                self.maximum_harmful_accepted_update_rate
            ),
            "maximum_fallback_rate": self.maximum_fallback_rate,
            "interval_width_floor_m": self.interval_width_floor_m,
        }

    @property
    def policy_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProspectiveV2PromotionPolicyV1",
                **self._payload(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "policy_id": self.policy_id}


@dataclass(frozen=True)
class ProspectiveV2CandidateV1:
    """One source-frozen candidate configuration in the fixed V2 ladder."""

    candidate_id: str
    candidate_kind: str
    configuration_artifact_id: str
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        candidate_id = _require_nonempty_string(self.candidate_id, name="candidate_id")
        candidate_kind = _require_nonempty_string(
            self.candidate_kind,
            name="candidate_kind",
        )
        if candidate_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS:
            raise ValueError("candidate_kind does not belong to the V2 ladder")
        configuration_id = _require_sha256(
            self.configuration_artifact_id,
            name="configuration_artifact_id",
        )
        if _require_bool(self.target_outcomes_used, name="target_outcomes_used"):
            raise ValueError("candidate configuration must remain target-free")
        metadata = _validated_source_metadata(
            self.metadata,
            name="candidate metadata",
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(self, "configuration_artifact_id", configuration_id)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "configuration_artifact_id": self.configuration_artifact_id,
            "target_outcomes_used": False,
            "metadata": plain_json(self.metadata),
        }

    @property
    def candidate_binding_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProspectiveV2CandidateV1",
                **self._payload(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "candidate_binding_id": self.candidate_binding_id}


@dataclass(frozen=True)
class ProspectiveV2EvaluationUnitV1:
    """One target artifact and independent group registered before opening."""

    unit_id: str
    endpoint: str
    protocol_id: str
    case_id: str
    session_id: str
    independent_group_id: str
    target_artifact_id: str
    factual_context_artifact_id: str
    counterfactual_query_artifact_id: str
    target_access_seal_id: str
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unit_id = _require_nonempty_string(self.unit_id, name="unit_id")
        endpoint = _require_nonempty_string(self.endpoint, name="endpoint")
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError("evaluation unit endpoint is invalid")
        protocol_id = _require_nonempty_string(self.protocol_id, name="protocol_id")
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        session_id = _require_nonempty_string(self.session_id, name="session_id")
        independent_group_id = _require_nonempty_string(
            self.independent_group_id,
            name="independent_group_id",
        )
        target_id = _require_sha256(
            self.target_artifact_id,
            name="target_artifact_id",
        )
        factual_id = _require_sha256(
            self.factual_context_artifact_id,
            name="factual_context_artifact_id",
        )
        query_id = _require_sha256(
            self.counterfactual_query_artifact_id,
            name="counterfactual_query_artifact_id",
        )
        seal_id = _require_sha256(
            self.target_access_seal_id,
            name="target_access_seal_id",
        )
        if _require_bool(self.target_outcomes_used, name="target_outcomes_used"):
            raise ValueError("evaluation units must be registered before target access")
        metadata = _validated_source_metadata(
            self.metadata,
            name="evaluation-unit metadata",
        )
        object.__setattr__(self, "unit_id", unit_id)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "protocol_id", protocol_id)
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "session_id", session_id)
        object.__setattr__(self, "independent_group_id", independent_group_id)
        object.__setattr__(self, "target_artifact_id", target_id)
        object.__setattr__(self, "factual_context_artifact_id", factual_id)
        object.__setattr__(self, "counterfactual_query_artifact_id", query_id)
        object.__setattr__(self, "target_access_seal_id", seal_id)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "endpoint": self.endpoint,
            "protocol_id": self.protocol_id,
            "case_id": self.case_id,
            "session_id": self.session_id,
            "independent_group_id": self.independent_group_id,
            "target_artifact_id": self.target_artifact_id,
            "factual_context_artifact_id": self.factual_context_artifact_id,
            "counterfactual_query_artifact_id": (self.counterfactual_query_artifact_id),
            "target_access_seal_id": self.target_access_seal_id,
            "target_outcomes_used": False,
            "metadata": plain_json(self.metadata),
        }

    @property
    def unit_binding_id(self) -> str:
        return _canonical_sha256(
            {
                "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
                "artifact_kind": "Causal4DProspectiveV2EvaluationUnitV1",
                **self._payload(),
            }
        )

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "unit_binding_id": self.unit_binding_id}


@dataclass(frozen=True)
class ProspectiveV2PromotionFreezeV1:
    """Complete target-free registration for one candidate-selection panel."""

    experiment_id: str
    stack_lock_id: str
    target_access_seal_id: str
    candidates: tuple[ProspectiveV2CandidateV1, ...]
    evaluation_units: tuple[ProspectiveV2EvaluationUnitV1, ...]
    metric_contract: ProspectiveV2MetricContractV1
    policy: ProspectiveV2PromotionPolicyV1
    source_artifact_ids: tuple[str, ...]
    target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        experiment_id = _require_nonempty_string(
            self.experiment_id,
            name="experiment_id",
        )
        stack_lock_id = _require_sha256(self.stack_lock_id, name="stack_lock_id")
        seal_id = _require_sha256(
            self.target_access_seal_id,
            name="target_access_seal_id",
        )
        candidates = tuple(self.candidates)
        if any(
            type(candidate) is not ProspectiveV2CandidateV1 for candidate in candidates
        ):
            raise ValueError("candidates must contain ProspectiveV2CandidateV1 values")
        if tuple(candidate.candidate_kind for candidate in candidates) != (
            PROSPECTIVE_V2_CANDIDATE_KINDS
        ):
            raise ValueError("candidate ladder must exactly match the V2 registration")
        candidate_ids = tuple(candidate.candidate_id for candidate in candidates)
        candidate_bindings = tuple(
            candidate.candidate_binding_id for candidate in candidates
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate IDs must be unique")
        if len(set(candidate_bindings)) != len(candidate_bindings):
            raise ValueError("candidate bindings must be unique")
        units = tuple(self.evaluation_units)
        if not units or any(
            type(unit) is not ProspectiveV2EvaluationUnitV1 for unit in units
        ):
            raise ValueError(
                "evaluation_units must contain ProspectiveV2EvaluationUnitV1 values"
            )
        if any(unit.target_access_seal_id != seal_id for unit in units):
            raise ValueError("every evaluation unit must bind the frozen access seal")
        unit_ids = tuple(unit.unit_id for unit in units)
        unit_bindings = tuple(unit.unit_binding_id for unit in units)
        target_ids = tuple(unit.target_artifact_id for unit in units)
        if len(set(unit_ids)) != len(unit_ids):
            raise ValueError("evaluation unit IDs must be unique")
        if len(set(unit_bindings)) != len(unit_bindings):
            raise ValueError("evaluation unit bindings must be unique")
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("registered target artifact IDs must be unique")
        independence_keys = tuple(
            (unit.endpoint, unit.independent_group_id) for unit in units
        )
        if len(set(independence_keys)) != len(independence_keys):
            raise ValueError(
                "independent_group_id must be unique within every endpoint"
            )
        if type(self.metric_contract) is not ProspectiveV2MetricContractV1:
            raise ValueError("metric_contract has the wrong type")
        if type(self.policy) is not ProspectiveV2PromotionPolicyV1:
            raise ValueError("policy has the wrong type")
        for endpoint in DECISION_TRACE_ENDPOINTS:
            count = sum(unit.endpoint == endpoint for unit in units)
            if count < self.policy.minimum_units_per_endpoint:
                raise ValueError(
                    f"endpoint {endpoint!r} has insufficient independent units"
                )
        source_ids = _validated_unique_strings(
            self.source_artifact_ids,
            name="source_artifact_ids",
            require_sha256=True,
        )
        required_source_ids = {
            stack_lock_id,
            seal_id,
            self.metric_contract.scoring_implementation_artifact_id,
            *(candidate.configuration_artifact_id for candidate in candidates),
            *(unit.factual_context_artifact_id for unit in units),
            *(unit.counterfactual_query_artifact_id for unit in units),
        }
        missing_source_ids = sorted(required_source_ids - set(source_ids))
        if missing_source_ids:
            raise ValueError(
                "source_artifact_ids do not bind every target-free source; "
                f"missing={missing_source_ids}"
            )
        if _require_bool(self.target_outcomes_used, name="target_outcomes_used"):
            raise ValueError("promotion freeze must predate target access")
        metadata = _validated_source_metadata(
            self.metadata,
            name="promotion-freeze metadata",
        )
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "stack_lock_id", stack_lock_id)
        object.__setattr__(self, "target_access_seal_id", seal_id)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "evaluation_units", units)
        object.__setattr__(self, "source_artifact_ids", source_ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def baseline_candidate(self) -> ProspectiveV2CandidateV1:
        return self.candidates[0]

    @property
    def candidate_by_id(self) -> dict[str, ProspectiveV2CandidateV1]:
        return {candidate.candidate_id: candidate for candidate in self.candidates}

    @property
    def unit_by_id(self) -> dict[str, ProspectiveV2EvaluationUnitV1]:
        return {unit.unit_id: unit for unit in self.evaluation_units}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2PromotionFreezeV1",
            "experiment_id": self.experiment_id,
            "stack_lock_id": self.stack_lock_id,
            "target_access_seal_id": self.target_access_seal_id,
            "candidates": [candidate.as_dict() for candidate in self.candidates],
            "evaluation_units": [unit.as_dict() for unit in self.evaluation_units],
            "metric_contract": self.metric_contract.as_dict(),
            "policy": self.policy.as_dict(),
            "source_artifact_ids": list(self.source_artifact_ids),
            "selection_panel_role": PROSPECTIVE_V2_SELECTION_PANEL_ROLE,
            "unbiased_post_selection_performance_claimed": False,
            "independent_confirmation_required": True,
            "target_outcomes_used": False,
            "metadata": plain_json(self.metadata),
        }

    @property
    def freeze_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "freeze_id": self.freeze_id}


@dataclass(frozen=True)
class ProspectiveV2TargetOpeningV1:
    """One content-addressed opening of the complete registered target inventory."""

    freeze_id: str
    target_access_seal_id: str
    target_artifact_ids: tuple[str, ...]
    opened_at_utc: str
    opened_by: str
    target_outcomes_used: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        freeze_id = _require_sha256(self.freeze_id, name="freeze_id")
        seal_id = _require_sha256(
            self.target_access_seal_id,
            name="target_access_seal_id",
        )
        target_ids = _validated_unique_strings(
            self.target_artifact_ids,
            name="target_artifact_ids",
            require_sha256=True,
        )
        opened_at = _utc_timestamp(self.opened_at_utc, name="opened_at_utc")
        opened_by = _require_nonempty_string(self.opened_by, name="opened_by")
        if not _require_bool(
            self.target_outcomes_used,
            name="target_outcomes_used",
        ):
            raise ValueError("target opening must explicitly record target access")
        metadata = _validated_target_metadata(
            self.metadata,
            name="target-opening metadata",
        )
        object.__setattr__(self, "freeze_id", freeze_id)
        object.__setattr__(self, "target_access_seal_id", seal_id)
        object.__setattr__(self, "target_artifact_ids", target_ids)
        object.__setattr__(self, "opened_at_utc", opened_at)
        object.__setattr__(self, "opened_by", opened_by)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2TargetOpeningV1",
            "freeze_id": self.freeze_id,
            "target_access_seal_id": self.target_access_seal_id,
            "target_artifact_ids": list(self.target_artifact_ids),
            "opened_at_utc": self.opened_at_utc,
            "opened_by": self.opened_by,
            "target_outcomes_used": True,
            "metadata": plain_json(self.metadata),
        }

    @property
    def opening_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "opening_id": self.opening_id}


def build_prospective_v2_target_opening_v1(
    freeze: ProspectiveV2PromotionFreezeV1,
    *,
    opened_at_utc: str,
    opened_by: str,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveV2TargetOpeningV1:
    """Open exactly the target inventory registered by ``freeze`` once."""

    if type(freeze) is not ProspectiveV2PromotionFreezeV1:
        raise ValueError("freeze has the wrong type")
    return ProspectiveV2TargetOpeningV1(
        freeze_id=freeze.freeze_id,
        target_access_seal_id=freeze.target_access_seal_id,
        target_artifact_ids=tuple(
            unit.target_artifact_id for unit in freeze.evaluation_units
        ),
        opened_at_utc=opened_at_utc,
        opened_by=opened_by,
        target_outcomes_used=True,
        metadata={} if metadata is None else metadata,
    )


@dataclass(frozen=True)
class ProspectiveV2UnitMetricValuesV1:
    """Raw baseline/candidate scores bound to exact target and predictions."""

    opening_id: str
    unit_binding_id: str
    candidate_binding_id: str
    target_artifact_id: str
    baseline_prediction_artifact_id: str
    candidate_prediction_artifact_id: str
    metric_contract_id: str
    scoring_run_artifact_id: str
    baseline_log_score: float
    candidate_log_score: float
    baseline_brier_score: float
    candidate_brier_score: float
    baseline_trajectory_error_m: float
    candidate_trajectory_error_m: float
    baseline_coverage: float
    candidate_coverage: float
    baseline_interval_width_m: float
    candidate_interval_width_m: float
    target_outcomes_used: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "opening_id",
            "unit_binding_id",
            "candidate_binding_id",
            "target_artifact_id",
            "baseline_prediction_artifact_id",
            "candidate_prediction_artifact_id",
            "metric_contract_id",
            "scoring_run_artifact_id",
        ):
            object.__setattr__(
                self,
                name,
                _require_sha256(getattr(self, name), name=name),
            )
        if self.baseline_prediction_artifact_id == (
            self.candidate_prediction_artifact_id
        ):
            raise ValueError("baseline and candidate predictions must differ")
        for name in ("baseline_log_score", "candidate_log_score"):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        for name in (
            "baseline_brier_score",
            "candidate_brier_score",
            "baseline_trajectory_error_m",
            "candidate_trajectory_error_m",
            "baseline_interval_width_m",
            "candidate_interval_width_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_float(getattr(self, name), name=name),
            )
        for name in ("baseline_coverage", "candidate_coverage"):
            object.__setattr__(self, name, _rate(getattr(self, name), name=name))
        if not _require_bool(
            self.target_outcomes_used,
            name="target_outcomes_used",
        ):
            raise ValueError("unit metrics must record target-outcome access")
        metadata = _validated_target_metadata(
            self.metadata,
            name="unit-metric metadata",
        )
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2UnitMetricValuesV1",
            "opening_id": self.opening_id,
            "unit_binding_id": self.unit_binding_id,
            "candidate_binding_id": self.candidate_binding_id,
            "target_artifact_id": self.target_artifact_id,
            "baseline_prediction_artifact_id": (self.baseline_prediction_artifact_id),
            "candidate_prediction_artifact_id": (self.candidate_prediction_artifact_id),
            "metric_contract_id": self.metric_contract_id,
            "scoring_run_artifact_id": self.scoring_run_artifact_id,
            "baseline_log_score": self.baseline_log_score,
            "candidate_log_score": self.candidate_log_score,
            "baseline_brier_score": self.baseline_brier_score,
            "candidate_brier_score": self.candidate_brier_score,
            "baseline_trajectory_error_m": self.baseline_trajectory_error_m,
            "candidate_trajectory_error_m": self.candidate_trajectory_error_m,
            "baseline_coverage": self.baseline_coverage,
            "candidate_coverage": self.candidate_coverage,
            "baseline_interval_width_m": self.baseline_interval_width_m,
            "candidate_interval_width_m": self.candidate_interval_width_m,
            "target_outcomes_used": True,
            "metadata": plain_json(self.metadata),
        }

    @property
    def metric_values_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "metric_values_id": self.metric_values_id}


@dataclass(frozen=True)
class ProspectiveV2UnitEvaluationV1:
    """Derived unit decision with no caller-supplied acceptance/fallback flags."""

    freeze_id: str
    stack_lock_id: str
    unit: ProspectiveV2EvaluationUnitV1
    candidate: ProspectiveV2CandidateV1
    opening: ProspectiveV2TargetOpeningV1
    metric_contract: ProspectiveV2MetricContractV1
    policy: ProspectiveV2PromotionPolicyV1
    trace: UnifiedDecisionTrace
    metric_values: ProspectiveV2UnitMetricValuesV1

    def __post_init__(self) -> None:
        freeze_id = _require_sha256(self.freeze_id, name="freeze_id")
        stack_lock_id = _require_sha256(self.stack_lock_id, name="stack_lock_id")
        if type(self.unit) is not ProspectiveV2EvaluationUnitV1:
            raise ValueError("unit has the wrong type")
        if type(self.candidate) is not ProspectiveV2CandidateV1:
            raise ValueError("candidate has the wrong type")
        if self.candidate.candidate_kind == PROSPECTIVE_V2_CANDIDATE_KINDS[0]:
            raise ValueError("unit evaluation requires a non-baseline candidate")
        if type(self.opening) is not ProspectiveV2TargetOpeningV1:
            raise ValueError("opening has the wrong type")
        if type(self.metric_contract) is not ProspectiveV2MetricContractV1:
            raise ValueError("metric_contract has the wrong type")
        if type(self.policy) is not ProspectiveV2PromotionPolicyV1:
            raise ValueError("policy has the wrong type")
        if type(self.trace) is not UnifiedDecisionTrace:
            raise ValueError("trace has the wrong type")
        if type(self.metric_values) is not ProspectiveV2UnitMetricValuesV1:
            raise ValueError("metric_values has the wrong type")
        if self.opening.freeze_id != freeze_id:
            raise ValueError("target opening does not bind the promotion freeze")
        if self.opening.target_access_seal_id != self.unit.target_access_seal_id:
            raise ValueError("target opening and evaluation unit use different seals")
        if self.unit.target_artifact_id not in self.opening.target_artifact_ids:
            raise ValueError("target opening does not include the unit target artifact")
        if self.trace.stack_lock_id != stack_lock_id:
            raise ValueError("decision trace stack lock does not match the freeze")
        if (
            self.trace.protocol_id,
            self.trace.case_id,
            self.trace.session_id,
            self.trace.endpoint,
        ) != (
            self.unit.protocol_id,
            self.unit.case_id,
            self.unit.session_id,
            self.unit.endpoint,
        ):
            raise ValueError("decision trace does not identify the evaluation unit")
        root_by_role = {
            artifact.role: artifact for artifact in self.trace.root_artifacts
        }
        if root_by_role["factual_evidence_context"].artifact_id != (
            self.unit.factual_context_artifact_id
        ):
            raise ValueError("decision trace factual-context binding changed")
        if root_by_role["counterfactual_query_context"].artifact_id != (
            self.unit.counterfactual_query_artifact_id
        ):
            raise ValueError("decision trace counterfactual-query binding changed")
        validation = validate_prospective_v2_decision_trace_v1(self.trace)
        if not validation.accepted:
            joined = ", ".join(validation.reasons)
            raise ValueError(f"decision trace fails the V2 profile: {joined}")
        expected_metadata = {
            "promotion_candidate_id": self.candidate.candidate_id,
            "promotion_candidate_configuration_id": (
                self.candidate.configuration_artifact_id
            ),
            "promotion_evaluation_unit_id": self.unit.unit_id,
            "promotion_target_access_seal_id": self.unit.target_access_seal_id,
        }
        for key, expected in expected_metadata.items():
            if self.trace.metadata.get(key) != expected:
                field = _TRACE_METADATA_FIELDS[key]
                raise ValueError(f"decision trace {field} binding changed")
        values = self.metric_values
        expected_metric_bindings = (
            self.opening.opening_id,
            self.unit.unit_binding_id,
            self.candidate.candidate_binding_id,
            self.unit.target_artifact_id,
            self.trace.selection.baseline_artifact_id,
            self.trace.selection.candidate_artifact_id,
            self.metric_contract.metric_contract_id,
        )
        observed_metric_bindings = (
            values.opening_id,
            values.unit_binding_id,
            values.candidate_binding_id,
            values.target_artifact_id,
            values.baseline_prediction_artifact_id,
            values.candidate_prediction_artifact_id,
            values.metric_contract_id,
        )
        if observed_metric_bindings != expected_metric_bindings:
            raise ValueError(
                "unit metrics do not bind the registered evaluation sources"
            )
        object.__setattr__(self, "freeze_id", freeze_id)
        object.__setattr__(self, "stack_lock_id", stack_lock_id)

    @property
    def trace_validation_id(self) -> str:
        return validate_prospective_v2_decision_trace_v1(self.trace).validation_id

    @property
    def candidate_selected(self) -> bool:
        return self.trace.selection.candidate_selected

    @property
    def fallback_used(self) -> bool:
        return not self.candidate_selected

    @property
    def log_score_gain(self) -> float:
        return self.metric_values.candidate_log_score - (
            self.metric_values.baseline_log_score
        )

    @property
    def brier_change(self) -> float:
        return self.metric_values.candidate_brier_score - (
            self.metric_values.baseline_brier_score
        )

    @property
    def trajectory_regret_m(self) -> float:
        return self.metric_values.candidate_trajectory_error_m - (
            self.metric_values.baseline_trajectory_error_m
        )

    @property
    def candidate_coverage_error(self) -> float:
        return abs(
            self.metric_values.candidate_coverage
            - self.metric_contract.nominal_coverage
        )

    @property
    def interval_width_ratio(self) -> float:
        denominator = max(
            self.metric_values.baseline_interval_width_m,
            self.policy.interval_width_floor_m,
        )
        return self.metric_values.candidate_interval_width_m / denominator

    @property
    def candidate_harmful(self) -> bool:
        return (
            self.trajectory_regret_m > self.metric_contract.harmful_regret_threshold_m
        )

    @property
    def harmful_accepted_update(self) -> bool:
        return self.candidate_selected and self.candidate_harmful

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2UnitEvaluationV1",
            "freeze_id": self.freeze_id,
            "stack_lock_id": self.stack_lock_id,
            "opening_id": self.opening.opening_id,
            "unit_binding_id": self.unit.unit_binding_id,
            "candidate_binding_id": self.candidate.candidate_binding_id,
            "metric_contract_id": self.metric_contract.metric_contract_id,
            "policy_id": self.policy.policy_id,
            "metric_values_id": self.metric_values.metric_values_id,
            "trace_id": self.trace.trace_id,
            "trace_validation_id": self.trace_validation_id,
            "unit_id": self.unit.unit_id,
            "endpoint": self.unit.endpoint,
            "candidate_id": self.candidate.candidate_id,
            "candidate_kind": self.candidate.candidate_kind,
            "baseline_prediction_artifact_id": (
                self.trace.selection.baseline_artifact_id
            ),
            "candidate_prediction_artifact_id": (
                self.trace.selection.candidate_artifact_id
            ),
            "deployed_prediction_artifact_id": (
                self.trace.selection.deployed_artifact_id
            ),
            "candidate_selected": self.candidate_selected,
            "fallback_used": self.fallback_used,
            "candidate_harmful": self.candidate_harmful,
            "harmful_accepted_update": self.harmful_accepted_update,
            "log_score_gain": self.log_score_gain,
            "brier_change": self.brier_change,
            "trajectory_regret_m": self.trajectory_regret_m,
            "candidate_coverage_error": self.candidate_coverage_error,
            "candidate_interval_width_m": (
                self.metric_values.candidate_interval_width_m
            ),
            "interval_width_ratio": self.interval_width_ratio,
            "target_outcomes_used": True,
        }

    @property
    def evaluation_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "evaluation_id": self.evaluation_id}


def build_prospective_v2_unit_evaluation_v1(
    freeze: ProspectiveV2PromotionFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    *,
    unit_id: str,
    candidate_id: str,
    trace: UnifiedDecisionTrace,
    metric_values: ProspectiveV2UnitMetricValuesV1,
) -> ProspectiveV2UnitEvaluationV1:
    """Derive one evaluation from frozen bindings, one trace, and raw scores."""

    if type(freeze) is not ProspectiveV2PromotionFreezeV1:
        raise ValueError("freeze has the wrong type")
    unit = freeze.unit_by_id.get(unit_id)
    if unit is None:
        raise ValueError("unit_id is not registered by the promotion freeze")
    candidate = freeze.candidate_by_id.get(candidate_id)
    if candidate is None:
        raise ValueError("candidate_id is not registered by the promotion freeze")
    return ProspectiveV2UnitEvaluationV1(
        freeze_id=freeze.freeze_id,
        stack_lock_id=freeze.stack_lock_id,
        unit=unit,
        candidate=candidate,
        opening=opening,
        metric_contract=freeze.metric_contract,
        policy=freeze.policy,
        trace=trace,
        metric_values=metric_values,
    )


__all__ = [
    "PROSPECTIVE_V2_CANDIDATE_KINDS",
    "PROSPECTIVE_V2_METRIC_SEMANTICS",
    "PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION",
    "PROSPECTIVE_V2_SELECTION_PANEL_ROLE",
    "ProspectiveV2CandidateV1",
    "ProspectiveV2EvaluationUnitV1",
    "ProspectiveV2MetricContractV1",
    "ProspectiveV2PromotionFreezeV1",
    "ProspectiveV2PromotionPolicyV1",
    "ProspectiveV2TargetOpeningV1",
    "ProspectiveV2UnitEvaluationV1",
    "ProspectiveV2UnitMetricValuesV1",
    "build_prospective_v2_target_opening_v1",
    "build_prospective_v2_unit_evaluation_v1",
]
