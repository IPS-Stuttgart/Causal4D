"""Evidence-bound target-opening promotion for prospective Causal4D V2.

This module aggregates unit evaluations that have already been bound to the
registered target inventory, fixed metric semantics, and validated prospective
V2 decision traces.  Promotion labels are recomputed from those sources.  The
selection panel is explicitly not an unbiased post-selection performance panel.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from causal4d._prospective_v2_promotion_evidence import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
    PROSPECTIVE_V2_SELECTION_PANEL_ROLE,
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2MetricContractV1,
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2PromotionPolicyV1,
    ProspectiveV2TargetOpeningV1,
    ProspectiveV2UnitEvaluationV1,
    ProspectiveV2UnitMetricValuesV1,
    build_prospective_v2_target_opening_v1,
    build_prospective_v2_unit_evaluation_v1,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS
from causal4d.immutable_json import plain_json, validated_json_mapping


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
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
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


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    result = float(np.mean(np.asarray(values, dtype=float)))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


@dataclass(frozen=True)
class ProspectiveV2EndpointMetricsV1:
    """One endpoint aggregate derived from bound independent-unit evaluations."""

    candidate_id: str
    endpoint: str
    unit_count: int
    mean_log_score_gain: float
    mean_brier_change: float
    mean_trajectory_regret_m: float
    mean_coverage_error: float
    mean_interval_width_m: float
    mean_interval_width_ratio: float
    accepted_update_rate: float
    harmful_accepted_update_rate: float
    fallback_rate: float
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate_id = _require_nonempty_string(
            self.candidate_id,
            name="candidate_id",
        )
        endpoint = _require_nonempty_string(self.endpoint, name="endpoint")
        if endpoint not in DECISION_TRACE_ENDPOINTS:
            raise ValueError("endpoint aggregate has an invalid endpoint")
        if type(self.unit_count) is not int or self.unit_count < 1:
            raise ValueError("unit_count must be a positive integer")
        for name in (
            "mean_log_score_gain",
            "mean_brier_change",
            "mean_trajectory_regret_m",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name=name),
            )
        for name in (
            "mean_coverage_error",
            "mean_interval_width_m",
            "mean_interval_width_ratio",
        ):
            object.__setattr__(
                self,
                name,
                _finite_nonnegative_float(getattr(self, name), name=name),
            )
        for name in (
            "accepted_update_rate",
            "harmful_accepted_update_rate",
            "fallback_rate",
        ):
            object.__setattr__(self, name, _rate(getattr(self, name), name=name))
        accepted = _require_bool(self.accepted, name="accepted")
        reasons = tuple(self.reasons)
        if accepted and reasons:
            raise ValueError("accepted endpoint metrics cannot contain reasons")
        if not accepted and not reasons:
            raise ValueError("rejected endpoint metrics require reasons")
        if len(set(reasons)) != len(reasons):
            raise ValueError("endpoint rejection reasons must be unique")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "endpoint", endpoint)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "endpoint": self.endpoint,
            "unit_count": self.unit_count,
            "mean_log_score_gain": self.mean_log_score_gain,
            "mean_brier_change": self.mean_brier_change,
            "mean_trajectory_regret_m": self.mean_trajectory_regret_m,
            "mean_coverage_error": self.mean_coverage_error,
            "mean_interval_width_m": self.mean_interval_width_m,
            "mean_interval_width_ratio": self.mean_interval_width_ratio,
            "accepted_update_rate": self.accepted_update_rate,
            "harmful_accepted_update_rate": self.harmful_accepted_update_rate,
            "fallback_rate": self.fallback_rate,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProspectiveV2CandidateResultV1:
    """All registered endpoint decisions for one non-baseline candidate."""

    candidate_id: str
    candidate_kind: str
    candidate_configuration_artifact_id: str
    endpoint_metrics: tuple[ProspectiveV2EndpointMetricsV1, ...]
    accepted: bool
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        candidate_id = _require_nonempty_string(
            self.candidate_id,
            name="candidate_id",
        )
        candidate_kind = _require_nonempty_string(
            self.candidate_kind,
            name="candidate_kind",
        )
        if candidate_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS[1:]:
            raise ValueError("candidate result must describe a non-baseline candidate")
        configuration_id = _require_sha256(
            self.candidate_configuration_artifact_id,
            name="candidate_configuration_artifact_id",
        )
        endpoint_metrics = tuple(self.endpoint_metrics)
        if tuple(metric.endpoint for metric in endpoint_metrics) != (
            DECISION_TRACE_ENDPOINTS
        ):
            raise ValueError("candidate result must cover every endpoint in order")
        if any(metric.candidate_id != candidate_id for metric in endpoint_metrics):
            raise ValueError("endpoint metrics identify the wrong candidate")
        expected_reasons = tuple(
            f"{metric.endpoint}:{reason}"
            for metric in endpoint_metrics
            for reason in metric.reasons
        )
        accepted = _require_bool(self.accepted, name="accepted")
        reasons = tuple(self.reasons)
        if reasons != expected_reasons or accepted is not (not expected_reasons):
            raise ValueError("candidate decision must exactly match endpoint evidence")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(
            self,
            "candidate_configuration_artifact_id",
            configuration_id,
        )
        object.__setattr__(self, "endpoint_metrics", endpoint_metrics)
        object.__setattr__(self, "accepted", accepted)
        object.__setattr__(self, "reasons", reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "candidate_configuration_artifact_id": (
                self.candidate_configuration_artifact_id
            ),
            "endpoint_metrics": [metric.as_dict() for metric in self.endpoint_metrics],
            "accepted": self.accepted,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class ProspectiveV2PromotionResultV1:
    """Selection-panel result with exact baseline configuration fallback."""

    freeze_id: str
    opening_id: str
    baseline_candidate_id: str
    baseline_configuration_artifact_id: str
    selected_candidate_id: str
    selected_candidate_kind: str
    selected_configuration_artifact_id: str
    candidate_results: tuple[ProspectiveV2CandidateResultV1, ...]
    evaluation_ids: tuple[str, ...]
    selection_panel_role: str = PROSPECTIVE_V2_SELECTION_PANEL_ROLE
    unbiased_post_selection_performance_claimed: bool = False
    independent_confirmation_required: bool = True
    target_outcomes_used: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        freeze_id = _require_sha256(self.freeze_id, name="freeze_id")
        opening_id = _require_sha256(self.opening_id, name="opening_id")
        baseline_candidate_id = _require_nonempty_string(
            self.baseline_candidate_id,
            name="baseline_candidate_id",
        )
        baseline_configuration_id = _require_sha256(
            self.baseline_configuration_artifact_id,
            name="baseline_configuration_artifact_id",
        )
        selected_candidate_id = _require_nonempty_string(
            self.selected_candidate_id,
            name="selected_candidate_id",
        )
        selected_kind = _require_nonempty_string(
            self.selected_candidate_kind,
            name="selected_candidate_kind",
        )
        if selected_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS:
            raise ValueError("selected_candidate_kind is invalid")
        selected_configuration_id = _require_sha256(
            self.selected_configuration_artifact_id,
            name="selected_configuration_artifact_id",
        )
        candidate_results = tuple(self.candidate_results)
        if any(
            type(result) is not ProspectiveV2CandidateResultV1
            for result in candidate_results
        ):
            raise ValueError(
                "candidate_results must contain ProspectiveV2CandidateResultV1 values"
            )
        if (
            tuple(result.candidate_kind for result in candidate_results)
            != (PROSPECTIVE_V2_CANDIDATE_KINDS[1:])
        ):
            raise ValueError("candidate results must follow the frozen V2 ladder")
        candidate_ids = tuple(result.candidate_id for result in candidate_results)
        configuration_ids = tuple(
            result.candidate_configuration_artifact_id for result in candidate_results
        )
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError("candidate result IDs must be unique")
        if len(set(configuration_ids)) != len(configuration_ids):
            raise ValueError("candidate result configurations must be unique")
        accepted_results = tuple(
            result for result in candidate_results if result.accepted
        )
        if accepted_results:
            expected_selected = accepted_results[-1]
            expected_fields = (
                expected_selected.candidate_id,
                expected_selected.candidate_kind,
                expected_selected.candidate_configuration_artifact_id,
            )
        else:
            expected_fields = (
                baseline_candidate_id,
                PROSPECTIVE_V2_CANDIDATE_KINDS[0],
                baseline_configuration_id,
            )
        if (
            selected_candidate_id,
            selected_kind,
            selected_configuration_id,
        ) != expected_fields:
            raise ValueError(
                "selection must equal the highest accepted candidate or baseline"
            )
        evaluation_ids = tuple(
            _require_sha256(value, name=f"evaluation_ids[{index}]")
            for index, value in enumerate(self.evaluation_ids)
        )
        if not evaluation_ids or len(set(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("evaluation_ids must be nonempty and unique")
        if self.selection_panel_role != PROSPECTIVE_V2_SELECTION_PANEL_ROLE:
            raise ValueError("selection_panel_role is fixed")
        if _require_bool(
            self.unbiased_post_selection_performance_claimed,
            name="unbiased_post_selection_performance_claimed",
        ):
            raise ValueError("the selection panel cannot support an unbiased claim")
        if not _require_bool(
            self.independent_confirmation_required,
            name="independent_confirmation_required",
        ):
            raise ValueError("independent confirmation must remain required")
        if not _require_bool(
            self.target_outcomes_used,
            name="target_outcomes_used",
        ):
            raise ValueError("promotion result must record target-outcome access")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="promotion-result metadata must contain finite JSON data",
        )
        object.__setattr__(self, "freeze_id", freeze_id)
        object.__setattr__(self, "opening_id", opening_id)
        object.__setattr__(self, "baseline_candidate_id", baseline_candidate_id)
        object.__setattr__(
            self,
            "baseline_configuration_artifact_id",
            baseline_configuration_id,
        )
        object.__setattr__(self, "selected_candidate_id", selected_candidate_id)
        object.__setattr__(self, "selected_candidate_kind", selected_kind)
        object.__setattr__(
            self,
            "selected_configuration_artifact_id",
            selected_configuration_id,
        )
        object.__setattr__(self, "candidate_results", candidate_results)
        object.__setattr__(self, "evaluation_ids", evaluation_ids)
        object.__setattr__(self, "metadata", metadata)

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2PromotionResultV1",
            "freeze_id": self.freeze_id,
            "opening_id": self.opening_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "baseline_configuration_artifact_id": (
                self.baseline_configuration_artifact_id
            ),
            "selected_candidate_id": self.selected_candidate_id,
            "selected_candidate_kind": self.selected_candidate_kind,
            "selected_configuration_artifact_id": (
                self.selected_configuration_artifact_id
            ),
            "candidate_results": [
                result.as_dict() for result in self.candidate_results
            ],
            "evaluation_ids": list(self.evaluation_ids),
            "selection_panel_role": self.selection_panel_role,
            "unbiased_post_selection_performance_claimed": (
                self.unbiased_post_selection_performance_claimed
            ),
            "independent_confirmation_required": (
                self.independent_confirmation_required
            ),
            "target_outcomes_used": self.target_outcomes_used,
            "metadata": plain_json(self.metadata),
        }

    @property
    def result_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "result_id": self.result_id}


def _evaluate_endpoint(
    candidate: ProspectiveV2CandidateV1,
    endpoint: str,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
    policy: ProspectiveV2PromotionPolicyV1,
) -> ProspectiveV2EndpointMetricsV1:
    unit_count = len(evaluations)
    accepted_count = sum(value.candidate_selected for value in evaluations)
    harmful_count = sum(value.harmful_accepted_update for value in evaluations)
    accepted_rate = accepted_count / unit_count
    harmful_accepted_rate = harmful_count / accepted_count if accepted_count else 0.0
    fallback_rate = sum(value.fallback_used for value in evaluations) / unit_count
    mean_log_gain = _mean(
        tuple(value.log_score_gain for value in evaluations),
        name="mean_log_score_gain",
    )
    mean_brier_change = _mean(
        tuple(value.brier_change for value in evaluations),
        name="mean_brier_change",
    )
    mean_regret = _mean(
        tuple(value.trajectory_regret_m for value in evaluations),
        name="mean_trajectory_regret_m",
    )
    mean_coverage_error = _mean(
        tuple(value.candidate_coverage_error for value in evaluations),
        name="mean_coverage_error",
    )
    mean_width = _mean(
        tuple(value.metric_values.candidate_interval_width_m for value in evaluations),
        name="mean_interval_width_m",
    )
    mean_width_ratio = _mean(
        tuple(value.interval_width_ratio for value in evaluations),
        name="mean_interval_width_ratio",
    )
    reasons: list[str] = []
    if unit_count < policy.minimum_units_per_endpoint:
        reasons.append("insufficient_independent_units")
    if mean_log_gain < policy.minimum_mean_log_score_gain:
        reasons.append("mean_log_score_gain_below_limit")
    if mean_brier_change > policy.maximum_mean_brier_change:
        reasons.append("mean_brier_change_exceeds_limit")
    if mean_regret > policy.maximum_mean_trajectory_regret_m:
        reasons.append("mean_trajectory_regret_exceeds_limit")
    if mean_coverage_error > policy.maximum_mean_coverage_error:
        reasons.append("mean_coverage_error_exceeds_limit")
    if mean_width_ratio > policy.maximum_mean_interval_width_ratio:
        reasons.append("mean_interval_width_ratio_exceeds_limit")
    if accepted_rate < policy.minimum_accepted_update_rate:
        reasons.append("accepted_update_rate_below_limit")
    if harmful_accepted_rate > policy.maximum_harmful_accepted_update_rate:
        reasons.append("harmful_accepted_update_rate_exceeds_limit")
    if fallback_rate > policy.maximum_fallback_rate:
        reasons.append("fallback_rate_exceeds_limit")
    return ProspectiveV2EndpointMetricsV1(
        candidate_id=candidate.candidate_id,
        endpoint=endpoint,
        unit_count=unit_count,
        mean_log_score_gain=mean_log_gain,
        mean_brier_change=mean_brier_change,
        mean_trajectory_regret_m=mean_regret,
        mean_coverage_error=mean_coverage_error,
        mean_interval_width_m=mean_width,
        mean_interval_width_ratio=mean_width_ratio,
        accepted_update_rate=accepted_rate,
        harmful_accepted_update_rate=harmful_accepted_rate,
        fallback_rate=fallback_rate,
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def evaluate_prospective_v2_promotion_v1(
    freeze: ProspectiveV2PromotionFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveV2PromotionResultV1:
    """Evaluate the complete frozen candidate/unit product exactly once."""

    if type(freeze) is not ProspectiveV2PromotionFreezeV1:
        raise ValueError("freeze has the wrong type")
    if type(opening) is not ProspectiveV2TargetOpeningV1:
        raise ValueError("opening has the wrong type")
    expected_target_ids = tuple(
        unit.target_artifact_id for unit in freeze.evaluation_units
    )
    if (
        opening.freeze_id != freeze.freeze_id
        or opening.target_access_seal_id != freeze.target_access_seal_id
        or opening.target_artifact_ids != expected_target_ids
    ):
        raise ValueError("target opening does not exactly match the frozen inventory")
    values = tuple(evaluations)
    if not values or any(
        type(value) is not ProspectiveV2UnitEvaluationV1 for value in values
    ):
        raise ValueError("evaluations must contain bound unit evaluations")
    expected_keys = {
        (unit.unit_id, candidate.candidate_id)
        for unit in freeze.evaluation_units
        for candidate in freeze.candidates[1:]
    }
    actual_keys = {
        (value.unit.unit_id, value.candidate.candidate_id) for value in values
    }
    if len(actual_keys) != len(values):
        raise ValueError("unit/candidate evaluation pairs must be unique")
    if actual_keys != expected_keys:
        missing = sorted(expected_keys - actual_keys)
        unexpected = sorted(actual_keys - expected_keys)
        raise ValueError(
            "evaluations must cover the frozen unit/candidate product; "
            f"missing={missing}, unexpected={unexpected}"
        )
    candidate_index = freeze.candidate_by_id
    unit_index = freeze.unit_by_id
    for value in values:
        if value.freeze_id != freeze.freeze_id:
            raise ValueError("unit evaluation identifies a different freeze")
        if value.opening.opening_id != opening.opening_id:
            raise ValueError("unit evaluations must share the frozen target opening")
        if value.metric_contract.metric_contract_id != (
            freeze.metric_contract.metric_contract_id
        ):
            raise ValueError("unit evaluation uses a different metric contract")
        if value.policy.policy_id != freeze.policy.policy_id:
            raise ValueError("unit evaluation uses a different promotion policy")
        if value.unit.unit_binding_id != (
            unit_index[value.unit.unit_id].unit_binding_id
        ):
            raise ValueError("unit evaluation binding differs from the freeze")
        if value.candidate.candidate_binding_id != (
            candidate_index[value.candidate.candidate_id].candidate_binding_id
        ):
            raise ValueError("candidate evaluation binding differs from the freeze")

    evaluation_index = {
        (value.unit.unit_id, value.candidate.candidate_id): value for value in values
    }
    candidate_results: list[ProspectiveV2CandidateResultV1] = []
    for candidate in freeze.candidates[1:]:
        endpoint_metrics = tuple(
            _evaluate_endpoint(
                candidate,
                endpoint,
                tuple(
                    evaluation_index[(unit.unit_id, candidate.candidate_id)]
                    for unit in freeze.evaluation_units
                    if unit.endpoint == endpoint
                ),
                freeze.policy,
            )
            for endpoint in DECISION_TRACE_ENDPOINTS
        )
        reasons = tuple(
            f"{metric.endpoint}:{reason}"
            for metric in endpoint_metrics
            for reason in metric.reasons
        )
        candidate_results.append(
            ProspectiveV2CandidateResultV1(
                candidate_id=candidate.candidate_id,
                candidate_kind=candidate.candidate_kind,
                candidate_configuration_artifact_id=(
                    candidate.configuration_artifact_id
                ),
                endpoint_metrics=endpoint_metrics,
                accepted=not reasons,
                reasons=reasons,
            )
        )
    accepted_ids = {
        result.candidate_id for result in candidate_results if result.accepted
    }
    selected = freeze.baseline_candidate
    for candidate in freeze.candidates[1:]:
        if candidate.candidate_id in accepted_ids:
            selected = candidate
    ordered_evaluation_ids = tuple(
        evaluation_index[(unit.unit_id, candidate.candidate_id)].evaluation_id
        for unit in freeze.evaluation_units
        for candidate in freeze.candidates[1:]
    )
    return ProspectiveV2PromotionResultV1(
        freeze_id=freeze.freeze_id,
        opening_id=opening.opening_id,
        baseline_candidate_id=freeze.baseline_candidate.candidate_id,
        baseline_configuration_artifact_id=(
            freeze.baseline_candidate.configuration_artifact_id
        ),
        selected_candidate_id=selected.candidate_id,
        selected_candidate_kind=selected.candidate_kind,
        selected_configuration_artifact_id=selected.configuration_artifact_id,
        candidate_results=tuple(candidate_results),
        evaluation_ids=ordered_evaluation_ids,
        selection_panel_role=PROSPECTIVE_V2_SELECTION_PANEL_ROLE,
        unbiased_post_selection_performance_claimed=False,
        independent_confirmation_required=True,
        target_outcomes_used=True,
        metadata={} if metadata is None else metadata,
    )


def validate_prospective_v2_promotion_result_v1(
    result: ProspectiveV2PromotionResultV1,
    freeze: ProspectiveV2PromotionFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
) -> ProspectiveV2PromotionResultV1:
    """Recompute and require exact equality with a published selection result."""

    if type(result) is not ProspectiveV2PromotionResultV1:
        raise ValueError("result has the wrong type")
    expected = evaluate_prospective_v2_promotion_v1(
        freeze,
        opening,
        evaluations,
        metadata=result.metadata,
    )
    if result != expected or result.result_id != expected.result_id:
        raise ValueError("promotion result does not match its bound source evidence")
    return result


def write_prospective_v2_promotion_freeze(
    path: str | Path,
    freeze: ProspectiveV2PromotionFreezeV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a target-free promotion freeze."""

    if type(freeze) is not ProspectiveV2PromotionFreezeV1:
        raise ValueError("freeze has the wrong type")
    atomic_write_json(path, freeze.as_dict(), overwrite=overwrite)


def write_prospective_v2_target_opening(
    path: str | Path,
    opening: ProspectiveV2TargetOpeningV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish the one target-opening inventory record."""

    if type(opening) is not ProspectiveV2TargetOpeningV1:
        raise ValueError("opening has the wrong type")
    atomic_write_json(path, opening.as_dict(), overwrite=overwrite)


def write_prospective_v2_promotion_result(
    path: str | Path,
    result: ProspectiveV2PromotionResultV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a recomputable candidate-selection result."""

    if type(result) is not ProspectiveV2PromotionResultV1:
        raise ValueError("result has the wrong type")
    atomic_write_json(path, result.as_dict(), overwrite=overwrite)


__all__ = [
    "PROSPECTIVE_V2_CANDIDATE_KINDS",
    "PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION",
    "PROSPECTIVE_V2_SELECTION_PANEL_ROLE",
    "ProspectiveV2CandidateResultV1",
    "ProspectiveV2CandidateV1",
    "ProspectiveV2EndpointMetricsV1",
    "ProspectiveV2EvaluationUnitV1",
    "ProspectiveV2MetricContractV1",
    "ProspectiveV2PromotionFreezeV1",
    "ProspectiveV2PromotionPolicyV1",
    "ProspectiveV2PromotionResultV1",
    "ProspectiveV2TargetOpeningV1",
    "ProspectiveV2UnitEvaluationV1",
    "ProspectiveV2UnitMetricValuesV1",
    "build_prospective_v2_target_opening_v1",
    "build_prospective_v2_unit_evaluation_v1",
    "evaluate_prospective_v2_promotion_v1",
    "validate_prospective_v2_promotion_result_v1",
    "write_prospective_v2_promotion_freeze",
    "write_prospective_v2_promotion_result",
    "write_prospective_v2_target_opening",
]
