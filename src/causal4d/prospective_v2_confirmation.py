"""Independent confirmation for a fixed prospective Causal4D V2 candidate.

The candidate-selection panel may choose one non-baseline configuration, but it
cannot support an unbiased post-selection performance claim. This module freezes
and evaluates a disjoint confirmation panel for that already selected candidate.
It never reopens the candidate ladder and falls back to the exact registered
baseline when the fixed candidate does not pass every confirmation endpoint.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from causal4d._prospective_v2_promotion_evidence import (
    PROSPECTIVE_V2_CANDIDATE_KINDS,
    ProspectiveV2CandidateV1,
    ProspectiveV2EvaluationUnitV1,
    ProspectiveV2MetricContractV1,
    ProspectiveV2PromotionFreezeV1,
    ProspectiveV2PromotionPolicyV1,
    ProspectiveV2TargetOpeningV1,
    ProspectiveV2UnitEvaluationV1,
    ProspectiveV2UnitMetricValuesV1,
    _canonical_sha256,
    _finite_float,
    _finite_nonnegative_float,
    _rate,
    _require_bool,
    _require_nonempty_string,
    _require_sha256,
    _validated_source_metadata,
    _validated_unique_strings,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.decision_trace import DECISION_TRACE_ENDPOINTS, UnifiedDecisionTrace
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.prospective_v2_promotion import (
    ProspectiveV2EndpointMetricsV1,
    ProspectiveV2PromotionResultV1,
    validate_prospective_v2_promotion_result_v1,
)


PROSPECTIVE_V2_CONFIRMATION_SCHEMA_VERSION = 1
PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE = "independent_confirmation"

_CONFIRMATION_TRACE_METADATA = {
    "confirmation_freeze_id": "confirmation_freeze_id",
    "confirmation_selection_result_id": "selection_result_id",
    "confirmation_panel_role": "confirmation_panel_role",
}


def _mean(values: Sequence[float], *, name: str) -> float:
    if not values:
        raise ValueError(f"{name} requires at least one value")
    result = float(np.mean(np.asarray(values, dtype=float)))
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _ordered_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _unit_identities(
    units: Sequence[ProspectiveV2EvaluationUnitV1],
) -> tuple[
    set[str],
    set[str],
    set[str],
    set[tuple[str, str]],
    set[tuple[str, str]],
]:
    return (
        {unit.unit_id for unit in units},
        {unit.unit_binding_id for unit in units},
        {unit.target_artifact_id for unit in units},
        {(unit.endpoint, unit.independent_group_id) for unit in units},
        {(unit.protocol_id, unit.session_id) for unit in units},
    )


def _require_confirmation_trace_metadata(
    trace: UnifiedDecisionTrace,
    freeze: ProspectiveV2ConfirmationFreezeV1,
) -> None:
    expected_metadata = {
        "confirmation_freeze_id": freeze.confirmation_freeze_id,
        "confirmation_selection_result_id": freeze.selection_result_id,
        "confirmation_panel_role": PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE,
    }
    for key, expected in expected_metadata.items():
        if trace.metadata.get(key) != expected:
            field = _CONFIRMATION_TRACE_METADATA[key]
            raise ValueError(f"decision trace {field} binding changed")


@dataclass(frozen=True)
class ProspectiveV2ConfirmationFreezeV1:
    """Target-free freeze for a fixed-candidate independent confirmation panel."""

    experiment_id: str
    selection_result_id: str
    selection_freeze_id: str
    selection_opening_id: str
    selection_evaluation_ids: tuple[str, ...]
    selection_trace_ids: tuple[str, ...]
    selection_metric_values_ids: tuple[str, ...]
    selection_scoring_run_artifact_ids: tuple[str, ...]
    stack_lock_id: str
    target_access_seal_id: str
    baseline_candidate: ProspectiveV2CandidateV1
    selected_candidate: ProspectiveV2CandidateV1
    selection_units: tuple[ProspectiveV2EvaluationUnitV1, ...]
    evaluation_units: tuple[ProspectiveV2EvaluationUnitV1, ...]
    metric_contract: ProspectiveV2MetricContractV1
    policy: ProspectiveV2PromotionPolicyV1
    bound_artifact_ids: tuple[str, ...]
    confirmation_target_outcomes_used: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        experiment_id = _require_nonempty_string(
            self.experiment_id,
            name="experiment_id",
        )
        selection_result_id = _require_sha256(
            self.selection_result_id,
            name="selection_result_id",
        )
        selection_freeze_id = _require_sha256(
            self.selection_freeze_id,
            name="selection_freeze_id",
        )
        selection_opening_id = _require_sha256(
            self.selection_opening_id,
            name="selection_opening_id",
        )
        selection_evaluation_ids = _validated_unique_strings(
            self.selection_evaluation_ids,
            name="selection_evaluation_ids",
            require_sha256=True,
        )
        selection_trace_ids = _validated_unique_strings(
            self.selection_trace_ids,
            name="selection_trace_ids",
            require_sha256=True,
        )
        selection_metric_values_ids = _validated_unique_strings(
            self.selection_metric_values_ids,
            name="selection_metric_values_ids",
            require_sha256=True,
        )
        selection_scoring_run_ids = _validated_unique_strings(
            self.selection_scoring_run_artifact_ids,
            name="selection_scoring_run_artifact_ids",
            require_sha256=True,
        )
        stack_lock_id = _require_sha256(self.stack_lock_id, name="stack_lock_id")
        seal_id = _require_sha256(
            self.target_access_seal_id,
            name="target_access_seal_id",
        )
        if type(self.baseline_candidate) is not ProspectiveV2CandidateV1:
            raise ValueError("baseline_candidate has the wrong type")
        if type(self.selected_candidate) is not ProspectiveV2CandidateV1:
            raise ValueError("selected_candidate has the wrong type")
        if self.baseline_candidate.candidate_kind != PROSPECTIVE_V2_CANDIDATE_KINDS[0]:
            raise ValueError("baseline_candidate must be the registered baseline")
        if self.selected_candidate.candidate_kind == PROSPECTIVE_V2_CANDIDATE_KINDS[0]:
            raise ValueError(
                "independent confirmation requires a non-baseline candidate"
            )
        if self.baseline_candidate.candidate_id == self.selected_candidate.candidate_id:
            raise ValueError("baseline and selected candidate IDs must differ")
        if (
            self.baseline_candidate.configuration_artifact_id
            == self.selected_candidate.configuration_artifact_id
        ):
            raise ValueError("baseline and selected configurations must differ")

        selection_units = tuple(self.selection_units)
        confirmation_units = tuple(self.evaluation_units)
        for name, units in (
            ("selection_units", selection_units),
            ("evaluation_units", confirmation_units),
        ):
            if not units or any(
                type(unit) is not ProspectiveV2EvaluationUnitV1 for unit in units
            ):
                raise ValueError(
                    f"{name} must contain ProspectiveV2EvaluationUnitV1 values"
                )
        selection_seals = {unit.target_access_seal_id for unit in selection_units}
        if len(selection_seals) != 1:
            raise ValueError("selection units must share one target-access seal")
        if seal_id in selection_seals:
            raise ValueError("confirmation must use a new target-access seal")
        if any(unit.target_access_seal_id != seal_id for unit in confirmation_units):
            raise ValueError("every confirmation unit must bind the new access seal")

        selection_identity = _unit_identities(selection_units)
        confirmation_identity = _unit_identities(confirmation_units)
        identity_names = (
            "unit IDs",
            "unit bindings",
            "target artifacts",
            "endpoint independence groups",
            "protocol sessions",
        )
        for name, selected_values, confirmation_values in zip(
            identity_names,
            selection_identity,
            confirmation_identity,
            strict=True,
        ):
            overlap = sorted(selected_values & confirmation_values)
            if overlap:
                raise ValueError(
                    f"confirmation panel reuses selection-panel {name}: {overlap}"
                )
        for name, values in zip(
            identity_names[:4],
            confirmation_identity[:4],
            strict=True,
        ):
            if len(values) != len(confirmation_units):
                raise ValueError(f"confirmation {name} must be unique")

        if type(self.metric_contract) is not ProspectiveV2MetricContractV1:
            raise ValueError("metric_contract has the wrong type")
        if type(self.policy) is not ProspectiveV2PromotionPolicyV1:
            raise ValueError("policy has the wrong type")
        for endpoint in DECISION_TRACE_ENDPOINTS:
            count = sum(unit.endpoint == endpoint for unit in confirmation_units)
            if count < self.policy.minimum_units_per_endpoint:
                raise ValueError(
                    f"confirmation endpoint {endpoint!r} has insufficient units"
                )

        bound_ids = _validated_unique_strings(
            self.bound_artifact_ids,
            name="bound_artifact_ids",
            require_sha256=True,
        )
        required_ids = {
            selection_result_id,
            selection_freeze_id,
            selection_opening_id,
            *selection_evaluation_ids,
            *selection_trace_ids,
            *selection_metric_values_ids,
            *selection_scoring_run_ids,
            stack_lock_id,
            seal_id,
            self.metric_contract.scoring_implementation_artifact_id,
            self.baseline_candidate.configuration_artifact_id,
            self.selected_candidate.configuration_artifact_id,
            *(unit.factual_context_artifact_id for unit in confirmation_units),
            *(unit.counterfactual_query_artifact_id for unit in confirmation_units),
        }
        missing = sorted(required_ids - set(bound_ids))
        if missing:
            raise ValueError(
                "bound_artifact_ids do not cover the confirmation sources; "
                f"missing={missing}"
            )
        if _require_bool(
            self.confirmation_target_outcomes_used,
            name="confirmation_target_outcomes_used",
        ):
            raise ValueError(
                "confirmation freeze must predate confirmation target access"
            )
        metadata = _validated_source_metadata(
            self.metadata,
            name="confirmation-freeze metadata",
        )
        object.__setattr__(self, "experiment_id", experiment_id)
        object.__setattr__(self, "selection_result_id", selection_result_id)
        object.__setattr__(self, "selection_freeze_id", selection_freeze_id)
        object.__setattr__(self, "selection_opening_id", selection_opening_id)
        object.__setattr__(
            self,
            "selection_evaluation_ids",
            selection_evaluation_ids,
        )
        object.__setattr__(self, "selection_trace_ids", selection_trace_ids)
        object.__setattr__(
            self,
            "selection_metric_values_ids",
            selection_metric_values_ids,
        )
        object.__setattr__(
            self,
            "selection_scoring_run_artifact_ids",
            selection_scoring_run_ids,
        )
        object.__setattr__(self, "stack_lock_id", stack_lock_id)
        object.__setattr__(self, "target_access_seal_id", seal_id)
        object.__setattr__(self, "selection_units", selection_units)
        object.__setattr__(self, "evaluation_units", confirmation_units)
        object.__setattr__(self, "bound_artifact_ids", bound_ids)
        object.__setattr__(self, "metadata", metadata)

    @property
    def unit_by_id(self) -> dict[str, ProspectiveV2EvaluationUnitV1]:
        return {unit.unit_id: unit for unit in self.evaluation_units}

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_CONFIRMATION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2ConfirmationFreezeV1",
            "experiment_id": self.experiment_id,
            "selection_result_id": self.selection_result_id,
            "selection_freeze_id": self.selection_freeze_id,
            "selection_opening_id": self.selection_opening_id,
            "selection_evaluation_ids": list(self.selection_evaluation_ids),
            "selection_trace_ids": list(self.selection_trace_ids),
            "selection_metric_values_ids": list(self.selection_metric_values_ids),
            "selection_scoring_run_artifact_ids": list(
                self.selection_scoring_run_artifact_ids
            ),
            "stack_lock_id": self.stack_lock_id,
            "target_access_seal_id": self.target_access_seal_id,
            "baseline_candidate": self.baseline_candidate.as_dict(),
            "selected_candidate": self.selected_candidate.as_dict(),
            "selection_units": [unit.as_dict() for unit in self.selection_units],
            "evaluation_units": [unit.as_dict() for unit in self.evaluation_units],
            "metric_contract": self.metric_contract.as_dict(),
            "policy": self.policy.as_dict(),
            "bound_artifact_ids": list(self.bound_artifact_ids),
            "panel_role": PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE,
            "candidate_selection_performed": False,
            "selection_panel_outcomes_used": True,
            "confirmation_target_outcomes_used": False,
            "metadata": plain_json(self.metadata),
        }

    @property
    def confirmation_freeze_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {
            **self._payload(),
            "confirmation_freeze_id": self.confirmation_freeze_id,
        }


@dataclass(frozen=True)
class ProspectiveV2ConfirmationResultV1:
    """Independent fixed-candidate confirmation with exact baseline fallback."""

    confirmation_freeze_id: str
    opening_id: str
    selection_result_id: str
    baseline_candidate_id: str
    baseline_configuration_artifact_id: str
    candidate_id: str
    candidate_kind: str
    candidate_configuration_artifact_id: str
    deployed_candidate_id: str
    deployed_configuration_artifact_id: str
    endpoint_metrics: tuple[ProspectiveV2EndpointMetricsV1, ...]
    evaluation_ids: tuple[str, ...]
    confirmation_passed: bool
    reasons: tuple[str, ...]
    exact_baseline_fallback_verified: bool = True
    confirmation_target_outcomes_used: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        confirmation_freeze_id = _require_sha256(
            self.confirmation_freeze_id,
            name="confirmation_freeze_id",
        )
        opening_id = _require_sha256(self.opening_id, name="opening_id")
        selection_result_id = _require_sha256(
            self.selection_result_id,
            name="selection_result_id",
        )
        baseline_candidate_id = _require_nonempty_string(
            self.baseline_candidate_id,
            name="baseline_candidate_id",
        )
        baseline_configuration_id = _require_sha256(
            self.baseline_configuration_artifact_id,
            name="baseline_configuration_artifact_id",
        )
        candidate_id = _require_nonempty_string(self.candidate_id, name="candidate_id")
        candidate_kind = _require_nonempty_string(
            self.candidate_kind,
            name="candidate_kind",
        )
        if candidate_kind not in PROSPECTIVE_V2_CANDIDATE_KINDS[1:]:
            raise ValueError("confirmation candidate must be non-baseline")
        candidate_configuration_id = _require_sha256(
            self.candidate_configuration_artifact_id,
            name="candidate_configuration_artifact_id",
        )
        endpoint_metrics = tuple(self.endpoint_metrics)
        if tuple(metric.endpoint for metric in endpoint_metrics) != (
            DECISION_TRACE_ENDPOINTS
        ):
            raise ValueError("confirmation result must cover every endpoint in order")
        if any(metric.candidate_id != candidate_id for metric in endpoint_metrics):
            raise ValueError("endpoint metrics identify a different candidate")
        expected_reasons = tuple(
            f"{metric.endpoint}:{reason}"
            for metric in endpoint_metrics
            for reason in metric.reasons
        )
        passed = _require_bool(self.confirmation_passed, name="confirmation_passed")
        reasons = tuple(self.reasons)
        if reasons != expected_reasons or passed is not (not expected_reasons):
            raise ValueError(
                "confirmation decision must exactly match endpoint evidence"
            )
        expected_deployed = (
            (candidate_id, candidate_configuration_id)
            if passed
            else (baseline_candidate_id, baseline_configuration_id)
        )
        deployed_candidate_id = _require_nonempty_string(
            self.deployed_candidate_id,
            name="deployed_candidate_id",
        )
        deployed_configuration_id = _require_sha256(
            self.deployed_configuration_artifact_id,
            name="deployed_configuration_artifact_id",
        )
        if (deployed_candidate_id, deployed_configuration_id) != expected_deployed:
            raise ValueError(
                "confirmation deployment must use the fixed candidate or exact baseline"
            )
        evaluation_ids = _validated_unique_strings(
            self.evaluation_ids,
            name="evaluation_ids",
            require_sha256=True,
        )
        if not _require_bool(
            self.exact_baseline_fallback_verified,
            name="exact_baseline_fallback_verified",
        ):
            raise ValueError("exact baseline fallback must remain verified")
        if not _require_bool(
            self.confirmation_target_outcomes_used,
            name="confirmation_target_outcomes_used",
        ):
            raise ValueError("confirmation result must record target-outcome access")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="confirmation-result metadata must contain finite JSON data",
        )
        object.__setattr__(
            self,
            "confirmation_freeze_id",
            confirmation_freeze_id,
        )
        object.__setattr__(self, "opening_id", opening_id)
        object.__setattr__(self, "selection_result_id", selection_result_id)
        object.__setattr__(self, "baseline_candidate_id", baseline_candidate_id)
        object.__setattr__(
            self,
            "baseline_configuration_artifact_id",
            baseline_configuration_id,
        )
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "candidate_kind", candidate_kind)
        object.__setattr__(
            self,
            "candidate_configuration_artifact_id",
            candidate_configuration_id,
        )
        object.__setattr__(self, "deployed_candidate_id", deployed_candidate_id)
        object.__setattr__(
            self,
            "deployed_configuration_artifact_id",
            deployed_configuration_id,
        )
        object.__setattr__(self, "endpoint_metrics", endpoint_metrics)
        object.__setattr__(self, "evaluation_ids", evaluation_ids)
        object.__setattr__(self, "confirmation_passed", passed)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "metadata", metadata)

    @property
    def fixed_candidate_confirmation_gate_passed(self) -> bool:
        return self.confirmation_passed

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": PROSPECTIVE_V2_CONFIRMATION_SCHEMA_VERSION,
            "artifact_kind": "Causal4DProspectiveV2ConfirmationResultV1",
            "confirmation_freeze_id": self.confirmation_freeze_id,
            "opening_id": self.opening_id,
            "selection_result_id": self.selection_result_id,
            "baseline_candidate_id": self.baseline_candidate_id,
            "baseline_configuration_artifact_id": (
                self.baseline_configuration_artifact_id
            ),
            "candidate_id": self.candidate_id,
            "candidate_kind": self.candidate_kind,
            "candidate_configuration_artifact_id": (
                self.candidate_configuration_artifact_id
            ),
            "deployed_candidate_id": self.deployed_candidate_id,
            "deployed_configuration_artifact_id": (
                self.deployed_configuration_artifact_id
            ),
            "endpoint_metrics": [metric.as_dict() for metric in self.endpoint_metrics],
            "evaluation_ids": list(self.evaluation_ids),
            "confirmation_passed": self.confirmation_passed,
            "reasons": list(self.reasons),
            "panel_role": PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE,
            "candidate_selection_performed": False,
            "selection_panel_performance_reused": False,
            "independent_panel_verified": True,
            "fixed_candidate_confirmation_gate_passed": (
                self.fixed_candidate_confirmation_gate_passed
            ),
            "exact_baseline_fallback_verified": (self.exact_baseline_fallback_verified),
            "confirmation_target_outcomes_used": (
                self.confirmation_target_outcomes_used
            ),
            "metadata": plain_json(self.metadata),
        }

    @property
    def result_id(self) -> str:
        return _canonical_sha256(self._payload())

    def as_dict(self) -> dict[str, Any]:
        return {**self._payload(), "result_id": self.result_id}


def build_prospective_v2_confirmation_freeze_v1(
    selection_result: ProspectiveV2PromotionResultV1,
    selection_freeze: ProspectiveV2PromotionFreezeV1,
    selection_opening: ProspectiveV2TargetOpeningV1,
    selection_evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
    *,
    experiment_id: str,
    target_access_seal_id: str,
    evaluation_units: Sequence[ProspectiveV2EvaluationUnitV1],
    additional_bound_artifact_ids: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveV2ConfirmationFreezeV1:
    """Freeze a disjoint panel for the exact candidate selected previously."""

    validate_prospective_v2_promotion_result_v1(
        selection_result,
        selection_freeze,
        selection_opening,
        selection_evaluations,
    )
    selected = selection_freeze.candidate_by_id.get(
        selection_result.selected_candidate_id
    )
    if selected is None:
        raise ValueError("selection result candidate is absent from its freeze")
    if selected.candidate_kind == PROSPECTIVE_V2_CANDIDATE_KINDS[0]:
        raise ValueError("selection panel did not select a candidate to confirm")
    if (
        selected.candidate_kind,
        selected.configuration_artifact_id,
    ) != (
        selection_result.selected_candidate_kind,
        selection_result.selected_configuration_artifact_id,
    ):
        raise ValueError("selection result candidate binding changed")
    selection_values = tuple(selection_evaluations)
    confirmation_units = tuple(evaluation_units)
    bound_ids = _ordered_unique(
        (
            selection_result.result_id,
            selection_freeze.freeze_id,
            selection_opening.opening_id,
            *selection_result.evaluation_ids,
            *(value.trace.trace_id for value in selection_values),
            *(value.metric_values.metric_values_id for value in selection_values),
            *(
                value.metric_values.scoring_run_artifact_id
                for value in selection_values
            ),
            selection_freeze.stack_lock_id,
            target_access_seal_id,
            selection_freeze.metric_contract.scoring_implementation_artifact_id,
            selection_freeze.baseline_candidate.configuration_artifact_id,
            selected.configuration_artifact_id,
            *(unit.factual_context_artifact_id for unit in confirmation_units),
            *(unit.counterfactual_query_artifact_id for unit in confirmation_units),
            *additional_bound_artifact_ids,
        )
    )
    return ProspectiveV2ConfirmationFreezeV1(
        experiment_id=experiment_id,
        selection_result_id=selection_result.result_id,
        selection_freeze_id=selection_freeze.freeze_id,
        selection_opening_id=selection_opening.opening_id,
        selection_evaluation_ids=selection_result.evaluation_ids,
        selection_trace_ids=tuple(value.trace.trace_id for value in selection_values),
        selection_metric_values_ids=tuple(
            value.metric_values.metric_values_id for value in selection_values
        ),
        selection_scoring_run_artifact_ids=tuple(
            value.metric_values.scoring_run_artifact_id for value in selection_values
        ),
        stack_lock_id=selection_freeze.stack_lock_id,
        target_access_seal_id=target_access_seal_id,
        baseline_candidate=selection_freeze.baseline_candidate,
        selected_candidate=selected,
        selection_units=selection_freeze.evaluation_units,
        evaluation_units=confirmation_units,
        metric_contract=selection_freeze.metric_contract,
        policy=selection_freeze.policy,
        bound_artifact_ids=bound_ids,
        confirmation_target_outcomes_used=False,
        metadata={} if metadata is None else metadata,
    )


def validate_prospective_v2_confirmation_freeze_v1(
    freeze: ProspectiveV2ConfirmationFreezeV1,
    selection_result: ProspectiveV2PromotionResultV1,
    selection_freeze: ProspectiveV2PromotionFreezeV1,
    selection_opening: ProspectiveV2TargetOpeningV1,
    selection_evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
) -> ProspectiveV2ConfirmationFreezeV1:
    """Recompute and require the exact confirmation freeze from its sources."""

    if type(freeze) is not ProspectiveV2ConfirmationFreezeV1:
        raise ValueError("freeze has the wrong type")
    expected = build_prospective_v2_confirmation_freeze_v1(
        selection_result,
        selection_freeze,
        selection_opening,
        selection_evaluations,
        experiment_id=freeze.experiment_id,
        target_access_seal_id=freeze.target_access_seal_id,
        evaluation_units=freeze.evaluation_units,
        additional_bound_artifact_ids=freeze.bound_artifact_ids,
        metadata=freeze.metadata,
    )
    if freeze != expected or freeze.confirmation_freeze_id != (
        expected.confirmation_freeze_id
    ):
        raise ValueError("confirmation freeze does not match its selection evidence")
    return freeze


def build_prospective_v2_confirmation_opening_v1(
    freeze: ProspectiveV2ConfirmationFreezeV1,
    *,
    opened_at_utc: str,
    opened_by: str,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveV2TargetOpeningV1:
    """Open exactly the disjoint confirmation inventory registered by ``freeze``."""

    if type(freeze) is not ProspectiveV2ConfirmationFreezeV1:
        raise ValueError("freeze has the wrong type")
    return ProspectiveV2TargetOpeningV1(
        freeze_id=freeze.confirmation_freeze_id,
        target_access_seal_id=freeze.target_access_seal_id,
        target_artifact_ids=tuple(
            unit.target_artifact_id for unit in freeze.evaluation_units
        ),
        opened_at_utc=opened_at_utc,
        opened_by=opened_by,
        target_outcomes_used=True,
        metadata={} if metadata is None else metadata,
    )


def validate_prospective_v2_confirmation_opening_v1(
    opening: ProspectiveV2TargetOpeningV1,
    freeze: ProspectiveV2ConfirmationFreezeV1,
) -> ProspectiveV2TargetOpeningV1:
    """Require an opening to equal the complete frozen confirmation inventory."""

    if type(opening) is not ProspectiveV2TargetOpeningV1:
        raise ValueError("opening has the wrong type")
    if type(freeze) is not ProspectiveV2ConfirmationFreezeV1:
        raise ValueError("freeze has the wrong type")
    expected_target_ids = tuple(
        unit.target_artifact_id for unit in freeze.evaluation_units
    )
    if (
        opening.freeze_id != freeze.confirmation_freeze_id
        or opening.target_access_seal_id != freeze.target_access_seal_id
        or opening.target_artifact_ids != expected_target_ids
    ):
        raise ValueError("confirmation opening does not match its frozen inventory")
    return opening


def build_prospective_v2_confirmation_unit_evaluation_v1(
    freeze: ProspectiveV2ConfirmationFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    *,
    unit_id: str,
    trace: UnifiedDecisionTrace,
    metric_values: ProspectiveV2UnitMetricValuesV1,
) -> ProspectiveV2UnitEvaluationV1:
    """Bind one fixed-candidate evaluation to the independent panel."""

    if type(freeze) is not ProspectiveV2ConfirmationFreezeV1:
        raise ValueError("freeze has the wrong type")
    unit = freeze.unit_by_id.get(unit_id)
    if unit is None:
        raise ValueError("unit_id is not registered by the confirmation freeze")
    validate_prospective_v2_confirmation_opening_v1(opening, freeze)
    if type(trace) is not UnifiedDecisionTrace:
        raise ValueError("trace has the wrong type")
    _require_confirmation_trace_metadata(trace, freeze)
    if trace.trace_id in freeze.selection_trace_ids:
        raise ValueError("confirmation reuses a selection-panel decision trace")
    if metric_values.metric_values_id in freeze.selection_metric_values_ids:
        raise ValueError("confirmation reuses selection-panel metric values")
    if (
        metric_values.scoring_run_artifact_id
        in freeze.selection_scoring_run_artifact_ids
    ):
        raise ValueError("confirmation reuses a selection-panel scoring run")
    return ProspectiveV2UnitEvaluationV1(
        freeze_id=freeze.confirmation_freeze_id,
        stack_lock_id=freeze.stack_lock_id,
        unit=unit,
        candidate=freeze.selected_candidate,
        opening=opening,
        metric_contract=freeze.metric_contract,
        policy=freeze.policy,
        trace=trace,
        metric_values=metric_values,
    )


def _evaluate_confirmation_endpoint(
    freeze: ProspectiveV2ConfirmationFreezeV1,
    endpoint: str,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
) -> ProspectiveV2EndpointMetricsV1:
    policy = freeze.policy
    unit_count = len(evaluations)
    accepted_count = sum(value.candidate_selected for value in evaluations)
    harmful_count = sum(value.harmful_accepted_update for value in evaluations)
    accepted_rate = accepted_count / unit_count
    harmful_rate = harmful_count / accepted_count if accepted_count else 0.0
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
    if harmful_rate > policy.maximum_harmful_accepted_update_rate:
        reasons.append("harmful_accepted_update_rate_exceeds_limit")
    if fallback_rate > policy.maximum_fallback_rate:
        reasons.append("fallback_rate_exceeds_limit")
    return ProspectiveV2EndpointMetricsV1(
        candidate_id=freeze.selected_candidate.candidate_id,
        endpoint=endpoint,
        unit_count=unit_count,
        mean_log_score_gain=_finite_float(
            mean_log_gain,
            name="mean_log_score_gain",
        ),
        mean_brier_change=_finite_float(
            mean_brier_change,
            name="mean_brier_change",
        ),
        mean_trajectory_regret_m=_finite_float(
            mean_regret,
            name="mean_trajectory_regret_m",
        ),
        mean_coverage_error=_finite_nonnegative_float(
            mean_coverage_error,
            name="mean_coverage_error",
        ),
        mean_interval_width_m=_finite_nonnegative_float(
            mean_width,
            name="mean_interval_width_m",
        ),
        mean_interval_width_ratio=_finite_nonnegative_float(
            mean_width_ratio,
            name="mean_interval_width_ratio",
        ),
        accepted_update_rate=_rate(accepted_rate, name="accepted_update_rate"),
        harmful_accepted_update_rate=_rate(
            harmful_rate,
            name="harmful_accepted_update_rate",
        ),
        fallback_rate=_rate(fallback_rate, name="fallback_rate"),
        accepted=not reasons,
        reasons=tuple(reasons),
    )


def evaluate_prospective_v2_confirmation_v1(
    freeze: ProspectiveV2ConfirmationFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ProspectiveV2ConfirmationResultV1:
    """Evaluate the fixed candidate on the complete independent panel."""

    if type(freeze) is not ProspectiveV2ConfirmationFreezeV1:
        raise ValueError("freeze has the wrong type")
    validate_prospective_v2_confirmation_opening_v1(opening, freeze)
    values = tuple(evaluations)
    if not values or any(
        type(value) is not ProspectiveV2UnitEvaluationV1 for value in values
    ):
        raise ValueError("evaluations must contain bound unit evaluations")
    expected_unit_ids = tuple(unit.unit_id for unit in freeze.evaluation_units)
    actual_unit_ids = tuple(value.unit.unit_id for value in values)
    if len(set(actual_unit_ids)) != len(actual_unit_ids):
        raise ValueError("confirmation evaluation units must be unique")
    if set(actual_unit_ids) != set(expected_unit_ids):
        missing = sorted(set(expected_unit_ids) - set(actual_unit_ids))
        unexpected = sorted(set(actual_unit_ids) - set(expected_unit_ids))
        raise ValueError(
            "evaluations must cover the frozen confirmation units; "
            f"missing={missing}, unexpected={unexpected}"
        )
    unit_index = freeze.unit_by_id
    evaluation_index = {value.unit.unit_id: value for value in values}
    for value in values:
        if value.freeze_id != freeze.confirmation_freeze_id:
            raise ValueError(
                "unit evaluation identifies a different confirmation freeze"
            )
        if value.opening.opening_id != opening.opening_id:
            raise ValueError("confirmation evaluations must share one opening")
        if value.stack_lock_id != freeze.stack_lock_id:
            raise ValueError("confirmation evaluation uses a different stack lock")
        if value.candidate.candidate_binding_id != (
            freeze.selected_candidate.candidate_binding_id
        ):
            raise ValueError("confirmation evaluation changed the fixed candidate")
        if value.unit.unit_binding_id != (
            unit_index[value.unit.unit_id].unit_binding_id
        ):
            raise ValueError("confirmation evaluation changed a unit binding")
        if value.metric_contract.metric_contract_id != (
            freeze.metric_contract.metric_contract_id
        ):
            raise ValueError("confirmation evaluation changed the metric contract")
        if value.policy.policy_id != freeze.policy.policy_id:
            raise ValueError("confirmation evaluation changed the promotion policy")
        _require_confirmation_trace_metadata(value.trace, freeze)
        if value.trace.trace_id in freeze.selection_trace_ids:
            raise ValueError("confirmation reuses a selection-panel decision trace")
        if value.metric_values.metric_values_id in freeze.selection_metric_values_ids:
            raise ValueError("confirmation reuses selection-panel metric values")
        if (
            value.metric_values.scoring_run_artifact_id
            in freeze.selection_scoring_run_artifact_ids
        ):
            raise ValueError("confirmation reuses a selection-panel scoring run")

    endpoint_metrics = tuple(
        _evaluate_confirmation_endpoint(
            freeze,
            endpoint,
            tuple(
                evaluation_index[unit.unit_id]
                for unit in freeze.evaluation_units
                if unit.endpoint == endpoint
            ),
        )
        for endpoint in DECISION_TRACE_ENDPOINTS
    )
    reasons = tuple(
        f"{metric.endpoint}:{reason}"
        for metric in endpoint_metrics
        for reason in metric.reasons
    )
    passed = not reasons
    deployed = freeze.selected_candidate if passed else freeze.baseline_candidate
    ordered_evaluation_ids = tuple(
        evaluation_index[unit.unit_id].evaluation_id for unit in freeze.evaluation_units
    )
    return ProspectiveV2ConfirmationResultV1(
        confirmation_freeze_id=freeze.confirmation_freeze_id,
        opening_id=opening.opening_id,
        selection_result_id=freeze.selection_result_id,
        baseline_candidate_id=freeze.baseline_candidate.candidate_id,
        baseline_configuration_artifact_id=(
            freeze.baseline_candidate.configuration_artifact_id
        ),
        candidate_id=freeze.selected_candidate.candidate_id,
        candidate_kind=freeze.selected_candidate.candidate_kind,
        candidate_configuration_artifact_id=(
            freeze.selected_candidate.configuration_artifact_id
        ),
        deployed_candidate_id=deployed.candidate_id,
        deployed_configuration_artifact_id=deployed.configuration_artifact_id,
        endpoint_metrics=endpoint_metrics,
        evaluation_ids=ordered_evaluation_ids,
        confirmation_passed=passed,
        reasons=reasons,
        exact_baseline_fallback_verified=True,
        confirmation_target_outcomes_used=True,
        metadata={} if metadata is None else metadata,
    )


def validate_prospective_v2_confirmation_result_v1(
    result: ProspectiveV2ConfirmationResultV1,
    freeze: ProspectiveV2ConfirmationFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    evaluations: Sequence[ProspectiveV2UnitEvaluationV1],
) -> ProspectiveV2ConfirmationResultV1:
    """Recompute and require exact equality with a confirmation result."""

    if type(result) is not ProspectiveV2ConfirmationResultV1:
        raise ValueError("result has the wrong type")
    expected = evaluate_prospective_v2_confirmation_v1(
        freeze,
        opening,
        evaluations,
        metadata=result.metadata,
    )
    if result != expected or result.result_id != expected.result_id:
        raise ValueError("confirmation result does not match its bound evidence")
    return result


def write_prospective_v2_confirmation_opening(
    path: str | Path,
    freeze: ProspectiveV2ConfirmationFreezeV1,
    opening: ProspectiveV2TargetOpeningV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish an opening after exact inventory validation."""

    validate_prospective_v2_confirmation_opening_v1(opening, freeze)
    atomic_write_json(path, opening.as_dict(), overwrite=overwrite)


def write_prospective_v2_confirmation_freeze(
    path: str | Path,
    freeze: ProspectiveV2ConfirmationFreezeV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a validated confirmation freeze."""

    if type(freeze) is not ProspectiveV2ConfirmationFreezeV1:
        raise ValueError("freeze has the wrong type")
    atomic_write_json(path, freeze.as_dict(), overwrite=overwrite)


def write_prospective_v2_confirmation_result(
    path: str | Path,
    result: ProspectiveV2ConfirmationResultV1,
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a recomputable confirmation result."""

    if type(result) is not ProspectiveV2ConfirmationResultV1:
        raise ValueError("result has the wrong type")
    atomic_write_json(path, result.as_dict(), overwrite=overwrite)


__all__ = [
    "PROSPECTIVE_V2_CONFIRMATION_PANEL_ROLE",
    "PROSPECTIVE_V2_CONFIRMATION_SCHEMA_VERSION",
    "ProspectiveV2ConfirmationFreezeV1",
    "ProspectiveV2ConfirmationResultV1",
    "build_prospective_v2_confirmation_freeze_v1",
    "build_prospective_v2_confirmation_opening_v1",
    "build_prospective_v2_confirmation_unit_evaluation_v1",
    "evaluate_prospective_v2_confirmation_v1",
    "validate_prospective_v2_confirmation_freeze_v1",
    "validate_prospective_v2_confirmation_opening_v1",
    "validate_prospective_v2_confirmation_result_v1",
    "write_prospective_v2_confirmation_freeze",
    "write_prospective_v2_confirmation_opening",
    "write_prospective_v2_confirmation_result",
]
