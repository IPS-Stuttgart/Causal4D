"""Requested-only sensor disclosure, trace sealing, and scoring."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ._sensor_reveal_case import SensorRevealManifest, SensorRevealTruth
from ._sensor_reveal_common import (
    SENSOR_REVEAL_SCORE_CLAIM_BOUNDARY,
    SENSOR_REVEAL_TRACE_VERSION,
    SensorTerminalMode,
    _content_id,
    _digest,
    _integer,
    _name,
    _number,
)
from ._sensor_reveal_submission import (
    SensorRevealPlan,
    SensorRevealSubmission,
    validate_sensor_reveal_submission,
)
from .sensor_reveal_verifier import (
    SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY,
    policy_node_id,
    verify_sensor_reveal_plan,
    verify_sensor_reveal_trace,
)


@dataclass(frozen=True)
class SensorRevealEvent:
    """One requested sensor disclosure along the realized policy path."""

    step_index: int
    sensor_index: int
    sensor_name: str
    outcome_index: int
    outcome_name: str
    payload_sha256: str
    adapter_id: str
    policy_node_id_before: str
    policy_node_id_after: str

    def __post_init__(self) -> None:
        for field in ("step_index", "sensor_index", "outcome_index"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        for field in ("sensor_name", "outcome_name"):
            object.__setattr__(self, field, _name(getattr(self, field), field))
        for field in (
            "payload_sha256",
            "adapter_id",
            "policy_node_id_before",
            "policy_node_id_after",
        ):
            object.__setattr__(self, field, _digest(getattr(self, field), field))

    def as_dict(self) -> dict[str, object]:
        return {
            "step_index": self.step_index,
            "sensor_index": self.sensor_index,
            "sensor_name": self.sensor_name,
            "outcome_index": self.outcome_index,
            "outcome_name": self.outcome_name,
            "payload_sha256": self.payload_sha256,
            "adapter_id": self.adapter_id,
            "policy_node_id_before": self.policy_node_id_before,
            "policy_node_id_after": self.policy_node_id_after,
        }


@dataclass(frozen=True)
class SensorRevealTrace:
    """Sealed requested-only acquisition path and terminal decision."""

    case_id: str
    manifest_id: str
    submission_id: str
    plan_id: str
    events: tuple[SensorRevealEvent, ...]
    terminal_mode: SensorTerminalMode
    action_index: int
    action_name: str
    fallback_used: bool
    revealed_sensor_indices: tuple[int, ...]
    revealed_sensor_names: tuple[str, ...]
    total_sensor_cost: float
    total_sensor_risk: float
    terminal_certificate_support_count: int
    terminal_certificate_reason_code: str
    sealed_before_scoring: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _name(self.case_id, "case_id"))
        for field in ("manifest_id", "submission_id", "plan_id"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        if not all(isinstance(event, SensorRevealEvent) for event in self.events):
            raise TypeError("events must contain SensorRevealEvent values")
        if self.terminal_mode not in ("act", "fallback"):
            raise ValueError("invalid terminal_mode")
        object.__setattr__(
            self,
            "action_index",
            _integer(self.action_index, "action_index"),
        )
        object.__setattr__(
            self,
            "action_name",
            _name(self.action_name, "action_name"),
        )
        if type(self.fallback_used) is not bool:
            raise ValueError("fallback_used must be boolean")
        object.__setattr__(
            self,
            "revealed_sensor_indices",
            tuple(
                _integer(value, "revealed sensor index")
                for value in self.revealed_sensor_indices
            ),
        )
        object.__setattr__(
            self,
            "revealed_sensor_names",
            tuple(
                _name(value, "revealed sensor name")
                for value in self.revealed_sensor_names
            ),
        )
        object.__setattr__(
            self,
            "total_sensor_cost",
            _number(self.total_sensor_cost, "total_sensor_cost", nonnegative=True),
        )
        object.__setattr__(
            self,
            "total_sensor_risk",
            _number(self.total_sensor_risk, "total_sensor_risk", nonnegative=True),
        )
        object.__setattr__(
            self,
            "terminal_certificate_support_count",
            _integer(
                self.terminal_certificate_support_count,
                "terminal_certificate_support_count",
                minimum=1,
            ),
        )
        object.__setattr__(
            self,
            "terminal_certificate_reason_code",
            _name(
                self.terminal_certificate_reason_code,
                "terminal_certificate_reason_code",
            ),
        )
        if self.sealed_before_scoring is not True:
            raise ValueError("trace must be sealed before scoring")

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SENSOR_REVEAL_TRACE_VERSION,
            "kind": "SensorRevealTrace",
            "case_id": self.case_id,
            "manifest_id": self.manifest_id,
            "submission_id": self.submission_id,
            "plan_id": self.plan_id,
            "events": [event.as_dict() for event in self.events],
            "terminal_mode": self.terminal_mode,
            "action_index": self.action_index,
            "action_name": self.action_name,
            "fallback_used": self.fallback_used,
            "revealed_sensor_indices": list(self.revealed_sensor_indices),
            "revealed_sensor_names": list(self.revealed_sensor_names),
            "total_sensor_cost": self.total_sensor_cost,
            "total_sensor_risk": self.total_sensor_risk,
            "terminal_certificate_support_count": (
                self.terminal_certificate_support_count
            ),
            "terminal_certificate_reason_code": (self.terminal_certificate_reason_code),
            "sealed_before_scoring": self.sealed_before_scoring,
            "claim_boundary": SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY,
        }

    @property
    def trace_id(self) -> str:
        return _content_id(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "trace_id": self.trace_id}


def _policy_branch(
    node: Mapping[str, object], outcome_index: int
) -> Mapping[str, object]:
    branches = node.get("outcomes")
    if not isinstance(branches, list):
        raise ValueError("policy outcomes are malformed")
    matches = [
        branch
        for branch in branches
        if isinstance(branch, dict) and branch.get("outcome_index") == outcome_index
    ]
    if len(matches) != 1 or not isinstance(matches[0].get("policy"), dict):
        raise ValueError("revealed outcome does not select one policy branch")
    return matches[0]["policy"]


def execute_sensor_reveal_plan(
    manifest: SensorRevealManifest,
    truth: SensorRevealTruth,
    submission: SensorRevealSubmission,
    plan: SensorRevealPlan,
) -> SensorRevealTrace:
    """Traverse a frozen policy and disclose only requested sensors."""

    if truth.manifest != manifest:
        raise ValueError("truth does not belong to the supplied manifest")
    validate_sensor_reveal_submission(manifest, submission)
    if plan.manifest_id != manifest.manifest_id:
        raise ValueError("plan manifest_id mismatch")
    if plan.submission_id != submission.submission_id:
        raise ValueError("plan submission_id mismatch")
    verify_sensor_reveal_plan(manifest.as_dict(), plan.as_dict())

    node: Mapping[str, object] = plan.policy
    events: list[SensorRevealEvent] = []
    revealed: set[int] = set()
    while node["mode"] == "probe":
        sensor_index = _integer(node["probe_index"], "policy probe_index")
        if sensor_index in revealed:
            raise ValueError("policy requests the same sensor more than once")
        outcome_index = truth.sensor_outcome_indices[sensor_index]
        child = _policy_branch(node, outcome_index)
        events.append(
            SensorRevealEvent(
                step_index=len(events),
                sensor_index=sensor_index,
                sensor_name=manifest.sensor_names[sensor_index],
                outcome_index=outcome_index,
                outcome_name=(
                    manifest.sensor_outcome_names[sensor_index][outcome_index]
                ),
                payload_sha256=truth.sensor_payload_sha256[sensor_index],
                adapter_id=truth.sensor_adapter_ids[sensor_index],
                policy_node_id_before=policy_node_id(node),
                policy_node_id_after=policy_node_id(child),
            )
        )
        revealed.add(sensor_index)
        node = child

    if node["mode"] == "act":
        action_index = _integer(node["action_index"], "policy action_index")
        terminal_mode: SensorTerminalMode = "act"
        fallback_used = False
    elif node["mode"] == "fallback":
        action_index = manifest.fallback_action_index
        terminal_mode = "fallback"
        fallback_used = True
    else:
        raise ValueError("policy did not terminate in act or fallback")
    certificate = node.get("certificate")
    if not isinstance(certificate, dict):
        raise ValueError("terminal policy certificate is malformed")
    support = certificate.get("support_indices")
    reason = certificate.get("reason_code")
    if not isinstance(support, list) or type(reason) is not str:
        raise ValueError("terminal policy certificate is malformed")

    revealed_indices = tuple(event.sensor_index for event in events)
    trace = SensorRevealTrace(
        case_id=manifest.case_id,
        manifest_id=manifest.manifest_id,
        submission_id=submission.submission_id,
        plan_id=plan.plan_id,
        events=tuple(events),
        terminal_mode=terminal_mode,
        action_index=action_index,
        action_name=manifest.action_names[action_index],
        fallback_used=fallback_used,
        revealed_sensor_indices=revealed_indices,
        revealed_sensor_names=tuple(event.sensor_name for event in events),
        total_sensor_cost=sum(
            manifest.sensor_costs[index] for index in revealed_indices
        ),
        total_sensor_risk=sum(
            manifest.sensor_risks[index] for index in revealed_indices
        ),
        terminal_certificate_support_count=len(support),
        terminal_certificate_reason_code=reason,
    )
    verify_sensor_reveal_trace(
        manifest.as_dict(),
        plan.as_dict(),
        trace.as_dict(),
    )
    return trace


@dataclass(frozen=True)
class SensorRevealScore:
    """Post-seal offline score for one sensor-reveal trace."""

    truth_id: str
    trace_id: str
    action_index: int
    action_name: str
    fallback_used: bool
    selected_realized_loss: float
    fallback_realized_loss: float
    best_realized_loss: float
    realized_regret: float
    improvement_vs_fallback: float
    total_sensor_cost: float
    cost_multiplier: float
    objective_with_sensor_cost: float

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SENSOR_REVEAL_TRACE_VERSION,
            "kind": "SensorRevealScore",
            "truth_id": self.truth_id,
            "trace_id": self.trace_id,
            "action_index": self.action_index,
            "action_name": self.action_name,
            "fallback_used": self.fallback_used,
            "selected_realized_loss": self.selected_realized_loss,
            "fallback_realized_loss": self.fallback_realized_loss,
            "best_realized_loss": self.best_realized_loss,
            "realized_regret": self.realized_regret,
            "improvement_vs_fallback": self.improvement_vs_fallback,
            "total_sensor_cost": self.total_sensor_cost,
            "cost_multiplier": self.cost_multiplier,
            "objective_with_sensor_cost": self.objective_with_sensor_cost,
            "claim_boundary": SENSOR_REVEAL_SCORE_CLAIM_BOUNDARY,
        }

    @property
    def score_id(self) -> str:
        return _content_id(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "score_id": self.score_id}


def score_sensor_reveal_trace(
    manifest: SensorRevealManifest,
    truth: SensorRevealTruth,
    submission: SensorRevealSubmission,
    plan: SensorRevealPlan,
    trace: SensorRevealTrace,
    *,
    cost_multiplier: float = 0.0,
) -> SensorRevealScore:
    """Open terminal losses only after the reveal trace is sealed."""

    if truth.manifest != manifest:
        raise ValueError("truth does not belong to the supplied manifest")
    validate_sensor_reveal_submission(manifest, submission)
    verify_sensor_reveal_trace(
        manifest.as_dict(),
        plan.as_dict(),
        trace.as_dict(),
    )
    for event in trace.events:
        index = event.sensor_index
        if event.outcome_index != truth.sensor_outcome_indices[index]:
            raise ValueError("trace outcome does not match challenge truth")
        if event.payload_sha256 != truth.sensor_payload_sha256[index]:
            raise ValueError("trace payload digest does not match challenge truth")
        if event.adapter_id != truth.sensor_adapter_ids[index]:
            raise ValueError("trace adapter id does not match challenge truth")
    multiplier = _number(cost_multiplier, "cost_multiplier", nonnegative=True)
    selected = truth.realized_action_losses[trace.action_index]
    fallback = truth.realized_action_losses[manifest.fallback_action_index]
    best = min(truth.realized_action_losses)
    return SensorRevealScore(
        truth_id=truth.truth_id,
        trace_id=trace.trace_id,
        action_index=trace.action_index,
        action_name=trace.action_name,
        fallback_used=trace.fallback_used,
        selected_realized_loss=selected,
        fallback_realized_loss=fallback,
        best_realized_loss=best,
        realized_regret=selected - best,
        improvement_vs_fallback=fallback - selected,
        total_sensor_cost=trace.total_sensor_cost,
        cost_multiplier=multiplier,
        objective_with_sensor_cost=(selected + multiplier * trace.total_sensor_cost),
    )
