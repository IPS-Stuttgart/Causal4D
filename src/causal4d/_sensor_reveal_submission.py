"""Provider submission and frozen policy for sensor-reveal replay."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass

from ._sensor_reveal_case import SensorRevealManifest
from ._sensor_reveal_common import (
    SENSOR_REVEAL_TRACE_VERSION,
    _ATOL,
    _content_id,
    _digest,
    _integer,
    _name,
    _number,
)
from .sensor_reveal_verifier import (
    SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY,
    verify_sensor_reveal_plan,
)
from .sequential_decision_identification import (
    FiniteProbe,
    SequentialObjective,
    solve_sequential_decision,
)


@dataclass(frozen=True)
class SensorRevealSubmission:
    """Provider-owned finite belief and sensor models; no held truth."""

    case_id: str
    manifest_id: str
    provider_id: str
    implementation_id: str
    hypothesis_weights: tuple[float, ...]
    hypothesis_losses: tuple[tuple[float, ...], ...]
    probes: tuple[FiniteProbe, ...]
    regret_tolerance: float = 0.0
    max_probes: int = 1
    risk_budget: float = 1.0
    objective: SequentialObjective = "expected_cost"
    maximum_nodes: int = 100_000

    def __post_init__(self) -> None:
        weights = tuple(
            _number(value, "hypothesis weight", nonnegative=True)
            for value in self.hypothesis_weights
        )
        if not weights or sum(weights) <= 0.0:
            raise ValueError("hypothesis weights must have positive mass")
        losses = tuple(
            tuple(_number(value, "hypothesis loss") for value in row)
            for row in self.hypothesis_losses
        )
        if not losses or any(not row for row in losses):
            raise ValueError("hypothesis losses must be nonempty")
        action_count = len(losses[0])
        if len(losses) != len(weights) or any(
            len(row) != action_count for row in losses
        ):
            raise ValueError("hypothesis belief/loss dimensions do not match")
        probes = tuple(self.probes)
        if not all(isinstance(probe, FiniteProbe) for probe in probes):
            raise TypeError("probes must contain FiniteProbe values")
        if self.objective not in ("expected_cost", "worst_case_cost"):
            raise ValueError("invalid sequential objective")
        object.__setattr__(self, "case_id", _name(self.case_id, "case_id"))
        for field in ("manifest_id", "provider_id", "implementation_id"):
            object.__setattr__(self, field, _digest(getattr(self, field), field))
        object.__setattr__(self, "hypothesis_weights", weights)
        object.__setattr__(self, "hypothesis_losses", losses)
        object.__setattr__(self, "probes", probes)
        object.__setattr__(
            self,
            "regret_tolerance",
            _number(self.regret_tolerance, "regret_tolerance", nonnegative=True),
        )
        object.__setattr__(
            self,
            "max_probes",
            _integer(self.max_probes, "max_probes"),
        )
        object.__setattr__(
            self,
            "risk_budget",
            _number(self.risk_budget, "risk_budget", nonnegative=True),
        )
        object.__setattr__(
            self,
            "maximum_nodes",
            _integer(self.maximum_nodes, "maximum_nodes", minimum=1),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "manifest_id": self.manifest_id,
            "provider_id": self.provider_id,
            "implementation_id": self.implementation_id,
            "hypothesis_weights": list(self.hypothesis_weights),
            "hypothesis_losses": [list(row) for row in self.hypothesis_losses],
            "probes": [probe.as_dict() for probe in self.probes],
            "regret_tolerance": self.regret_tolerance,
            "max_probes": self.max_probes,
            "risk_budget": self.risk_budget,
            "objective": self.objective,
            "maximum_nodes": self.maximum_nodes,
        }

    @property
    def submission_id(self) -> str:
        return _content_id(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "submission_id": self.submission_id}


def validate_sensor_reveal_submission(
    manifest: SensorRevealManifest,
    submission: SensorRevealSubmission,
) -> None:
    """Ensure provider sensor models match the public challenge interface."""

    if submission.case_id != manifest.case_id:
        raise ValueError("submission case_id mismatch")
    if submission.manifest_id != manifest.manifest_id:
        raise ValueError("submission manifest_id mismatch")
    if len(submission.probes) != len(manifest.sensor_names):
        raise ValueError("submission sensor count mismatch")
    if any(
        len(row) != len(manifest.action_names) for row in submission.hypothesis_losses
    ):
        raise ValueError("submission action count mismatch")
    for index, probe in enumerate(submission.probes):
        if probe.name != manifest.sensor_names[index]:
            raise ValueError("submission sensor name mismatch")
        if probe.outcome_names != manifest.sensor_outcome_names[index]:
            raise ValueError("submission sensor outcome roster mismatch")
        if not math.isclose(
            probe.cost,
            manifest.sensor_costs[index],
            rel_tol=0.0,
            abs_tol=_ATOL,
        ):
            raise ValueError("submission sensor cost mismatch")
        if not math.isclose(
            probe.risk,
            manifest.sensor_risks[index],
            rel_tol=0.0,
            abs_tol=_ATOL,
        ):
            raise ValueError("submission sensor risk mismatch")


@dataclass(frozen=True)
class SensorRevealPlan:
    """Content-addressed sequential policy frozen before disclosure."""

    manifest_id: str
    submission_id: str
    policy: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "manifest_id",
            _digest(self.manifest_id, "manifest_id"),
        )
        object.__setattr__(
            self,
            "submission_id",
            _digest(self.submission_id, "submission_id"),
        )
        if not isinstance(self.policy, Mapping):
            raise ValueError("policy must be a mapping")
        object.__setattr__(
            self,
            "policy",
            json.loads(json.dumps(self.policy, sort_keys=True, allow_nan=False)),
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": SENSOR_REVEAL_TRACE_VERSION,
            "kind": "SensorRevealPlan",
            "manifest_id": self.manifest_id,
            "submission_id": self.submission_id,
            "policy": dict(self.policy),
            "claim_boundary": SENSOR_REVEAL_TRACE_CLAIM_BOUNDARY,
        }

    @property
    def plan_id(self) -> str:
        return _content_id(self._payload())

    def as_dict(self) -> dict[str, object]:
        return {**self._payload(), "plan_id": self.plan_id}


def build_sensor_reveal_plan(
    manifest: SensorRevealManifest,
    submission: SensorRevealSubmission,
) -> SensorRevealPlan:
    """Freeze an exact finite-horizon policy before any optional sensor reveal."""

    validate_sensor_reveal_submission(manifest, submission)
    policy = solve_sequential_decision(
        submission.hypothesis_losses,
        submission.hypothesis_weights,
        submission.probes,
        regret_tolerance=submission.regret_tolerance,
        max_probes=submission.max_probes,
        risk_budget=submission.risk_budget,
        objective=submission.objective,
        maximum_nodes=submission.maximum_nodes,
    )
    plan = SensorRevealPlan(
        manifest_id=manifest.manifest_id,
        submission_id=submission.submission_id,
        policy=policy.as_dict(),
    )
    verify_sensor_reveal_plan(manifest.as_dict(), plan.as_dict())
    return plan
