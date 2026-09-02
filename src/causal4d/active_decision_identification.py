"""One-step active decision identification over supplied finite certificates.

The module composes Causal4D's certificate consumer with a prospective roster of
finite-outcome probes.  It never reconstructs a latent physical state.  If the
current certificate uniquely identifies an action, that action is returned
immediately.  Otherwise, safe probes are ranked by expected reduction in the
certificate's minimax worst-case regret, net of registered probe cost.  If no
probe meets the registered value and certification-probability thresholds, the
caller-owned fallback action is returned exactly.

Every guarantee remains conditional on the supplied certificates, outcome
probabilities, action roster, ambiguity support, risk values, and costs.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .decision_identifiable_intervention import (
    DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY,
    DecisionIdentifiableInterventionV1,
    consume_query_decision_certificate,
)

ActiveDecisionMode = Literal["act", "probe", "fallback"]

ACTIVE_DECISION_IDENTIFICATION_VERSION = 1
ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY = (
    "Causal4D selects among caller-supplied finite probes using exact "
    "source-certificate regret summaries and returns the caller-owned fallback "
    "when no safe positive-value probe qualifies. This does not validate the "
    "physical hypothesis support, quotient, probe likelihoods, risk model, "
    "costs, held-out transport, target coverage, deployment authorization, or "
    "safety."
)

_NUMERICAL_ATOL = 1e-12


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _probability(value: object, *, name: str) -> float:
    result = _finite_nonnegative(value, name=name)
    if result > 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _nonempty_name(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


@dataclass(frozen=True)
class CertificateOutcomeV1:
    """One possible probe outcome and its supplied posterior certificate."""

    probability: float
    certificate: object

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probability",
            _probability(self.probability, name="outcome probability"),
        )


@dataclass(frozen=True)
class CertificateProbeV1:
    """A finite probe with posterior certificates for every outcome."""

    name: str
    outcomes: tuple[CertificateOutcomeV1, ...]
    physical_risk: float = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_name(self.name, name="probe name"))
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("probe outcomes must be nonempty")
        if not all(isinstance(item, CertificateOutcomeV1) for item in outcomes):
            raise TypeError("probe outcomes must contain CertificateOutcomeV1 values")
        total = float(sum(item.probability for item in outcomes))
        if not np.isclose(total, 1.0, atol=_NUMERICAL_ATOL, rtol=0.0):
            raise ValueError("probe outcome probabilities must sum to one")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(
            self,
            "physical_risk",
            _probability(self.physical_risk, name="physical_risk"),
        )
        object.__setattr__(
            self,
            "cost",
            _finite_nonnegative(self.cost, name="cost"),
        )


@dataclass(frozen=True)
class CertificateProbeReportV1:
    """Prospective structural value of one supplied probe."""

    name: str
    safe: bool
    physical_risk: float
    cost: float
    current_minimax_worst_case_regret: float
    expected_posterior_minimax_worst_case_regret: float
    expected_regret_reduction: float
    certification_probability: float
    robust_certification_probability: float
    net_value: float
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.name, name="name")
        _probability(self.physical_risk, name="physical_risk")
        _finite_nonnegative(self.cost, name="cost")
        _finite_nonnegative(
            self.current_minimax_worst_case_regret,
            name="current_minimax_worst_case_regret",
        )
        _finite_nonnegative(
            self.expected_posterior_minimax_worst_case_regret,
            name="expected_posterior_minimax_worst_case_regret",
        )
        if not np.isfinite(self.expected_regret_reduction):
            raise ValueError("expected_regret_reduction must be finite")
        _probability(
            self.certification_probability,
            name="certification_probability",
        )
        _probability(
            self.robust_certification_probability,
            name="robust_certification_probability",
        )
        if self.robust_certification_probability > (
            self.certification_probability + _NUMERICAL_ATOL
        ):
            raise ValueError(
                "robust certification probability cannot exceed total certification"
            )
        if not np.isfinite(self.net_value):
            raise ValueError("net_value must be finite")


@dataclass(frozen=True)
class ActiveDecisionPlanV1:
    """Act now, acquire one probe, or return exact fallback."""

    mode: ActiveDecisionMode
    action_name: str | None
    selected_probe_name: str | None
    fallback_action_name: str
    current_intervention: DecisionIdentifiableInterventionV1
    probe_reports: tuple[CertificateProbeReportV1, ...]
    score: float
    reason_code: str

    def __post_init__(self) -> None:
        fallback = _nonempty_name(
            self.fallback_action_name,
            name="fallback_action_name",
        )
        if self.mode not in ("act", "probe", "fallback"):
            raise ValueError("unknown active decision mode")
        if self.mode == "act":
            if (
                self.current_intervention.used_exact_fallback
                or self.action_name != self.current_intervention.action_name
                or self.selected_probe_name is not None
            ):
                raise ValueError("act plans require the current certified action")
        elif self.mode == "probe":
            if self.action_name is not None or self.selected_probe_name is None:
                raise ValueError("probe plans require exactly one selected probe")
        else:
            if self.action_name != fallback or self.selected_probe_name is not None:
                raise ValueError("fallback plans must return the exact fallback action")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.reason_code:
            raise ValueError("reason_code must be nonempty")
        object.__setattr__(self, "fallback_action_name", fallback)

    @property
    def used_exact_fallback(self) -> bool:
        return self.mode == "fallback"

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ACTIVE_DECISION_IDENTIFICATION_VERSION,
            "mode": self.mode,
            "action_name": self.action_name,
            "selected_probe_name": self.selected_probe_name,
            "fallback_action_name": self.fallback_action_name,
            "used_exact_fallback": self.used_exact_fallback,
            "score": self.score,
            "reason_code": self.reason_code,
            "current_intervention": self.current_intervention.as_dict(),
            "probe_reports": [
                {
                    "name": report.name,
                    "safe": report.safe,
                    "physical_risk": report.physical_risk,
                    "cost": report.cost,
                    "current_minimax_worst_case_regret": (
                        report.current_minimax_worst_case_regret
                    ),
                    "expected_posterior_minimax_worst_case_regret": (
                        report.expected_posterior_minimax_worst_case_regret
                    ),
                    "expected_regret_reduction": report.expected_regret_reduction,
                    "certification_probability": report.certification_probability,
                    "robust_certification_probability": (
                        report.robust_certification_probability
                    ),
                    "net_value": report.net_value,
                    "reason_codes": list(report.reason_codes),
                }
                for report in self.probe_reports
            ],
            "source_intervention_claim_boundary": (
                DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY
            ),
            "claim_boundary": ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY,
        }


def evaluate_certificate_probe(
    current_intervention: DecisionIdentifiableInterventionV1,
    probe: CertificateProbeV1,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
) -> CertificateProbeReportV1:
    """Evaluate one probe from supplied posterior certificate branches."""

    if not isinstance(current_intervention, DecisionIdentifiableInterventionV1):
        raise TypeError(
            "current_intervention must be DecisionIdentifiableInterventionV1"
        )
    cap = _probability(risk_cap, name="risk_cap")
    multiplier = _finite_nonnegative(cost_multiplier, name="cost_multiplier")
    current_regret = current_intervention.minimax_worst_case_regret

    expected_regret = 0.0
    certification_probability = 0.0
    robust_probability = 0.0
    for outcome in probe.outcomes:
        posterior = consume_query_decision_certificate(
            outcome.certificate,
            action_names,
            fallback_action_name=fallback_action_name,
        )
        expected_regret += outcome.probability * posterior.minimax_worst_case_regret
        if not posterior.used_exact_fallback:
            certification_probability += outcome.probability
            if posterior.certificate_level == "robustly-optimal":
                robust_probability += outcome.probability

    reduction = current_regret - expected_regret
    safe = probe.physical_risk <= cap + _NUMERICAL_ATOL
    reasons: list[str] = []
    if not safe:
        reasons.append("prospective-physical-risk-cap-exceeded")
    if reduction <= _NUMERICAL_ATOL:
        reasons.append("no-positive-expected-regret-reduction")
    if certification_probability <= _NUMERICAL_ATOL:
        reasons.append("no-positive-mass-certified-outcome")

    return CertificateProbeReportV1(
        name=probe.name,
        safe=safe,
        physical_risk=probe.physical_risk,
        cost=probe.cost,
        current_minimax_worst_case_regret=current_regret,
        expected_posterior_minimax_worst_case_regret=max(expected_regret, 0.0),
        expected_regret_reduction=reduction,
        certification_probability=min(max(certification_probability, 0.0), 1.0),
        robust_certification_probability=min(max(robust_probability, 0.0), 1.0),
        net_value=reduction - multiplier * probe.cost,
        reason_codes=tuple(reasons),
    )


def plan_active_decision(
    current_certificate: object,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
    probes: tuple[CertificateProbeV1, ...] = (),
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
    minimum_net_value: float = 0.0,
    minimum_certification_probability: float = 1.0,
) -> ActiveDecisionPlanV1:
    """Choose act, one safe decision-identifying probe, or exact fallback."""

    threshold = _finite_nonnegative(
        minimum_net_value,
        name="minimum_net_value",
    )
    certification_threshold = _probability(
        minimum_certification_probability,
        name="minimum_certification_probability",
    )
    current = consume_query_decision_certificate(
        current_certificate,
        action_names,
        fallback_action_name=fallback_action_name,
    )
    if not current.used_exact_fallback:
        return ActiveDecisionPlanV1(
            mode="act",
            action_name=current.action_name,
            selected_probe_name=None,
            fallback_action_name=fallback_action_name,
            current_intervention=current,
            probe_reports=(),
            score=current.minimax_worst_case_regret,
            reason_code="current-decision-uniquely-certified",
        )

    roster = tuple(probes)
    names = [probe.name for probe in roster]
    if len(set(names)) != len(names):
        raise ValueError("probe names must be unique")

    reports = tuple(
        evaluate_certificate_probe(
            current,
            probe,
            action_names,
            fallback_action_name=fallback_action_name,
            risk_cap=risk_cap,
            cost_multiplier=cost_multiplier,
        )
        for probe in roster
    )
    eligible = [
        report
        for report in reports
        if report.safe
        and report.certification_probability
        >= certification_threshold - _NUMERICAL_ATOL
        and report.net_value > threshold + _NUMERICAL_ATOL
    ]
    if eligible:
        eligible.sort(
            key=lambda report: (
                -report.net_value,
                -report.certification_probability,
                -report.robust_certification_probability,
                report.physical_risk,
                report.cost,
                report.name,
            )
        )
        winner = eligible[0]
        return ActiveDecisionPlanV1(
            mode="probe",
            action_name=None,
            selected_probe_name=winner.name,
            fallback_action_name=fallback_action_name,
            current_intervention=current,
            probe_reports=reports,
            score=winner.net_value,
            reason_code="selected-safe-positive-decision-identification-value",
        )

    if not reports:
        reason = "no-candidate-probes"
    elif not any(report.safe for report in reports):
        reason = "no-safe-probe"
    elif not any(
        report.certification_probability >= certification_threshold - _NUMERICAL_ATOL
        for report in reports
    ):
        reason = "no-probe-meets-certification-probability"
    else:
        reason = "no-positive-net-decision-identification-value"
    return ActiveDecisionPlanV1(
        mode="fallback",
        action_name=fallback_action_name,
        selected_probe_name=None,
        fallback_action_name=fallback_action_name,
        current_intervention=current,
        probe_reports=reports,
        score=0.0,
        reason_code=reason,
    )


def consume_probe_outcome(
    probe: CertificateProbeV1,
    outcome_index: int,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
) -> DecisionIdentifiableInterventionV1:
    """Consume one realized probe outcome through the existing fail-closed router."""

    if isinstance(outcome_index, bool) or not isinstance(
        outcome_index,
        (int, np.integer),
    ):
        raise TypeError("outcome_index must be an integer")
    index = int(outcome_index)
    if not 0 <= index < len(probe.outcomes):
        raise ValueError("outcome_index is outside the probe outcome support")
    return consume_query_decision_certificate(
        probe.outcomes[index].certificate,
        action_names,
        fallback_action_name=fallback_action_name,
    )


__all__ = [
    "ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY",
    "ACTIVE_DECISION_IDENTIFICATION_VERSION",
    "ActiveDecisionMode",
    "ActiveDecisionPlanV1",
    "CertificateOutcomeV1",
    "CertificateProbeReportV1",
    "CertificateProbeV1",
    "consume_probe_outcome",
    "evaluate_certificate_probe",
    "plan_active_decision",
]
