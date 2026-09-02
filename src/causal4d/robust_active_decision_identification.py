"""Distributionally robust active decision identification.

The nominal active-decision interface assumes one supplied probability for every
probe outcome.  This module replaces those point probabilities by a
box-constrained simplex and permits a nonnegative additive regret inflation for
each outcome.  It then selects an action, a probe, or the exact caller-owned
fallback using worst-case expectations and worst-case certification probability.

All statements remain conditional on the supplied certificates, probability
bounds, regret inflations, physical-risk upper bounds, costs, and action roster.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Literal

import numpy as np

from .active_decision_identification import CertificateProbeV1
from .active_decision_calibration import (
    CategoricalProbabilityBoxV1,
    CompleteGroupRegretCalibrationV1,
)
from .decision_identifiable_intervention import (
    DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY,
    DecisionIdentifiableInterventionV1,
    consume_query_decision_certificate,
)

RobustActiveDecisionMode = Literal["act", "probe", "fallback"]
RobustCertificateLevel = Literal[
    "robustly-optimal",
    "tolerance-admissible",
    "uncertified",
]

ROBUST_ACTIVE_DECISION_IDENTIFICATION_VERSION = 1
ROBUST_ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY = (
    "Causal4D optimizes over a caller-supplied box-constrained outcome-probability "
    "simplex, additive branch-regret inflations, and prospective physical-risk "
    "upper bounds. The guarantee is conditional on those sets containing the "
    "deployed process and does not validate physical support completeness, "
    "exchangeability, calibration, probe channels, costs, deployment "
    "authorization, or safety."
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


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim != 1 or not result.size or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a nonempty finite vector")
    return result


def extremal_box_simplex_distribution(
    lower_bounds: Sequence[float],
    upper_bounds: Sequence[float],
    values: Sequence[float],
    *,
    maximize: bool,
) -> tuple[float, ...]:
    """Return an exact extremizer over a box-constrained probability simplex.

    The ambiguity set is ``lower <= p <= upper`` and ``sum(p) == 1``.  A linear
    objective is optimized by assigning all remaining mass, after satisfying the
    lower bounds, to outcomes in descending value order for maximization or
    ascending value order for minimization.
    """

    lower = _finite_vector(lower_bounds, name="lower_bounds")
    upper = _finite_vector(upper_bounds, name="upper_bounds")
    objective = _finite_vector(values, name="values")
    if lower.shape != upper.shape or lower.shape != objective.shape:
        raise ValueError("lower_bounds, upper_bounds, and values must align")
    if np.any(lower < 0.0) or np.any(upper > 1.0):
        raise ValueError("probability bounds must lie in [0, 1]")
    if np.any(lower > upper + _NUMERICAL_ATOL):
        raise ValueError("each lower probability bound must not exceed its upper bound")

    lower_sum = float(np.sum(lower))
    upper_sum = float(np.sum(upper))
    if lower_sum > 1.0 + _NUMERICAL_ATOL or upper_sum < 1.0 - _NUMERICAL_ATOL:
        raise ValueError("probability bounds define an empty simplex")

    probability = lower.copy()
    remaining = 1.0 - float(np.sum(probability))
    if remaining < 0.0:
        # Permit only a floating-point-sized violation of the simplex sum.
        probability[0] += remaining
        remaining = 0.0

    order = sorted(
        range(objective.size),
        key=lambda index: (
            -float(objective[index]) if maximize else float(objective[index]),
            index,
        ),
    )
    for index in order:
        if remaining <= _NUMERICAL_ATOL:
            break
        capacity = max(float(upper[index] - probability[index]), 0.0)
        assigned = min(capacity, remaining)
        probability[index] += assigned
        remaining -= assigned

    if remaining > _NUMERICAL_ATOL:
        raise ValueError("probability bounds define an empty simplex")
    residual = 1.0 - float(np.sum(probability))
    if abs(residual) > _NUMERICAL_ATOL:
        raise RuntimeError("failed to construct an extremal simplex distribution")
    if residual != 0.0:
        if residual > 0.0:
            candidates = [
                index
                for index in range(probability.size)
                if upper[index] - probability[index] >= residual - _NUMERICAL_ATOL
            ]
        else:
            candidates = [
                index
                for index in range(probability.size)
                if probability[index] - lower[index] >= -residual - _NUMERICAL_ATOL
            ]
        if not candidates:
            candidates = [0]
        probability[candidates[0]] += residual

    if np.any(probability < lower - _NUMERICAL_ATOL) or np.any(
        probability > upper + _NUMERICAL_ATOL
    ):
        raise RuntimeError("constructed extremizer violates probability bounds")
    return tuple(float(item) for item in probability)


@dataclass(frozen=True)
class AmbiguousCertificateOutcomeV1:
    """A probe outcome with probability bounds and an additive regret margin."""

    probability_lower: float
    probability_upper: float
    certificate: object
    regret_inflation: float = 0.0

    def __post_init__(self) -> None:
        lower = _probability(
            self.probability_lower,
            name="probability_lower",
        )
        upper = _probability(
            self.probability_upper,
            name="probability_upper",
        )
        if lower > upper + _NUMERICAL_ATOL:
            raise ValueError("probability_lower must not exceed probability_upper")
        object.__setattr__(self, "probability_lower", lower)
        object.__setattr__(self, "probability_upper", upper)
        object.__setattr__(
            self,
            "regret_inflation",
            _finite_nonnegative(
                self.regret_inflation,
                name="regret_inflation",
            ),
        )


@dataclass(frozen=True)
class AmbiguousCertificateProbeV1:
    """A finite probe with a box-constrained outcome distribution."""

    name: str
    outcomes: tuple[AmbiguousCertificateOutcomeV1, ...]
    physical_risk_upper: float = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _nonempty_name(self.name, name="probe name"))
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("probe outcomes must be nonempty")
        if not all(
            isinstance(item, AmbiguousCertificateOutcomeV1) for item in outcomes
        ):
            raise TypeError(
                "probe outcomes must contain AmbiguousCertificateOutcomeV1 values"
            )
        lower_sum = float(sum(item.probability_lower for item in outcomes))
        upper_sum = float(sum(item.probability_upper for item in outcomes))
        if lower_sum > 1.0 + _NUMERICAL_ATOL or upper_sum < 1.0 - _NUMERICAL_ATOL:
            raise ValueError("probe outcome probability bounds define an empty simplex")
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(
            self,
            "physical_risk_upper",
            _probability(
                self.physical_risk_upper,
                name="physical_risk_upper",
            ),
        )
        object.__setattr__(
            self,
            "cost",
            _finite_nonnegative(self.cost, name="cost"),
        )


@dataclass(frozen=True)
class RobustOutcomeDecisionV1:
    """One realized outcome routed through structural and inflated certificates."""

    action_name: str
    certified_action_name: str | None
    fallback_action_name: str
    used_exact_fallback: bool
    certificate_level: RobustCertificateLevel
    structural_intervention: DecisionIdentifiableInterventionV1
    regret_inflation: float
    inflated_selected_worst_case_regret: float | None
    inflated_minimax_worst_case_regret: float
    reason_code: str

    def __post_init__(self) -> None:
        action = _nonempty_name(self.action_name, name="action_name")
        fallback = _nonempty_name(
            self.fallback_action_name,
            name="fallback_action_name",
        )
        if not isinstance(
            self.structural_intervention,
            DecisionIdentifiableInterventionV1,
        ):
            raise TypeError(
                "structural_intervention must be DecisionIdentifiableInterventionV1"
            )
        if self.certificate_level not in (
            "robustly-optimal",
            "tolerance-admissible",
            "uncertified",
        ):
            raise ValueError("unknown certificate_level")
        if self.used_exact_fallback != (self.certified_action_name is None):
            raise ValueError("fallback flag must agree with certified_action_name")
        if self.used_exact_fallback:
            if action != fallback or self.certificate_level != "uncertified":
                raise ValueError("uncertified outcomes must return exact fallback")
        else:
            certified = _nonempty_name(
                self.certified_action_name,
                name="certified_action_name",
            )
            if action != certified or self.certificate_level == "uncertified":
                raise ValueError("certified outcome fields are inconsistent")
        _finite_nonnegative(self.regret_inflation, name="regret_inflation")
        if self.inflated_selected_worst_case_regret is not None:
            _finite_nonnegative(
                self.inflated_selected_worst_case_regret,
                name="inflated_selected_worst_case_regret",
            )
        _finite_nonnegative(
            self.inflated_minimax_worst_case_regret,
            name="inflated_minimax_worst_case_regret",
        )
        if not self.reason_code:
            raise ValueError("reason_code must be nonempty")
        object.__setattr__(self, "action_name", action)
        object.__setattr__(self, "fallback_action_name", fallback)

    def as_dict(self) -> dict[str, object]:
        return {
            "action_name": self.action_name,
            "certified_action_name": self.certified_action_name,
            "fallback_action_name": self.fallback_action_name,
            "used_exact_fallback": self.used_exact_fallback,
            "certificate_level": self.certificate_level,
            "regret_inflation": self.regret_inflation,
            "inflated_selected_worst_case_regret": (
                self.inflated_selected_worst_case_regret
            ),
            "inflated_minimax_worst_case_regret": (
                self.inflated_minimax_worst_case_regret
            ),
            "reason_code": self.reason_code,
            "structural_intervention": self.structural_intervention.as_dict(),
        }


@dataclass(frozen=True)
class RobustCertificateProbeReportV1:
    """Worst-case value and certification report for one ambiguous probe."""

    name: str
    safe: bool
    physical_risk_upper: float
    cost: float
    current_minimax_worst_case_regret: float
    worst_case_expected_posterior_minimax_worst_case_regret: float
    guaranteed_regret_reduction: float
    worst_case_certification_probability: float
    worst_case_robust_certification_probability: float
    net_guaranteed_value: float
    probability_lower_bounds: tuple[float, ...]
    probability_upper_bounds: tuple[float, ...]
    worst_case_regret_distribution: tuple[float, ...]
    worst_case_certification_distribution: tuple[float, ...]
    outcome_decisions: tuple[RobustOutcomeDecisionV1, ...]
    reason_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonempty_name(self.name, name="name")
        _probability(self.physical_risk_upper, name="physical_risk_upper")
        _finite_nonnegative(self.cost, name="cost")
        _finite_nonnegative(
            self.current_minimax_worst_case_regret,
            name="current_minimax_worst_case_regret",
        )
        _finite_nonnegative(
            self.worst_case_expected_posterior_minimax_worst_case_regret,
            name="worst_case_expected_posterior_minimax_worst_case_regret",
        )
        if not np.isfinite(self.guaranteed_regret_reduction):
            raise ValueError("guaranteed_regret_reduction must be finite")
        _probability(
            self.worst_case_certification_probability,
            name="worst_case_certification_probability",
        )
        _probability(
            self.worst_case_robust_certification_probability,
            name="worst_case_robust_certification_probability",
        )
        if self.worst_case_robust_certification_probability > (
            self.worst_case_certification_probability + _NUMERICAL_ATOL
        ):
            raise ValueError(
                "robust certification probability cannot exceed total certification"
            )
        if not np.isfinite(self.net_guaranteed_value):
            raise ValueError("net_guaranteed_value must be finite")
        decisions = tuple(self.outcome_decisions)
        size = len(decisions)
        if not size or not all(
            isinstance(item, RobustOutcomeDecisionV1) for item in decisions
        ):
            raise TypeError(
                "outcome_decisions must contain RobustOutcomeDecisionV1 values"
            )
        lower = _finite_vector(
            self.probability_lower_bounds,
            name="probability_lower_bounds",
        )
        upper = _finite_vector(
            self.probability_upper_bounds,
            name="probability_upper_bounds",
        )
        regret_distribution = _finite_vector(
            self.worst_case_regret_distribution,
            name="worst_case_regret_distribution",
        )
        certification_distribution = _finite_vector(
            self.worst_case_certification_distribution,
            name="worst_case_certification_distribution",
        )
        vectors = (lower, upper, regret_distribution, certification_distribution)
        if any(vector.size != size for vector in vectors):
            raise ValueError("probe report outcome vectors must align")
        if np.any(lower < 0.0) or np.any(upper > 1.0) or np.any(lower > upper):
            raise ValueError("probe report probability bounds are invalid")
        for distribution in (regret_distribution, certification_distribution):
            if np.any(distribution < 0.0) or not np.isclose(
                float(np.sum(distribution)),
                1.0,
                atol=_NUMERICAL_ATOL,
                rtol=0.0,
            ):
                raise ValueError("worst-case witnesses must be distributions")
            if np.any(distribution < lower - _NUMERICAL_ATOL) or np.any(
                distribution > upper + _NUMERICAL_ATOL
            ):
                raise ValueError("worst-case witnesses violate probability bounds")
        inflated_regret = np.asarray(
            [item.inflated_minimax_worst_case_regret for item in decisions],
            dtype=np.float64,
        )
        expected = float(np.dot(regret_distribution, inflated_regret))
        if not np.isclose(
            expected,
            self.worst_case_expected_posterior_minimax_worst_case_regret,
            atol=_NUMERICAL_ATOL,
            rtol=0.0,
        ):
            raise ValueError("worst-case regret witness is inconsistent")
        certified = np.asarray(
            [0.0 if item.used_exact_fallback else 1.0 for item in decisions],
            dtype=np.float64,
        )
        certified_probability = float(np.dot(certification_distribution, certified))
        if not np.isclose(
            certified_probability,
            self.worst_case_certification_probability,
            atol=_NUMERICAL_ATOL,
            rtol=0.0,
        ):
            raise ValueError("worst-case certification witness is inconsistent")
        object.__setattr__(self, "outcome_decisions", decisions)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "safe": self.safe,
            "physical_risk_upper": self.physical_risk_upper,
            "cost": self.cost,
            "current_minimax_worst_case_regret": (
                self.current_minimax_worst_case_regret
            ),
            "worst_case_expected_posterior_minimax_worst_case_regret": (
                self.worst_case_expected_posterior_minimax_worst_case_regret
            ),
            "guaranteed_regret_reduction": self.guaranteed_regret_reduction,
            "worst_case_certification_probability": (
                self.worst_case_certification_probability
            ),
            "worst_case_robust_certification_probability": (
                self.worst_case_robust_certification_probability
            ),
            "net_guaranteed_value": self.net_guaranteed_value,
            "probability_lower_bounds": list(self.probability_lower_bounds),
            "probability_upper_bounds": list(self.probability_upper_bounds),
            "worst_case_regret_distribution": list(self.worst_case_regret_distribution),
            "worst_case_certification_distribution": list(
                self.worst_case_certification_distribution
            ),
            "outcome_decisions": [
                decision.as_dict() for decision in self.outcome_decisions
            ],
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class RobustActiveDecisionPlanV1:
    """Act now, acquire a robustly valuable probe, or return exact fallback."""

    mode: RobustActiveDecisionMode
    action_name: str | None
    selected_probe_name: str | None
    fallback_action_name: str
    current_intervention: DecisionIdentifiableInterventionV1
    probe_reports: tuple[RobustCertificateProbeReportV1, ...]
    score: float
    reason_code: str

    def __post_init__(self) -> None:
        fallback = _nonempty_name(
            self.fallback_action_name,
            name="fallback_action_name",
        )
        if not isinstance(
            self.current_intervention,
            DecisionIdentifiableInterventionV1,
        ):
            raise TypeError(
                "current_intervention must be DecisionIdentifiableInterventionV1"
            )
        reports = tuple(self.probe_reports)
        if not all(
            isinstance(item, RobustCertificateProbeReportV1) for item in reports
        ):
            raise TypeError(
                "probe_reports must contain RobustCertificateProbeReportV1 values"
            )
        if self.mode not in ("act", "probe", "fallback"):
            raise ValueError("unknown robust active decision mode")
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
        if self.mode == "probe" and self.selected_probe_name not in {
            report.name for report in reports
        }:
            raise ValueError("selected_probe_name is absent from probe_reports")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.reason_code:
            raise ValueError("reason_code must be nonempty")
        object.__setattr__(self, "fallback_action_name", fallback)
        object.__setattr__(self, "probe_reports", reports)

    @property
    def used_exact_fallback(self) -> bool:
        return self.mode == "fallback"

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ROBUST_ACTIVE_DECISION_IDENTIFICATION_VERSION,
            "mode": self.mode,
            "action_name": self.action_name,
            "selected_probe_name": self.selected_probe_name,
            "fallback_action_name": self.fallback_action_name,
            "used_exact_fallback": self.used_exact_fallback,
            "score": self.score,
            "reason_code": self.reason_code,
            "current_intervention": self.current_intervention.as_dict(),
            "probe_reports": [report.as_dict() for report in self.probe_reports],
            "source_intervention_claim_boundary": (
                DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY
            ),
            "claim_boundary": ROBUST_ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY,
        }


def assess_ambiguous_certificate_outcome(
    outcome: AmbiguousCertificateOutcomeV1,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
) -> RobustOutcomeDecisionV1:
    """Apply an additive branch-regret margin and fail closed if it revokes action."""

    if not isinstance(outcome, AmbiguousCertificateOutcomeV1):
        raise TypeError("outcome must be AmbiguousCertificateOutcomeV1")
    structural = consume_query_decision_certificate(
        outcome.certificate,
        action_names,
        fallback_action_name=fallback_action_name,
    )
    inflated_minimax = structural.minimax_worst_case_regret + outcome.regret_inflation
    if structural.used_exact_fallback:
        return RobustOutcomeDecisionV1(
            action_name=fallback_action_name,
            certified_action_name=None,
            fallback_action_name=fallback_action_name,
            used_exact_fallback=True,
            certificate_level="uncertified",
            structural_intervention=structural,
            regret_inflation=outcome.regret_inflation,
            inflated_selected_worst_case_regret=None,
            inflated_minimax_worst_case_regret=inflated_minimax,
            reason_code="structural-certificate-unresolved",
        )

    if structural.selected_worst_case_regret is None:
        raise RuntimeError("certified structural intervention lacks selected regret")
    inflated_selected = structural.selected_worst_case_regret + outcome.regret_inflation
    if inflated_selected > structural.regret_tolerance + _NUMERICAL_ATOL:
        return RobustOutcomeDecisionV1(
            action_name=fallback_action_name,
            certified_action_name=None,
            fallback_action_name=fallback_action_name,
            used_exact_fallback=True,
            certificate_level="uncertified",
            structural_intervention=structural,
            regret_inflation=outcome.regret_inflation,
            inflated_selected_worst_case_regret=inflated_selected,
            inflated_minimax_worst_case_regret=inflated_minimax,
            reason_code="regret-inflation-exceeds-tolerance",
        )

    robust = (
        structural.certificate_level == "robustly-optimal"
        and outcome.regret_inflation <= _NUMERICAL_ATOL
    )
    level: RobustCertificateLevel = (
        "robustly-optimal" if robust else "tolerance-admissible"
    )
    reason = (
        "robust-certification-preserved"
        if robust
        else "inflated-regret-within-tolerance"
    )
    return RobustOutcomeDecisionV1(
        action_name=structural.action_name,
        certified_action_name=structural.action_name,
        fallback_action_name=fallback_action_name,
        used_exact_fallback=False,
        certificate_level=level,
        structural_intervention=structural,
        regret_inflation=outcome.regret_inflation,
        inflated_selected_worst_case_regret=inflated_selected,
        inflated_minimax_worst_case_regret=inflated_minimax,
        reason_code=reason,
    )


def evaluate_ambiguous_certificate_probe(
    current_intervention: DecisionIdentifiableInterventionV1,
    probe: AmbiguousCertificateProbeV1,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
) -> RobustCertificateProbeReportV1:
    """Evaluate one probe against every distribution in its ambiguity set."""

    if not isinstance(current_intervention, DecisionIdentifiableInterventionV1):
        raise TypeError(
            "current_intervention must be DecisionIdentifiableInterventionV1"
        )
    if not isinstance(probe, AmbiguousCertificateProbeV1):
        raise TypeError("probe must be AmbiguousCertificateProbeV1")
    cap = _probability(risk_cap, name="risk_cap")
    multiplier = _finite_nonnegative(cost_multiplier, name="cost_multiplier")

    decisions = tuple(
        assess_ambiguous_certificate_outcome(
            outcome,
            action_names,
            fallback_action_name=fallback_action_name,
        )
        for outcome in probe.outcomes
    )
    lower = tuple(outcome.probability_lower for outcome in probe.outcomes)
    upper = tuple(outcome.probability_upper for outcome in probe.outcomes)
    inflated_regret = tuple(
        decision.inflated_minimax_worst_case_regret for decision in decisions
    )
    certified = tuple(
        0.0 if decision.used_exact_fallback else 1.0 for decision in decisions
    )
    robust = tuple(
        1.0 if decision.certificate_level == "robustly-optimal" else 0.0
        for decision in decisions
    )

    regret_distribution = extremal_box_simplex_distribution(
        lower,
        upper,
        inflated_regret,
        maximize=True,
    )
    certification_distribution = extremal_box_simplex_distribution(
        lower,
        upper,
        certified,
        maximize=False,
    )
    robust_distribution = extremal_box_simplex_distribution(
        lower,
        upper,
        robust,
        maximize=False,
    )
    worst_expected = float(np.dot(regret_distribution, inflated_regret))
    worst_certification = float(np.dot(certification_distribution, certified))
    worst_robust = float(np.dot(robust_distribution, robust))
    current_regret = current_intervention.minimax_worst_case_regret
    reduction = current_regret - worst_expected
    net_value = reduction - multiplier * probe.cost
    safe = probe.physical_risk_upper <= cap + _NUMERICAL_ATOL

    reasons: list[str] = []
    if not safe:
        reasons.append("prospective-physical-risk-upper-bound-exceeds-cap")
    if reduction <= _NUMERICAL_ATOL:
        reasons.append("no-positive-guaranteed-regret-reduction")
    if worst_certification <= _NUMERICAL_ATOL:
        reasons.append("no-positive-worst-case-certification-probability")

    return RobustCertificateProbeReportV1(
        name=probe.name,
        safe=safe,
        physical_risk_upper=probe.physical_risk_upper,
        cost=probe.cost,
        current_minimax_worst_case_regret=current_regret,
        worst_case_expected_posterior_minimax_worst_case_regret=worst_expected,
        guaranteed_regret_reduction=reduction,
        worst_case_certification_probability=worst_certification,
        worst_case_robust_certification_probability=worst_robust,
        net_guaranteed_value=net_value,
        probability_lower_bounds=lower,
        probability_upper_bounds=upper,
        worst_case_regret_distribution=regret_distribution,
        worst_case_certification_distribution=certification_distribution,
        outcome_decisions=decisions,
        reason_codes=tuple(reasons),
    )


def plan_distributionally_robust_active_decision(
    current_certificate: object,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
    probes: tuple[AmbiguousCertificateProbeV1, ...] = (),
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
    minimum_net_guaranteed_value: float = 0.0,
    minimum_worst_case_certification_probability: float = 1.0,
) -> RobustActiveDecisionPlanV1:
    """Choose act, a robustly valuable probe, or the exact fallback."""

    threshold = _finite_nonnegative(
        minimum_net_guaranteed_value,
        name="minimum_net_guaranteed_value",
    )
    certification_threshold = _probability(
        minimum_worst_case_certification_probability,
        name="minimum_worst_case_certification_probability",
    )
    current = consume_query_decision_certificate(
        current_certificate,
        action_names,
        fallback_action_name=fallback_action_name,
    )
    if not current.used_exact_fallback:
        return RobustActiveDecisionPlanV1(
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
        evaluate_ambiguous_certificate_probe(
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
        and report.worst_case_certification_probability
        >= certification_threshold - _NUMERICAL_ATOL
        and report.net_guaranteed_value > threshold + _NUMERICAL_ATOL
    ]
    if eligible:
        eligible.sort(
            key=lambda report: (
                -report.net_guaranteed_value,
                -report.worst_case_certification_probability,
                -report.worst_case_robust_certification_probability,
                report.physical_risk_upper,
                report.cost,
                report.name,
            )
        )
        winner = eligible[0]
        return RobustActiveDecisionPlanV1(
            mode="probe",
            action_name=None,
            selected_probe_name=winner.name,
            fallback_action_name=fallback_action_name,
            current_intervention=current,
            probe_reports=reports,
            score=winner.net_guaranteed_value,
            reason_code="selected-positive-worst-case-decision-identification-value",
        )

    if not reports:
        reason = "no-candidate-probes"
    elif not any(report.safe for report in reports):
        reason = "no-safe-probe-under-risk-upper-bound"
    elif not any(
        report.worst_case_certification_probability
        >= certification_threshold - _NUMERICAL_ATOL
        for report in reports
    ):
        reason = "no-probe-meets-worst-case-certification-probability"
    else:
        reason = "no-positive-net-guaranteed-decision-identification-value"
    return RobustActiveDecisionPlanV1(
        mode="fallback",
        action_name=fallback_action_name,
        selected_probe_name=None,
        fallback_action_name=fallback_action_name,
        current_intervention=current,
        probe_reports=reports,
        score=0.0,
        reason_code=reason,
    )


def point_identified_probe(
    probe: CertificateProbeV1,
    *,
    regret_inflation: float = 0.0,
) -> AmbiguousCertificateProbeV1:
    """Embed an existing nominal certificate probe as a singleton ambiguity set."""

    if not isinstance(probe, CertificateProbeV1):
        raise TypeError("probe must be CertificateProbeV1")
    inflation = _finite_nonnegative(
        regret_inflation,
        name="regret_inflation",
    )
    return AmbiguousCertificateProbeV1(
        name=probe.name,
        outcomes=tuple(
            AmbiguousCertificateOutcomeV1(
                probability_lower=outcome.probability,
                probability_upper=outcome.probability,
                certificate=outcome.certificate,
                regret_inflation=inflation,
            )
            for outcome in probe.outcomes
        ),
        physical_risk_upper=probe.physical_risk,
        cost=probe.cost,
    )


def apply_probability_box_to_probe(
    probe: AmbiguousCertificateProbeV1,
    probability_box: CategoricalProbabilityBoxV1,
) -> AmbiguousCertificateProbeV1:
    """Replace a probe's outcome bounds by one simultaneous calibrated box."""

    if not isinstance(probe, AmbiguousCertificateProbeV1):
        raise TypeError("probe must be AmbiguousCertificateProbeV1")
    if not isinstance(probability_box, CategoricalProbabilityBoxV1):
        raise TypeError("probability_box must be CategoricalProbabilityBoxV1")
    if len(probe.outcomes) != probability_box.outcome_count:
        raise ValueError("probability box and probe outcome counts differ")
    return AmbiguousCertificateProbeV1(
        name=probe.name,
        outcomes=tuple(
            AmbiguousCertificateOutcomeV1(
                probability_lower=probability_box.lower_bounds[index],
                probability_upper=probability_box.upper_bounds[index],
                certificate=outcome.certificate,
                regret_inflation=outcome.regret_inflation,
            )
            for index, outcome in enumerate(probe.outcomes)
        ),
        physical_risk_upper=probe.physical_risk_upper,
        cost=probe.cost,
    )


def apply_complete_group_calibration_to_probe(
    probe: AmbiguousCertificateProbeV1,
    calibration: CompleteGroupRegretCalibrationV1,
) -> AmbiguousCertificateProbeV1:
    """Add one finite complete-group conformal margin to every probe branch."""

    if not isinstance(probe, AmbiguousCertificateProbeV1):
        raise TypeError("probe must be AmbiguousCertificateProbeV1")
    if not isinstance(calibration, CompleteGroupRegretCalibrationV1):
        raise TypeError("calibration must be CompleteGroupRegretCalibrationV1")
    margin = calibration.require_finite_margin()
    return AmbiguousCertificateProbeV1(
        name=probe.name,
        outcomes=tuple(
            AmbiguousCertificateOutcomeV1(
                probability_lower=outcome.probability_lower,
                probability_upper=outcome.probability_upper,
                certificate=outcome.certificate,
                regret_inflation=outcome.regret_inflation + margin,
            )
            for outcome in probe.outcomes
        ),
        physical_risk_upper=probe.physical_risk_upper,
        cost=probe.cost,
    )


def consume_ambiguous_probe_outcome(
    probe: AmbiguousCertificateProbeV1,
    outcome_index: int,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
) -> RobustOutcomeDecisionV1:
    """Route a realized outcome through its inflated certificate."""

    if isinstance(outcome_index, bool) or not isinstance(
        outcome_index,
        (int, np.integer),
    ):
        raise TypeError("outcome_index must be an integer")
    index = int(outcome_index)
    if not 0 <= index < len(probe.outcomes):
        raise ValueError("outcome_index is outside the probe outcome support")
    return assess_ambiguous_certificate_outcome(
        probe.outcomes[index],
        action_names,
        fallback_action_name=fallback_action_name,
    )


__all__ = [
    "ROBUST_ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY",
    "ROBUST_ACTIVE_DECISION_IDENTIFICATION_VERSION",
    "AmbiguousCertificateOutcomeV1",
    "AmbiguousCertificateProbeV1",
    "RobustActiveDecisionMode",
    "RobustActiveDecisionPlanV1",
    "RobustCertificateLevel",
    "RobustCertificateProbeReportV1",
    "RobustOutcomeDecisionV1",
    "apply_complete_group_calibration_to_probe",
    "apply_probability_box_to_probe",
    "assess_ambiguous_certificate_outcome",
    "consume_ambiguous_probe_outcome",
    "evaluate_ambiguous_certificate_probe",
    "extremal_box_simplex_distribution",
    "plan_distributionally_robust_active_decision",
    "point_identified_probe",
]
