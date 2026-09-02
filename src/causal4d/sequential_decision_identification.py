"""Exact finite-horizon decision-identifying acquisition.

The module separates robust decision certification from predictive weighting.
A terminal action is certified only when it is the unique action whose regret is
below the registered tolerance for every currently supported hypothesis.
Predictive weights influence probe-outcome probabilities and expected sensing
cost, but never relax the support-wise certificate.

For a fixed finite interface, hypotheses with the same normalized action-loss
signature and the same likelihood row for every registered probe form the
coarsest probe--action quotient. Aggregating predictive mass over that quotient
preserves every finite-horizon adaptive policy produced here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, product
from numbers import Integral, Real
from typing import Literal

import numpy as np

SequentialDecisionMode = Literal["act", "probe", "fallback"]
SequentialObjective = Literal["expected_cost", "worst_case_cost"]

SEQUENTIAL_DECISION_IDENTIFICATION_VERSION = 1
SEQUENTIAL_DECISION_IDENTIFICATION_CLAIM_BOUNDARY = (
    "Exact only for the supplied finite hypothesis support, terminal losses, "
    "conditionally independent finite probe channels, additive registered probe "
    "cost/risk charges, regret tolerance, and finite horizon. It does not validate "
    "the physical hypotheses, probe models, costs, risks, target transport, "
    "exchangeability, deployment authorization, or safety."
)

_ATOL = 1e-12


def _nonempty_name(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return result


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return result


def _loss_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("losses must be a nonempty hypothesis-by-action matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("losses must be finite")
    return matrix


def _weights(value: object, *, expected_size: int | None = None) -> np.ndarray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.ndim != 1 or weights.size == 0:
        raise ValueError("weights must be a nonempty vector")
    if expected_size is not None and weights.size != expected_size:
        raise ValueError("weights do not match the hypothesis count")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("weights must be finite and nonnegative")
    total = float(np.sum(weights))
    if total <= 0.0:
        raise ValueError("weights must have positive total mass")
    return weights / total


def _canonical(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class FiniteProbe:
    """One registered finite-outcome diagnostic probe."""

    name: str
    likelihood: object
    cost: float = 0.0
    risk: float = 0.0
    outcome_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _nonempty_name(self.name, name="probe name")
        likelihood = np.asarray(self.likelihood, dtype=np.float64)
        if likelihood.ndim != 2 or min(likelihood.shape) == 0:
            raise ValueError(
                "probe likelihood must be a nonempty hypothesis-by-outcome matrix"
            )
        if not np.isfinite(likelihood).all() or np.any(likelihood < 0.0):
            raise ValueError("probe likelihood must be finite and nonnegative")
        if not np.allclose(
            np.sum(likelihood, axis=1),
            1.0,
            rtol=0.0,
            atol=_ATOL,
        ):
            raise ValueError("every probe likelihood row must sum to one")
        names = tuple(self.outcome_names)
        if not names:
            names = tuple(f"outcome-{index}" for index in range(likelihood.shape[1]))
        if len(names) != likelihood.shape[1]:
            raise ValueError("outcome_names do not match the probe outcome count")
        names = tuple(
            _nonempty_name(item, name="probe outcome name") for item in names
        )
        if len(set(names)) != len(names):
            raise ValueError("probe outcome names must be unique")
        canonical = tuple(
            tuple(_canonical(value) for value in row) for row in likelihood
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "likelihood", canonical)
        object.__setattr__(self, "cost", _finite_nonnegative(self.cost, name="cost"))
        object.__setattr__(self, "risk", _finite_nonnegative(self.risk, name="risk"))
        object.__setattr__(self, "outcome_names", names)

    @property
    def hypothesis_count(self) -> int:
        return len(self.likelihood)

    @property
    def outcome_count(self) -> int:
        return len(self.outcome_names)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "likelihood": [list(row) for row in self.likelihood],
            "cost": self.cost,
            "risk": self.risk,
            "outcome_names": list(self.outcome_names),
        }


@dataclass(frozen=True)
class FiniteDecisionCertificate:
    """Support-wise terminal-action regret certificate."""

    weights: tuple[float, ...]
    support_indices: tuple[int, ...]
    worst_case_regret: tuple[float, ...]
    admissible_action_indices: tuple[int, ...]
    selected_action_index: int | None
    minimax_action_index: int
    minimax_worst_case_regret: float
    regret_tolerance: float
    reason_code: str

    @property
    def certified(self) -> bool:
        return self.selected_action_index is not None

    def as_dict(self) -> dict[str, object]:
        return {
            "weights": list(self.weights),
            "support_indices": list(self.support_indices),
            "worst_case_regret": list(self.worst_case_regret),
            "admissible_action_indices": list(self.admissible_action_indices),
            "selected_action_index": self.selected_action_index,
            "minimax_action_index": self.minimax_action_index,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "regret_tolerance": self.regret_tolerance,
            "certified": self.certified,
            "reason_code": self.reason_code,
        }


def finite_decision_certificate(
    losses: object,
    weights: object,
    *,
    regret_tolerance: float = 0.0,
) -> FiniteDecisionCertificate:
    """Certify one action uniformly over every positive-weight hypothesis."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    tolerance = _finite_nonnegative(
        regret_tolerance,
        name="regret_tolerance",
    )
    support = np.flatnonzero(probability > 0.0)
    oracle = np.min(loss_matrix[support], axis=1)
    regrets = np.max(loss_matrix[support] - oracle[:, None], axis=0)
    regrets = np.maximum(regrets, 0.0)
    admissible = np.flatnonzero(regrets <= tolerance + _ATOL)
    minimum = float(np.min(regrets))
    minimax = int(np.flatnonzero(np.isclose(regrets, minimum, atol=_ATOL))[0])
    if admissible.size == 1:
        selected: int | None = int(admissible[0])
        reason = "unique-support-wise-admissible-action"
    elif admissible.size == 0:
        selected = None
        reason = "no-support-wise-admissible-action"
    else:
        selected = None
        reason = "multiple-support-wise-admissible-actions"
    return FiniteDecisionCertificate(
        weights=tuple(float(value) for value in probability),
        support_indices=tuple(int(value) for value in support),
        worst_case_regret=tuple(float(value) for value in regrets),
        admissible_action_indices=tuple(int(value) for value in admissible),
        selected_action_index=selected,
        minimax_action_index=minimax,
        minimax_worst_case_regret=minimum,
        regret_tolerance=tolerance,
        reason_code=reason,
    )


def _validate_probes(
    probes: Sequence[FiniteProbe],
    *,
    hypothesis_count: int,
) -> tuple[FiniteProbe, ...]:
    roster = tuple(probes)
    if not all(isinstance(probe, FiniteProbe) for probe in roster):
        raise TypeError("probes must contain FiniteProbe values")
    if len({probe.name for probe in roster}) != len(roster):
        raise ValueError("probe names must be unique")
    for probe in roster:
        if probe.hypothesis_count != hypothesis_count:
            raise ValueError(f"probe {probe.name!r} has the wrong hypothesis count")
    return roster


def _probe_outcome_probabilities(
    weights: np.ndarray,
    probe: FiniteProbe,
) -> np.ndarray:
    likelihood = np.asarray(probe.likelihood, dtype=np.float64)
    probabilities = weights @ likelihood
    probabilities[np.abs(probabilities) <= _ATOL] = 0.0
    return probabilities


def _posterior_weights(
    weights: np.ndarray,
    probe: FiniteProbe,
    outcome_index: int,
    outcome_probability: float,
) -> np.ndarray:
    likelihood = np.asarray(probe.likelihood, dtype=np.float64)
    posterior = weights * likelihood[:, outcome_index]
    return posterior / outcome_probability


def _mutual_information(weights: np.ndarray, probe: FiniteProbe) -> float:
    likelihood = np.asarray(probe.likelihood, dtype=np.float64)
    outcome_probability = weights @ likelihood
    total = 0.0
    for hypothesis in range(weights.size):
        if weights[hypothesis] <= 0.0:
            continue
        for outcome in range(probe.outcome_count):
            conditional = likelihood[hypothesis, outcome]
            marginal = outcome_probability[outcome]
            if conditional > 0.0 and marginal > 0.0:
                total += (
                    weights[hypothesis]
                    * conditional
                    * float(np.log(conditional / marginal))
                )
    return max(total, 0.0)


@dataclass(frozen=True)
class FiniteProbeEvaluation:
    """One-step decision value of one finite probe."""

    probe_index: int
    probe_name: str
    safe: bool
    outcome_probabilities: tuple[float, ...]
    posterior_minimax_regrets: tuple[float | None, ...]
    posterior_action_indices: tuple[int | None, ...]
    expected_post_probe_regret: float
    worst_post_probe_regret: float
    expected_regret_reduction: float
    certification_probability: float
    all_possible_outcomes_certified: bool
    mutual_information: float
    net_value: float

    def as_dict(self) -> dict[str, object]:
        return {
            "probe_index": self.probe_index,
            "probe_name": self.probe_name,
            "safe": self.safe,
            "outcome_probabilities": list(self.outcome_probabilities),
            "posterior_minimax_regrets": list(self.posterior_minimax_regrets),
            "posterior_action_indices": list(self.posterior_action_indices),
            "expected_post_probe_regret": self.expected_post_probe_regret,
            "worst_post_probe_regret": self.worst_post_probe_regret,
            "expected_regret_reduction": self.expected_regret_reduction,
            "certification_probability": self.certification_probability,
            "all_possible_outcomes_certified": (
                self.all_possible_outcomes_certified
            ),
            "mutual_information": self.mutual_information,
            "net_value": self.net_value,
        }


def evaluate_probe(
    losses: object,
    weights: object,
    probe: FiniteProbe,
    *,
    probe_index: int = 0,
    regret_tolerance: float = 0.0,
    risk_cap: float = 1.0,
    cost_multiplier: float = 1.0,
) -> FiniteProbeEvaluation:
    """Return exact one-step support-wise decision value for ``probe``."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    if not isinstance(probe, FiniteProbe):
        raise TypeError("probe must be a FiniteProbe")
    if probe.hypothesis_count != loss_matrix.shape[0]:
        raise ValueError("probe has the wrong hypothesis count")
    index = _integer(probe_index, name="probe_index")
    cap = _finite_nonnegative(risk_cap, name="risk_cap")
    multiplier = _finite_nonnegative(cost_multiplier, name="cost_multiplier")
    current = finite_decision_certificate(
        loss_matrix,
        probability,
        regret_tolerance=regret_tolerance,
    )
    outcome_probabilities = _probe_outcome_probabilities(probability, probe)
    posterior_regrets: list[float | None] = []
    posterior_actions: list[int | None] = []
    expected_regret = 0.0
    worst_regret = 0.0
    certification_probability = 0.0
    possible_count = 0
    certified_count = 0
    for outcome, outcome_probability in enumerate(outcome_probabilities):
        if outcome_probability <= 0.0:
            posterior_regrets.append(None)
            posterior_actions.append(None)
            continue
        possible_count += 1
        posterior = _posterior_weights(
            probability,
            probe,
            outcome,
            float(outcome_probability),
        )
        certificate = finite_decision_certificate(
            loss_matrix,
            posterior,
            regret_tolerance=regret_tolerance,
        )
        posterior_regrets.append(certificate.minimax_worst_case_regret)
        posterior_actions.append(certificate.selected_action_index)
        expected_regret += (
            float(outcome_probability) * certificate.minimax_worst_case_regret
        )
        worst_regret = max(worst_regret, certificate.minimax_worst_case_regret)
        if certificate.certified:
            certified_count += 1
            certification_probability += float(outcome_probability)
    all_certified = possible_count > 0 and certified_count == possible_count
    reduction = current.minimax_worst_case_regret - expected_regret
    return FiniteProbeEvaluation(
        probe_index=index,
        probe_name=probe.name,
        safe=probe.risk <= cap + _ATOL,
        outcome_probabilities=tuple(float(value) for value in outcome_probabilities),
        posterior_minimax_regrets=tuple(posterior_regrets),
        posterior_action_indices=tuple(posterior_actions),
        expected_post_probe_regret=max(expected_regret, 0.0),
        worst_post_probe_regret=max(worst_regret, 0.0),
        expected_regret_reduction=reduction,
        certification_probability=min(max(certification_probability, 0.0), 1.0),
        all_possible_outcomes_certified=all_certified,
        mutual_information=_mutual_information(probability, probe),
        net_value=reduction - multiplier * probe.cost,
    )


@dataclass(frozen=True)
class ActiveDecisionSelection:
    """One-step act, probe, or exact-fallback selection."""

    mode: SequentialDecisionMode
    action_index: int | None
    probe_index: int | None
    current_certificate: FiniteDecisionCertificate
    probe_evaluations: tuple[FiniteProbeEvaluation, ...]
    reason_code: str

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "action_index": self.action_index,
            "probe_index": self.probe_index,
            "current_certificate": self.current_certificate.as_dict(),
            "probe_evaluations": [item.as_dict() for item in self.probe_evaluations],
            "reason_code": self.reason_code,
        }


def select_active_decision(
    losses: object,
    weights: object,
    probes: Sequence[FiniteProbe],
    *,
    regret_tolerance: float = 0.0,
    risk_cap: float = 1.0,
    cost_multiplier: float = 1.0,
    minimum_net_value: float = 0.0,
    require_all_outcomes_certified: bool = True,
) -> ActiveDecisionSelection:
    """Choose an immediate action, one decision-valued probe, or fallback."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_probes(probes, hypothesis_count=loss_matrix.shape[0])
    current = finite_decision_certificate(
        loss_matrix,
        probability,
        regret_tolerance=regret_tolerance,
    )
    if current.certified:
        return ActiveDecisionSelection(
            mode="act",
            action_index=current.selected_action_index,
            probe_index=None,
            current_certificate=current,
            probe_evaluations=(),
            reason_code="current-decision-certified",
        )
    threshold = _finite_nonnegative(
        minimum_net_value,
        name="minimum_net_value",
    )
    evaluations = tuple(
        evaluate_probe(
            loss_matrix,
            probability,
            probe,
            probe_index=index,
            regret_tolerance=regret_tolerance,
            risk_cap=risk_cap,
            cost_multiplier=cost_multiplier,
        )
        for index, probe in enumerate(roster)
    )
    eligible = [
        item
        for item in evaluations
        if item.safe
        and item.net_value > threshold + _ATOL
        and (
            item.all_possible_outcomes_certified
            if require_all_outcomes_certified
            else item.certification_probability > 0.0
        )
    ]
    if eligible:
        eligible.sort(
            key=lambda item: (
                -item.net_value,
                -item.certification_probability,
                roster[item.probe_index].risk,
                roster[item.probe_index].cost,
                item.probe_name,
            )
        )
        selected = eligible[0]
        return ActiveDecisionSelection(
            mode="probe",
            action_index=None,
            probe_index=selected.probe_index,
            current_certificate=current,
            probe_evaluations=evaluations,
            reason_code="selected-positive-decision-value-probe",
        )
    return ActiveDecisionSelection(
        mode="fallback",
        action_index=None,
        probe_index=None,
        current_certificate=current,
        probe_evaluations=evaluations,
        reason_code="no-eligible-one-step-probe",
    )


def select_information_probe(
    weights: object,
    probes: Sequence[FiniteProbe],
    *,
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
) -> int | None:
    """Select the safe probe with maximum generic mutual information."""

    roster = tuple(probes)
    if not roster:
        return None
    probability = _weights(weights, expected_size=roster[0].hypothesis_count)
    roster = _validate_probes(roster, hypothesis_count=probability.size)
    cap = _finite_nonnegative(risk_cap, name="risk_cap")
    multiplier = _finite_nonnegative(cost_multiplier, name="cost_multiplier")
    candidates = [
        (
            _mutual_information(probability, probe) - multiplier * probe.cost,
            probe.risk,
            probe.cost,
            probe.name,
            index,
        )
        for index, probe in enumerate(roster)
        if probe.risk <= cap + _ATOL
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
    return int(candidates[0][4])


@dataclass(frozen=True)
class ProbeActionQuotient:
    """Coarsest exact quotient for the registered probe--action interface."""

    normalized_losses: tuple[tuple[float, ...], ...]
    class_weights: tuple[float, ...]
    class_index: tuple[int, ...]
    class_members: tuple[tuple[int, ...], ...]
    probes: tuple[FiniteProbe, ...]

    @property
    def original_hypothesis_count(self) -> int:
        return len(self.class_index)

    @property
    def class_count(self) -> int:
        return len(self.class_members)

    def as_dict(self) -> dict[str, object]:
        return {
            "normalized_losses": [list(row) for row in self.normalized_losses],
            "class_weights": list(self.class_weights),
            "class_index": list(self.class_index),
            "class_members": [list(row) for row in self.class_members],
            "probes": [probe.as_dict() for probe in self.probes],
        }


def build_probe_action_quotient(
    losses: object,
    weights: object,
    probes: Sequence[FiniteProbe],
) -> ProbeActionQuotient:
    """Aggregate states with identical action gaps and all probe likelihood rows."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_probes(probes, hypothesis_count=loss_matrix.shape[0])
    normalized = loss_matrix - loss_matrix[:, [0]]
    signatures: dict[tuple[float, ...], int] = {}
    class_index: list[int] = []
    members: list[list[int]] = []
    representatives: list[int] = []
    for hypothesis in range(loss_matrix.shape[0]):
        signature_values = [
            _canonical(value) for value in normalized[hypothesis]
        ]
        for probe in roster:
            signature_values.extend(
                _canonical(value) for value in probe.likelihood[hypothesis]
            )
        signature = tuple(signature_values)
        class_id = signatures.get(signature)
        if class_id is None:
            class_id = len(members)
            signatures[signature] = class_id
            members.append([])
            representatives.append(hypothesis)
        members[class_id].append(hypothesis)
        class_index.append(class_id)
    class_weights = np.bincount(
        np.asarray(class_index, dtype=np.int64),
        weights=probability,
        minlength=len(members),
    )
    quotient_probes = tuple(
        FiniteProbe(
            name=probe.name,
            likelihood=tuple(
                probe.likelihood[representative]
                for representative in representatives
            ),
            cost=probe.cost,
            risk=probe.risk,
            outcome_names=probe.outcome_names,
        )
        for probe in roster
    )
    return ProbeActionQuotient(
        normalized_losses=tuple(
            tuple(_canonical(value) for value in normalized[representative])
            for representative in representatives
        ),
        class_weights=tuple(float(value) for value in class_weights),
        class_index=tuple(class_index),
        class_members=tuple(tuple(row) for row in members),
        probes=quotient_probes,
    )


@dataclass(frozen=True)
class SequentialOutcomeBranch:
    outcome_index: int
    outcome_name: str
    probability: float
    policy: SequentialDecisionPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome_index": self.outcome_index,
            "outcome_name": self.outcome_name,
            "probability": self.probability,
            "policy": self.policy.as_dict(),
        }


@dataclass(frozen=True)
class SequentialDecisionPolicy:
    """Exact finite-horizon adaptive acquisition policy tree."""

    mode: SequentialDecisionMode
    action_index: int | None
    probe_index: int | None
    probe_name: str | None
    certificate: FiniteDecisionCertificate
    outcomes: tuple[SequentialOutcomeBranch, ...]
    expected_probe_cost: float
    worst_case_probe_cost: float
    worst_case_risk: float
    guaranteed_certification: bool
    horizon_remaining: int
    reason_code: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": SEQUENTIAL_DECISION_IDENTIFICATION_VERSION,
            "mode": self.mode,
            "action_index": self.action_index,
            "probe_index": self.probe_index,
            "probe_name": self.probe_name,
            "certificate": self.certificate.as_dict(),
            "outcomes": [branch.as_dict() for branch in self.outcomes],
            "expected_probe_cost": self.expected_probe_cost,
            "worst_case_probe_cost": self.worst_case_probe_cost,
            "worst_case_risk": self.worst_case_risk,
            "guaranteed_certification": self.guaranteed_certification,
            "horizon_remaining": self.horizon_remaining,
            "reason_code": self.reason_code,
            "claim_boundary": SEQUENTIAL_DECISION_IDENTIFICATION_CLAIM_BOUNDARY,
        }


def solve_sequential_decision(
    losses: object,
    weights: object,
    probes: Sequence[FiniteProbe],
    *,
    regret_tolerance: float = 0.0,
    max_probes: int = 1,
    risk_budget: float = 1.0,
    objective: SequentialObjective = "expected_cost",
    maximum_nodes: int = 100_000,
) -> SequentialDecisionPolicy:
    """Find the minimum-cost adaptive policy certifying every possible branch.

    Each registered probe may be acquired at most once. A probe policy is
    feasible only when every outcome with positive predictive probability leads
    to a certified terminal action within the remaining horizon and cumulative
    registered risk budget. If no such policy exists, the exact fallback mode is
    returned rather than a latent-state completion.
    """

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_probes(probes, hypothesis_count=loss_matrix.shape[0])
    horizon = _integer(max_probes, name="max_probes")
    budget = _finite_nonnegative(risk_budget, name="risk_budget")
    node_limit = _integer(maximum_nodes, name="maximum_nodes", minimum=1)
    if objective not in ("expected_cost", "worst_case_cost"):
        raise ValueError("objective must be 'expected_cost' or 'worst_case_cost'")
    nodes = 0

    def recurse(
        current_weights: np.ndarray,
        remaining: tuple[int, ...],
        remaining_horizon: int,
        remaining_risk: float,
    ) -> SequentialDecisionPolicy:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise RuntimeError("sequential policy search exceeded maximum_nodes")
        certificate = finite_decision_certificate(
            loss_matrix,
            current_weights,
            regret_tolerance=regret_tolerance,
        )
        if certificate.certified:
            return SequentialDecisionPolicy(
                mode="act",
                action_index=certificate.selected_action_index,
                probe_index=None,
                probe_name=None,
                certificate=certificate,
                outcomes=(),
                expected_probe_cost=0.0,
                worst_case_probe_cost=0.0,
                worst_case_risk=0.0,
                guaranteed_certification=True,
                horizon_remaining=remaining_horizon,
                reason_code="current-decision-certified",
            )
        if remaining_horizon == 0 or not remaining:
            return SequentialDecisionPolicy(
                mode="fallback",
                action_index=None,
                probe_index=None,
                probe_name=None,
                certificate=certificate,
                outcomes=(),
                expected_probe_cost=0.0,
                worst_case_probe_cost=0.0,
                worst_case_risk=0.0,
                guaranteed_certification=False,
                horizon_remaining=remaining_horizon,
                reason_code="horizon-exhausted-without-certificate",
            )

        candidates: list[SequentialDecisionPolicy] = []
        for probe_index in remaining:
            probe = roster[probe_index]
            if probe.risk > remaining_risk + _ATOL:
                continue
            outcome_probabilities = _probe_outcome_probabilities(
                current_weights,
                probe,
            )
            next_remaining = tuple(
                index for index in remaining if index != probe_index
            )
            branches: list[SequentialOutcomeBranch] = []
            feasible = True
            expected_child_cost = 0.0
            worst_child_cost = 0.0
            worst_child_risk = 0.0
            for outcome, outcome_probability in enumerate(outcome_probabilities):
                if outcome_probability <= 0.0:
                    continue
                posterior = _posterior_weights(
                    current_weights,
                    probe,
                    outcome,
                    float(outcome_probability),
                )
                child = recurse(
                    posterior,
                    next_remaining,
                    remaining_horizon - 1,
                    max(remaining_risk - probe.risk, 0.0),
                )
                if not child.guaranteed_certification:
                    feasible = False
                    break
                branches.append(
                    SequentialOutcomeBranch(
                        outcome_index=outcome,
                        outcome_name=probe.outcome_names[outcome],
                        probability=float(outcome_probability),
                        policy=child,
                    )
                )
                expected_child_cost += (
                    float(outcome_probability) * child.expected_probe_cost
                )
                worst_child_cost = max(
                    worst_child_cost,
                    child.worst_case_probe_cost,
                )
                worst_child_risk = max(worst_child_risk, child.worst_case_risk)
            if not feasible or not branches:
                continue
            candidates.append(
                SequentialDecisionPolicy(
                    mode="probe",
                    action_index=None,
                    probe_index=probe_index,
                    probe_name=probe.name,
                    certificate=certificate,
                    outcomes=tuple(branches),
                    expected_probe_cost=probe.cost + expected_child_cost,
                    worst_case_probe_cost=probe.cost + worst_child_cost,
                    worst_case_risk=probe.risk + worst_child_risk,
                    guaranteed_certification=True,
                    horizon_remaining=remaining_horizon,
                    reason_code="minimum-cost-guaranteed-sequential-certificate",
                )
            )
        if not candidates:
            return SequentialDecisionPolicy(
                mode="fallback",
                action_index=None,
                probe_index=None,
                probe_name=None,
                certificate=certificate,
                outcomes=(),
                expected_probe_cost=0.0,
                worst_case_probe_cost=0.0,
                worst_case_risk=0.0,
                guaranteed_certification=False,
                horizon_remaining=remaining_horizon,
                reason_code="no-safe-guaranteed-sequential-certificate",
            )
        if objective == "expected_cost":
            candidates.sort(
                key=lambda item: (
                    item.expected_probe_cost,
                    item.worst_case_probe_cost,
                    item.worst_case_risk,
                    item.probe_name or "",
                )
            )
        else:
            candidates.sort(
                key=lambda item: (
                    item.worst_case_probe_cost,
                    item.expected_probe_cost,
                    item.worst_case_risk,
                    item.probe_name or "",
                )
            )
        return candidates[0]

    return recurse(probability, tuple(range(len(roster))), horizon, budget)


@dataclass(frozen=True)
class NonAdaptiveProbeSet:
    probe_indices: tuple[int, ...]
    probe_names: tuple[str, ...]
    total_cost: float
    total_risk: float
    guaranteed_certification: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "probe_indices": list(self.probe_indices),
            "probe_names": list(self.probe_names),
            "total_cost": self.total_cost,
            "total_risk": self.total_risk,
            "guaranteed_certification": self.guaranteed_certification,
        }


def minimum_nonadaptive_probe_set(
    losses: object,
    weights: object,
    probes: Sequence[FiniteProbe],
    *,
    regret_tolerance: float = 0.0,
    max_probes: int | None = None,
    risk_budget: float = 1.0,
    maximum_outcome_combinations: int = 100_000,
) -> NonAdaptiveProbeSet | None:
    """Return the cheapest fixed probe set certifying every joint outcome."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_probes(probes, hypothesis_count=loss_matrix.shape[0])
    limit = len(roster) if max_probes is None else _integer(
        max_probes,
        name="max_probes",
    )
    limit = min(limit, len(roster))
    budget = _finite_nonnegative(risk_budget, name="risk_budget")
    combination_limit = _integer(
        maximum_outcome_combinations,
        name="maximum_outcome_combinations",
        minimum=1,
    )
    candidates: list[NonAdaptiveProbeSet] = []
    evaluated = 0
    for count in range(limit + 1):
        for indices in combinations(range(len(roster)), count):
            total_risk = float(sum(roster[index].risk for index in indices))
            if total_risk > budget + _ATOL:
                continue
            sufficient = True
            outcome_ranges = [range(roster[index].outcome_count) for index in indices]
            for outcomes in product(*outcome_ranges):
                evaluated += 1
                if evaluated > combination_limit:
                    raise RuntimeError(
                        "nonadaptive search exceeded maximum_outcome_combinations"
                    )
                posterior = probability.copy()
                possible = True
                for probe_index, outcome in zip(indices, outcomes, strict=True):
                    probe = roster[probe_index]
                    likelihood = np.asarray(probe.likelihood, dtype=np.float64)
                    posterior *= likelihood[:, outcome]
                    mass = float(np.sum(posterior))
                    if mass <= 0.0:
                        possible = False
                        break
                    posterior /= mass
                if not possible:
                    continue
                certificate = finite_decision_certificate(
                    loss_matrix,
                    posterior,
                    regret_tolerance=regret_tolerance,
                )
                if not certificate.certified:
                    sufficient = False
                    break
            if sufficient:
                candidates.append(
                    NonAdaptiveProbeSet(
                        probe_indices=tuple(indices),
                        probe_names=tuple(roster[index].name for index in indices),
                        total_cost=float(sum(roster[index].cost for index in indices)),
                        total_risk=total_risk,
                        guaranteed_certification=True,
                    )
                )
    if not candidates:
        return None
    candidates.sort(
        key=lambda item: (
            item.total_cost,
            item.total_risk,
            len(item.probe_indices),
            item.probe_names,
        )
    )
    return candidates[0]


__all__ = [
    "SEQUENTIAL_DECISION_IDENTIFICATION_CLAIM_BOUNDARY",
    "SEQUENTIAL_DECISION_IDENTIFICATION_VERSION",
    "ActiveDecisionSelection",
    "FiniteDecisionCertificate",
    "FiniteProbe",
    "FiniteProbeEvaluation",
    "NonAdaptiveProbeSet",
    "ProbeActionQuotient",
    "SequentialDecisionMode",
    "SequentialDecisionPolicy",
    "SequentialObjective",
    "SequentialOutcomeBranch",
    "build_probe_action_quotient",
    "evaluate_probe",
    "finite_decision_certificate",
    "minimum_nonadaptive_probe_set",
    "select_active_decision",
    "select_information_probe",
    "solve_sequential_decision",
]
