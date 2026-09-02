"""Exact decision identification under state-changing diagnostic interventions.

Unlike a passive probe, a controlled intervention can both change the hidden
physical state and emit an observation.  The finite kernel is

    K_e[h, h_next, y] = P(H_next=h_next, Y=y | H=h, e).

The solver propagates complete beliefs through these kernels, certifies terminal
actions support-wise, and returns exact fallback whenever no branch-complete
finite-horizon intervention policy exists.

For a fixed registered interface, the coarsest controlled decision quotient is
the greatest refinement of terminal action-loss equivalence that is lumpable
for every registered transition-observation kernel.  Aggregating belief mass on
that quotient preserves every finite-horizon policy produced here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import permutations
from numbers import Integral, Real
from typing import Literal

import numpy as np

from .sequential_decision_identification import (
    FiniteDecisionCertificate,
    FiniteProbe,
    finite_decision_certificate,
)

CONTROLLED_DECISION_IDENTIFICATION_VERSION = 1
CONTROLLED_DECISION_IDENTIFICATION_CLAIM_BOUNDARY = (
    "Exact only for the supplied finite physical-state support, terminal losses, "
    "registered finite transition-observation kernels, additive intervention "
    "cost/risk charges, regret tolerance, and finite horizon. It does not validate "
    "the physical state roster, transition or sensor models, costs, risks, target "
    "transport, exchangeability, deployment authorization, or safety."
)

ControlledDecisionMode = Literal["act", "intervene", "fallback"]
ControlledObjective = Literal["expected_cost", "worst_case_cost"]

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
        raise ValueError("losses must be a nonempty state-by-action matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("losses must be finite")
    return matrix


def _weights(value: object, *, expected_size: int) -> np.ndarray:
    weights = np.asarray(value, dtype=np.float64)
    if weights.ndim != 1 or weights.size != expected_size:
        raise ValueError("weights do not match the physical-state count")
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
class FiniteControlledIntervention:
    """One finite state-changing intervention and its joint observation kernel."""

    name: str
    kernel: object
    cost: float = 0.0
    risk: float = 0.0
    outcome_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _nonempty_name(self.name, name="intervention name")
        kernel = np.asarray(self.kernel, dtype=np.float64)
        if kernel.ndim != 3 or min(kernel.shape) == 0:
            raise ValueError(
                "intervention kernel must have shape "
                "(current_state, next_state, outcome)"
            )
        if kernel.shape[0] != kernel.shape[1]:
            raise ValueError("current and next physical-state counts must match")
        if not np.isfinite(kernel).all() or np.any(kernel < 0.0):
            raise ValueError("intervention kernel must be finite and nonnegative")
        if not np.allclose(
            np.sum(kernel, axis=(1, 2)),
            1.0,
            rtol=0.0,
            atol=_ATOL,
        ):
            raise ValueError(
                "every current-state intervention-kernel row must sum to one"
            )
        names = tuple(self.outcome_names)
        if not names:
            names = tuple(f"outcome-{index}" for index in range(kernel.shape[2]))
        if len(names) != kernel.shape[2]:
            raise ValueError("outcome_names do not match the kernel outcome count")
        names = tuple(
            _nonempty_name(item, name="intervention outcome name") for item in names
        )
        if len(set(names)) != len(names):
            raise ValueError("intervention outcome names must be unique")
        canonical = tuple(
            tuple(
                tuple(_canonical(value) for value in outcome_row)
                for outcome_row in next_rows
            )
            for next_rows in kernel
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "kernel", canonical)
        object.__setattr__(self, "cost", _finite_nonnegative(self.cost, name="cost"))
        object.__setattr__(self, "risk", _finite_nonnegative(self.risk, name="risk"))
        object.__setattr__(self, "outcome_names", names)

    @property
    def state_count(self) -> int:
        return len(self.kernel)

    @property
    def outcome_count(self) -> int:
        return len(self.outcome_names)

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kernel": [
                [list(outcome_row) for outcome_row in next_rows]
                for next_rows in self.kernel
            ],
            "cost": self.cost,
            "risk": self.risk,
            "outcome_names": list(self.outcome_names),
        }


def factorized_controlled_intervention(
    name: str,
    transition: object,
    observation_likelihood: object,
    *,
    cost: float = 0.0,
    risk: float = 0.0,
    outcome_names: tuple[str, ...] = (),
) -> FiniteControlledIntervention:
    """Combine a state transition and next-state observation channel.

    ``transition[h, h_next]`` is row stochastic and
    ``observation_likelihood[h_next, y]`` is the observation law after the
    transition.  The returned joint kernel is their product.
    """

    transition_matrix = np.asarray(transition, dtype=np.float64)
    likelihood = np.asarray(observation_likelihood, dtype=np.float64)
    if (
        transition_matrix.ndim != 2
        or transition_matrix.shape[0] == 0
        or transition_matrix.shape[0] != transition_matrix.shape[1]
    ):
        raise ValueError("transition must be a nonempty square matrix")
    if not np.isfinite(transition_matrix).all() or np.any(transition_matrix < 0.0):
        raise ValueError("transition must be finite and nonnegative")
    if not np.allclose(
        np.sum(transition_matrix, axis=1),
        1.0,
        rtol=0.0,
        atol=_ATOL,
    ):
        raise ValueError("every transition row must sum to one")
    if (
        likelihood.ndim != 2
        or likelihood.shape[0] != transition_matrix.shape[0]
        or likelihood.shape[1] == 0
    ):
        raise ValueError("observation_likelihood must have shape (next_state, outcome)")
    if not np.isfinite(likelihood).all() or np.any(likelihood < 0.0):
        raise ValueError("observation_likelihood must be finite and nonnegative")
    if not np.allclose(
        np.sum(likelihood, axis=1),
        1.0,
        rtol=0.0,
        atol=_ATOL,
    ):
        raise ValueError("every observation-likelihood row must sum to one")
    kernel = transition_matrix[:, :, None] * likelihood[None, :, :]
    return FiniteControlledIntervention(
        name=name,
        kernel=kernel,
        cost=cost,
        risk=risk,
        outcome_names=outcome_names,
    )


def controlled_intervention_from_probe(
    probe: FiniteProbe,
) -> FiniteControlledIntervention:
    """Embed a passive probe as an identity-state controlled intervention."""

    if not isinstance(probe, FiniteProbe):
        raise TypeError("probe must be a FiniteProbe")
    likelihood = np.asarray(probe.likelihood, dtype=np.float64)
    kernel = np.zeros(
        (probe.hypothesis_count, probe.hypothesis_count, probe.outcome_count),
        dtype=np.float64,
    )
    for state in range(probe.hypothesis_count):
        kernel[state, state] = likelihood[state]
    return FiniteControlledIntervention(
        name=probe.name,
        kernel=kernel,
        cost=probe.cost,
        risk=probe.risk,
        outcome_names=probe.outcome_names,
    )


def marginal_static_probe(
    intervention: FiniteControlledIntervention,
) -> FiniteProbe:
    """Discard state transitions and retain only the observation marginal."""

    if not isinstance(intervention, FiniteControlledIntervention):
        raise TypeError("intervention must be a FiniteControlledIntervention")
    kernel = np.asarray(intervention.kernel, dtype=np.float64)
    return FiniteProbe(
        name=intervention.name,
        likelihood=np.sum(kernel, axis=1),
        cost=intervention.cost,
        risk=intervention.risk,
        outcome_names=intervention.outcome_names,
    )


def _validate_interventions(
    interventions: Sequence[FiniteControlledIntervention],
    *,
    state_count: int,
) -> tuple[FiniteControlledIntervention, ...]:
    roster = tuple(interventions)
    if not all(isinstance(item, FiniteControlledIntervention) for item in roster):
        raise TypeError(
            "interventions must contain FiniteControlledIntervention values"
        )
    if len({item.name for item in roster}) != len(roster):
        raise ValueError("intervention names must be unique")
    for item in roster:
        if item.state_count != state_count:
            raise ValueError(
                f"intervention {item.name!r} has the wrong physical-state count"
            )
    return roster


def controlled_outcome_probabilities(
    weights: object,
    intervention: FiniteControlledIntervention,
) -> tuple[float, ...]:
    """Return predictive outcome probabilities under one intervention."""

    if not isinstance(intervention, FiniteControlledIntervention):
        raise TypeError("intervention must be a FiniteControlledIntervention")
    probability = _weights(weights, expected_size=intervention.state_count)
    kernel = np.asarray(intervention.kernel, dtype=np.float64)
    outcomes = np.einsum("i,ijo->o", probability, kernel, optimize=True)
    outcomes[np.abs(outcomes) <= _ATOL] = 0.0
    return tuple(float(value) for value in outcomes)


def controlled_posterior_weights(
    weights: object,
    intervention: FiniteControlledIntervention,
    outcome_index: int,
) -> tuple[float, ...]:
    """Propagate and condition a complete belief through one intervention."""

    if not isinstance(intervention, FiniteControlledIntervention):
        raise TypeError("intervention must be a FiniteControlledIntervention")
    probability = _weights(weights, expected_size=intervention.state_count)
    outcome = _integer(outcome_index, name="outcome_index")
    if outcome >= intervention.outcome_count:
        raise ValueError("outcome_index is outside the intervention outcome roster")
    kernel = np.asarray(intervention.kernel, dtype=np.float64)
    next_mass = np.einsum(
        "i,ij->j",
        probability,
        kernel[:, :, outcome],
        optimize=True,
    )
    mass = float(np.sum(next_mass))
    if mass <= 0.0:
        raise ValueError("selected outcome has zero predictive probability")
    next_mass /= mass
    next_mass[np.abs(next_mass) <= _ATOL] = 0.0
    return tuple(float(value) for value in next_mass)


@dataclass(frozen=True)
class ControlledOutcomeBranch:
    outcome_index: int
    outcome_name: str
    probability: float
    policy: ControlledDecisionPolicy

    def as_dict(self) -> dict[str, object]:
        return {
            "outcome_index": self.outcome_index,
            "outcome_name": self.outcome_name,
            "probability": self.probability,
            "policy": self.policy.as_dict(),
        }


@dataclass(frozen=True)
class ControlledDecisionPolicy:
    """Exact finite-horizon act/intervene/fallback policy tree."""

    mode: ControlledDecisionMode
    action_index: int | None
    intervention_index: int | None
    intervention_name: str | None
    certificate: FiniteDecisionCertificate
    outcomes: tuple[ControlledOutcomeBranch, ...]
    expected_intervention_cost: float
    worst_case_intervention_cost: float
    worst_case_risk: float
    guaranteed_certification: bool
    horizon_remaining: int
    reason_code: str

    def as_dict(self) -> dict[str, object]:
        return {
            "version": CONTROLLED_DECISION_IDENTIFICATION_VERSION,
            "mode": self.mode,
            "action_index": self.action_index,
            "intervention_index": self.intervention_index,
            "intervention_name": self.intervention_name,
            "certificate": self.certificate.as_dict(),
            "outcomes": [branch.as_dict() for branch in self.outcomes],
            "expected_intervention_cost": self.expected_intervention_cost,
            "worst_case_intervention_cost": self.worst_case_intervention_cost,
            "worst_case_risk": self.worst_case_risk,
            "guaranteed_certification": self.guaranteed_certification,
            "horizon_remaining": self.horizon_remaining,
            "reason_code": self.reason_code,
            "claim_boundary": CONTROLLED_DECISION_IDENTIFICATION_CLAIM_BOUNDARY,
        }


def solve_controlled_decision(
    losses: object,
    weights: object,
    interventions: Sequence[FiniteControlledIntervention],
    *,
    regret_tolerance: float = 0.0,
    max_interventions: int = 1,
    risk_budget: float = 1.0,
    objective: ControlledObjective = "expected_cost",
    maximum_nodes: int = 100_000,
) -> ControlledDecisionPolicy:
    """Find a minimum-cost branch-complete policy under controlled dynamics.

    Every registered intervention may be used at most once.  A candidate is
    feasible only when each positive-probability observation branch reaches a
    support-wise certified terminal action within the remaining horizon and
    additive risk budget.  Failure returns exact fallback rather than a selected
    latent state.
    """

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_interventions(
        interventions,
        state_count=loss_matrix.shape[0],
    )
    horizon = _integer(max_interventions, name="max_interventions")
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
    ) -> ControlledDecisionPolicy:
        nonlocal nodes
        nodes += 1
        if nodes > node_limit:
            raise RuntimeError("controlled policy search exceeded maximum_nodes")
        certificate = finite_decision_certificate(
            loss_matrix,
            current_weights,
            regret_tolerance=regret_tolerance,
        )
        if certificate.certified:
            return ControlledDecisionPolicy(
                mode="act",
                action_index=certificate.selected_action_index,
                intervention_index=None,
                intervention_name=None,
                certificate=certificate,
                outcomes=(),
                expected_intervention_cost=0.0,
                worst_case_intervention_cost=0.0,
                worst_case_risk=0.0,
                guaranteed_certification=True,
                horizon_remaining=remaining_horizon,
                reason_code="current-decision-certified",
            )
        if remaining_horizon == 0 or not remaining:
            return ControlledDecisionPolicy(
                mode="fallback",
                action_index=None,
                intervention_index=None,
                intervention_name=None,
                certificate=certificate,
                outcomes=(),
                expected_intervention_cost=0.0,
                worst_case_intervention_cost=0.0,
                worst_case_risk=0.0,
                guaranteed_certification=False,
                horizon_remaining=remaining_horizon,
                reason_code="horizon-exhausted-without-certificate",
            )

        candidates: list[ControlledDecisionPolicy] = []
        for intervention_index in remaining:
            intervention = roster[intervention_index]
            if intervention.risk > remaining_risk + _ATOL:
                continue
            outcome_probabilities = np.asarray(
                controlled_outcome_probabilities(current_weights, intervention),
                dtype=np.float64,
            )
            next_remaining = tuple(
                index for index in remaining if index != intervention_index
            )
            branches: list[ControlledOutcomeBranch] = []
            expected_child_cost = 0.0
            worst_child_cost = 0.0
            worst_child_risk = 0.0
            feasible = True
            for outcome, outcome_probability in enumerate(outcome_probabilities):
                if outcome_probability <= 0.0:
                    continue
                posterior = np.asarray(
                    controlled_posterior_weights(
                        current_weights,
                        intervention,
                        outcome,
                    ),
                    dtype=np.float64,
                )
                child = recurse(
                    posterior,
                    next_remaining,
                    remaining_horizon - 1,
                    max(remaining_risk - intervention.risk, 0.0),
                )
                if not child.guaranteed_certification:
                    feasible = False
                    break
                branches.append(
                    ControlledOutcomeBranch(
                        outcome_index=outcome,
                        outcome_name=intervention.outcome_names[outcome],
                        probability=float(outcome_probability),
                        policy=child,
                    )
                )
                expected_child_cost += (
                    float(outcome_probability) * child.expected_intervention_cost
                )
                worst_child_cost = max(
                    worst_child_cost,
                    child.worst_case_intervention_cost,
                )
                worst_child_risk = max(worst_child_risk, child.worst_case_risk)
            if not feasible or not branches:
                continue
            candidates.append(
                ControlledDecisionPolicy(
                    mode="intervene",
                    action_index=None,
                    intervention_index=intervention_index,
                    intervention_name=intervention.name,
                    certificate=certificate,
                    outcomes=tuple(branches),
                    expected_intervention_cost=(
                        intervention.cost + expected_child_cost
                    ),
                    worst_case_intervention_cost=(intervention.cost + worst_child_cost),
                    worst_case_risk=intervention.risk + worst_child_risk,
                    guaranteed_certification=True,
                    horizon_remaining=remaining_horizon,
                    reason_code="minimum-cost-controlled-decision-certificate",
                )
            )
        if not candidates:
            return ControlledDecisionPolicy(
                mode="fallback",
                action_index=None,
                intervention_index=None,
                intervention_name=None,
                certificate=certificate,
                outcomes=(),
                expected_intervention_cost=0.0,
                worst_case_intervention_cost=0.0,
                worst_case_risk=0.0,
                guaranteed_certification=False,
                horizon_remaining=remaining_horizon,
                reason_code="no-safe-controlled-certificate",
            )
        if objective == "expected_cost":
            candidates.sort(
                key=lambda item: (
                    item.expected_intervention_cost,
                    item.worst_case_intervention_cost,
                    item.worst_case_risk,
                    item.intervention_name or "",
                )
            )
        else:
            candidates.sort(
                key=lambda item: (
                    item.worst_case_intervention_cost,
                    item.expected_intervention_cost,
                    item.worst_case_risk,
                    item.intervention_name or "",
                )
            )
        return candidates[0]

    return recurse(probability, tuple(range(len(roster))), horizon, budget)


@dataclass(frozen=True)
class NonAdaptiveControlledSequence:
    intervention_indices: tuple[int, ...]
    intervention_names: tuple[str, ...]
    total_cost: float
    total_risk: float
    guaranteed_certification: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "intervention_indices": list(self.intervention_indices),
            "intervention_names": list(self.intervention_names),
            "total_cost": self.total_cost,
            "total_risk": self.total_risk,
            "guaranteed_certification": self.guaranteed_certification,
        }


def minimum_nonadaptive_control_sequence(
    losses: object,
    weights: object,
    interventions: Sequence[FiniteControlledIntervention],
    *,
    regret_tolerance: float = 0.0,
    max_interventions: int | None = None,
    risk_budget: float = 1.0,
    maximum_outcome_histories: int = 100_000,
) -> NonAdaptiveControlledSequence | None:
    """Return the cheapest fixed intervention sequence certifying every history."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_interventions(
        interventions,
        state_count=loss_matrix.shape[0],
    )
    limit = (
        len(roster)
        if max_interventions is None
        else _integer(
            max_interventions,
            name="max_interventions",
        )
    )
    limit = min(limit, len(roster))
    budget = _finite_nonnegative(risk_budget, name="risk_budget")
    history_limit = _integer(
        maximum_outcome_histories,
        name="maximum_outcome_histories",
        minimum=1,
    )
    evaluated_histories = 0
    candidates: list[NonAdaptiveControlledSequence] = []

    for length in range(limit + 1):
        for indices in permutations(range(len(roster)), length):
            total_risk = float(sum(roster[index].risk for index in indices))
            if total_risk > budget + _ATOL:
                continue
            frontier = [probability]
            for intervention_index in indices:
                intervention = roster[intervention_index]
                next_frontier: list[np.ndarray] = []
                for current in frontier:
                    outcomes = controlled_outcome_probabilities(current, intervention)
                    for outcome, mass in enumerate(outcomes):
                        if mass <= 0.0:
                            continue
                        evaluated_histories += 1
                        if evaluated_histories > history_limit:
                            raise RuntimeError(
                                "nonadaptive controlled search exceeded "
                                "maximum_outcome_histories"
                            )
                        next_frontier.append(
                            np.asarray(
                                controlled_posterior_weights(
                                    current,
                                    intervention,
                                    outcome,
                                ),
                                dtype=np.float64,
                            )
                        )
                frontier = next_frontier
                if not frontier:
                    break
            if not frontier:
                continue
            if not all(
                finite_decision_certificate(
                    loss_matrix,
                    current,
                    regret_tolerance=regret_tolerance,
                ).certified
                for current in frontier
            ):
                continue
            candidates.append(
                NonAdaptiveControlledSequence(
                    intervention_indices=tuple(indices),
                    intervention_names=tuple(roster[index].name for index in indices),
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
            len(item.intervention_indices),
            item.intervention_names,
        )
    )
    return candidates[0]


def _partition(
    signatures: Sequence[tuple[float | int, ...]],
) -> tuple[tuple[int, ...], tuple[tuple[int, ...], ...]]:
    class_by_signature: dict[tuple[float | int, ...], int] = {}
    class_index: list[int] = []
    members: list[list[int]] = []
    for state, signature in enumerate(signatures):
        class_id = class_by_signature.get(signature)
        if class_id is None:
            class_id = len(members)
            class_by_signature[signature] = class_id
            members.append([])
        members[class_id].append(state)
        class_index.append(class_id)
    return tuple(class_index), tuple(tuple(row) for row in members)


@dataclass(frozen=True)
class ControlledLumpabilityWitness:
    """Two terminal-decision-equivalent states separated by controlled dynamics."""

    decision_class_index: int
    first_state_index: int
    second_state_index: int
    intervention_index: int
    intervention_name: str
    outcome_index: int
    outcome_name: str
    next_controlled_class_index: int
    first_probability: float
    second_probability: float

    def as_dict(self) -> dict[str, object]:
        return {
            "decision_class_index": self.decision_class_index,
            "first_state_index": self.first_state_index,
            "second_state_index": self.second_state_index,
            "intervention_index": self.intervention_index,
            "intervention_name": self.intervention_name,
            "outcome_index": self.outcome_index,
            "outcome_name": self.outcome_name,
            "next_controlled_class_index": self.next_controlled_class_index,
            "first_probability": self.first_probability,
            "second_probability": self.second_probability,
        }


@dataclass(frozen=True)
class ControlledDecisionQuotient:
    """Coarsest finite quotient preserving controlled terminal decisions."""

    normalized_losses: tuple[tuple[float, ...], ...]
    class_weights: tuple[float, ...]
    decision_class_index: tuple[int, ...]
    decision_class_members: tuple[tuple[int, ...], ...]
    class_index: tuple[int, ...]
    class_members: tuple[tuple[int, ...], ...]
    interventions: tuple[FiniteControlledIntervention, ...]
    refinement_iterations: int
    passive_decision_quotient_sufficient: bool
    witnesses: tuple[ControlledLumpabilityWitness, ...]

    @property
    def original_state_count(self) -> int:
        return len(self.class_index)

    @property
    def decision_class_count(self) -> int:
        return len(self.decision_class_members)

    @property
    def class_count(self) -> int:
        return len(self.class_members)

    def as_dict(self) -> dict[str, object]:
        return {
            "normalized_losses": [list(row) for row in self.normalized_losses],
            "class_weights": list(self.class_weights),
            "decision_class_index": list(self.decision_class_index),
            "decision_class_members": [
                list(row) for row in self.decision_class_members
            ],
            "class_index": list(self.class_index),
            "class_members": [list(row) for row in self.class_members],
            "interventions": [item.as_dict() for item in self.interventions],
            "refinement_iterations": self.refinement_iterations,
            "passive_decision_quotient_sufficient": (
                self.passive_decision_quotient_sufficient
            ),
            "witnesses": [item.as_dict() for item in self.witnesses],
        }


def build_controlled_decision_quotient(
    losses: object,
    weights: object,
    interventions: Sequence[FiniteControlledIntervention],
) -> ControlledDecisionQuotient:
    """Build the coarsest action-loss and controlled-kernel stable partition."""

    loss_matrix = _loss_matrix(losses)
    probability = _weights(weights, expected_size=loss_matrix.shape[0])
    roster = _validate_interventions(
        interventions,
        state_count=loss_matrix.shape[0],
    )
    normalized = loss_matrix - loss_matrix[:, [0]]
    decision_signatures = tuple(
        tuple(_canonical(value) for value in normalized[state])
        for state in range(loss_matrix.shape[0])
    )
    decision_index, decision_members = _partition(decision_signatures)
    class_index = decision_index
    class_members = decision_members
    iterations = 0
    while True:
        signatures: list[tuple[float | int, ...]] = []
        kernel_arrays = [np.asarray(item.kernel, dtype=np.float64) for item in roster]
        for state in range(loss_matrix.shape[0]):
            signature: list[float | int] = [class_index[state]]
            for intervention, kernel in zip(roster, kernel_arrays, strict=True):
                for outcome in range(intervention.outcome_count):
                    signature.extend(
                        _canonical(float(np.sum(kernel[state, members, outcome])))
                        for members in class_members
                    )
            signatures.append(tuple(signature))
        next_index, next_members = _partition(signatures)
        if next_index == class_index:
            break
        class_index = next_index
        class_members = next_members
        iterations += 1
        if iterations > loss_matrix.shape[0]:
            raise RuntimeError("controlled quotient refinement did not converge")

    representatives = [members[0] for members in class_members]
    class_weights = np.bincount(
        np.asarray(class_index, dtype=np.int64),
        weights=probability,
        minlength=len(class_members),
    )
    quotient_interventions: list[FiniteControlledIntervention] = []
    for intervention in roster:
        kernel = np.asarray(intervention.kernel, dtype=np.float64)
        quotient_kernel = np.zeros(
            (len(class_members), len(class_members), intervention.outcome_count),
            dtype=np.float64,
        )
        for current_class, representative in enumerate(representatives):
            for next_class, members in enumerate(class_members):
                quotient_kernel[current_class, next_class] = np.sum(
                    kernel[representative, members, :],
                    axis=0,
                )
        quotient_interventions.append(
            FiniteControlledIntervention(
                name=intervention.name,
                kernel=quotient_kernel,
                cost=intervention.cost,
                risk=intervention.risk,
                outcome_names=intervention.outcome_names,
            )
        )

    witnesses: list[ControlledLumpabilityWitness] = []
    for decision_class, members in enumerate(decision_members):
        first = members[0]
        for second in members[1:]:
            witness_found = False
            for intervention_index, intervention in enumerate(roster):
                kernel = np.asarray(intervention.kernel, dtype=np.float64)
                for outcome in range(intervention.outcome_count):
                    for next_class, controlled_members in enumerate(class_members):
                        first_probability = float(
                            np.sum(kernel[first, controlled_members, outcome])
                        )
                        second_probability = float(
                            np.sum(kernel[second, controlled_members, outcome])
                        )
                        if np.isclose(
                            first_probability,
                            second_probability,
                            rtol=0.0,
                            atol=_ATOL,
                        ):
                            continue
                        witnesses.append(
                            ControlledLumpabilityWitness(
                                decision_class_index=decision_class,
                                first_state_index=first,
                                second_state_index=second,
                                intervention_index=intervention_index,
                                intervention_name=intervention.name,
                                outcome_index=outcome,
                                outcome_name=intervention.outcome_names[outcome],
                                next_controlled_class_index=next_class,
                                first_probability=first_probability,
                                second_probability=second_probability,
                            )
                        )
                        witness_found = True
                        break
                    if witness_found:
                        break
                if witness_found:
                    break

    return ControlledDecisionQuotient(
        normalized_losses=tuple(
            tuple(_canonical(value) for value in normalized[representative])
            for representative in representatives
        ),
        class_weights=tuple(float(value) for value in class_weights),
        decision_class_index=decision_index,
        decision_class_members=decision_members,
        class_index=class_index,
        class_members=class_members,
        interventions=tuple(quotient_interventions),
        refinement_iterations=iterations,
        passive_decision_quotient_sufficient=(class_index == decision_index),
        witnesses=tuple(witnesses),
    )


__all__ = [
    "CONTROLLED_DECISION_IDENTIFICATION_CLAIM_BOUNDARY",
    "CONTROLLED_DECISION_IDENTIFICATION_VERSION",
    "ControlledDecisionMode",
    "ControlledDecisionPolicy",
    "ControlledDecisionQuotient",
    "ControlledLumpabilityWitness",
    "ControlledObjective",
    "ControlledOutcomeBranch",
    "FiniteControlledIntervention",
    "NonAdaptiveControlledSequence",
    "build_controlled_decision_quotient",
    "controlled_intervention_from_probe",
    "factorized_controlled_intervention",
    "controlled_outcome_probabilities",
    "controlled_posterior_weights",
    "marginal_static_probe",
    "minimum_nonadaptive_control_sequence",
    "solve_controlled_decision",
]
