"""Task-conditioned finite-hypothesis experimental design.

The routines in this module evaluate candidate observations by the expected
reduction in a registered downstream Bayes risk.  They deliberately separate
task value from generic mutual information and from a scalar physical-risk
guard.  All likelihoods and query/loss tables are caller supplied; this module
does not infer a physical model or authorize an intervention.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.floating[Any]]
IntArray: TypeAlias = NDArray[np.integer[Any]]
Objective = Literal["query", "decision", "information"]


def _readonly_float(
    value: object,
    *,
    name: str,
    ndim: int,
) -> FloatArray:
    array = np.asarray(value, dtype=np.float64).copy()
    if array.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    array.setflags(write=False)
    return array


def _probability(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    numeric = float(value)
    if not np.isfinite(numeric) or not 0.0 <= numeric <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return numeric


def _nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a real numeric scalar")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return numeric


def normalized_prior(prior_weights: FloatArray) -> FloatArray:
    """Return immutable normalized finite-hypothesis masses."""
    weights = _readonly_float(prior_weights, name="prior_weights", ndim=1).copy()
    if not weights.size or np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError(
            "prior_weights must contain nonnegative mass with positive total"
        )
    weights /= np.max(weights)
    weights /= np.sum(weights)
    weights.setflags(write=False)
    return weights


def _validate_likelihood(
    likelihood: FloatArray,
    *,
    hypotheses: int | None = None,
) -> FloatArray:
    matrix = _readonly_float(
        likelihood,
        name="outcome_likelihood",
        ndim=2,
    )
    if matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError("outcome_likelihood must be nonempty")
    if hypotheses is not None and matrix.shape[0] != hypotheses:
        raise ValueError("outcome_likelihood must have one row per hypothesis")
    if np.any(matrix < 0.0) or np.any(matrix > 1.0):
        raise ValueError("outcome_likelihood entries must lie in [0, 1]")
    if not np.allclose(
        np.sum(matrix, axis=1),
        1.0,
        atol=1e-12,
        rtol=1e-12,
    ):
        raise ValueError("each outcome_likelihood row must sum to one")
    return matrix


def _metric(value: FloatArray | None, *, dimension: int) -> FloatArray:
    if value is None:
        result: FloatArray = np.eye(dimension, dtype=np.float64)
        result.setflags(write=False)
        return result
    matrix = _readonly_float(value, name="query_metric", ndim=2).copy()
    if matrix.shape != (dimension, dimension):
        raise ValueError("query_metric has the wrong query dimension")
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=1e-12):
        raise ValueError("query_metric must be symmetric")
    matrix = 0.5 * (matrix + matrix.T)
    scale = max(float(np.max(np.abs(matrix), initial=0.0)), 1.0)
    if float(np.min(np.linalg.eigvalsh(matrix))) < -1e-12 * scale:
        raise ValueError("query_metric must be positive semidefinite")
    if not np.any(np.linalg.eigvalsh(matrix) > 1e-12 * scale):
        raise ValueError("query_metric must have a positive direction")
    matrix.setflags(write=False)
    return matrix


def _validate_objective(objective: Objective) -> Objective:
    if objective not in ("query", "decision", "information"):
        raise ValueError(f"unknown objective {objective!r}")
    return objective


@dataclass(frozen=True)
class FiniteProbe:
    """A finite observation experiment with a scalar prospective risk."""

    name: str
    outcome_likelihood: FloatArray
    physical_risk: float = 0.0
    cost: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("name must be a string")
        name = self.name.strip()
        if not name:
            raise ValueError("name must be nonempty")
        likelihood = _validate_likelihood(self.outcome_likelihood)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "outcome_likelihood", likelihood)
        object.__setattr__(
            self,
            "physical_risk",
            _probability(self.physical_risk, name="physical_risk"),
        )
        object.__setattr__(
            self,
            "cost",
            _nonnegative(self.cost, name="cost"),
        )

    @property
    def num_hypotheses(self) -> int:
        return int(self.outcome_likelihood.shape[0])

    @property
    def num_outcomes(self) -> int:
        return int(self.outcome_likelihood.shape[1])


@dataclass(frozen=True)
class ProbeValueReport:
    """Prospective task, decision, information, cost, and safety values."""

    name: str
    safe: bool
    physical_risk: float
    cost: float
    prior_query_risk: float
    expected_posterior_query_risk: float
    query_value: float
    prior_decision_risk: float | None
    expected_posterior_decision_risk: float | None
    decision_value: float | None
    mutual_information_nats: float
    query_net_value: float
    decision_net_value: float | None
    information_net_value: float
    reason_codes: tuple[str, ...]

    def score(self, objective: Objective) -> float | None:
        objective = _validate_objective(objective)
        if objective == "query":
            return self.query_net_value
        if objective == "decision":
            return self.decision_net_value
        if objective == "information":
            return self.information_net_value
        raise ValueError(f"unknown objective {objective!r}")


@dataclass(frozen=True)
class ProbeSelectionDecision:
    """A deterministic selection or exact no-probe fallback decision."""

    objective: Objective
    selected_probe_name: str | None
    exact_no_probe_fallback: bool
    score: float
    reason_code: str

    def __post_init__(self) -> None:
        _validate_objective(self.objective)
        if self.exact_no_probe_fallback != (self.selected_probe_name is None):
            raise ValueError("fallback flag must agree with selected_probe_name")
        if not np.isfinite(self.score):
            raise ValueError("score must be finite")
        if not self.reason_code:
            raise ValueError("reason_code must be nonempty")


def posterior_weights(
    prior_weights: FloatArray,
    outcome_likelihood: FloatArray,
    outcome_index: int,
) -> FloatArray:
    """Return p(hypothesis | one finite outcome).

    An impossible outcome is rejected instead of producing an arbitrary
    posterior.  Outcome labels are indices into the likelihood columns.
    """
    prior = normalized_prior(prior_weights)
    likelihood = _validate_likelihood(
        outcome_likelihood,
        hypotheses=prior.size,
    )
    if isinstance(outcome_index, bool) or not isinstance(
        outcome_index, (int, np.integer)
    ):
        raise TypeError("outcome_index must be an integer")
    index = int(outcome_index)
    if not 0 <= index < likelihood.shape[1]:
        raise ValueError("outcome_index is outside the outcome support")
    joint = prior * likelihood[:, index]
    total = float(np.sum(joint))
    if total <= 0.0:
        raise ValueError("the requested outcome has zero predictive mass")
    result = joint / total
    result.setflags(write=False)
    return result


def mutual_information_nats(
    prior_weights: FloatArray,
    outcome_likelihood: FloatArray,
) -> float:
    """Return I(H;Y) for a finite hypothesis and observation model."""
    prior = normalized_prior(prior_weights)
    likelihood = _validate_likelihood(
        outcome_likelihood,
        hypotheses=prior.size,
    )
    joint = prior[:, None] * likelihood
    outcome_mass = np.sum(joint, axis=0)
    positive = joint > 0.0
    denominator = np.broadcast_to(outcome_mass, joint.shape)
    value = np.sum(
        joint[positive] * (np.log(likelihood[positive]) - np.log(denominator[positive]))
    )
    return float(max(value, 0.0))


def query_bayes_risk(
    prior_weights: FloatArray,
    query_values: FloatArray,
    *,
    query_metric: FloatArray | None = None,
) -> float:
    """Bayes risk under metric-weighted squared query error."""
    prior = normalized_prior(prior_weights)
    query = _readonly_float(
        query_values,
        name="query_values",
        ndim=2,
    )
    if query.shape[0] != prior.size or query.shape[1] < 1:
        raise ValueError("query_values must have one nonempty row per hypothesis")
    metric = _metric(query_metric, dimension=query.shape[1])
    mean = prior @ query
    centered = query - mean
    return float(
        np.sum(
            prior
            * np.einsum(
                "hi,ij,hj->h",
                centered,
                metric,
                centered,
            )
        )
    )


def expected_posterior_query_risk(
    prior_weights: FloatArray,
    outcome_likelihood: FloatArray,
    query_values: FloatArray,
    *,
    query_metric: FloatArray | None = None,
) -> float:
    """Expected posterior Bayes risk for one prospective experiment."""
    prior = normalized_prior(prior_weights)
    likelihood = _validate_likelihood(
        outcome_likelihood,
        hypotheses=prior.size,
    )
    query = _readonly_float(
        query_values,
        name="query_values",
        ndim=2,
    )
    if query.shape[0] != prior.size or query.shape[1] < 1:
        raise ValueError("query_values must have one nonempty row per hypothesis")
    metric = _metric(query_metric, dimension=query.shape[1])
    joint = prior[:, None] * likelihood
    outcome_mass = np.sum(joint, axis=0)
    expected = 0.0
    for outcome, mass in enumerate(outcome_mass):
        if mass <= 0.0:
            continue
        posterior = joint[:, outcome] / mass
        mean = posterior @ query
        centered = query - mean
        conditional = float(
            np.sum(
                posterior
                * np.einsum(
                    "hi,ij,hj->h",
                    centered,
                    metric,
                    centered,
                )
            )
        )
        expected += float(mass) * conditional
    return expected


def decision_bayes_risk(
    prior_weights: FloatArray,
    decision_loss: FloatArray,
) -> float:
    """Minimum expected loss over a finite registered decision set."""
    prior = normalized_prior(prior_weights)
    loss = _readonly_float(
        decision_loss,
        name="decision_loss",
        ndim=2,
    )
    if loss.shape[0] < 1 or loss.shape[1] != prior.size:
        raise ValueError("decision_loss must have shape (decisions, hypotheses)")
    if np.any(loss < 0.0):
        raise ValueError("decision_loss must be nonnegative")
    return float(np.min(loss @ prior))


def expected_posterior_decision_risk(
    prior_weights: FloatArray,
    outcome_likelihood: FloatArray,
    decision_loss: FloatArray,
) -> float:
    """Expected finite-decision Bayes risk after a candidate observation."""
    prior = normalized_prior(prior_weights)
    likelihood = _validate_likelihood(
        outcome_likelihood,
        hypotheses=prior.size,
    )
    loss = _readonly_float(
        decision_loss,
        name="decision_loss",
        ndim=2,
    )
    if loss.shape[0] < 1 or loss.shape[1] != prior.size:
        raise ValueError("decision_loss must have shape (decisions, hypotheses)")
    if np.any(loss < 0.0):
        raise ValueError("decision_loss must be nonnegative")
    joint = prior[:, None] * likelihood
    # For each outcome, p(y) * posterior risk equals the minimum
    # unnormalized joint expected loss. This avoids dividing rare outcomes.
    outcome_weighted_risks = np.min(loss @ joint, axis=0)
    return float(np.sum(outcome_weighted_risks))


def evaluate_probe(
    prior_weights: FloatArray,
    probe: FiniteProbe,
    query_values: FloatArray,
    *,
    query_metric: FloatArray | None = None,
    decision_loss: FloatArray | None = None,
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
) -> ProbeValueReport:
    """Evaluate one probe without observing its outcome."""
    prior = normalized_prior(prior_weights)
    if probe.num_hypotheses != prior.size:
        raise ValueError("probe and prior hypothesis counts differ")
    cap = _probability(risk_cap, name="risk_cap")
    multiplier = _nonnegative(
        cost_multiplier,
        name="cost_multiplier",
    )
    prior_query = query_bayes_risk(
        prior,
        query_values,
        query_metric=query_metric,
    )
    posterior_query = expected_posterior_query_risk(
        prior,
        probe.outcome_likelihood,
        query_values,
        query_metric=query_metric,
    )
    query_value = max(prior_query - posterior_query, 0.0)

    prior_decision: float | None = None
    posterior_decision: float | None = None
    decision_value: float | None = None
    decision_net: float | None = None
    if decision_loss is not None:
        prior_decision = decision_bayes_risk(prior, decision_loss)
        posterior_decision = expected_posterior_decision_risk(
            prior,
            probe.outcome_likelihood,
            decision_loss,
        )
        decision_value = max(
            prior_decision - posterior_decision,
            0.0,
        )
        decision_net = decision_value - multiplier * probe.cost

    information = mutual_information_nats(
        prior,
        probe.outcome_likelihood,
    )
    safe = probe.physical_risk <= cap
    reasons = () if safe else ("prospective-physical-risk-cap-exceeded",)
    return ProbeValueReport(
        name=probe.name,
        safe=safe,
        physical_risk=probe.physical_risk,
        cost=probe.cost,
        prior_query_risk=prior_query,
        expected_posterior_query_risk=posterior_query,
        query_value=query_value,
        prior_decision_risk=prior_decision,
        expected_posterior_decision_risk=posterior_decision,
        decision_value=decision_value,
        mutual_information_nats=information,
        query_net_value=query_value - multiplier * probe.cost,
        decision_net_value=decision_net,
        information_net_value=information - multiplier * probe.cost,
        reason_codes=reasons,
    )


def evaluate_probes(
    prior_weights: FloatArray,
    probes: tuple[FiniteProbe, ...],
    query_values: FloatArray,
    *,
    query_metric: FloatArray | None = None,
    decision_loss: FloatArray | None = None,
    risk_cap: float = 1.0,
    cost_multiplier: float = 0.0,
) -> tuple[ProbeValueReport, ...]:
    """Evaluate a nonempty, uniquely named candidate roster."""
    if not probes:
        raise ValueError("probes must be nonempty")
    names = [probe.name for probe in probes]
    if len(set(names)) != len(names):
        raise ValueError("probe names must be unique")
    return tuple(
        evaluate_probe(
            prior_weights,
            probe,
            query_values,
            query_metric=query_metric,
            decision_loss=decision_loss,
            risk_cap=risk_cap,
            cost_multiplier=cost_multiplier,
        )
        for probe in probes
    )


def select_probe(
    reports: tuple[ProbeValueReport, ...],
    *,
    objective: Objective,
    minimum_net_value: float = 0.0,
) -> ProbeSelectionDecision:
    """Select a safe positive-value probe or return exact no-probe fallback."""
    objective = _validate_objective(objective)
    threshold = _nonnegative(
        minimum_net_value,
        name="minimum_net_value",
    )
    if not reports:
        return ProbeSelectionDecision(
            objective=objective,
            selected_probe_name=None,
            exact_no_probe_fallback=True,
            score=0.0,
            reason_code="no-candidates",
        )
    safe = [report for report in reports if report.safe]
    if not safe:
        return ProbeSelectionDecision(
            objective=objective,
            selected_probe_name=None,
            exact_no_probe_fallback=True,
            score=0.0,
            reason_code="no-safe-candidate",
        )
    scored: list[tuple[float, ProbeValueReport]] = []
    for report in safe:
        score = report.score(objective)
        if score is not None:
            scored.append((float(score), report))
    if not scored:
        return ProbeSelectionDecision(
            objective=objective,
            selected_probe_name=None,
            exact_no_probe_fallback=True,
            score=0.0,
            reason_code="objective-unavailable",
        )
    scored.sort(
        key=lambda item: (
            -item[0],
            item[1].physical_risk,
            item[1].cost,
            item[1].name,
        )
    )
    score, winner = scored[0]
    if score <= threshold:
        return ProbeSelectionDecision(
            objective=objective,
            selected_probe_name=None,
            exact_no_probe_fallback=True,
            score=score,
            reason_code="no-positive-net-value",
        )
    return ProbeSelectionDecision(
        objective=objective,
        selected_probe_name=winner.name,
        exact_no_probe_fallback=False,
        score=score,
        reason_code="selected-safe-positive-value",
    )


def weight_preserving_query_permutation(
    prior_weights: FloatArray,
    query_values: FloatArray,
    permutation: IntArray,
) -> FloatArray:
    """Permute hypothesis-query alignment without changing weighted marginals.

    This is a diagnostic control, not a physical baseline.  It is accepted only
    when the permutation preserves prior masses exactly (within numerical
    tolerance), so the weighted marginal multiset of query values is unchanged.
    """
    prior = normalized_prior(prior_weights)
    query = _readonly_float(
        query_values,
        name="query_values",
        ndim=2,
    )
    if query.shape[0] != prior.size:
        raise ValueError("query_values and prior hypothesis counts differ")
    indices = np.asarray(permutation)
    if (
        indices.ndim != 1
        or indices.shape != (prior.size,)
        or indices.dtype.kind not in "iu"
    ):
        raise ValueError("permutation must be a one-dimensional integer vector")
    indices = indices.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(indices), np.arange(prior.size)):
        raise ValueError("permutation must contain each hypothesis index once")
    if not np.allclose(
        prior,
        prior[indices],
        atol=1e-14,
        rtol=1e-14,
    ):
        raise ValueError("permutation does not preserve prior masses")
    result = np.asarray(query[indices], dtype=np.float64).copy()
    result.setflags(write=False)
    return result


def weight_preserving_loss_permutation(
    prior_weights: FloatArray,
    decision_loss: FloatArray,
    permutation: IntArray,
) -> FloatArray:
    """Apply the same weight-preserving control to finite loss columns."""
    prior = normalized_prior(prior_weights)
    loss = _readonly_float(
        decision_loss,
        name="decision_loss",
        ndim=2,
    )
    if loss.shape[1] != prior.size:
        raise ValueError("decision_loss and prior hypothesis counts differ")
    indices = np.asarray(permutation)
    if (
        indices.ndim != 1
        or indices.shape != (prior.size,)
        or indices.dtype.kind not in "iu"
    ):
        raise ValueError("permutation must be a one-dimensional integer vector")
    indices = indices.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(indices), np.arange(prior.size)):
        raise ValueError("permutation must contain each hypothesis index once")
    if not np.allclose(
        prior,
        prior[indices],
        atol=1e-14,
        rtol=1e-14,
    ):
        raise ValueError("permutation does not preserve prior masses")
    result = np.asarray(loss[:, indices], dtype=np.float64).copy()
    result.setflags(write=False)
    return result


__all__ = [
    "FiniteProbe",
    "ProbeSelectionDecision",
    "ProbeValueReport",
    "decision_bayes_risk",
    "evaluate_probe",
    "evaluate_probes",
    "expected_posterior_decision_risk",
    "expected_posterior_query_risk",
    "mutual_information_nats",
    "normalized_prior",
    "posterior_weights",
    "query_bayes_risk",
    "select_probe",
    "weight_preserving_loss_permutation",
    "weight_preserving_query_permutation",
]
