"""Finite-support session-level intervention hierarchy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causal4d.immutable_array import readonly_array as _readonly_array
from causal4d.weighting import log_weights_from_probabilities


def _logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    maximum = np.max(array, axis=axis, keepdims=True)
    safe_maximum = np.where(np.isfinite(maximum), maximum, 0.0)
    summed = np.sum(np.exp(array - safe_maximum), axis=axis, keepdims=True)
    with np.errstate(divide="ignore"):
        result = safe_maximum + np.log(summed)
    if axis is not None:
        result = np.squeeze(result, axis=axis)
    return np.where(np.squeeze(np.isfinite(maximum), axis=axis), result, -np.inf)


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    normalizer = float(np.ravel(_logsumexp(np.asarray(log_weights, dtype=float)))[0])
    if not np.isfinite(normalizer):
        raise RuntimeError("session hierarchy posterior normalization failed")
    return np.exp(log_weights - normalizer)


def _probability_vector(values: np.ndarray, count: int, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=float)
    if probabilities.shape != (count,):
        raise ValueError(f"{name} must have shape ({count},)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    if float(np.sum(probabilities)) <= 0.0:
        raise ValueError(f"{name} must contain positive mass")
    return probabilities


def _transition_matrix(values: np.ndarray, phi_count: int) -> np.ndarray:
    transition = np.asarray(values, dtype=float)
    if transition.shape != (phi_count, phi_count):
        raise ValueError("session_phi_transition must have shape (Phi, Phi)")
    if not np.all(np.isfinite(transition)) or np.any(transition < 0.0):
        raise ValueError("session_phi_transition must be finite and nonnegative")
    row_sums = np.sum(transition, axis=1)
    if not np.allclose(row_sums, 1.0, rtol=0.0, atol=1.0e-12):
        raise ValueError("session_phi_transition rows must sum to one")
    return transition


@dataclass(frozen=True)
class SessionHierarchyPosterior:
    """Posterior over global and session-local finite ``phi`` support."""

    global_weights: np.ndarray
    session_ids: tuple[str, ...]
    execution_session_ids: tuple[str, ...]
    session_joint_weights: tuple[np.ndarray, ...]
    session_log_evidence: tuple[np.ndarray, ...]
    session_phi_transition: np.ndarray
    execution_evidence_powers: np.ndarray
    mode: str

    def __post_init__(self) -> None:
        global_weights = _readonly_array(self.global_weights)
        transition = _readonly_array(self.session_phi_transition)
        powers = _readonly_array(self.execution_evidence_powers)
        if global_weights.ndim != 2 or min(global_weights.shape) < 1:
            raise ValueError("global_weights must have shape (Phi, P)")
        phi_count, parameter_count = global_weights.shape
        if (
            not np.all(np.isfinite(global_weights))
            or np.any(global_weights < 0.0)
            or not np.isclose(np.sum(global_weights), 1.0)
        ):
            raise ValueError("global_weights must be finite and sum to one")
        if transition.shape != (phi_count, phi_count):
            raise ValueError("session_phi_transition must match global phi support")
        if (
            not np.all(np.isfinite(transition))
            or np.any(transition < 0.0)
            or not np.allclose(
                np.sum(transition, axis=1),
                1.0,
                rtol=0.0,
                atol=1.0e-12,
            )
        ):
            raise ValueError("session_phi_transition must be row-stochastic")
        session_ids = tuple(map(str, self.session_ids))
        if not session_ids or len(set(session_ids)) != len(session_ids):
            raise ValueError("session_ids must be unique and nonempty")
        execution_session_ids = tuple(map(str, self.execution_session_ids))
        if (
            len(execution_session_ids) != len(powers)
            or any(not value for value in execution_session_ids)
            or tuple(dict.fromkeys(execution_session_ids)) != session_ids
        ):
            raise ValueError(
                "execution_session_ids must bind every power to the session order"
            )
        if len(self.session_joint_weights) != len(session_ids) or len(
            self.session_log_evidence
        ) != len(session_ids):
            raise ValueError("session posterior arrays must match session_ids")
        session_weights = []
        session_evidence = []
        parameter_marginal = np.sum(global_weights, axis=0)
        for weights, evidence in zip(
            self.session_joint_weights,
            self.session_log_evidence,
            strict=True,
        ):
            joint = _readonly_array(weights)
            log_evidence = _readonly_array(evidence)
            if joint.shape != (phi_count, parameter_count):
                raise ValueError("session weights must have shape (Phi, P)")
            if (
                not np.all(np.isfinite(joint))
                or np.any(joint < 0.0)
                or not np.isclose(np.sum(joint), 1.0)
            ):
                raise ValueError("session weights must be finite and sum to one")
            if not np.allclose(
                np.sum(joint, axis=0),
                parameter_marginal,
                rtol=1.0e-10,
                atol=1.0e-12,
            ):
                raise ValueError("session weights must preserve parameter marginal")
            if log_evidence.shape != (phi_count, parameter_count):
                raise ValueError("session log evidence must have shape (Phi, P)")
            if np.any(np.isnan(log_evidence)) or np.any(np.isposinf(log_evidence)):
                raise ValueError("session log evidence must not contain NaN or +inf")
            session_weights.append(joint)
            session_evidence.append(log_evidence)
        if powers.ndim != 1:
            raise ValueError("execution_evidence_powers must be a vector")
        if not np.all(np.isfinite(powers)) or np.any(powers <= 0.0):
            raise ValueError("execution_evidence_powers must be finite and positive")
        if not self.mode:
            raise ValueError("mode must be nonempty")
        object.__setattr__(self, "global_weights", global_weights)
        object.__setattr__(self, "session_ids", session_ids)
        object.__setattr__(self, "execution_session_ids", execution_session_ids)
        object.__setattr__(self, "session_joint_weights", tuple(session_weights))
        object.__setattr__(self, "session_log_evidence", tuple(session_evidence))
        object.__setattr__(self, "session_phi_transition", transition)
        object.__setattr__(self, "execution_evidence_powers", powers)

    @property
    def global_phi_marginal(self) -> np.ndarray:
        return np.sum(self.global_weights, axis=1)

    @property
    def parameter_marginal(self) -> np.ndarray:
        return np.sum(self.global_weights, axis=0)

    @property
    def session_phi_marginals(self) -> tuple[np.ndarray, ...]:
        return tuple(np.sum(weights, axis=1) for weights in self.session_joint_weights)

    @property
    def predictive_session_joint_weights(self) -> np.ndarray:
        """Posterior predictive ``(phi_s, theta)`` weights for a new session."""

        predictive = np.einsum(
            "gp,gf->fp",
            self.global_weights,
            self.session_phi_transition,
        )
        return _readonly_array(predictive)


def infer_session_phi_hierarchy(
    execution_log_evidence: Sequence[np.ndarray],
    *,
    phi_prior: np.ndarray,
    parameter_prior: np.ndarray,
    session_ids: Sequence[str],
    execution_evidence_powers: Sequence[float],
    session_phi_transition: np.ndarray,
) -> SessionHierarchyPosterior:
    """Infer global ``phi_bar`` and session-local ``phi_s`` on finite support.

    ``session_phi_transition[g, f]`` is ``p(phi_s=f | phi_bar=g)``. Exact zeros
    preserve excluded support. An identity transition is the zero-session-
    variance limit and reproduces the shared-``phi`` posterior exactly.
    """

    evidence = tuple(
        np.asarray(values, dtype=float) for values in execution_log_evidence
    )
    if not evidence:
        raise ValueError("at least one execution log evidence matrix is required")
    if evidence[0].ndim != 2 or min(evidence[0].shape) < 1:
        raise ValueError("execution log evidence must have shape (Phi, P)")
    phi_count, parameter_count = evidence[0].shape
    for values in evidence:
        if values.shape != (phi_count, parameter_count):
            raise ValueError("execution log evidence matrices must share shape")
        if np.any(np.isnan(values)) or np.any(np.isposinf(values)):
            raise ValueError("execution log evidence must not contain NaN or +inf")

    identifiers = tuple(map(str, session_ids))
    if len(identifiers) != len(evidence) or any(not value for value in identifiers):
        raise ValueError("session_ids must contain one nonempty value per execution")
    powers = np.asarray(tuple(execution_evidence_powers), dtype=float)
    if powers.shape != (len(evidence),):
        raise ValueError("execution_evidence_powers must match executions")
    if not np.all(np.isfinite(powers)) or np.any(powers <= 0.0):
        raise ValueError("execution_evidence_powers must be finite and positive")

    normalized_phi_prior = _probability_vector(phi_prior, phi_count, "phi_prior")
    normalized_parameter_prior = _probability_vector(
        parameter_prior,
        parameter_count,
        "parameter_prior",
    )
    transition = _transition_matrix(session_phi_transition, phi_count)
    session_order = tuple(dict.fromkeys(identifiers))
    session_lookup = {
        identifier: index for index, identifier in enumerate(session_order)
    }
    session_evidence = [
        np.zeros((phi_count, parameter_count), dtype=float) for _ in session_order
    ]
    for identifier, power, values in zip(identifiers, powers, evidence, strict=True):
        session_evidence[session_lookup[identifier]] += float(power) * values

    global_log_weights = (
        log_weights_from_probabilities(
            normalized_phi_prior,
            name="global phi prior",
        )[:, None]
        + log_weights_from_probabilities(
            normalized_parameter_prior,
            name="parameter prior",
        )[None]
    )
    identity = np.eye(phi_count, dtype=float)
    if np.array_equal(transition, identity):
        for power, values in zip(powers, evidence, strict=True):
            global_log_weights += float(power) * values
        global_weights = _normalize_log_weights(global_log_weights)
        session_joint_weights = tuple(global_weights.copy() for _ in session_order)
        mode = "zero_variance_identity"
    else:
        log_transition = log_weights_from_probabilities(
            transition,
            name="session phi transition",
        )
        session_log_marginals = []
        for values in session_evidence:
            numerator = log_transition[:, :, None] + values[None]
            marginal = _logsumexp(numerator, axis=1)
            session_log_marginals.append(marginal)
            global_log_weights += marginal
        global_weights = _normalize_log_weights(global_log_weights)

        session_joint_weights_list = []
        for values, marginal in zip(
            session_evidence,
            session_log_marginals,
            strict=True,
        ):
            numerator = log_transition[:, :, None] + values[None]
            conditional_log = np.full_like(numerator, -np.inf)
            valid = np.isfinite(marginal)
            np.subtract(
                numerator,
                marginal[:, None],
                out=conditional_log,
                where=valid[:, None],
            )
            conditional = np.exp(conditional_log)
            joint = np.sum(global_weights[:, None] * conditional, axis=0)
            total = float(np.sum(joint))
            if not np.isfinite(total) or total <= 0.0:
                raise RuntimeError("session phi posterior normalization failed")
            session_joint_weights_list.append(joint / total)
        session_joint_weights = tuple(session_joint_weights_list)
        mode = "finite_session_transition"

    return SessionHierarchyPosterior(
        global_weights=global_weights,
        session_ids=session_order,
        execution_session_ids=identifiers,
        session_joint_weights=session_joint_weights,
        session_log_evidence=tuple(session_evidence),
        session_phi_transition=transition,
        execution_evidence_powers=powers,
        mode=mode,
    )
