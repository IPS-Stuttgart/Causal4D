"""Consume a BayesianPhysTwin query-decision certificate fail closed.

BayesianPhysTwin owns construction of the exact finite-action certificate.
Causal4D consumes that certificate to authorize one downstream intervention
without selecting an unsupported within-class physical explanation.  A unique
robustly optimal action is preferred; otherwise a unique action within the
registered regret tolerance is accepted.  If neither decision is unique,
Causal4D returns the caller-owned fallback exactly.

This module validates certificate semantics and internal numerical consistency,
but it does not re-establish the physical quotient, provider competence, loss
model, regret tolerance, held-out transport, deployment authorization, or
safety.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Real
from typing import Any, Final, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
CertificateLevel = Literal[
    "robustly-optimal",
    "tolerance-admissible",
    "uncertified",
]

QUERY_DECISION_CERTIFICATE_VERSION: Final = 1
QUERY_DECISION_CERTIFICATE_SEMANTICS: Final = (
    "exact-worst-case-regret-over-registered-query-quotient-and-prior-support-v1"
)
DECISION_IDENTIFIABLE_INTERVENTION_VERSION: Final = 1
DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY: Final = (
    "Causal4D consumes a supplied finite-action certificate and fails closed when "
    "a unique registered action is not certified. This does not validate the "
    "physical quotient, provider, hypothesis roster, loss matrix, regret "
    "tolerance, held-out transport, deployment authorization, or safety."
)

_NUMERICAL_ATOL: Final = 1e-12


def _required_field(value: object, name: str) -> object:
    if isinstance(value, Mapping):
        if name not in value:
            raise ValueError(f"certificate is missing {name}")
        return value[name]
    if not hasattr(value, name):
        raise ValueError(f"certificate is missing {name}")
    return getattr(value, name)


def _certificate_summary(value: object) -> Mapping[str, object]:
    summary: object
    if isinstance(value, Mapping) and "summary" in value:
        summary = value["summary"]
    else:
        method = getattr(value, "summary", None)
        summary = method() if callable(method) else value
    if not isinstance(summary, Mapping):
        raise ValueError("certificate summary must be a mapping")
    version = summary.get("version")
    if type(version) is not int or version != QUERY_DECISION_CERTIFICATE_VERSION:
        raise ValueError("unsupported query-decision certificate version")
    if summary.get("semantics") != QUERY_DECISION_CERTIFICATE_SEMANTICS:
        raise ValueError("unsupported query-decision certificate semantics")
    return summary


def _finite_nonnegative(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _float_vector(value: object, *, name: str, size: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (size,) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite vector of length {size}")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _float_matrix(value: object, *, name: str, size: int) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.shape != (size, size) or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be a finite {size} by {size} matrix")
    return result


def _bool_vector(value: object, *, name: str, size: int) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b" or raw.shape != (size,):
        raise ValueError(f"{name} must be a Boolean vector of length {size}")
    return np.asarray(raw, dtype=np.bool_)


def _action_names(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("action_names must be a sequence of strings")
    actions = tuple(values)
    if len(actions) < 2:
        raise ValueError("at least two action names are required")
    for index, value in enumerate(actions):
        if type(value) is not str or not value.strip():
            raise ValueError(f"action_names[{index}] must be a nonempty string")
    if len(set(actions)) != len(actions):
        raise ValueError("action_names must be unique")
    return actions


def _nonempty_string(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class DecisionIdentifiableInterventionV1:
    """One certified action or the exact caller-owned fallback."""

    action_name: str
    certified_action_name: str | None
    fallback_action_name: str
    used_exact_fallback: bool
    certificate_level: CertificateLevel
    selected_worst_case_regret: float | None
    minimax_action_name: str
    minimax_worst_case_regret: float
    regret_tolerance: float
    tolerance_admissible_action_names: tuple[str, ...]
    robustly_optimal_action_names: tuple[str, ...]
    reason_code: str
    source_certificate_version: int = QUERY_DECISION_CERTIFICATE_VERSION
    source_certificate_semantics: str = QUERY_DECISION_CERTIFICATE_SEMANTICS

    def __post_init__(self) -> None:
        action = _nonempty_string(self.action_name, name="action_name")
        fallback = _nonempty_string(
            self.fallback_action_name,
            name="fallback_action_name",
        )
        minimax = _nonempty_string(
            self.minimax_action_name,
            name="minimax_action_name",
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
                raise ValueError("uncertified decisions must return exact fallback")
            if self.selected_worst_case_regret is not None:
                raise ValueError("fallback decisions have no selected certified regret")
        else:
            certified = _nonempty_string(
                self.certified_action_name,
                name="certified_action_name",
            )
            if action != certified or self.certificate_level == "uncertified":
                raise ValueError("certified action fields are inconsistent")
            if self.selected_worst_case_regret is None:
                raise ValueError("certified actions require a worst-case regret")
            _finite_nonnegative(
                self.selected_worst_case_regret,
                name="selected_worst_case_regret",
            )
        _finite_nonnegative(
            self.minimax_worst_case_regret,
            name="minimax_worst_case_regret",
        )
        _finite_nonnegative(self.regret_tolerance, name="regret_tolerance")
        if not self.reason_code:
            raise ValueError("reason_code must be nonempty")
        if self.source_certificate_version != QUERY_DECISION_CERTIFICATE_VERSION:
            raise ValueError("unsupported source certificate version")
        if self.source_certificate_semantics != QUERY_DECISION_CERTIFICATE_SEMANTICS:
            raise ValueError("unsupported source certificate semantics")
        object.__setattr__(self, "action_name", action)
        object.__setattr__(self, "fallback_action_name", fallback)
        object.__setattr__(self, "minimax_action_name", minimax)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": DECISION_IDENTIFIABLE_INTERVENTION_VERSION,
            "action_name": self.action_name,
            "certified_action_name": self.certified_action_name,
            "fallback_action_name": self.fallback_action_name,
            "used_exact_fallback": self.used_exact_fallback,
            "certificate_level": self.certificate_level,
            "selected_worst_case_regret": self.selected_worst_case_regret,
            "minimax_action_name": self.minimax_action_name,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "regret_tolerance": self.regret_tolerance,
            "tolerance_admissible_action_names": list(
                self.tolerance_admissible_action_names
            ),
            "robustly_optimal_action_names": list(
                self.robustly_optimal_action_names
            ),
            "reason_code": self.reason_code,
            "source_certificate_version": self.source_certificate_version,
            "source_certificate_semantics": self.source_certificate_semantics,
            "claim_boundary": DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY,
        }


def consume_query_decision_certificate(
    certificate: object,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
) -> DecisionIdentifiableInterventionV1:
    """Authorize one uniquely certified action or return exact fallback.

    The consumer independently checks the action-wise regret, pairwise loss-gap
    matrix, robust/tolerance masks, deterministic minimax index, and summary
    fields.  It deliberately refuses to choose among multiple admissible actions:
    such a set may be useful to a higher-level planner, but it does not identify
    one intervention under this contract.
    """

    actions = _action_names(action_names)
    fallback = _nonempty_string(
        fallback_action_name,
        name="fallback_action_name",
    )
    if fallback in actions:
        raise ValueError("fallback_action_name must be outside the candidate roster")
    summary = _certificate_summary(certificate)
    size = len(actions)
    if summary.get("action_count") != size:
        raise ValueError("certificate action count does not match action_names")

    pairwise = _float_matrix(
        _required_field(certificate, "pairwise_worst_case_loss_gap"),
        name="pairwise_worst_case_loss_gap",
        size=size,
    )
    if not np.allclose(
        np.diag(pairwise),
        0.0,
        atol=_NUMERICAL_ATOL,
        rtol=0.0,
    ):
        raise ValueError("pairwise certificate diagonal must be zero")
    regret = _float_vector(
        _required_field(certificate, "worst_case_regret"),
        name="worst_case_regret",
        size=size,
    )
    expected_regret = np.maximum(np.max(pairwise, axis=1), 0.0)
    if not np.allclose(
        regret,
        expected_regret,
        atol=_NUMERICAL_ATOL,
        rtol=0.0,
    ):
        raise ValueError("worst_case_regret is inconsistent with pairwise gaps")

    tolerance = _finite_nonnegative(
        _required_field(certificate, "regret_tolerance"),
        name="regret_tolerance",
    )
    tolerance_mask = _bool_vector(
        _required_field(certificate, "tolerance_admissible_action_mask"),
        name="tolerance_admissible_action_mask",
        size=size,
    )
    expected_tolerance = regret <= tolerance + _NUMERICAL_ATOL
    if not np.array_equal(tolerance_mask, expected_tolerance):
        raise ValueError("tolerance-admissible mask is inconsistent with regret")

    robust_mask = _bool_vector(
        _required_field(certificate, "robustly_optimal_action_mask"),
        name="robustly_optimal_action_mask",
        size=size,
    )
    expected_robust = np.all(pairwise <= _NUMERICAL_ATOL, axis=1)
    if not np.array_equal(robust_mask, expected_robust):
        raise ValueError("robust-optimal mask is inconsistent with pairwise gaps")

    minimax_index = _integer(
        _required_field(certificate, "minimax_action_index"),
        name="minimax_action_index",
    )
    if not 0 <= minimax_index < size:
        raise ValueError("minimax_action_index is outside the action roster")
    minimum = float(np.min(regret))
    expected_indices = np.flatnonzero(
        np.isclose(regret, minimum, atol=_NUMERICAL_ATOL, rtol=0.0)
    )
    if minimax_index != int(expected_indices[0]):
        raise ValueError("minimax_action_index is not the deterministic minimizer")
    minimax_regret = _finite_nonnegative(
        _required_field(certificate, "minimax_worst_case_regret"),
        name="minimax_worst_case_regret",
    )
    if not np.isclose(
        minimax_regret,
        minimum,
        atol=_NUMERICAL_ATOL,
        rtol=0.0,
    ):
        raise ValueError("minimax_worst_case_regret is inconsistent")

    tolerance_indices = np.flatnonzero(tolerance_mask)
    robust_indices = np.flatnonzero(robust_mask)
    tolerance_names = tuple(actions[int(index)] for index in tolerance_indices)
    robust_names = tuple(actions[int(index)] for index in robust_indices)

    expected_summary = {
        "minimax_action_index": minimax_index,
        "minimax_worst_case_regret": minimax_regret,
        "regret_tolerance": tolerance,
        "has_tolerance_admissible_action": bool(tolerance_indices.size),
        "uniquely_tolerance_identified": bool(tolerance_indices.size == 1),
        "has_robustly_optimal_action": bool(robust_indices.size),
        "uniquely_robustly_optimal": bool(robust_indices.size == 1),
    }
    for name, expected in expected_summary.items():
        actual = summary.get(name)
        if isinstance(expected, float):
            if not isinstance(actual, Real) or not np.isclose(
                float(actual),
                expected,
                atol=_NUMERICAL_ATOL,
                rtol=0.0,
            ):
                raise ValueError(f"certificate summary field {name} is inconsistent")
        elif actual != expected:
            raise ValueError(f"certificate summary field {name} is inconsistent")

    if robust_indices.size == 1:
        selected = int(robust_indices[0])
        return DecisionIdentifiableInterventionV1(
            action_name=actions[selected],
            certified_action_name=actions[selected],
            fallback_action_name=fallback,
            used_exact_fallback=False,
            certificate_level="robustly-optimal",
            selected_worst_case_regret=float(regret[selected]),
            minimax_action_name=actions[minimax_index],
            minimax_worst_case_regret=minimax_regret,
            regret_tolerance=tolerance,
            tolerance_admissible_action_names=tolerance_names,
            robustly_optimal_action_names=robust_names,
            reason_code="unique-robustly-optimal-action",
        )
    if tolerance_indices.size == 1:
        selected = int(tolerance_indices[0])
        return DecisionIdentifiableInterventionV1(
            action_name=actions[selected],
            certified_action_name=actions[selected],
            fallback_action_name=fallback,
            used_exact_fallback=False,
            certificate_level="tolerance-admissible",
            selected_worst_case_regret=float(regret[selected]),
            minimax_action_name=actions[minimax_index],
            minimax_worst_case_regret=minimax_regret,
            regret_tolerance=tolerance,
            tolerance_admissible_action_names=tolerance_names,
            robustly_optimal_action_names=robust_names,
            reason_code="unique-tolerance-admissible-action",
        )

    reason = (
        "no-tolerance-admissible-action"
        if tolerance_indices.size == 0
        else "decision-not-uniquely-identified"
    )
    return DecisionIdentifiableInterventionV1(
        action_name=fallback,
        certified_action_name=None,
        fallback_action_name=fallback,
        used_exact_fallback=True,
        certificate_level="uncertified",
        selected_worst_case_regret=None,
        minimax_action_name=actions[minimax_index],
        minimax_worst_case_regret=minimax_regret,
        regret_tolerance=tolerance,
        tolerance_admissible_action_names=tolerance_names,
        robustly_optimal_action_names=robust_names,
        reason_code=reason,
    )


def decision_identifiable_intervention_from_quotient(
    prior_weights: object,
    quotient_weights: object,
    class_index: object,
    loss_by_hypothesis_action: object,
    action_names: Sequence[str],
    *,
    fallback_action_name: str,
    regret_tolerance: float = 0.0,
) -> DecisionIdentifiableInterventionV1:
    """Construct the BayesianPhysTwin certificate and consume it in Causal4D.

    ``causal4d[phystwin]`` (or another compatible BayesianPhysTwin installation)
    must expose ``query_decision_certificate_v1``.  The import is intentionally
    local so Causal4D's core package remains usable without the optional provider.
    """

    try:
        from bayesian_phystwin.query_decision_certificate_v1 import (
            query_decision_certificate,
        )
    except ImportError as error:
        raise RuntimeError(
            "decision-identifiable quotient integration requires a compatible "
            "BayesianPhysTwin installation; install causal4d[phystwin]"
        ) from error

    certificate = query_decision_certificate(
        prior_weights,
        quotient_weights,
        class_index,
        loss_by_hypothesis_action,
        regret_tolerance=regret_tolerance,
    )
    return consume_query_decision_certificate(
        certificate,
        action_names,
        fallback_action_name=fallback_action_name,
    )


__all__ = [
    "DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY",
    "DECISION_IDENTIFIABLE_INTERVENTION_VERSION",
    "QUERY_DECISION_CERTIFICATE_SEMANTICS",
    "QUERY_DECISION_CERTIFICATE_VERSION",
    "DecisionIdentifiableInterventionV1",
    "consume_query_decision_certificate",
    "decision_identifiable_intervention_from_quotient",
]
