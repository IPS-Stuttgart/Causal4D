"""Counterexample-guided certification over continuous physical support.

The finite decision-identification layers are exact only for a supplied finite
hypothesis roster. This module replaces that roster by one compact axis-aligned
parameter domain and a deterministic loss oracle with registered Lipschitz
constants. It uses sound branch-and-bound envelopes to certify or refute
uniform action regret without treating a parameter grid as complete support.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np

ContinuousDecisionStatus = Literal[
    "certified",
    "no-admissible-action",
    "multiple-admissible-actions",
    "inconclusive",
]

CONTINUOUS_DECISION_CERTIFICATION_VERSION = 1
CONTINUOUS_DECISION_CERTIFICATION_CLAIM_BOUNDARY = (
    "Sound only for the supplied compact axis-aligned parameter domain, "
    "deterministic finite loss oracle, valid global action-loss Lipschitz "
    "constants in the L-infinity parameter metric, regret tolerance, and "
    "numerical arithmetic. It does not validate the physical parameter set, "
    "loss oracle, Lipschitz constants, target transport, deployment, or safety."
)

_ATOL = 1e-12


def _finite_real(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real")
    return result


def _finite_nonnegative(value: object, *, name: str) -> float:
    result = _finite_real(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _canonical(value: float) -> float:
    result = float(value)
    return 0.0 if result == 0.0 else result


@dataclass(frozen=True)
class ParameterBox:
    """One compact axis-aligned region of continuous physical parameters."""

    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        if lower.ndim != 1 or upper.ndim != 1 or lower.size == 0:
            raise ValueError("box bounds must be nonempty vectors")
        if lower.shape != upper.shape:
            raise ValueError("box lower and upper bounds must have equal shape")
        if not np.isfinite(lower).all() or not np.isfinite(upper).all():
            raise ValueError("box bounds must be finite")
        if np.any(lower > upper):
            raise ValueError("box lower bounds must not exceed upper bounds")
        object.__setattr__(
            self, "lower", tuple(_canonical(value) for value in lower)
        )
        object.__setattr__(
            self, "upper", tuple(_canonical(value) for value in upper)
        )

    @property
    def dimension(self) -> int:
        return len(self.lower)

    @property
    def center(self) -> tuple[float, ...]:
        return tuple((low + high) / 2.0 for low, high in zip(self.lower, self.upper))

    @property
    def widths(self) -> tuple[float, ...]:
        return tuple(high - low for low, high in zip(self.lower, self.upper))

    @property
    def linf_radius(self) -> float:
        return max(self.widths) / 2.0

    @property
    def maximum_width(self) -> float:
        return max(self.widths)

    def split(self) -> tuple[ParameterBox, ParameterBox] | None:
        widths = self.widths
        dimension = int(np.argmax(np.asarray(widths, dtype=np.float64)))
        if widths[dimension] <= 0.0:
            return None
        midpoint = (self.lower[dimension] + self.upper[dimension]) / 2.0
        first_upper = list(self.upper)
        first_upper[dimension] = midpoint
        second_lower = list(self.lower)
        second_lower[dimension] = midpoint
        return (
            ParameterBox(self.lower, tuple(first_upper)),
            ParameterBox(tuple(second_lower), self.upper),
        )

    def contains(self, point: Sequence[float]) -> bool:
        values = np.asarray(point, dtype=np.float64)
        if values.shape != (self.dimension,):
            return False
        lower = np.asarray(self.lower, dtype=np.float64)
        upper = np.asarray(self.upper, dtype=np.float64)
        return bool(
            np.all(values >= lower - _ATOL) and np.all(values <= upper + _ATOL)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "lower": list(self.lower),
            "upper": list(self.upper),
            "center": list(self.center),
            "linf_radius": self.linf_radius,
        }


@dataclass(frozen=True)
class ContinuousActionBound:
    """Verified and witnessed bounds on one action's worst-case regret."""

    action_index: int
    witnessed_lower_bound: float
    verified_upper_bound: float
    certified_admissible: bool
    witnessed_inadmissible: bool
    witness_parameter: tuple[float, ...]
    witness_losses: tuple[float, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "action_index": self.action_index,
            "witnessed_lower_bound": self.witnessed_lower_bound,
            "verified_upper_bound": self.verified_upper_bound,
            "certified_admissible": self.certified_admissible,
            "witnessed_inadmissible": self.witnessed_inadmissible,
            "witness_parameter": list(self.witness_parameter),
            "witness_losses": list(self.witness_losses),
        }


@dataclass(frozen=True)
class ContinuousDecisionCertificate:
    """Fail-closed continuous-support decision certificate."""

    status: ContinuousDecisionStatus
    selected_action_index: int | None
    action_bounds: tuple[ContinuousActionBound, ...]
    regret_tolerance: float
    evaluated_points: int
    active_boxes: int
    maximum_remaining_radius: float
    reason_code: str

    @property
    def certified(self) -> bool:
        return self.status == "certified"

    @property
    def used_exact_fallback(self) -> bool:
        return not self.certified

    def as_dict(self) -> dict[str, object]:
        return {
            "version": CONTINUOUS_DECISION_CERTIFICATION_VERSION,
            "status": self.status,
            "selected_action_index": self.selected_action_index,
            "action_bounds": [bound.as_dict() for bound in self.action_bounds],
            "regret_tolerance": self.regret_tolerance,
            "evaluated_points": self.evaluated_points,
            "active_boxes": self.active_boxes,
            "maximum_remaining_radius": self.maximum_remaining_radius,
            "reason_code": self.reason_code,
            "used_exact_fallback": self.used_exact_fallback,
            "claim_boundary": CONTINUOUS_DECISION_CERTIFICATION_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class _EvaluatedBox:
    box: ParameterBox
    depth: int
    losses: tuple[float, ...]
    regret: tuple[float, ...]
    regret_upper: tuple[float, ...]


def _losses_at(
    loss_oracle: Callable[[np.ndarray], Sequence[float]],
    point: tuple[float, ...],
    *,
    expected_actions: int | None,
) -> np.ndarray:
    losses = np.asarray(
        loss_oracle(np.asarray(point, dtype=np.float64)),
        dtype=np.float64,
    )
    if losses.ndim != 1 or losses.size < 2:
        raise ValueError("loss oracle must return at least two action losses")
    if expected_actions is not None and losses.size != expected_actions:
        raise ValueError("loss oracle changed the action count")
    if not np.isfinite(losses).all():
        raise ValueError("loss oracle returned a nonfinite value")
    return losses


def _evaluate_box(
    box: ParameterBox,
    depth: int,
    loss_oracle: Callable[[np.ndarray], Sequence[float]],
    lipschitz: np.ndarray,
) -> _EvaluatedBox:
    losses = _losses_at(loss_oracle, box.center, expected_actions=lipschitz.size)
    regret = losses - float(np.min(losses))
    radius = box.linf_radius
    upper = np.empty(losses.size, dtype=np.float64)
    for action in range(losses.size):
        gaps = []
        for competitor in range(losses.size):
            if competitor == action:
                gaps.append(0.0)
            else:
                gaps.append(
                    float(losses[action] - losses[competitor])
                    + float(lipschitz[action] + lipschitz[competitor]) * radius
                )
        upper[action] = max(0.0, max(gaps))
    return _EvaluatedBox(
        box=box,
        depth=depth,
        losses=tuple(float(value) for value in losses),
        regret=tuple(max(0.0, float(value)) for value in regret),
        regret_upper=tuple(max(0.0, float(value)) for value in upper),
    )


def _summarize(
    active: Sequence[_EvaluatedBox],
    samples: Sequence[_EvaluatedBox],
    *,
    regret_tolerance: float,
) -> tuple[ContinuousActionBound, ...]:
    action_count = len(active[0].losses)
    upper = np.max(
        np.asarray([box.regret_upper for box in active], dtype=np.float64),
        axis=0,
    )
    sample_regret = np.asarray([box.regret for box in samples], dtype=np.float64)
    lower = np.max(sample_regret, axis=0)
    bounds: list[ContinuousActionBound] = []
    for action in range(action_count):
        witness_index = int(np.argmax(sample_regret[:, action]))
        witness = samples[witness_index]
        certified = bool(upper[action] <= regret_tolerance + _ATOL)
        rejected = bool(lower[action] > regret_tolerance + _ATOL)
        if certified and rejected:
            raise RuntimeError("inconsistent continuous regret bounds")
        bounds.append(
            ContinuousActionBound(
                action_index=action,
                witnessed_lower_bound=float(lower[action]),
                verified_upper_bound=float(upper[action]),
                certified_admissible=certified,
                witnessed_inadmissible=rejected,
                witness_parameter=witness.box.center,
                witness_losses=witness.losses,
            )
        )
    return tuple(bounds)


def _terminal_status(
    bounds: Sequence[ContinuousActionBound],
) -> tuple[ContinuousDecisionStatus | None, int | None, str | None]:
    certified = [bound.action_index for bound in bounds if bound.certified_admissible]
    rejected = [bound.action_index for bound in bounds if bound.witnessed_inadmissible]
    if len(certified) >= 2:
        return (
            "multiple-admissible-actions",
            None,
            "multiple-actions-uniformly-within-regret-tolerance",
        )
    if len(rejected) == len(bounds):
        return "no-admissible-action", None, "every-action-has-counterexample"
    if len(certified) == 1 and len(rejected) == len(bounds) - 1:
        return "certified", certified[0], "unique-continuous-support-admissible-action"
    return None, None, None


def certify_continuous_decision(
    loss_oracle: Callable[[np.ndarray], Sequence[float]],
    domain: ParameterBox,
    action_loss_lipschitz: Sequence[float],
    *,
    regret_tolerance: float = 0.0,
    maximum_evaluations: int = 4097,
    maximum_depth: int = 40,
    minimum_box_width: float = 0.0,
) -> ContinuousDecisionCertificate:
    """Certify a unique action or fail closed over a continuous parameter box.

    The registered loss of action ``a`` must be globally ``K[a]``-Lipschitz in
    the L-infinity parameter metric on ``domain``. For every box center ``c``
    and competitor ``b``,

    ``loss_a(theta) - loss_b(theta)``

    is then at most its value at ``c`` plus ``(K[a] + K[b]) * radius``. The
    branch-and-bound loop refines the box that carries the largest unresolved
    regret excess. A sampled violating point is a concrete counterexample; a
    verified upper bound below tolerance is a certificate. Exhausting the
    registered search budget returns ``inconclusive`` and exact fallback.
    """

    if not isinstance(domain, ParameterBox):
        raise TypeError("domain must be a ParameterBox")
    lipschitz = np.asarray(action_loss_lipschitz, dtype=np.float64)
    if lipschitz.ndim != 1 or lipschitz.size < 2:
        raise ValueError("action_loss_lipschitz must contain at least two values")
    if not np.isfinite(lipschitz).all() or np.any(lipschitz < 0.0):
        raise ValueError("action_loss_lipschitz must be finite and nonnegative")
    tolerance = _finite_nonnegative(regret_tolerance, name="regret_tolerance")
    evaluation_limit = _positive_integer(
        maximum_evaluations, name="maximum_evaluations"
    )
    depth_limit = _positive_integer(maximum_depth, name="maximum_depth")
    minimum_width = _finite_nonnegative(
        minimum_box_width, name="minimum_box_width"
    )

    initial = _evaluate_box(domain, 0, loss_oracle, lipschitz)
    active: list[_EvaluatedBox] = [initial]
    samples: list[_EvaluatedBox] = [initial]

    while True:
        bounds = _summarize(active, samples, regret_tolerance=tolerance)
        status, selected, reason = _terminal_status(bounds)
        if status is not None and reason is not None:
            return ContinuousDecisionCertificate(
                status=status,
                selected_action_index=selected,
                action_bounds=bounds,
                regret_tolerance=tolerance,
                evaluated_points=len(samples),
                active_boxes=len(active),
                maximum_remaining_radius=max(item.box.linf_radius for item in active),
                reason_code=reason,
            )

        unresolved = {
            bound.action_index
            for bound in bounds
            if not bound.certified_admissible and not bound.witnessed_inadmissible
        }
        if not unresolved:
            return ContinuousDecisionCertificate(
                status="inconclusive",
                selected_action_index=None,
                action_bounds=bounds,
                regret_tolerance=tolerance,
                evaluated_points=len(samples),
                active_boxes=len(active),
                maximum_remaining_radius=max(item.box.linf_radius for item in active),
                reason_code="continuous-bounds-do-not-identify-unique-action",
            )

        splittable = [
            (index, item)
            for index, item in enumerate(active)
            if item.depth < depth_limit
            and item.box.maximum_width > minimum_width + _ATOL
            and item.box.split() is not None
        ]
        if not splittable or len(samples) + 2 > evaluation_limit:
            return ContinuousDecisionCertificate(
                status="inconclusive",
                selected_action_index=None,
                action_bounds=bounds,
                regret_tolerance=tolerance,
                evaluated_points=len(samples),
                active_boxes=len(active),
                maximum_remaining_radius=max(item.box.linf_radius for item in active),
                reason_code="continuous-search-budget-exhausted",
            )

        def priority(entry: tuple[int, _EvaluatedBox]) -> tuple[object, ...]:
            index, item = entry
            unresolved_upper = max(item.regret_upper[action] for action in unresolved)
            unresolved_gap = max(
                item.regret_upper[action] - item.regret[action]
                for action in unresolved
            )
            return (
                unresolved_upper - tolerance,
                unresolved_gap,
                item.box.linf_radius,
                -item.depth,
                tuple(-value for value in item.box.lower),
                tuple(-value for value in item.box.upper),
                -index,
            )

        split_index, parent = max(splittable, key=priority)
        children = parent.box.split()
        if children is None:
            raise RuntimeError("selected box is not splittable")
        active.pop(split_index)
        evaluated_children = [
            _evaluate_box(child, parent.depth + 1, loss_oracle, lipschitz)
            for child in children
        ]
        active.extend(evaluated_children)
        samples.extend(evaluated_children)


__all__ = [
    "CONTINUOUS_DECISION_CERTIFICATION_CLAIM_BOUNDARY",
    "CONTINUOUS_DECISION_CERTIFICATION_VERSION",
    "ContinuousActionBound",
    "ContinuousDecisionCertificate",
    "ContinuousDecisionStatus",
    "ParameterBox",
    "certify_continuous_decision",
]
