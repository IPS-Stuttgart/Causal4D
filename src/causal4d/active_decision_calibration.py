"""Finite-sample calibration for active decision identification.

This module provides two complementary calibration objects:

* a complete-group split-conformal additive regret margin; and
* a familywise simultaneous Hoeffding box for categorical probe outcomes.

The regret score keeps every registered action, branch, horizon, and query for a
physical object or trajectory inside one complete group.  The probability box
can account for all registered probe-outcome probabilities before a probe is
selected by passing their total family size.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

import numpy as np

from .immutable_array import readonly_array

ACTIVE_DECISION_CALIBRATION_VERSION = 1
ACTIVE_DECISION_CALIBRATION_CLAIM_BOUNDARY = (
    "The split-conformal margin has finite-sample group-marginal coverage only "
    "when calibration groups and the future group are exchangeable and the "
    "structural procedure was fixed independently of calibration outcomes. The "
    "Hoeffding probability box requires identically distributed independent "
    "complete trials for each registered probe context. Neither construction "
    "validates its assumptions, gives conditional coverage, or establishes "
    "deployment safety."
)


def _alpha(value: object) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError("alpha must be a finite real number in (0, 1)")
    result = float(value)
    if not np.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError("alpha must be a finite real number in (0, 1)")
    return result


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonempty_name(value: object, *, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value.strip()


def _regret_array(value: object, *, name: str) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    result = np.asarray(raw, dtype=np.float64)
    if result.ndim < 1 or result.shape[0] < 1 or not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must have a nonempty finite group axis")
    if np.any(result < 0.0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _count_vector(value: object) -> np.ndarray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iu":
        raise ValueError("outcome_counts must contain integers")
    result = np.asarray(raw, dtype=np.int64)
    if result.ndim != 1 or result.size < 2 or np.any(result < 0):
        raise ValueError(
            "outcome_counts must be a nonnegative vector with at least two outcomes"
        )
    if int(np.sum(result)) < 1:
        raise ValueError("outcome_counts must contain at least one complete trial")
    return result


@dataclass(frozen=True)
class CompleteGroupRegretCalibrationV1:
    """A one-sided additive split-conformal regret margin."""

    alpha: float
    primary_unit: str
    group_count: int
    comparisons_per_group: int
    conformal_rank: int
    raw_order_statistic: float | None
    additive_margin: float
    nonconformity_scores: tuple[float, ...]
    finite_margin_available: bool
    negative_margin_was_clamped: bool

    def __post_init__(self) -> None:
        level = _alpha(self.alpha)
        unit = _nonempty_name(self.primary_unit, name="primary_unit")
        group_count = _positive_integer(self.group_count, name="group_count")
        comparisons = _positive_integer(
            self.comparisons_per_group,
            name="comparisons_per_group",
        )
        rank = _positive_integer(self.conformal_rank, name="conformal_rank")
        scores = tuple(float(item) for item in self.nonconformity_scores)
        if len(scores) != group_count or not np.all(np.isfinite(scores)):
            raise ValueError("nonconformity_scores must align with group_count")
        margin = float(self.additive_margin)
        raw = self.raw_order_statistic
        if self.finite_margin_available:
            if raw is None or not np.isfinite(raw) or rank > group_count:
                raise ValueError("finite calibration order-statistic fields disagree")
            if not np.isfinite(margin) or margin < 0.0:
                raise ValueError("finite additive_margin must be nonnegative")
            expected = max(float(raw), 0.0)
            if not np.isclose(margin, expected, atol=1e-12, rtol=0.0):
                raise ValueError("additive_margin is inconsistent with raw statistic")
            if self.negative_margin_was_clamped != (float(raw) < 0.0):
                raise ValueError("negative-margin clamp flag is inconsistent")
        else:
            if raw is not None or margin != math.inf:
                raise ValueError(
                    "unavailable calibration margins require no statistic and infinity"
                )
            if self.negative_margin_was_clamped:
                raise ValueError("unavailable margins cannot be marked as clamped")
        object.__setattr__(self, "alpha", level)
        object.__setattr__(self, "primary_unit", unit)
        object.__setattr__(self, "group_count", group_count)
        object.__setattr__(self, "comparisons_per_group", comparisons)
        object.__setattr__(self, "conformal_rank", rank)
        object.__setattr__(self, "nonconformity_scores", scores)
        object.__setattr__(self, "additive_margin", margin)

    def require_finite_margin(self) -> float:
        """Return the margin or fail closed when its conformal rank is unavailable."""

        if not self.finite_margin_available:
            raise ValueError(
                "too few complete calibration groups for a finite conformal margin"
            )
        return self.additive_margin

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ACTIVE_DECISION_CALIBRATION_VERSION,
            "kind": "complete-group-split-conformal-regret-margin",
            "alpha": self.alpha,
            "primary_unit": self.primary_unit,
            "group_count": self.group_count,
            "comparisons_per_group": self.comparisons_per_group,
            "conformal_rank": self.conformal_rank,
            "raw_order_statistic": self.raw_order_statistic,
            "additive_margin": (
                self.additive_margin if self.finite_margin_available else None
            ),
            "nonconformity_scores": list(self.nonconformity_scores),
            "finite_margin_available": self.finite_margin_available,
            "negative_margin_was_clamped": self.negative_margin_was_clamped,
            "claim_boundary": ACTIVE_DECISION_CALIBRATION_CLAIM_BOUNDARY,
        }


@dataclass(frozen=True)
class CategoricalProbabilityBoxV1:
    """A simultaneous Hoeffding box for one categorical probe distribution."""

    alpha: float
    primary_unit: str
    group_count: int
    outcome_count: int
    registered_probability_count: int
    outcome_counts: tuple[int, ...]
    empirical_probabilities: tuple[float, ...]
    simultaneous_radius: float
    lower_bounds: tuple[float, ...]
    upper_bounds: tuple[float, ...]

    def __post_init__(self) -> None:
        level = _alpha(self.alpha)
        unit = _nonempty_name(self.primary_unit, name="primary_unit")
        groups = _positive_integer(self.group_count, name="group_count")
        outcomes = _positive_integer(self.outcome_count, name="outcome_count")
        family = _positive_integer(
            self.registered_probability_count,
            name="registered_probability_count",
        )
        if outcomes < 2 or family < outcomes:
            raise ValueError(
                "registered_probability_count must cover at least two outcomes"
            )
        count_array = _count_vector(self.outcome_counts)
        counts = tuple(int(item) for item in count_array)
        empirical = tuple(float(item) for item in self.empirical_probabilities)
        lower = tuple(float(item) for item in self.lower_bounds)
        upper = tuple(float(item) for item in self.upper_bounds)
        vectors = (counts, empirical, lower, upper)
        if any(len(vector) != outcomes for vector in vectors):
            raise ValueError("categorical calibration vectors must align")
        if sum(counts) != groups:
            raise ValueError("outcome_counts must sum to group_count")
        radius = float(self.simultaneous_radius)
        if not np.isfinite(radius) or not 0.0 <= radius <= 1.0:
            raise ValueError("simultaneous_radius must lie in [0, 1]")
        expected_empirical = np.asarray(counts, dtype=np.float64) / groups
        if not np.allclose(empirical, expected_empirical, atol=1e-12, rtol=0.0):
            raise ValueError("empirical probabilities are inconsistent with counts")
        if not all(
            0.0 <= lower[index] <= empirical[index] <= upper[index] <= 1.0
            for index in range(outcomes)
        ):
            raise ValueError("probability-box bounds are internally inconsistent")
        if sum(lower) > 1.0 + 1e-12 or sum(upper) < 1.0 - 1e-12:
            raise ValueError("probability-box bounds define an empty simplex")
        object.__setattr__(self, "alpha", level)
        object.__setattr__(self, "primary_unit", unit)
        object.__setattr__(self, "group_count", groups)
        object.__setattr__(self, "outcome_count", outcomes)
        object.__setattr__(self, "registered_probability_count", family)
        object.__setattr__(self, "outcome_counts", counts)
        object.__setattr__(self, "empirical_probabilities", empirical)
        object.__setattr__(self, "simultaneous_radius", radius)
        object.__setattr__(self, "lower_bounds", lower)
        object.__setattr__(self, "upper_bounds", upper)

    def as_dict(self) -> dict[str, object]:
        return {
            "version": ACTIVE_DECISION_CALIBRATION_VERSION,
            "kind": "familywise-two-sided-hoeffding-probability-box",
            "alpha": self.alpha,
            "primary_unit": self.primary_unit,
            "group_count": self.group_count,
            "outcome_count": self.outcome_count,
            "registered_probability_count": self.registered_probability_count,
            "outcome_counts": list(self.outcome_counts),
            "empirical_probabilities": list(self.empirical_probabilities),
            "simultaneous_radius": self.simultaneous_radius,
            "lower_bounds": list(self.lower_bounds),
            "upper_bounds": list(self.upper_bounds),
            "claim_boundary": ACTIVE_DECISION_CALIBRATION_CLAIM_BOUNDARY,
        }


def complete_group_nonconformity_scores(
    structural_regret_bounds: object,
    realized_regrets: object,
) -> tuple[float, ...]:
    """Return max(realized - structural) within each complete group."""

    structural = _regret_array(
        structural_regret_bounds,
        name="structural_regret_bounds",
    )
    realized = _regret_array(realized_regrets, name="realized_regrets")
    if structural.shape != realized.shape:
        raise ValueError("structural and realized regret arrays must have equal shape")
    difference = realized - structural
    flattened = difference.reshape(difference.shape[0], -1)
    return tuple(float(item) for item in np.max(flattened, axis=1))


def calibrate_complete_group_regret(
    structural_regret_bounds: object,
    realized_regrets: object,
    *,
    alpha: float,
    primary_unit: str = "complete-physical-group",
    minimum_groups: int = 1,
) -> CompleteGroupRegretCalibrationV1:
    """Fit a one-sided split-conformal complete-group regret margin.

    The one-indexed order-statistic rank is
    ``ceil((n + 1) * (1 - alpha))``.  When that rank exceeds the number of
    calibration groups, the returned margin is positive infinity and downstream
    code must fail closed.
    """

    level = _alpha(alpha)
    unit = _nonempty_name(primary_unit, name="primary_unit")
    minimum = _positive_integer(minimum_groups, name="minimum_groups")
    structural = _regret_array(
        structural_regret_bounds,
        name="structural_regret_bounds",
    )
    realized = _regret_array(realized_regrets, name="realized_regrets")
    if structural.shape != realized.shape:
        raise ValueError("structural and realized regret arrays must have equal shape")
    scores = complete_group_nonconformity_scores(structural, realized)
    group_count = len(scores)
    comparisons = int(np.prod(structural.shape[1:], dtype=np.int64))
    comparisons = max(comparisons, 1)
    rank = int(math.ceil((group_count + 1) * (1.0 - level)))
    available = group_count >= minimum and rank <= group_count
    raw: float | None
    if available:
        raw = float(np.sort(np.asarray(scores, dtype=np.float64))[rank - 1])
        margin = max(raw, 0.0)
        clamped = raw < 0.0
    else:
        raw = None
        margin = math.inf
        clamped = False
    return CompleteGroupRegretCalibrationV1(
        alpha=level,
        primary_unit=unit,
        group_count=group_count,
        comparisons_per_group=comparisons,
        conformal_rank=rank,
        raw_order_statistic=raw,
        additive_margin=margin,
        nonconformity_scores=scores,
        finite_margin_available=available,
        negative_margin_was_clamped=clamped,
    )


def simultaneous_hoeffding_probability_box(
    outcome_counts: object,
    *,
    alpha: float,
    primary_unit: str = "complete-probe-trial",
    registered_probability_count: int | None = None,
) -> CategoricalProbabilityBoxV1:
    """Construct a selection-safe categorical probability ambiguity box.

    For ``M`` registered probability coordinates, each coordinate uses the
    two-sided Hoeffding radius
    ``sqrt(log(2 * M / alpha) / (2 * n))``.  The union bound therefore covers all
    ``M`` pre-registered coordinates with probability at least ``1 - alpha``.
    Passing a family size larger than this probe's outcome count protects later
    selection among multiple registered probes.
    """

    counts = _count_vector(outcome_counts)
    level = _alpha(alpha)
    unit = _nonempty_name(primary_unit, name="primary_unit")
    outcome_count = int(counts.size)
    if registered_probability_count is None:
        family = outcome_count
    else:
        family = _positive_integer(
            registered_probability_count,
            name="registered_probability_count",
        )
    if family < outcome_count:
        raise ValueError(
            "registered_probability_count cannot be smaller than outcome count"
        )
    group_count = int(np.sum(counts))
    empirical = counts.astype(np.float64) / group_count
    radius = math.sqrt(math.log(2.0 * family / level) / (2.0 * group_count))
    radius = min(radius, 1.0)
    lower = np.maximum(empirical - radius, 0.0)
    upper = np.minimum(empirical + radius, 1.0)
    return CategoricalProbabilityBoxV1(
        alpha=level,
        primary_unit=unit,
        group_count=group_count,
        outcome_count=outcome_count,
        registered_probability_count=family,
        outcome_counts=tuple(int(item) for item in counts),
        empirical_probabilities=tuple(float(item) for item in empirical),
        simultaneous_radius=radius,
        lower_bounds=tuple(float(item) for item in lower),
        upper_bounds=tuple(float(item) for item in upper),
    )


def apply_complete_group_regret_margin(
    structural_regret_bounds: object,
    calibration: CompleteGroupRegretCalibrationV1,
) -> np.ndarray:
    """Inflate structural bounds by one finite complete-group margin."""

    if not isinstance(calibration, CompleteGroupRegretCalibrationV1):
        raise TypeError("calibration must be CompleteGroupRegretCalibrationV1")
    structural = _regret_array(
        structural_regret_bounds,
        name="structural_regret_bounds",
    )
    return readonly_array(
        structural + calibration.require_finite_margin(),
        dtype=np.float64,
    )


__all__ = [
    "ACTIVE_DECISION_CALIBRATION_CLAIM_BOUNDARY",
    "ACTIVE_DECISION_CALIBRATION_VERSION",
    "CategoricalProbabilityBoxV1",
    "CompleteGroupRegretCalibrationV1",
    "apply_complete_group_regret_margin",
    "calibrate_complete_group_regret",
    "complete_group_nonconformity_scores",
    "simultaneous_hoeffding_probability_box",
]
