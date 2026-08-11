"""Registered confidence intervals for session-clustered real effects.

The physical protocol treats one target grasp session as the independent unit.
This module keeps the interval algorithms separate from artifact publication so
both the registered report and target-free operating-characteristic diagnostics
exercise exactly the same implementation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t as student_t_distribution


REAL_EFFECT_CONFIDENCE_LEVEL = 0.95
REAL_EFFECT_BOOTSTRAP_REPLICATES = 20_000
REAL_EFFECT_BOOTSTRAP_SEED = 20_260_726

_PRIMARY_INTERVAL_METHOD = "target_session_bootstrap_t"
_ROBUSTNESS_INTERVAL_METHOD = "student_t_mean"

FloatArray: TypeAlias = NDArray[np.float64]


def _validated_values(values: Sequence[float]) -> FloatArray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("session effects must be one-dimensional")
    if not np.all(np.isfinite(array)):
        raise ValueError("session effects must be finite")
    return array


def _validated_interval_settings(
    *,
    confidence_level: float,
    replicates: int | None = None,
    seed: int | None = None,
) -> None:
    if not np.isfinite(confidence_level) or not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    if replicates is not None and (type(replicates) is not int or replicates < 1):
        raise ValueError("replicates must be a positive integer")
    if seed is not None and type(seed) is not int:
        raise ValueError("seed must be an integer")


def _not_estimable_interval(
    *,
    method: str,
    sample_count: int,
    confidence_level: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "estimable": False,
        "method": method,
        "confidence_level": confidence_level,
        "sample_count": sample_count,
        "point_estimate": None,
        "lower": None,
        "upper": None,
        "reason": reason,
        "finite_sample_coverage_guaranteed": False,
    }


def percentile_bootstrap_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = REAL_EFFECT_CONFIDENCE_LEVEL,
    replicates: int = REAL_EFFECT_BOOTSTRAP_REPLICATES,
    seed: int = REAL_EFFECT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return the historical deterministic percentile interval for the mean."""

    _validated_interval_settings(
        confidence_level=confidence_level,
        replicates=replicates,
        seed=seed,
    )
    array = _validated_values(values)
    if len(array) < 2:
        result = _not_estimable_interval(
            method="target_session_percentile_bootstrap",
            sample_count=len(array),
            confidence_level=confidence_level,
            reason="at least two included sessions are required",
        )
        result.update(replicates=replicates, seed=seed)
        return result

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        len(array),
        size=(replicates, len(array)),
    )
    means = np.mean(array[indices], axis=1)
    tail = 0.5 * (1.0 - confidence_level)
    lower, upper = np.quantile(means, [tail, 1.0 - tail], method="linear")
    return {
        "estimable": True,
        "method": "target_session_percentile_bootstrap",
        "confidence_level": confidence_level,
        "sample_count": len(array),
        "point_estimate": float(np.mean(array)),
        "lower": float(lower),
        "upper": float(upper),
        "replicates": replicates,
        "seed": seed,
        "degenerate_sample": float(np.std(array, ddof=1)) == 0.0,
        "finite_sample_coverage_guaranteed": False,
    }


def student_t_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = REAL_EFFECT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Return a transparent small-sample mean interval under t assumptions."""

    _validated_interval_settings(confidence_level=confidence_level)
    array = _validated_values(values)
    if len(array) < 2:
        return _not_estimable_interval(
            method=_ROBUSTNESS_INTERVAL_METHOD,
            sample_count=len(array),
            confidence_level=confidence_level,
            reason="at least two included sessions are required",
        )

    point = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    standard_error = sample_sd / math.sqrt(len(array))
    assumptions = [
        "independent session effects",
        "approximately normal sampling distribution of the session mean",
    ]
    if sample_sd == 0.0:
        result = _not_estimable_interval(
            method=_ROBUSTNESS_INTERVAL_METHOD,
            sample_count=len(array),
            confidence_level=confidence_level,
            reason="zero between-session variance prevents uncertainty estimation",
        )
        result.update(
            degrees_of_freedom=len(array) - 1,
            point_estimate=point,
            sample_standard_deviation=sample_sd,
            standard_error=standard_error,
            degenerate_sample=True,
            coverage_assumptions=assumptions,
        )
        return result

    tail = 0.5 * (1.0 - confidence_level)
    critical_value = float(student_t_distribution.ppf(1.0 - tail, df=len(array) - 1))
    half_width = critical_value * standard_error
    return {
        "estimable": True,
        "method": _ROBUSTNESS_INTERVAL_METHOD,
        "confidence_level": confidence_level,
        "sample_count": len(array),
        "degrees_of_freedom": len(array) - 1,
        "point_estimate": point,
        "sample_standard_deviation": sample_sd,
        "standard_error": standard_error,
        "critical_value": critical_value,
        "lower": point - half_width,
        "upper": point + half_width,
        "degenerate_sample": False,
        "coverage_assumptions": assumptions,
        "finite_sample_coverage_guaranteed": False,
    }


def bootstrap_t_mean_interval(
    values: Sequence[float],
    *,
    confidence_level: float = REAL_EFFECT_CONFIDENCE_LEVEL,
    replicates: int = REAL_EFFECT_BOOTSTRAP_REPLICATES,
    seed: int = REAL_EFFECT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return a deterministic studentized bootstrap interval for the mean."""

    _validated_interval_settings(
        confidence_level=confidence_level,
        replicates=replicates,
        seed=seed,
    )
    array = _validated_values(values)
    if len(array) < 2:
        result = _not_estimable_interval(
            method=_PRIMARY_INTERVAL_METHOD,
            sample_count=len(array),
            confidence_level=confidence_level,
            reason="at least two included sessions are required",
        )
        result.update(replicates=replicates, seed=seed)
        return result

    sample_count = len(array)
    point = float(np.mean(array))
    sample_sd = float(np.std(array, ddof=1))
    sample_standard_error = sample_sd / math.sqrt(sample_count)
    if sample_standard_error == 0.0:
        result = _not_estimable_interval(
            method=_PRIMARY_INTERVAL_METHOD,
            sample_count=sample_count,
            confidence_level=confidence_level,
            reason="zero between-session variance prevents studentization",
        )
        result.update(
            point_estimate=point,
            sample_standard_deviation=sample_sd,
            standard_error=sample_standard_error,
            replicates=replicates,
            seed=seed,
            finite_studentized_replicate_count=0,
            finite_studentized_replicate_fraction=0.0,
            degenerate_sample=True,
        )
        return result

    generator = np.random.default_rng(seed)
    indices = generator.integers(
        0,
        sample_count,
        size=(replicates, sample_count),
    )
    resamples = array[indices]
    resample_means = np.mean(resamples, axis=1)
    resample_sd = np.std(resamples, axis=1, ddof=1)
    resample_standard_error = resample_sd / math.sqrt(sample_count)
    differences = resample_means - point
    studentized = np.empty(replicates, dtype=np.float64)
    finite = resample_standard_error > 0.0
    studentized[finite] = differences[finite] / resample_standard_error[finite]
    zero_error = ~finite
    studentized[zero_error] = np.where(
        differences[zero_error] > 0.0,
        np.inf,
        np.where(differences[zero_error] < 0.0, -np.inf, 0.0),
    )

    tail = 0.5 * (1.0 - confidence_level)
    with np.errstate(invalid="ignore"):
        lower_pivot, upper_pivot = np.quantile(
            studentized,
            [tail, 1.0 - tail],
            method="linear",
        )
    lower = point - float(upper_pivot) * sample_standard_error
    upper = point - float(lower_pivot) * sample_standard_error
    if not np.isfinite(lower) or not np.isfinite(upper):
        result = _not_estimable_interval(
            method=_PRIMARY_INTERVAL_METHOD,
            sample_count=sample_count,
            confidence_level=confidence_level,
            reason="too many degenerate bootstrap resamples for finite pivots",
        )
        result.update(
            point_estimate=point,
            sample_standard_deviation=sample_sd,
            standard_error=sample_standard_error,
            replicates=replicates,
            seed=seed,
            finite_studentized_replicate_count=int(np.sum(finite)),
            finite_studentized_replicate_fraction=float(np.mean(finite)),
            degenerate_sample=False,
        )
        return result

    return {
        "estimable": True,
        "method": _PRIMARY_INTERVAL_METHOD,
        "confidence_level": confidence_level,
        "sample_count": sample_count,
        "point_estimate": point,
        "sample_standard_deviation": sample_sd,
        "standard_error": sample_standard_error,
        "lower_pivot_quantile": float(lower_pivot),
        "upper_pivot_quantile": float(upper_pivot),
        "lower": lower,
        "upper": upper,
        "replicates": replicates,
        "seed": seed,
        "finite_studentized_replicate_count": int(np.sum(finite)),
        "finite_studentized_replicate_fraction": float(np.mean(finite)),
        "degenerate_sample": False,
        "finite_sample_coverage_guaranteed": False,
    }


def interval_excludes_nonpositive_effect(interval: Mapping[str, Any]) -> bool:
    """Return whether one usable interval has a strictly positive lower bound."""

    if interval.get("estimable") is not True:
        return False
    if interval.get("degenerate_sample") is True:
        return False
    lower = interval.get("lower")
    return type(lower) in {int, float} and np.isfinite(float(lower)) and lower > 0.0


def _validated_registered_interval_pair(
    primary: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> int:
    if primary.get("method") != _PRIMARY_INTERVAL_METHOD:
        raise ValueError("primary interval method is not the registered bootstrap-t")
    if robustness.get("method") != _ROBUSTNESS_INTERVAL_METHOD:
        raise ValueError("robustness interval method is not the registered Student-t")

    primary_count = primary.get("sample_count")
    robustness_count = robustness.get("sample_count")
    if type(primary_count) is not int or primary_count < 0:
        raise ValueError("primary interval sample_count must be nonnegative")
    if type(robustness_count) is not int or robustness_count < 0:
        raise ValueError("robustness interval sample_count must be nonnegative")
    if primary_count != robustness_count:
        raise ValueError("registered intervals must use the same session sample")

    for name, interval in (("primary", primary), ("robustness", robustness)):
        confidence_level = interval.get("confidence_level")
        if type(confidence_level) not in {int, float} or not np.isclose(
            float(confidence_level),
            REAL_EFFECT_CONFIDENCE_LEVEL,
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(
                f"{name} interval confidence_level differs from the registered value"
            )

    if primary.get("estimable") is True and robustness.get("estimable") is True:
        primary_point = primary.get("point_estimate")
        robustness_point = robustness.get("point_estimate")
        if type(primary_point) not in {int, float} or not np.isfinite(
            float(primary_point)
        ):
            raise ValueError("primary interval point estimate must be finite")
        if type(robustness_point) not in {int, float} or not np.isfinite(
            float(robustness_point)
        ):
            raise ValueError("robustness interval point estimate must be finite")
        if not np.isclose(
            float(primary_point),
            float(robustness_point),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError("registered intervals must estimate the same session mean")
    return primary_count


def registered_positive_effect_interval_decision(
    primary: Mapping[str, Any],
    robustness: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the predeclared two-interval positive-claim rule.

    Bootstrap-t is primary. Student-t may veto a positive claim but can never
    rescue a primary interval that includes zero or is not estimable. The gate
    also fails closed for a zero-variance session panel because neither required
    method can estimate between-session uncertainty in that case.
    """

    sample_count = _validated_registered_interval_pair(primary, robustness)
    primary_passed = interval_excludes_nonpositive_effect(primary)
    robustness_passed = interval_excludes_nonpositive_effect(robustness)
    degenerate = (
        primary.get("degenerate_sample") is True
        or robustness.get("degenerate_sample") is True
    )
    return {
        "primary_interval_method": primary.get("method"),
        "required_robustness_interval_method": robustness.get("method"),
        "sample_count": sample_count,
        "registered_interval_inputs_match": True,
        "primary_interval_estimable": primary.get("estimable") is True,
        "required_robustness_interval_estimable": (robustness.get("estimable") is True),
        "degenerate_session_panel": degenerate,
        "degenerate_session_panel_blocks_positive_claim": True,
        "primary_interval_excludes_nonpositive_effect": primary_passed,
        "required_robustness_interval_excludes_nonpositive_effect": (robustness_passed),
        "positive_claim_interval_gate_passed": (
            primary_passed and robustness_passed and not degenerate
        ),
        "robustness_interval_may_veto_positive_claim": True,
        "robustness_interval_may_rescue_primary_failure": False,
        "negative_or_bounded_result_remains_reportable": True,
    }


__all__ = [
    "REAL_EFFECT_BOOTSTRAP_REPLICATES",
    "REAL_EFFECT_BOOTSTRAP_SEED",
    "REAL_EFFECT_CONFIDENCE_LEVEL",
    "bootstrap_t_mean_interval",
    "interval_excludes_nonpositive_effect",
    "percentile_bootstrap_mean_interval",
    "registered_positive_effect_interval_decision",
    "student_t_mean_interval",
]
