"""Consume trajectory-calibrated finite-action regret envelopes fail closed.

BayesianPhysTwin may augment a registered finite-support worst-case regret vector
with a split-conformal trajectory-level inflation radius.  This module verifies
that record independently before Causal4D selects a physical intervention.  It
never trusts a supplied selected index, admissibility mask, or inflated bound.

A nonfallback action is emitted only when it is the unique minimum among the
candidate actions whose independently reconstructed inflated regret is below
the declared tolerance.  Infinite radii, malformed records, and admissible ties
all return or raise before an unsupported intervention can be selected.

The conformal statement is marginal over an exchangeable complete trajectory
and simultaneous only for the registered decisions and actions represented by
the score.  It is not pointwise conditional validity, unseen-object transport,
or a deployment-safety certificate.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Real
from typing import Any, Final, NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt

FloatArray: TypeAlias = npt.NDArray[np.float64]
BoolArray: TypeAlias = npt.NDArray[np.bool_]

SUPPORT_ROBUST_INTERVENTION_VERSION: Final = 1
SUPPORT_ROBUST_INTERVENTION_SEMANTICS: Final = (
    "independently-verified-trajectory-conformal-regret-intervention-v1"
)
BAYESIAN_PHYSTWIN_ENVELOPE_VERSION: Final = 1
BAYESIAN_PHYSTWIN_ENVELOPE_SEMANTICS: Final = (
    "trajectory-split-conformal-simultaneous-action-regret-inflation-v1"
)
SUPPORT_ROBUST_INTERVENTION_CLAIM_BOUNDARY: Final = (
    "The consumer verifies arithmetic and fail-closed action-selection semantics "
    "for the supplied finite action record. The trajectory-level conformal claim "
    "still requires exchangeable complete trajectories and remains marginal over "
    "a future trajectory. This module does not validate exchangeability, the "
    "physical support, provider, action set, loss, regret budget, unseen-object "
    "transport, deployment, or safety."
)

_ATOL: Final = 1e-12


def _immutable_float64(value: object) -> FloatArray:
    array = np.ascontiguousarray(value, dtype=np.float64)
    array.setflags(write=False)
    return array


def _immutable_bool(value: object) -> BoolArray:
    array = np.ascontiguousarray(value, dtype=np.bool_)
    array.setflags(write=False)
    return array


def _field(record: object, name: str) -> object:
    if isinstance(record, Mapping):
        if name not in record:
            raise ValueError(f"support-robust record is missing {name!r}")
        return record[name]
    if not hasattr(record, name):
        raise ValueError(f"support-robust record is missing {name!r}")
    return getattr(record, name)


def _optional_field(record: object, name: str) -> object | None:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def _summary(record: object) -> Mapping[str, object] | None:
    direct = _optional_field(record, "summary")
    if isinstance(direct, Mapping):
        return direct
    if callable(direct):
        value = direct()
        if not isinstance(value, Mapping):
            raise ValueError("support-robust summary() must return a mapping")
        return value
    return None


def _finite_nonnegative_vector(value: object, *, name: str) -> FloatArray:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must contain real numeric values")
    array = np.ascontiguousarray(raw, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError(f"{name} must contain at least two actions")
    if not np.all(np.isfinite(array)) or np.any(array < 0.0):
        raise ValueError(f"{name} must be finite and nonnegative")
    return _immutable_float64(array)


def _boolean_vector(value: object, *, name: str, size: int) -> BoolArray:
    raw = np.asarray(value)
    if raw.dtype.kind != "b":
        raise ValueError(f"{name} must contain boolean values")
    array = np.ascontiguousarray(raw, dtype=np.bool_)
    if array.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},)")
    return _immutable_bool(array)


def _nonnegative_real_or_infinity(value: object, *, name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a nonnegative real or positive infinity")
    result = float(value)
    if np.isnan(result) or result < 0.0:
        raise ValueError(f"{name} must be a nonnegative real or positive infinity")
    return result


def _finite_nonnegative_real(value: object, *, name: str) -> float:
    result = _nonnegative_real_or_infinity(value, name=name)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _integer(value: object, *, name: str, size: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise ValueError(f"{name} must be an integer")
    result = int(value)
    if not 0 <= result < size:
        raise ValueError(f"{name} is out of range")
    return result


def _action_names(value: Sequence[str], *, size: int) -> tuple[str, ...]:
    names = tuple(value)
    if len(names) != size:
        raise ValueError(f"action_names must contain exactly {size} entries")
    if any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError("action_names must contain nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("action_names must be unique")
    return names


def _validate_optional_metadata(record: object) -> None:
    summary = _summary(record)
    if summary is None:
        return
    version = summary.get("version")
    semantics = summary.get("semantics")
    if version != BAYESIAN_PHYSTWIN_ENVELOPE_VERSION:
        raise ValueError("unsupported BayesianPhysTwin envelope version")
    if semantics != BAYESIAN_PHYSTWIN_ENVELOPE_SEMANTICS:
        raise ValueError("unsupported BayesianPhysTwin envelope semantics")


class SupportRobustInterventionV1(NamedTuple):
    """Independently reconstructed Causal4D physical-intervention decision."""

    action_names: tuple[str, ...]
    registered_worst_case_regret: FloatArray
    conformal_radius: float
    inflated_regret_upper_bound: FloatArray
    regret_tolerance: float
    candidate_action_mask: BoolArray
    tolerance_admissible_action_mask: BoolArray
    selected_action_index: int
    selected_action_name: str
    fallback_action_index: int
    fallback_action_name: str
    used_fallback: bool

    def summary(self) -> dict[str, object]:
        return {
            "version": SUPPORT_ROBUST_INTERVENTION_VERSION,
            "semantics": SUPPORT_ROBUST_INTERVENTION_SEMANTICS,
            "selected_action_index": self.selected_action_index,
            "selected_action_name": self.selected_action_name,
            "fallback_action_index": self.fallback_action_index,
            "fallback_action_name": self.fallback_action_name,
            "used_fallback": self.used_fallback,
            "conformal_radius": self.conformal_radius,
            "regret_tolerance": self.regret_tolerance,
            "admissible_action_count": int(
                np.count_nonzero(self.tolerance_admissible_action_mask)
            ),
            "claim_boundary": SUPPORT_ROBUST_INTERVENTION_CLAIM_BOUNDARY,
        }


def consume_support_robust_decision(
    record: object,
    action_names: Sequence[str],
    *,
    expected_fallback_action_index: int | None = None,
) -> SupportRobustInterventionV1:
    """Verify a BayesianPhysTwin support-robust record and select fail closed.

    ``record`` may be a live ``SupportRobustDecisionV1`` object or a mapping
    retaining all of its fields.  Optional ``summary`` metadata is checked when
    present.  The selected index, masks, and inflated regrets are reconstructed
    from the registered regret vector, conformal radius, and tolerance before
    any action name is returned.
    """

    _validate_optional_metadata(record)
    registered = _finite_nonnegative_vector(
        _field(record, "registered_worst_case_regret"),
        name="registered_worst_case_regret",
    )
    size = int(registered.size)
    names = _action_names(action_names, size=size)
    radius = _nonnegative_real_or_infinity(
        _field(record, "conformal_radius"),
        name="conformal_radius",
    )
    tolerance = _finite_nonnegative_real(
        _field(record, "regret_tolerance"),
        name="regret_tolerance",
    )
    fallback = _integer(
        _field(record, "fallback_action_index"),
        name="fallback_action_index",
        size=size,
    )
    if expected_fallback_action_index is not None:
        expected = _integer(
            expected_fallback_action_index,
            name="expected_fallback_action_index",
            size=size,
        )
        if fallback != expected:
            raise ValueError("support-robust fallback action does not match caller")

    supplied_candidates = _boolean_vector(
        _field(record, "candidate_action_mask"),
        name="candidate_action_mask",
        size=size,
    )
    candidates = np.asarray(supplied_candidates, dtype=np.bool_).copy()
    if candidates[fallback]:
        raise ValueError("candidate_action_mask must exclude the fallback action")
    if not np.any(candidates):
        raise ValueError("candidate_action_mask must select a nonfallback action")

    reconstructed_inflated = np.asarray(registered, dtype=np.float64) + radius
    reconstructed_admissible = candidates & (
        reconstructed_inflated <= tolerance + _ATOL
    )
    reconstructed_selected = fallback
    reconstructed_used_fallback = True
    if np.any(reconstructed_admissible):
        minimum = float(np.min(reconstructed_inflated[reconstructed_admissible]))
        minimizers = np.flatnonzero(
            reconstructed_admissible
            & np.isclose(
                reconstructed_inflated,
                minimum,
                rtol=0.0,
                atol=_ATOL,
            )
        )
        if minimizers.size == 1:
            reconstructed_selected = int(minimizers[0])
            reconstructed_used_fallback = False

    supplied_inflated = np.asarray(
        _field(record, "inflated_regret_upper_bound"),
        dtype=np.float64,
    )
    if supplied_inflated.shape != (size,) or np.any(np.isnan(supplied_inflated)):
        raise ValueError("inflated_regret_upper_bound has invalid shape or NaN")
    if not np.allclose(
        supplied_inflated,
        reconstructed_inflated,
        rtol=0.0,
        atol=_ATOL,
    ):
        raise ValueError("supplied inflated regret bounds failed verification")

    supplied_admissible = _boolean_vector(
        _field(record, "tolerance_admissible_action_mask"),
        name="tolerance_admissible_action_mask",
        size=size,
    )
    if not np.array_equal(supplied_admissible, reconstructed_admissible):
        raise ValueError("supplied admissibility mask failed verification")

    supplied_selected = _integer(
        _field(record, "selected_action_index"),
        name="selected_action_index",
        size=size,
    )
    if supplied_selected != reconstructed_selected:
        raise ValueError("supplied selected action failed verification")
    supplied_used_fallback = _field(record, "used_fallback")
    if not isinstance(supplied_used_fallback, (bool, np.bool_)):
        raise ValueError("used_fallback must be boolean")
    if bool(supplied_used_fallback) != reconstructed_used_fallback:
        raise ValueError("supplied fallback flag failed verification")

    return SupportRobustInterventionV1(
        action_names=names,
        registered_worst_case_regret=registered,
        conformal_radius=radius,
        inflated_regret_upper_bound=_immutable_float64(reconstructed_inflated),
        regret_tolerance=tolerance,
        candidate_action_mask=_immutable_bool(candidates),
        tolerance_admissible_action_mask=_immutable_bool(
            reconstructed_admissible
        ),
        selected_action_index=reconstructed_selected,
        selected_action_name=names[reconstructed_selected],
        fallback_action_index=fallback,
        fallback_action_name=names[fallback],
        used_fallback=reconstructed_used_fallback,
    )


__all__ = [
    "BAYESIAN_PHYSTWIN_ENVELOPE_SEMANTICS",
    "BAYESIAN_PHYSTWIN_ENVELOPE_VERSION",
    "SUPPORT_ROBUST_INTERVENTION_CLAIM_BOUNDARY",
    "SUPPORT_ROBUST_INTERVENTION_SEMANTICS",
    "SUPPORT_ROBUST_INTERVENTION_VERSION",
    "SupportRobustInterventionV1",
    "consume_support_robust_decision",
]
