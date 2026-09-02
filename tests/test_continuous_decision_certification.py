from __future__ import annotations

import numpy as np
import pytest

from causal4d.continuous_decision_certification import (
    ParameterBox,
    certify_continuous_decision,
)


def _triangular_losses(parameter: np.ndarray) -> tuple[float, float]:
    x = float(parameter[0])
    center = 0.375
    half_width = 0.125
    bump = max(0.0, 1.0 - abs(x - center) / half_width)
    return bump, 0.1


def test_continuous_certificate_finds_grid_missed_counterexample() -> None:
    result = certify_continuous_decision(
        _triangular_losses,
        ParameterBox((-1.0,), (1.0,)),
        (8.0, 0.0),
        regret_tolerance=0.15,
        maximum_evaluations=1025,
    )
    assert result.status == "certified"
    assert result.selected_action_index == 1
    action_zero = result.action_bounds[0]
    action_one = result.action_bounds[1]
    assert action_zero.witnessed_inadmissible
    assert action_zero.witnessed_lower_bound >= 0.89
    assert 0.30 <= action_zero.witness_parameter[0] <= 0.45
    assert action_one.certified_admissible
    assert action_one.verified_upper_bound <= 0.15 + 1e-12


def test_restricted_domain_certifies_action_without_identifying_nuisance() -> None:
    result = certify_continuous_decision(
        _triangular_losses,
        ParameterBox((-1.0, -1.0), (0.20, 1.0)),
        (0.0, 0.0),
        regret_tolerance=0.05,
        maximum_evaluations=9,
    )
    assert result.status == "certified"
    assert result.selected_action_index == 0
    assert result.action_bounds[0].verified_upper_bound <= 0.05 + 1e-12
    assert result.action_bounds[1].witnessed_inadmissible
    assert result.maximum_remaining_radius > 0.0


def test_claim_boundary_exposes_unvalidated_lipschitz_assumption() -> None:
    result = certify_continuous_decision(
        _triangular_losses,
        ParameterBox((-1.0,), (1.0,)),
        (0.0, 0.0),
        regret_tolerance=0.01,
        maximum_evaluations=9,
    )
    assert result.status == "certified"
    assert result.selected_action_index == 0
    assert "valid global action-loss Lipschitz constants" in result.as_dict()[
        "claim_boundary"
    ]


def test_budget_exhaustion_fails_closed() -> None:
    def losses(parameter: np.ndarray) -> tuple[float, float]:
        return float(parameter[0] ** 2), 0.2

    result = certify_continuous_decision(
        losses,
        ParameterBox((-1.0,), (1.0,)),
        (2.0, 0.0),
        regret_tolerance=0.05,
        maximum_evaluations=1,
    )
    assert result.status == "inconclusive"
    assert result.used_exact_fallback
    assert result.reason_code == "continuous-search-budget-exhausted"


def test_multiple_uniformly_admissible_actions_fail_closed() -> None:
    result = certify_continuous_decision(
        lambda _: (0.0, 0.0),
        ParameterBox((-1.0,), (1.0,)),
        (0.0, 0.0),
        regret_tolerance=0.0,
    )
    assert result.status == "multiple-admissible-actions"
    assert result.selected_action_index is None


def test_no_uniformly_admissible_action_returns_counterexamples() -> None:
    result = certify_continuous_decision(
        lambda value: (
            float((value[0] - 0.5) ** 2),
            float((value[0] + 0.5) ** 2),
        ),
        ParameterBox((-1.0,), (1.0,)),
        (3.0, 3.0),
        regret_tolerance=0.1,
        maximum_evaluations=1025,
    )
    assert result.status == "no-admissible-action"
    assert all(bound.witnessed_inadmissible for bound in result.action_bounds)


def test_parameter_box_validation_and_containment() -> None:
    box = ParameterBox((-1.0, 0.0), (1.0, 2.0))
    assert box.center == (0.0, 1.0)
    assert box.contains((0.5, 1.5))
    assert not box.contains((2.0, 1.5))
    with pytest.raises(ValueError):
        ParameterBox((1.0,), (-1.0,))


def test_verified_regret_envelopes_cover_dense_affine_reference() -> None:
    coefficients = np.asarray(
        (
            (1.0, -2.0),
            (-0.5, 0.75),
            (0.25, 0.10),
        ),
        dtype=np.float64,
    )
    offsets = np.asarray((0.1, -0.2, 0.05), dtype=np.float64)

    def losses(parameter: np.ndarray) -> np.ndarray:
        return coefficients @ parameter + offsets

    lipschitz = np.sum(np.abs(coefficients), axis=1)
    result = certify_continuous_decision(
        losses,
        ParameterBox((-1.0, -1.0), (1.0, 1.0)),
        lipschitz,
        regret_tolerance=0.01,
        maximum_evaluations=65,
    )
    grid = np.linspace(-1.0, 1.0, 101)
    reference = []
    for first in grid:
        for second in grid:
            point_losses = losses(np.asarray((first, second)))
            reference.append(point_losses - np.min(point_losses))
    reference_worst = np.max(np.asarray(reference), axis=0)
    for bound, actual in zip(result.action_bounds, reference_worst):
        assert bound.witnessed_lower_bound <= actual + 1e-12
        assert bound.verified_upper_bound >= actual - 1e-12


def test_continuous_certificate_is_deterministic() -> None:
    arguments = (
        _triangular_losses,
        ParameterBox((-1.0,), (1.0,)),
        (8.0, 0.0),
    )
    first = certify_continuous_decision(
        *arguments,
        regret_tolerance=0.15,
        maximum_evaluations=1025,
    )
    second = certify_continuous_decision(
        *arguments,
        regret_tolerance=0.15,
        maximum_evaluations=1025,
    )
    assert first.as_dict() == second.as_dict()
