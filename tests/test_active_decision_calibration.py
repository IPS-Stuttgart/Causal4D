from __future__ import annotations

import math

import numpy as np
import pytest

from causal4d.active_decision_calibration import (
    apply_complete_group_regret_margin,
    calibrate_complete_group_regret,
    complete_group_nonconformity_scores,
    simultaneous_hoeffding_probability_box,
)
from causal4d.robust_active_decision_identification import (
    AmbiguousCertificateOutcomeV1,
    AmbiguousCertificateProbeV1,
    apply_complete_group_calibration_to_probe,
    apply_probability_box_to_probe,
)


def test_scores_take_maximum_inside_each_complete_group() -> None:
    structural = np.asarray(
        [
            [[0.1, 0.2], [0.3, 0.4]],
            [[0.2, 0.2], [0.2, 0.2]],
        ]
    )
    realized = np.asarray(
        [
            [[0.0, 0.4], [0.35, 0.1]],
            [[0.3, 0.1], [0.5, 0.0]],
        ]
    )
    assert complete_group_nonconformity_scores(structural, realized) == pytest.approx(
        (0.2, 0.3)
    )


def test_split_conformal_rank_and_margin_are_exact() -> None:
    structural = np.zeros((10, 2), dtype=np.float64)
    realized = np.column_stack(
        (
            np.arange(10, dtype=np.float64) / 100.0,
            np.zeros(10, dtype=np.float64),
        )
    )
    calibration = calibrate_complete_group_regret(
        structural,
        realized,
        alpha=0.2,
        primary_unit="complete-object",
    )
    assert calibration.conformal_rank == 9
    assert calibration.additive_margin == pytest.approx(0.08)
    assert calibration.finite_margin_available
    assert calibration.comparisons_per_group == 2
    assert calibration.primary_unit == "complete-object"


def test_too_few_groups_produce_infinite_fail_closed_margin() -> None:
    calibration = calibrate_complete_group_regret(
        np.zeros((8, 1)),
        np.zeros((8, 1)),
        alpha=0.1,
    )
    assert not calibration.finite_margin_available
    assert calibration.additive_margin == math.inf
    assert calibration.as_dict()["additive_margin"] is None
    with pytest.raises(ValueError, match="too few"):
        calibration.require_finite_margin()


def test_negative_empirical_excess_does_not_tighten_structural_bound() -> None:
    structural = np.ones((9, 3), dtype=np.float64)
    realized = np.zeros((9, 3), dtype=np.float64)
    calibration = calibrate_complete_group_regret(
        structural,
        realized,
        alpha=0.2,
    )
    assert calibration.additive_margin == 0.0
    assert calibration.negative_margin_was_clamped


def test_margin_application_is_readonly_and_group_wide() -> None:
    structural = np.zeros((9, 2), dtype=np.float64)
    realized = np.full((9, 2), 0.15, dtype=np.float64)
    calibration = calibrate_complete_group_regret(
        structural,
        realized,
        alpha=0.2,
    )
    inflated = apply_complete_group_regret_margin(
        np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        calibration,
    )
    assert inflated == pytest.approx(np.asarray([[0.25, 0.35], [0.45, 0.55]]))
    assert not inflated.flags.writeable


def test_calibration_margin_is_added_to_every_probe_branch() -> None:
    calibration = calibrate_complete_group_regret(
        np.zeros((9, 2)),
        np.full((9, 2), 0.15),
        alpha=0.2,
    )
    probe = AmbiguousCertificateProbeV1(
        "probe",
        (
            AmbiguousCertificateOutcomeV1(0.4, 0.6, object(), 0.02),
            AmbiguousCertificateOutcomeV1(0.4, 0.6, object(), 0.03),
        ),
        physical_risk_upper=0.1,
        cost=0.2,
    )
    inflated = apply_complete_group_calibration_to_probe(probe, calibration)
    assert [item.regret_inflation for item in inflated.outcomes] == pytest.approx(
        [0.17, 0.18]
    )
    assert inflated.physical_risk_upper == probe.physical_risk_upper
    assert inflated.cost == probe.cost


def test_hoeffding_box_is_simultaneous_and_selection_safe() -> None:
    local = simultaneous_hoeffding_probability_box(
        (95, 5),
        alpha=0.1,
        registered_probability_count=2,
    )
    familywise = simultaneous_hoeffding_probability_box(
        (95, 5),
        alpha=0.1,
        registered_probability_count=8,
    )
    assert local.empirical_probabilities == pytest.approx((0.95, 0.05))
    assert local.lower_bounds[0] <= 0.95 <= local.upper_bounds[0]
    assert local.lower_bounds[1] <= 0.05 <= local.upper_bounds[1]
    assert sum(local.lower_bounds) <= 1.0 <= sum(local.upper_bounds)
    assert familywise.simultaneous_radius > local.simultaneous_radius
    assert familywise.registered_probability_count == 8


def test_probability_box_replaces_bounds_without_changing_certificates() -> None:
    box = simultaneous_hoeffding_probability_box(
        (80, 20),
        alpha=0.1,
        registered_probability_count=4,
    )
    first = object()
    second = object()
    probe = AmbiguousCertificateProbeV1(
        "probe",
        (
            AmbiguousCertificateOutcomeV1(0.5, 0.5, first, 0.02),
            AmbiguousCertificateOutcomeV1(0.5, 0.5, second, 0.03),
        ),
        physical_risk_upper=0.1,
        cost=0.2,
    )
    calibrated = apply_probability_box_to_probe(probe, box)
    assert [item.probability_lower for item in calibrated.outcomes] == pytest.approx(
        box.lower_bounds
    )
    assert [item.probability_upper for item in calibrated.outcomes] == pytest.approx(
        box.upper_bounds
    )
    assert calibrated.outcomes[0].certificate is first
    assert calibrated.outcomes[1].certificate is second
    assert [item.regret_inflation for item in calibrated.outcomes] == pytest.approx(
        (0.02, 0.03)
    )
