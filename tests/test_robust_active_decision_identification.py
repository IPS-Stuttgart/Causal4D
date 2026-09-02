from __future__ import annotations

import numpy as np
import pytest

from causal4d.active_decision_identification import (
    CertificateOutcomeV1,
    CertificateProbeV1,
)
from causal4d.robust_active_decision_identification import (
    AmbiguousCertificateOutcomeV1,
    AmbiguousCertificateProbeV1,
    consume_ambiguous_probe_outcome,
    extremal_box_simplex_distribution,
    plan_distributionally_robust_active_decision,
    point_identified_probe,
)

ACTIONS = ("left", "right")
FALLBACK = "hold"


def certificate(
    pairwise: tuple[tuple[float, ...], ...],
    *,
    tolerance: float = 0.0,
) -> dict[str, object]:
    matrix = np.asarray(pairwise, dtype=np.float64)
    regret = np.maximum(np.max(matrix, axis=1), 0.0)
    robust = np.all(matrix <= 1e-12, axis=1)
    admissible = regret <= tolerance + 1e-12
    minimum = float(np.min(regret))
    minimax = int(np.flatnonzero(np.isclose(regret, minimum, atol=1e-12))[0])
    return {
        "summary": {
            "version": 1,
            "semantics": (
                "exact-worst-case-regret-over-registered-query-quotient-"
                "and-prior-support-v1"
            ),
            "action_count": int(matrix.shape[0]),
            "minimax_action_index": minimax,
            "minimax_worst_case_regret": minimum,
            "regret_tolerance": tolerance,
            "has_tolerance_admissible_action": bool(np.any(admissible)),
            "uniquely_tolerance_identified": bool(np.count_nonzero(admissible) == 1),
            "has_robustly_optimal_action": bool(np.any(robust)),
            "uniquely_robustly_optimal": bool(np.count_nonzero(robust) == 1),
        },
        "pairwise_worst_case_loss_gap": matrix.tolist(),
        "worst_case_regret": regret.tolist(),
        "minimax_action_index": minimax,
        "minimax_worst_case_regret": minimum,
        "regret_tolerance": tolerance,
        "tolerance_admissible_action_mask": admissible.tolist(),
        "robustly_optimal_action_mask": robust.tolist(),
    }


CURRENT = certificate(((0.0, 1.0), (1.0, 0.0)))
LEFT = certificate(((0.0, -1.0), (1.0, 0.0)))
RIGHT = certificate(((0.0, 1.0), (-1.0, 0.0)))
LEFT_TOLERANT = certificate(((0.0, -1.0), (1.0, 0.0)), tolerance=0.2)
RIGHT_TOLERANT = certificate(((0.0, 1.0), (-1.0, 0.0)), tolerance=0.2)


def ambiguous_outcome(
    lower: float,
    upper: float,
    cert: object,
    *,
    inflation: float = 0.0,
) -> AmbiguousCertificateOutcomeV1:
    return AmbiguousCertificateOutcomeV1(lower, upper, cert, inflation)


def test_extremal_box_simplex_distribution_is_exact() -> None:
    lower = (0.1, 0.2, 0.0)
    upper = (0.5, 0.7, 0.6)
    values = (1.0, 3.0, 2.0)
    maximum = extremal_box_simplex_distribution(
        lower,
        upper,
        values,
        maximize=True,
    )
    minimum = extremal_box_simplex_distribution(
        lower,
        upper,
        values,
        maximize=False,
    )
    assert maximum == pytest.approx((0.1, 0.7, 0.2))
    assert minimum == pytest.approx((0.5, 0.2, 0.3))
    assert np.dot(maximum, values) == pytest.approx(2.6)
    assert np.dot(minimum, values) == pytest.approx(1.7)


def test_point_identified_embedding_recovers_nominal_value() -> None:
    nominal = CertificateProbeV1(
        "task",
        (
            CertificateOutcomeV1(0.5, LEFT),
            CertificateOutcomeV1(0.5, RIGHT),
        ),
        physical_risk=0.01,
        cost=0.05,
    )
    plan = plan_distributionally_robust_active_decision(
        CURRENT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(point_identified_probe(nominal),),
        risk_cap=0.05,
        cost_multiplier=1.0,
    )
    report = plan.probe_reports[0]
    assert plan.mode == "probe"
    assert plan.selected_probe_name == "task"
    assert report.worst_case_expected_posterior_minimax_worst_case_regret == 0.0
    assert report.worst_case_certification_probability == 1.0
    assert report.net_guaranteed_value == pytest.approx(0.95)


def test_robust_policy_rejects_nominally_best_fragile_probe() -> None:
    nominal_fragile = CertificateProbeV1(
        "fragile",
        (
            CertificateOutcomeV1(0.95, LEFT),
            CertificateOutcomeV1(0.05, CURRENT),
        ),
        physical_risk=0.01,
        cost=0.05,
    )
    nominal_stable = CertificateProbeV1(
        "stable",
        (
            CertificateOutcomeV1(0.5, LEFT_TOLERANT),
            CertificateOutcomeV1(0.5, RIGHT_TOLERANT),
        ),
        physical_risk=0.01,
        cost=0.05,
    )
    nominal_plan = plan_distributionally_robust_active_decision(
        CURRENT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(
            point_identified_probe(nominal_fragile),
            point_identified_probe(nominal_stable, regret_inflation=0.15),
        ),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_worst_case_certification_probability=0.9,
    )
    assert nominal_plan.selected_probe_name == "fragile"

    fragile = AmbiguousCertificateProbeV1(
        "fragile",
        (
            ambiguous_outcome(0.4, 0.95, LEFT),
            ambiguous_outcome(0.05, 0.6, CURRENT),
        ),
        physical_risk_upper=0.01,
        cost=0.05,
    )
    stable = AmbiguousCertificateProbeV1(
        "stable",
        (
            ambiguous_outcome(0.0, 1.0, LEFT_TOLERANT, inflation=0.15),
            ambiguous_outcome(0.0, 1.0, RIGHT_TOLERANT, inflation=0.15),
        ),
        physical_risk_upper=0.01,
        cost=0.05,
    )
    robust_plan = plan_distributionally_robust_active_decision(
        CURRENT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(fragile, stable),
        risk_cap=0.05,
        cost_multiplier=1.0,
        minimum_worst_case_certification_probability=1.0,
    )
    reports = {report.name: report for report in robust_plan.probe_reports}
    assert robust_plan.selected_probe_name == "stable"
    assert reports["fragile"].worst_case_certification_probability == pytest.approx(0.4)
    assert reports[
        "fragile"
    ].worst_case_expected_posterior_minimax_worst_case_regret == pytest.approx(0.6)
    assert reports["stable"].worst_case_certification_probability == 1.0
    assert reports["stable"].net_guaranteed_value == pytest.approx(0.8)


def test_regret_inflation_revokes_structural_certificate() -> None:
    probe = AmbiguousCertificateProbeV1(
        "inflated",
        (ambiguous_outcome(1.0, 1.0, LEFT_TOLERANT, inflation=0.21),),
    )
    plan = plan_distributionally_robust_active_decision(
        CURRENT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(probe,),
    )
    outcome = plan.probe_reports[0].outcome_decisions[0]
    assert plan.mode == "fallback"
    assert outcome.used_exact_fallback
    assert outcome.reason_code == "regret-inflation-exceeds-tolerance"
    assert outcome.inflated_selected_worst_case_regret == pytest.approx(0.21)


def test_risk_upper_bound_rejects_high_value_probe() -> None:
    probe = AmbiguousCertificateProbeV1(
        "unsafe",
        (
            ambiguous_outcome(0.5, 0.5, LEFT),
            ambiguous_outcome(0.5, 0.5, RIGHT),
        ),
        physical_risk_upper=0.2,
    )
    plan = plan_distributionally_robust_active_decision(
        CURRENT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(probe,),
        risk_cap=0.05,
    )
    assert plan.mode == "fallback"
    assert plan.reason_code == "no-safe-probe-under-risk-upper-bound"
    assert not plan.probe_reports[0].safe


def test_current_certificate_acts_without_probing() -> None:
    plan = plan_distributionally_robust_active_decision(
        LEFT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(),
    )
    assert plan.mode == "act"
    assert plan.action_name == "left"
    assert plan.probe_reports == ()


def test_realized_outcome_uses_inflated_fail_closed_router() -> None:
    probe = AmbiguousCertificateProbeV1(
        "task",
        (
            ambiguous_outcome(0.5, 0.5, LEFT_TOLERANT, inflation=0.1),
            ambiguous_outcome(0.5, 0.5, RIGHT_TOLERANT, inflation=0.3),
        ),
    )
    accepted = consume_ambiguous_probe_outcome(
        probe,
        0,
        ACTIONS,
        fallback_action_name=FALLBACK,
    )
    rejected = consume_ambiguous_probe_outcome(
        probe,
        1,
        ACTIONS,
        fallback_action_name=FALLBACK,
    )
    assert accepted.action_name == "left"
    assert accepted.certificate_level == "tolerance-admissible"
    assert rejected.action_name == FALLBACK
    assert rejected.used_exact_fallback


def test_probability_ambiguity_can_force_fallback() -> None:
    probe = AmbiguousCertificateProbeV1(
        "fragile",
        (
            ambiguous_outcome(0.1, 0.9, LEFT),
            ambiguous_outcome(0.1, 0.9, CURRENT),
        ),
    )
    plan = plan_distributionally_robust_active_decision(
        CURRENT,
        ACTIONS,
        fallback_action_name=FALLBACK,
        probes=(probe,),
        minimum_worst_case_certification_probability=0.8,
    )
    assert plan.mode == "fallback"
    assert plan.reason_code == "no-probe-meets-worst-case-certification-probability"
    assert plan.probe_reports[0].worst_case_certification_probability == pytest.approx(
        0.1
    )


def test_invalid_probability_boxes_fail_closed() -> None:
    with pytest.raises(ValueError, match="empty simplex"):
        AmbiguousCertificateProbeV1(
            "bad",
            (
                ambiguous_outcome(0.6, 0.7, LEFT),
                ambiguous_outcome(0.6, 0.7, RIGHT),
            ),
        )
    with pytest.raises(ValueError, match="empty simplex"):
        AmbiguousCertificateProbeV1(
            "bad",
            (
                ambiguous_outcome(0.0, 0.4, LEFT),
                ambiguous_outcome(0.0, 0.4, RIGHT),
            ),
        )


def test_duplicate_probe_names_fail_closed() -> None:
    probe = AmbiguousCertificateProbeV1(
        "same",
        (ambiguous_outcome(1.0, 1.0, LEFT),),
    )
    with pytest.raises(ValueError, match="unique"):
        plan_distributionally_robust_active_decision(
            CURRENT,
            ACTIONS,
            fallback_action_name=FALLBACK,
            probes=(probe, probe),
        )
