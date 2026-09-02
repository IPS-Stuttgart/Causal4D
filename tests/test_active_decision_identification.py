from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from causal4d.active_decision_identification import (
    ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY,
    ActiveDecisionPlanV1,
    CertificateOutcomeV1,
    CertificateProbeV1,
    consume_probe_outcome,
    plan_active_decision,
)
from causal4d.decision_identifiable_intervention import (
    QUERY_DECISION_CERTIFICATE_SEMANTICS,
)


@dataclass
class CertificateFixture:
    pairwise_worst_case_loss_gap: np.ndarray
    worst_case_regret: np.ndarray
    minimax_action_index: int
    minimax_worst_case_regret: float
    regret_tolerance: float
    tolerance_admissible_action_mask: np.ndarray
    robustly_optimal_action_mask: np.ndarray

    def summary(self) -> dict[str, object]:
        tolerance_count = int(np.count_nonzero(self.tolerance_admissible_action_mask))
        robust_count = int(np.count_nonzero(self.robustly_optimal_action_mask))
        return {
            "version": 1,
            "semantics": QUERY_DECISION_CERTIFICATE_SEMANTICS,
            "action_count": int(self.worst_case_regret.size),
            "minimax_action_index": self.minimax_action_index,
            "minimax_worst_case_regret": self.minimax_worst_case_regret,
            "regret_tolerance": self.regret_tolerance,
            "has_tolerance_admissible_action": tolerance_count > 0,
            "uniquely_tolerance_identified": tolerance_count == 1,
            "has_robustly_optimal_action": robust_count > 0,
            "uniquely_robustly_optimal": robust_count == 1,
        }


def certificate(
    pairwise: list[list[float]],
    *,
    tolerance: float = 0.05,
) -> CertificateFixture:
    matrix = np.asarray(pairwise, dtype=np.float64)
    regret = np.maximum(np.max(matrix, axis=1), 0.0)
    robust = np.all(matrix <= 1e-12, axis=1)
    admissible = regret <= tolerance + 1e-12
    minimum = float(np.min(regret))
    minimax = int(np.flatnonzero(np.isclose(regret, minimum, atol=1e-12))[0])
    return CertificateFixture(
        pairwise_worst_case_loss_gap=matrix,
        worst_case_regret=regret,
        minimax_action_index=minimax,
        minimax_worst_case_regret=minimum,
        regret_tolerance=tolerance,
        tolerance_admissible_action_mask=admissible,
        robustly_optimal_action_mask=robust,
    )


def ambiguous() -> CertificateFixture:
    return certificate([[0.0, 0.4], [0.4, 0.0]])


def left_certified() -> CertificateFixture:
    return certificate([[0.0, -0.2], [0.8, 0.0]])


def right_certified() -> CertificateFixture:
    return certificate([[0.0, 0.8], [-0.2, 0.0]])


def task_probe(
    *,
    name: str = "task-probe",
    physical_risk: float = 0.02,
    cost: float = 0.05,
) -> CertificateProbeV1:
    return CertificateProbeV1(
        name=name,
        outcomes=(
            CertificateOutcomeV1(0.5, left_certified()),
            CertificateOutcomeV1(0.5, right_certified()),
        ),
        physical_risk=physical_risk,
        cost=cost,
    )


def uninformative_probe(
    *,
    name: str = "nuisance-probe",
    physical_risk: float = 0.01,
) -> CertificateProbeV1:
    return CertificateProbeV1(
        name=name,
        outcomes=(
            CertificateOutcomeV1(0.5, ambiguous()),
            CertificateOutcomeV1(0.5, ambiguous()),
        ),
        physical_risk=physical_risk,
    )


def test_current_certificate_acts_immediately_without_probing() -> None:
    plan = plan_active_decision(
        left_certified(),
        ("left", "right"),
        fallback_action_name="hold",
        probes=(task_probe(),),
        risk_cap=0.1,
    )
    assert plan.mode == "act"
    assert plan.action_name == "left"
    assert plan.selected_probe_name is None
    assert plan.probe_reports == ()
    assert not plan.used_exact_fallback


def test_task_probe_is_selected_over_uninformative_and_unsafe_probes() -> None:
    plan = plan_active_decision(
        ambiguous(),
        ("left", "right"),
        fallback_action_name="hold",
        probes=(
            uninformative_probe(),
            task_probe(),
            task_probe(name="risky-perfect", physical_risk=0.8, cost=0.0),
        ),
        risk_cap=0.1,
        cost_multiplier=1.0,
        minimum_certification_probability=1.0,
    )
    assert plan.mode == "probe"
    assert plan.action_name is None
    assert plan.selected_probe_name == "task-probe"
    reports = {report.name: report for report in plan.probe_reports}
    assert reports["task-probe"].expected_regret_reduction == pytest.approx(0.4)
    assert reports["task-probe"].certification_probability == pytest.approx(1.0)
    assert reports["task-probe"].net_value == pytest.approx(0.35)
    assert reports["nuisance-probe"].expected_regret_reduction == pytest.approx(0.0)
    assert not reports["risky-perfect"].safe


def test_dependence_destroyed_or_zero_value_probe_returns_exact_fallback() -> None:
    plan = plan_active_decision(
        ambiguous(),
        ("left", "right"),
        fallback_action_name="hold",
        probes=(uninformative_probe(name="dependence-destroyed"),),
        risk_cap=0.1,
    )
    assert plan.mode == "fallback"
    assert plan.action_name == "hold"
    assert plan.used_exact_fallback
    assert plan.reason_code == "no-probe-meets-certification-probability"


def test_minimum_certification_probability_blocks_partial_probe() -> None:
    partial = CertificateProbeV1(
        name="partial",
        outcomes=(
            CertificateOutcomeV1(0.5, left_certified()),
            CertificateOutcomeV1(
                0.5,
                certificate([[0.0, 0.2], [0.3, 0.0]]),
            ),
        ),
    )
    plan = plan_active_decision(
        ambiguous(),
        ("left", "right"),
        fallback_action_name="hold",
        probes=(partial,),
        minimum_certification_probability=0.75,
    )
    assert plan.mode == "fallback"
    assert plan.reason_code == "no-probe-meets-certification-probability"
    assert plan.probe_reports[0].certification_probability == pytest.approx(0.5)


def test_realized_probe_outcome_routes_through_existing_certificate_consumer() -> None:
    probe = task_probe()
    left = consume_probe_outcome(
        probe,
        0,
        ("left", "right"),
        fallback_action_name="hold",
    )
    right = consume_probe_outcome(
        probe,
        1,
        ("left", "right"),
        fallback_action_name="hold",
    )
    assert left.action_name == "left"
    assert right.action_name == "right"
    assert not left.used_exact_fallback
    assert not right.used_exact_fallback


def test_deterministic_tie_break_prefers_lower_risk_then_name() -> None:
    plan = plan_active_decision(
        ambiguous(),
        ("left", "right"),
        fallback_action_name="hold",
        probes=(
            task_probe(name="zeta", physical_risk=0.02, cost=0.0),
            task_probe(name="alpha", physical_risk=0.01, cost=0.0),
        ),
        risk_cap=0.1,
    )
    assert plan.selected_probe_name == "alpha"


def test_invalid_probe_contracts_and_duplicate_names_fail_closed() -> None:
    with pytest.raises(ValueError, match="sum to one"):
        CertificateProbeV1(
            name="bad",
            outcomes=(CertificateOutcomeV1(0.4, ambiguous()),),
        )
    with pytest.raises(ValueError, match="physical_risk"):
        task_probe(physical_risk=1.1)
    with pytest.raises(ValueError, match="unique"):
        plan_active_decision(
            ambiguous(),
            ("left", "right"),
            fallback_action_name="hold",
            probes=(task_probe(name="same"), task_probe(name="same")),
        )
    with pytest.raises(ValueError, match="outside"):
        consume_probe_outcome(
            task_probe(),
            2,
            ("left", "right"),
            fallback_action_name="hold",
        )


def test_plan_is_frozen_and_serializes_claim_boundaries() -> None:
    plan = plan_active_decision(
        ambiguous(),
        ("left", "right"),
        fallback_action_name="hold",
        probes=(task_probe(),),
        risk_cap=0.1,
    )
    payload = plan.as_dict()
    assert payload["claim_boundary"] == ACTIVE_DECISION_IDENTIFICATION_CLAIM_BOUNDARY
    assert payload["mode"] == "probe"
    assert payload["probe_reports"][0]["certification_probability"] == pytest.approx(
        1.0
    )
    with pytest.raises(Exception):
        plan.mode = "act"  # type: ignore[misc]


def test_plan_record_rejects_inconsistent_modes() -> None:
    current = plan_active_decision(
        left_certified(),
        ("left", "right"),
        fallback_action_name="hold",
    ).current_intervention
    with pytest.raises(ValueError, match="probe plans"):
        ActiveDecisionPlanV1(
            mode="probe",
            action_name="left",
            selected_probe_name=None,
            fallback_action_name="hold",
            current_intervention=current,
            probe_reports=(),
            score=0.0,
            reason_code="bad",
        )
