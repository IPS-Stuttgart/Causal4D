from __future__ import annotations

from dataclasses import dataclass
import sys
import types
from typing import Any

import numpy as np
import pytest

from causal4d.decision_identifiable_intervention import (
    DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY,
    QUERY_DECISION_CERTIFICATE_SEMANTICS,
    DecisionIdentifiableInterventionV1,
    consume_query_decision_certificate,
    decision_identifiable_intervention_from_quotient,
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
    pairwise: np.ndarray,
    *,
    tolerance: float,
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


def test_unique_robust_action_is_authorized_even_with_broad_tolerance() -> None:
    result = consume_query_decision_certificate(
        certificate(
            np.array([[0.0, -0.2], [0.5, 0.0]]),
            tolerance=1.0,
        ),
        ("retain", "update"),
        fallback_action_name="caller-fallback",
    )
    assert result.action_name == "retain"
    assert result.certified_action_name == "retain"
    assert not result.used_exact_fallback
    assert result.certificate_level == "robustly-optimal"
    assert result.selected_worst_case_regret == pytest.approx(0.0)
    assert result.tolerance_admissible_action_names == ("retain", "update")
    assert result.robustly_optimal_action_names == ("retain",)
    assert result.reason_code == "unique-robustly-optimal-action"


def test_unique_epsilon_admissible_action_is_authorized() -> None:
    result = consume_query_decision_certificate(
        certificate(
            np.array([[0.0, 0.1], [0.4, 0.0]]),
            tolerance=0.2,
        ),
        ("left", "right"),
        fallback_action_name="stop",
    )
    assert result.action_name == "left"
    assert result.certificate_level == "tolerance-admissible"
    assert result.selected_worst_case_regret == pytest.approx(0.1)
    assert result.robustly_optimal_action_names == ()
    assert result.tolerance_admissible_action_names == ("left",)


def test_multiple_admissible_actions_return_exact_fallback() -> None:
    result = consume_query_decision_certificate(
        certificate(
            np.array([[0.0, 0.1], [0.1, 0.0]]),
            tolerance=0.2,
        ),
        ("left", "right"),
        fallback_action_name="hold",
    )
    assert result.action_name == "hold"
    assert result.certified_action_name is None
    assert result.used_exact_fallback
    assert result.certificate_level == "uncertified"
    assert result.selected_worst_case_regret is None
    assert result.reason_code == "decision-not-uniquely-identified"


def test_no_admissible_action_returns_exact_fallback() -> None:
    result = consume_query_decision_certificate(
        certificate(
            np.array([[0.0, 0.1], [0.1, 0.0]]),
            tolerance=0.05,
        ),
        ("left", "right"),
        fallback_action_name="hold",
    )
    assert result.action_name == "hold"
    assert result.used_exact_fallback
    assert result.reason_code == "no-tolerance-admissible-action"


def test_serialized_mapping_certificate_is_supported() -> None:
    fixture = certificate(
        np.array([[0.0, -0.5], [2.75, 0.0]]),
        tolerance=0.0,
    )
    payload: dict[str, Any] = {
        "summary": fixture.summary(),
        "pairwise_worst_case_loss_gap": fixture.pairwise_worst_case_loss_gap.tolist(),
        "worst_case_regret": fixture.worst_case_regret.tolist(),
        "minimax_action_index": fixture.minimax_action_index,
        "minimax_worst_case_regret": fixture.minimax_worst_case_regret,
        "regret_tolerance": fixture.regret_tolerance,
        "tolerance_admissible_action_mask": (
            fixture.tolerance_admissible_action_mask.tolist()
        ),
        "robustly_optimal_action_mask": (fixture.robustly_optimal_action_mask.tolist()),
    }
    result = consume_query_decision_certificate(
        payload,
        ("retain", "update"),
        fallback_action_name="physical-fallback",
    )
    assert result.action_name == "retain"
    assert result.as_dict()["claim_boundary"] == (
        DECISION_IDENTIFIABLE_INTERVENTION_CLAIM_BOUNDARY
    )


def test_inconsistent_certificate_fails_closed() -> None:
    fixture = certificate(
        np.array([[0.0, -0.5], [2.75, 0.0]]),
        tolerance=0.0,
    )
    fixture.worst_case_regret = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="inconsistent with pairwise"):
        consume_query_decision_certificate(
            fixture,
            ("retain", "update"),
            fallback_action_name="fallback",
        )


def test_summary_semantics_and_masks_are_verified() -> None:
    fixture = certificate(
        np.array([[0.0, -0.5], [2.75, 0.0]]),
        tolerance=0.0,
    )
    bad_summary = fixture.summary()
    bad_summary["semantics"] = "other"
    payload = {
        "summary": bad_summary,
        "pairwise_worst_case_loss_gap": fixture.pairwise_worst_case_loss_gap,
        "worst_case_regret": fixture.worst_case_regret,
        "minimax_action_index": fixture.minimax_action_index,
        "minimax_worst_case_regret": fixture.minimax_worst_case_regret,
        "regret_tolerance": fixture.regret_tolerance,
        "tolerance_admissible_action_mask": (fixture.tolerance_admissible_action_mask),
        "robustly_optimal_action_mask": fixture.robustly_optimal_action_mask,
    }
    with pytest.raises(ValueError, match="semantics"):
        consume_query_decision_certificate(
            payload,
            ("retain", "update"),
            fallback_action_name="fallback",
        )

    fixture.robustly_optimal_action_mask = np.array([False, False])
    with pytest.raises(ValueError, match="robust-optimal mask"):
        consume_query_decision_certificate(
            fixture,
            ("retain", "update"),
            fallback_action_name="fallback",
        )


def test_action_roster_and_fallback_are_fail_closed() -> None:
    fixture = certificate(
        np.array([[0.0, -0.5], [2.75, 0.0]]),
        tolerance=0.0,
    )
    with pytest.raises(ValueError, match="unique"):
        consume_query_decision_certificate(
            fixture,
            ("same", "same"),
            fallback_action_name="fallback",
        )
    with pytest.raises(ValueError, match="outside"):
        consume_query_decision_certificate(
            fixture,
            ("retain", "update"),
            fallback_action_name="retain",
        )


def test_optional_bayesian_phystwin_constructor_is_consumed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = certificate(
        np.array([[0.0, -0.5], [2.75, 0.0]]),
        tolerance=0.0,
    )
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def fake_certificate(*args: object, **kwargs: object) -> CertificateFixture:
        calls.append((args, kwargs))
        return fixture

    parent = types.ModuleType("bayesian_phystwin")
    parent.__path__ = []  # type: ignore[attr-defined]
    child = types.ModuleType("bayesian_phystwin.query_decision_certificate_v1")
    child.query_decision_certificate = fake_certificate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "bayesian_phystwin", parent)
    monkeypatch.setitem(
        sys.modules,
        "bayesian_phystwin.query_decision_certificate_v1",
        child,
    )

    result = decision_identifiable_intervention_from_quotient(
        np.array([0.5, 0.5]),
        np.array([1.0]),
        np.array([0, 0]),
        np.array([[0.0, 1.0], [0.0, 2.0]]),
        ("retain", "update"),
        fallback_action_name="fallback",
        regret_tolerance=0.0,
    )
    assert result.action_name == "retain"
    assert len(calls) == 1
    assert calls[0][1] == {"regret_tolerance": 0.0}


def test_decision_record_is_frozen() -> None:
    result = DecisionIdentifiableInterventionV1(
        action_name="fallback",
        certified_action_name=None,
        fallback_action_name="fallback",
        used_exact_fallback=True,
        certificate_level="uncertified",
        selected_worst_case_regret=None,
        minimax_action_name="candidate",
        minimax_worst_case_regret=1.0,
        regret_tolerance=0.0,
        tolerance_admissible_action_names=(),
        robustly_optimal_action_names=(),
        reason_code="no-tolerance-admissible-action",
    )
    with pytest.raises(Exception):
        result.action_name = "changed"  # type: ignore[misc]
