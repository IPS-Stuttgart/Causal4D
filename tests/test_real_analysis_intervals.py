from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
    bind_repository_interval_amendment,
    expected_real_analysis_interval_amendment,
    expected_real_analysis_interval_evidence,
    load_real_analysis_interval_amendment,
    load_real_analysis_interval_evidence,
    validate_real_analysis_interval_amendment,
)
from causal4d.real_analysis_intervals import (
    REAL_EFFECT_BOOTSTRAP_REPLICATES,
    REAL_EFFECT_BOOTSTRAP_SEED,
    bootstrap_t_mean_interval,
    percentile_bootstrap_mean_interval,
    registered_positive_effect_interval_decision,
    student_t_mean_interval,
)


ROOT = Path(__file__).resolve().parents[1]


def _interval(
    *,
    method: str,
    lower: float,
    sample_count: int = 12,
    point_estimate: float = 0.2,
) -> dict[str, object]:
    return {
        "estimable": True,
        "method": method,
        "confidence_level": 0.95,
        "sample_count": sample_count,
        "point_estimate": point_estimate,
        "lower": lower,
        "degenerate_sample": False,
    }


def test_registered_interval_implementations_are_deterministic() -> None:
    values = np.linspace(-0.25, 0.75, 18).tolist()

    first = bootstrap_t_mean_interval(values)
    second = bootstrap_t_mean_interval(values)
    historical = percentile_bootstrap_mean_interval(values)
    robustness = student_t_mean_interval(values)

    assert first == second
    assert first["method"] == "target_session_bootstrap_t"
    assert first["replicates"] == REAL_EFFECT_BOOTSTRAP_REPLICATES
    assert first["seed"] == REAL_EFFECT_BOOTSTRAP_SEED
    assert first["lower"] <= first["point_estimate"] <= first["upper"]
    assert historical["method"] == "target_session_percentile_bootstrap"
    assert robustness["method"] == "student_t_mean"


def test_degenerate_session_panel_cannot_authorize_a_positive_claim() -> None:
    values = [1.25] * 12

    primary = bootstrap_t_mean_interval(values)
    robustness = student_t_mean_interval(values)
    historical = percentile_bootstrap_mean_interval(values)

    for interval in (primary, robustness):
        assert interval["estimable"] is False
        assert interval["point_estimate"] == pytest.approx(1.25)
        assert interval["lower"] is None
        assert interval["upper"] is None
        assert interval["degenerate_sample"] is True

    assert historical["estimable"] is True
    assert historical["point_estimate"] == pytest.approx(1.25)
    assert historical["lower"] == pytest.approx(1.25)
    assert historical["upper"] == pytest.approx(1.25)
    assert historical["degenerate_sample"] is True

    decision = registered_positive_effect_interval_decision(primary, robustness)
    assert decision["registered_interval_inputs_match"] is True
    assert decision["degenerate_session_panel"] is True
    assert decision["degenerate_session_panel_blocks_positive_claim"] is True
    assert decision["positive_claim_interval_gate_passed"] is False


def test_student_t_can_veto_but_never_rescue_primary_interval() -> None:
    primary_pass = _interval(
        method="target_session_bootstrap_t",
        lower=0.1,
    )
    robustness_fail = _interval(
        method="student_t_mean",
        lower=-0.01,
    )
    decision = registered_positive_effect_interval_decision(
        primary_pass,
        robustness_fail,
    )
    assert decision["primary_interval_excludes_nonpositive_effect"] is True
    assert decision["required_robustness_interval_excludes_nonpositive_effect"] is False
    assert decision["positive_claim_interval_gate_passed"] is False

    primary_fail = {**primary_pass, "lower": 0.0}
    robustness_pass = {**robustness_fail, "lower": 0.1}
    decision = registered_positive_effect_interval_decision(
        primary_fail,
        robustness_pass,
    )
    assert decision["positive_claim_interval_gate_passed"] is False
    assert decision["robustness_interval_may_rescue_primary_failure"] is False


@pytest.mark.parametrize(
    ("primary_update", "robustness_update", "message"),
    [
        (
            {"method": "target_session_percentile_bootstrap"},
            {},
            "primary interval method",
        ),
        ({}, {"method": "bootstrap_t_mean"}, "robustness interval method"),
        ({"sample_count": 11}, {}, "same session sample"),
        ({"confidence_level": 0.9}, {}, "registered value"),
        ({"point_estimate": 0.3}, {}, "same session mean"),
    ],
)
def test_positive_claim_gate_rejects_mismatched_interval_inputs(
    primary_update: dict[str, object],
    robustness_update: dict[str, object],
    message: str,
) -> None:
    primary = {
        **_interval(method="target_session_bootstrap_t", lower=0.1),
        **primary_update,
    }
    robustness = {
        **_interval(method="student_t_mean", lower=0.1),
        **robustness_update,
    }

    with pytest.raises(ValueError, match=message):
        registered_positive_effect_interval_decision(primary, robustness)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("replicates", True, "replicates must be a positive integer"),
        ("replicates", 0, "replicates must be a positive integer"),
        ("seed", True, "seed must be an integer"),
    ],
)
def test_bootstrap_t_rejects_coercive_settings(
    keyword: str,
    value: object,
    message: str,
) -> None:
    arguments = {keyword: value}
    with pytest.raises(ValueError, match=message):
        bootstrap_t_mean_interval([0.0, 1.0], **arguments)  # type: ignore[arg-type]


def test_checked_in_interval_amendment_is_exact_and_content_bound() -> None:
    expected = expected_real_analysis_interval_amendment()
    path = ROOT / REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH
    loaded, snapshot = load_real_analysis_interval_amendment(path)
    evidence_path = ROOT / REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH
    evidence, evidence_snapshot = load_real_analysis_interval_evidence(evidence_path)
    binding = bind_repository_interval_amendment(ROOT)

    assert loaded == expected
    assert evidence == expected_real_analysis_interval_evidence()
    assert binding["contract"] == expected
    assert binding["amendment_id"] == expected["amendment_id"]
    assert binding["sha256"] == snapshot.sha256
    assert binding["bytes"] == snapshot.byte_count
    assert binding["operating_characteristic_evidence"] == {
        "repository_path": REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
        "result_sha256": evidence["result_sha256"],
        "sha256": evidence_snapshot.sha256,
        "bytes": evidence_snapshot.byte_count,
    }
    assert (
        expected["information_boundary"]["physical_execution_count_at_registration"]
        == 0
    )


def test_interval_amendment_rejects_consistently_readdressed_tampering() -> None:
    tampered = copy.deepcopy(expected_real_analysis_interval_amendment())
    tampered["primary_interval"]["method"] = "target_session_percentile_bootstrap"
    with pytest.raises(ValueError, match="interval amendment changed"):
        validate_real_analysis_interval_amendment(tampered)
