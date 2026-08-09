from __future__ import annotations

import json

import pytest

from causal4d.selective_prediction import (
    SessionRiskCoverageRankingContract,
    SessionRiskCoverageRecord,
    build_session_risk_coverage_diagnostic,
    validate_session_risk_coverage_diagnostic,
    write_session_risk_coverage_diagnostic,
)


def _contract(**overrides: object) -> SessionRiskCoverageRankingContract:
    values: dict[str, object] = {
        "ranking_artifact_id": "a" * 64,
        "score_name": "source_predictive_scale",
        "score_semantics": "maximum source-only predictive scale in metres",
        "frozen_before_target_access": True,
        "target_outcomes_used": False,
    }
    values.update(overrides)
    return SessionRiskCoverageRankingContract(**values)  # type: ignore[arg-type]


def _record(
    unit_id: str,
    session_id: str,
    risk: float,
    score: float,
) -> SessionRiskCoverageRecord:
    return SessionRiskCoverageRecord(
        unit_id=unit_id,
        session_id=session_id,
        included=True,
        risk=risk,
        abstention_score=score,
    )


def test_curve_clusters_units_by_session_and_keeps_score_ties() -> None:
    records = [
        _record("s1-u1", "s1", 1.0, 0.1),
        _record("s1-u2", "s1", 3.0, 0.2),
        _record("s2-u1", "s2", 4.0, 0.1),
        _record("s3-u1", "s3", 2.0, 0.2),
    ]

    result = build_session_risk_coverage_diagnostic(
        records,
        _contract(),
        risk_name="track_error",
        risk_unit="m",
    )

    assert [row["session_id"] for row in result["session_summaries"]] == [
        "s1",
        "s2",
        "s3",
    ]
    by_session = {
        row["session_id"]: row for row in result["session_summaries"]
    }
    assert by_session["s1"]["session_risk"] == pytest.approx(2.0)
    assert by_session["s1"]["session_abstention_score"] == pytest.approx(0.2)

    curve = result["curve"]
    assert len(curve) == 2
    assert curve[0]["newly_admitted_session_ids"] == ["s2"]
    assert curve[0]["eligible_session_coverage"] == pytest.approx(1.0 / 3.0)
    assert curve[0]["mean_session_risk"] == pytest.approx(4.0)
    assert curve[1]["newly_admitted_session_ids"] == ["s1", "s3"]
    assert curve[1]["eligible_session_coverage"] == 1.0
    assert curve[1]["mean_session_risk"] == pytest.approx(8.0 / 3.0)
    assert result["scientific_boundary"]["primary_decision_eligible"] is False


def test_curve_is_permutation_invariant() -> None:
    records = [
        _record("b", "s2", 2.0, 0.4),
        _record("a", "s1", 1.0, 0.2),
        _record("c", "s3", 3.0, 0.4),
    ]
    forward = build_session_risk_coverage_diagnostic(
        records,
        _contract(),
        risk_name="loss",
        risk_unit="m",
    )
    reverse = build_session_risk_coverage_diagnostic(
        list(reversed(records)),
        _contract(),
        risk_name="loss",
        risk_unit="m",
    )
    assert reverse == forward


def test_partial_session_is_excluded_as_a_whole() -> None:
    records = [
        _record("s1-u1", "s1", 1.0, 0.1),
        SessionRiskCoverageRecord(
            unit_id="s1-u2",
            session_id="s1",
            included=False,
            risk=None,
            abstention_score=None,
            exclusion_reason="registered timing failure",
        ),
        _record("s2-u1", "s2", 2.0, 0.2),
    ]

    result = build_session_risk_coverage_diagnostic(
        records,
        _contract(),
        risk_name="loss",
        risk_unit="m",
    )

    accounting = result["accounting"]
    assert accounting["registered_session_count"] == 2
    assert accounting["eligible_session_count"] == 1
    assert accounting["excluded_session_count"] == 1
    assert result["curve"][-1]["eligible_session_coverage"] == 1.0
    assert result["curve"][-1]["registered_session_coverage"] == 0.5
    assert result["session_summaries"][0]["session_id"] == "s2"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"frozen_before_target_access": False}, "frozen before target"),
        ({"target_outcomes_used": True}, "may not be used"),
        ({"lower_score_more_confident": False}, "lower-is-more-confident"),
    ],
)
def test_ranking_contract_fails_closed(
    overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _contract(**overrides)


def test_records_reject_target_values_for_excluded_units() -> None:
    with pytest.raises(ValueError, match="must not contain"):
        SessionRiskCoverageRecord(
            unit_id="u1",
            session_id="s1",
            included=False,
            risk=1.0,
            abstention_score=0.2,
            exclusion_reason="technical failure",
        )


def test_duplicate_units_are_rejected() -> None:
    record = _record("u1", "s1", 1.0, 0.2)
    with pytest.raises(ValueError, match="unique"):
        build_session_risk_coverage_diagnostic(
            [record, record],
            _contract(),
            risk_name="loss",
            risk_unit="m",
        )


def test_validation_recomputes_the_content_bound_result() -> None:
    records = [
        _record("u1", "s1", 1.0, 0.2),
        _record("u2", "s2", 2.0, 0.3),
    ]
    contract = _contract()
    result = build_session_risk_coverage_diagnostic(
        records,
        contract,
        risk_name="loss",
        risk_unit="m",
    )
    assert (
        validate_session_risk_coverage_diagnostic(
            result,
            records,
            contract,
            risk_name="loss",
            risk_unit="m",
        )
        == result
    )

    changed = json.loads(json.dumps(result))
    changed["curve"][-1]["mean_session_risk"] = 99.0
    with pytest.raises(ValueError, match="differs from its sources"):
        validate_session_risk_coverage_diagnostic(
            changed,
            records,
            contract,
            risk_name="loss",
            risk_unit="m",
        )


def test_atomic_writer_rejects_a_tampered_content_id(tmp_path) -> None:
    records = [
        _record("u1", "s1", 1.0, 0.2),
        _record("u2", "s2", 2.0, 0.3),
    ]
    result = build_session_risk_coverage_diagnostic(
        records,
        _contract(),
        risk_name="loss",
        risk_unit="m",
    )
    output = tmp_path / "risk-coverage.json"
    write_session_risk_coverage_diagnostic(output, result)
    assert json.loads(output.read_text(encoding="utf-8")) == result

    changed = json.loads(json.dumps(result))
    changed["risk_unit"] = "mm"
    with pytest.raises(ValueError, match="invalid content ID"):
        write_session_risk_coverage_diagnostic(
            tmp_path / "tampered.json",
            changed,
        )
