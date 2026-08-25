from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_source_panel_review as review
import causal4d.preacquisition_source_panel_review_publication as publication
from causal4d.preacquisition_protocol_v5 import single_operator_governance_policy
from causal4d.cli import preacquisition_readiness as readiness_cli


def _operator(
    operator_id: str,
    person_digest: str,
    roles: list[str],
) -> dict:
    return {
        "operator_id": operator_id,
        "person_identity_sha256": person_digest,
        "active": True,
        "roles": sorted(roles),
    }


def _registry(*, same_person: bool = False) -> tuple[dict, dict]:
    reviewer_digest = "1" * 64
    publisher_digest = reviewer_digest if same_person else "2" * 64
    registry = {
        "sealed_at_utc": "2026-08-04T07:00:00Z",
        "operators": [
            _operator("reviewer", reviewer_digest, ["gate_approver"]),
            _operator("publisher", publisher_digest, ["freezer"]),
        ],
    }
    result = {
        "valid": True,
        "artifact_sha256": "3" * 64,
        "error": None,
    }
    return result, registry


def _preflight(source: Path) -> dict:
    data = source.read_bytes()
    return {
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "execution_id": "source-01",
        "session_id": "session-01",
        "source_manifest_relative_path": "staging/source-01.json",
        "source_manifest_sha256": hashlib.sha256(data).hexdigest(),
        "source_manifest_bytes": len(data),
        "source_panel_evidence_sha256_before": "c" * 64,
        "evidence_sha256": "d" * 64,
        "status_sha256": "e" * 64,
    }


def _patch(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    *,
    same_person: bool = False,
) -> dict:
    preflight = _preflight(source)
    registry_result, registry = _registry(same_person=same_person)
    monkeypatch.setattr(
        review,
        "load_registered_preacquisition_chain",
        lambda root: (
            {
                "protocol_id": "protocol",
                "design_sha256": "a" * 64,
            },
            {},
            {},
            {
                "plan_id": "plan",
                "amendment_sha256": "b" * 64,
            },
        ),
    )
    monkeypatch.setattr(
        review,
        "verify_source_panel_manifest_staging",
        lambda *args: deepcopy(preflight),
    )
    monkeypatch.setattr(
        review,
        "_registry",
        lambda *args: (deepcopy(registry_result), deepcopy(registry)),
    )
    return preflight


def _write_source(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ended_at_utc": "2026-08-04T08:01:00Z",
                "target_outcomes_used": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def test_review_receipt_binds_preflight_and_registered_reviewer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    preflight = _patch(monkeypatch, source)

    receipt = review.build_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        reviewed_by="reviewer",
        reviewed_at_utc="2026-08-04T08:02:00Z",
    )

    assert receipt["execution_id"] == "source-01"
    assert receipt["staging_preflight_evidence_sha256"] == preflight["evidence_sha256"]
    assert receipt["source_manifest_sha256"] == preflight["source_manifest_sha256"]
    assert receipt["reviewer_operator_id"] == "reviewer"
    assert receipt["reviewer_roles"] == ["gate_approver"]
    assert receipt["approved_for_exactly_once_publication"] is True
    assert receipt["target_outcomes_used"] is False
    assert receipt["artifact_sha256"] == (
        review.source_panel_review_receipt_sha256(receipt)
    )


def test_review_cannot_predate_execution_completion(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    _patch(monkeypatch, source)

    with pytest.raises(ValueError, match="predates execution completion"):
        review.build_source_panel_review_receipt(
            tmp_path,
            tmp_path,
            source,
            reviewed_by="reviewer",
            reviewed_at_utc="2026-08-04T08:00:00Z",
        )


def test_review_receipt_is_published_once_at_canonical_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    _patch(monkeypatch, source)
    receipt = review.build_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        reviewed_by="reviewer",
        reviewed_at_utc="2026-08-04T08:02:00Z",
    )

    output = review.write_source_panel_review_receipt(tmp_path, receipt)

    assert output == tmp_path / "staging" / "reviews" / "source-01.json"
    assert json.loads(output.read_text(encoding="utf-8")) == receipt
    with pytest.raises(FileExistsError):
        review.write_source_panel_review_receipt(tmp_path, receipt)


def _write_receipt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: Path,
) -> tuple[Path, dict]:
    _patch(monkeypatch, source)
    receipt = review.build_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        reviewed_by="reviewer",
        reviewed_at_utc="2026-08-04T08:02:00Z",
    )
    path = review.write_source_panel_review_receipt(tmp_path, receipt)
    return path, receipt


def test_publication_validation_requires_distinct_people(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    receipt_path, _ = _write_receipt(monkeypatch, tmp_path, source)

    result = review.validate_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        receipt_path,
        published_by="publisher",
    )

    assert result["independent_people"] is True
    assert result["reviewer_operator_id"] == "reviewer"
    assert result["publisher_operator_id"] == "publisher"
    assert result["review_receipt"]["path"] == ("staging/reviews/source-01.json")


def test_self_review_and_publication_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    _patch(monkeypatch, source, same_person=True)
    receipt = review.build_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        reviewed_by="reviewer",
        reviewed_at_utc="2026-08-04T08:02:00Z",
    )
    receipt_path = review.write_source_panel_review_receipt(tmp_path, receipt)

    with pytest.raises(ValueError, match="require distinct people"):
        review.validate_source_panel_review_receipt(
            tmp_path,
            tmp_path,
            source,
            receipt_path,
            published_by="publisher",
        )


def test_stale_receipt_is_rejected_after_source_change(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    receipt_path, _ = _write_receipt(monkeypatch, tmp_path, source)
    changed = _preflight(source)
    changed["evidence_sha256"] = "0" * 64
    monkeypatch.setattr(
        review,
        "verify_source_panel_manifest_staging",
        lambda *args: deepcopy(changed),
    )

    with pytest.raises(ValueError, match="preflight_evidence_sha256 mismatch"):
        review.validate_source_panel_review_receipt(
            tmp_path,
            tmp_path,
            source,
            receipt_path,
            published_by="publisher",
        )


def test_receipt_digest_tamper_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    receipt_path, receipt = _write_receipt(monkeypatch, tmp_path, source)
    receipt["reviewed_at_utc"] = "2026-08-04T08:03:00Z"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="receipt digest mismatch"):
        review.validate_source_panel_review_receipt(
            tmp_path,
            tmp_path,
            source,
            receipt_path,
            published_by="publisher",
        )


def test_reviewed_publisher_validates_before_claim_bearing_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        publication,
        "validate_source_panel_review_receipt",
        lambda *args, **kwargs: (
            calls.append("review")
            or {
                "execution_id": "source-01",
                "session_id": "session-01",
                "review_receipt": {"path": "receipt.json"},
                "reviewer_operator_id": "reviewer",
                "reviewer_person_identity_sha256": "1" * 64,
                "publisher_operator_id": "publisher",
                "publisher_person_identity_sha256": "2" * 64,
                "independent_people": True,
                "preflight_evidence_sha256": "3" * 64,
            }
        ),
    )
    monkeypatch.setattr(
        publication,
        "publish_source_panel_manifest",
        lambda *args, **kwargs: (
            calls.append("publish")
            or {
                "passed": True,
                "execution_id": "source-01",
                "session_id": "session-01",
                "target_outcomes_used": False,
            }
        ),
    )

    result = publication.publish_reviewed_source_panel_manifest(
        tmp_path,
        tmp_path,
        tmp_path / "source.json",
        review_receipt_json=tmp_path / "receipt.json",
        published_by="publisher",
    )

    assert calls == ["review", "publish"]
    assert result["review_required"] is True
    assert result["independent_people"] is True


def test_cli_requires_receipt_and_publisher_for_publication() -> None:
    parser = readiness_cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["source-panel-publish", "repo", "dataset", "source.json"])
    args = parser.parse_args(
        [
            "source-panel-publish",
            "repo",
            "dataset",
            "source.json",
            "--review-receipt",
            "receipt.json",
            "--published-by",
            "publisher",
        ]
    )
    assert args.review_receipt == "receipt.json"
    assert args.published_by == "publisher"


def test_cli_review_and_publication_routes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        readiness_cli,
        "review_source_panel_manifest_staging",
        lambda *args, **kwargs: calls.append("review") or {"passed": True},
    )
    monkeypatch.setattr(
        readiness_cli,
        "publish_reviewed_source_panel_manifest",
        lambda *args, **kwargs: calls.append("publish") or {"passed": True},
    )

    assert (
        readiness_cli.main(
            [
                "source-panel-review-staged",
                "repo",
                "dataset",
                "source.json",
                "--reviewed-by",
                "reviewer",
            ]
        )
        == 0
    )
    assert (
        readiness_cli.main(
            [
                "source-panel-publish",
                "repo",
                "dataset",
                "source.json",
                "--review-receipt",
                "receipt.json",
                "--published-by",
                "publisher",
            ]
        )
        == 0
    )
    assert calls == ["review", "publish"]


def test_v5_allows_disclosed_same_operator_review_and_publication(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    source = tmp_path / "staging" / "source-01.json"
    _write_source(source)
    preflight = _preflight(source)
    operator = _operator(
        "florianpfaff",
        "4" * 64,
        ["freezer", "gate_approver"],
    )
    monkeypatch.setattr(
        review,
        "load_registered_preacquisition_chain",
        lambda root: (
            {"protocol_id": "protocol", "design_sha256": "a" * 64},
            {},
            {},
            {
                "plan_id": "plan",
                "amendment_sha256": "b" * 64,
                "governance": single_operator_governance_policy(),
            },
        ),
    )
    monkeypatch.setattr(
        review,
        "verify_source_panel_manifest_staging",
        lambda *args: deepcopy(preflight),
    )
    monkeypatch.setattr(
        review,
        "_registry",
        lambda *args: (
            {"valid": True, "artifact_sha256": "5" * 64, "error": None},
            {
                "sealed_at_utc": "2026-08-04T07:00:00Z",
                "operators": [operator],
            },
        ),
    )
    receipt = review.build_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        reviewed_by="florianpfaff",
        reviewed_at_utc="2026-08-04T08:02:00Z",
    )
    receipt_path = review.write_source_panel_review_receipt(tmp_path, receipt)

    result = review.validate_source_panel_review_receipt(
        tmp_path,
        tmp_path,
        source,
        receipt_path,
        published_by="florianpfaff",
    )

    assert result["independent_people"] is False
    assert result["independent_preacquisition_attestation_claimed"] is False
    assert result["governance_mode"] == "single_operator_self_attested"
