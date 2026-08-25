"""Governance-bound review receipts for staged source-panel publication."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _parse_utc,
    _reject_target_outcomes,
    _require,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.operator_registry import (
    ROLE_GATE_APPROVER,
    ROLE_INDEPENDENT_VERIFIER,
    load_registered_operator_registry,
    require_distinct_operator_people,
    require_registry_precedes_event,
    resolve_operator,
)
from causal4d.preacquisition_protocol_v5 import governance_allows_single_operator
from causal4d.preacquisition_readiness_contracts import (
    _canonical_sha256,
    _read_json_mapping,
    _sha256_file,
    load_registered_preacquisition_chain,
)
from causal4d.preacquisition_source_panel_staging import (
    verify_source_panel_manifest_staging,
)

SOURCE_PANEL_REVIEW_RECEIPT_SCHEMA_VERSION = 1
SOURCE_PANEL_REVIEW_RECEIPT_ARTIFACT_KIND = "Causal4DSourcePanelStagingReviewReceipt"
_REVIEWER_ROLES = frozenset({ROLE_GATE_APPROVER, ROLE_INDEPENDENT_VERIFIER})
_RECEIPT_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_plan_id",
        "preacquisition_amendment_sha256",
        "execution_id",
        "session_id",
        "source_manifest_relative_path",
        "source_manifest_sha256",
        "source_manifest_bytes",
        "source_execution_ended_at_utc",
        "staging_preflight_evidence_sha256",
        "staging_preflight_status_sha256",
        "source_panel_evidence_sha256_before",
        "operator_registry_artifact_sha256",
        "reviewer_operator_id",
        "reviewer_person_identity_sha256",
        "reviewer_roles",
        "reviewed_at_utc",
        "approved_for_exactly_once_publication",
        "changes_registered_method",
        "target_outcomes_used",
        "artifact_sha256",
    }
)


def source_panel_review_receipt_sha256(values: Mapping[str, Any]) -> str:
    """Return the canonical digest sealing one review receipt."""

    return _canonical_sha256(values, omitted_field="artifact_sha256")


def _registry(
    repository_root: str | Path,
    dataset_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result, registry = load_registered_operator_registry(
        repository_root,
        dataset_root,
    )
    _require(
        result.get("valid") is True and isinstance(registry, Mapping),
        str(result.get("error") or "operator registry is invalid"),
    )
    return result, dict(registry)


def _source_end_time(source_json: Path, expected_sha256: str) -> str:
    digest_before, _ = _sha256_file(source_json)
    _require(
        digest_before == expected_sha256,
        "staged source manifest changed after preflight",
    )
    payload = _read_json_mapping(source_json, name="source-panel staging manifest")
    _reject_target_outcomes(payload)
    ended_at = payload.get("ended_at_utc")
    ended = _parse_utc(ended_at, name="source execution ended_at_utc")
    digest_after, _ = _sha256_file(source_json)
    _require(
        digest_after == digest_before,
        "staged source manifest changed during review",
    )
    return ended.isoformat().replace("+00:00", "Z")


def build_source_panel_review_receipt(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
    *,
    reviewed_by: str,
    reviewed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Re-run preflight and bind one registered human review."""

    protocol, _, _, v4 = load_registered_preacquisition_chain(repository_root)
    preflight = verify_source_panel_manifest_staging(
        repository_root,
        dataset_root,
        source_json,
    )
    _, _, _, preacquisition = load_registered_preacquisition_chain(repository_root)
    single_operator = governance_allows_single_operator(preacquisition)
    registry_result, registry = _registry(repository_root, dataset_root)
    reviewer = resolve_operator(
        registry,
        reviewed_by,
        any_role=_REVIEWER_ROLES,
        name="source-panel staging reviewer",
    )
    timestamp = reviewed_at_utc or datetime.now(timezone.utc).isoformat()
    reviewed_at = _parse_utc(timestamp, name="source-panel review timestamp")
    require_registry_precedes_event(
        registry,
        timestamp,
        event_name="source-panel staging review",
    )

    source = Path(source_json).resolve(strict=True)
    ended_at_text = _source_end_time(
        source,
        str(preflight["source_manifest_sha256"]),
    )
    ended_at = _parse_utc(
        ended_at_text,
        name="source execution ended_at_utc",
    )
    _require(
        reviewed_at >= ended_at,
        "source-panel review predates execution completion",
    )

    receipt: dict[str, Any] = {
        "schema_version": SOURCE_PANEL_REVIEW_RECEIPT_SCHEMA_VERSION,
        "artifact_kind": SOURCE_PANEL_REVIEW_RECEIPT_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "execution_id": preflight["execution_id"],
        "session_id": preflight["session_id"],
        "source_manifest_relative_path": preflight["source_manifest_relative_path"],
        "source_manifest_sha256": preflight["source_manifest_sha256"],
        "source_manifest_bytes": preflight["source_manifest_bytes"],
        "source_execution_ended_at_utc": ended_at_text,
        "staging_preflight_evidence_sha256": preflight["evidence_sha256"],
        "staging_preflight_status_sha256": preflight["status_sha256"],
        "source_panel_evidence_sha256_before": preflight[
            "source_panel_evidence_sha256_before"
        ],
        "operator_registry_artifact_sha256": registry_result["artifact_sha256"],
        "reviewer_operator_id": reviewer["operator_id"],
        "reviewer_person_identity_sha256": reviewer["person_identity_sha256"],
        "reviewer_roles": sorted(reviewer["roles"]),
        "reviewed_at_utc": reviewed_at.isoformat().replace("+00:00", "Z"),
        "approved_for_exactly_once_publication": True,
        "changes_registered_method": False,
        "target_outcomes_used": False,
        "artifact_sha256": None,
    }
    receipt["artifact_sha256"] = source_panel_review_receipt_sha256(receipt)
    return receipt


def source_panel_review_receipt_path(
    dataset_root: str | Path,
    execution_id: str,
) -> Path:
    """Return the canonical once-only receipt path."""

    return Path(dataset_root) / "staging" / "reviews" / f"{execution_id}.json"


def write_source_panel_review_receipt(
    dataset_root: str | Path,
    receipt: Mapping[str, Any],
) -> Path:
    """Atomically publish one non-overwriting review receipt."""

    execution_id = receipt.get("execution_id")
    _require(
        isinstance(execution_id, str) and bool(execution_id),
        "review receipt execution id is missing",
    )
    target = source_panel_review_receipt_path(dataset_root, execution_id)
    atomic_write_json(target, dict(receipt), overwrite=False)
    return target


def review_source_panel_manifest_staging(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
    *,
    reviewed_by: str,
    reviewed_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and publish one canonical review receipt."""

    receipt = build_source_panel_review_receipt(
        repository_root,
        dataset_root,
        source_json,
        reviewed_by=reviewed_by,
        reviewed_at_utc=reviewed_at_utc,
    )
    output = write_source_panel_review_receipt(dataset_root, receipt)
    digest, byte_count = _sha256_file(output)
    return {
        **receipt,
        "receipt_file": {
            "path": output.resolve()
            .relative_to(Path(dataset_root).resolve())
            .as_posix(),
            "sha256": digest,
            "bytes": byte_count,
        },
        "output": str(output.resolve()),
        "passed": True,
    }


def _validate_receipt_fields(receipt: Mapping[str, Any]) -> None:
    _require(
        set(receipt) == _RECEIPT_FIELDS,
        "source-panel review receipt fields differ from schema version 1",
    )
    _require(
        receipt.get("schema_version") == SOURCE_PANEL_REVIEW_RECEIPT_SCHEMA_VERSION,
        "unsupported source-panel review receipt schema",
    )
    _require(
        receipt.get("artifact_kind") == SOURCE_PANEL_REVIEW_RECEIPT_ARTIFACT_KIND,
        "unexpected source-panel review receipt artifact kind",
    )
    _require(
        receipt.get("approved_for_exactly_once_publication") is True,
        "source-panel review receipt does not approve publication",
    )
    _require(
        receipt.get("changes_registered_method") is False,
        "source-panel review receipt permits a registered-method change",
    )
    _require(
        receipt.get("target_outcomes_used") is False,
        "target outcomes entered source-panel review",
    )
    _require(
        receipt.get("artifact_sha256") == source_panel_review_receipt_sha256(receipt),
        "source-panel review receipt digest mismatch",
    )


def validate_source_panel_review_receipt(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
    receipt_json: str | Path | None,
    *,
    published_by: str | None,
) -> dict[str, Any]:
    """Require a current review and the registered publication policy."""

    _require(receipt_json is not None, "source-panel review receipt is required")
    _require(
        isinstance(published_by, str) and bool(published_by),
        "source-panel publisher id is required",
    )
    preflight = verify_source_panel_manifest_staging(
        repository_root,
        dataset_root,
        source_json,
    )
    root = Path(dataset_root).resolve()
    expected_path = source_panel_review_receipt_path(
        root,
        str(preflight["execution_id"]),
    ).resolve()
    receipt_path = Path(receipt_json)
    _assert_no_symlink_components(receipt_path, name="source-panel review receipt")
    _require(receipt_path.is_file(), "source-panel review receipt is missing")
    receipt_path = receipt_path.resolve(strict=True)
    _require(
        receipt_path == expected_path,
        "source-panel review receipt is not at the canonical execution path",
    )
    digest_before, bytes_before = _sha256_file(receipt_path)
    receipt = _read_json_mapping(receipt_path, name="source-panel review receipt")
    _reject_target_outcomes(receipt)
    _validate_receipt_fields(receipt)

    registry_result, registry = _registry(repository_root, dataset_root)
    reviewer = resolve_operator(
        registry,
        receipt.get("reviewer_operator_id"),
        any_role=_REVIEWER_ROLES,
        name="source-panel staging reviewer",
    )
    publisher = resolve_operator(
        registry,
        published_by,
        name="source-panel publisher",
    )
    if not single_operator:
        require_distinct_operator_people(
            reviewer,
            publisher,
            relationship=(
                "source-panel staging review and publication require distinct people"
            ),
        )
    require_registry_precedes_event(
        registry,
        receipt.get("reviewed_at_utc"),
        event_name="source-panel staging review",
    )

    expected = {
        "protocol_id": preflight["protocol_id"],
        "protocol_design_sha256": preflight["protocol_design_sha256"],
        "preacquisition_plan_id": preflight["preacquisition_plan_id"],
        "preacquisition_amendment_sha256": preflight["preacquisition_amendment_sha256"],
        "execution_id": preflight["execution_id"],
        "session_id": preflight["session_id"],
        "source_manifest_relative_path": preflight["source_manifest_relative_path"],
        "source_manifest_sha256": preflight["source_manifest_sha256"],
        "source_manifest_bytes": preflight["source_manifest_bytes"],
        "staging_preflight_evidence_sha256": preflight["evidence_sha256"],
        "staging_preflight_status_sha256": preflight["status_sha256"],
        "source_panel_evidence_sha256_before": preflight[
            "source_panel_evidence_sha256_before"
        ],
        "operator_registry_artifact_sha256": registry_result["artifact_sha256"],
        "reviewer_person_identity_sha256": reviewer["person_identity_sha256"],
        "reviewer_roles": sorted(reviewer["roles"]),
    }
    for field, value in expected.items():
        _require(
            receipt.get(field) == value,
            f"source-panel review receipt {field} mismatch",
        )
    ended_at = _parse_utc(
        receipt.get("source_execution_ended_at_utc"),
        name="source execution ended_at_utc",
    )
    reviewed_at = _parse_utc(
        receipt.get("reviewed_at_utc"),
        name="source-panel review timestamp",
    )
    _require(
        reviewed_at >= ended_at,
        "source-panel review predates execution completion",
    )
    digest_after, bytes_after = _sha256_file(receipt_path)
    _require(
        (digest_after, bytes_after) == (digest_before, bytes_before),
        "source-panel review receipt changed during validation",
    )
    return {
        "passed": True,
        "execution_id": preflight["execution_id"],
        "session_id": preflight["session_id"],
        "review_receipt": {
            "path": receipt_path.relative_to(root).as_posix(),
            "sha256": digest_after,
            "bytes": bytes_after,
            "artifact_sha256": receipt["artifact_sha256"],
        },
        "reviewer_operator_id": reviewer["operator_id"],
        "reviewer_person_identity_sha256": reviewer["person_identity_sha256"],
        "publisher_operator_id": publisher["operator_id"],
        "publisher_person_identity_sha256": publisher["person_identity_sha256"],
        "independent_people": (
            reviewer["person_identity_sha256"]
            != publisher["person_identity_sha256"]
        ),
        "governance_mode": preacquisition["governance"]["mode"],
        "independent_preacquisition_attestation_claimed": not single_operator,
        "preflight_evidence_sha256": preflight["evidence_sha256"],
        "target_outcomes_used": False,
    }


__all__ = [
    "SOURCE_PANEL_REVIEW_RECEIPT_ARTIFACT_KIND",
    "SOURCE_PANEL_REVIEW_RECEIPT_SCHEMA_VERSION",
    "build_source_panel_review_receipt",
    "review_source_panel_manifest_staging",
    "source_panel_review_receipt_path",
    "source_panel_review_receipt_sha256",
    "validate_source_panel_review_receipt",
    "write_source_panel_review_receipt",
]
