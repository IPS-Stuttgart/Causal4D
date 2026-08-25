"""Review-gated publication for staged physical source evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _assert_ordinary_file_or_missing,
    _reject_target_outcomes,
    _require,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    _read_json_mapping,
    _sha256_file,
    load_registered_preacquisition_chain,
    source_panel_execution_manifest_template,
)
from causal4d.preacquisition_source_panel_control import (
    _resolved_dataset_root,
    _write_manifest_payload,
    build_source_panel_status,
)
from causal4d.preacquisition_source_panel_review import (
    validate_source_panel_review_receipt,
)
from causal4d.preacquisition_source_panel_staging import (
    verify_source_panel_manifest_staging,
)
from causal4d.preacquisition_source_validation import (
    _validate_source_execution_manifest,
)


def publish_source_panel_manifest(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
    *,
    review: Mapping[str, Any],
) -> dict[str, Any]:
    """Atomically publish only the source bytes bound by a validated review."""

    preflight = verify_source_panel_manifest_staging(
        repository_root,
        dataset_root,
        source_json,
    )
    _require(
        preflight["evidence_sha256"] == review["preflight_evidence_sha256"],
        "source-panel preflight changed after receipt validation",
    )
    _require(
        preflight["execution_id"] == review["execution_id"],
        "reviewed source execution changed before publication",
    )
    _require(
        preflight["session_id"] == review["session_id"],
        "reviewed source session changed before publication",
    )

    protocol, _, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = _resolved_dataset_root(dataset_root)
    status_before = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(status_before["valid"] is True, "source-panel status is invalid")
    _require(status_before["complete"] is False, "source panel is already complete")
    _require(
        status_before["status_sha256"]
        == preflight["source_panel_status_sha256_before"],
        "source-panel status changed after the reviewed preflight",
    )
    next_execution = status_before.get("next_execution")
    _require(isinstance(next_execution, Mapping), "source panel has no next execution")
    _require(
        next_execution.get("template_present") is True
        and next_execution.get("template_valid") is True,
        "next source-panel manifest template is missing or invalid",
    )

    source = Path(source_json)
    _assert_no_symlink_components(source, name="source-panel publication source")
    _require(source.is_file(), "source-panel publication source is missing")
    source = source.resolve(strict=True)
    digest_before, bytes_before = _sha256_file(source)
    _require(
        digest_before == preflight["source_manifest_sha256"],
        "staged source bytes differ from the reviewed preflight",
    )
    _require(
        bytes_before == preflight["source_manifest_bytes"],
        "staged source byte count differs from the reviewed preflight",
    )
    payload = _read_json_mapping(source, name="source-panel publication source")
    _reject_target_outcomes(payload)
    digest_after, bytes_after = _sha256_file(source)
    _require(
        (digest_after, bytes_after) == (digest_before, bytes_before),
        "source-panel publication source changed while being read",
    )

    expected_template = source_panel_execution_manifest_template(
        next_execution,
        protocol,
        v4,
    )
    _require(
        set(payload) == set(expected_template),
        "source-panel manifest fields differ from schema version 1",
    )
    execution_id = str(next_execution["execution_id"])
    session_id = str(next_execution["session_id"])
    _require(
        payload.get("execution_id") == execution_id,
        "source-panel manifest is not the next registered execution",
    )
    _require(
        payload.get("session_id") == session_id,
        "source-panel manifest names the wrong session",
    )

    manifest_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
    final_path = root / manifest_relative
    _assert_ordinary_file_or_missing(final_path, name="source-panel manifest")
    _require(not final_path.exists(), "source-panel manifest already exists")
    final_path.parent.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(
        final_path.parent,
        name="source-panel manifest parent",
    )

    def validate(temporary: Path) -> None:
        relative = temporary.resolve().relative_to(root).as_posix()
        _validate_source_execution_manifest(
            root,
            relative,
            protocol=protocol,
            v4=v4,
            execution_id=execution_id,
            session_id=session_id,
            verify_file_hashes=True,
        )

    atomic_write_binary(
        final_path,
        lambda handle: _write_manifest_payload(handle, payload),
        overwrite=False,
        validate=validate,
    )
    digest, byte_count = _sha256_file(final_path)
    status_after = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(
        status_after["valid"] is True,
        "published source-panel status is invalid",
    )
    _require(
        execution_id in status_after["completed_execution_ids"],
        "published source-panel manifest was not admitted",
    )
    return {
        "passed": True,
        "execution_id": execution_id,
        "session_id": session_id,
        "published_manifest": {
            "path": manifest_relative,
            "sha256": digest,
            "bytes": byte_count,
        },
        "source_panel_status": status_after,
        "staging_preflight_evidence_sha256": preflight["evidence_sha256"],
        "reviewed_source_manifest_sha256": digest_before,
        "reviewed_source_manifest_bytes": bytes_before,
        "target_outcomes_used": False,
    }


def publish_reviewed_source_panel_manifest(
    repository_root: str | Path,
    dataset_root: str | Path,
    source_json: str | Path,
    *,
    review_receipt_json: str | Path,
    published_by: str,
) -> dict[str, Any]:
    """Require a current governance-bound receipt, then publish exactly once."""

    review = validate_source_panel_review_receipt(
        repository_root,
        dataset_root,
        source_json,
        review_receipt_json,
        published_by=published_by,
    )
    published = publish_source_panel_manifest(
        repository_root,
        dataset_root,
        source_json,
        review=review,
    )
    if review["execution_id"] != published["execution_id"]:
        raise RuntimeError("reviewed and published source executions differ")
    if review["session_id"] != published["session_id"]:
        raise RuntimeError("reviewed and published source sessions differ")
    return {
        **published,
        "review_receipt": review["review_receipt"],
        "reviewer_operator_id": review["reviewer_operator_id"],
        "reviewer_person_identity_sha256": review["reviewer_person_identity_sha256"],
        "publisher_operator_id": review["publisher_operator_id"],
        "publisher_person_identity_sha256": review["publisher_person_identity_sha256"],
        "independent_people": review["independent_people"],
        "review_required": True,
        "target_outcomes_used": False,
    }


__all__ = ["publish_reviewed_source_panel_manifest"]
