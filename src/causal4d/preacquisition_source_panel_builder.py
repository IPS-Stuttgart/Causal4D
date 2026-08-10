"""Safe construction of the next source-panel staging manifest."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from causal4d.acquisition_flight_common import (
    _assert_no_symlink_components,
    _assert_ordinary_file_or_missing,
    _fsync_directory,
    _reject_target_outcomes,
    _require,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.preacquisition_readiness_contracts import (
    SOURCE_PANEL_MANIFEST_PATH,
    SOURCE_PANEL_MANIFEST_TEMPLATE_PATH,
    _parse_utc_timestamp,
    _read_json_mapping,
    _sha256_file,
    load_registered_preacquisition_chain,
    source_panel_execution_manifest_template,
)
from causal4d.preacquisition_source_panel_control import build_source_panel_status
from causal4d.preacquisition_source_validation import (
    _validate_source_execution_manifest,
)

SOURCE_PANEL_STAGING_RESULT_SCHEMA_VERSION = 1
SOURCE_PANEL_STAGING_RESULT_ARTIFACT_KIND = "Causal4DSourcePanelStagingResult"


def _resolved_dataset_root(dataset_root: str | Path) -> Path:
    candidate = Path(dataset_root)
    _assert_no_symlink_components(candidate, name="dataset root")
    _require(candidate.is_dir(), "dataset root must exist")
    return candidate.resolve()


def _resolved_artifact(
    dataset_root: Path,
    value: str | Path,
    *,
    execution_id: str,
    index: int,
) -> tuple[str, Path]:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else dataset_root / raw
    name = f"source execution artifact {index}"
    _assert_no_symlink_components(candidate, name=name)
    _require(candidate.is_file(), f"{name} file is missing")
    resolved = candidate.resolve(strict=True)
    _require(
        resolved.is_relative_to(dataset_root),
        f"{name} path escapes the dataset root",
    )
    relative = resolved.relative_to(dataset_root)
    execution_root = (
        Path("preacquisition") / "source_panel" / "executions" / execution_id
    )
    _require(
        relative.is_relative_to(execution_root),
        f"{name} must be below {execution_root.as_posix()}",
    )
    _require(
        relative
        not in {
            execution_root / "manifest.json",
            execution_root / "manifest.template.json",
        },
        f"{name} cannot be a source-panel manifest",
    )
    return relative.as_posix(), resolved


def _artifact_descriptors(
    dataset_root: Path,
    artifacts: Sequence[str | Path],
    *,
    execution_id: str,
) -> list[dict[str, Any]]:
    _require(
        not isinstance(artifacts, (str, bytes))
        and isinstance(artifacts, Sequence)
        and bool(artifacts),
        "source execution artifacts must be a nonempty sequence",
    )
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(artifacts):
        _require(
            isinstance(value, (str, Path)),
            f"source execution artifact {index} path is invalid",
        )
        relative, path = _resolved_artifact(
            dataset_root,
            value,
            execution_id=execution_id,
            index=index,
        )
        _require(
            relative not in seen,
            f"source execution artifacts contain a duplicate path: {relative}",
        )
        digest, byte_count = _sha256_file(path)
        rows.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": byte_count,
            }
        )
        seen.add(relative)
    return sorted(rows, key=lambda row: str(row["path"]))


def _rollback_owned_staging_file(path: Path) -> None:
    path.unlink(missing_ok=True)
    try:
        _fsync_directory(path.parent)
    except OSError:
        pass


def stage_source_panel_manifest(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    started_at_utc: str,
    ended_at_utc: str,
    artifacts: Sequence[str | Path],
) -> dict[str, Any]:
    """Build the exact next completed manifest without verifying or publishing it."""

    protocol, _, _, v4 = load_registered_preacquisition_chain(repository_root)
    root = _resolved_dataset_root(dataset_root)
    status_before = build_source_panel_status(
        repository_root,
        root,
        verify_file_hashes=True,
    )
    _require(status_before["valid"] is True, "source-panel status is invalid")
    _require(status_before["complete"] is False, "source panel is already complete")
    next_execution = status_before.get("next_execution")
    _require(isinstance(next_execution, Mapping), "source panel has no next execution")
    _require(
        next_execution.get("template_present") is True
        and next_execution.get("template_valid") is True,
        "next source-panel manifest template is missing or invalid",
    )

    execution_id = str(next_execution["execution_id"])
    session_id = str(next_execution["session_id"])
    expected_template = source_panel_execution_manifest_template(
        next_execution,
        protocol,
        v4,
    )
    template_relative = SOURCE_PANEL_MANIFEST_TEMPLATE_PATH.format(
        execution_id=execution_id
    )
    template_path = root / template_relative
    _assert_no_symlink_components(template_path, name="source-panel worksheet template")
    _require(template_path.is_file(), "source-panel worksheet template is missing")
    template_digest_before, template_bytes_before = _sha256_file(template_path)
    template = _read_json_mapping(
        template_path,
        name="source-panel worksheet template",
    )
    _require(
        template == expected_template,
        "source-panel worksheet template differs from the registered template",
    )

    started = _parse_utc_timestamp(
        started_at_utc,
        name="source execution started_at_utc",
    )
    ended = _parse_utc_timestamp(
        ended_at_utc,
        name="source execution ended_at_utc",
    )
    _require(ended >= started, "source execution ends before it starts")
    descriptors_before = _artifact_descriptors(
        root,
        artifacts,
        execution_id=execution_id,
    )

    manifest = deepcopy(template)
    manifest.update(
        {
            "status": "complete",
            "included": True,
            "quality_gate_failures": [],
            "started_at_utc": started_at_utc,
            "ended_at_utc": ended_at_utc,
            "artifacts": descriptors_before,
        }
    )
    _reject_target_outcomes(manifest)

    final_relative = SOURCE_PANEL_MANIFEST_PATH.format(execution_id=execution_id)
    final_path = root / final_relative
    _assert_ordinary_file_or_missing(final_path, name="source-panel manifest")
    _require(not final_path.exists(), "source-panel manifest already exists")

    staging_root = root / "staging"
    _assert_no_symlink_components(staging_root, name="source-panel staging directory")
    staging_root.mkdir(parents=True, exist_ok=True)
    _assert_no_symlink_components(staging_root, name="source-panel staging directory")
    staging_relative = (Path("staging") / f"{execution_id}.json").as_posix()
    target = root / staging_relative
    _assert_ordinary_file_or_missing(target, name="source-panel staging manifest")
    _require(not target.exists(), "source-panel staging manifest already exists")

    created = False
    try:
        atomic_write_json(target, manifest, overwrite=False)
        created = True
        staged = _read_json_mapping(target, name="source-panel staging manifest")
        _require(staged == manifest, "staged source-panel manifest changed on write")
        _validate_source_execution_manifest(
            root,
            staging_relative,
            protocol=protocol,
            v4=v4,
            execution_id=execution_id,
            session_id=session_id,
            verify_file_hashes=True,
        )
        staged_digest, staged_bytes = _sha256_file(target)

        template_digest_after, template_bytes_after = _sha256_file(template_path)
        _require(
            (template_digest_after, template_bytes_after)
            == (template_digest_before, template_bytes_before),
            "source-panel worksheet template changed while staging",
        )
        descriptors_after = _artifact_descriptors(
            root,
            [str(row["path"]) for row in descriptors_before],
            execution_id=execution_id,
        )
        _require(
            descriptors_after == descriptors_before,
            "source-panel artifacts changed while staging",
        )

        status_after = build_source_panel_status(
            repository_root,
            root,
            verify_file_hashes=True,
        )
        _require(
            status_after["status_sha256"] == status_before["status_sha256"],
            "source-panel status changed while staging",
        )
        next_after = status_after.get("next_execution")
        _require(
            isinstance(next_after, Mapping)
            and next_after.get("execution_id") == execution_id,
            "next source-panel execution changed while staging",
        )
        _require(
            not final_path.exists(),
            "staging unexpectedly created the final source-panel manifest",
        )
    except BaseException:
        if created:
            _rollback_owned_staging_file(target)
        raise

    repository = str(Path(repository_root).resolve())
    verification_command = [
        "causal4d",
        "protocol",
        "readiness",
        "source-panel-verify-staged",
        repository,
        str(root),
        str(target.resolve(strict=True)),
    ]
    return {
        "schema_version": SOURCE_PANEL_STAGING_RESULT_SCHEMA_VERSION,
        "artifact_kind": SOURCE_PANEL_STAGING_RESULT_ARTIFACT_KIND,
        "protocol_id": protocol["protocol_id"],
        "protocol_design_sha256": protocol["design_sha256"],
        "preacquisition_plan_id": v4["plan_id"],
        "preacquisition_amendment_sha256": v4["amendment_sha256"],
        "repository_root": repository,
        "dataset_root": str(root),
        "execution_id": execution_id,
        "session_id": session_id,
        "source_panel_execution_index": next_execution["source_panel_execution_index"],
        "command_profile_id": next_execution["command_profile_id"],
        "template_relative_path": template_relative,
        "template_sha256": template_digest_before,
        "template_bytes": template_bytes_before,
        "source_json": str(target.resolve(strict=True)),
        "source_manifest_relative_path": staging_relative,
        "source_manifest_sha256": staged_digest,
        "source_manifest_bytes": staged_bytes,
        "started_at_utc": started_at_utc,
        "ended_at_utc": ended_at_utc,
        "artifact_count": len(descriptors_before),
        "artifacts": descriptors_before,
        "final_manifest_path": final_relative,
        "final_manifest_present": False,
        "source_panel_evidence_sha256_before": status_before["evidence_sha256"],
        "source_panel_status_sha256_before": status_before["status_sha256"],
        "source_panel_status_sha256_after": status_after["status_sha256"],
        "source_panel_status_stable": True,
        "staged_verification_command_argv": verification_command,
        "staged_verification_command_text": shlex.join(verification_command),
        "ready_for_staged_verification": True,
        "published": False,
        "claim_bearing_evidence_mutated": False,
        "changes_registered_method": False,
        "target_outcomes_used": False,
        "valid": True,
        "complete": True,
        "passed": True,
    }


__all__ = [
    "SOURCE_PANEL_STAGING_RESULT_ARTIFACT_KIND",
    "SOURCE_PANEL_STAGING_RESULT_SCHEMA_VERSION",
    "stage_source_panel_manifest",
]
