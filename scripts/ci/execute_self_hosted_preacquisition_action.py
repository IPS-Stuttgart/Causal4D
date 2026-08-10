#!/usr/bin/env python3
"""Execute one allowlisted, nonphysical registered pre-acquisition action."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from causal4d.operator_registry import (
    OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND,
    OPERATOR_REGISTRY_TEMPLATE_PATH,
    scaffold_operator_registry,
)
from causal4d.preacquisition_operator_flow import (
    build_preacquisition_operator_next_action,
)


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_KIND = "Causal4DSelfHostedAutomatablePreacquisitionAction"
ALLOWED_ACTION_ID = "scaffold_operator_registry"
EXPECTED_AFTER_ACTION_ID = "seal_operator_registry"
_MAXIMUM_SNAPSHOT_FILES = 50_000
_MAXIMUM_SNAPSHOT_BYTES = 4 * 1024**3


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _require_ordinary_directory(path: Path, *, name: str) -> Path:
    _require(
        not _contains_symlink_component(path),
        f"{name} contains a symlink component",
    )
    _require(path.is_dir(), f"{name} must be an ordinary directory")
    return path.resolve()


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _snapshot_regular_files(root: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        _require(not path.is_symlink(), f"dataset contains a symlink: {path}")
        if path.is_dir():
            continue
        _require(path.is_file(), f"dataset contains a non-regular entry: {path}")
        relative = path.relative_to(root).as_posix()
        sha256, byte_count = _sha256_file(path)
        snapshot[relative] = {"sha256": sha256, "bytes": byte_count}
        total_bytes += byte_count
        _require(
            len(snapshot) <= _MAXIMUM_SNAPSHOT_FILES,
            "dataset snapshot exceeds the registered file-count guard",
        )
        _require(
            total_bytes <= _MAXIMUM_SNAPSHOT_BYTES,
            "dataset snapshot exceeds the registered byte-count guard",
        )
    return snapshot


def _snapshot_sha256(snapshot: Mapping[str, Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        snapshot,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _snapshot_delta(
    before: Mapping[str, Mapping[str, Any]],
    after: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[str]]:
    before_paths = set(before)
    after_paths = set(after)
    return {
        "added": sorted(after_paths - before_paths),
        "removed": sorted(before_paths - after_paths),
        "modified": sorted(
            path
            for path in before_paths & after_paths
            if dict(before[path]) != dict(after[path])
        ),
    }


def _load_json_mapping(path: Path, *, name: str) -> dict[str, Any]:
    _require(
        not _contains_symlink_component(path),
        f"{name} contains a symlink component",
    )
    _require(path.is_file(), f"{name} is missing")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, Mapping), f"{name} must contain a JSON object")
    return dict(value)


def _expected_command(repository_root: Path, dataset_root: Path) -> list[str]:
    return [
        "causal4d",
        "protocol",
        "readiness",
        "scaffold-operator-registry",
        str(repository_root),
        str(dataset_root),
    ]


def _registered_action(
    decision: Mapping[str, Any],
    *,
    expected_action_id: str,
    repository_root: Path,
    dataset_root: Path,
) -> dict[str, Any]:
    _require(
        decision.get("valid") is True,
        "registered next-action decision is invalid",
    )
    _require(
        decision.get("target_outcomes_used") is False,
        "registered next-action decision admits target outcomes",
    )
    action_value = decision.get("action")
    _require(isinstance(action_value, Mapping), "registered next action is missing")
    action = dict(cast(Mapping[str, Any], action_value))
    _require(
        action.get("action_id") == expected_action_id,
        "registered next action differs from the requested allowlisted action",
    )
    _require(
        action.get("category") == "scaffold",
        "registered action is not a scaffold",
    )
    _require(action.get("automatable") is True, "registered action is not automatable")
    _require(
        action.get("physical_acquisition_required") is False,
        "registered action requires physical acquisition",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "registered action permits target outcomes",
    )
    _require(
        action.get("changes_registered_method") is False,
        "registered action changes the registered method",
    )
    _require(
        action.get("command_argv") == _expected_command(repository_root, dataset_root),
        "registered action command differs from the exact allowlisted command",
    )
    return action


def _action_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    action_value = decision.get("action")
    _require(isinstance(action_value, Mapping), "registered next action is missing")
    action = dict(cast(Mapping[str, Any], action_value))
    execution = action.get("registered_execution")
    execution_summary = None
    if isinstance(execution, Mapping):
        execution_summary = {
            "execution_id": execution.get("execution_id"),
            "session_id": execution.get("session_id"),
            "command_profile_id": execution.get("command_profile_id"),
        }
    return {
        "action_id": action.get("action_id"),
        "category": action.get("category"),
        "operator_role": action.get("operator_role"),
        "automatable": action.get("automatable") is True,
        "physical_acquisition_required": (
            action.get("physical_acquisition_required") is True
        ),
        "target_outcomes_permitted": action.get("target_outcomes_permitted") is True,
        "blocking_item_count": len(action.get("blocking_items", [])),
        "registered_execution": execution_summary,
        "evidence_sha256": decision.get("evidence_sha256"),
        "status_sha256": decision.get("status_sha256"),
    }


def _validate_created_template(
    path: Path,
    *,
    decision_before: Mapping[str, Any],
) -> dict[str, Any]:
    template = _load_json_mapping(path, name="operator registry template")
    _require(
        template.get("artifact_kind") == OPERATOR_REGISTRY_TEMPLATE_ARTIFACT_KIND,
        "operator registry template artifact kind is invalid",
    )
    _require(
        template.get("status") == "template",
        "operator registry is not a template",
    )
    _require(
        template.get("target_outcomes_used") is False,
        "target outcomes entered the operator registry template",
    )
    _require(template.get("operators") == [], "operator registry template is not empty")
    _require(
        template.get("sealed_at_utc") is None
        and template.get("sealed_by_operator_id") is None
        and template.get("artifact_sha256") is None,
        "operator registry template contains seal metadata",
    )
    for field in (
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_plan_id",
        "preacquisition_amendment_sha256",
    ):
        _require(
            template.get(field) == decision_before.get(field),
            f"operator registry template {field} mismatch",
        )
    sha256, byte_count = _sha256_file(path)
    return {
        "relative_path": OPERATOR_REGISTRY_TEMPLATE_PATH,
        "sha256": sha256,
        "bytes": byte_count,
        "artifact_kind": template["artifact_kind"],
        "status": template["status"],
        "operator_count": 0,
        "target_outcomes_used": False,
    }


def _validate_after_action(decision: Mapping[str, Any]) -> dict[str, Any]:
    _require(decision.get("valid") is True, "post-action decision is invalid")
    _require(
        decision.get("target_outcomes_used") is False,
        "post-action decision admits target outcomes",
    )
    action_value = decision.get("action")
    _require(isinstance(action_value, Mapping), "post-action decision has no action")
    action = dict(cast(Mapping[str, Any], action_value))
    _require(
        action.get("action_id") == EXPECTED_AFTER_ACTION_ID,
        "operator registry scaffold did not reach the registered sealing boundary",
    )
    _require(
        action.get("automatable") is False,
        "post-scaffold operator registry action must require a human role",
    )
    _require(
        action.get("physical_acquisition_required") is False,
        "post-scaffold operator registry action unexpectedly requires acquisition",
    )
    _require(
        action.get("target_outcomes_permitted") is False,
        "post-scaffold operator registry action permits target outcomes",
    )
    return _action_summary(decision)


def execute_allowlisted_action(
    *,
    repository_root: Path,
    dataset_root: Path,
    expected_action_id: str,
) -> dict[str, Any]:
    _require(
        expected_action_id == ALLOWED_ACTION_ID,
        "requested action is not in the self-hosted allowlist",
    )
    repository = _require_ordinary_directory(
        repository_root,
        name="repository root",
    )
    dataset = _require_ordinary_directory(dataset_root, name="dataset root")
    target = dataset / OPERATOR_REGISTRY_TEMPLATE_PATH
    _require(
        not _contains_symlink_component(target),
        "operator registry template path contains a symlink component",
    )
    _require(
        not target.exists(),
        "operator registry template already exists; refusing a non-exact action",
    )

    snapshot_before = _snapshot_regular_files(dataset)
    decision_before = build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )
    action_before = _registered_action(
        decision_before,
        expected_action_id=expected_action_id,
        repository_root=repository,
        dataset_root=dataset,
    )

    scaffold_result = scaffold_operator_registry(repository, dataset)
    _require(scaffold_result.get("passed") is True, "operator registry scaffold failed")
    _require(
        scaffold_result.get("created") is True
        and scaffold_result.get("existing") is False,
        "operator registry scaffold did not create exactly one new template",
    )

    snapshot_after = _snapshot_regular_files(dataset)
    delta = _snapshot_delta(snapshot_before, snapshot_after)
    _require(
        delta
        == {
            "added": [OPERATOR_REGISTRY_TEMPLATE_PATH],
            "removed": [],
            "modified": [],
        },
        "allowlisted action changed files outside the registered template path",
    )
    template_summary = _validate_created_template(
        target,
        decision_before=decision_before,
    )

    decision_after = build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )
    action_after = _validate_after_action(decision_after)

    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "artifact_kind": REPORT_ARTIFACT_KIND,
        "reviewed_main_commit": os.environ.get("GITHUB_SHA"),
        "repository_root": str(repository),
        "dataset_root": str(dataset),
        "executed_action": {
            "action_id": action_before["action_id"],
            "category": action_before["category"],
            "operator_role": action_before["operator_role"],
            "command_argv": list(action_before["command_argv"]),
            "automatable": True,
            "physical_acquisition_required": False,
            "target_outcomes_permitted": False,
            "changes_registered_method": False,
        },
        "scaffold_result": dict(scaffold_result),
        "created_template": template_summary,
        "dataset_delta": delta,
        "snapshot_before_sha256": _snapshot_sha256(snapshot_before),
        "snapshot_after_sha256": _snapshot_sha256(snapshot_after),
        "next_action": action_after,
        "target_outcomes_used": False,
        "device_nodes_opened": False,
        "physical_command_sent": False,
        "dataset_modified": True,
        "physical_evidence_increment": 0,
        "claim_boundary": (
            "Exactly one empty operator-registry template was created by the "
            "registered nonphysical scaffold action. No identity was supplied, "
            "no gate was sealed or approved, no device was opened, no robot or "
            "sensor command was sent, no target outcome was read, and physical "
            "evidence was not incremented."
        ),
    }
    encoded = json.dumps(
        report,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    report["report_sha256"] = hashlib.sha256(encoded).hexdigest()
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--expected-action-id",
        choices=(ALLOWED_ACTION_ID,),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = execute_allowlisted_action(
        repository_root=arguments.repository_root,
        dataset_root=arguments.dataset_root,
        expected_action_id=arguments.expected_action_id,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    arguments.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
