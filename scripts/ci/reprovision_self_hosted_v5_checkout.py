#!/usr/bin/env python3
"""Atomically reprovision the registered Causal4D v5 checkout before freeze.

This utility replaces only the registered software checkout. It never writes to
registered acquisition data and refuses to run after method freeze or physical
collection has begun. The previous clean checkout is retained as a rollback
snapshot next to the deployed path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = 1
REPORT_ARTIFACT_KIND = "Causal4DV5CheckoutReprovisionReport"
EXPECTED_NEXT_ACTION = "complete_object_registration"
CANONICAL_REPOSITORY_URL = "https://github.com/IPS-Stuttgart/Causal4D.git"
REQUIRED_SOURCE_PATHS = (
    "configs/causal4d/sloth_preacquisition_v5.json",
    "configs/causal4d/sloth_multi_action_v1.json",
    "scripts/ci/probe_self_hosted_acquisition.py",
)
FORBIDDEN_DATASET_PATHS = (
    "method_freeze.json",
    "method_freeze_validation.json",
    "registered-analysis.json",
)
FORBIDDEN_DATASET_PATTERNS = (
    "executions/*/manifest.json",
    "sessions/*/session.json",
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

DecisionBuilder = Callable[[Path, Path], Mapping[str, Any]]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _contains_symlink_component(path: Path) -> bool:
    candidate = path.absolute()
    return any(component.is_symlink() for component in (candidate, *candidate.parents))


def _require_ordinary_directory(path: Path, *, name: str) -> Path:
    absolute = path.absolute()
    _require(
        not _contains_symlink_component(absolute),
        f"{name} contains a symlink component",
    )
    _require(absolute.is_dir(), f"{name} must be an ordinary directory")
    _require(not absolute.is_symlink(), f"{name} must not be a symlink")
    return absolute.resolve()


def _require_nonexistent(path: Path, *, name: str) -> None:
    _require(
        not os.path.lexists(path),
        f"{name} already exists: {path}",
    )


def _run(
    arguments: list[str],
    *,
    cwd: Path | None = None,
) -> str:
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        command = " ".join(arguments)
        detail = (completed.stderr or completed.stdout).strip()
        raise ValueError(
            f"command failed ({completed.returncode}): {command}: {detail}"
        )
    return completed.stdout.strip()


def _git(repository: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(repository), *arguments])


def _git_head(repository: Path) -> str:
    value = _git(repository, "rev-parse", "HEAD")
    _require(_COMMIT_RE.fullmatch(value) is not None, "git HEAD is not a full commit")
    return value


def _git_tree(repository: Path) -> str:
    value = _git(repository, "rev-parse", "HEAD^{tree}")
    _require(_COMMIT_RE.fullmatch(value) is not None, "git tree is not a full SHA")
    return value


def _require_clean_repository(repository: Path, *, name: str) -> None:
    _require((repository / ".git").is_dir(), f"{name} is not a Git checkout")
    status = _git(repository, "status", "--porcelain=v1", "--untracked-files=all")
    _require(not status, f"{name} is not clean")


def _sha256_file(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
            byte_count += len(block)
    return digest.hexdigest(), byte_count


def _dataset_snapshot(root: Path) -> dict[str, Any]:
    descriptors: list[dict[str, Any]] = []
    for current_root, directory_names, filenames in os.walk(root, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            directory = current / name
            _require(
                not directory.is_symlink(),
                f"dataset contains a symlink directory: {directory.relative_to(root)}",
            )
        for name in filenames:
            path = current / name
            relative = path.relative_to(root).as_posix()
            _require(
                not path.is_symlink(),
                f"dataset contains a symlink file: {relative}",
            )
            mode = path.stat().st_mode
            _require(
                stat.S_ISREG(mode),
                f"dataset contains a non-regular file: {relative}",
            )
            sha256, byte_count = _sha256_file(path)
            descriptors.append(
                {
                    "path": relative,
                    "sha256": sha256,
                    "bytes": byte_count,
                }
            )
    descriptors.sort(key=lambda item: str(item["path"]))
    encoded = json.dumps(
        descriptors,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "file_count": len(descriptors),
        "tree_sha256": hashlib.sha256(encoded).hexdigest(),
        "members": descriptors,
    }


def _require_pre_freeze_dataset(dataset: Path) -> None:
    for relative in FORBIDDEN_DATASET_PATHS:
        _require(
            not os.path.lexists(dataset / relative),
            f"checkout reprovision is too late; governed evidence exists: {relative}",
        )
    for pattern in FORBIDDEN_DATASET_PATTERNS:
        _require(
            not any(dataset.glob(pattern)),
            f"checkout reprovision is too late; governed evidence matches: {pattern}",
        )


def _default_decision_builder(
    repository: Path,
    dataset: Path,
) -> Mapping[str, Any]:
    from causal4d.preacquisition_operator_flow import (
        build_preacquisition_operator_next_action,
    )

    return build_preacquisition_operator_next_action(
        repository,
        dataset,
        verify_file_hashes=True,
    )


def _decision_summary(decision: Mapping[str, Any]) -> dict[str, Any]:
    action = decision.get("action")
    _require(isinstance(action, Mapping), "next-action decision has no action object")
    action_mapping = dict(action)
    summary = {
        "protocol_id": decision.get("protocol_id"),
        "valid": decision.get("valid") is True,
        "ready": decision.get("ready") is True,
        "verify_file_hashes": decision.get("verify_file_hashes") is True,
        "evidence_sha256": decision.get("evidence_sha256"),
        "status_sha256": decision.get("status_sha256"),
        "action_id": action_mapping.get("action_id"),
        "category": action_mapping.get("category"),
        "operator_role": action_mapping.get("operator_role"),
        "automatable": action_mapping.get("automatable") is True,
        "physical_acquisition_required": (
            action_mapping.get("physical_acquisition_required") is True
        ),
        "target_outcomes_permitted": (
            action_mapping.get("target_outcomes_permitted") is True
        ),
        "changes_registered_method": (
            action_mapping.get("changes_registered_method") is True
        ),
    }
    _require(summary["valid"], "registered next-action decision is invalid")
    _require(
        summary["verify_file_hashes"],
        "registered next-action decision did not verify file hashes",
    )
    _require(
        summary["action_id"] == EXPECTED_NEXT_ACTION,
        "checkout reprovision is allowed only at complete_object_registration",
    )
    _require(
        not summary["automatable"],
        "complete_object_registration unexpectedly became automatable",
    )
    _require(
        not summary["physical_acquisition_required"],
        "checkout reprovision cannot precede a physical action",
    )
    _require(
        not summary["target_outcomes_permitted"],
        "checkout reprovision cannot permit target outcomes",
    )
    _require(
        not summary["changes_registered_method"],
        "checkout reprovision cannot change the registered method",
    )
    return summary


def _require_source_paths(repository: Path) -> None:
    for relative in REQUIRED_SOURCE_PATHS:
        path = repository / relative
        _require(
            path.is_file() and not path.is_symlink(),
            f"source checkout lacks required ordinary file: {relative}",
        )


def _clone_exact_source(
    source: Path,
    staging: Path,
    *,
    expected_commit: str,
) -> None:
    _run(
        [
            "git",
            "clone",
            "--no-local",
            "--no-hardlinks",
            "--quiet",
            str(source),
            str(staging),
        ]
    )
    _git(staging, "checkout", "--detach", "--quiet", expected_commit)
    _git(staging, "remote", "set-url", "origin", CANONICAL_REPOSITORY_URL)
    _require_clean_repository(staging, name="staged checkout")
    _require(
        _git_head(staging) == expected_commit,
        "staged checkout commit differs from the reviewed source",
    )
    _require_source_paths(staging)


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json_write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    _require_nonexistent(temporary, name="report staging file")
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()


def _canonical_sha256(payload: Mapping[str, Any], *, field: str) -> str:
    value = dict(payload)
    value.pop(field, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def reprovision_checkout(
    source_repository: Path,
    target_repository: Path,
    dataset_root: Path,
    *,
    expected_source_commit: str,
    decision_builder: DecisionBuilder = _default_decision_builder,
) -> dict[str, Any]:
    """Replace a clean stale checkout with an exact clean reviewed checkout."""

    _require(
        _COMMIT_RE.fullmatch(expected_source_commit) is not None,
        "expected_source_commit must be 40 lowercase hexadecimal characters",
    )
    source = _require_ordinary_directory(
        source_repository,
        name="reviewed source checkout",
    )
    target = _require_ordinary_directory(
        target_repository,
        name="registered target checkout",
    )
    dataset = _require_ordinary_directory(dataset_root, name="registered v5 dataset")
    _require(source != target, "source and target checkouts must differ")

    _require_clean_repository(source, name="reviewed source checkout")
    _require_clean_repository(target, name="registered target checkout")
    source_head = _git_head(source)
    _require(
        source_head == expected_source_commit,
        "reviewed source checkout does not match expected_source_commit",
    )
    source_tree = _git_tree(source)
    target_head = _git_head(target)
    target_tree = _git_tree(target)
    _require_source_paths(source)
    _require_pre_freeze_dataset(dataset)

    dataset_before = _dataset_snapshot(dataset)
    source_decision = _decision_summary(decision_builder(source, dataset))

    parent = _require_ordinary_directory(target.parent, name="checkout parent")
    staging = parent / f".{target.name}.reprovision-{expected_source_commit[:12]}.tmp"
    backup = parent / (
        f"{target.name}.before-{target_head[:12]}-for-{expected_source_commit[:12]}"
    )
    failed = parent / f".{target.name}.failed-{expected_source_commit[:12]}"
    _require_nonexistent(staging, name="checkout staging path")
    _require_nonexistent(backup, name="checkout backup path")
    _require_nonexistent(failed, name="failed-checkout quarantine path")

    moved_old = False
    installed_new = False
    try:
        _clone_exact_source(
            source,
            staging,
            expected_commit=expected_source_commit,
        )
        staged_decision = _decision_summary(decision_builder(staging, dataset))
        _require(
            staged_decision["action_id"] == source_decision["action_id"],
            "staged checkout changes the registered next action",
        )
        if source_decision["evidence_sha256"] is not None:
            _require(
                staged_decision["evidence_sha256"]
                == source_decision["evidence_sha256"],
                "staged checkout changes the portable readiness evidence identity",
            )

        os.rename(target, backup)
        moved_old = True
        os.rename(staging, target)
        installed_new = True
        _fsync_directory(parent)

        _require_clean_repository(target, name="deployed target checkout")
        _require(
            _git_head(target) == expected_source_commit,
            "deployed target checkout commit mismatch",
        )
        _require(
            _git_tree(target) == source_tree,
            "deployed target checkout tree mismatch",
        )
        _require_source_paths(target)
        deployed_decision = _decision_summary(decision_builder(target, dataset))
        _require(
            deployed_decision["action_id"] == source_decision["action_id"],
            "deployed checkout changes the registered next action",
        )
        if source_decision["evidence_sha256"] is not None:
            _require(
                deployed_decision["evidence_sha256"]
                == source_decision["evidence_sha256"],
                "deployed checkout changes the portable readiness evidence identity",
            )

        dataset_after = _dataset_snapshot(dataset)
        _require(
            dataset_after["tree_sha256"] == dataset_before["tree_sha256"],
            "registered dataset changed during checkout reprovision",
        )
        _require(
            dataset_after["members"] == dataset_before["members"],
            "registered dataset members changed during checkout reprovision",
        )

        report: dict[str, Any] = {
            "schema_version": REPORT_SCHEMA_VERSION,
            "artifact_kind": REPORT_ARTIFACT_KIND,
            "reviewed_source_commit": source_head,
            "reviewed_source_tree": source_tree,
            "previous_target_commit": target_head,
            "previous_target_tree": target_tree,
            "deployed_target_commit": _git_head(target),
            "deployed_target_tree": _git_tree(target),
            "target_repository": str(target),
            "retained_backup_repository": str(backup),
            "registered_dataset": str(dataset),
            "source_next_action": source_decision,
            "deployed_next_action": deployed_decision,
            "dataset_before": {
                "file_count": dataset_before["file_count"],
                "tree_sha256": dataset_before["tree_sha256"],
            },
            "dataset_after": {
                "file_count": dataset_after["file_count"],
                "tree_sha256": dataset_after["tree_sha256"],
            },
            "dataset_modified": False,
            "target_outcomes_used": False,
            "device_nodes_opened": False,
            "physical_command_sent": False,
            "registered_method_changed": False,
            "physical_evidence_increment": 0,
            "claim_boundary": (
                "Pre-freeze software-checkout reprovision only. The registered "
                "dataset is byte-preserved, no target outcome is read, and no "
                "physical evidence is created."
            ),
        }
        report["report_sha256"] = _canonical_sha256(report, field="report_sha256")
        return report
    except BaseException:
        if installed_new and os.path.lexists(target):
            os.rename(target, failed)
            installed_new = False
        if moved_old and not os.path.lexists(target):
            os.rename(backup, target)
            moved_old = False
        _fsync_directory(parent)
        raise
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repository", type=Path, required=True)
    parser.add_argument("--target-repository", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    report = reprovision_checkout(
        arguments.source_repository,
        arguments.target_repository,
        arguments.dataset_root,
        expected_source_commit=arguments.expected_source_commit,
    )
    _atomic_json_write(arguments.output, report)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
