from __future__ import annotations

import json
from pathlib import Path

import pytest

from reviewed_agent_branch_cleanup_test_support import (
    BRANCH,
    OTHER_SHA,
    REPOSITORY,
    CleanupError,
    _decision_payload,
    _entry,
    _write_review_files,
    load_cleanup,
)


def test_load_cleanup_binds_exact_manifest_and_report(tmp_path: Path) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)

    loaded, _ = load_cleanup(tmp_path, manifest_path, manifest_sha)

    assert loaded.manifest.repository == REPOSITORY
    assert loaded.manifest.entries[0].name == BRANCH
    assert loaded.manifest_sha256 == manifest_sha

    report = tmp_path / "ci" / "branch-cleanup-manifests" / "report.json"
    report.write_text(report.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(CleanupError, match="source report SHA-256"):
        load_cleanup(tmp_path, manifest_path, manifest_sha)


def test_manifest_rejects_sensitive_substrings_and_unreviewed_entries(
    tmp_path: Path,
) -> None:
    sensitive = _decision_payload(branch="agent/acquisitionfreeze-archive")
    manifest_path, manifest_sha = _write_review_files(
        tmp_path,
        decisions=[sensitive],
    )
    with pytest.raises(CleanupError, match="protected archival tokens"):
        load_cleanup(tmp_path, manifest_path, manifest_sha)

    safe_root = tmp_path / "safe"
    safe = _decision_payload()
    entry = _entry(safe)
    entry["artifact_reference_reviewed"] = False
    manifest_path, manifest_sha = _write_review_files(
        safe_root,
        decisions=[safe],
        entries=[entry],
    )
    with pytest.raises(CleanupError, match="artifact_reference_reviewed"):
        load_cleanup(safe_root, manifest_path, manifest_sha)


def test_unselected_sensitive_report_candidate_does_not_block_safe_entry(
    tmp_path: Path,
) -> None:
    safe = _decision_payload()
    sensitive = _decision_payload(
        branch="agent/evidence/archive",
        sha=OTHER_SHA,
    )
    manifest_path, manifest_sha = _write_review_files(
        tmp_path,
        decisions=[safe, sensitive],
        entries=[_entry(safe)],
    )

    loaded, _ = load_cleanup(tmp_path, manifest_path, manifest_sha)

    assert [entry.name for entry in loaded.manifest.entries] == [BRANCH]


def test_manifest_and_allowlist_reject_symlinks(tmp_path: Path) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    manifest = tmp_path / manifest_path
    target = manifest.with_name("manifest-target.json")
    manifest.replace(target)
    manifest.symlink_to(target.name)

    with pytest.raises(CleanupError, match="must not traverse a symlink"):
        load_cleanup(tmp_path, manifest_path, manifest_sha)

    root = tmp_path / "allowlist-case"
    manifest_path, manifest_sha = _write_review_files(root)
    allowlist = root / ".github" / "stale-agent-branch-allowlist.json"
    allowlist_target = allowlist.with_name("allowlist-target.json")
    allowllist.replace(allowlist_target)
    allowlist.symlink_to(allowlist_target.name)
    with pytest.raises(CleanupError, match="must not traverse a symlink"):
        load_cleanup(root, manifest_path, manifest_sha)


def test_report_counts_are_recomputed_fail_closed(tmp_path: Path) -> None:
    manifest_path, manifest_sha = _write_review_files(
        tmp_path,
        reason_counts={"tip_reachable_from_default": 0},
    )

    with pytest.raises(CleanupError, match="reason_counts is inconsistent"):
        load_cleanup(tmp_path, manifest_path, manifest_sha)


def test_invalid_allowlist_is_wrapped_as_cleanup_error(tmp_path: Path) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    allowlist = tmp_path / ".github" / "stale-agent-branch-allowlist.json"
    allowlist.write_text(
        json.dumps({"schema_version": 99, "branches": [], "prefixes": []}),
        encoding="utf-8",
    )

    with pytest.raises(CleanupError, match="invalid branch allowlist"):
        load_cleanup(tmp_path, manifest_path, manifest_sha)
