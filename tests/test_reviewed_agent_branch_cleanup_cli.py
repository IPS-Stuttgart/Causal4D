from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from reviewed_agent_branch_cleanup_test_support import (
    BRANCH,
    MODULE,
    SHA,
    CleanupExecutionError,
    _write_review_files,
    main,
)


def test_validate_only_cli_requires_no_token(tmp_path: Path, monkeypatch: Any) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    output = tmp_path / "validation.json"
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    status = main(
        [
            "--repository-root",
            str(tmp_path),
            "--manifest",
            manifest_path,
            "--manifest-sha256",
            manifest_sha,
            "--validate-only",
            "--output-json",
            str(output),
        ]
    )

    assert status == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["valid"] is True
    assert payload["mutation_performed"] is False


def test_cli_writes_failure_receipt_for_invalid_manifest(tmp_path: Path) -> None:
    output = tmp_path / "failure.json"

    status = main(
        [
            "--repository-root",
            str(tmp_path),
            "--manifest",
            "ci/branch-cleanup-manifests/missing.json",
            "--manifest-sha256",
            "0" * 64,
            "--validate-only",
            "--output-json",
            str(output),
        ]
    )

    assert status == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["complete"] is False
    assert payload["mutation_performed"] is False
    assert payload["artifact_kind"].endswith("CleanupFailure")


def test_cli_persists_partial_execution_receipt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    output = tmp_path / "execution.json"
    partial = {
        "schema_version": 1,
        "artifact_kind": "Causal4DReviewedAgentBranchCleanupExecution",
        "complete": False,
        "mutation_performed": True,
        "deleted_count": 1,
        "deleted_branches": [{"name": BRANCH, "sha": SHA}],
        "failure": {"branch": "agent/next", "phase": "delete_ref"},
    }

    def failed(*args: Any, **kwargs: Any) -> Any:
        raise CleanupExecutionError("partial failure", partial)

    monkeypatch.setattr(MODULE, "execute_cleanup", failed)
    monkeypatch.setattr(MODULE, "CleanupGitHubApi", lambda token: object())
    monkeypatch.setenv("GITHUB_TOKEN", "token")

    status = main(
        [
            "--repository-root",
            str(tmp_path),
            "--manifest",
            manifest_path,
            "--manifest-sha256",
            manifest_sha,
            "--approval-phrase",
            "delete-reviewed-agent-branches",
            "--output-json",
            str(output),
        ]
    )

    assert status == 1
    assert json.loads(output.read_text(encoding="utf-8")) == partial
