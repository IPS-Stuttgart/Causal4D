#!/usr/bin/env python3
"""Execute a SHA-bound reviewed cleanup manifest for stale ``agent/*`` refs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from reviewed_agent_branch_cleanup_contract import (
    CleanupError,
    CleanupManifest,
    LoadedCleanup,
    _utc,
    _utc_text,
    load_cleanup,
)
from stale_agent_branches import (
    BranchAllowlist,
    BranchDecision,
    GitHubApi,
    GitHubApiError,
    _pull_request_numbers,
    inspect_agent_branches,
)


_APPROVAL = "delete-reviewed-agent-branches"
_SHA1 = re.compile(r"[0-9a-f]{40}")


class CleanupExecutionError(CleanupError):
    """Raised after an execution attempt with a durable partial receipt."""

    def __init__(self, message: str, receipt: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.receipt = dict(receipt)


class CleanupGitHubApi(GitHubApi):
    """The report client plus narrowly bounded ref deletion and confirmation."""

    def __init__(self, token: str) -> None:
        super().__init__(token)
        self._cleanup_token = token

    def _request(self, method: str, path: str) -> bytes:
        request = Request(
            f"https://api.github.com{path}",
            method=method,
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._cleanup_token}",
                "User-Agent": "causal4d-reviewed-agent-branch-cleanup",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                return response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise GitHubApiError(
                f"GitHub API {method} {path} failed with {error.code}: {body}"
            ) from error
        except URLError as error:
            raise GitHubApiError(
                f"GitHub API {method} {path} could not be reached: {error}"
            ) from error

    def delete_branch_ref(self, repository: str, branch: str) -> None:
        path = f"/repos/{repository}/git/refs/heads/{quote(branch, safe='')}"
        self._request("DELETE", path)

    def branch_exists(self, repository: str, branch: str) -> bool:
        """Return false only for an explicit 404; malformed responses fail closed."""

        path = f"/repos/{repository}/branches/{quote(branch, safe='')}"
        request = Request(
            f"https://api.github.com{path}",
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._cleanup_token}",
                "User-Agent": "causal4d-reviewed-agent-branch-cleanup",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                raw = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            if error.code == 404:
                return False
            raise GitHubApiError(
                f"GitHub API GET {path} failed with {error.code}: {body}"
            ) from error
        except URLError as error:
            raise GitHubApiError(
                f"GitHub API GET {path} could not be reached: {error}"
            ) from error
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as error:
            raise GitHubApiError(
                f"GitHub API GET {path} returned invalid JSON"
            ) from error
        if not isinstance(payload, Mapping) or payload.get("name") != branch:
            raise GitHubApiError(
                f"GitHub API GET {path} did not return the exact branch"
            )
        commit = payload.get("commit")
        sha = commit.get("sha") if isinstance(commit, Mapping) else None
        if type(sha) is not str or _SHA1.fullmatch(sha) is None:
            raise GitHubApiError(
                f"GitHub API GET {path} returned an invalid branch tip"
            )
        return True


def _live_candidates(
    api: GitHubApi,
    manifest: CleanupManifest,
    allowlist: BranchAllowlist,
    now: datetime,
) -> tuple[BranchDecision, ...]:
    if now.tzinfo is None or now.utcoffset() is None:
        raise CleanupError("cleanup time must be timezone-aware")
    if now < _utc(manifest.reviewed_at_utc, "reviewed_at_utc"):
        raise CleanupError("cleanup cannot predate review")
    if now > _utc(manifest.expires_at_utc, "expires_at_utc"):
        raise CleanupError("cleanup manifest has expired")
    default, decisions = inspect_agent_branches(
        api,
        repository=manifest.repository,
        prefix="agent/",
        allowlist=allowlist,
        minimum_age_days=manifest.minimum_age_days,
        now=now,
    )
    if default != manifest.default_branch:
        raise CleanupError("repository default branch changed after review")
    by_name = {decision.inspection.name: decision for decision in decisions}
    selected: list[BranchDecision] = []
    for entry in manifest.entries:
        decision = by_name.get(entry.name)
        if decision is None or not decision.cleanup_candidate:
            raise CleanupError(f"live branch is no longer eligible: {entry.name}")
        inspection = decision.inspection
        if (
            inspection.sha != entry.expected_sha
            or decision.reason != entry.eligibility_reason
            or inspection.exact_tip_merged_pull_requests != entry.merged_pull_requests
        ):
            raise CleanupError(f"live branch lineage changed: {entry.name}")
        selected.append(decision)
    return tuple(selected)


def _immediate_recheck(
    api: GitHubApi,
    manifest: CleanupManifest,
    allowlist: BranchAllowlist,
    decision: BranchDecision,
) -> None:
    inspection = decision.inspection
    if allowlist.matches(inspection.name):
        raise CleanupError(f"branch became allowlisted: {inspection.name}")
    repository = api.get_json(f"/repos/{manifest.repository}")
    if not isinstance(repository, Mapping):
        raise GitHubApiError("repository response must be an object")
    if repository.get("default_branch") != manifest.default_branch:
        raise CleanupError("default branch changed immediately before deletion")
    branch = api.get_json(
        f"/repos/{manifest.repository}/branches/{quote(inspection.name, safe='')}"
    )
    if not isinstance(branch, Mapping) or branch.get("name") != inspection.name:
        raise CleanupError(f"branch no longer resolves exactly: {inspection.name}")
    commit = branch.get("commit")
    sha = commit.get("sha") if isinstance(commit, Mapping) else None
    if branch.get("protected") is not False or sha != inspection.sha:
        raise CleanupError(f"branch protection or tip changed: {inspection.name}")
    open_pulls = api.paginated(
        f"/repos/{manifest.repository}/pulls",
        query={"state": "open"},
    )
    numbers = _pull_request_numbers(
        open_pulls,
        repository=manifest.repository,
        branch=inspection.name,
    )
    if numbers:
        raise CleanupError(
            f"branch gained open pull requests {numbers}: {inspection.name}"
        )


def _base_execution_receipt(
    loaded: LoadedCleanup,
    now: datetime,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DReviewedAgentBranchCleanupExecution",
        "repository": loaded.manifest.repository,
        "default_branch": loaded.manifest.default_branch,
        "manifest_sha256": loaded.manifest_sha256,
        "source_report_sha256": loaded.source_report_sha256,
        "reviewed_by": loaded.manifest.reviewed_by,
        "issue_url": loaded.manifest.issue_url,
        "executed_at_utc": _utc_text(now),
        "requested_count": len(loaded.manifest.entries),
        "deleted_count": 0,
        "complete": False,
        "mutation_performed": False,
        "deleted_branches": [],
        "failure": None,
    }


def execute_cleanup(
    api: CleanupGitHubApi,
    loaded: LoadedCleanup,
    allowlist: BranchAllowlist,
    now: datetime,
) -> dict[str, Any]:
    """Preflight the tranche, then recheck, delete, and confirm each exact ref."""

    selected = _live_candidates(api, loaded.manifest, allowlist, now)
    receipt = _base_execution_receipt(loaded, now)
    deleted = receipt["deleted_branches"]
    if not isinstance(deleted, list):
        raise RuntimeError("execution receipt deleted_branches must be a list")
    for decision in selected:
        inspection = decision.inspection
        phase = "immediate_recheck"
        try:
            _immediate_recheck(api, loaded.manifest, allowlist, decision)
            phase = "delete_ref"
            api.delete_branch_ref(loaded.manifest.repository, inspection.name)
            phase = "confirm_absence"
            if api.branch_exists(loaded.manifest.repository, inspection.name):
                raise CleanupError(f"deleted branch still resolves: {inspection.name}")
        except (CleanupError, GitHubApiError, OSError) as error:
            receipt["deleted_count"] = len(deleted)
            receipt["mutation_performed"] = (
                bool(deleted) or phase != "immediate_recheck"
            )
            receipt["failure"] = {
                "branch": inspection.name,
                "expected_sha": inspection.sha,
                "phase": phase,
                "message": str(error),
            }
            raise CleanupExecutionError(str(error), receipt) from error
        deleted.append({**decision.as_dict(), "deleted": True})
    receipt["deleted_count"] = len(deleted)
    receipt["complete"] = len(deleted) == len(loaded.manifest.entries)
    receipt["mutation_performed"] = bool(deleted)
    return receipt


def _write(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _failure_receipt(
    *,
    manifest_path: str,
    expected_manifest_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DReviewedAgentBranchCleanupFailure",
        "manifest_path": manifest_path,
        "expected_manifest_sha256": expected_manifest_sha256,
        "complete": False,
        "mutation_performed": False,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=".")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--approval-phrase")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--now", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", required=True)
    arguments = parser.parse_args(argv)
    try:
        loaded, allowlist = load_cleanup(
            arguments.repository_root,
            arguments.manifest,
            arguments.manifest_sha256,
        )
        if arguments.validate_only:
            result: Mapping[str, Any] = {
                "schema_version": 1,
                "artifact_kind": "Causal4DReviewedAgentBranchCleanupValidation",
                "manifest_sha256": loaded.manifest_sha256,
                "source_report_sha256": loaded.source_report_sha256,
                "requested_count": len(loaded.manifest.entries),
                "valid": True,
                "mutation_performed": False,
            }
        else:
            if arguments.approval_phrase != _APPROVAL:
                raise CleanupError(f"--approval-phrase must be exactly {_APPROVAL!r}")
            token = os.environ.get(arguments.token_env, "")
            if not token:
                raise CleanupError(
                    f"environment variable {arguments.token_env} is required"
                )
            now = (
                datetime.now(timezone.utc)
                if arguments.now is None
                else _utc(arguments.now, "--now")
            )
            result = execute_cleanup(
                CleanupGitHubApi(token),
                loaded,
                allowlist,
                now,
            )
    except CleanupExecutionError as error:
        _write(arguments.output_json, error.receipt)
        print(f"reviewed branch cleanup failed: {error}", file=sys.stderr)
        return 1
    except (CleanupError, GitHubApiError, OSError) as error:
        failure = _failure_receipt(
            manifest_path=arguments.manifest,
            expected_manifest_sha256=arguments.manifest_sha256,
            error=error,
        )
        _write(arguments.output_json, failure)
        print(f"reviewed branch cleanup failed: {error}", file=sys.stderr)
        return 1
    _write(arguments.output_json, result)
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
