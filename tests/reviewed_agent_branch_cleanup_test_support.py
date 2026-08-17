from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
STALE_PATH = ROOT / "scripts" / "ci" / "stale_agent_branches.py"
STALE_SPEC = importlib.util.spec_from_file_location(
    "stale_agent_branches",
    STALE_PATH,
)
assert STALE_SPEC is not None and STALE_SPEC.loader is not None
STALE_MODULE = importlib.util.module_from_spec(STALE_SPEC)
sys.modules[STALE_SPEC.name] = STALE_MODULE
STALE_SPEC.loader.exec_module(STALE_MODULE)

CONTRACT_PATH = ROOT / "scripts" / "ci" / "reviewed_agent_branch_cleanup_contract.py"
CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "reviewed_agent_branch_cleanup_contract",
    CONTRACT_PATH,
)
assert CONTRACT_SPEC is not None and CONTRACT_SPEC.loader is not None
CONTRACT_MODULE = importlib.util.module_from_spec(CONTRACT_SPEC)
sys.modules[CONTRACT_SPEC.name] = CONTRACT_MODULE
CONTRACT_SPEC.loader.exec_module(CONTRACT_MODULE)

MODULE_PATH = ROOT / "scripts" / "ci" / "reviewed_agent_branch_cleanup.py"
SPEC = importlib.util.spec_from_file_location(
    "reviewed_agent_branch_cleanup",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


BranchAllowlist = STALE_MODULE.BranchAllowlist
BranchDecision = STALE_MODULE.BranchDecision
BranchInspection = STALE_MODULE.BranchInspection
CleanupError = CONTRACT_MODULE.CleanupError
CleanupExecutionError = MODULE.CleanupExecutionError
CleanupGitHubApi = MODULE.CleanupGitHubApi
GitHubApiError = MODULE.GitHubApiError
execute_cleanup = MODULE.execute_cleanup
load_cleanup = CONTRACT_MODULE.load_cleanup
main = MODULE.main


NOW = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
GENERATED = datetime(2026, 8, 16, 9, 0, tzinfo=timezone.utc)
REVIEWED = datetime(2026, 8, 16, 12, 0, tzinfo=timezone.utc)
EXPIRES = datetime(2026, 8, 20, 12, 0, tzinfo=timezone.utc)
COMMITTED = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
SHA = "1" * 40
OTHER_SHA = "2" * 40
REPOSITORY = "IPS-Stuttgart/Causal4D"
BRANCH = "agent/old-merged-change"


def _utc(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _decision_payload(
    *,
    branch: str = BRANCH,
    sha: str = SHA,
    reason: str = "tip_reachable_from_default",
    merged: tuple[int, ...] = (),
    candidate: bool = True,
) -> dict[str, Any]:
    return {
        "name": branch,
        "sha": sha,
        "committed_at_utc": _utc(COMMITTED),
        "protected": False,
        "allowlisted": False,
        "open_pull_requests": [],
        "exact_tip_merged_pull_requests": list(merged),
        "tip_reachable_from_default": reason == "tip_reachable_from_default",
        "cleanup_candidate": candidate,
        "reason": reason,
        "age_days": 76.0,
    }


def _live_decision(
    *,
    branch: str = BRANCH,
    sha: str = SHA,
    reason: str = "tip_reachable_from_default",
    merged: tuple[int, ...] = (),
    candidate: bool = True,
) -> Any:
    inspection = BranchInspection(
        name=branch,
        sha=sha,
        committed_at_utc=_utc(COMMITTED),
        protected=False,
        allowlisted=False,
        open_pull_requests=(),
        exact_tip_merged_pull_requests=merged,
        tip_reachable_from_default=reason == "tip_reachable_from_default",
    )
    return BranchDecision(inspection, candidate, reason, 76.0)


def _entry(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": item["name"],
        "expected_sha": item["sha"],
        "eligibility_reason": item["reason"],
        "merged_pull_requests": item["exact_tip_merged_pull_requests"],
        "artifact_reference_reviewed": True,
        "review_note": "Tip is retained elsewhere; no unique artifact ref.",
    }


def _write_review_files(
    root: Path,
    *,
    decisions: list[dict[str, Any]] | None = None,
    entries: list[dict[str, Any]] | None = None,
    expires: datetime = EXPIRES,
    reason_counts: Mapping[str, int] | None = None,
) -> tuple[str, str]:
    directory = root / "ci" / "branch-cleanup-manifests"
    directory.mkdir(parents=True)
    selected_decisions = decisions or [_decision_payload()]
    counts: dict[str, int] = {}
    for decision in selected_decisions:
        reason = str(decision["reason"])
        counts[reason] = counts.get(reason, 0) + 1
    report = {
        "schema_version": 1,
        "artifact_kind": "Causal4DStaleAgentBranchReport",
        "generated_at_utc": _utc(GENERATED),
        "repository": REPOSITORY,
        "default_branch": "main",
        "prefix": "agent/",
        "minimum_age_days": 30,
        "report_only": True,
        "branch_count": len(selected_decisions),
        "cleanup_candidate_count": sum(
            item["cleanup_candidate"] is True for item in selected_decisions
        ),
        "excluded_count": sum(
            item["cleanup_candidate"] is not True for item in selected_decisions
        ),
        "reason_counts": dict(reason_counts or counts),
        "decisions": selected_decisions,
    }
    report_path = directory / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_entries = entries
    if manifest_entries is None:
        manifest_entries = [
            _entry(item)
            for item in selected_decisions
            if item["cleanup_candidate"] is True
        ]
    manifest = {
        "schema_version": 1,
        "artifact_kind": "Causal4DReviewedAgentBranchCleanupManifest",
        "repository": REPOSITORY,
        "default_branch": "main",
        "source_report_path": "ci/branch-cleanup-manifests/report.json",
        "source_report_sha256": _sha256(report_path),
        "source_report_generated_at_utc": _utc(GENERATED),
        "minimum_age_days": 30,
        "reviewed_at_utc": _utc(REVIEWED),
        "expires_at_utc": _utc(expires),
        "reviewed_by": "FlorianPfaff",
        "issue_url": "https://github.com/IPS-Stuttgart/Causal4D/issues/336",
        "entries": manifest_entries,
    }
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    allowlist_path = root / ".github" / "stale-agent-branch-allowlist.json"
    allowlist_path.parent.mkdir(parents=True)
    allowlist_path.write_text(
        json.dumps({"schema_version": 1, "branches": [], "prefixes": []}),
        encoding="utf-8",
    )
    return manifest_path.relative_to(root).as_posix(), _sha256(manifest_path)


class SequenceValue:
    def __init__(self, *values: Any) -> None:
        self.values = list(values)

    def next(self) -> Any:
        if not self.values:
            raise AssertionError("sequence exhausted")
        return self.values.pop(0)


class FakeApi:
    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.open_pulls: list[Any] = []
        self.get_calls: list[str] = []
        self.deleted: list[tuple[str, str]] = []
        self.existing_after_delete: set[str] = set()
        self.delete_failures: dict[str, Exception] = {}

    def get_json(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        self.get_calls.append(path)
        if path not in self.responses:
            raise AssertionError(f"unexpected GET: {path}, query={query}")
        response = self.responses[path]
        if isinstance(response, SequenceValue):
            response = response.next()
        if isinstance(response, Exception):
            raise response
        return response

    def paginated(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> list[Any]:
        assert path == f"/repos/{REPOSITORY}/pulls"
        assert query == {"state": "open"}
        return self.open_pulls

    def delete_branch_ref(self, repository: str, branch: str) -> None:
        failure = self.delete_failures.get(branch)
        if failure is not None:
            raise failure
        self.deleted.append((repository, branch))

    def branch_exists(self, repository: str, branch: str) -> bool:
        assert repository == REPOSITORY
        return branch in self.existing_after_delete


def _branch_payload(branch: str = BRANCH, sha: str = SHA) -> dict[str, Any]:
    return {
        "name": branch,
        "protected": False,
        "commit": {"sha": sha},
    }


def _configure_immediate(
    api: FakeApi,
    *,
    branch: str = BRANCH,
    sha: str = SHA,
    branch_response: Any | None = None,
) -> None:
    api.responses[f"/repos/{REPOSITORY}"] = {"default_branch": "main"}
    encoded = branch.replace("/", "%2F")
    api.responses[f"/repos/{REPOSITORY}/branches/{encoded}"] = (
        _branch_payload(branch, sha) if branch_response is None else branch_response
    )


def _set_live(
    monkeypatch: Any,
    *decisions: Any,
    default_branch: str = "main",
) -> None:
    def inspect(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...]]:
        return default_branch, tuple(decisions)

    monkeypatch.setattr(MODULE, "inspect_agent_branches", inspect)
