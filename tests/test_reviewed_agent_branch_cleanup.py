from __future__ import annotations

from datetime import datetime, timezone
from email.message import Message
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping
from urllib.error import HTTPError

import pytest


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

CONTRACT_PATH = (
    ROOT / "scripts" / "ci" / "reviewed_agent_branch_cleanup_contract.py"
)
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
        _branch_payload(branch, sha)
        if branch_response is None
        else branch_response
    )


def _set_live(
    monkeypatch: Any,
    *decisions: Any,
    default_branch: str = "main",
) -> None:
    def inspect(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...]]:
        return default_branch, tuple(decisions)

    monkeypatch.setattr(MODULE, "inspect_agent_branches", inspect)


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
    allowlist.replace(allowlist_target)
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


def test_complete_live_preflight_occurs_before_any_delete(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    second_branch = "agent/second-old-change"
    decisions = [
        _decision_payload(),
        _decision_payload(branch=second_branch, sha=OTHER_SHA),
    ]
    manifest_path, manifest_sha = _write_review_files(tmp_path, decisions=decisions)
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()
    _set_live(
        monkeypatch,
        _live_decision(),
        _live_decision(branch=second_branch, sha=SHA),
    )

    with pytest.raises(CleanupError, match="lineage changed"):
        execute_cleanup(api, loaded, allowlist, NOW)

    assert api.deleted == []
    assert api.get_calls == []


def test_immediate_tip_change_stops_before_delete(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()
    _set_live(monkeypatch, _live_decision())
    _configure_immediate(api, branch_response=_branch_payload(sha=OTHER_SHA))

    with pytest.raises(CleanupExecutionError, match="tip changed") as raised:
        execute_cleanup(api, loaded, allowlist, NOW)

    assert api.deleted == []
    assert raised.value.receipt["mutation_performed"] is False
    assert raised.value.receipt["failure"]["phase"] == "immediate_recheck"


def test_open_pull_request_stops_immediate_deletion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()
    _set_live(monkeypatch, _live_decision())
    _configure_immediate(api)
    api.open_pulls = [
        {
            "number": 999,
            "head": {
                "ref": BRANCH,
                "sha": SHA,
                "repo": {"full_name": REPOSITORY},
            },
        }
    ]

    with pytest.raises(CleanupExecutionError, match="gained open pull requests"):
        execute_cleanup(api, loaded, allowlist, NOW)

    assert api.deleted == []


def test_execute_cleanup_deletes_and_confirms_exact_ref(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()
    _set_live(monkeypatch, _live_decision())
    _configure_immediate(api)

    result = execute_cleanup(api, loaded, allowlist, NOW)

    assert api.deleted == [(REPOSITORY, BRANCH)]
    assert result["complete"] is True
    assert result["deleted_count"] == 1
    assert result["mutation_performed"] is True
    assert result["executed_at_utc"] == _utc(NOW)
    assert result["deleted_branches"][0]["sha"] == SHA


def test_merged_tip_requires_exact_live_pull_request_lineage(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    report = _decision_payload(
        reason="exact_tip_merged_in_pull_request",
        merged=(259,),
    )
    manifest_path, manifest_sha = _write_review_files(tmp_path, decisions=[report])
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()
    _set_live(
        monkeypatch,
        _live_decision(
            reason="exact_tip_merged_in_pull_request",
            merged=(260,),
        ),
    )

    with pytest.raises(CleanupError, match="lineage changed"):
        execute_cleanup(api, loaded, allowlist, NOW)

    assert api.deleted == []


def test_expired_manifest_fails_before_live_inspection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    manifest_path, manifest_sha = _write_review_files(
        tmp_path,
        expires=datetime(2026, 8, 17, 10, 0, tzinfo=timezone.utc),
    )
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()

    def unexpected(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("live inspection must not run after expiry")

    monkeypatch.setattr(MODULE, "inspect_agent_branches", unexpected)
    with pytest.raises(CleanupError, match="expired"):
        execute_cleanup(api, loaded, allowlist, NOW)

    assert api.get_calls == []


def test_partial_failure_raises_with_durable_receipt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    second_branch = "agent/second-old-change"
    decisions = [
        _decision_payload(),
        _decision_payload(branch=second_branch, sha=OTHER_SHA),
    ]
    manifest_path, manifest_sha = _write_review_files(tmp_path, decisions=decisions)
    loaded, allowlist = load_cleanup(tmp_path, manifest_path, manifest_sha)
    api = FakeApi()
    _set_live(
        monkeypatch,
        _live_decision(),
        _live_decision(branch=second_branch, sha=OTHER_SHA),
    )
    _configure_immediate(api)
    _configure_immediate(api, branch=second_branch, sha=OTHER_SHA)
    api.delete_failures[second_branch] = GitHubApiError("injected delete failure")

    with pytest.raises(
        CleanupExecutionError,
        match="injected delete failure",
    ) as raised:
        execute_cleanup(api, loaded, allowlist, NOW)

    receipt = raised.value.receipt
    assert api.deleted == [(REPOSITORY, BRANCH)]
    assert receipt["complete"] is False
    assert receipt["mutation_performed"] is True
    assert receipt["deleted_count"] == 1
    assert receipt["failure"]["branch"] == second_branch
    assert receipt["failure"]["phase"] == "delete_ref"


def test_branch_confirmation_treats_only_404_as_absent(monkeypatch: Any) -> None:
    api = CleanupGitHubApi("token")
    headers = Message()

    def missing(*args: Any, **kwargs: Any) -> Any:
        raise HTTPError("url", 404, "missing", headers, None)

    monkeypatch.setattr(MODULE, "urlopen", missing)
    assert api.branch_exists(REPOSITORY, BRANCH) is False

    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise HTTPError("url", 403, "forbidden", headers, None)

    monkeypatch.setattr(MODULE, "urlopen", forbidden)
    with pytest.raises(GitHubApiError, match="403"):
        api.branch_exists(REPOSITORY, BRANCH)


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


def test_invalid_allowlist_is_wrapped_as_cleanup_error(tmp_path: Path) -> None:
    manifest_path, manifest_sha = _write_review_files(tmp_path)
    allowlist = tmp_path / ".github" / "stale-agent-branch-allowlist.json"
    allowlist.write_text(
        json.dumps({"schema_version": 99, "branches": [], "prefixes": []}),
        encoding="utf-8",
    )

    with pytest.raises(CleanupError, match="invalid branch allowlist"):
        load_cleanup(tmp_path, manifest_path, manifest_sha)
