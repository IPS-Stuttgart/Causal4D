from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "stale_agent_branches.py"
SPEC = importlib.util.spec_from_file_location("stale_agent_branches", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


BranchAllowlist = MODULE.BranchAllowlist
BranchInspection = MODULE.BranchInspection
GitHubApiError = MODULE.GitHubApiError
build_report = MODULE.build_report
classify_branch = MODULE.classify_branch
inspect_agent_branches = MODULE.inspect_agent_branches
load_allowlist = MODULE.load_allowlist
render_markdown = MODULE.render_markdown


NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)
OLD = (NOW - timedelta(days=45)).isoformat().replace("+00:00", "Z")
YOUNG = (NOW - timedelta(days=4)).isoformat().replace("+00:00", "Z")
SHA = "1" * 40
OTHER_SHA = "2" * 40


def _inspection(**overrides: Any) -> Any:
    values: dict[str, Any] = {
        "name": "agent/example",
        "sha": SHA,
        "committed_at_utc": OLD,
        "protected": False,
        "allowlisted": False,
        "open_pull_requests": (),
        "exact_tip_merged_pull_requests": (),
        "tip_reachable_from_default": False,
    }
    values.update(overrides)
    return BranchInspection(**values)


class FakeApi:
    def __init__(self) -> None:
        self.responses: dict[str, Any] = {}
        self.paginated_responses: dict[str, Any] = {}
        self.calls: list[tuple[str, Mapping[str, str | int] | None]] = []
        self.paginated_calls: list[tuple[str, Mapping[str, str | int] | None]] = []

    def get_json(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        self.calls.append((path, query))
        if path not in self.responses:
            raise AssertionError(f"unexpected request: {path}, query={query}")
        response = self.responses[path]
        if isinstance(response, Exception):
            raise response
        return response

    def paginated(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> list[Any]:
        self.paginated_calls.append((path, query))
        if path not in self.paginated_responses:
            raise AssertionError(f"unexpected pagination: {path}, query={query}")
        response = self.paginated_responses[path]
        if isinstance(response, dict):
            key = None if query is None else query.get("state")
            if key not in response:
                raise AssertionError(f"unexpected pagination state: {path}, {query}")
            return response[key]
        return response


def test_allowlist_schema_is_strict_and_supports_exact_and_prefix() -> None:
    allowlist = BranchAllowlist.from_mapping(
        {
            "schema_version": 1,
            "branches": ["agent/exact"],
            "prefixes": ["agent/evidence/"],
        }
    )

    assert allowlist.matches("agent/exact") is True
    assert allowlist.matches("agent/evidence/run-1") is True
    assert allowlist.matches("agent/ordinary") is False

    with pytest.raises(ValueError, match="must contain"):
        BranchAllowlist.from_mapping(
            {
                "schema_version": 1,
                "branches": [],
                "prefixes": [],
                "unknown": True,
            }
        )
    with pytest.raises(ValueError, match="unique"):
        BranchAllowlist.from_mapping(
            {
                "schema_version": 1,
                "branches": ["agent/a", "agent/a"],
                "prefixes": [],
            }
        )


def test_load_allowlist_rejects_duplicate_keys_and_symlink(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text(
        '{"schema_version":1,"branches":[],"branches":[],"prefixes":[]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        load_allowlist(duplicate)

    target = tmp_path / "target.json"
    target.write_text(
        json.dumps({"schema_version": 1, "branches": [], "prefixes": []}),
        encoding="utf-8",
    )
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="ordinary file"):
        load_allowlist(link)


def test_classification_is_fail_closed() -> None:
    cases = [
        (_inspection(protected=True), "protected_branch", False),
        (_inspection(allowlisted=True), "allowlisted_branch", False),
        (_inspection(open_pull_requests=(17,)), "open_pull_request", False),
        (_inspection(committed_at_utc=YOUNG), "younger_than_threshold", False),
        (_inspection(), "unmerged_tip", False),
        (
            _inspection(tip_reachable_from_default=True),
            "tip_reachable_from_default",
            True,
        ),
        (
            _inspection(exact_tip_merged_pull_requests=(259,)),
            "exact_tip_merged_in_pull_request",
            True,
        ),
    ]

    for inspection, reason, expected_candidate in cases:
        decision = classify_branch(inspection, now=NOW, minimum_age_days=30)
        assert decision.reason == reason
        assert decision.cleanup_candidate is expected_candidate


def test_inspection_rejects_nonpositive_age_before_api_access() -> None:
    api = FakeApi()

    with pytest.raises(ValueError, match="minimum_age_days must be positive"):
        inspect_agent_branches(
            api,
            repository="IPS-Stuttgart/Causal4D",
            prefix="agent/",
            allowlist=BranchAllowlist(frozenset(), ()),
            minimum_age_days=0,
            now=NOW,
        )

    assert api.calls == []
    assert api.paginated_calls == []


def test_inspection_avoids_history_queries_for_preliminary_exclusions() -> None:
    repository = "IPS-Stuttgart/Causal4D"
    api = FakeApi()
    api.responses[f"/repos/{repository}"] = {"default_branch": "main"}
    api.responses[f"/repos/{repository}/commits/{SHA}"] = {
        "commit": {"committer": {"date": YOUNG}}
    }
    api.paginated_responses[f"/repos/{repository}/pulls"] = []
    api.paginated_responses[f"/repos/{repository}/branches"] = [
        {"name": "agent/young", "protected": False, "commit": {"sha": SHA}},
        {"name": "main", "protected": True, "commit": {"sha": OTHER_SHA}},
    ]

    default, decisions = inspect_agent_branches(
        api,
        repository=repository,
        prefix="agent/",
        allowlist=BranchAllowlist(frozenset(), ()),
        minimum_age_days=30,
        now=NOW,
    )

    assert default == "main"
    assert [decision.reason for decision in decisions] == ["younger_than_threshold"]
    assert all("/compare/" not in path for path, _ in api.calls)


def test_inspection_accepts_only_exact_merged_tip() -> None:
    repository = "IPS-Stuttgart/Causal4D"
    branch = "agent/merged"
    api = FakeApi()
    api.responses[f"/repos/{repository}"] = {"default_branch": "main"}
    api.responses[f"/repos/{repository}/commits/{SHA}"] = {
        "commit": {"committer": {"date": OLD}}
    }
    api.responses[f"/repos/{repository}/compare/{SHA}...main"] = {
        "merge_base_commit": {"sha": OTHER_SHA}
    }
    api.paginated_responses[f"/repos/{repository}/pulls"] = []
    api.paginated_responses[f"/repos/{repository}/branches"] = [
        {"name": branch, "protected": False, "commit": {"sha": SHA}}
    ]
    api.paginated_responses[f"/repos/{repository}/commits/{SHA}/pulls"] = [
        {
            "number": 259,
            "merged_at": "2026-08-08T21:20:12Z",
            "head": {
                "ref": branch,
                "sha": SHA,
                "repo": {"full_name": repository},
            },
        },
        {
            "number": 258,
            "merged_at": "2026-08-08T20:00:00Z",
            "head": {
                "ref": branch,
                "sha": OTHER_SHA,
                "repo": {"full_name": repository},
            },
        },
    ]

    _, decisions = inspect_agent_branches(
        api,
        repository=repository,
        prefix="agent/",
        allowlist=BranchAllowlist(frozenset(), ()),
        minimum_age_days=30,
        now=NOW,
    )

    assert len(decisions) == 1
    assert decisions[0].cleanup_candidate is True
    assert decisions[0].inspection.exact_tip_merged_pull_requests == (259,)


def test_malformed_api_payload_fails_closed() -> None:
    api = FakeApi()
    api.responses["/repos/IPS-Stuttgart/Causal4D"] = []

    with pytest.raises(GitHubApiError, match="repository response"):
        inspect_agent_branches(
            api,
            repository="IPS-Stuttgart/Causal4D",
            prefix="agent/",
            allowlist=BranchAllowlist(frozenset(), ()),
            minimum_age_days=30,
            now=NOW,
        )


def test_report_has_no_mutation_or_deletion_state() -> None:
    decision = classify_branch(
        _inspection(tip_reachable_from_default=True),
        now=NOW,
        minimum_age_days=30,
    )
    report = build_report(
        repository="IPS-Stuttgart/Causal4D",
        default_branch="main",
        prefix="agent/",
        minimum_age_days=30,
        now=NOW,
        decisions=(decision,),
    )

    assert report["report_only"] is True
    assert report["cleanup_candidate_count"] == 1
    assert "deleted" not in report
    markdown = render_markdown(report)
    assert "read-only" in markdown
    assert "never deletes" in markdown
    assert "agent/example" in markdown
