from __future__ import annotations

from datetime import datetime, timezone
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

from reviewed_agent_branch_cleanup_test_support import (
    BRANCH,
    MODULE,
    NOW,
    OTHER_SHA,
    REPOSITORY,
    SHA,
    CleanupError,
    CleanupExecutionError,
    CleanupGitHubApi,
    FakeApi,
    GitHubApiError,
    _branch_payload,
    _configure_immediate,
    _decision_payload,
    _live_decision,
    _set_live,
    _utc,
    _write_review_files,
    execute_cleanup,
    load_cleanup,
)


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
