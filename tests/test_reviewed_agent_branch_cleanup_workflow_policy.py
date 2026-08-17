from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "reviewed-agent-branch-cleanup.yml"
SCRIPT = ROOT / "scripts" / "ci" / "reviewed_agent_branch_cleanup.py"


def _job(name: str, next_name: str | None = None) -> str:
    text = WORKFLOW.read_text(encoding="utf-8")
    section = text.split(f"\n  {name}:\n", 1)[1]
    if next_name is not None:
        section = section.split(f"\n  {next_name}:\n", 1)[0]
    return section


def test_cleanup_workflow_is_manual_main_only_and_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in text
    assert "schedule:" not in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert text.count("refs/heads/main") >= 4
    assert text.count("runs-on: ubuntu-latest") == 2
    assert "self-hosted" not in text


def test_write_permission_is_isolated_to_execution_job() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    header = text.split("\njobs:\n", 1)[0]
    validation = _job("validate", "cleanup")
    cleanup = _job("cleanup")

    assert "permissions: {}" in header
    assert "contents: write" not in header
    assert "contents: read" in validation
    assert "contents: write" not in validation
    assert "pull-requests: read" not in validation
    assert "contents: write" in cleanup
    assert "pull-requests: read" in cleanup
    assert text.count("contents: write") == 1
    assert text.count("pull-requests: read") == 1
    assert "issues: write" not in text
    assert "actions: write" not in text


def test_dispatch_binds_reviewed_path_hash_and_approval_phrase() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "manifest_path:" in text
    assert "manifest_sha256:" in text
    assert "approval_phrase:" in text
    assert text.count("delete-reviewed-agent-branches") >= 2
    assert text.count("ci/branch-cleanup-manifests/*.json") == 2
    assert text.count("--manifest-sha256") == 2
    assert "--validate-only" in text
    assert "reviewed-branch-cleanup-validation.json" in text
    assert "reviewed-branch-cleanup-execution.json" in text
    assert "GITHUB_TOKEN: ${{ github.token }}" in text
    assert "needs.validate.result == 'success'" in text


def test_both_checkouts_are_exact_shallow_and_uncredentialed() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("ref: ${{ github.sha }}") == 2
    assert text.count("fetch-depth: 1") == 2
    assert "fetch-depth: 0" not in text
    assert text.count("persist-credentials: false") == 2
    assert text.count('test "$(git rev-parse HEAD)" = "$GITHUB_SHA"') == 2
    assert text.count('test -z "$(git status --porcelain)"') == 2


def test_every_action_is_pinned_and_both_receipts_are_retained() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    action_lines = [line.strip() for line in text.splitlines() if "uses:" in line]

    assert action_lines
    assert all(
        re.search(r"uses: [^@\s]+@[0-9a-f]{40}(?:\s|$)", line)
        for line in action_lines
    )
    assert text.count("if: always()") == 3
    assert text.count("retention-days: 90") == 2
    assert "partial-failure execution receipt" in text
    assert SCRIPT.is_file()
    assert text.count("scripts/ci/reviewed_agent_branch_cleanup.py") == 2
