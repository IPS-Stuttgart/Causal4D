from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "stale-agent-branch-hygiene.yml"
SCRIPT = ROOT / "scripts" / "ci" / "stale_agent_branches.py"
ALLOWLIST = ROOT / ".github" / "stale-agent-branch-allowlist.json"


def test_workflow_is_manual_or_scheduled_and_never_self_hosted() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "schedule:" in text
    assert "workflow_dispatch:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "self-hosted" not in text
    assert "runs-on: ubuntu-latest" in text


def test_workflow_is_strictly_read_only() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    script = SCRIPT.read_text(encoding="utf-8")

    assert "contents: read" in workflow
    assert "pull-requests: read" in workflow
    assert "contents: write" not in workflow
    assert "DELETE_AGENT_BRANCHES" not in workflow
    assert "maximum_deletions" not in workflow
    assert 'method="GET"' in script
    assert '"DELETE"' not in script
    assert "/git/refs/heads/" not in script


def test_every_action_is_pinned_and_checkout_drops_credentials() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")
    action_lines = [line.strip() for line in text.splitlines() if "uses:" in line]

    assert action_lines
    assert all(
        re.search(r"uses: [^@\s]+@[0-9a-f]{40}(?:\s|$)", line) for line in action_lines
    )
    assert text.count("persist-credentials: false") == 1
    assert text.count("ref: ${{ github.sha }}") == 1
    assert text.count('test "$(git rev-parse HEAD)" = "$GITHUB_SHA"') == 1


def test_script_and_allowlist_are_bound_into_workflow() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert SCRIPT.is_file()
    assert ALLOWLIST.is_file()
    assert text.count("scripts/ci/stale_agent_branches.py") == 1
    assert text.count(".github/stale-agent-branch-allowlist.json") == 1
