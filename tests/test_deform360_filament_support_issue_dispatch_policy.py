from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "deform360-filament-support-issue-dispatch.yml"
)


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_dispatcher_uses_only_the_exact_maintainer_issue() -> None:
    text = _workflow_text()

    assert "on:\n  issues:\n    types: [opened]\n" in text
    assert "github.event_name == 'issues'" in text
    assert "github.event.action == 'opened'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert "'[self-hosted] run Causal4D Deform360 filament support'" in text
    assert text.count("github.event.issue.title") == 1
    assert "github.event.issue.body" not in text
    assert "github.event.issue.labels" not in text
    assert "github.event.comment" not in text


def test_dispatcher_is_hosted_and_dispatches_fixed_reviewed_main() -> None:
    text = _workflow_text()

    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted]" not in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "issues: write" in text
    assert "GH_TOKEN: ${{ github.token }}" in text
    assert "${{ secrets." not in text
    assert "commits/main" in text
    assert 'test "${main_sha}" = "${GITHUB_SHA}"' in text
    assert "gh workflow run deform360-filament-support.yml" in text
    assert "--ref main" in text
    assert "-f run_source_diagnostic=true" in text
    assert 'test "${identity[0]}" = "workflow_dispatch"' in text
    assert 'test "${identity[1]}" = "${MAIN_SHA}"' in text


def test_dispatcher_retains_a_nonphysical_receipt() -> None:
    text = _workflow_text()

    assert "Causal4DDeform360FilamentSupportDispatch" in text
    assert '"target_outcomes_used": False' in text
    assert '"registered_physical_dataset_modified": False' in text
    assert '"physical_evidence_increment": 0' in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "causal4d-deform360-filament-support-dispatch" in text
