from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "controlled-execution-campaign.yml"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_campaign_uses_an_exact_maintainer_issue_trigger() -> None:
    text = _workflow_text()

    assert "on:\n  issues:\n    types: [opened]\n" in text
    assert "github.event_name == 'issues'" in text
    assert "github.event.action == 'opened'" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert (
        "'[self-hosted] run Causal4D controlled execution campaign'" in text
    )
    assert text.count("github.event.issue.title") == 1
    for forbidden in (
        "github.event.issue.body",
        "github.event.issue.labels",
        "github.event.comment",
    ):
        assert forbidden not in text


def test_campaign_dispatcher_is_hosted_and_secret_free() -> None:
    text = _workflow_text()

    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: [self-hosted" not in text
    assert "actions: write" in text
    assert "contents: read" in text
    assert "issues: write" in text
    assert "${{ secrets." not in text
    assert "GH_TOKEN: ${{ github.token }}" in text
    assert "commits/main" in text
    assert 'test "${main_sha}" = "${GITHUB_SHA}"' in text


def test_campaign_dispatches_only_fixed_reviewed_workflows_and_inputs() -> None:
    text = _workflow_text()
    workflows = (
        "self-hosted-evaluation.yml",
        "workstation2-evaluation.yml",
        "deform360-reset-mechanics.yml",
        "deform360-prefix-kinematics.yml",
        "deform360-filament-support.yml",
        "deform360-contact-support.yml",
    )

    for workflow in workflows:
        assert workflow in text
    assert text.count("--ref main") == 3
    assert "-f profile=full" in text
    assert "-f run_bpt=true" in text
    assert "-f cuda_visible_devices=0" in text
    assert "-f extended_seeds=500:540" in text
    assert "-f run_source_diagnostic=true" in text
    assert "github.event.issue.body" not in text


def test_campaign_retains_a_nonclaiming_dispatch_receipt() -> None:
    text = _workflow_text()

    assert "Causal4DControlledExecutionCampaignDispatch" in text
    assert '"target_outcomes_used": False' in text
    assert '"registered_physical_dataset_modified": False' in text
    assert '"physical_evidence_increment": 0' in text
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a" in text
    assert "cannot authorize the registered 36-execution campaign" in text
