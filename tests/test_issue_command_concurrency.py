from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / ".github" / "workflows"

ISSUE_COMMAND_JOBS = {
    "acquisition-readiness-self-hosted.yml": (
        "inspect",
        "acquisition-readiness-${{ github.sha }}",
    ),
    "automatable-preacquisition-self-hosted.yml": (
        "execute",
        "automatable-preacquisition-${{ github.sha }}",
    ),
    "controlled-execution-campaign.yml": (
        "dispatch",
        "controlled-execution-campaign-${{ github.sha }}",
    ),
    "correct-operator-registry-self-hosted.yml": (
        "correct",
        "correct-operator-registry-${{ github.sha }}",
    ),
    "deform360-reset-mechanics-issue-dispatch.yml": (
        "dispatch",
        "deform360-reset-mechanics-issue-dispatch-${{ github.sha }}",
    ),
    "prepared-joint-observation-self-hosted.yml": (
        "validate",
        "prepared-joint-observation-${{ github.sha }}",
    ),
}


def _job_block(text: str, job_name: str) -> str:
    lines = text.splitlines()
    marker = f"  {job_name}:"
    start = lines.index(marker)
    end = len(lines)
    for index in range(start + 1, len(lines)):
        if re.fullmatch(r"  [A-Za-z0-9_-]+:", lines[index]):
            end = index
            break
    return "\n".join(lines[start:end])


def test_every_opened_issue_workflow_is_registered() -> None:
    actual = {
        path.name
        for path in WORKFLOWS.glob("*.yml")
        if "\n  issues:\n    types: [opened]\n" in path.read_text(encoding="utf-8")
    }

    assert actual == set(ISSUE_COMMAND_JOBS)


def test_unrelated_issue_events_cannot_replace_pending_authorized_jobs() -> None:
    for filename, (job_name, group) in ISSUE_COMMAND_JOBS.items():
        text = (WORKFLOWS / filename).read_text(encoding="utf-8")
        block = _job_block(text, job_name)

        assert "concurrency:" not in {
            line for line in text.splitlines() if not line.startswith(" ")
        }
        assert "github.event.issue.title" in block
        assert "    concurrency:\n" in block
        assert f"      group: {group}\n" in block
        assert "      cancel-in-progress: false\n" in block
        assert "      queue: max\n" in block


def test_issue_command_concurrency_remains_job_scoped() -> None:
    for filename, (job_name, _) in ISSUE_COMMAND_JOBS.items():
        text = (WORKFLOWS / filename).read_text(encoding="utf-8")
        jobs_prefix, jobs_text = text.split("\njobs:\n", maxsplit=1)
        block = _job_block(text, job_name)

        assert "\nconcurrency:\n" not in f"\n{jobs_prefix}\n"
        assert block.count("    concurrency:\n") == 1
        assert jobs_text.count("    concurrency:\n") == 1
