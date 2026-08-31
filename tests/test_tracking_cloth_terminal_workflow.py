from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUEST_PATH = (
    ROOT / "ops" / "tracking-cloth-query-observation-gpuserver4090-request.json"
)
TERMINAL_PATH = (
    ROOT
    / "evidence"
    / "negative"
    / "tracking-cloth-shake-to-twist-20260830"
    / "result.json"
)
TERMINALIZER_PATH = (
    ROOT
    / ".github"
    / "workflows"
    / "tracking-cloth-query-observation-file-dispatch.yml"
)


def test_terminal_tracking_cloth_request_is_disabled() -> None:
    request = json.loads(REQUEST_PATH.read_text(encoding="utf-8"))
    record = json.loads(TERMINAL_PATH.read_text(encoding="utf-8"))

    assert request == {
        "enabled": False,
        "ref": "main",
        "request": "record-terminal-tracking-cloth-negative-result",
        "request_id": (
            "3e9f77bd804f91545dbdac1098c324b3b155167c938c186a2e9238c712cdeca0"
        ),
        "schema_version": 1,
        "supersedes_run_id": 33363286850,
        "terminal_record": (
            "evidence/negative/tracking-cloth-shake-to-twist-20260830/result.json"
        ),
        "workflow": "tracking-cloth-query-observation.yml",
    }
    assert record["status"] == "completed_negative_result"
    assert record["terminal_for_registered_protocol"] is True
    assert record["creates_new_experiment_requirement"] is False
    assert record["paper_claim_authorized"] is False
    decision = record["primary_evaluation"]["decision"]
    assert decision["physics_transfer_beats_persistence"] is False


def test_terminalizer_cancels_obsolete_run_without_dispatch() -> None:
    workflow = TERMINALIZER_PATH.read_text(encoding="utf-8")

    assert "group: tracking-cloth-query-observation-${{ github.ref }}" in workflow
    assert "cancel-in-progress: true" in workflow
    assert "actions: write" not in workflow
    assert "/dispatches" not in workflow
    assert "dispatch_performed=false" in workflow
    assert '"dispatch_performed": False' in workflow
    assert '"self_hosted_runner_allocated": False' in workflow
    assert "33363286850" in workflow
    assert '"evidence/negative/"' in workflow
    assert '"tracking-cloth-shake-to-twist-20260830/result.json"' in workflow
