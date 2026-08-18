from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.write_filament_support_dispatch_receipt import (
    DispatchReceiptError,
    build_receipt,
    main,
)


REPOSITORY = "IPS-Stuttgart/Causal4D"
MAIN_SHA = "a" * 40
RUN_ID = 123456789
RUN_URL = f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID}"


def test_receipt_records_only_source_dispatch_evidence() -> None:
    receipt = build_receipt(
        repository=REPOSITORY,
        reviewed_main_sha=MAIN_SHA,
        trigger_issue_number=388,
        workflow_run_id=RUN_ID,
        workflow_run_url=RUN_URL,
    )

    assert receipt.repository == REPOSITORY
    assert receipt.reviewed_main_sha == MAIN_SHA
    assert receipt.run_source_diagnostic is True
    assert receipt.target_outcomes_used is False
    assert receipt.registered_physical_dataset_modified is False
    assert receipt.physical_evidence_increment == 0


def test_receipt_rejects_run_url_from_a_different_identity() -> None:
    with pytest.raises(DispatchReceiptError, match="does not identify"):
        build_receipt(
            repository=REPOSITORY,
            reviewed_main_sha=MAIN_SHA,
            trigger_issue_number=388,
            workflow_run_id=RUN_ID,
            workflow_run_url=(
                f"https://github.com/{REPOSITORY}/actions/runs/{RUN_ID + 1}"
            ),
        )


def test_cli_writes_deterministic_receipt_without_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "dispatch.json"
    argv = [
        "--repository",
        REPOSITORY,
        "--reviewed-main-sha",
        MAIN_SHA,
        "--trigger-issue-number",
        "388",
        "--workflow-run-id",
        str(RUN_ID),
        "--workflow-run-url",
        RUN_URL,
        "--output",
        str(output),
    ]

    assert main(argv) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "Causal4DDeform360FilamentSupportDispatch"
    assert payload["workflow_run_id"] == RUN_ID
    assert payload["physical_evidence_increment"] == 0

    with pytest.raises(DispatchReceiptError, match="refusing to overwrite"):
        main(argv)
