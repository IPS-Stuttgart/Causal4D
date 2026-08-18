#!/usr/bin/env python3
"""Write a validated receipt for the filament-support issue dispatcher."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_REPOSITORY = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_COMMIT_SHA = re.compile(r"^[0-9a-f]{40}$")


class DispatchReceiptError(ValueError):
    """Raised when dispatch metadata is malformed or inconsistent."""


@dataclass(frozen=True)
class DispatchReceipt:
    """Non-physical audit record for one source-only workflow dispatch."""

    schema_version: int
    artifact_kind: str
    repository: str
    reviewed_main_sha: str
    trigger_issue_number: int
    workflow_run_id: int
    workflow_run_url: str
    run_source_diagnostic: bool
    target_outcomes_used: bool
    registered_physical_dataset_modified: bool
    physical_evidence_increment: int


def build_receipt(
    *,
    repository: str,
    reviewed_main_sha: str,
    trigger_issue_number: int,
    workflow_run_id: int,
    workflow_run_url: str,
) -> DispatchReceipt:
    """Validate dispatch identity and return its immutable receipt."""

    if _REPOSITORY.fullmatch(repository) is None:
        raise DispatchReceiptError(
            "repository must use the non-empty 'owner/name' form"
        )
    if _COMMIT_SHA.fullmatch(reviewed_main_sha) is None:
        raise DispatchReceiptError(
            "reviewed_main_sha must be a lowercase 40-character commit SHA"
        )
    if trigger_issue_number <= 0:
        raise DispatchReceiptError("trigger_issue_number must be positive")
    if workflow_run_id <= 0:
        raise DispatchReceiptError("workflow_run_id must be positive")

    expected_url = f"https://github.com/{repository}/actions/runs/{workflow_run_id}"
    if workflow_run_url != expected_url:
        raise DispatchReceiptError(
            "workflow_run_url does not identify workflow_run_id in repository: "
            f"expected {expected_url!r}, received {workflow_run_url!r}"
        )

    return DispatchReceipt(
        schema_version=1,
        artifact_kind="Causal4DDeform360FilamentSupportDispatch",
        repository=repository,
        reviewed_main_sha=reviewed_main_sha,
        trigger_issue_number=trigger_issue_number,
        workflow_run_id=workflow_run_id,
        workflow_run_url=workflow_run_url,
        run_source_diagnostic=True,
        target_outcomes_used=False,
        registered_physical_dataset_modified=False,
        physical_evidence_increment=0,
    )


def write_receipt(receipt: DispatchReceipt, output: Path) -> None:
    """Write ``receipt`` deterministically, replacing no existing artifact."""

    if output.exists():
        raise DispatchReceiptError(f"refusing to overwrite existing receipt: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(asdict(receipt), indent=2, sort_keys=True) + "\n"
    output.write_text(text, encoding="utf-8")
    print(text, end="")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--reviewed-main-sha", required=True)
    parser.add_argument("--trigger-issue-number", type=int, required=True)
    parser.add_argument("--workflow-run-id", type=int, required=True)
    parser.add_argument("--workflow-run-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    receipt = build_receipt(
        repository=args.repository,
        reviewed_main_sha=args.reviewed_main_sha,
        trigger_issue_number=args.trigger_issue_number,
        workflow_run_id=args.workflow_run_id,
        workflow_run_url=args.workflow_run_url,
    )
    write_receipt(receipt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
