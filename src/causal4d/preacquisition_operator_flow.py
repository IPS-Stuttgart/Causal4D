"""Operator sequencing for pre-acquisition next-action decisions."""

from __future__ import annotations

import shlex
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from causal4d.atomic_io import atomic_write_json, atomic_write_text
from causal4d.preacquisition_next_action import (
    build_preacquisition_next_action as _build_next_action,
    derive_preacquisition_next_action as _derive_next_action,
    next_action_evidence_sha256,
    next_action_status_sha256,
)

OPERATOR_FLOW_SCHEMA_VERSION = 2
NEXT_ACTION_SCHEMA_VERSION = 3


def _command_pair(argv: list[str]) -> tuple[list[str], str]:
    return argv, shlex.join(argv)


def _expected_publication_command(
    repository_root: str,
    dataset_root: str,
    staging_path: str,
) -> list[str]:
    return [
        "causal4d",
        "protocol",
        "readiness",
        "source-panel-publish",
        repository_root,
        dataset_root,
        staging_path,
    ]


def enrich_preacquisition_next_action(
    decision: Mapping[str, Any],
) -> dict[str, Any]:
    """Insert safe staging, verification, and governed publication."""

    result = deepcopy(dict(decision))
    action_value = result.get("action")
    if not isinstance(action_value, Mapping):
        raise ValueError("next-action decision has no action object")
    action = deepcopy(dict(action_value))
    governance = result.get("governance", {})
    single_operator = bool(
        isinstance(governance, Mapping)
        and governance.get("mode") == "single_operator_self_attested"
        and governance.get("single_operator_allowed") is True
        and governance.get("independent_verifier_required") is False
    )
    result["schema_version"] = NEXT_ACTION_SCHEMA_VERSION
    result["operator_flow_schema_version"] = OPERATOR_FLOW_SCHEMA_VERSION

    if action.get("action_id") == "acquire_next_source_panel_execution":
        repository = result.get("repository_root")
        dataset = result.get("dataset_root")
        execution = action.get("registered_execution")
        if not isinstance(repository, str) or not repository:
            raise ValueError("next-action repository root is missing")
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("next-action dataset root is missing")
        if not isinstance(execution, Mapping):
            raise ValueError("registered source execution is missing")
        execution_id = execution.get("execution_id")
        if not isinstance(execution_id, str) or not execution_id:
            raise ValueError("registered source execution id is missing")

        root = Path(dataset)
        staging = str(root / "staging" / f"{execution_id}.json")
        preflight = str(root / "operator" / f"{execution_id}-preflight.json")
        receipt = str(root / "staging" / "reviews" / f"{execution_id}.json")
        expected_publication = _expected_publication_command(
            repository,
            dataset,
            staging,
        )
        legacy_publication = action.get("after_completion_argv")
        if legacy_publication != expected_publication:
            raise ValueError(
                "source action publication command differs from the registered path"
            )

        staging_argv, staging_text = _command_pair(
            [
                "causal4d",
                "protocol",
                "readiness",
                "source-panel-stage",
                repository,
                dataset,
                "--started-at-utc",
                "<execution-started-at-utc>",
                "--ended-at-utc",
                "<execution-ended-at-utc>",
                "--artifact",
                "<dataset-relative-artifact-path>",
            ]
        )
        verification_argv, verification_text = _command_pair(
            [
                "causal4d",
                "protocol",
                "readiness",
                "source-panel-verify-staged",
                repository,
                dataset,
                staging,
                "--output-json",
                preflight,
            ]
        )
        review_argv, review_text = _command_pair(
            [
                "causal4d",
                "protocol",
                "readiness",
                "source-panel-review-staged",
                repository,
                dataset,
                staging,
                "--reviewed-by",
                (
                    "<registered-self-attesting-operator-id>"
                    if single_operator
                    else "<registered-reviewer-id>"
                ),
            ]
        )
        publication_argv, publication_text = _command_pair(
            [
                *expected_publication,
                "--review-receipt",
                receipt,
                "--published-by",
                (
                    "<registered-self-attesting-operator-id>"
                    if single_operator
                    else "<registered-publisher-id>"
                ),
            ]
        )
        action["after_completion_argv"] = None
        action["after_completion_text"] = None
        action["staged_manifest_build_argv"] = staging_argv
        action["staged_manifest_build_text"] = staging_text
        action["staged_manifest_path"] = staging
        action["staged_manifest_build_note"] = (
            "Repeat --artifact once for every ordinary artifact file below the "
            "registered execution directory."
        )
        action["post_acquisition_verification_argv"] = verification_argv
        action["post_acquisition_verification_text"] = verification_text
        action["preflight_report_path"] = preflight
        action["staged_review_argv"] = review_argv
        action["staged_review_text"] = review_text
        action["review_receipt_path"] = receipt
        action["two_person_publication_required"] = not single_operator
        action["reviewer_identity_placeholder"] = (
            "<registered-self-attesting-operator-id>"
            if single_operator
            else "<registered-reviewer-id>"
        )
        action["publisher_identity_placeholder"] = (
            "<registered-self-attesting-operator-id>"
            if single_operator
            else "<registered-publisher-id>"
        )
        action["independent_review_required_before_publication"] = not single_operator
        action["self_review_required_before_publication"] = single_operator
        action["claim_bearing_publication_argv"] = publication_argv
        action["claim_bearing_publication_text"] = publication_text
        action["operator_sequence"] = [
            "acquire_registered_source_execution",
            "build_staged_manifest_from_registered_template_and_artifact_bytes",
            "verify_staged_manifest_and_artifacts",
            (
                "self_review_of_preflight_report"
                if single_operator
                else "independent_review_of_preflight_report"
            ),
            "publish_exactly_once",
            "recompute_next_action",
        ]
        outputs = list(action.get("output_paths", []))
        staged_outputs = (staging, preflight, receipt)
        action["output_paths"] = [
            *staged_outputs,
            *(path for path in outputs if path not in staged_outputs),
        ]
    else:
        action.setdefault("staged_manifest_build_argv", None)
        action.setdefault("staged_manifest_build_text", None)
        action.setdefault("staged_manifest_path", None)
        action.setdefault("staged_manifest_build_note", None)
        action.setdefault("post_acquisition_verification_argv", None)
        action.setdefault("post_acquisition_verification_text", None)
        action.setdefault("preflight_report_path", None)
        action.setdefault("staged_review_argv", None)
        action.setdefault("staged_review_text", None)
        action.setdefault("review_receipt_path", None)
        action.setdefault("two_person_publication_required", False)
        action.setdefault("reviewer_identity_placeholder", None)
        action.setdefault("publisher_identity_placeholder", None)
        action.setdefault("independent_review_required_before_publication", False)
        action.setdefault("self_review_required_before_publication", False)
        action.setdefault("claim_bearing_publication_argv", None)
        action.setdefault("claim_bearing_publication_text", None)
        action.setdefault("operator_sequence", [])

    result["action"] = action
    result.pop("evidence_sha256", None)
    result.pop("status_sha256", None)
    result["evidence_sha256"] = next_action_evidence_sha256(result)
    result["status_sha256"] = next_action_status_sha256(result)
    return result


def build_preacquisition_operator_next_action(
    repository_root: str | Path,
    dataset_root: str | Path,
    *,
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    """Build one decision with explicit staging and publication boundaries."""

    return enrich_preacquisition_next_action(
        _build_next_action(
            repository_root,
            dataset_root,
            verify_file_hashes=verify_file_hashes,
        )
    )


def derive_preacquisition_operator_next_action(
    readiness: Mapping[str, Any],
    source_panel: Mapping[str, Any],
    *,
    repository_root: str | Path,
    dataset_root: str | Path,
) -> dict[str, Any]:
    """Derive an operator-sequenced decision from validated status snapshots."""

    return enrich_preacquisition_next_action(
        _derive_next_action(
            readiness,
            source_panel,
            repository_root=repository_root,
            dataset_root=dataset_root,
        )
    )


def render_preacquisition_operator_next_action_markdown(
    decision: Mapping[str, Any],
) -> str:
    """Render the explicit operator sequence without collapsing publication."""

    action = decision.get("action")
    if not isinstance(action, Mapping):
        raise ValueError("next-action decision has no action object")
    self_review = action.get("self_review_required_before_publication") is True
    lines = [
        "# Causal4D pre-acquisition next action",
        "",
        f"- Protocol: `{decision['protocol_id']}`",
        f"- Valid: `{str(decision['valid']).lower()}`",
        f"- Ready: `{str(decision['ready']).lower()}`",
        "- Target outcomes permitted: `false`",
        "",
        f"## {action['title']}",
        "",
        f"- Action ID: `{action['action_id']}`",
        f"- Operator role: `{action['operator_role']}`",
        "- Physical acquisition required: "
        f"`{str(action['physical_acquisition_required']).lower()}`",
    ]
    sections = (
        ("Command", "command_text", None),
        (
            "Build staged manifest",
            "staged_manifest_build_text",
            "staged_manifest_build_note",
        ),
        ("Verify staged evidence", "post_acquisition_verification_text", None),
        ("Review and seal receipt", "staged_review_text", None),
        (
            (
                "Publish after registered self-review"
                if self_review
                else "Publish after independent review"
            ),
            "claim_bearing_publication_text",
            None,
        ),
        ("Completion check", "completion_check_text", None),
    )
    for heading, field, note_field in sections:
        value = action.get(field)
        if value:
            lines += ["", f"### {heading}", "", "```bash", str(value), "```"]
            if note_field and action.get(note_field):
                lines += ["", str(action[note_field])]
    if self_review:
        lines += [
            "",
            (
                "Publication is claim-bearing and requires a registered self-review "
                "receipt. Review and publication are performed by the same disclosed "
                "operator; no independent review is claimed."
            ),
        ]
    elif action.get("independent_review_required_before_publication") is True:
        lines += [
            "",
            "Publication is claim-bearing and requires independent review by a ",
            "registered reviewer plus publication by a distinct registered person.",
        ]
    blockers = action.get("blocking_items")
    if isinstance(blockers, list) and blockers:
        lines += ["", "### Blocking items", ""]
        lines += [f"- `{item}`" for item in blockers]
    if action.get("action_id") == "stop_independent_verifier_unavailable":
        lines += [
            "",
            "### Permitted governance resolutions",
            "",
            (
                "The current independently attested protocol cannot proceed with "
                "a one-person registry."
            ),
            "",
            (
                "1. Register a real, distinct person as independent verifier, "
                "seal the corrected registry, and recompute the next action."
            ),
            (
                "2. Before any target access or physical acquisition, approve a "
                "separately versioned protocol amendment that removes the "
                "independent-attestation claim while preserving the method, split, "
                "threshold, and reporting locks."
            ),
            "",
            (
                "Do not create aliases, duplicate identities, or multiple operator "
                "IDs for one person. They do not satisfy person-level independence."
            ),
            "",
            (
                "See `docs/independent_verifier_onboarding.md` for the verifier's "
                "bounded role and checklist."
            ),
        ]
    execution = action.get("registered_execution")
    if isinstance(execution, Mapping):
        lines += [
            "",
            "### Registered source execution",
            "",
            f"- Execution: `{execution['execution_id']}`",
            f"- Session: `{execution['session_id']}`",
            f"- Command profile: `{execution['command_profile_id']}`",
        ]
    lines += [
        "",
        "---",
        "",
        f"Evidence SHA-256: `{decision['evidence_sha256']}`  ",
        f"Host-local status SHA-256: `{decision['status_sha256']}`",
        "",
    ]
    return "\n".join(lines)


def write_preacquisition_operator_next_action(
    path: str | Path,
    decision: Mapping[str, Any],
) -> Path:
    """Atomically write one operator-sequenced JSON decision."""

    output = Path(path)
    atomic_write_json(output, dict(decision))
    return output


def write_preacquisition_operator_next_action_markdown(
    path: str | Path,
    decision: Mapping[str, Any],
) -> Path:
    """Atomically write one operator-sequenced Markdown report."""

    output = Path(path)
    atomic_write_text(
        output,
        render_preacquisition_operator_next_action_markdown(decision),
    )
    return output


__all__ = [
    "NEXT_ACTION_SCHEMA_VERSION",
    "OPERATOR_FLOW_SCHEMA_VERSION",
    "build_preacquisition_operator_next_action",
    "derive_preacquisition_operator_next_action",
    "enrich_preacquisition_next_action",
    "render_preacquisition_operator_next_action_markdown",
    "write_preacquisition_operator_next_action",
    "write_preacquisition_operator_next_action_markdown",
]
