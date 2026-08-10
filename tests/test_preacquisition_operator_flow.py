from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

import causal4d.preacquisition_operator_flow as operator_flow
from causal4d.preacquisition_next_action import (
    next_action_evidence_sha256,
    next_action_status_sha256,
)


def _source_decision() -> dict:
    repository = "/opt/causal4d-frozen"
    dataset = "/data/causal4d"
    execution_id = "source-lift_high-r1"
    staging = f"{dataset}/staging/{execution_id}.json"
    publication = [
        "causal4d",
        "protocol",
        "readiness",
        "source-panel-publish",
        repository,
        dataset,
        staging,
    ]
    decision = {
        "schema_version": 1,
        "artifact_kind": "Causal4DPreacquisitionNextAction",
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "repository_root": repository,
        "dataset_root": dataset,
        "readiness_evidence_sha256": "c" * 64,
        "readiness_status_sha256": "d" * 64,
        "source_panel_evidence_sha256": "e" * 64,
        "source_panel_status_sha256": "f" * 64,
        "readiness_valid": True,
        "source_panel_valid": True,
        "ready": False,
        "complete": False,
        "passed": False,
        "valid": True,
        "target_outcomes_used": False,
        "action": {
            "action_id": "acquire_next_source_panel_execution",
            "category": "physical_source_execution",
            "title": f"Acquire registered source execution {execution_id}",
            "operator_role": "acquisition_operator",
            "physical_acquisition_required": True,
            "automatable": False,
            "changes_registered_method": False,
            "target_outcomes_permitted": False,
            "command_argv": [
                "causal4d",
                "protocol",
                "readiness",
                "source-panel-status",
                repository,
                dataset,
                "--verify-file-hashes",
            ],
            "command_text": (
                "causal4d protocol readiness source-panel-status "
                f"{repository} {dataset} --verify-file-hashes"
            ),
            "completion_check_argv": [
                "causal4d",
                "protocol",
                "readiness",
                "next-action",
                repository,
                dataset,
            ],
            "completion_check_text": (
                f"causal4d protocol readiness next-action {repository} {dataset}"
            ),
            "after_completion_argv": publication,
            "after_completion_text": " ".join(publication),
            "input_paths": [f"{dataset}/template.json"],
            "output_paths": [staging, f"{dataset}/final.json"],
            "blocking_items": [],
            "registered_execution": {
                "execution_id": execution_id,
                "session_id": execution_id,
                "command_profile_id": "lift_high",
            },
        },
    }
    decision["evidence_sha256"] = next_action_evidence_sha256(decision)
    decision["status_sha256"] = next_action_status_sha256(decision)
    return decision


def test_source_action_inserts_staging_preflight_and_independent_review() -> None:
    result = operator_flow.enrich_preacquisition_next_action(_source_decision())

    action = result["action"]
    assert result["schema_version"] == 3
    assert result["operator_flow_schema_version"] == 2
    assert action["after_completion_argv"] is None
    assert action["after_completion_text"] is None
    assert action["staged_manifest_build_argv"][3] == "source-panel-stage"
    assert action["staged_manifest_build_argv"].count("--artifact") == 1
    assert action["staged_manifest_path"].endswith(
        "staging/source-lift_high-r1.json"
    )
    assert "Repeat --artifact" in action["staged_manifest_build_note"]
    assert action["post_acquisition_verification_argv"][3] == (
        "source-panel-verify-staged"
    )
    assert action["post_acquisition_verification_argv"][-2] == "--output-json"
    assert action["preflight_report_path"].endswith(
        "source-lift_high-r1-preflight.json"
    )
    assert action["independent_review_required_before_publication"] is True
    assert action["claim_bearing_publication_argv"][3] == "source-panel-publish"
    assert action["operator_sequence"] == [
        "acquire_registered_source_execution",
        "build_staged_manifest_from_registered_template_and_artifact_bytes",
        "verify_staged_manifest_and_artifacts",
        "independent_review_of_preflight_report",
        "publish_exactly_once",
        "recompute_next_action",
    ]
    assert action["staged_manifest_path"] in action["output_paths"]
    assert action["preflight_report_path"] in action["output_paths"]
    assert result["evidence_sha256"] == next_action_evidence_sha256(result)
    assert result["status_sha256"] == next_action_status_sha256(result)


def test_non_source_action_gets_explicit_empty_publication_boundary() -> None:
    decision = _source_decision()
    decision["action"] = {
        **decision["action"],
        "action_id": "seal_method_freeze",
        "category": "freeze",
        "after_completion_argv": None,
        "after_completion_text": None,
        "registered_execution": None,
    }

    result = operator_flow.enrich_preacquisition_next_action(decision)

    action = result["action"]
    assert action["staged_manifest_build_argv"] is None
    assert action["staged_manifest_path"] is None
    assert action["post_acquisition_verification_argv"] is None
    assert action["claim_bearing_publication_argv"] is None
    assert action["independent_review_required_before_publication"] is False
    assert action["operator_sequence"] == []


def test_source_action_rejects_drifted_publication_command() -> None:
    decision = _source_decision()
    decision["action"]["after_completion_argv"][-1] = "/wrong/staging.json"

    with pytest.raises(ValueError, match="publication command differs"):
        operator_flow.enrich_preacquisition_next_action(decision)


def test_source_action_requires_registered_execution_identity() -> None:
    decision = _source_decision()
    decision["action"].pop("registered_execution")

    with pytest.raises(ValueError, match="registered source execution is missing"):
        operator_flow.enrich_preacquisition_next_action(decision)


def test_markdown_orders_staging_verification_and_publication() -> None:
    result = operator_flow.enrich_preacquisition_next_action(_source_decision())

    markdown = operator_flow.render_preacquisition_operator_next_action_markdown(result)

    staging = markdown.index("### Build staged manifest")
    verification = markdown.index("### Verify staged evidence")
    publication = markdown.index("### Publish after independent review")
    completion = markdown.index("### Completion check")
    assert staging < verification < publication < completion
    assert "Repeat --artifact" in markdown
    assert "requires independent review" in markdown
    assert result["evidence_sha256"] in markdown


def test_operator_flow_hash_is_mount_independent() -> None:
    first = operator_flow.enrich_preacquisition_next_action(_source_decision())
    second = deepcopy(first)
    old_repository = second["repository_root"]
    old_dataset = second["dataset_root"]
    second["repository_root"] = "/relocated/repository"
    second["dataset_root"] = "/relocated/dataset"

    def relocate(value):
        if isinstance(value, dict):
            return {key: relocate(item) for key, item in value.items()}
        if isinstance(value, list):
            return [relocate(item) for item in value]
        if isinstance(value, str):
            return value.replace(old_repository, second["repository_root"]).replace(
                old_dataset, second["dataset_root"]
            )
        return value

    second["action"] = relocate(second["action"])
    second.pop("evidence_sha256")
    second.pop("status_sha256")

    assert next_action_evidence_sha256(first) == next_action_evidence_sha256(second)


def test_build_wrapper_enriches_underlying_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        operator_flow,
        "_build_next_action",
        lambda *args, **kwargs: _source_decision(),
    )

    result = operator_flow.build_preacquisition_operator_next_action(
        "/repo",
        "/data",
    )

    assert result["action"]["staged_manifest_build_argv"] is not None
    assert result["action"]["post_acquisition_verification_argv"] is not None


def test_json_and_markdown_writers_are_atomic(tmp_path: Path) -> None:
    result = operator_flow.enrich_preacquisition_next_action(_source_decision())
    json_path = tmp_path / "operator" / "next.json"
    markdown_path = tmp_path / "operator" / "next.md"

    assert (
        operator_flow.write_preacquisition_operator_next_action(
            json_path,
            result,
        )
        == json_path
    )
    assert (
        operator_flow.write_preacquisition_operator_next_action_markdown(
            markdown_path,
            result,
        )
        == markdown_path
    )
    assert json_path.is_file()
    assert markdown_path.is_file()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "Build staged manifest" in markdown
    assert "Verify staged evidence" in markdown
