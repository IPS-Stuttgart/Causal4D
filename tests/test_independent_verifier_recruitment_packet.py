from __future__ import annotations

from pathlib import Path
from typing import Any

from causal4d.preacquisition_next_action import (
    derive_preacquisition_next_action,
    render_preacquisition_next_action_markdown,
)

ROOT = Path(__file__).resolve().parents[1]
MATERIALS = (
    "docs/independent_verifier_onboarding.md",
    "docs/independent_verifier_invitation_template.md",
    "docs/independent_verifier_self_declaration_template.md",
)


def _prerequisite() -> dict[str, object]:
    return {
        "present": True,
        "valid": True,
        "template": False,
        "identity_pending": False,
        "error": None,
    }


def _gate() -> dict[str, object]:
    return {
        "present": True,
        "valid": True,
        "template": False,
        "identity_pending": False,
        "error": None,
    }


def _readiness() -> dict[str, Any]:
    prerequisites = {
        name: _prerequisite()
        for name in (
            "dataset_protocol",
            "acquisition_schedule",
            "object_registration",
            "slip_pilot",
            "timebase_calibration",
            "contact_registration",
            "operator_registry",
            "operator_identity_bindings",
            "method_freeze",
            "method_freeze_validation",
        )
    }
    prerequisites["operator_registry"]["independent_verifier_available"] = False
    gates = {
        name: _gate()
        for name in (
            "signature_panel_complete",
            "actuator_sync_passed",
            "support_registration_passed",
            "end_to_end_dry_run_passed",
            "software_environment_locked",
        )
    }
    return {
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "verify_file_hashes": True,
        "prerequisites": prerequisites,
        "operational_gates": gates,
        "missing_prerequisites": [],
        "malformed_prerequisites": [],
        "missing_or_template_gates": [],
        "malformed_gates": [],
        "chronology_blockers": [],
        "blockers": [],
        "confirmatory_collection": {
            "manifest_executions": 0,
            "acquired_executions": 0,
            "validated_executions": 0,
            "not_started": True,
        },
        "evidence_sha256": "c" * 64,
        "status_sha256": "d" * 64,
        "valid": True,
        "ready": False,
    }


def _source_panel() -> dict[str, Any]:
    return {
        "protocol_id": "protocol",
        "protocol_design_sha256": "a" * 64,
        "preacquisition_plan_id": "plan",
        "preacquisition_amendment_sha256": "b" * 64,
        "missing_template_ids": [],
        "blockers": [],
        "next_execution": None,
        "evidence_sha256": "e" * 64,
        "status_sha256": "f" * 64,
        "valid": True,
        "complete": True,
        "target_outcomes_used": False,
    }


def _decision(repository: str, dataset: str) -> dict[str, Any]:
    return derive_preacquisition_next_action(
        _readiness(),
        _source_panel(),
        repository_root=repository,
        dataset_root=dataset,
    )


def _normalized(relative: str) -> str:
    return " ".join((ROOT / relative).read_text(encoding="utf-8").split())


def test_verifier_stop_packet_surfaces_only_blank_recruitment_materials() -> None:
    decision = _decision("/repo", "/data/run")
    action = decision["action"]

    assert action["action_id"] == "stop_independent_verifier_unavailable"
    assert action["category"] == "governance_blocker"
    assert action["automatable"] is False
    assert action["physical_acquisition_required"] is False
    assert action["target_outcomes_permitted"] is False
    assert action["changes_registered_method"] is False
    assert action["command_argv"] is None
    assert action["command_text"] is None
    assert action["after_completion_argv"] is None
    assert action["output_paths"] == []
    assert action["input_paths"] == [f"/repo/{path}" for path in MATERIALS]
    assert action["blocking_items"] == [
        "single_operator_project_cannot_satisfy_independent_verification"
    ]
    assert decision["ready"] is False
    assert decision["target_outcomes_used"] is False

    markdown = render_preacquisition_next_action_markdown(decision)
    assert "### Input materials" in markdown
    for path in action["input_paths"]:
        assert f"`{path}`" in markdown
    assert "Physical acquisition required: `false`" in markdown


def test_verifier_stop_packet_identity_is_mount_invariant() -> None:
    first = _decision("/repo", "/data/run")
    second = _decision("/different/repo", "/different/data")

    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["status_sha256"] != second["status_sha256"]


def test_recruitment_documents_preserve_the_governance_boundary() -> None:
    for relative in MATERIALS:
        path = ROOT / relative
        assert path.is_file()
        assert not path.is_symlink()

    invitation = _normalized("docs/independent_verifier_invitation_template.md")
    declaration = _normalized("docs/independent_verifier_self_declaration_template.md")
    onboarding = _normalized("docs/independent_verifier_onboarding.md")

    for marker in (
        "current evidence state is `0/36` acquired",
        "would not authorize acquisition by itself",
        "A second account, alias, service identity, bot, or unattended workflow",
        "Do not commit the completed message",
    ):
        assert marker in invitation

    for marker in (
        (
            "This is a private onboarding aid, not a repository artifact and "
            "not an attestation"
        ),
        "I am a real person",
        "I am not FlorianPfaff",
        "did not use confirmatory target outcomes",
        "does not by itself authorize physical work",
        "must remain outside the repository and acquisition dataset",
    ):
        assert marker in declaration

    for marker in (
        "independent_verifier_invitation_template.md",
        "independent_verifier_self_declaration_template.md",
        "Receiving a private declaration does not itself register the candidate",
        "stop_independent_verifier_unavailable",
        "Confirmatory execution 1 remains forbidden",
    ):
        assert marker in onboarding
