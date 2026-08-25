from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT
    / ".github"
    / "workflows"
    / "bootstrap-single-operator-v5-self-hosted.yml"
)
SCRIPT = ROOT / "scripts" / "ci" / "bootstrap_self_hosted_v5_operator_scaffold.py"
SELF_HOSTED_REGISTRY = ROOT / ".github" / "self-hosted-jobs.json"
TRIGGER = "[self-hosted] bootstrap Causal4D v5 operator scaffold"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_v5_bootstrap_has_one_exact_maintainer_issue_trigger() -> None:
    text = _workflow_text()

    assert "issues:\n    types: [opened]" in text
    assert "workflow_dispatch:" not in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert f"'{TRIGGER}'" in text
    for forbidden in (
        "github.event.issue.body",
        "github.event.issue.labels",
        "github.event.comment",
        "${{ secrets.",
    ):
        assert forbidden not in text


def test_v5_bootstrap_uses_fresh_fixed_root_and_exact_wheel() -> None:
    text = _workflow_text()

    assert (
        "/mnt/lexar4tb/causal4d-physical/"
        "causal4d-sloth-multi-action-v1\n"
    ) in text
    assert (
        "/mnt/lexar4tb/causal4d-physical/"
        "causal4d-sloth-multi-action-v1-v5\n"
    ) in text
    assert "scripts/ci/bootstrap_self_hosted_v5_operator_scaffold.py" in text
    assert '--repository-root "${GITHUB_WORKSPACE}"' in text
    assert "ref: ${{ github.sha }}" in text
    assert "persist-credentials: false" in text
    assert "PYTHONPATH=\"\"" in text
    assert "installed import" in text


def test_v5_bootstrap_upload_is_sanitized_and_nonphysical() -> None:
    text = _workflow_text()
    script = SCRIPT.read_text(encoding="utf-8")

    upload = text[
        text.index("- name: Upload sanitized bootstrap evidence") : text.index(
            "- name: Remove isolated environment and wheel"
        )
    ]
    for allowed in (
        "report.json",
        "report.txt",
        "runner.txt",
        "nvidia-smi-L.txt",
        "wheel-sha256.txt",
    ):
        assert allowed in upload
    for forbidden in (
        "operator_registry.json",
        "operator_registry.template.json",
        "single_operator_v5_bootstrap.json",
        "person_identity_sha256",
        "object_registration",
        "executions/",
        "sessions/",
    ):
        assert forbidden not in upload

    assert '"target_outcomes_used": False' in script
    assert '"device_nodes_opened": False' in script
    assert '"physical_command_sent": False' in script
    assert '"registered_method_changed": False' in script
    assert '"physical_evidence_increment": 0' in script
    assert "target_outcomes_permitted" in text
    assert "does not authorize target access" in text


def test_v5_bootstrap_pins_all_external_actions() -> None:
    for line in _workflow_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.rsplit("@", 1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_v5_bootstrap_is_registered_as_self_hosted() -> None:
    payload = json.loads(SELF_HOSTED_REGISTRY.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in payload["jobs"]
        if entry["workflow"]
        == "bootstrap-single-operator-v5-self-hosted.yml"
        and entry["job"] == "bootstrap"
    ]

    assert len(matches) == 1
    entry = matches[0]
    assert entry["authorization_model"] == "maintainer-issue-main"
    assert entry["secrets_allowed"] is False
    assert entry["dataset_access"] == (
        "registered-local-preacquisition-v5-scaffold-bootstrap"
    )
