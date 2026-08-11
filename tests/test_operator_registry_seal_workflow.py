from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "seal-operator-registry-self-hosted.yml"
SCRIPT = ROOT / "scripts" / "ci" / "seal_self_hosted_operator_registry.py"
SELF_HOSTED_REGISTRY = ROOT / ".github" / "self-hosted-jobs.json"
TRIGGER = "[self-hosted] seal Causal4D operator registry"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_operator_registry_seal_has_exact_maintainer_guard() -> None:
    text = _workflow_text()

    assert "workflow_dispatch:" in text
    assert "issues:\n    types: [opened]" in text
    assert "github.ref == 'refs/heads/main'" in text
    assert "github.event_name == 'issues'" in text
    assert "github.event.action == 'opened'" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert f"'{TRIGGER}'" in text
    for forbidden in (
        "github.event.issue.body",
        "github.event.issue.labels",
        "github.event.comment",
    ):
        assert forbidden not in text


def test_operator_registry_seal_uses_fixed_local_paths_and_no_secrets() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read\n" in text
    assert "${{ secrets." not in text
    assert "/mnt/lexar4tb/causal4d-physical/causal4d-frozen" in text
    assert "/mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1" in text
    assert "/mnt/lexar4tb/causal4d-physical/private/operator-registry-v1" in text
    assert "scripts/ci/seal_self_hosted_operator_registry.py" in text
    assert "ref: ${{ github.sha }}" in text
    assert "persist-credentials: false" in text
    assert 'test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"' in text
    assert 'test -z "$(git status --porcelain=v1)"' in text


def test_operator_registry_seal_uploads_only_sanitized_evidence() -> None:
    text = _workflow_text()
    upload_start = text.index("- name: Upload sanitized seal evidence")
    cleanup_start = text.index(
        "- name: Remove isolated environment and wheel",
        upload_start,
    )
    upload = text[upload_start:cleanup_start]

    assert "report.json" in upload
    assert "runner.txt" in upload
    assert "wheel-sha256.txt" in upload
    assert "operator-identity-hmac-v1.key" not in upload
    assert "operator-principals-v1.json" not in upload
    assert "operator_registry.template.json" not in upload
    assert "operator_registry.json" not in upload
    assert "/private/" not in upload


def test_operator_registry_seal_keeps_the_nonphysical_claim_boundary() -> None:
    text = _workflow_text()
    script = SCRIPT.read_text(encoding="utf-8")

    for required in (
        "Target outcomes used: \\`false\\`",
        "Device nodes opened: \\`false\\`",
        "Physical command sent: \\`false\\`",
        "Physical evidence increment:",
    ):
        assert required in text
    assert '"target_outcomes_used": False' in script
    assert '"device_nodes_opened": False' in script
    assert '"physical_command_sent": False' in script
    assert '"registered_method_changed": False' in script
    assert '"physical_evidence_increment": 0' in script


def test_operator_registry_seal_pins_all_external_actions() -> None:
    text = _workflow_text()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.rsplit("@", 1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_operator_registry_seal_is_registered_as_self_hosted() -> None:
    payload = json.loads(SELF_HOSTED_REGISTRY.read_text(encoding="utf-8"))
    matches = [
        entry
        for entry in payload["jobs"]
        if entry["workflow"] == "seal-operator-registry-self-hosted.yml"
        and entry["job"] == "seal"
    ]

    assert len(matches) == 1
    entry = matches[0]
    assert entry["authorization_model"] == "main-only"
    assert entry["secrets_allowed"] is False
    assert entry["dataset_access"] == (
        "registered-local-preacquisition-operator-registry-write"
    )
