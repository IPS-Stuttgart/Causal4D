from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "acquisition-readiness-self-hosted.yml"
DOCUMENTATION = ROOT / "docs" / "self_hosted_acquisition_readiness.md"

CANONICAL_V5 = "/data/causal4d-sloth-multi-action-v1-v5"
PERSISTENT_V5 = (
    "/mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1-v5"
)
CANONICAL_HISTORICAL = "/data/causal4d-sloth-multi-action-v1"
PERSISTENT_HISTORICAL = (
    "/mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1"
)


def _workflow_paths() -> set[str]:
    paths: set[str] = set()
    for line in WORKFLOW.read_text(encoding="utf-8").splitlines():
        value = line.strip()
        if value.endswith("\\"):
            value = value[:-1].strip()
        if value.startswith("/"):
            paths.add(value)
    return paths


def test_readiness_inspects_only_authoritative_v5_dataset_roots() -> None:
    paths = _workflow_paths()

    assert CANONICAL_V5 in paths
    assert PERSISTENT_V5 in paths
    assert CANONICAL_HISTORICAL not in paths
    assert PERSISTENT_HISTORICAL not in paths


def test_readiness_trigger_and_read_only_boundary_are_unchanged() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "[self-hosted] inspect Causal4D acquisition readiness" in text
    assert "github.event.issue.user.login == 'FlorianPfaff'" in text
    assert "github.event.issue.user.id == 6773539" in text
    assert "permissions:\n  contents: read" in text
    assert "scripts/ci/probe_self_hosted_acquisition.py" in text
    assert "- Physical evidence increment: \\`0\\`" in text


def test_readiness_documentation_matches_v5_root_binding() -> None:
    text = DOCUMENTATION.read_text(encoding="utf-8")

    assert f"`{CANONICAL_V5}`" in text
    assert f"`{PERSISTENT_V5}`" in text
    assert f"`{CANONICAL_HISTORICAL}` |" not in text
    assert f"`{PERSISTENT_HISTORICAL}` |" not in text
    assert "historical non-v5 dataset is" in text
    assert "not an admissible readiness candidate" in text
