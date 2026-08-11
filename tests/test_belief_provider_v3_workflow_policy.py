from __future__ import annotations

from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    _REPOSITORY_ROOT / ".github" / "workflows" / "belief-provider-v3-integration.yml"
)
BPT_BELIEF_PROVIDER_V3_REVISION = "62dff353903dcad273ffcd96644e3c2b3f9e5fd1"
CAUSAL4D_HEAD_REF = "${{ github.event.pull_request.head.sha || github.sha }}"


def test_belief_provider_v3_workflow_uses_exact_public_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert f"ref: {BPT_BELIEF_PROVIDER_V3_REVISION}" in text
    assert "persist-credentials: false" in text
    assert "BPT_READ_SSH_KEY" not in text


def test_belief_provider_v3_workflow_records_exact_causal4d_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert f"ref: {CAUSAL4D_HEAD_REF}" in text
    assert f"EXPECTED_CAUSAL4D_SHA: {CAUSAL4D_HEAD_REF}" in text
    assert 'actual_sha="$(git rev-parse HEAD)"' in text
    assert 'test "$actual_sha" = "$EXPECTED_CAUSAL4D_SHA"' in text


def test_belief_provider_v3_workflow_exercises_installed_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m build --wheel" in text
    assert "belief-provider-v3-wheelhouse" in text
    assert "belief-provider-v3-venv" in text
    assert "--import-mode=importlib" in text
    assert "env -u PYTHONPATH" in text
    assert 'CAUSAL4D_REQUIRE_BPT_BELIEF_PROVIDER_V3: "1"' in text
    assert "test_belief_provider_v2_contract.py" in text
    assert "test_belief_provider_v3_contract.py" in text
    assert "test_bpt_provider_registry.py" in text


def test_belief_provider_v3_workflow_is_hosted_and_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "permissions:\n  contents: read\n" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text


def test_belief_provider_v3_workflow_runs_contract_before_quality() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    contract_index = text.index(
        "- name: Exercise the installed cross-repository contract"
    )
    quality_index = text.index("- name: Check Causal4D source quality")

    assert contract_index < quality_index
    assert "if: always()" in text[quality_index:]
    assert "python -m mypy" in text[quality_index:]


def test_belief_provider_v3_workflow_pins_external_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1") == 2
    assert "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97" in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:"):
            assert "@" in stripped
            reference = stripped.rsplit("@", 1)[1].split()[0]
            assert len(reference) == 40
            assert all(character in "0123456789abcdef" for character in reference)
