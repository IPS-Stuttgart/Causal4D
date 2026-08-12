from __future__ import annotations

from pathlib import Path


WORKFLOW_PATH = ".github/workflows/recursive-belief-handoff-integration.yml"
WORKFLOW = Path(__file__).resolve().parents[1] / WORKFLOW_PATH
BPT_RECURSIVE_PROVIDER_REVISION = "d2b38ce72a0ecb07cce7e252a0af6fb32c9411dc"
CAUSAL4D_HEAD_REF = "${{ github.event.pull_request.head.sha || github.sha }}"
CHECKOUT_ACTION = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_PYTHON_ACTION = "actions/setup-python@5fda3b95a4ea91299a34e894583c3862153e4b97"


def test_recursive_handoff_workflow_uses_exact_public_provider_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert f"ref: {BPT_RECURSIVE_PROVIDER_REVISION}" in text
    assert "persist-credentials: false" in text
    assert "BPT_READ_SSH_KEY" not in text


def test_recursive_handoff_workflow_records_exact_causal4d_revision() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert f"ref: {CAUSAL4D_HEAD_REF}" in text
    assert f"EXPECTED_CAUSAL4D_SHA: {CAUSAL4D_HEAD_REF}" in text
    assert 'actual_sha="$(git rev-parse HEAD)"' in text
    assert 'test "$actual_sha" = "$EXPECTED_CAUSAL4D_SHA"' in text


def test_recursive_handoff_workflow_exercises_installed_wheels() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "python -m build --wheel" in text
    assert "recursive-belief-handoff-wheelhouse" in text
    assert "recursive-belief-handoff-venv" in text
    assert "--import-mode=importlib" in text
    assert "env -u PYTHONPATH" in text
    assert "test_belief_provider_v2_recursive_contract.py" in text
    assert "test_recursive_bpt_belief_handoff.py" in text
    assert "test_evidence_ownership.py" in text
    assert "test_bpt_provider_import_boundary.py" in text


def test_recursive_handoff_pull_request_validation_is_hosted_read_only() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "pull_request:" in text
    assert "permissions:\n  contents: read\n" in text
    assert "runs-on: ubuntu-latest" in text
    assert "self-hosted" not in text


def test_recursive_handoff_contract_runs_before_local_quality() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    contract_index = text.index(
        "- name: Exercise the installed recursive handoff contract"
    )
    quality_index = text.index("- name: Check Causal4D source quality")

    assert contract_index < quality_index
    assert "if: always()" in text[quality_index:]


def test_recursive_handoff_workflow_pins_external_actions() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count(CHECKOUT_ACTION) == 2
    assert SETUP_PYTHON_ACTION in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("uses:"):
            assert "@" in stripped
            reference = stripped.rsplit("@", 1)[1].split()[0]
            assert len(reference) == 40
            assert all(character in "0123456789abcdef" for character in reference)
