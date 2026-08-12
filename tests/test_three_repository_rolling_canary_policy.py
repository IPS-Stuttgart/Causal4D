from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = (
    ROOT / ".github" / "workflows" / "three-repository-rolling-canary.yml"
)


def _text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_canary_is_scheduled_manual_and_non_claim_bearing() -> None:
    text = _text()

    assert "name: Rolling three-repository compatibility canary" in text
    assert "  workflow_dispatch:" in text
    assert "  schedule:" in text
    assert "pull_request:" not in text
    assert "push:" not in text
    assert "issues:" not in text
    assert "CAUSAL4D_COMPATIBILITY_LANE: rolling-non-claim-bearing" in text
    assert '"claim_bearing": False' in text
    assert '"frozen_pins_used": False' in text
    assert "does not update frozen pins" in text


def test_canary_follows_all_three_current_default_branches() -> None:
    text = _text()

    assert "repository: IPS-Stuttgart/Causal4D" in text
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in text
    assert "repository: IPS-Stuttgart/Prob4D" in text
    assert text.count("          ref: main\n") == 3
    assert "bayesian-phystwin-three-repository.sha" not in text
    assert "prob4d-three-repository.sha" not in text
    assert "Record exact current-head revisions" in text
    assert "rolling-three-repository-revisions.json" in text


def test_canary_builds_installs_and_exercises_the_contract_stack() -> None:
    text = _text()

    assert "Build all current-head wheels" in text
    assert text.count("python -m build --wheel") == 3
    assert "Install current-head wheels in isolation" in text
    assert "Verify imports originate only from installed wheels" in text
    assert "source-tree import detected" in text
    assert "Create and verify an ephemeral current-head stack lock" in text
    assert "three_repository_golden_path.py" in text
    assert "three_repository_provider_v2_attestation.py" in text
    assert "Run focused current-head contract tests" in text
    assert "test_claim_bearing_observation.py" in text
    assert "test_prob4d_causal_lineage.py" in text
    assert "test_decision_trace.py" in text
    assert "three-repository-rolling-canary" in text


def test_canary_pins_external_actions_and_has_read_only_permissions() -> None:
    text = _text()

    assert "permissions:\n  contents: read\n" in text
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.rsplit("@", maxsplit=1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)
