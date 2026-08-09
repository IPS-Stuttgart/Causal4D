from __future__ import annotations

import re
from pathlib import Path


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW = (
    _REPOSITORY_ROOT
    / ".github"
    / "workflows"
    / "tree-block-query-provider-integration.yml"
)
_QUERY_PIN = (
    _REPOSITORY_ROOT
    / "requirements"
    / "ci"
    / "bayesian-phystwin-tree-block-query-v1.sha"
)
_HISTORICAL_PIN = (
    _REPOSITORY_ROOT / "requirements" / "ci" / "bayesian-phystwin-provider-v1.sha"
)


def _pin(path: Path) -> str:
    value = path.read_text(encoding="utf-8").strip()
    assert re.fullmatch(r"[0-9a-f]{40}", value)
    return value


def test_tree_block_query_workflow_uses_exact_dedicated_provider_pin() -> None:
    query_pin = _pin(_QUERY_PIN)
    workflow = _WORKFLOW.read_text(encoding="utf-8")

    assert f"ref: {query_pin}" in workflow
    assert f"BAYESIAN_PHYSTWIN_REVISION: {query_pin}" in workflow
    assert "repository: IPS-Stuttgart/BayesianPhysTwin" in workflow
    assert "persist-credentials: false" in workflow
    assert 'CAUSAL4D_REQUIRE_TREE_BLOCK_QUERY_PROVIDER: "1"' in workflow
    assert "tests/test_bpt_tree_block_query_provider_integration.py" in workflow


def test_new_provider_pin_does_not_rewrite_historical_compatibility_pin() -> None:
    assert _pin(_QUERY_PIN) != _pin(_HISTORICAL_PIN)
