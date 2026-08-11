from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "merge-gate.yml"
BPT_PIN = (
    (ROOT / "requirements" / "ci" / "bayesian-phystwin-provider-v1.sha")
    .read_text(encoding="utf-8")
    .strip()
)


def test_merge_gate_has_one_stable_pull_request_status() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert text.count("name: Causal4D merge gate") == 2
    assert "pull_request:" in text
    assert "branches: [main]" in text
    assert "  merge-gate:" in text


def test_merge_gate_validates_the_pull_request_merge_result() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    checkout_index = text.index("- name: Check out pull-request merge result")
    record_index = text.index("- name: Record tested merge and head revisions")
    checkout_block = text[checkout_index:record_index]

    assert "ref: ${{ github.sha }}" in checkout_block
    assert "fetch-depth: 1" in checkout_block
    assert "fetch-depth: 0" not in checkout_block
    assert "persist-credentials: false" in checkout_block
    assert "EXPECTED_MERGE_SHA: ${{ github.sha }}" in text
    assert "PR_HEAD_SHA: ${{ github.event.pull_request.head.sha }}" in text
    assert 'actual_merge_sha="$(git rev-parse HEAD)"' in text
    assert 'test "$actual_merge_sha" = "$EXPECTED_MERGE_SHA"' in text


def test_merge_gate_covers_quality_tests_build_and_provider_integration() -> None:
    text = WORKFLOW.read_text(encoding="utf-8")

    assert 'PYTHONPYCACHEPREFIX="$RUNNER_TEMP/python-syntax-cache"' in text
    assert (
        "python -W error::SyntaxWarning -m compileall -q -f src tests scripts" in text
    )
    assert "python -m ruff check ." in text
    assert "python -m mypy --python-version 3.12" in text
    assert "src/causal4d/belief_provider_v2_contract.py" in text
    assert "python -m pytest --junitxml=pytest-merge-gate.xml" in text
    assert "python -m build" in text
    assert "python -m twine check dist/*" in text
    assert f"ref: {BPT_PIN}" in text
    assert "python scripts/ci/read_bpt_pin.py" not in text
    assert "steps.pin.outputs.sha" not in text
    assert "scripts/ci/run_bpt_integration_tests.py" in text
    assert 'CAUSAL4D_REQUIRE_BPT_PROVIDER: "1"' in text


def test_merge_gate_pins_external_actions() -> None:
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
