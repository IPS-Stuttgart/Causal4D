"""Policy checks for the permanent exact-head validation workflow."""

from pathlib import Path


WORKFLOW = Path(".github/workflows/exact-head-validation.yml")


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def _job_section(name: str, next_name: str | None = None) -> str:
    text = _workflow_text()
    section = text.split(f"\n  {name}:\n", 1)[1]
    if next_name is not None:
        section = section.split(f"\n  {next_name}:\n", 1)[0]
    return section


def test_exact_head_validation_has_manual_and_scheduled_queue() -> None:
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    assert "schedule:" in text
    assert '<!-- exact-head-validation: queued -->' in text
    assert "same-repository PRs only" in text
    assert 'pull["base"]["ref"] != "main"' in text
    assert 'WORKFLOW_REF"] != "refs/heads/main"' in text
    assert "merge_base_commit" in text
    assert "compare/{pull['base']['sha']}...{pull['head']['sha']}" in text


def test_exact_head_checkout_is_shallow_immutable_and_uncredentialed() -> None:
    text = _workflow_text()
    assert "fetch-depth: 0" not in text
    assert text.count("fetch-depth: 1") == 2
    assert text.count("- name: Fetch exact comparison commits") == 2
    assert text.count('git -c http.sslCAInfo="$ca_file" fetch') == 2
    assert text.count('--no-tags --depth=1 origin "$sha"') == 2
    assert text.count('git cat-file -e "$EXPECTED_BASE_SHA^{commit}"') >= 2
    assert (
        text.count('git cat-file -e "$EXPECTED_MERGE_BASE_SHA^{commit}"') >= 2
    )
    assert '--merge-base "$EXPECTED_MERGE_BASE_SHA"' in text
    assert text.count("persist-credentials: false") >= 3
    assert text.count("git rev-parse HEAD") >= 2
    assert text.count("pull-request head changed during exact-head validation") >= 2
    assert text.count("pull-request base changed during exact-head validation") >= 2
    assert "git merge-tree --write-tree" in text
    assert "pull_request_target" not in text
    assert "contents: write" not in text


def test_exact_head_write_permissions_are_attestation_only() -> None:
    text = _workflow_text()
    header = text.split("\njobs:\n", 1)[0]
    assert "permissions: {}" in header
    assert "pull-requests: write" not in header
    assert "issues: write" not in header

    for section in (
        _job_section("select", "core"),
        _job_section("core", "quality-package-provider"),
        _job_section("quality-package-provider", "attest"),
    ):
        assert "contents: read" in section
        assert "pull-requests: read" in section
        assert "pull-requests: write" not in section
        assert "issues: write" not in section

    attest = _job_section("attest")
    assert "contents: read" in attest
    assert "pull-requests: write" in attest
    assert "issues: write" in attest
    assert text.count("pull-requests: write") == 1
    assert text.count("issues: write") == 1


def test_exact_head_validation_matches_authoritative_stack() -> None:
    text = _workflow_text()
    for version in ('"3.10"', '"3.12"', '"3.14"'):
        assert version in text
    for command in (
        "python -W error::SyntaxWarning -m compileall",
        "python -m pytest",
        "python -m ruff check .",
        "python -m ruff format --check",
        "python -m mypy --python-version 3.12",
        "python -m build",
        "python -m twine check dist/*",
        "python scripts/ci/read_bpt_pin.py",
        "scripts/ci/run_bpt_integration_tests.py",
    ):
        assert command in text


def test_exact_head_result_is_bound_and_one_shot() -> None:
    text = _workflow_text()
    assert '"artifact_kind": "Causal4DExactHeadValidation"' in text
    assert '"head_sha": expected_head' in text
    assert '"base_ref": os.environ["EXPECTED_BASE_REF"]' in text
    assert '"base_sha": os.environ["EXPECTED_BASE_SHA"]' in text
    assert '"merge_base_sha": os.environ["EXPECTED_MERGE_BASE_SHA"]' in text
    assert '"head_still_current": head_current' in text
    assert '"base_still_current": base_current' in text
    assert "and base_current" in text
    assert "exact-head-validation.json" in text
    assert "body.replace(marker, completion, 1)" in text
