from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release-metadata-integrity.yml"
CHECKER = ROOT / "scripts" / "ci" / "check_release_metadata.py"


def _workflow_text() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_release_integrity_runs_for_metadata_changes_and_tags() -> None:
    text = _workflow_text()

    assert "name: Release metadata integrity" in text
    assert "pull_request:" in text
    assert "push:" in text
    assert 'tags: ["v*"]' in text
    assert "workflow_dispatch:" in text
    for path in (
        ".github/workflows/release-metadata-integrity.yml",
        "CHANGELOG.md",
        "CITATION.cff",
        "ci/project_status_v2.json",
        "pyproject.toml",
        "scripts/ci/check_release_metadata.py",
        "src/causal4d/__init__.py",
        "tests/test_release_metadata_policy.py",
        "tests/test_release_metadata_workflow_policy.py",
    ):
        assert text.count(f'"{path}"') == 2


def test_release_integrity_is_read_only_and_pins_external_actions() -> None:
    text = _workflow_text()

    assert "permissions:\n  contents: read\n" in text
    assert "contents: write" not in text
    assert "packages: write" not in text
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("uses:"):
            continue
        reference = stripped.rsplit("@", maxsplit=1)[1].split()[0]
        assert len(reference) == 40
        assert all(character in "0123456789abcdef" for character in reference)


def test_release_integrity_validates_tag_and_built_artifact_versions() -> None:
    text = _workflow_text()

    assert "check_release_metadata.py" in text
    assert 'if [[ "${GITHUB_REF_TYPE}" == "tag" ]]' in text
    assert 'arguments+=(--tag "${GITHUB_REF_NAME}")' in text
    assert "python -m build --sdist --wheel" in text
    assert "python -m twine check dist/*" in text
    assert "parse_wheel_filename" in text
    assert "parse_sdist_filename" in text
    assert 'wheel_metadata["Version"]' in text
    assert 'sdist_metadata["Version"]' in text
    assert 'version("causal4d")' in text
    assert "causal4d.__version__" in text
    assert "distribution-sha256.txt" in text
    assert "causal4d-release-metadata-integrity" in text


def test_release_checker_fails_closed_on_unreleased_tagging() -> None:
    text = CHECKER.read_text(encoding="utf-8")

    assert "tagged releases require an empty Unreleased changelog section" in text
    assert 'expected_tag = f"v{package_version}"' in text
    assert "development/prerelease package versions cannot be published" in text
    assert "project-status required_version does not match" in text
    assert 'attribute != "causal4d.__version__"' in text
