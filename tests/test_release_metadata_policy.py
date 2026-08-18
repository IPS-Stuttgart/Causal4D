from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "ci" / "check_release_metadata.py"
SPEC = importlib.util.spec_from_file_location("check_release_metadata", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)
ReleaseMetadataError = MODULE.ReleaseMetadataError
inspect_repository = MODULE.inspect_repository


def _write_fixture(
    root: Path,
    *,
    package_version: str = "1.2.3",
    citation_version: str | None = None,
    status_version: str | None = None,
    setuptools_attribute: str = "causal4d.__version__",
    changelog: str | None = None,
) -> Path:
    citation_version = citation_version or package_version
    status_version = status_version or package_version
    if changelog is None:
        changelog = f"# Changelog\n\n## Unreleased\n\n## {package_version}\n"

    (root / "src" / "causal4d").mkdir(parents=True)
    (root / "ci").mkdir(parents=True)
    (root / "src" / "causal4d" / "__init__.py").write_text(
        f'__version__ = "{package_version}"\n',
        encoding="utf-8",
    )
    (root / "CITATION.cff").write_text(
        f"cff-version: 1.2.0\ntitle: Causal4D\nversion: {citation_version}\n",
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project]\n"
        'name = "causal4d"\n'
        'dynamic = ["version"]\n\n'
        "[tool.setuptools.dynamic]\n"
        f'version = {{ attr = "{setuptools_attribute}" }}\n',
        encoding="utf-8",
    )
    (root / "ci" / "project_status_v2.json").write_text(
        json.dumps(
            {
                "packages": {
                    "causal4d": {
                        "required_version": status_version,
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "CHANGELOG.md").write_text(changelog, encoding="utf-8")
    return root


def test_current_repository_metadata_is_internally_consistent() -> None:
    summary = inspect_repository(ROOT)

    assert summary.package_version == summary.citation_version
    assert summary.setuptools_version_attribute == "causal4d.__version__"
    assert summary.tag_validated is False
    assert summary.release_ready is (
        not summary.prerelease
        and summary.changelog_heading_present
        and not summary.unreleased_changes_present
        and summary.project_status_required_version == summary.package_version
    )


def test_current_release_tag_obeys_unreleased_boundary() -> None:
    summary = inspect_repository(ROOT)

    if summary.prerelease:
        with pytest.raises(ReleaseMetadataError, match="cannot be published"):
            inspect_repository(ROOT, tag=f"v{summary.package_version}")
    elif summary.unreleased_changes_present:
        with pytest.raises(ReleaseMetadataError, match="empty Unreleased"):
            inspect_repository(ROOT, tag=f"v{summary.package_version}")
    else:
        tagged = inspect_repository(ROOT, tag=f"v{summary.package_version}")
        assert tagged.tag_validated is True


def test_clean_stable_release_validates_exact_tag(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)

    summary = inspect_repository(root, tag="v1.2.3")

    assert summary.tag_validated is True
    assert summary.release_ready is True
    assert summary.unreleased_changes_present is False


def test_release_tag_must_match_package_version(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path)

    with pytest.raises(ReleaseMetadataError, match="release tag must be"):
        inspect_repository(root, tag="v1.2.4")


@pytest.mark.parametrize("version", ["2.0.0.dev0", "2.0.0rc1", "2.0.0a2", "2.0.0b3"])
def test_prerelease_versions_are_non_publishable(version: str, tmp_path: Path) -> None:
    root = _write_fixture(
        tmp_path,
        package_version=version,
        status_version=version,
        changelog="# Changelog\n\n## Unreleased\n\n- development work\n",
    )

    summary = inspect_repository(root)
    assert summary.prerelease is True
    assert summary.release_ready is False
    with pytest.raises(ReleaseMetadataError, match="cannot be published"):
        inspect_repository(root, tag=f"v{version}")


def test_local_build_metadata_is_not_misclassified_as_prerelease(
    tmp_path: Path,
) -> None:
    root = _write_fixture(
        tmp_path,
        package_version="1.2.3+build.7",
        changelog="# Changelog\n\n## Unreleased\n\n## 1.2.3+build.7\n",
    )

    summary = inspect_repository(root)

    assert summary.prerelease is False
    assert summary.release_ready is True


def test_citation_version_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path, citation_version="1.2.4")

    with pytest.raises(ReleaseMetadataError, match="CITATION.cff version"):
        inspect_repository(root)


def test_setuptools_version_source_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path, setuptools_attribute="causal4d._version")

    with pytest.raises(ReleaseMetadataError, match="setuptools must derive"):
        inspect_repository(root)


def test_project_status_version_mismatch_fails_closed(tmp_path: Path) -> None:
    root = _write_fixture(tmp_path, status_version="1.2.2")

    with pytest.raises(ReleaseMetadataError, match="project-status required_version"):
        inspect_repository(root)


def test_stable_version_requires_changelog_release_heading(tmp_path: Path) -> None:
    root = _write_fixture(
        tmp_path,
        changelog="# Changelog\n\n## Unreleased\n",
    )

    with pytest.raises(ReleaseMetadataError, match="no exact"):
        inspect_repository(root)
