#!/usr/bin/env python3
"""Validate Causal4D package, citation, changelog, and release-tag metadata."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


_VERSION_ASSIGNMENT = "__version__"
_CFF_SCALAR = re.compile(r"(?m)^{key}:\s*([^\n]+)$")
_PRERELEASE = re.compile(
    r"(?:\d)(?:alpha|beta|preview|pre|rc|dev|a|b|c)\d*"
    r"(?=$|[.+-])|(?:^|[._-])dev\d*(?=$|[.+-])",
    re.IGNORECASE,
)
_TOML_SECTION = re.compile(r"(?m)^\[([^\]]+)\]\s*$")


class ReleaseMetadataError(ValueError):
    """Raised when release-facing metadata is inconsistent."""


@dataclass(frozen=True)
class ReleaseMetadataSummary:
    schema_version: int
    artifact_kind: str
    package_version: str
    citation_version: str
    project_status_required_version: str
    setuptools_version_attribute: str
    changelog_heading_present: bool
    unreleased_section_present: bool
    unreleased_changes_present: bool
    prerelease: bool
    tag: str | None
    tag_validated: bool
    release_ready: bool


def _required_file(root: Path, relative: str) -> Path:
    path = root / relative
    if not path.is_file():
        raise ReleaseMetadataError(f"required metadata file is missing: {relative}")
    return path


def _package_version(root: Path) -> str:
    path = _required_file(root, "src/causal4d/__init__.py")
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: list[str] = []
    for node in module.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        targets: tuple[ast.expr, ...]
        value: ast.expr | None
        if isinstance(node, ast.Assign):
            targets = tuple(node.targets)
            value = node.value
        else:
            targets = (node.target,)
            value = node.value
        if not any(
            isinstance(target, ast.Name) and target.id == _VERSION_ASSIGNMENT
            for target in targets
        ):
            continue
        if not isinstance(value, ast.Constant) or not isinstance(value.value, str):
            raise ReleaseMetadataError("causal4d.__version__ must be a string literal")
        values.append(value.value)
    if len(values) != 1:
        raise ReleaseMetadataError(
            "src/causal4d/__init__.py must define exactly one string __version__"
        )
    version = values[0].strip()
    if not version or any(character.isspace() for character in version):
        raise ReleaseMetadataError(f"invalid package version: {version!r}")
    return version


def _cff_scalar(root: Path, key: str) -> str:
    path = _required_file(root, "CITATION.cff")
    text = path.read_text(encoding="utf-8")
    match = re.search(
        _CFF_SCALAR.pattern.format(key=re.escape(key)),
        text,
        flags=_CFF_SCALAR.flags,
    )
    if match is None:
        raise ReleaseMetadataError(f"CITATION.cff is missing {key!r}")
    return match.group(1).strip().strip("\"'")


def _toml_section(text: str, name: str) -> str:
    matches = list(_TOML_SECTION.finditer(text))
    for index, match in enumerate(matches):
        if match.group(1).strip() != name:
            continue
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        return text[start:end]
    raise ReleaseMetadataError(f"pyproject.toml is missing [{name}]")


def _setuptools_version_attribute(root: Path) -> str:
    path = _required_file(root, "pyproject.toml")
    text = path.read_text(encoding="utf-8")
    project = _toml_section(text, "project")
    dynamic_match = re.search(
        r"(?ms)^dynamic\s*=\s*\[(?P<body>.*?)\]\s*$",
        project,
    )
    if dynamic_match is None:
        raise ReleaseMetadataError("project.version must remain dynamic")
    dynamic_values = re.findall(r"[\"']([^\"']+)[\"']", dynamic_match.group("body"))
    if "version" not in dynamic_values:
        raise ReleaseMetadataError("project.version must remain dynamic")

    setuptools_dynamic = _toml_section(text, "tool.setuptools.dynamic")
    attribute_match = re.search(
        r"(?m)^version\s*=\s*\{\s*attr\s*=\s*[\"']([^\"']+)[\"']\s*\}\s*$",
        setuptools_dynamic,
    )
    if attribute_match is None:
        raise ReleaseMetadataError(
            "pyproject.toml must derive version from a setuptools attribute"
        )
    return attribute_match.group(1)


def _project_status_required_version(root: Path) -> str:
    path = _required_file(root, "ci/project_status_v2.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        version = payload["packages"]["causal4d"]["required_version"]
    except (KeyError, TypeError) as error:
        raise ReleaseMetadataError(
            "ci/project_status_v2.json is missing packages.causal4d.required_version"
        ) from error
    if not isinstance(version, str) or not version:
        raise ReleaseMetadataError(
            "project-status Causal4D required_version must be a non-empty string"
        )
    return version


def _changelog_state(root: Path, version: str) -> tuple[bool, bool, bool]:
    changelog = _required_file(root, "CHANGELOG.md").read_text(encoding="utf-8")
    heading_present = bool(
        re.search(rf"(?m)^##\s+{re.escape(version)}\s*$", changelog)
    )
    unreleased_match = re.search(
        r"(?ms)^##\s+Unreleased\s*$\n(?P<body>.*?)(?=^##\s+|\Z)",
        changelog,
    )
    unreleased_present = unreleased_match is not None
    unreleased_changes = bool(
        unreleased_match is not None and unreleased_match.group("body").strip()
    )
    return heading_present, unreleased_present, unreleased_changes


def inspect_repository(
    root: Path,
    *,
    tag: str | None = None,
) -> ReleaseMetadataSummary:
    """Return validated release metadata for ``root`` or raise fail closed."""

    resolved_root = root.resolve()
    package_version = _package_version(resolved_root)
    citation_version = _cff_scalar(resolved_root, "version")
    if citation_version != package_version:
        raise ReleaseMetadataError(
            "CITATION.cff version does not match causal4d.__version__: "
            f"{citation_version!r} != {package_version!r}"
        )

    attribute = _setuptools_version_attribute(resolved_root)
    if attribute != "causal4d.__version__":
        raise ReleaseMetadataError(
            "setuptools must derive distribution metadata from causal4d.__version__"
        )

    heading_present, unreleased_present, unreleased_changes = _changelog_state(
        resolved_root,
        package_version,
    )
    prerelease = _PRERELEASE.search(package_version) is not None
    if prerelease:
        if not unreleased_present:
            raise ReleaseMetadataError(
                "development/prerelease versions require an Unreleased "
                "changelog section"
            )
    elif not heading_present:
        raise ReleaseMetadataError(
            f"CHANGELOG.md has no exact '## {package_version}' release heading"
        )

    project_status_version = _project_status_required_version(resolved_root)
    if project_status_version != package_version:
        raise ReleaseMetadataError(
            "project-status required_version does not match causal4d.__version__: "
            f"{project_status_version!r} != {package_version!r}"
        )

    tag_validated = False
    if tag is not None:
        expected_tag = f"v{package_version}"
        if tag != expected_tag:
            raise ReleaseMetadataError(
                f"release tag must be {expected_tag!r}, received {tag!r}"
            )
        if prerelease:
            raise ReleaseMetadataError(
                "development/prerelease package versions cannot be published "
                "as releases"
            )
        if unreleased_changes:
            raise ReleaseMetadataError(
                "tagged releases require an empty Unreleased changelog section"
            )
        tag_validated = True

    release_ready = (
        not prerelease
        and heading_present
        and not unreleased_changes
        and citation_version == package_version
        and project_status_version == package_version
    )
    return ReleaseMetadataSummary(
        schema_version=1,
        artifact_kind="Causal4DReleaseMetadataIntegrity",
        package_version=package_version,
        citation_version=citation_version,
        project_status_required_version=project_status_version,
        setuptools_version_attribute=attribute,
        changelog_heading_present=heading_present,
        unreleased_section_present=unreleased_present,
        unreleased_changes_present=unreleased_changes,
        prerelease=prerelease,
        tag=tag,
        tag_validated=tag_validated,
        release_ready=release_ready,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--tag")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        summary = inspect_repository(args.root, tag=args.tag)
    except (ReleaseMetadataError, json.JSONDecodeError, SyntaxError) as error:
        print(f"release metadata integrity failed: {error}", file=sys.stderr)
        return 1

    rendered = json.dumps(asdict(summary), indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
