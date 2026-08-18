from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripts.ci.validate_document_lifecycle import (
    git_blob_sha1,
    validate_document_lifecycle,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "lifecycle_registry.json"


def _write_registry(
    root: Path,
    *,
    documents: list[dict[str, Any]],
    required_active_roles: list[str] | None = None,
    covered_path_patterns: list[str] | None = None,
) -> Path:
    registry = root / "docs" / "lifecycle_registry.json"
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "required_active_roles": sorted(required_active_roles or []),
                "covered_path_patterns": sorted(covered_path_patterns or []),
                "documents": sorted(documents, key=lambda entry: entry["path"]),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return registry


def _document(
    path: str,
    *,
    title: str,
    kind: str,
    status: str,
    operational_role: str | None,
    successor: str | None = None,
    git_blob_sha1_value: str | None = None,
) -> dict[str, Any]:
    return {
        "path": path,
        "title": title,
        "kind": kind,
        "status": status,
        "claim_bearing": True,
        "operational_role": operational_role,
        "successor": successor,
        "git_blob_sha1": git_blob_sha1_value,
    }


def test_repository_document_lifecycle_registry_is_valid() -> None:
    assert validate_document_lifecycle(root=ROOT, registry_path=REGISTRY) == {
        "active_documents": 11,
        "current_documents": 10,
        "frozen_documents": 1,
        "historical_documents": 0,
        "registered_documents": 13,
        "superseded_documents": 2,
    }


def test_unregistered_versioned_document_is_rejected(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "[registry](lifecycle_registry.json)\n[current](current.md)\n",
        encoding="utf-8",
    )
    (docs / "current.md").write_text("current\n", encoding="utf-8")
    (docs / "protocol_v2.md").write_text("unregistered\n", encoding="utf-8")
    registry = _write_registry(
        tmp_path,
        documents=[
            _document(
                "docs/current.md",
                title="Current",
                kind="runbook",
                status="current",
                operational_role="current-runbook",
            )
        ],
        required_active_roles=["current-runbook"],
        covered_path_patterns=["docs/protocol_v*.md"],
    )

    with pytest.raises(ValueError, match="not registered"):
        validate_document_lifecycle(root=tmp_path, registry_path=registry)


def test_superseded_successor_cycle_is_rejected(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "[registry](lifecycle_registry.json)\n",
        encoding="utf-8",
    )
    first = docs / "first.md"
    second = docs / "second.md"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    registry = _write_registry(
        tmp_path,
        documents=[
            _document(
                "docs/first.md",
                title="First",
                kind="protocol",
                status="superseded",
                operational_role=None,
                successor="docs/second.md",
                git_blob_sha1_value=git_blob_sha1(first),
            ),
            _document(
                "docs/second.md",
                title="Second",
                kind="protocol",
                status="superseded",
                operational_role=None,
                successor="docs/first.md",
                git_blob_sha1_value=git_blob_sha1(second),
            ),
        ],
    )

    with pytest.raises(ValueError, match="successor cycle"):
        validate_document_lifecycle(root=tmp_path, registry_path=registry)


def test_frozen_document_drift_is_rejected(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    frozen = docs / "frozen.md"
    frozen.write_text("frozen\n", encoding="utf-8")
    (docs / "README.md").write_text(
        "[registry](lifecycle_registry.json)\n[frozen](frozen.md)\n",
        encoding="utf-8",
    )
    registry = _write_registry(
        tmp_path,
        documents=[
            _document(
                "docs/frozen.md",
                title="Frozen",
                kind="protocol",
                status="frozen",
                operational_role="frozen-protocol",
                git_blob_sha1_value=git_blob_sha1(frozen),
            )
        ],
        required_active_roles=["frozen-protocol"],
    )
    frozen.write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Git blob SHA-1 changed"):
        validate_document_lifecycle(root=tmp_path, registry_path=registry)


def test_active_document_must_be_linked_from_index(tmp_path: Path) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "README.md").write_text(
        "[registry](lifecycle_registry.json)\n",
        encoding="utf-8",
    )
    (docs / "current.md").write_text("current\n", encoding="utf-8")
    registry = _write_registry(
        tmp_path,
        documents=[
            _document(
                "docs/current.md",
                title="Current",
                kind="runbook",
                status="current",
                operational_role="current-runbook",
            )
        ],
        required_active_roles=["current-runbook"],
    )

    with pytest.raises(ValueError, match="not linked"):
        validate_document_lifecycle(root=tmp_path, registry_path=registry)
