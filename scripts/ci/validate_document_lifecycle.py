"""Validate the lifecycle registry for operational Causal4D documentation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast


ROOT: Final = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY: Final = ROOT / "docs" / "lifecycle_registry.json"
DEFAULT_INDEX: Final = ROOT / "docs" / "README.md"
ALLOWED_STATUSES: Final = frozenset(
    {"current", "frozen", "historical", "superseded"}
)
ALLOWED_KINDS: Final = frozenset(
    {"archive", "concept", "contract", "protocol", "result", "runbook"}
)
ACTIVE_STATUSES: Final = frozenset({"current", "frozen"})
IMMUTABLE_STATUSES: Final = frozenset({"frozen", "superseded"})
HEX_40: Final = re.compile(r"[0-9a-f]{40}")
TOP_LEVEL_FIELDS: Final = frozenset(
    {
        "schema_version",
        "required_active_roles",
        "covered_path_patterns",
        "documents",
    }
)
DOCUMENT_FIELDS: Final = frozenset(
    {
        "path",
        "title",
        "kind",
        "status",
        "claim_bearing",
        "operational_role",
        "successor",
        "git_blob_sha1",
    }
)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(
                f"document lifecycle registry contains duplicate key {key!r}"
            )
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(
        f"document lifecycle registry contains non-finite value {value!r}"
    )


def _load_registry(path: Path) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_constant,
        )
    except json.JSONDecodeError as error:
        raise ValueError(
            f"document lifecycle registry is not valid JSON: {path}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("document lifecycle registry must be a JSON object")
    return cast(Mapping[str, Any], payload)


def _require_exact_fields(
    values: Mapping[str, Any],
    *,
    expected: frozenset[str],
    label: str,
) -> None:
    actual = set(values)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{label} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )


def _require_string(value: Any, *, label: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _require_optional_string(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _require_string(value, label=label)


def _require_string_sequence(value: Any, *, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence of strings")
    result = tuple(
        _require_string(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    )
    if len(result) != len(set(result)):
        raise ValueError(f"{label} must contain unique values")
    if result != tuple(sorted(result)):
        raise ValueError(f"{label} must be sorted")
    return result


def git_blob_sha1(path: Path) -> str:
    """Return the Git blob identity of one ordinary file."""

    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data, usedforsecurity=False).hexdigest()


def _resolve_document(root: Path, raw_path: str) -> Path:
    relative = Path(raw_path)
    if relative.is_absolute() or relative.as_posix() != raw_path:
        raise ValueError(
            "document path must be a normalized relative POSIX path: "
            f"{raw_path}"
        )
    if not relative.parts or relative.parts[0] != "docs" or ".." in relative.parts:
        raise ValueError(f"document path must stay below docs/: {raw_path}")
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"document path escapes repository root: {raw_path}")
    if candidate.suffix != ".md" or not candidate.is_file():
        raise ValueError(f"registered document does not exist as Markdown: {raw_path}")
    return candidate


def _validate_successor_chains(
    documents: Mapping[str, Mapping[str, Any]],
) -> None:
    for start_path, start in documents.items():
        if start["status"] != "superseded":
            continue
        seen: set[str] = set()
        chain: list[str] = []
        current_path = start_path
        while documents[current_path]["status"] == "superseded":
            if current_path in seen:
                cycle = " -> ".join((*chain, current_path))
                raise ValueError(
                    f"superseded document successor cycle detected: {cycle}"
                )
            seen.add(current_path)
            chain.append(current_path)
            successor = documents[current_path]["successor"]
            if successor not in documents:
                raise ValueError(
                    f"superseded document successor is not registered: "
                    f"{current_path} -> {successor}"
                )
            current_path = cast(str, successor)
        if documents[current_path]["status"] not in ACTIVE_STATUSES:
            raise ValueError(
                f"superseded document chain must terminate at an active document: "
                f"{start_path} -> {current_path}"
            )


def validate_document_lifecycle(
    *,
    root: Path = ROOT,
    registry_path: Path | None = None,
    index_path: Path | None = None,
) -> dict[str, int]:
    """Validate lifecycle metadata, immutable identities, and index coverage."""

    repository_root = root.resolve()
    registry = (
        registry_path.resolve()
        if registry_path is not None
        else repository_root / "docs" / "lifecycle_registry.json"
    )
    index = (
        index_path.resolve()
        if index_path is not None
        else repository_root / "docs" / "README.md"
    )
    if not registry.is_file():
        raise ValueError(f"document lifecycle registry does not exist: {registry}")
    if not index.is_file():
        raise ValueError(f"documentation index does not exist: {index}")

    payload = _load_registry(registry)
    if any(type(key) is not str for key in payload):
        raise ValueError("document lifecycle registry keys must be strings")
    _require_exact_fields(payload, expected=TOP_LEVEL_FIELDS, label="registry")
    if payload["schema_version"] != 1:
        raise ValueError("document lifecycle registry schema_version must equal 1")

    required_roles = _require_string_sequence(
        payload["required_active_roles"],
        label="required_active_roles",
    )
    covered_patterns = _require_string_sequence(
        payload["covered_path_patterns"],
        label="covered_path_patterns",
    )
    raw_documents = payload["documents"]
    if isinstance(raw_documents, (str, bytes)) or not isinstance(
        raw_documents, Sequence
    ):
        raise ValueError("documents must be a sequence of objects")

    normalized: dict[str, Mapping[str, Any]] = {}
    paths: list[str] = []
    active_roles: dict[str, str] = {}
    index_text = index.read_text(encoding="utf-8")
    if "(lifecycle_registry.json)" not in index_text:
        raise ValueError("docs/README.md must link lifecycle_registry.json")

    for position, raw_document in enumerate(raw_documents):
        if not isinstance(raw_document, Mapping):
            raise ValueError(f"documents[{position}] must be an object")
        if any(type(key) is not str for key in raw_document):
            raise ValueError(f"documents[{position}] keys must be strings")
        document = cast(Mapping[str, Any], raw_document)
        _require_exact_fields(
            document,
            expected=DOCUMENT_FIELDS,
            label=f"documents[{position}]",
        )

        path = _require_string(document["path"], label=f"documents[{position}].path")
        title = _require_string(
            document["title"], label=f"documents[{position}].title"
        )
        kind = _require_string(document["kind"], label=f"documents[{position}].kind")
        status = _require_string(
            document["status"], label=f"documents[{position}].status"
        )
        role = _require_optional_string(
            document["operational_role"],
            label=f"documents[{position}].operational_role",
        )
        successor = _require_optional_string(
            document["successor"],
            label=f"documents[{position}].successor",
        )
        blob_sha = _require_optional_string(
            document["git_blob_sha1"],
            label=f"documents[{position}].git_blob_sha1",
        )
        if type(document["claim_bearing"]) is not bool:
            raise ValueError(f"documents[{position}].claim_bearing must be boolean")
        if kind not in ALLOWED_KINDS:
            raise ValueError(f"unsupported document kind {kind!r}: {path}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"unsupported document status {status!r}: {path}")
        if path in normalized:
            raise ValueError(f"document path is registered more than once: {path}")

        file_path = _resolve_document(repository_root, path)
        if status in IMMUTABLE_STATUSES:
            if blob_sha is None or HEX_40.fullmatch(blob_sha) is None:
                raise ValueError(
                    f"immutable document requires a lowercase Git blob SHA-1: {path}"
                )
            actual_blob_sha = git_blob_sha1(file_path)
            if actual_blob_sha != blob_sha:
                raise ValueError(
                    f"immutable document Git blob SHA-1 changed: "
                    f"{path}; expected={blob_sha}, actual={actual_blob_sha}"
                )
        elif blob_sha is not None:
            raise ValueError(
                f"mutable document must not carry a frozen Git blob SHA-1: {path}"
            )

        if status == "superseded":
            if successor is None:
                raise ValueError(f"superseded document requires a successor: {path}")
            if role is not None:
                raise ValueError(
                    f"superseded document must not retain an operational role: {path}"
                )
        elif successor is not None:
            raise ValueError(
                f"only superseded documents may declare a successor: {path}"
            )

        if status in ACTIVE_STATUSES:
            if role is None:
                raise ValueError(
                    f"active document requires an operational role: {path}"
                )
            if role in active_roles:
                raise ValueError(
                    f"active operational role is ambiguous: {role!r}; "
                    f"{active_roles[role]} and {path}"
                )
            active_roles[role] = path
            relative_link = Path(path).relative_to("docs").as_posix()
            if f"({relative_link})" not in index_text:
                raise ValueError(
                    f"active document is not linked from docs/README.md: {path}"
                )
        elif role is not None:
            raise ValueError(
                f"inactive document must not declare an operational role: {path}"
            )

        normalized[path] = {
            "path": path,
            "title": title,
            "kind": kind,
            "status": status,
            "claim_bearing": document["claim_bearing"],
            "operational_role": role,
            "successor": successor,
            "git_blob_sha1": blob_sha,
        }
        paths.append(path)

    if paths != sorted(paths):
        raise ValueError("documents must be sorted by path")
    if set(required_roles) != set(active_roles):
        missing = sorted(set(required_roles) - set(active_roles))
        unexpected = sorted(set(active_roles) - set(required_roles))
        raise ValueError(
            "active operational roles do not match the registry contract; "
            f"missing={missing}, unexpected={unexpected}"
        )

    _validate_successor_chains(normalized)

    registered_paths = set(normalized)
    for pattern in covered_patterns:
        matches = sorted(
            path.relative_to(repository_root).as_posix()
            for path in repository_root.glob(pattern)
            if path.is_file()
        )
        if not matches:
            raise ValueError(f"covered_path_patterns entry matches no files: {pattern}")
        unregistered = sorted(set(matches) - registered_paths)
        if unregistered:
            raise ValueError(
                f"covered lifecycle documents are not registered for {pattern!r}: "
                f"{unregistered}"
            )

    status_counts = {
        status: sum(
            1 for document in normalized.values() if document["status"] == status
        )
        for status in sorted(ALLOWED_STATUSES)
    }
    return {
        "active_documents": sum(
            count
            for status, count in status_counts.items()
            if status in ACTIVE_STATUSES
        ),
        "current_documents": status_counts["current"],
        "frozen_documents": status_counts["frozen"],
        "historical_documents": status_counts["historical"],
        "registered_documents": len(normalized),
        "superseded_documents": status_counts["superseded"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the operational-document lifecycle registry."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Repository root (default: detected checkout root).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        help="Registry path (default: <root>/docs/lifecycle_registry.json).",
    )
    parser.add_argument(
        "--index",
        type=Path,
        help="Documentation index path (default: <root>/docs/README.md).",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the registry and print a deterministic JSON summary."""

    arguments = _build_parser().parse_args(argv)
    try:
        summary = validate_document_lifecycle(
            root=arguments.root,
            registry_path=arguments.registry,
            index_path=arguments.index,
        )
    except (OSError, ValueError) as error:
        print(f"document lifecycle validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
