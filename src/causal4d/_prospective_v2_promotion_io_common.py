"""Shared strict-JSON helpers for prospective V2 promotion replay."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from causal4d._prospective_v2_promotion_evidence import (
    PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION,
)
from causal4d.artifact_io import load_strict_json_object, read_regular_file


def load_object(path: str | Path, *, name: str) -> dict[str, Any]:
    snapshot = read_regular_file(path, name=name)
    return load_strict_json_object(snapshot.payload, name=name)


def require_fields(
    values: Any,
    *,
    expected: frozenset[str],
    name: str,
) -> Mapping[str, Any]:
    if not isinstance(values, Mapping) or any(type(key) is not str for key in values):
        raise ValueError(f"{name} must be a string-keyed JSON object")
    actual = set(values)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields changed; missing={missing}, unexpected={unexpected}"
        )
    return values


def require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(type(key) is not str for key in value):
        raise ValueError(f"{name} must be a string-keyed JSON object")
    return value


def require_list(value: Any, *, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a JSON array")
    return value


def require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def require_expected_identity(
    actual: str,
    expected: str | None,
    *,
    name: str,
) -> None:
    if expected is None:
        return
    expected_id = require_sha256(expected, name=f"expected_{name}")
    if actual != expected_id:
        raise ValueError(f"{name} does not match expected_{name}")


def require_schema(
    values: Mapping[str, Any],
    *,
    artifact_kind: str,
    name: str,
) -> None:
    if (
        type(values["schema_version"]) is not int
        or values["schema_version"] != PROSPECTIVE_V2_PROMOTION_SCHEMA_VERSION
    ):
        raise ValueError(f"unsupported {name} schema version")
    if values["artifact_kind"] != artifact_kind:
        raise ValueError(f"unexpected {name} artifact kind")
