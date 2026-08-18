#!/usr/bin/env python3
"""Resolve a bounded exact-history fetch for one locked Git ancestor."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, NoReturn
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


_SHA = re.compile(r"[0-9a-f]{40}")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_MAX_FETCH_DEPTH = 2048
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AncestorPlanError(ValueError):
    """Raised when the lock or comparison cannot prove exact ancestry."""


@dataclass(frozen=True)
class LockedAncestorFetchPlan:
    schema_version: int
    artifact_kind: str
    repository: str
    head_sha: str
    required_parent_commit: str
    comparison_status: str
    ahead_by: int
    behind_by: int
    total_commits: int
    fetch_depth: int
    maximum_fetch_depth: int
    merge_base_sha: str
    ancestry_verified_by: str


def _raise(message: str) -> NoReturn:
    raise AncestorPlanError(message)


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise AncestorPlanError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _load_json_bytes(raw: bytes, *, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: _raise(
                f"non-finite JSON constant in {name}: {value}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AncestorPlanError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise AncestorPlanError(f"{name} must be a JSON object")
    return payload


def _ordinary_file(path: Path, *, name: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise AncestorPlanError(f"{name} must be an ordinary file")
    return path.read_bytes()


def _sha(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA.fullmatch(value) is None:
        raise AncestorPlanError(f"{name} must be a lowercase 40-character SHA")
    return value


def _repository(value: Any) -> str:
    if type(value) is not str or _REPOSITORY.fullmatch(value) is None:
        raise AncestorPlanError("repository must use owner/name syntax")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise AncestorPlanError(f"{name} must be a nonnegative integer")
    return value


def required_parent_from_lock(payload: Mapping[str, Any]) -> str:
    """Return the strict locked parent commit from a configuration object."""

    return _sha(payload.get("required_parent_commit"), name="required_parent_commit")


def _nested_sha(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise AncestorPlanError(f"comparison {key} must be an object")
    return _sha(value.get("sha"), name=f"comparison {key}.sha")


def resolve_fetch_plan(
    comparison: Mapping[str, Any],
    *,
    repository: str,
    head_sha: str,
    required_parent_commit: str,
    maximum_fetch_depth: int = _MAX_FETCH_DEPTH,
) -> LockedAncestorFetchPlan:
    """Validate GitHub comparison metadata and derive the exact fetch depth."""

    validated_repository = _repository(repository)
    head = _sha(head_sha, name="head_sha")
    parent = _sha(required_parent_commit, name="required_parent_commit")
    if type(maximum_fetch_depth) is not int or maximum_fetch_depth < 1:
        raise AncestorPlanError("maximum_fetch_depth must be a positive integer")

    status = comparison.get("status")
    if status not in {"ahead", "identical"}:
        raise AncestorPlanError(
            "required parent must be identical to or an ancestor of the exact head"
        )
    base_sha = _nested_sha(comparison, "base_commit")
    merge_base_sha = _nested_sha(comparison, "merge_base_commit")
    compared_head_sha = _nested_sha(comparison, "head_commit")
    if base_sha != parent or merge_base_sha != parent:
        raise AncestorPlanError("comparison is not rooted at the required parent")
    if compared_head_sha != head:
        raise AncestorPlanError("comparison head does not match the workflow head")

    ahead_by = _nonnegative_integer(comparison.get("ahead_by"), name="ahead_by")
    behind_by = _nonnegative_integer(comparison.get("behind_by"), name="behind_by")
    total_commits = _nonnegative_integer(
        comparison.get("total_commits"),
        name="total_commits",
    )
    if behind_by != 0:
        raise AncestorPlanError("comparison head is behind the required parent")
    if total_commits != ahead_by:
        raise AncestorPlanError("comparison commit counts are inconsistent")
    if status == "identical":
        if parent != head or ahead_by != 0:
            raise AncestorPlanError("identical comparison metadata is inconsistent")
    elif parent == head or ahead_by < 1:
        raise AncestorPlanError("ahead comparison metadata is inconsistent")

    fetch_depth = ahead_by + 1
    if fetch_depth > maximum_fetch_depth:
        raise AncestorPlanError(
            "required ancestry exceeds the bounded fetch-depth policy: "
            f"{fetch_depth} > {maximum_fetch_depth}"
        )
    return LockedAncestorFetchPlan(
        schema_version=1,
        artifact_kind="Causal4DLockedAncestorFetchPlan",
        repository=validated_repository,
        head_sha=head,
        required_parent_commit=parent,
        comparison_status=str(status),
        ahead_by=ahead_by,
        behind_by=behind_by,
        total_commits=total_commits,
        fetch_depth=fetch_depth,
        maximum_fetch_depth=maximum_fetch_depth,
        merge_base_sha=merge_base_sha,
        ancestry_verified_by="github_compare_api_and_local_git",
    )


def _comparison_from_github(
    *,
    repository: str,
    parent: str,
    head: str,
    token: str,
) -> Mapping[str, Any]:
    encoded_repository = quote(repository, safe="/")
    url = (
        f"https://api.github.com/repos/{encoded_repository}/compare/"
        f"{parent}...{head}?per_page=1&page=2"
    )
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "causal4d-locked-ancestor-fetch-plan",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = Request(url, headers=headers)
    try:
        with urlopen(request, timeout=60) as response:
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise AncestorPlanError(
            f"GitHub comparison failed with HTTP {error.code}: {body}"
        ) from error
    except URLError as error:
        raise AncestorPlanError(
            f"GitHub comparison could not be reached: {error}"
        ) from error
    if len(raw) > _MAX_RESPONSE_BYTES:
        raise AncestorPlanError("GitHub comparison response exceeds the size limit")
    return _load_json_bytes(raw, name="GitHub comparison")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--compare-json", type=Path)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--maximum-fetch-depth", type=int, default=_MAX_FETCH_DEPTH)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    try:
        repository = _repository(arguments.repository)
        head = _sha(arguments.head_sha, name="head_sha")
        lock = _load_json_bytes(
            _ordinary_file(arguments.lock, name="lock"),
            name="lock",
        )
        parent = required_parent_from_lock(lock)
        if arguments.compare_json is None:
            comparison = _comparison_from_github(
                repository=repository,
                parent=parent,
                head=head,
                token=os.environ.get(arguments.token_env, ""),
            )
        else:
            comparison = _load_json_bytes(
                _ordinary_file(arguments.compare_json, name="comparison fixture"),
                name="comparison fixture",
            )
        plan = resolve_fetch_plan(
            comparison,
            repository=repository,
            head_sha=head,
            required_parent_commit=parent,
            maximum_fetch_depth=arguments.maximum_fetch_depth,
        )
    except (AncestorPlanError, OSError) as error:
        print(f"locked ancestor fetch planning failed: {error}", file=sys.stderr)
        return 1
    _write_json(arguments.output, asdict(plan))
    print(plan.fetch_depth)
    return 0


if __name__ == "__main__":
    sys.exit(main())
