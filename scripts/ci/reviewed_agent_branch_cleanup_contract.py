#!/usr/bin/env python3
"""Strict reviewed-manifest and report contracts for agent-branch cleanup."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn, Sequence

from stale_agent_branches import BranchAllowlist


_MANIFEST_ROOT = Path("ci/branch-cleanup-manifests")
_MAX_DELETIONS = 20
_REASONS = {
    "tip_reachable_from_default",
    "exact_tip_merged_in_pull_request",
}
_SENSITIVE = {
    "acquisition",
    "evidence",
    "freeze",
    "frozen",
    "provider",
    "registered",
    "release",
    "replication",
}
_BAD_REF = re.compile(r"[\x00-\x20~^:?*\[\\]")
_SHA1 = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")
_REPORT_DECISION_KEYS = {
    "name",
    "sha",
    "committed_at_utc",
    "protected",
    "allowlisted",
    "open_pull_requests",
    "exact_tip_merged_pull_requests",
    "tip_reachable_from_default",
    "cleanup_candidate",
    "reason",
    "age_days",
}
_REPORT_KEYS = {
    "schema_version",
    "artifact_kind",
    "generated_at_utc",
    "repository",
    "default_branch",
    "prefix",
    "minimum_age_days",
    "report_only",
    "branch_count",
    "cleanup_candidate_count",
    "excluded_count",
    "reason_counts",
    "decisions",
}


class CleanupError(ValueError):
    """Raised when a manifest or source report fails closed."""


@dataclass(frozen=True)
class CleanupEntry:
    name: str
    expected_sha: str
    eligibility_reason: str
    merged_pull_requests: tuple[int, ...]
    artifact_reference_reviewed: bool
    review_note: str

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CleanupEntry:
        _keys(
            payload,
            {
                "name",
                "expected_sha",
                "eligibility_reason",
                "merged_pull_requests",
                "artifact_reference_reviewed",
                "review_note",
            },
            "cleanup entry",
        )
        name = _branch(payload["name"], reject_sensitive=True)
        sha = _sha(payload["expected_sha"], 40, "expected_sha")
        reason = _string(payload["eligibility_reason"], "eligibility_reason")
        if reason not in _REASONS:
            raise CleanupError(f"unsupported eligibility reason: {reason}")
        merged = _positive_ints(payload["merged_pull_requests"], "merged PRs")
        if reason == "tip_reachable_from_default" and merged:
            raise CleanupError("reachable candidates cannot name merged PRs")
        if reason == "exact_tip_merged_in_pull_request" and not merged:
            raise CleanupError("merged-tip candidates must name a merged PR")
        if payload["artifact_reference_reviewed"] is not True:
            raise CleanupError("artifact_reference_reviewed must be true")
        note = _string(payload["review_note"], "review_note").strip()
        if len(note) < 12:
            raise CleanupError("review_note must explain archival-reference safety")
        return cls(name, sha, reason, merged, True, note)


@dataclass(frozen=True)
class CleanupManifest:
    repository: str
    default_branch: str
    source_report_path: str
    source_report_sha256: str
    source_report_generated_at_utc: str
    minimum_age_days: int
    reviewed_at_utc: str
    expires_at_utc: str
    reviewed_by: str
    issue_url: str
    entries: tuple[CleanupEntry, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> CleanupManifest:
        _keys(
            payload,
            {
                "schema_version",
                "artifact_kind",
                "repository",
                "default_branch",
                "source_report_path",
                "source_report_sha256",
                "source_report_generated_at_utc",
                "minimum_age_days",
                "reviewed_at_utc",
                "expires_at_utc",
                "reviewed_by",
                "issue_url",
                "entries",
            },
            "cleanup manifest",
        )
        if payload["schema_version"] != 1:
            raise CleanupError("unsupported cleanup manifest schema")
        if payload["artifact_kind"] != "Causal4DReviewedAgentBranchCleanupManifest":
            raise CleanupError("unexpected cleanup manifest artifact_kind")
        repository = _repository(payload["repository"])
        default = _string(payload["default_branch"], "default_branch")
        if default != "main":
            raise CleanupError("reviewed cleanup currently supports main only")
        report_path = _relative_json_path(
            payload["source_report_path"],
            "source_report_path",
            required_prefix=_MANIFEST_ROOT,
        )
        report_sha = _sha(payload["source_report_sha256"], 64, "report SHA")
        generated = _utc(payload["source_report_generated_at_utc"], "report time")
        minimum_age = _positive_int(payload["minimum_age_days"], "minimum age")
        if minimum_age < 30:
            raise CleanupError("minimum_age_days must be at least 30")
        reviewed = _utc(payload["reviewed_at_utc"], "reviewed_at_utc")
        expires = _utc(payload["expires_at_utc"], "expires_at_utc")
        if reviewed < generated or reviewed - generated > timedelta(days=7):
            raise CleanupError("source report must be reviewed within seven days")
        if expires <= reviewed or expires - reviewed > timedelta(days=7):
            raise CleanupError("cleanup approval must expire within seven days")
        reviewed_by = _string(payload["reviewed_by"], "reviewed_by")
        issue_url = _string(payload["issue_url"], "issue_url")
        expected_issue = f"https://github.com/{repository}/issues/336"
        if issue_url != expected_issue:
            raise CleanupError(f"issue_url must be {expected_issue}")
        raw_entries = payload["entries"]
        if isinstance(raw_entries, (str, bytes)) or not isinstance(
            raw_entries, Sequence
        ):
            raise CleanupError("entries must be a sequence")
        entries: list[CleanupEntry] = []
        for index, item in enumerate(raw_entries):
            if not isinstance(item, Mapping):
                raise CleanupError(f"entries[{index}] must be an object")
            entries.append(CleanupEntry.from_mapping(item))
        if not entries or len(entries) > _MAX_DELETIONS:
            raise CleanupError("manifest must contain between 1 and 20 entries")
        if len({entry.name for entry in entries}) != len(entries):
            raise CleanupError("manifest branch names must be unique")
        return cls(
            repository,
            default,
            report_path,
            report_sha,
            _utc_text(generated),
            minimum_age,
            _utc_text(reviewed),
            _utc_text(expires),
            reviewed_by,
            issue_url,
            tuple(entries),
        )


@dataclass(frozen=True)
class LoadedCleanup:
    manifest: CleanupManifest
    manifest_sha256: str
    source_report_sha256: str


def _raise(message: str) -> NoReturn:
    raise CleanupError(message)


def _keys(payload: Mapping[str, Any], expected: set[str], name: str) -> None:
    if set(payload) != expected:
        raise CleanupError(f"{name} has unexpected or missing keys")


def _string(value: Any, name: str) -> str:
    if type(value) is not str or not value.strip():
        raise CleanupError(f"{name} must be a nonempty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise CleanupError(f"{name} must be a nonnegative integer")
    return value


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 1:
        raise CleanupError(f"{name} must be a positive integer")
    return value


def _positive_ints(value: Any, name: str) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise CleanupError(f"{name} must be a sequence")
    result = tuple(
        _positive_int(item, f"{name}[{index}]")
        for index, item in enumerate(value)
    )
    if tuple(sorted(set(result))) != result:
        raise CleanupError(f"{name} must be sorted and unique")
    return result


def _sha(value: Any, length: int, name: str) -> str:
    text = _string(value, name)
    pattern = _SHA1 if length == 40 else _SHA256
    if pattern.fullmatch(text) is None:
        raise CleanupError(
            f"{name} must be lowercase hexadecimal with length {length}"
        )
    return text


def _repository(value: Any) -> str:
    repository = _string(value, "repository")
    parts = repository.split("/")
    if len(parts) != 2 or any(
        not part or part in {".", ".."} for part in parts
    ):
        raise CleanupError("repository must use owner/name syntax")
    return repository


def _branch(value: Any, *, reject_sensitive: bool) -> str:
    name = _string(value, "branch name")
    if not name.startswith("agent/") or name == "agent/":
        raise CleanupError("cleanup is restricted to nonempty agent/* branches")
    if (
        name.endswith(("/", ".", ".lock"))
        or "//" in name
        or ".." in name
        or "@{" in name
        or _BAD_REF.search(name)
        or any(part.startswith(".") for part in name.split("/"))
    ):
        raise CleanupError(f"unsafe branch name: {name!r}")
    lowered = name.lower()
    sensitive = sorted(token for token in _SENSITIVE if token in lowered)
    if reject_sensitive and sensitive:
        raise CleanupError(f"branch contains protected archival tokens: {sensitive}")
    return name


def _relative_json_path(
    value: Any,
    name: str,
    *,
    required_prefix: Path,
) -> str:
    text = _string(value, name)
    path = Path(text)
    if (
        path.is_absolute()
        or path.suffix != ".json"
        or text.endswith("/")
        or "\\" in text
        or "//" in text
        or any(part in {"", ".", ".."} for part in path.parts)
        or path.parts[: len(required_prefix.parts)] != required_prefix.parts
    ):
        raise CleanupError(
            f"{name} must be a JSON file below {required_prefix.as_posix()}/"
        )
    return path.as_posix()


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CleanupError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _ordinary_bytes(root: Path, relative: str, name: str) -> bytes:
    if root.is_symlink():
        raise CleanupError("repository root must not be a symlink")
    resolved_root = root.resolve()
    current = resolved_root
    for part in Path(relative).parts:
        current /= part
        if current.is_symlink():
            raise CleanupError(f"{name} must not traverse a symlink")
    if not current.is_file():
        raise CleanupError(f"{name} must be an ordinary file")
    try:
        current.resolve().relative_to(resolved_root)
    except ValueError as error:
        raise CleanupError(f"{name} escapes the repository root") from error
    return current.read_bytes()


def _json(raw: bytes, name: str) -> Mapping[str, Any]:
    try:
        payload = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: _raise(
                f"non-finite JSON constant in {name}: {value}"
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CleanupError(f"{name} must be strict UTF-8 JSON") from error
    if not isinstance(payload, Mapping):
        raise CleanupError(f"{name} must be a JSON object")
    return payload


def _utc(value: Any, name: str) -> datetime:
    text = _string(value, name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise CleanupError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise CleanupError(f"{name} must use UTC")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise CleanupError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _load_allowlist(root: Path) -> BranchAllowlist:
    relative = ".github/stale-agent-branch-allowlist.json"
    raw = _ordinary_bytes(root, relative, "allowlist")
    try:
        return BranchAllowlist.from_mapping(_json(raw, "allowlist"))
    except ValueError as error:
        raise CleanupError(f"invalid branch allowlist: {error}") from error


def _validate_report_decision(item: Mapping[str, Any]) -> tuple[str, str]:
    _keys(item, _REPORT_DECISION_KEYS, "source report decision")
    name = _branch(item["name"], reject_sensitive=False)
    _sha(item["sha"], 40, f"report SHA for {name}")
    _utc(item["committed_at_utc"], f"report commit time for {name}")
    if type(item["protected"]) is not bool:
        raise CleanupError(f"report protected flag is invalid for {name}")
    if type(item["allowlisted"]) is not bool:
        raise CleanupError(f"report allowlisted flag is invalid for {name}")
    _positive_ints(item["open_pull_requests"], f"report open PRs for {name}")
    _positive_ints(
        item["exact_tip_merged_pull_requests"],
        f"report merged PRs for {name}",
    )
    if type(item["tip_reachable_from_default"]) is not bool:
        raise CleanupError(f"report reachability flag is invalid for {name}")
    if type(item["cleanup_candidate"]) is not bool:
        raise CleanupError(f"report candidate flag is invalid for {name}")
    reason = _string(item["reason"], f"report reason for {name}")
    age = item["age_days"]
    if isinstance(age, bool) or not isinstance(age, (int, float)):
        raise CleanupError(f"report age is invalid for {name}")
    if not math.isfinite(float(age)):
        raise CleanupError(f"report age must be finite for {name}")
    return name, reason


def _validate_report(manifest: CleanupManifest, report: Mapping[str, Any]) -> None:
    _keys(report, _REPORT_KEYS, "source report")
    expected = (
        report["schema_version"] == 1
        and report["artifact_kind"] == "Causal4DStaleAgentBranchReport"
        and report["report_only"] is True
        and report["repository"] == manifest.repository
        and report["default_branch"] == manifest.default_branch
        and report["prefix"] == "agent/"
        and report["minimum_age_days"] == manifest.minimum_age_days
        and report["generated_at_utc"]
        == manifest.source_report_generated_at_utc
    )
    if not expected:
        raise CleanupError("source report metadata does not match the manifest")
    decisions = report["decisions"]
    if isinstance(decisions, (str, bytes)) or not isinstance(decisions, Sequence):
        raise CleanupError("source report decisions must be a sequence")
    candidates: dict[str, Mapping[str, Any]] = {}
    reason_counts: dict[str, int] = {}
    for item in decisions:
        if not isinstance(item, Mapping):
            raise CleanupError("source report decisions must be objects")
        name, reason = _validate_report_decision(item)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if item["cleanup_candidate"]:
            if name in candidates:
                raise CleanupError("source report candidate names must be unique")
            candidates[name] = item
    branch_count = _nonnegative_int(report["branch_count"], "branch_count")
    candidate_count = _nonnegative_int(
        report["cleanup_candidate_count"],
        "cleanup_candidate_count",
    )
    excluded_count = _nonnegative_int(report["excluded_count"], "excluded_count")
    if branch_count != len(decisions):
        raise CleanupError("source report branch_count is inconsistent")
    if candidate_count != len(candidates):
        raise CleanupError("source report candidate count is inconsistent")
    if excluded_count != len(decisions) - len(candidates):
        raise CleanupError("source report excluded count is inconsistent")
    raw_reason_counts = report["reason_counts"]
    if not isinstance(raw_reason_counts, Mapping):
        raise CleanupError("source report reason_counts must be an object")
    validated_counts: dict[str, int] = {}
    for key, value in raw_reason_counts.items():
        reason = _string(key, "reason_counts key")
        validated_counts[reason] = _nonnegative_int(
            value,
            f"reason_counts[{reason}]",
        )
    if validated_counts != dict(sorted(reason_counts.items())):
        raise CleanupError("source report reason_counts is inconsistent")
    for entry in manifest.entries:
        item = candidates.get(entry.name)
        if item is None:
            raise CleanupError(f"{entry.name} is not a reviewed report candidate")
        merged = _positive_ints(
            item["exact_tip_merged_pull_requests"],
            f"report merged PRs for {entry.name}",
        )
        if float(item["age_days"]) < manifest.minimum_age_days:
            raise CleanupError(f"report age is below the threshold for {entry.name}")
        if (
            item["sha"] != entry.expected_sha
            or item["reason"] != entry.eligibility_reason
            or merged != entry.merged_pull_requests
            or item["protected"] is not False
            or item["allowlisted"] is not False
            or item["open_pull_requests"] != []
        ):
            raise CleanupError(f"report evidence differs for {entry.name}")
        if entry.eligibility_reason == "tip_reachable_from_default":
            if item["tip_reachable_from_default"] is not True:
                raise CleanupError(f"report reachability differs for {entry.name}")


def load_cleanup(
    root: str | Path,
    manifest_path: str,
    expected_manifest_sha256: str,
) -> tuple[LoadedCleanup, BranchAllowlist]:
    root_path = Path(root)
    relative_manifest = _relative_json_path(
        manifest_path,
        "manifest path",
        required_prefix=_MANIFEST_ROOT,
    )
    manifest_raw = _ordinary_bytes(root_path, relative_manifest, "manifest")
    manifest_sha = _digest(manifest_raw)
    if manifest_sha != _sha(expected_manifest_sha256, 64, "manifest SHA"):
        raise CleanupError("manifest SHA-256 differs from the dispatch input")
    manifest = CleanupManifest.from_mapping(_json(manifest_raw, "manifest"))
    report_raw = _ordinary_bytes(
        root_path,
        manifest.source_report_path,
        "source report",
    )
    report_sha = _digest(report_raw)
    if report_sha != manifest.source_report_sha256:
        raise CleanupError("source report SHA-256 differs from the manifest")
    _validate_report(manifest, _json(report_raw, "source report"))
    allowlist = _load_allowlist(root_path)
    return LoadedCleanup(manifest, manifest_sha, report_sha), allowlist

