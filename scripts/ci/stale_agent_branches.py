#!/usr/bin/env python3
"""Produce a fail-closed, read-only report for stale ``agent/*`` branches."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


_API_ROOT = "https://api.github.com"


class GitHubApiError(RuntimeError):
    """Raised when GitHub returns an unexpected or malformed response."""


class ApiClient(Protocol):
    def get_json(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> Any: ...

    def paginated(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> list[Any]: ...


class GitHubApi:
    """Small dependency-free, GET-only GitHub REST client."""

    def __init__(self, token: str) -> None:
        if not token:
            raise ValueError("GitHub token must be nonempty")
        self._token = token

    def get_json(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> Any:
        suffix = f"?{urlencode(query)}" if query else ""
        request = Request(
            f"{_API_ROOT}{path}{suffix}",
            method="GET",
            headers={
                "Accept": "application/vnd.github+json",
                "Authorization": f"Bearer {self._token}",
                "User-Agent": "causal4d-stale-agent-branch-report",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )
        try:
            with urlopen(request, timeout=60) as response:
                payload = response.read()
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise GitHubApiError(
                f"GitHub API GET {path} failed with {error.code}: {body}"
            ) from error
        except URLError as error:
            raise GitHubApiError(
                f"GitHub API GET {path} could not be reached: {error}"
            ) from error
        if not payload:
            return None
        try:
            return json.loads(payload)
        except json.JSONDecodeError as error:
            raise GitHubApiError(
                f"GitHub API GET {path} returned invalid JSON"
            ) from error

    def paginated(
        self,
        path: str,
        *,
        query: Mapping[str, str | int] | None = None,
    ) -> list[Any]:
        result: list[Any] = []
        page = 1
        while True:
            parameters: dict[str, str | int] = dict(query or {})
            parameters.update({"per_page": 100, "page": page})
            payload = self.get_json(path, query=parameters)
            if not isinstance(payload, list):
                raise GitHubApiError(f"GitHub API GET {path} did not return a list")
            result.extend(payload)
            if len(payload) < 100:
                return result
            page += 1


@dataclass(frozen=True)
class BranchAllowlist:
    """Exact branches and prefixes excluded from cleanup candidacy."""

    branches: frozenset[str]
    prefixes: tuple[str, ...]

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BranchAllowlist:
        if set(payload) != {"schema_version", "branches", "prefixes"}:
            raise ValueError(
                "branch allowlist must contain schema_version, branches, and prefixes"
            )
        if payload["schema_version"] != 1:
            raise ValueError("unsupported branch allowlist schema")
        branches = _string_sequence(payload["branches"], name="branches")
        prefixes = _string_sequence(payload["prefixes"], name="prefixes")
        if len(set(branches)) != len(branches):
            raise ValueError("allowlisted branches must be unique")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("allowlisted prefixes must be unique")
        return cls(frozenset(branches), tuple(prefixes))

    def matches(self, branch: str) -> bool:
        return branch in self.branches or any(
            branch.startswith(prefix) for prefix in self.prefixes
        )


@dataclass(frozen=True)
class BranchInspection:
    """Evidence used to classify one branch conservatively."""

    name: str
    sha: str
    committed_at_utc: str
    protected: bool
    allowlisted: bool
    open_pull_requests: tuple[int, ...]
    exact_tip_merged_pull_requests: tuple[int, ...]
    tip_reachable_from_default: bool

    @property
    def committed_at(self) -> datetime:
        return _parse_utc(self.committed_at_utc, name="committed_at_utc")


@dataclass(frozen=True)
class BranchDecision:
    """One report-only cleanup-candidate decision."""

    inspection: BranchInspection
    cleanup_candidate: bool
    reason: str
    age_days: float

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self.inspection),
            "open_pull_requests": list(self.inspection.open_pull_requests),
            "exact_tip_merged_pull_requests": list(
                self.inspection.exact_tip_merged_pull_requests
            ),
            "cleanup_candidate": self.cleanup_candidate,
            "reason": self.reason,
            "age_days": self.age_days,
        }


def _string_sequence(value: Any, *, name: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        if type(item) is not str or not item:
            raise ValueError(f"{name}[{index}] must be a nonempty string")
        result.append(item)
    return tuple(result)


def _parse_utc(value: Any, *, name: str) -> datetime:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty ISO-8601 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} must be an ISO-8601 timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must use UTC")
    return parsed.astimezone(timezone.utc)


def _utc_text(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _repository_parts(repository: str) -> tuple[str, str]:
    parts = repository.split("/")
    if len(parts) != 2 or any(not part for part in parts):
        raise ValueError("repository must use owner/name syntax")
    if any(part in {".", ".."} for part in parts):
        raise ValueError("repository owner and name must be ordinary path segments")
    return parts[0], parts[1]


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_allowlist(path: str | Path) -> BranchAllowlist:
    source = Path(path)
    if source.is_symlink() or not source.is_file():
        raise ValueError("branch allowlist must be an ordinary file")
    payload = json.loads(
        source.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_object,
        parse_constant=lambda value: (_ for _ in ()).throw(
            ValueError(f"non-finite JSON constant: {value}")
        ),
    )
    if not isinstance(payload, Mapping):
        raise ValueError("branch allowlist must be a JSON object")
    return BranchAllowlist.from_mapping(payload)


def classify_branch(
    inspection: BranchInspection,
    *,
    now: datetime,
    minimum_age_days: int,
) -> BranchDecision:
    if minimum_age_days < 1:
        raise ValueError("minimum_age_days must be positive")
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    age = now.astimezone(timezone.utc) - inspection.committed_at
    age_days = age.total_seconds() / 86_400.0
    if age.total_seconds() < 0.0:
        return BranchDecision(inspection, False, "future_commit_timestamp", age_days)
    if inspection.protected:
        return BranchDecision(inspection, False, "protected_branch", age_days)
    if inspection.allowlisted:
        return BranchDecision(inspection, False, "allowlisted_branch", age_days)
    if inspection.open_pull_requests:
        return BranchDecision(inspection, False, "open_pull_request", age_days)
    if age < timedelta(days=minimum_age_days):
        return BranchDecision(inspection, False, "younger_than_threshold", age_days)
    if inspection.tip_reachable_from_default:
        return BranchDecision(inspection, True, "tip_reachable_from_default", age_days)
    if inspection.exact_tip_merged_pull_requests:
        return BranchDecision(
            inspection,
            True,
            "exact_tip_merged_in_pull_request",
            age_days,
        )
    return BranchDecision(inspection, False, "unmerged_tip", age_days)


def _commit_timestamp(payload: Any) -> str:
    if not isinstance(payload, Mapping):
        raise GitHubApiError("commit response must be a JSON object")
    commit = payload.get("commit")
    if not isinstance(commit, Mapping):
        raise GitHubApiError("commit response is missing commit metadata")
    for actor_name in ("committer", "author"):
        actor = commit.get(actor_name)
        if isinstance(actor, Mapping) and actor.get("date") is not None:
            return _utc_text(
                _parse_utc(actor["date"], name=f"commit.{actor_name}.date")
            )
    raise GitHubApiError("commit response has no usable UTC timestamp")


def _pull_request_numbers(
    pulls: Sequence[Any],
    *,
    repository: str,
    branch: str,
    sha: str | None = None,
    require_merged: bool = False,
) -> tuple[int, ...]:
    numbers: list[int] = []
    for pull in pulls:
        if not isinstance(pull, Mapping):
            continue
        head = pull.get("head")
        if not isinstance(head, Mapping):
            continue
        head_repo = head.get("repo")
        full_name = (
            head_repo.get("full_name") if isinstance(head_repo, Mapping) else None
        )
        if full_name != repository or head.get("ref") != branch:
            continue
        if sha is not None and head.get("sha") != sha:
            continue
        if require_merged and pull.get("merged_at") is None:
            continue
        number = pull.get("number")
        if type(number) is int:
            numbers.append(number)
    return tuple(sorted(set(numbers)))


def _tip_reachable_from_default(
    api: ApiClient,
    *,
    repository: str,
    sha: str,
    default_branch: str,
) -> bool:
    payload = api.get_json(
        (
            f"/repos/{repository}/compare/{quote(sha, safe='')}..."
            f"{quote(default_branch, safe='')}"
        )
    )
    if not isinstance(payload, Mapping):
        raise GitHubApiError("compare response must be a JSON object")
    merge_base = payload.get("merge_base_commit")
    return isinstance(merge_base, Mapping) and merge_base.get("sha") == sha


def _exact_tip_merged_pull_requests(
    api: ApiClient,
    *,
    repository: str,
    branch: str,
    sha: str,
) -> tuple[int, ...]:
    pulls = api.paginated(f"/repos/{repository}/commits/{sha}/pulls")
    return _pull_request_numbers(
        pulls,
        repository=repository,
        branch=branch,
        sha=sha,
        require_merged=True,
    )


def inspect_agent_branches(
    api: ApiClient,
    *,
    repository: str,
    prefix: str,
    allowlist: BranchAllowlist,
    minimum_age_days: int,
    now: datetime,
) -> tuple[str, tuple[BranchDecision, ...]]:
    if minimum_age_days < 1:
        raise ValueError("minimum_age_days must be positive")
    _repository_parts(repository)
    if not prefix or prefix.startswith("/") or prefix.endswith("//"):
        raise ValueError("prefix must be a nonempty branch-name prefix")
    repository_payload = api.get_json(f"/repos/{repository}")
    if not isinstance(repository_payload, Mapping):
        raise GitHubApiError("repository response must be a JSON object")
    default_branch = repository_payload.get("default_branch")
    if type(default_branch) is not str or not default_branch:
        raise GitHubApiError("repository response has no default branch")

    open_pulls = api.paginated(
        f"/repos/{repository}/pulls",
        query={"state": "open"},
    )
    branches = api.paginated(f"/repos/{repository}/branches")
    decisions: list[BranchDecision] = []
    for branch_payload in sorted(
        branches,
        key=lambda item: str(item.get("name")) if isinstance(item, Mapping) else "",
    ):
        if not isinstance(branch_payload, Mapping):
            raise GitHubApiError("branch response contains a non-object entry")
        name = branch_payload.get("name")
        if type(name) is not str or not name.startswith(prefix):
            continue
        if name == default_branch:
            continue
        commit = branch_payload.get("commit")
        sha = commit.get("sha") if isinstance(commit, Mapping) else None
        if type(sha) is not str or len(sha) != 40:
            raise GitHubApiError(f"branch {name} has no valid tip SHA")
        protected = branch_payload.get("protected") is True
        commit_payload = api.get_json(f"/repos/{repository}/commits/{sha}")
        preliminary = BranchInspection(
            name=name,
            sha=sha,
            committed_at_utc=_commit_timestamp(commit_payload),
            protected=protected,
            allowlisted=allowlist.matches(name),
            open_pull_requests=_pull_request_numbers(
                open_pulls,
                repository=repository,
                branch=name,
            ),
            exact_tip_merged_pull_requests=(),
            tip_reachable_from_default=False,
        )
        preliminary_decision = classify_branch(
            preliminary,
            now=now,
            minimum_age_days=minimum_age_days,
        )
        if preliminary_decision.reason in {
            "protected_branch",
            "allowlisted_branch",
            "open_pull_request",
            "younger_than_threshold",
            "future_commit_timestamp",
        }:
            decisions.append(preliminary_decision)
            continue
        reachable = _tip_reachable_from_default(
            api,
            repository=repository,
            sha=sha,
            default_branch=default_branch,
        )
        merged_numbers = (
            ()
            if reachable
            else _exact_tip_merged_pull_requests(
                api,
                repository=repository,
                branch=name,
                sha=sha,
            )
        )
        decisions.append(
            classify_branch(
                BranchInspection(
                    name=name,
                    sha=sha,
                    committed_at_utc=preliminary.committed_at_utc,
                    protected=protected,
                    allowlisted=allowlist.matches(name),
                    open_pull_requests=preliminary.open_pull_requests,
                    exact_tip_merged_pull_requests=merged_numbers,
                    tip_reachable_from_default=reachable,
                ),
                now=now,
                minimum_age_days=minimum_age_days,
            )
        )
    return default_branch, tuple(decisions)


def build_report(
    *,
    repository: str,
    default_branch: str,
    prefix: str,
    minimum_age_days: int,
    now: datetime,
    decisions: Sequence[BranchDecision],
) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    for decision in decisions:
        reason_counts[decision.reason] = reason_counts.get(decision.reason, 0) + 1
    candidate_count = sum(decision.cleanup_candidate for decision in decisions)
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DStaleAgentBranchReport",
        "generated_at_utc": _utc_text(now),
        "repository": repository,
        "default_branch": default_branch,
        "prefix": prefix,
        "minimum_age_days": minimum_age_days,
        "report_only": True,
        "branch_count": len(decisions),
        "cleanup_candidate_count": candidate_count,
        "excluded_count": len(decisions) - candidate_count,
        "reason_counts": dict(sorted(reason_counts.items())),
        "decisions": [decision.as_dict() for decision in decisions],
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Stale agent branch report",
        "",
        "This workflow is read-only. It never deletes or updates a branch.",
        "",
        f"- Repository: `{report['repository']}`",
        f"- Default branch: `{report['default_branch']}`",
        f"- Prefix: `{report['prefix']}`",
        f"- Minimum age: `{report['minimum_age_days']}` days",
        f"- Inspected: `{report['branch_count']}`",
        f"- Manual cleanup candidates: `{report['cleanup_candidate_count']}`",
        "",
        "## Manual cleanup candidates",
        "",
    ]
    candidates = [item for item in report["decisions"] if item["cleanup_candidate"]]
    if candidates:
        lines.extend(
            f"- `{item['name']}` at `{item['sha']}`: {item['reason']} "
            f"({item['age_days']:.1f} days old)"
            for item in candidates
        )
    else:
        lines.append("No branch is a cleanup candidate under the fail-closed policy.")
    lines.extend(["", "## Exclusion counts", ""])
    lines.extend(
        f"- `{reason}`: {count}"
        for reason, count in report["reason_counts"].items()
        if reason
        not in {
            "tip_reachable_from_default",
            "exact_tip_merged_in_pull_request",
        }
    )
    lines.extend(
        [
            "",
            "Review the report before removing any candidate manually through GitHub.",
            "",
        ]
    )
    return "\n".join(lines)


def _atomic_write(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def _now_from_argument(value: str | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    return _parse_utc(value, name="--now")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository",
        default=os.environ.get("GITHUB_REPOSITORY"),
        help="Repository in owner/name form; defaults to GITHUB_REPOSITORY.",
    )
    parser.add_argument("--prefix", default="agent/")
    parser.add_argument(
        "--allowlist",
        default=".github/stale-agent-branch-allowlist.json",
    )
    parser.add_argument("--minimum-age-days", type=int, default=30)
    parser.add_argument("--token-env", default="GITHUB_TOKEN")
    parser.add_argument("--now", help=argparse.SUPPRESS)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    arguments = parser.parse_args(argv)

    if not arguments.repository:
        parser.error("--repository or GITHUB_REPOSITORY is required")
    repository = str(arguments.repository)
    _repository_parts(repository)
    token = os.environ.get(arguments.token_env, "")
    if not token:
        parser.error(f"environment variable {arguments.token_env} is required")
    now = _now_from_argument(arguments.now)
    default_branch, decisions = inspect_agent_branches(
        GitHubApi(token),
        repository=repository,
        prefix=arguments.prefix,
        allowlist=load_allowlist(arguments.allowlist),
        minimum_age_days=arguments.minimum_age_days,
        now=now,
    )
    report = build_report(
        repository=repository,
        default_branch=default_branch,
        prefix=arguments.prefix,
        minimum_age_days=arguments.minimum_age_days,
        now=now,
        decisions=decisions,
    )
    _atomic_write(
        arguments.output_json,
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _atomic_write(arguments.output_markdown, render_markdown(report))
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
