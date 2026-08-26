from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIR = ROOT / ".github" / "workflows"
REGISTRY = ROOT / ".github" / "self-hosted-jobs.json"
JOB_HEADER = re.compile(r"^  (?P<job>[A-Za-z0-9_-]+):\s*$")
PINNED_ACTION = re.compile(r"[0-9a-f]{40}")


def _job_blocks(text: str) -> dict[str, str]:
    lines = text.splitlines(keepends=True)
    try:
        jobs_start = next(
            index for index, line in enumerate(lines) if line.rstrip() == "jobs:"
        )
    except StopIteration:
        return {}

    starts: list[tuple[str, int]] = []
    for index in range(jobs_start + 1, len(lines)):
        line = lines[index]
        if line and not line[0].isspace() and line.strip():
            break
        match = JOB_HEADER.match(line.rstrip("\n"))
        if match is not None:
            starts.append((match.group("job"), index))

    blocks: dict[str, str] = {}
    for position, (job, start) in enumerate(starts):
        end = starts[position + 1][1] if position + 1 < len(starts) else len(lines)
        blocks[job] = "".join(lines[start:end])
    return blocks


def _job_property(block: str, name: str) -> str:
    lines = block.splitlines(keepends=True)
    prefix = f"    {name}:"
    for start, line in enumerate(lines):
        if not line.startswith(prefix):
            continue
        collected = [line]
        for candidate in lines[start + 1 :]:
            if re.match(r"^    [A-Za-z0-9_-]+:\s*", candidate):
                break
            collected.append(candidate)
        return "".join(collected)
    return ""


def _uses_self_hosted_runner(block: str) -> bool:
    runs_on = _job_property(block, "runs-on")
    if "self-hosted" in runs_on:
        return True
    strategy = _job_property(block, "strategy")
    return "matrix." in runs_on and "self-hosted" in strategy


def _discover_self_hosted_jobs() -> dict[tuple[str, str], tuple[str, str]]:
    discovered: dict[tuple[str, str], tuple[str, str]] = {}
    paths = sorted((*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")))
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for job, block in _job_blocks(text).items():
            if _uses_self_hosted_runner(block):
                discovered[(path.name, job)] = (text, block)
    return discovered


def _single_event_workflow(text: str, event: str) -> bool:
    prefix = text.split("permissions:", maxsplit=1)[0]
    if re.search(rf"^  {event}:\s*$", prefix, re.MULTILINE) is None:
        return False
    other_events = (
        "issues",
        "pull_request",
        "push",
        "schedule",
        "workflow_call",
        "workflow_dispatch",
    )
    return all(
        candidate == event
        or re.search(rf"^  {candidate}:\s*$", prefix, re.MULTILINE) is None
        for candidate in other_events
    )


def _dispatch_only_workflow(text: str) -> bool:
    return _single_event_workflow(text, "workflow_dispatch")


def _issue_only_workflow(text: str) -> bool:
    return _single_event_workflow(text, "issues")


def _common_self_hosted_errors(block: str) -> list[str]:
    errors: list[str] = []
    if "github.ref == 'refs/heads/main'" not in block:
        errors.append("missing job-level main guard")
    if "ref: ${{ github.sha }}" not in block:
        errors.append("checkout is not bound to github.sha")
    if "git rev-parse HEAD" not in block or "GITHUB_SHA" not in block:
        errors.append("exact checkout SHA is not verified")
    if "git status --porcelain" not in block:
        errors.append("clean checkout is not verified")

    checkout_count = block.count("uses: actions/checkout@")
    if block.count("persist-credentials: false") < checkout_count:
        errors.append("one or more checkouts retain credentials")

    for line in block.splitlines():
        stripped = line.strip()
        match = re.match(r"-?\s*uses:\s*(?P<target>[^#\s]+)", stripped)
        if match is None:
            continue
        target = match.group("target")
        if target.startswith("./"):
            continue
        if (
            "@" not in target
            or PINNED_ACTION.fullmatch(target.rsplit("@", 1)[1]) is None
        ):
            errors.append(f"action is not pinned by full commit SHA: {target}")

    if "${{ secrets." in block:
        errors.append("self-hosted job references a GitHub secret")
    permissions = _job_property(block, "permissions")
    for forbidden in ("contents: write", "issues: write", "pull-requests: write"):
        if forbidden in permissions:
            errors.append(f"self-hosted job requests {forbidden}")
    return errors


def _main_only_errors(workflow_text: str, block: str) -> list[str]:
    errors = _common_self_hosted_errors(block)
    if (
        "github.event_name == 'workflow_dispatch'" not in block
        and not _dispatch_only_workflow(workflow_text)
    ):
        errors.append("missing dispatch-only authorization")
    return errors


def _maintainer_issue_main_errors(workflow_text: str, block: str) -> list[str]:
    errors = _common_self_hosted_errors(block)
    required = {
        "github.event_name == 'issues'": "missing issue-event authorization",
        "github.event.action == 'opened'": "missing issue-open authorization",
        (
            "github.event.issue.user.login == 'FlorianPfaff'"
        ): "missing exact maintainer-login authorization",
        (
            "github.event.issue.user.id == 6773539"
        ): "missing exact maintainer-ID authorization",
        (
            "'[self-hosted] validate prepared joint observation'"
        ): "missing exact trigger-title authorization",
    }
    for fragment, message in required.items():
        if fragment not in block:
            errors.append(message)
    if block.count("github.event.issue.title") != 1:
        errors.append("issue title must be used only by the exact guard")
    for forbidden in (
        "github.event.issue.body",
        "github.event.issue.labels",
        "github.event.comment",
    ):
        if forbidden in block:
            errors.append(
                f"untrusted issue payload reaches self-hosted job: {forbidden}"
            )
    if not _issue_only_workflow(workflow_text):
        errors.append("workflow is not issue-only")
    return errors


def _single_operator_v5_issue_main_errors(
    workflow_text: str,
    block: str,
) -> list[str]:
    errors = _common_self_hosted_errors(block)
    required = {
        "github.event_name == 'issues'": "missing issue-event authorization",
        "github.event.action == 'opened'": "missing issue-open authorization",
        (
            "github.event.issue.user.login == 'FlorianPfaff'"
        ): "missing exact maintainer-login authorization",
        (
            "github.event.issue.user.id == 6773539"
        ): "missing exact maintainer-ID authorization",
        (
            "'[self-hosted] bootstrap Causal4D v5 owner identity scaffold v2'"
        ): "missing exact trigger-title authorization",
    }
    for fragment, message in required.items():
        if fragment not in block:
            errors.append(message)
    if block.count("github.event.issue.title") != 1:
        errors.append("issue title must be used only by the exact guard")
    for forbidden in (
        "github.event.issue.body",
        "github.event.issue.labels",
        "github.event.comment",
    ):
        if forbidden in block:
            errors.append(
                f"untrusted issue payload reaches self-hosted job: {forbidden}"
            )
    if not _issue_only_workflow(workflow_text):
        errors.append("workflow is not issue-only")
    return errors


def _authorization_errors(
    authorization_model: str,
    workflow_text: str,
    block: str,
) -> list[str]:
    if authorization_model == "main-only":
        return _main_only_errors(workflow_text, block)
    if authorization_model == "maintainer-issue-main":
        return _maintainer_issue_main_errors(workflow_text, block)
    if authorization_model == "single-operator-v5-issue-main":
        return _single_operator_v5_issue_main_errors(workflow_text, block)
    return [f"unsupported authorization model: {authorization_model}"]


def test_self_hosted_job_registry_is_complete_and_unique() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert set(payload["authorization_models"]) == {
        "main-only",
        "maintainer-issue-main",
        "single-operator-v5-issue-main",
    }

    entries = payload["jobs"]
    keys = [(entry["workflow"], entry["job"]) for entry in entries]
    assert len(keys) == len(set(keys))
    assert keys == sorted(keys)
    assert set(keys) == set(_discover_self_hosted_jobs())


def test_every_self_hosted_job_is_exact_sha_and_secret_free() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    discovered = _discover_self_hosted_jobs()

    for entry in payload["jobs"]:
        key = (entry["workflow"], entry["job"])
        workflow_text, block = discovered[key]
        assert entry["authorization_model"] in payload["authorization_models"]
        assert entry["secrets_allowed"] is False
        assert "permissions:\n  contents: read\n" in workflow_text
        assert "contents: write" not in block
        assert "issues: write" not in block
        assert "pull-requests: write" not in block
        for label in entry["runner_labels"]:
            assert label in block
        assert (
            _authorization_errors(
                entry["authorization_model"],
                workflow_text,
                block,
            )
            == []
        ), key


def test_runner_discovery_ignores_hosted_jobs_that_only_mention_self_hosted() -> None:
    block = """  contract:
    runs-on: ubuntu-latest
    steps:
      - run: python tests/test_self_hosted_workflow_policy.py
"""
    assert _uses_self_hosted_runner(block) is False


def test_main_only_validator_accepts_a_reviewed_dispatch_fixture() -> None:
    workflow = """on:
  workflow_dispatch:
permissions:
  contents: read
"""
    block = """  evaluate:
    if: >-
      github.event_name == 'workflow_dispatch' &&
      github.ref == 'refs/heads/main'
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
      - run: |
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"
"""
    assert _main_only_errors(workflow, block) == []


def test_maintainer_issue_validator_accepts_exact_trigger() -> None:
    workflow = """on:
  issues:
    types: [opened]
permissions:
  contents: read
"""
    block = """  validate:
    if: >-
      github.event_name == 'issues' &&
      github.event.action == 'opened' &&
      github.ref == 'refs/heads/main' &&
      github.event.issue.user.login == 'FlorianPfaff' &&
      github.event.issue.user.id == 6773539 &&
      github.event.issue.title ==
        '[self-hosted] validate prepared joint observation'
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
      - run: |
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"
"""
    assert _maintainer_issue_main_errors(workflow, block) == []


def test_main_only_validator_rejects_an_unauthorized_or_stale_fixture() -> None:
    workflow = """on:
  workflow_dispatch:
  pull_request:
permissions:
  contents: read
"""
    block = """  evaluate:
    if: github.ref == 'refs/heads/feature'
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: actions/checkout@v7
"""
    errors = _main_only_errors(workflow, block)
    assert "missing job-level main guard" in errors
    assert "missing dispatch-only authorization" in errors
    assert "checkout is not bound to github.sha" in errors
    assert "exact checkout SHA is not verified" in errors
    assert "clean checkout is not verified" in errors
    assert "one or more checkouts retain credentials" in errors
    assert any(error.startswith("action is not pinned") for error in errors)


def test_maintainer_issue_validator_rejects_broad_issue_execution() -> None:
    workflow = """on:
  issues:
permissions:
  contents: read
"""
    block = """  validate:
    if: github.ref == 'refs/heads/main'
    runs-on: [self-hosted, Linux, X64]
    steps:
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1
        with:
          ref: ${{ github.sha }}
          persist-credentials: false
      - run: |
          test "$(git rev-parse HEAD)" = "${GITHUB_SHA}"
          test -z "$(git status --porcelain=v1)"
"""
    errors = _maintainer_issue_main_errors(workflow, block)
    assert "missing issue-event authorization" in errors
    assert "missing issue-open authorization" in errors
    assert "missing exact maintainer-login authorization" in errors
    assert "missing exact maintainer-ID authorization" in errors
    assert "missing exact trigger-title authorization" in errors


def test_maintainer_issue_reporter_is_hosted_and_has_no_checkout() -> None:
    path = WORKFLOW_DIR / "prepared-joint-observation-self-hosted.yml"
    text = path.read_text(encoding="utf-8")
    report = _job_blocks(text)["report"]

    assert "runs-on: ubuntu-latest" in report
    assert "needs: validate" in report
    assert "issues: write" in report
    assert "uses: actions/checkout@" not in report
    assert "runs-on: [self-hosted" not in report
