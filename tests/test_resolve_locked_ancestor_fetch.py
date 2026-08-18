from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "ci" / "resolve_locked_ancestor_fetch.py"
SPEC = importlib.util.spec_from_file_location(
    "resolve_locked_ancestor_fetch",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


AncestorPlanError = MODULE.AncestorPlanError
main = MODULE.main
resolve_fetch_plan = MODULE.resolve_fetch_plan

PARENT = "1" * 40
HEAD = "2" * 40
OTHER = "3" * 40
REPOSITORY = "IPS-Stuttgart/Causal4D"


def _comparison(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": "ahead",
        "base_commit": {"sha": PARENT},
        "merge_base_commit": {"sha": PARENT},
        "head_commit": {"sha": HEAD},
        "ahead_by": 17,
        "behind_by": 0,
        "total_commits": 17,
    }
    payload.update(updates)
    return payload


def test_descendant_plan_fetches_exact_commit_distance_plus_parent() -> None:
    plan = resolve_fetch_plan(
        _comparison(),
        repository=REPOSITORY,
        head_sha=HEAD,
        required_parent_commit=PARENT,
    )

    assert plan.fetch_depth == 18
    assert plan.ahead_by == 17
    assert plan.merge_base_sha == PARENT
    assert plan.ancestry_verified_by == "github_compare_api_and_local_git"


def test_identical_parent_and_head_use_depth_one() -> None:
    payload = _comparison(
        status="identical",
        head_commit={"sha": PARENT},
        ahead_by=0,
        total_commits=0,
    )

    plan = resolve_fetch_plan(
        payload,
        repository=REPOSITORY,
        head_sha=PARENT,
        required_parent_commit=PARENT,
    )

    assert plan.fetch_depth == 1


def test_diverged_or_stale_comparison_fails_closed() -> None:
    with pytest.raises(AncestorPlanError, match="ancestor"):
        resolve_fetch_plan(
            _comparison(status="diverged", behind_by=2),
            repository=REPOSITORY,
            head_sha=HEAD,
            required_parent_commit=PARENT,
        )

    with pytest.raises(AncestorPlanError, match="workflow head"):
        resolve_fetch_plan(
            _comparison(head_commit={"sha": OTHER}),
            repository=REPOSITORY,
            head_sha=HEAD,
            required_parent_commit=PARENT,
        )


def test_wrong_merge_base_or_inconsistent_counts_fail_closed() -> None:
    with pytest.raises(AncestorPlanError, match="rooted"):
        resolve_fetch_plan(
            _comparison(merge_base_commit={"sha": OTHER}),
            repository=REPOSITORY,
            head_sha=HEAD,
            required_parent_commit=PARENT,
        )

    with pytest.raises(AncestorPlanError, match="counts"):
        resolve_fetch_plan(
            _comparison(total_commits=16),
            repository=REPOSITORY,
            head_sha=HEAD,
            required_parent_commit=PARENT,
        )


def test_excessive_history_is_rejected_instead_of_unshallowing() -> None:
    with pytest.raises(AncestorPlanError, match="bounded fetch-depth"):
        resolve_fetch_plan(
            _comparison(ahead_by=2048, total_commits=2048),
            repository=REPOSITORY,
            head_sha=HEAD,
            required_parent_commit=PARENT,
        )


def test_fixture_cli_writes_plan_and_prints_depth(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps({"required_parent_commit": PARENT}),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(_comparison()), encoding="utf-8")
    output = tmp_path / "plan.json"

    status = main(
        [
            "--repository",
            REPOSITORY,
            "--head-sha",
            HEAD,
            "--lock",
            str(lock),
            "--compare-json",
            str(comparison),
            "--output",
            str(output),
        ]
    )

    assert status == 0
    assert capsys.readouterr().out == "18\n"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["artifact_kind"] == "Causal4DLockedAncestorFetchPlan"
    assert payload["fetch_depth"] == 18
    assert payload["required_parent_commit"] == PARENT


def test_cli_rejects_duplicate_lock_keys(tmp_path: Path) -> None:
    lock = tmp_path / "lock.json"
    lock.write_text(
        '{"required_parent_commit":"' + PARENT + '",'
        '"required_parent_commit":"' + PARENT + '"}',
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(json.dumps(_comparison()), encoding="utf-8")

    status = main(
        [
            "--repository",
            REPOSITORY,
            "--head-sha",
            HEAD,
            "--lock",
            str(lock),
            "--compare-json",
            str(comparison),
            "--output",
            str(tmp_path / "plan.json"),
        ]
    )

    assert status == 1
    assert not (tmp_path / "plan.json").exists()
