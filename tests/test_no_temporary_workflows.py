from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_DIRECTORY = ROOT / ".github" / "workflows"
TEMPORARY_WORKFLOW_STEMS = (
    "temporary-*",
    "publish-reviewed-*",
    "*one-shot*",
    "*one_shot*",
    "agent-apply-*",
    "agent_apply_*",
    "branch-advance-*",
    "branch_advance_*",
)


def test_no_one_shot_workflow_can_reach_a_mergeable_head() -> None:
    one_shot = sorted(
        {
            path.relative_to(ROOT).as_posix()
            for stem in TEMPORARY_WORKFLOW_STEMS
            for suffix in (".yml", ".yaml")
            for path in WORKFLOW_DIRECTORY.glob(f"{stem}{suffix}")
        }
    )
    assert one_shot == [], (
        "temporary workflows must be removed before review and merge; "
        f"use a reusable workflow plus immutable run inputs instead: {one_shot}"
    )
