#!/usr/bin/env python3
"""Merge runner-local Deform360 fragment audits into an admission report."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "bayesian-phystwin-paper.deform360-fragment-audit/v1"
SCHEMA = "bayesian-phystwin-paper.deform360-fragment-admission/v1"


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != AUDIT_SCHEMA:
        raise ValueError(f"unexpected audit schema in {path}")
    if payload.get("source_only") is not True:
        raise ValueError(f"audit is not source-only: {path}")
    return payload


def _candidate(entry: dict[str, Any]) -> bool:
    return int(entry.get("structurally_processible_episode_count", 0)) >= 3


def _query_ready(entry: dict[str, Any]) -> bool:
    return (
        entry.get("source_kind") == "aligned"
        and int(entry.get("query_target_ready_episode_count", 0)) >= 3
    )


def _known_actions(entry: dict[str, Any]) -> set[str]:
    actions = set(entry.get("known_action_labels", []))
    metadata = entry.get("metadata", {})
    actions.update(metadata.get("known_action_labels", []))
    for episode in entry.get("episodes", []):
        actions.update(episode.get("metadata", {}).get("known_action_labels", []))
    return actions


def merge(paths: list[Path]) -> dict[str, Any]:
    reports = [_load(path) for path in paths]
    by_object: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in reports:
        for entry in report["objects"]:
            enriched = dict(entry)
            enriched["server_id"] = report["server_id"]
            by_object[entry["object_id"]].append(enriched)

    object_rows: list[dict[str, Any]] = []
    for object_id, entries in sorted(by_object.items()):
        processible = any(_candidate(entry) for entry in entries)
        query_ready = any(_query_ready(entry) for entry in entries)
        actions: set[str] = set()
        for entry in entries:
            actions.update(_known_actions(entry))
        object_rows.append(
            {
                "object_id": object_id,
                "servers": sorted({entry["server_id"] for entry in entries}),
                "source_kinds": sorted({entry["source_kind"] for entry in entries}),
                "maximum_processible_episode_count": max(
                    int(entry.get("structurally_processible_episode_count", 0))
                    for entry in entries
                ),
                "maximum_strong_360_episode_count": max(
                    int(entry.get("strong_360_episode_count", 0)) for entry in entries
                ),
                "maximum_query_ready_episode_count": max(
                    int(entry.get("query_target_ready_episode_count", 0))
                    for entry in entries
                ),
                "repeated_processible": processible,
                "repeated_query_ready_now": query_ready,
                "known_action_labels": sorted(actions),
                "episode_action_metadata_present": bool(actions),
                "sources": [
                    {
                        "server_id": entry["server_id"],
                        "root_label": entry["root_label"],
                        "source_kind": entry["source_kind"],
                        "path": entry["path"],
                    }
                    for entry in entries
                ],
            }
        )

    repeated = [row for row in object_rows if row["repeated_processible"]]
    repeated_with_actions = [
        row for row in repeated if row["episode_action_metadata_present"]
    ]
    query_ready = [row for row in object_rows if row["repeated_query_ready_now"]]
    action_support: dict[str, list[str]] = defaultdict(list)
    for row in repeated:
        for action in row["known_action_labels"]:
            action_support[action].append(row["object_id"])
    common_actions = {
        action: sorted(objects)
        for action, objects in sorted(action_support.items())
        if len(objects) >= 2
    }

    engineering_tiers = {
        "inventory_complete_for_listed_roots": all(
            root["exists"] for report in reports for root in report["roots"]
        ),
        "within_object_pipeline_pilot_after_processing": len(repeated) >= 1,
        "multi_object_source_pilot_after_processing": len(repeated) >= 5,
        "minimum_held_out_object_design_now": (
            len(query_ready) >= 12 and bool(common_actions)
        ),
        "minimum_held_out_object_design_after_processing": (
            len(repeated_with_actions) >= 12 and bool(common_actions)
        ),
        "headline_benefit_claim_admitted_by_inventory": False,
    }

    if engineering_tiers["minimum_held_out_object_design_now"]:
        conclusion = "minimum-held-out-object-design-ready-now"
    elif engineering_tiers["multi_object_source_pilot_after_processing"]:
        conclusion = "multi-object-source-pilot-ready-after-processing"
    elif engineering_tiers["within_object_pipeline_pilot_after_processing"]:
        conclusion = "within-object-pilot-ready-after-processing"
    else:
        conclusion = "insufficient-even-for-repeated-episode-pilot"

    return {
        "schema": SCHEMA,
        "source_only": True,
        "dataset_modified": False,
        "input_reports": [
            {
                "path": str(path),
                "server_id": report["server_id"],
                "object_count": report["object_count"],
            }
            for path, report in zip(paths, reports)
        ],
        "object_count": len(object_rows),
        "repeated_processible_object_count": len(repeated),
        "repeated_processible_with_action_metadata_count": len(repeated_with_actions),
        "repeated_query_ready_object_count": len(query_ready),
        "common_known_action_support": common_actions,
        "engineering_tiers": engineering_tiers,
        "conclusion": conclusion,
        "objects": object_rows,
        "claim_boundary": (
            "This inventory can admit source-only processing and pilot design. "
            "It cannot demonstrate task-conditioned intervention benefit, "
            "calibration, real-provider competence, or a held-out physical claim."
        ),
    }


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Deform360 fragment-admission audit",
        "",
        f"- Distinct object IDs: **{payload['object_count']}**",
        (
            "- Objects with at least three structurally processible episodes: "
            f"**{payload['repeated_processible_object_count']}**"
        ),
        (
            "- Repeated objects with recognized action metadata: "
            f"**{payload['repeated_processible_with_action_metadata_count']}**"
        ),
        (
            "- Objects already carrying at least three query-target-ready aligned "
            f"episodes: **{payload['repeated_query_ready_object_count']}**"
        ),
        f"- Admission conclusion: **`{payload['conclusion']}`**",
        "",
        "## Engineering tiers",
        "",
    ]
    for name, value in payload["engineering_tiers"].items():
        lines.append(f"- `{name}`: **{str(value).lower()}**")
    lines.extend(
        [
            "",
            "## Repeated-episode objects",
            "",
            "| Object | Max processible episodes | Query-ready episodes | Actions | Servers |",
            "|---|---:|---:|---|---|",
        ]
    )
    for row in payload["objects"]:
        if not row["repeated_processible"]:
            continue
        actions = ", ".join(row["known_action_labels"]) or "not recovered"
        servers = ", ".join(row["servers"])
        lines.append(
            f"| `{row['object_id']}` | "
            f"{row['maximum_processible_episode_count']} | "
            f"{row['maximum_query_ready_episode_count']} | "
            f"{actions} | {servers} |"
        )
    lines.extend(["", f"> {payload['claim_boundary']}", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    payload = merge(args.reports)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    args.output_markdown.write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "object_count": payload["object_count"],
                "repeated_processible_object_count": payload[
                    "repeated_processible_object_count"
                ],
                "repeated_query_ready_object_count": payload[
                    "repeated_query_ready_object_count"
                ],
                "conclusion": payload["conclusion"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
