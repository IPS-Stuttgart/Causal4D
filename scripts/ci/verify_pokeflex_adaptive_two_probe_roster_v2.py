#!/usr/bin/env python3
"""Verify a PokeFlex adaptive two-probe structural roster."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d/pokeflex-adaptive-two-probe-structural-roster"
PROTOCOL_ID = "pokeflex-adaptive-two-probe-drop-protocol-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roster", type=Path)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_roster(
    roster: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    require(roster.get("schema") == SCHEMA, "unexpected roster schema")
    require(roster.get("schema_version") == 1, "unexpected roster version")
    require(
        roster.get("protocol_id") == PROTOCOL_ID,
        "unexpected protocol binding",
    )
    require(
        roster.get("protocol_sha256") == protocol.get("protocol_sha256"),
        "protocol digest binding changed",
    )
    require(
        roster.get("metadata_audit_id")
        == protocol["dataset"]["metadata_audit_id"],
        "metadata audit binding changed",
    )
    require(
        roster.get("status") == "adaptive-two-probe-structure-ready",
        "roster did not pass",
    )

    stored_id = roster.get("roster_id")
    require(
        isinstance(stored_id, str) and len(stored_id) == 64,
        "invalid roster id",
    )
    canonical = dict(roster)
    canonical.pop("roster_id", None)
    actual_id = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
    require(actual_id == stored_id, "roster digest mismatch")

    boundary = roster["information_boundary"]
    require(
        boundary
        == {
            "archive_member_payload_opened": False,
            "archive_member_payload_bytes_read": 0,
            "probe_response_used": False,
            "drop_outcome_used": False,
        },
        "information boundary changed",
    )
    require(
        roster["decision"]["proceed"] is True,
        "roster decision did not proceed",
    )

    summary = roster["summary"]
    require(summary["object_count"] == 18, "object count changed")
    require(summary["source_object_count"] == 9, "source count changed")
    require(
        summary["calibration_object_count"] == 3,
        "calibration count changed",
    )
    require(summary["target_object_count"] == 6, "target count changed")
    require(
        summary["candidate_probe_count_per_object"] == 4,
        "candidate count changed",
    )
    require(
        summary["ordered_two_probe_paths_per_object"] == 12,
        "ordered pair count changed",
    )
    require(summary["target_query_count"] == 2, "query count changed")
    require(
        summary["target_object_query_count"] == 12,
        "target object-query count changed",
    )

    objects = roster["objects"]
    require(len(objects) == 18, "object rows changed")
    require(
        len({item["object_id"] for item in objects}) == 18,
        "duplicate object rows",
    )
    targets = [item for item in objects if item["role"] == "target"]
    require(len(targets) == 6, "target rows changed")
    for item in objects:
        library = item["candidate_probe_take_ids"]
        require(len(library) == 4, f"{item['object_id']}: bad library size")
        require(
            len(set(library)) == 4,
            f"{item['object_id']}: duplicate probe identity",
        )
        pairs = item["ordered_distinct_probe_pairs"]
        require(len(pairs) == 12, f"{item['object_id']}: bad pair count")
        expected_pairs = {
            (first, second)
            for first in library
            for second in library
            if second != first
        }
        require(
            {tuple(pair) for pair in pairs} == expected_pairs,
            f"{item['object_id']}: ordered pair roster changed",
        )
        require(
            item["first_stage_choice_count"] == 4,
            f"{item['object_id']}: first-stage count changed",
        )
        require(
            item["second_stage_choices_per_first"] == 3,
            f"{item['object_id']}: second-stage count changed",
        )
        require(
            item["ordered_distinct_probe_pair_count"] == 12,
            f"{item['object_id']}: reported pair count changed",
        )
        require(
            item["structurally_feasible"] is True,
            f"{item['object_id']}: infeasible object retained",
        )
        if item["role"] == "target":
            require(
                item["complete_drop_count"] >= 2,
                f"{item['object_id']}: insufficient drop challenges",
            )

    require(
        all(roster["checks"].values()),
        "at least one structural check failed",
    )

    return {
        "schema": (
            "causal4d/pokeflex-adaptive-two-probe-structural-roster-verification"
        ),
        "schema_version": 1,
        "status": "verified-metadata-only-adaptive-two-probe-feasibility",
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": roster["protocol_sha256"],
        "metadata_audit_id": roster["metadata_audit_id"],
        "roster_id": stored_id,
        "target_object_count": 6,
        "candidate_probe_count_per_object": 4,
        "ordered_probe_pairs_per_object": 12,
        "probe_response_used": False,
        "drop_outcome_used": False,
        "claim_boundary_verified": True,
    }


def main() -> int:
    args = parse_args()
    roster = json.loads(args.roster.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    verification = verify_roster(roster, protocol)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(verification, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
