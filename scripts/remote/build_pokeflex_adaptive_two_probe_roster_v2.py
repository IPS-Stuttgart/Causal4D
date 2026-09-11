#!/usr/bin/env python3
"""Build a metadata-only PokeFlex adaptive two-probe structural roster."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

AUDIT_SCHEMA = "causal4d.pokeflex_probe_challenge_fold_audit"
OUTPUT_SCHEMA = "causal4d/pokeflex-adaptive-two-probe-structural-roster"
PROTOCOL_ID = "pokeflex-adaptive-two-probe-drop-protocol-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--protocol-json", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def salted_order(
    values: list[str],
    *,
    salt: str,
    object_id: str,
) -> list[str]:
    return sorted(
        values,
        key=lambda value: (
            hashlib.sha256(
                f"{salt}\0candidate\0{object_id}\0{value}".encode("utf-8")
            ).hexdigest(),
            value,
        ),
    )


def role_map(protocol: dict[str, Any]) -> dict[str, str]:
    registered = protocol["object_split"]["expected_if_all_18_objects_are_eligible"]
    roles: dict[str, str] = {}
    for values in registered["target"].values():
        for object_id in values:
            roles[object_id] = "target"
    for values in registered["calibration"].values():
        for object_id in values:
            roles[object_id] = "calibration"
    for object_id in registered["source"]:
        roles[object_id] = "source"
    return roles


def family_map(protocol: dict[str, Any]) -> dict[str, str]:
    registered = protocol["object_split"]["expected_if_all_18_objects_are_eligible"]
    families: dict[str, str] = {}
    for family, values in registered["target"].items():
        for object_id in values:
            families[object_id] = family
    for family, values in registered["calibration"].items():
        for object_id in values:
            families[object_id] = family
    for object_id in registered["source"]:
        if object_id.startswith("3dPrinted"):
            families[object_id] = "printed"
        elif "Foam" in object_id or object_id in {
            "Sponge",
            "ToiletPaperRoll",
        }:
            families[object_id] = "foam"
        else:
            families[object_id] = "soft"
    return families


def complete_pokes(panel: dict[str, Any]) -> list[str]:
    values = list(panel["candidate_probe_take_ids"])
    for key in ("calibration_poke_take_id", "poke_challenge_take_id"):
        value = panel.get(key)
        if value is not None:
            values.append(value)
    return sorted(set(values))


def build_roster(
    audit: dict[str, Any],
    protocol: dict[str, Any],
) -> dict[str, Any]:
    require(audit.get("schema") == AUDIT_SCHEMA, "unexpected audit schema")
    require(audit.get("schema_version") == 1, "unexpected audit version")
    require(
        audit.get("audit_id") == protocol["dataset"]["metadata_audit_id"],
        "audit identity does not match protocol",
    )
    require(
        audit.get("decision", {}).get("proceed") is True,
        "metadata audit did not pass",
    )
    boundary = audit.get("information_boundary", {})
    require(
        boundary.get("archive_member_payload_opened") is False,
        "audit opened archive payload",
    )
    require(
        boundary.get("archive_member_payload_bytes_read") == 0,
        "audit read archive payload bytes",
    )
    require(
        boundary.get("challenge_outcome_used") is False,
        "audit used challenge outcome",
    )

    summary = audit["summary"]
    require(summary["archive_count"] == 170, "archive count changed")
    require(summary["poking_count"] == 116, "poking count changed")
    require(summary["dropping_count"] == 54, "dropping count changed")
    require(summary["object_count"] == 18, "object count changed")

    roles = role_map(protocol)
    families = family_map(protocol)
    require(len(roles) == 18, "protocol roles do not cover 18 objects")
    require(set(roles) == set(families), "family and role maps disagree")

    panels = {panel["object_id"]: panel for panel in audit["object_panels"]}
    require(set(panels) == set(roles), "audit object roster changed")

    drops_by_object: dict[str, list[str]] = defaultdict(list)
    for archive in audit["archives"]:
        if archive["action_class"] == "dropping" and archive["has_state_carrier"]:
            drops_by_object[archive["object_id"]].append(archive["take_id"])

    roles_config = protocol["take_roles"]
    candidate_count = roles_config["target_candidate_probe_count"]
    minimum_source = roles_config["minimum_source_candidate_probe_count"]
    salt = roles_config["candidate_probe_selection_salt"]

    objects: list[dict[str, Any]] = []
    checks: dict[str, bool] = {}
    for object_id in sorted(roles):
        panel = panels[object_id]
        available_pokes = complete_pokes(panel)
        ordered = salted_order(
            available_pokes,
            salt=salt,
            object_id=object_id,
        )
        library = ordered[:candidate_count]
        ordered_pairs = [
            [first, second]
            for first in library
            for second in library
            if second != first
        ]
        drops = sorted(set(drops_by_object[object_id]))
        role = roles[object_id]
        enough_pokes = len(available_pokes) >= minimum_source
        enough_drops = (
            len(drops) >= roles_config["minimum_target_drops_per_object"]
            if role == "target"
            else True
        )
        exact_library = len(library) == candidate_count
        exact_pair_count = len(ordered_pairs) == (
            candidate_count * (candidate_count - 1)
        )
        checks[f"{object_id}:minimum-complete-pokes"] = enough_pokes
        checks[f"{object_id}:exact-four-probe-library"] = exact_library
        checks[f"{object_id}:twelve-ordered-two-probe-paths"] = exact_pair_count
        checks[f"{object_id}:target-drop-support"] = enough_drops
        objects.append(
            {
                "object_id": object_id,
                "family": families[object_id],
                "role": role,
                "available_complete_poke_count": len(available_pokes),
                "candidate_probe_take_ids": library,
                "first_stage_choice_count": len(library),
                "second_stage_choices_per_first": max(len(library) - 1, 0),
                "ordered_distinct_probe_pairs": ordered_pairs,
                "ordered_distinct_probe_pair_count": len(ordered_pairs),
                "complete_drop_take_ids": drops,
                "complete_drop_count": len(drops),
                "structurally_feasible": (
                    enough_pokes and exact_library and exact_pair_count and enough_drops
                ),
            }
        )

    counts = {
        role: sum(item["role"] == role for item in objects)
        for role in ("source", "calibration", "target")
    }
    checks["role-count-source-nine"] = counts["source"] == 9
    checks["role-count-calibration-three"] = counts["calibration"] == 3
    checks["role-count-target-six"] = counts["target"] == 6
    checks["all-targets-have-four-first-stage-choices"] = all(
        item["first_stage_choice_count"] == 4
        for item in objects
        if item["role"] == "target"
    )
    checks["all-targets-have-three-second-stage-choices-per-first"] = all(
        item["second_stage_choices_per_first"] == 3
        for item in objects
        if item["role"] == "target"
    )
    checks["all-targets-have-twelve-ordered-probe-pairs"] = all(
        item["ordered_distinct_probe_pair_count"] == 12
        for item in objects
        if item["role"] == "target"
    )
    checks["all-targets-have-at-least-two-drop-challenges"] = all(
        item["complete_drop_count"] >= 2 for item in objects if item["role"] == "target"
    )
    proceed = all(checks.values())

    result: dict[str, Any] = {
        "schema": OUTPUT_SCHEMA,
        "schema_version": 1,
        "status": (
            "adaptive-two-probe-structure-ready"
            if proceed
            else "adaptive-two-probe-structure-not-ready"
        ),
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": protocol["protocol_sha256"],
        "metadata_audit_id": audit["audit_id"],
        "metadata_identity_sha256": audit["dataset"]["metadata_identity_sha256"],
        "information_boundary": {
            "archive_member_payload_opened": False,
            "archive_member_payload_bytes_read": 0,
            "probe_response_used": False,
            "drop_outcome_used": False,
        },
        "summary": {
            "object_count": len(objects),
            "source_object_count": counts["source"],
            "calibration_object_count": counts["calibration"],
            "target_object_count": counts["target"],
            "candidate_probe_count_per_object": candidate_count,
            "ordered_two_probe_paths_per_object": (
                candidate_count * (candidate_count - 1)
            ),
            "target_query_count": len(protocol["registered_queries"]),
            "target_object_query_count": (
                counts["target"] * len(protocol["registered_queries"])
            ),
        },
        "checks": checks,
        "objects": objects,
        "decision": {
            "proceed": proceed,
            "next_stage": (
                "exact-historical-exposure-scan-and-source-only-modeling"
                if proceed
                else "repair-probe-or-drop-roster-before-payload-access"
            ),
        },
        "claim_boundary": [
            "Structural feasibility only; no probe response was read.",
            "No drop outcome, target score, or policy value was computed.",
            "Logged two-probe paths do not establish online sequential execution.",
        ],
    }
    result["roster_id"] = canonical_sha256(
        {key: value for key, value in result.items() if key != "roster_id"}
    )
    return result


def markdown_report(result: dict[str, Any]) -> str:
    summary = result["summary"]
    lines = [
        "# PokeFlex adaptive two-probe structural roster",
        "",
        f"- Status: `{result['status']}`",
        f"- Roster ID: `{result['roster_id']}`",
        f"- Objects: {summary['object_count']}",
        (
            "- Source / calibration / target objects: "
            f"{summary['source_object_count']} / "
            f"{summary['calibration_object_count']} / "
            f"{summary['target_object_count']}"
        ),
        (
            "- Candidate probes / ordered distinct pairs per object: "
            f"{summary['candidate_probe_count_per_object']} / "
            f"{summary['ordered_two_probe_paths_per_object']}"
        ),
        "- Archive member payload bytes read: 0",
        "- Probe responses used: false",
        "- Drop outcomes used: false",
        "",
        "## Target panel",
        "",
        (
            "| Object | Family | Complete pokes | Candidate probes | "
            "Ordered pairs | Drops |"
        ),
        "|---|---|---:|---:|---:|---:|",
    ]
    for item in result["objects"]:
        if item["role"] != "target":
            continue
        lines.append(
            f"| `{item['object_id']}` | {item['family']} | "
            f"{item['available_complete_poke_count']} | "
            f"{item['first_stage_choice_count']} | "
            f"{item['ordered_distinct_probe_pair_count']} | "
            f"{item['complete_drop_count']} |"
        )
    lines.extend(
        [
            "",
            "## Decision",
            "",
            f"- Proceed: `{str(result['decision']['proceed']).lower()}`",
            f"- Next stage: `{result['decision']['next_stage']}`",
            "",
            (
                "This result proves only that the frozen public-data roster can "
                "support response-conditioned selection of a second distinct "
                "logged poke. It does not establish probe value."
            ),
        ]
    )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    audit = json.loads(args.audit_json.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol_json.read_text(encoding="utf-8"))
    result = build_roster(audit, protocol)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.output_md.write_text(
        markdown_report(result) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "roster_id": result["roster_id"],
                "summary": result["summary"],
                "decision": result["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["decision"]["proceed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
