#!/usr/bin/env python3
"""Verify a target-blind PokeFlex probe/challenge fold audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d.pokeflex_probe_challenge_fold_audit"
SCHEMA_VERSION = 1
VERIFY_SCHEMA = "causal4d.pokeflex_probe_challenge_fold_audit_verification"
VERIFY_VERSION = 1
FORBIDDEN_ARCHIVE_KEYS = {
    "member_names",
    "members",
    "member_payload",
    "payload",
    "payload_values",
    "samples",
    "response",
    "challenge_outcome",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--selection-salt", required=True)
    parser.add_argument("--expected-archives", type=int, default=170)
    parser.add_argument("--expected-poking", type=int, default=116)
    parser.add_argument("--expected-dropping", type=int, default=54)
    parser.add_argument("--minimum-eligible-objects", type=int, default=12)
    return parser.parse_args()


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def main() -> int:
    args = parse_args()
    result = json.loads(args.audit.read_text(encoding="utf-8"))
    require(result.get("schema") == SCHEMA, "unexpected audit schema")
    require(result.get("schema_version") == SCHEMA_VERSION, "unexpected audit version")
    require(result.get("request_id") == args.request_id, "request id mismatch")
    require(result.get("selection_salt") == args.selection_salt, "selection salt mismatch")

    stored_id = result.get("audit_id")
    require(isinstance(stored_id, str) and len(stored_id) == 64, "invalid audit id")
    identity_payload = {key: value for key, value in result.items() if key != "audit_id"}
    recomputed_id = hashlib.sha256(canonical_json(identity_payload)).hexdigest()
    require(stored_id == recomputed_id, "audit id mismatch")

    boundary = result.get("information_boundary", {})
    expected_boundary = {
        "archive_member_payload_opened": False,
        "archive_member_payload_bytes_read": 0,
        "archive_member_decompressed": False,
        "archive_member_extracted": False,
        "target_response_payload_used": False,
        "challenge_outcome_used": False,
    }
    require(boundary == expected_boundary, "information boundary changed")

    summary = result.get("summary", {})
    require(summary.get("archive_count") == args.expected_archives, "archive count mismatch")
    require(summary.get("audited_archive_count") == args.expected_archives, "not all archives audited")
    require(summary.get("poking_count") == args.expected_poking, "poking count mismatch")
    require(summary.get("dropping_count") == args.expected_dropping, "dropping count mismatch")
    require(summary.get("unknown_count") == 0, "unknown action archives remain")
    require(
        summary.get("dual_query_eligible_objects", 0) >= args.minimum_eligible_objects,
        "insufficient dual-query eligible objects",
    )
    require(summary.get("suspicious_member_path_count") == 0, "unsafe ZIP member path found")
    require(not result.get("archive_errors"), "archive metadata errors remain")
    require(result.get("status") == "ready-for-source-only-protocol", "audit not ready")
    decision = result.get("decision", {})
    require(decision.get("proceed") is True, "audit did not authorize next stage")
    require(
        decision.get("next_stage")
        == "freeze-source-only-action-and-response-carrier-contract",
        "unexpected next-stage decision",
    )

    archives = result.get("archives")
    require(isinstance(archives, list) and len(archives) == args.expected_archives, "archive records invalid")
    archive_take_ids: set[str] = set()
    for record in archives:
        require(isinstance(record, dict), "archive record must be an object")
        forbidden = FORBIDDEN_ARCHIVE_KEYS.intersection(record)
        require(not forbidden, f"archive record exposes forbidden payload key(s): {sorted(forbidden)}")
        require(
            set(record).issuperset(
                {
                    "relative_path",
                    "take_id",
                    "object_id",
                    "action_class",
                    "member_count",
                    "member_name_sha256",
                    "central_directory_crc_sha256",
                }
            ),
            "archive record missing metadata field",
        )
        take_key = f"{record['object_id']}::{record['take_id']}"
        require(take_key not in archive_take_ids, f"duplicate object/take identity: {take_key}")
        archive_take_ids.add(take_key)

    panels = result.get("object_panels")
    folds = result.get("frozen_folds")
    require(isinstance(panels, list), "object panels missing")
    require(isinstance(folds, list), "frozen folds missing")
    eligible_objects = {p["object_id"] for p in panels if p.get("eligible_dual_query")}
    require(len(eligible_objects) == summary["dual_query_eligible_objects"], "eligible-object count mismatch")
    require(len(folds) == 2 * len(eligible_objects), "expected two folds per eligible object")

    by_object: dict[str, list[dict[str, Any]]] = {}
    for fold in folds:
        require(isinstance(fold, dict), "fold must be an object")
        object_id = fold.get("object_id")
        require(object_id in eligible_objects, "fold belongs to ineligible object")
        candidates = fold.get("candidate_probe_take_ids")
        require(isinstance(candidates, list) and candidates, "candidate probe roster empty")
        require(len(candidates) == len(set(candidates)), "duplicate candidate probe")
        calibration = fold.get("calibration_take_id")
        challenge = fold.get("challenge_take_id")
        require(calibration not in candidates, "calibration take leaked into candidates")
        require(challenge not in candidates, "challenge take leaked into candidates")
        require(calibration != challenge, "calibration and challenge are identical")
        require(
            fold.get("query_id") in {"held-poke-response", "held-drop-response"},
            "unexpected query id",
        )
        by_object.setdefault(object_id, []).append(fold)

    for object_id in eligible_objects:
        object_folds = by_object.get(object_id, [])
        require(len(object_folds) == 2, f"{object_id}: expected two query folds")
        require(
            {fold["query_id"] for fold in object_folds}
            == {"held-poke-response", "held-drop-response"},
            f"{object_id}: missing registered query fold",
        )
        probe_rosters = {tuple(fold["candidate_probe_take_ids"]) for fold in object_folds}
        require(len(probe_rosters) == 1, f"{object_id}: query folds use different probe rosters")
        calibrations = {fold["calibration_take_id"] for fold in object_folds}
        require(len(calibrations) == 1, f"{object_id}: query folds use different calibration take")

    verification = {
        "schema": VERIFY_SCHEMA,
        "schema_version": VERIFY_VERSION,
        "status": "verified-ready-for-source-only-protocol",
        "audit_id": stored_id,
        "request_id": args.request_id,
        "selection_salt": args.selection_salt,
        "summary": {
            "archive_count": summary["archive_count"],
            "object_count": summary["object_count"],
            "dual_query_eligible_objects": summary["dual_query_eligible_objects"],
            "frozen_fold_count": summary["frozen_fold_count"],
        },
        "information_boundary_verified": True,
        "claim_boundary": [
            "Verification covers metadata-only roster feasibility.",
            "It does not establish probe value, prediction gain, or online control.",
        ],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(verification, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
