#!/usr/bin/env python3
"""Verify a PokeFlex continuous probe-descriptor audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d/pokeflex-continuous-probe-descriptor-audit"
TARGET_OBJECTS = {
    "3dPrintedBunny",
    "3dPrintedCylinder",
    "Sponge",
    "MemoryFoam",
    "Beanbag",
    "Pillow",
}


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def verify(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == SCHEMA, "unexpected result schema")
    require(payload.get("schema_version") == 1, "unexpected result version")
    stored = payload.get("content_sha256")
    require(isinstance(stored, str) and len(stored) == 64, "bad result digest")
    canonical = dict(payload)
    canonical.pop("content_sha256", None)
    actual = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
    require(actual == stored, "result digest mismatch")

    objects = payload["source_and_calibration_objects"]
    require(len(objects) == 12 and len(set(objects)) == 12, "object panel changed")
    require(not (set(objects) & TARGET_OBJECTS), "target object appears in audit")
    require(payload["take_indices"] == [1, 2, 3, 4, 5, 6], "take roster changed")
    require(len(payload["records"]) == 72, "record panel is incomplete")
    observed = {
        (row["object_id"], int(row["take_index"])) for row in payload["records"]
    }
    expected = {(object_id, take) for object_id in objects for take in range(1, 7)}
    require(observed == expected, "record identities changed")

    prior = payload["prior_action_class_audit"]
    require(prior["gate_passed"] is False, "prior negative result changed")
    require(
        prior["content_sha256"]
        == "12450e44d7f32330d2d7608f3b13535e6373427e3bc60c0815038ccd10979247",
        "prior result identity changed",
    )

    boundary = payload["information_boundary"]
    require(boundary["target_archive_open_count"] == 0, "target archive was opened")
    require(boundary["drop_archive_open_count"] == 0, "drop archive was opened")
    require(boundary["other_member_open_count"] == 0, "unauthorized member was opened")
    require(boundary["robot_member_open_count"] == 72, "robot member count changed")
    require(
        boundary["initial_mesh_member_open_count"] == 72,
        "initial mesh member count changed",
    )
    require(boundary["challenge_outcome_read"] is False, "challenge outcome was read")

    tip = payload["tip_model"]
    require(int(tip["axis"]) in {0, 1, 2}, "tool axis is invalid")
    require(abs(float(tip["offset_m"])) <= 0.25, "tip offset exceeds frozen grid")
    require(len(payload["descriptor_names"]) == 11, "descriptor dimension changed")

    gate = payload["gate"]
    checks = gate["checks"]
    require(gate["passed"] == all(checks.values()), "gate/check mismatch")
    positive = gate["passed"]
    require(
        payload["status"]
        == (
            "continuous-probe-interface-qualified"
            if positive
            else "continuous-probe-interface-not-qualified"
        ),
        "status and gate differ",
    )
    require(
        gate["next_stage"]
        == (
            "run-source-only-continuous-sequential-probe-qualification"
            if positive
            else "retain-continuous-actions-but-revise-contact-localization"
        ),
        "next-stage decision changed",
    )

    validation = payload["contact_validation"]
    geometry = payload["descriptor_geometry"]
    return {
        "schema": "causal4d/pokeflex-continuous-probe-descriptor-audit-verification",
        "schema_version": 1,
        "status": "verified",
        "content_sha256": stored,
        "gate_passed": positive,
        "median_contact_index_error": validation[
            "median_absolute_contact_index_error"
        ],
        "p90_contact_index_error": validation["p90_absolute_contact_index_error"],
        "descriptor_intrinsic_rank": geometry["intrinsic_rank"],
        "target_archive_open_count": boundary["target_archive_open_count"],
        "drop_archive_open_count": boundary["drop_archive_open_count"],
        "challenge_outcome_read": boundary["challenge_outcome_read"],
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.result.read_text(encoding="utf-8"))
    result = verify(payload)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
