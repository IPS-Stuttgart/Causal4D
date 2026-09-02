#!/usr/bin/env python3
"""Verify a source-only PokeFlex probe action-class audit artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d/pokeflex-probe-action-class-audit"
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

    boundary = payload["information_boundary"]
    require(boundary["target_archive_open_count"] == 0, "target archive was opened")
    require(boundary["non_robot_member_open_count"] == 0, "non-robot member was opened")
    require(boundary["mesh_payload_bytes_read"] == 0, "mesh payload was read")
    require(boundary["image_payload_bytes_read"] == 0, "image payload was read")
    require(boundary["drop_payload_bytes_read"] == 0, "drop payload was read")
    require(boundary["challenge_outcome_read"] is False, "challenge outcome was read")
    require(boundary["robot_member_open_count"] == 72, "robot member count changed")

    gate = payload["gate"]
    checks = gate["checks"]
    require(gate["passed"] == all(checks.values()), "gate/check mismatch")
    if gate["passed"]:
        require(
            payload["status"] == "source-action-classes-qualified",
            "positive status changed",
        )
        require(
            gate["next_stage"]
            == "run-source-only-sequential-probe-to-drop-qualification",
            "positive next stage changed",
        )
    else:
        require(
            payload["status"] == "source-action-classes-not-qualified",
            "negative status changed",
        )
        require(
            gate["next_stage"] == "do-not-treat-take-index-as-action-class",
            "negative next stage changed",
        )

    return {
        "schema": "causal4d/pokeflex-probe-action-class-audit-verification",
        "schema_version": 1,
        "status": "verified",
        "content_sha256": stored,
        "gate_passed": gate["passed"],
        "loo_take_accuracy": payload["summary"]["loo_take_accuracy"],
        "between_to_within_ratio": payload["summary"][
            "between_to_within_ratio"
        ],
        "target_archive_open_count": boundary["target_archive_open_count"],
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
