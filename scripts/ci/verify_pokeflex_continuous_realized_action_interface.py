#!/usr/bin/env python3
"""Verify a continuous realized-action PokeFlex interface artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

SCHEMA = "causal4d/pokeflex-continuous-realized-action-interface"
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
    require(not (set(objects) & TARGET_OBJECTS), "target object appears in result")
    require(len(payload["records"]) == 72, "record panel is incomplete")
    representation = payload["representation"]
    require(
        representation["semantics"] == "complete realized tool trajectory",
        "action semantics changed",
    )
    require(representation["uses_nominal_take_identity"] is False, "take label used")
    require(representation["uses_contact_frame"] is False, "contact frame used")

    boundary = payload["information_boundary"]
    require(boundary["target_archive_open_count"] == 0, "target archive was opened")
    require(boundary["drop_archive_open_count"] == 0, "drop archive was opened")
    require(boundary["non_robot_member_open_count"] == 0, "non-robot payload opened")
    require(boundary["challenge_outcome_read"] is False, "challenge outcome was read")
    require(boundary["nominal_take_label_used_as_action"] is False, "take label leaked")
    require(boundary["contact_frame_inferred"] is False, "contact inference leaked")

    prior = payload["prior_results"]
    require(prior["nominal_take_classes"]["gate_passed"] is False, "nominal gate changed")
    require(
        prior["contact_localized_interface"]["gate_passed"] is False,
        "contact gate changed",
    )

    gate = payload["gate"]
    require(gate["passed"] == all(gate["checks"].values()), "gate/check mismatch")
    require(
        payload["status"]
        == (
            "continuous-realized-action-interface-qualified"
            if gate["passed"]
            else "continuous-realized-action-interface-not-qualified"
        ),
        "status and gate differ",
    )
    require(
        gate["next_stage"]
        == (
            "run-source-only-continuous-sequential-probe-to-drop-study"
            if gate["passed"]
            else "do-not-open-target-probe-or-drop-payloads"
        ),
        "next stage changed",
    )

    return {
        "schema": "causal4d/pokeflex-continuous-realized-action-interface-verification",
        "schema_version": 1,
        "status": "verified",
        "content_sha256": stored,
        "gate_passed": gate["passed"],
        "intrinsic_rank": payload["geometry"]["intrinsic_rank"],
        "minimum_object_median_pairwise_distance": payload["geometry"][
            "minimum_object_median_pairwise_distance"
        ],
        "p90_nearest_other_object_distance": payload["geometry"][
            "p90_nearest_other_object_distance"
        ],
        "target_archive_open_count": boundary["target_archive_open_count"],
        "drop_archive_open_count": boundary["drop_archive_open_count"],
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
