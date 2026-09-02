#!/usr/bin/env python3
"""Verify a compact source-only PokeFlex active-probing panel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "causal4d/pokeflex-source-active-probe-panel"
TARGET_OBJECTS = {
    "3dPrintedBunny",
    "3dPrintedCylinder",
    "Sponge",
    "MemoryFoam",
    "Beanbag",
    "Pillow",
}
EXPECTED_ACTION_INTERFACE_SHA = (
    "d94815f7bb832fc18350a6a42f624cf8bdced5e5914cf58977927bbbfd53e577"
)


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
    parser.add_argument("panel", type=Path)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def verify(payload: dict[str, Any]) -> dict[str, Any]:
    require(payload.get("schema") == SCHEMA, "unexpected panel schema")
    require(payload.get("schema_version") == 1, "unexpected panel version")
    require(
        payload.get("status") == "source-active-probe-panel-complete",
        "panel status is not complete",
    )
    stored = payload.get("content_sha256")
    require(isinstance(stored, str) and len(stored) == 64, "bad panel digest")
    canonical = dict(payload)
    canonical.pop("content_sha256", None)
    actual = hashlib.sha256(canonical_bytes(canonical)).hexdigest()
    require(actual == stored, "panel digest mismatch")

    prior = payload["continuous_action_interface"]
    require(prior["gate_passed"] is True, "continuous action gate changed")
    require(
        prior["content_sha256"] == EXPECTED_ACTION_INTERFACE_SHA,
        "continuous action evidence identity changed",
    )

    source = list(map(str, payload["source_objects"]))
    calibration = list(map(str, payload["calibration_objects"]))
    require(len(source) == 9 and len(set(source)) == 9, "source roster changed")
    require(
        len(calibration) == 3 and len(set(calibration)) == 3,
        "calibration roster changed",
    )
    require(not (set(source) & set(calibration)), "source/calibration overlap")
    require(not ((set(source) | set(calibration)) & TARGET_OBJECTS), "target object entered panel")
    require(set(payload["target_objects_excluded"]) == TARGET_OBJECTS, "target roster changed")

    objects = payload["object_records"]
    pokes = payload["poke_records"]
    drops = payload["drop_records"]
    require(len(objects) == 12, "object panel is incomplete")
    require(len(pokes) == 72, "poke panel is incomplete")
    require(len(drops) == 36, "drop panel is incomplete")
    expected_objects = set(source) | set(calibration)
    require({row["object_id"] for row in objects} == expected_objects, "object identities changed")
    require(
        {(row["object_id"], int(row["take_index_for_file_identity_only"])) for row in pokes}
        == {(object_id, take) for object_id in expected_objects for take in range(1, 7)},
        "poke identities changed",
    )
    require(
        {(row["object_id"], int(row["take_index"])) for row in drops}
        == {(object_id, take) for object_id in expected_objects for take in range(1, 4)},
        "drop identities changed",
    )

    names = payload["feature_names"]
    require(len(names["initial"]) == 7, "initial feature dimension changed")
    require(len(names["action"]) == 7, "action feature dimension changed")
    require(len(names["force_response"]) == 5, "force feature dimension changed")
    require(len(names["geometry_response"]) == 7, "geometry response dimension changed")
    require(len(names["response"]) == 12, "probe response dimension changed")
    require(len(names["drop_shape_query"]) == 4, "shape query dimension changed")
    require(len(names["drop_pose_query"]) == 3, "pose query dimension changed")

    arrays = []
    for row in objects:
        arrays.extend(
            [
                row["initial_features"],
                row["initial_feature_std"],
                row["drop_shape_query_mean"],
                row["drop_shape_query_std"],
                row["drop_pose_query_mean"],
                row["drop_pose_query_std"],
            ]
        )
    for row in pokes:
        arrays.extend(
            [
                row["action_features"],
                row["force_response_features"],
                row["geometry_response_features"],
                row["response_features"],
            ]
        )
    for row in drops:
        arrays.extend(
            [
                row["query_shape_features"],
                row["query_pose_features"],
                row["all_geometry_change_features"],
            ]
        )
    require(
        all(np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays),
        "panel contains non-finite features",
    )

    boundary = payload["information_boundary"]
    require(boundary["source_and_calibration_poking_archive_open_count"] == 72, "poke-open count changed")
    require(boundary["source_and_calibration_dropping_archive_open_count"] == 36, "drop-open count changed")
    require(boundary["target_archive_open_count"] == 0, "target archive was opened")
    require(boundary["image_member_open_count"] == 0, "image member was opened")
    require(boundary["point_cloud_member_open_count"] == 0, "point cloud was opened")
    require(boundary["camera_member_open_count"] == 0, "camera member was opened")
    require(boundary["target_outcome_read"] is False, "target outcome was read")
    require(all(payload["quality_checks"].values()), "panel quality check failed")

    return {
        "schema": "causal4d/pokeflex-source-active-probe-panel-verification",
        "schema_version": 1,
        "status": "verified",
        "content_sha256": stored,
        "object_count": len(objects),
        "poke_count": len(pokes),
        "drop_count": len(drops),
        "target_archive_open_count": boundary["target_archive_open_count"],
        "target_outcome_read": boundary["target_outcome_read"],
    }


def main() -> int:
    args = parse_args()
    payload = json.loads(args.panel.read_text(encoding="utf-8"))
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
