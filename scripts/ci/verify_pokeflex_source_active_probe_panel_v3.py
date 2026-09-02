#!/usr/bin/env python3
"""Verify the PokeFlex source panel with explicit invalid-drop accounting."""

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
EXPECTED_INVALID = {
    "dropping:PlushDice_T2",
    "dropping:PlushDice_T3",
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
        payload.get("status")
        == "source-active-probe-panel-complete-with-recorded-missingness",
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

    source = set(map(str, payload["source_objects"]))
    calibration = set(map(str, payload["calibration_objects"]))
    require(len(source) == 9, "source roster changed")
    require(len(calibration) == 3, "calibration roster changed")
    require(source.isdisjoint(calibration), "source/calibration overlap")
    require((source | calibration).isdisjoint(TARGET_OBJECTS), "target object entered panel")
    require(set(payload["target_objects_excluded"]) == TARGET_OBJECTS, "target roster changed")

    objects = payload["object_records"]
    pokes = payload["poke_records"]
    drops = payload["drop_records"]
    invalid = payload["invalid_drop_records"]
    require(len(objects) == 12, "object panel is incomplete")
    require(len(pokes) == 72, "poke panel is incomplete")
    require(len(drops) == 34, "valid drop panel changed")
    require(len(invalid) == 2, "invalid drop count changed")
    require({row["take_id"] for row in invalid} == EXPECTED_INVALID, "invalid identities changed")
    require(
        all(row["payload_used_as_query"] is False for row in invalid),
        "invalid drop was used as a query",
    )
    require(
        {row["object_id"] for row in objects} == source | calibration,
        "object identities changed",
    )
    require(
        {(row["object_id"], int(row["take_index_for_file_identity_only"])) for row in pokes}
        == {(object_id, take) for object_id in source | calibration for take in range(1, 7)},
        "poke identities changed",
    )
    observed_valid = {(row["object_id"], int(row["take_index"])) for row in drops}
    expected_all = {(object_id, take) for object_id in source | calibration for take in range(1, 4)}
    expected_invalid_pairs = {("PlushDice", 2), ("PlushDice", 3)}
    require(observed_valid == expected_all - expected_invalid_pairs, "valid drop identities changed")

    missingness = payload["missingness_contract"]
    require(missingness["expected_invalid_drop_ids"] == sorted(EXPECTED_INVALID), "missingness contract changed")
    require(missingness["invalid_drop_queries_imputed"] is False, "invalid drops were imputed")
    require(missingness["objects_with_missing_drops_retained"] is True, "object retention changed")
    require(
        missingness["object_query_is_mean_of_available_valid_drops"] is True,
        "query aggregation changed",
    )
    per_object = {row["object_id"]: row for row in objects}
    require(per_object["PlushDice"]["valid_drop_count"] == 1, "PlushDice valid count changed")
    require(per_object["PlushDice"]["invalid_drop_count"] == 2, "PlushDice invalid count changed")
    require(
        all(row["valid_drop_count"] == 3 for name, row in per_object.items() if name != "PlushDice"),
        "a complete object lost a drop",
    )

    names = payload["feature_names"]
    require(len(names["initial"]) == 7, "initial feature dimension changed")
    require(len(names["action"]) == 7, "action feature dimension changed")
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
        arrays.extend([row["action_features"], row["response_features"]])
    for row in drops:
        arrays.extend([row["query_shape_features"], row["query_pose_features"]])
    require(
        all(np.all(np.isfinite(np.asarray(value, dtype=float))) for value in arrays),
        "panel contains non-finite features",
    )

    boundary = payload["information_boundary"]
    require(boundary["source_and_calibration_poking_archive_open_count"] == 72, "poke-open count changed")
    require(boundary["source_and_calibration_dropping_archive_open_count"] == 36, "drop-open count changed")
    require(boundary["valid_drop_outcome_count"] == 34, "valid drop count changed")
    require(boundary["invalid_drop_outcome_count"] == 2, "invalid drop count changed")
    require(boundary["target_archive_open_count"] == 0, "target archive was opened")
    require(boundary["image_member_open_count"] == 0, "image member was opened")
    require(boundary["point_cloud_member_open_count"] == 0, "point cloud was opened")
    require(boundary["camera_member_open_count"] == 0, "camera member was opened")
    require(boundary["target_outcome_read"] is False, "target outcome was read")
    require(all(payload["quality_checks"].values()), "panel quality check failed")

    return {
        "schema": "causal4d/pokeflex-source-active-probe-panel-v3-verification",
        "schema_version": 1,
        "status": "verified",
        "content_sha256": stored,
        "object_count": len(objects),
        "poke_count": len(pokes),
        "valid_drop_count": len(drops),
        "invalid_drop_ids": sorted(EXPECTED_INVALID),
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
