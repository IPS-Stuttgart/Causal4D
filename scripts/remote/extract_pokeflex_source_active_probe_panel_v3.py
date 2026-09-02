#!/usr/bin/env python3
"""Extract a source PokeFlex panel with explicit invalid-drop accounting.

The source-only carrier audit found exactly two public recordings whose every
OBJ frame is an empty reconstruction: ``PlushDice_T2`` and ``PlushDice_T3``.
This revision retains the physical object, uses its one valid drop recording,
and records the two invalid recordings rather than fabricating geometry or
changing the frozen target roster.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np


BASE_PATH = (
    Path(__file__).resolve().parent
    / "extract_pokeflex_source_active_probe_panel.py"
)
V2_PATH = (
    Path(__file__).resolve().parent
    / "extract_pokeflex_source_active_probe_panel_v2.py"
)
EXPECTED_INVALID_DROP_IDS = {
    "dropping:PlushDice_T2",
    "dropping:PlushDice_T3",
}


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_module(BASE_PATH, "pokeflex_source_panel_base_v3")
V2 = load_module(V2_PATH, "pokeflex_source_panel_endpoint_v2")


def try_read_drop_episode(
    root: Path,
    object_id: str,
    take_index: int,
    maximum_points: int,
) -> tuple[dict[str, Any] | None, np.ndarray | None, dict[str, Any] | None]:
    archive_path = BASE.locate_archive(root, "dropping", object_id, take_index)
    take_stem = f"{object_id}_T{take_index}"
    with ZipFile(archive_path) as archive:
        members = BASE.mesh_members(archive)
        try:
            first, last, endpoint = V2.valid_endpoint_meshes(archive, members)
        except ValueError as error:
            identity = f"dropping:{take_stem}"
            return None, None, {
                "object_id": object_id,
                "take_index": take_index,
                "take_id": identity,
                "archive_relative_path": str(archive_path.relative_to(root)),
                "reason": str(error),
                "mesh_member_count": len(members),
                "mesh_member_uncompressed_bytes_min": min(
                    archive.getinfo(member).file_size for _, member in members
                ),
                "mesh_member_uncompressed_bytes_max": max(
                    archive.getinfo(member).file_size for _, member in members
                ),
                "payload_used_as_query": False,
            }
    first_frame, first_member, first_payload, first_vertices = first
    last_frame, last_member, last_payload, last_vertices = last
    initial, initial_metadata = BASE.initial_features(first_vertices)
    change, change_metadata = BASE.geometry_change(
        first_vertices,
        last_vertices,
        maximum_points,
    )
    record = {
        "object_id": object_id,
        "take_index": take_index,
        "take_id": f"dropping:{take_stem}",
        "archive_relative_path": str(archive_path.relative_to(root)),
        "first_mesh_member": first_member,
        "first_mesh_frame": first_frame,
        "first_mesh_sha256": hashlib.sha256(first_payload).hexdigest(),
        "last_mesh_member": last_member,
        "last_mesh_frame": last_frame,
        "last_mesh_sha256": hashlib.sha256(last_payload).hexdigest(),
        "query_shape_features": change[3:].tolist(),
        "query_pose_features": change[:3].tolist(),
        "all_geometry_change_features": change.tolist(),
        "initial_metadata": initial_metadata,
        "change_metadata": change_metadata,
        "endpoint_selection": endpoint,
    }
    return record, initial, None


def run(root: Path, request: dict[str, Any], action_module: Any) -> dict[str, Any]:
    root = root.resolve()
    BASE.require(root.is_dir(), "PokeFlex root does not exist")
    objects = [str(value) for value in request["source_and_calibration_objects"]]
    maximum_points = int(request["geometry"]["maximum_chamfer_points"])
    poke_records = []
    drop_records = []
    invalid_drop_records = []
    object_records = []
    total_payload_members = 0
    for object_id in objects:
        BASE.require(
            object_id not in BASE.TARGET_OBJECTS,
            "target object entered source panel",
        )
        initial_rows = []
        object_pokes = []
        object_drops = []
        object_invalid_drops = []
        for take_index in request["poking_take_indices"]:
            record, initial = V2.read_poke_episode(
                root,
                object_id,
                int(take_index),
                action_module,
                maximum_points,
            )
            poke_records.append(record)
            object_pokes.append(record)
            initial_rows.append(initial)
            total_payload_members += 3
        for take_index in request["dropping_take_indices"]:
            record, initial, invalid = try_read_drop_episode(
                root,
                object_id,
                int(take_index),
                maximum_points,
            )
            if invalid is not None:
                invalid_drop_records.append(invalid)
                object_invalid_drops.append(invalid)
                continue
            if record is None or initial is None:
                raise RuntimeError("drop reader returned an inconsistent state")
            drop_records.append(record)
            object_drops.append(record)
            initial_rows.append(initial)
            total_payload_members += 2
        BASE.require(
            len(object_drops) >= int(request["quality_gate"]["minimum_valid_drops_per_object"]),
            f"{object_id} has too few valid drops",
        )
        initial_matrix = np.asarray(initial_rows, dtype=np.float64)
        shape_matrix = np.asarray(
            [record["query_shape_features"] for record in object_drops],
            dtype=np.float64,
        )
        pose_matrix = np.asarray(
            [record["query_pose_features"] for record in object_drops],
            dtype=np.float64,
        )
        object_records.append(
            {
                "object_id": object_id,
                "role": request["object_roles"][object_id],
                "initial_features": np.mean(initial_matrix, axis=0).tolist(),
                "initial_feature_std": np.std(initial_matrix, axis=0, ddof=0).tolist(),
                "drop_shape_query_mean": np.mean(shape_matrix, axis=0).tolist(),
                "drop_shape_query_std": np.std(shape_matrix, axis=0, ddof=0).tolist(),
                "drop_pose_query_mean": np.mean(pose_matrix, axis=0).tolist(),
                "drop_pose_query_std": np.std(pose_matrix, axis=0, ddof=0).tolist(),
                "valid_drop_count": len(object_drops),
                "invalid_drop_count": len(object_invalid_drops),
                "poking_take_ids": [record["take_id"] for record in object_pokes],
                "valid_dropping_take_ids": [record["take_id"] for record in object_drops],
                "invalid_dropping_take_ids": [
                    record["take_id"] for record in object_invalid_drops
                ],
            }
        )

    invalid_ids = {record["take_id"] for record in invalid_drop_records}
    checks = {
        "complete_object_panel": len(object_records) == 12,
        "complete_poking_panel": len(poke_records) == 72,
        "minimum_valid_drop_panel": len(drop_records)
        >= int(request["quality_gate"]["minimum_total_valid_drop_records"]),
        "expected_invalid_drop_set": invalid_ids
        == set(request["quality_gate"]["expected_invalid_drop_ids"]),
        "minimum_one_valid_drop_per_object": all(
            int(record["valid_drop_count"])
            >= int(request["quality_gate"]["minimum_valid_drops_per_object"])
            for record in object_records
        ),
        "all_features_finite": all(
            np.all(np.isfinite(np.asarray(value, dtype=np.float64)))
            for record in object_records
            for value in (
                record["initial_features"],
                record["drop_shape_query_mean"],
                record["drop_pose_query_mean"],
            )
        ),
        "target_objects_absent": not (
            {record["object_id"] for record in object_records}
            & BASE.TARGET_OBJECTS
        ),
    }
    BASE.require(all(checks.values()), f"source panel failed: {checks}")
    payload = {
        "schema": BASE.SCHEMA,
        "schema_version": BASE.SCHEMA_VERSION,
        "request_id": request["request_id"],
        "status": "source-active-probe-panel-complete-with-recorded-missingness",
        "continuous_action_interface": request["continuous_action_interface"],
        "source_objects": request["source_objects"],
        "calibration_objects": request["calibration_objects"],
        "target_objects_excluded": request["target_objects_excluded"],
        "feature_names": {
            "initial": list(BASE.INITIAL_FEATURE_NAMES),
            "action": list(action_module.ACTION_FEATURE_NAMES),
            "force_response": list(action_module.RESPONSE_FEATURE_NAMES),
            "geometry_response": list(BASE.GEOMETRY_CHANGE_NAMES),
            "response": [
                *action_module.RESPONSE_FEATURE_NAMES,
                *BASE.GEOMETRY_CHANGE_NAMES,
            ],
            "drop_shape_query": list(BASE.DROP_SHAPE_QUERY_NAMES),
            "drop_pose_query": list(BASE.DROP_POSE_QUERY_NAMES),
        },
        "geometry_contract": request["geometry"],
        "missingness_contract": {
            "minimum_valid_drops_per_object": request["quality_gate"][
                "minimum_valid_drops_per_object"
            ],
            "expected_invalid_drop_ids": sorted(EXPECTED_INVALID_DROP_IDS),
            "invalid_drop_queries_imputed": False,
            "objects_with_missing_drops_retained": True,
            "object_query_is_mean_of_available_valid_drops": True,
        },
        "object_records": object_records,
        "poke_records": poke_records,
        "drop_records": drop_records,
        "invalid_drop_records": invalid_drop_records,
        "quality_checks": checks,
        "information_boundary": {
            "source_and_calibration_poking_archive_open_count": 72,
            "source_and_calibration_dropping_archive_open_count": 36,
            "valid_drop_outcome_count": len(drop_records),
            "invalid_drop_outcome_count": len(invalid_drop_records),
            "target_archive_open_count": 0,
            "opened_payload_member_count": total_payload_members,
            "image_member_open_count": 0,
            "point_cloud_member_open_count": 0,
            "camera_member_open_count": 0,
            "target_outcome_read": False,
        },
        "claim_boundary": [
            "This is a source/calibration feature panel, not a target result.",
            "Two known empty PlushDice drop reconstructions are recorded, not imputed.",
            "Poking take indices are file identities only; actions are continuous trajectories.",
        ],
        "content_sha256": "",
    }
    payload["content_sha256"] = BASE.content_sha256(payload)
    return payload


def write_outputs(output_dir: Path, payload: dict[str, Any], cache_path: Path | None) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (output_dir / "panel.json").write_bytes(encoded)
    if cache_path is not None:
        BASE.write_atomic(cache_path, encoded)
        (output_dir / "cache.json").write_text(
            json.dumps(
                {
                    "cache_path": str(cache_path),
                    "content_sha256": payload["content_sha256"],
                    "bytes": len(encoded),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    lines = [
        "# PokeFlex source active-probe panel v3",
        "",
        f"- Objects: **{len(payload['object_records'])}**",
        f"- Poking responses: **{len(payload['poke_records'])}**",
        f"- Valid dropping outcomes: **{len(payload['drop_records'])}**",
        f"- Recorded invalid drops: **{len(payload['invalid_drop_records'])}**",
        (
            "- Invalid identities: "
            f"`{', '.join(row['take_id'] for row in payload['invalid_drop_records'])}`"
        ),
        f"- Content SHA-256: `{payload['content_sha256']}`",
        "- Target archive opens: **0**",
        "",
        "No missing drop geometry was fabricated or imputed.",
        "",
    ]
    (output_dir / "panel.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = BASE.parse_args()
    request = BASE.load_request(args.request)
    action_module = BASE.load_action_module(args.action_module)
    payload = run(args.root, request, action_module)
    write_outputs(args.output_dir, payload, args.cache_path)
    print(
        json.dumps(
            {
                "content_sha256": payload["content_sha256"],
                "objects": len(payload["object_records"]),
                "pokes": len(payload["poke_records"]),
                "valid_drops": len(payload["drop_records"]),
                "invalid_drops": [
                    row["take_id"] for row in payload["invalid_drop_records"]
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
