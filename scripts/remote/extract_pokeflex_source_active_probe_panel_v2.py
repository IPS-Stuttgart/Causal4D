#!/usr/bin/env python3
"""Extract the PokeFlex source panel using first/last valid mesh frames.

Revision 1 exposed that some public drop archives begin with an OBJ placeholder
that has no usable vertex payload. This technical revision keeps the cohort,
features, metrics, and information boundary unchanged while selecting the first
and last mesh members that independently parse to at least four finite vertices.
Every skipped endpoint placeholder is retained in the derived record.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "extract_pokeflex_source_active_probe_panel.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "pokeflex_source_active_probe_panel_v1",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load source active-probe panel module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def valid_endpoint_meshes(
    archive: ZipFile,
    members: list[tuple[int, str]],
) -> tuple[
    tuple[int, str, bytes, np.ndarray],
    tuple[int, str, bytes, np.ndarray],
    dict[str, Any],
]:
    parsed: dict[str, tuple[bytes, np.ndarray]] = {}
    invalid: list[dict[str, Any]] = []

    def parse(frame: int, member: str) -> tuple[bytes, np.ndarray] | None:
        if member in parsed:
            return parsed[member]
        payload = archive.read(member)
        try:
            vertices = BASE.parse_obj_vertices(payload)
        except (UnicodeDecodeError, ValueError) as error:
            invalid.append(
                {
                    "frame": frame,
                    "member": member,
                    "bytes": len(payload),
                    "reason": str(error),
                }
            )
            return None
        parsed[member] = (payload, vertices)
        return payload, vertices

    first = None
    for frame, member in members:
        value = parse(frame, member)
        if value is not None:
            first = (frame, member, *value)
            break
    if first is None:
        raise ValueError("archive contains no valid mesh frame")

    last = None
    for frame, member in reversed(members):
        value = parse(frame, member)
        if value is not None and frame > first[0]:
            last = (frame, member, *value)
            break
    if last is None:
        raise ValueError("archive contains fewer than two valid mesh frames")

    return first, last, {
        "invalid_endpoint_members": invalid,
        "invalid_endpoint_member_count": len(invalid),
        "first_valid_frame": first[0],
        "last_valid_frame": last[0],
    }


def read_poke_episode(
    root: Path,
    object_id: str,
    take_index: int,
    action_module: Any,
    maximum_points: int,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_path = BASE.locate_archive(root, "poking", object_id, take_index)
    take_stem = f"{object_id}_T{take_index}"
    with ZipFile(archive_path) as archive:
        robot_member = action_module.robot_member_name(archive, take_stem)
        members = BASE.mesh_members(archive)
        robot_payload = archive.read(robot_member)
        first, last, endpoint = valid_endpoint_meshes(archive, members)
    first_frame, first_member, first_payload, first_vertices = first
    last_frame, last_member, last_payload, last_vertices = last
    raw_records = json.loads(robot_payload.decode("utf-8"))
    if not isinstance(raw_records, list):
        raise ValueError("robot_data.json is not a list")
    action, force_response, action_metadata = action_module.action_response_features(
        raw_records
    )
    initial, initial_metadata = BASE.initial_features(first_vertices)
    change, change_metadata = BASE.geometry_change(
        first_vertices,
        last_vertices,
        maximum_points,
    )
    response = np.concatenate([force_response, change])
    record = {
        "object_id": object_id,
        "take_index_for_file_identity_only": take_index,
        "take_id": f"poking:{take_stem}",
        "archive_relative_path": str(archive_path.relative_to(root)),
        "robot_member": robot_member,
        "robot_sha256": hashlib.sha256(robot_payload).hexdigest(),
        "first_mesh_member": first_member,
        "first_mesh_frame": first_frame,
        "first_mesh_sha256": hashlib.sha256(first_payload).hexdigest(),
        "last_mesh_member": last_member,
        "last_mesh_frame": last_frame,
        "last_mesh_sha256": hashlib.sha256(last_payload).hexdigest(),
        "action_features": action.tolist(),
        "force_response_features": force_response.tolist(),
        "geometry_response_features": change.tolist(),
        "response_features": response.tolist(),
        "action_metadata": action_metadata,
        "initial_metadata": initial_metadata,
        "change_metadata": change_metadata,
        "endpoint_selection": endpoint,
    }
    return record, initial


def read_drop_episode(
    root: Path,
    object_id: str,
    take_index: int,
    maximum_points: int,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_path = BASE.locate_archive(root, "dropping", object_id, take_index)
    take_stem = f"{object_id}_T{take_index}"
    with ZipFile(archive_path) as archive:
        members = BASE.mesh_members(archive)
        first, last, endpoint = valid_endpoint_meshes(archive, members)
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
    return record, initial


def main() -> int:
    BASE.read_poke_episode = read_poke_episode
    BASE.read_drop_episode = read_drop_episode
    args = BASE.parse_args()
    request = BASE.load_request(args.request)
    action_module = BASE.load_action_module(args.action_module)
    payload = BASE.run(args.root, request, action_module)
    payload["endpoint_contract"] = {
        "revision": 2,
        "first_frame": "first mesh member with at least four finite vertices",
        "last_frame": "last later mesh member with at least four finite vertices",
        "invalid_placeholders_recorded": True,
    }
    payload["content_sha256"] = BASE.content_sha256(payload)
    BASE.write_outputs(args.output_dir, payload, args.cache_path)
    skipped = sum(
        row["endpoint_selection"]["invalid_endpoint_member_count"]
        for row in [*payload["poke_records"], *payload["drop_records"]]
    )
    print(
        json.dumps(
            {
                "content_sha256": payload["content_sha256"],
                "objects": len(payload["object_records"]),
                "pokes": len(payload["poke_records"]),
                "drops": len(payload["drop_records"]),
                "invalid_endpoint_placeholders_skipped": skipped,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
