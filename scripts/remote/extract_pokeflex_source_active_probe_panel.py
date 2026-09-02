#!/usr/bin/env python3
"""Extract a compact source-only PokeFlex active-probing panel."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

MESH_PATTERN = re.compile(r"/meshes/mesh-f(\d{5})\.obj$")
SCHEMA = "causal4d/pokeflex-source-active-probe-panel"
SCHEMA_VERSION = 1
TARGET_OBJECTS = frozenset(
    {
        "3dPrintedBunny",
        "3dPrintedCylinder",
        "Sponge",
        "MemoryFoam",
        "Beanbag",
        "Pillow",
    }
)
INITIAL_FEATURE_NAMES = (
    "log_diagonal_mm",
    "extent_x_over_diagonal",
    "extent_y_over_diagonal",
    "extent_z_over_diagonal",
    "cov_eigenvalue_1_over_diagonal_sq",
    "cov_eigenvalue_2_over_diagonal_sq",
    "cov_eigenvalue_3_over_diagonal_sq",
)
GEOMETRY_CHANGE_NAMES = (
    "centroid_dx_over_diagonal",
    "centroid_dy_over_diagonal",
    "centroid_dz_over_diagonal",
    "extent_dx_over_diagonal",
    "extent_dy_over_diagonal",
    "extent_dz_over_diagonal",
    "centered_symmetric_chamfer_over_diagonal",
)
DROP_SHAPE_QUERY_NAMES = (
    "extent_dx_over_diagonal",
    "extent_dy_over_diagonal",
    "extent_dz_over_diagonal",
    "centered_symmetric_chamfer_over_diagonal",
)
DROP_POSE_QUERY_NAMES = (
    "centroid_dx_over_diagonal",
    "centroid_dy_over_diagonal",
    "centroid_dz_over_diagonal",
)


def load_action_module(path: Path):
    spec = importlib.util.spec_from_file_location("pokeflex_action_panel_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load action module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("content_sha256", None)
    return hashlib.sha256(canonical_bytes(canonical)).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--action-module", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-path", type=Path)
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema")
        == "causal4d/pokeflex-source-active-probe-panel-request",
        "unexpected request schema",
    )
    require(payload.get("schema_version") == 1, "unexpected request version")
    objects = payload.get("source_and_calibration_objects")
    require(isinstance(objects, list) and len(objects) == 12, "expected 12 objects")
    require(len(set(objects)) == 12, "object roster repeats")
    require(not (set(objects) & TARGET_OBJECTS), "target object entered source panel")
    require(
        payload.get("poking_take_indices") == [1, 2, 3, 4, 5, 6],
        "poke roster changed",
    )
    require(
        payload.get("dropping_take_indices") == [1, 2, 3],
        "drop roster changed",
    )
    prior = payload.get("continuous_action_interface")
    require(
        isinstance(prior, dict) and prior.get("gate_passed") is True,
        "continuous action interface is not authorized",
    )
    require(
        prior.get("content_sha256")
        == "d94815f7bb832fc18350a6a42f624cf8bdced5e5914cf58977927bbbfd53e577",
        "continuous action interface identity changed",
    )
    return payload


def locate_archive(
    root: Path,
    action: str,
    object_id: str,
    take_index: int,
) -> Path:
    expected = root / action / object_id / f"{object_id}_T{take_index}.zip"
    if expected.is_file():
        return expected
    matches = sorted((root / action).rglob(f"{object_id}_T{take_index}.zip"))
    require(
        len(matches) == 1,
        (
            f"expected one {action} archive for {object_id} T{take_index}; "
            f"found {len(matches)}"
        ),
    )
    return matches[0]


def mesh_members(archive: ZipFile) -> list[tuple[int, str]]:
    values = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        match = MESH_PATTERN.search(info.filename)
        if match is not None:
            values.append((int(match.group(1)), info.filename))
    values.sort()
    require(len(values) >= 2, "archive contains fewer than two mesh frames")
    return values


def parse_obj_vertices(payload: bytes) -> np.ndarray:
    vertices = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        if line.startswith("v "):
            fields = line.split()
            require(len(fields) >= 4, "invalid OBJ vertex")
            vertices.append(tuple(float(value) for value in fields[1:4]))
    array = np.asarray(vertices, dtype=np.float64)
    require(
        array.ndim == 2 and array.shape[1] == 3 and len(array) >= 4,
        "OBJ contains too few vertices",
    )
    require(np.all(np.isfinite(array)), "OBJ contains non-finite vertices")
    return array


def deterministic_sample(vertices: np.ndarray, maximum: int) -> np.ndarray:
    if len(vertices) <= maximum:
        return vertices
    indices = np.linspace(0, len(vertices) - 1, maximum, dtype=np.int64)
    return vertices[indices]


def symmetric_chamfer(first: np.ndarray, last: np.ndarray, maximum: int) -> float:
    first_sample = deterministic_sample(first, maximum)
    last_sample = deterministic_sample(last, maximum)
    first_centered = first_sample - np.mean(first_sample, axis=0)
    last_centered = last_sample - np.mean(last_sample, axis=0)
    first_tree = cKDTree(first_centered)
    last_tree = cKDTree(last_centered)
    first_to_last, _ = last_tree.query(first_centered, k=1, workers=1)
    last_to_first, _ = first_tree.query(last_centered, k=1, workers=1)
    return float(0.5 * (np.mean(first_to_last) + np.mean(last_to_first)))


def initial_features(vertices: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    extent = maximum - minimum
    diagonal = float(np.linalg.norm(extent))
    require(diagonal > 1e-9, "mesh diagonal is degenerate")
    centered = vertices - np.mean(vertices, axis=0)
    covariance = centered.T @ centered / max(len(vertices), 1)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    features = np.asarray(
        [
            np.log1p(diagonal),
            *(extent / diagonal).tolist(),
            *(eigenvalues / (diagonal * diagonal)).tolist(),
        ],
        dtype=np.float64,
    )
    require(np.all(np.isfinite(features)), "initial features are non-finite")
    return features, {
        "vertex_count": int(len(vertices)),
        "diagonal_mm": diagonal,
        "centroid_mm": np.mean(vertices, axis=0).tolist(),
        "extent_mm": extent.tolist(),
    }


def geometry_change(
    first: np.ndarray,
    last: np.ndarray,
    maximum: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    first_minimum = np.min(first, axis=0)
    first_maximum = np.max(first, axis=0)
    first_extent = first_maximum - first_minimum
    diagonal = float(np.linalg.norm(first_extent))
    require(diagonal > 1e-9, "first mesh diagonal is degenerate")
    last_extent = np.max(last, axis=0) - np.min(last, axis=0)
    centroid_change = (np.mean(last, axis=0) - np.mean(first, axis=0)) / diagonal
    extent_change = (last_extent - first_extent) / diagonal
    chamfer = symmetric_chamfer(first, last, maximum) / diagonal
    features = np.asarray(
        [*centroid_change.tolist(), *extent_change.tolist(), chamfer],
        dtype=np.float64,
    )
    require(np.all(np.isfinite(features)), "geometry change is non-finite")
    return features, {
        "first_vertex_count": int(len(first)),
        "last_vertex_count": int(len(last)),
        "first_diagonal_mm": diagonal,
    }


def read_poke_episode(
    root: Path,
    object_id: str,
    take_index: int,
    action_module: Any,
    maximum_points: int,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_path = locate_archive(root, "poking", object_id, take_index)
    take_stem = f"{object_id}_T{take_index}"
    with ZipFile(archive_path) as archive:
        robot_member = action_module.robot_member_name(archive, take_stem)
        members = mesh_members(archive)
        first_member = members[0][1]
        last_member = members[-1][1]
        robot_payload = archive.read(robot_member)
        first_payload = archive.read(first_member)
        last_payload = archive.read(last_member)
    raw_records = json.loads(robot_payload.decode("utf-8"))
    require(isinstance(raw_records, list), "robot_data.json is not a list")
    action, force_response, action_metadata = action_module.action_response_features(
        raw_records
    )
    first = parse_obj_vertices(first_payload)
    last = parse_obj_vertices(last_payload)
    initial, initial_metadata = initial_features(first)
    change, change_metadata = geometry_change(first, last, maximum_points)
    response = np.concatenate([force_response, change])
    record = {
        "object_id": object_id,
        "take_index_for_file_identity_only": take_index,
        "take_id": f"poking:{take_stem}",
        "archive_relative_path": str(archive_path.relative_to(root)),
        "robot_member": robot_member,
        "robot_sha256": hashlib.sha256(robot_payload).hexdigest(),
        "first_mesh_member": first_member,
        "first_mesh_sha256": hashlib.sha256(first_payload).hexdigest(),
        "last_mesh_member": last_member,
        "last_mesh_sha256": hashlib.sha256(last_payload).hexdigest(),
        "action_features": action.tolist(),
        "force_response_features": force_response.tolist(),
        "geometry_response_features": change.tolist(),
        "response_features": response.tolist(),
        "action_metadata": action_metadata,
        "initial_metadata": initial_metadata,
        "change_metadata": change_metadata,
    }
    return record, initial


def read_drop_episode(
    root: Path,
    object_id: str,
    take_index: int,
    maximum_points: int,
) -> tuple[dict[str, Any], np.ndarray]:
    archive_path = locate_archive(root, "dropping", object_id, take_index)
    take_stem = f"{object_id}_T{take_index}"
    with ZipFile(archive_path) as archive:
        members = mesh_members(archive)
        first_member = members[0][1]
        last_member = members[-1][1]
        first_payload = archive.read(first_member)
        last_payload = archive.read(last_member)
    first = parse_obj_vertices(first_payload)
    last = parse_obj_vertices(last_payload)
    initial, initial_metadata = initial_features(first)
    change, change_metadata = geometry_change(first, last, maximum_points)
    record = {
        "object_id": object_id,
        "take_index": take_index,
        "take_id": f"dropping:{take_stem}",
        "archive_relative_path": str(archive_path.relative_to(root)),
        "first_mesh_member": first_member,
        "first_mesh_sha256": hashlib.sha256(first_payload).hexdigest(),
        "last_mesh_member": last_member,
        "last_mesh_sha256": hashlib.sha256(last_payload).hexdigest(),
        "query_shape_features": change[3:].tolist(),
        "query_pose_features": change[:3].tolist(),
        "all_geometry_change_features": change.tolist(),
        "initial_metadata": initial_metadata,
        "change_metadata": change_metadata,
    }
    return record, initial


def run(root: Path, request: dict[str, Any], action_module: Any) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), "PokeFlex root does not exist")
    objects = [str(value) for value in request["source_and_calibration_objects"]]
    maximum_points = int(request["geometry"]["maximum_chamfer_points"])
    poke_records = []
    drop_records = []
    object_records = []
    total_payload_members = 0
    for object_id in objects:
        require(object_id not in TARGET_OBJECTS, "target object entered source panel")
        initial_rows = []
        object_pokes = []
        object_drops = []
        for take_index in request["poking_take_indices"]:
            record, initial = read_poke_episode(
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
            record, initial = read_drop_episode(
                root,
                object_id,
                int(take_index),
                maximum_points,
            )
            drop_records.append(record)
            object_drops.append(record)
            initial_rows.append(initial)
            total_payload_members += 2
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
                "initial_feature_std": np.std(
                    initial_matrix, axis=0, ddof=0
                ).tolist(),
                "drop_shape_query_mean": np.mean(shape_matrix, axis=0).tolist(),
                "drop_shape_query_std": np.std(
                    shape_matrix, axis=0, ddof=0
                ).tolist(),
                "drop_pose_query_mean": np.mean(pose_matrix, axis=0).tolist(),
                "drop_pose_query_std": np.std(
                    pose_matrix, axis=0, ddof=0
                ).tolist(),
                "poking_take_ids": [record["take_id"] for record in object_pokes],
                "dropping_take_ids": [record["take_id"] for record in object_drops],
            }
        )

    checks = {
        "complete_object_panel": len(object_records) == 12,
        "complete_poking_panel": len(poke_records) == 72,
        "complete_dropping_panel": len(drop_records) == 36,
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
            {record["object_id"] for record in object_records} & TARGET_OBJECTS
        ),
    }
    require(all(checks.values()), f"source panel failed: {checks}")
    payload = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "status": "source-active-probe-panel-complete",
        "continuous_action_interface": request["continuous_action_interface"],
        "source_objects": request["source_objects"],
        "calibration_objects": request["calibration_objects"],
        "target_objects_excluded": request["target_objects_excluded"],
        "feature_names": {
            "initial": list(INITIAL_FEATURE_NAMES),
            "action": list(action_module.ACTION_FEATURE_NAMES),
            "force_response": list(action_module.RESPONSE_FEATURE_NAMES),
            "geometry_response": list(GEOMETRY_CHANGE_NAMES),
            "response": [
                *action_module.RESPONSE_FEATURE_NAMES,
                *GEOMETRY_CHANGE_NAMES,
            ],
            "drop_shape_query": list(DROP_SHAPE_QUERY_NAMES),
            "drop_pose_query": list(DROP_POSE_QUERY_NAMES),
        },
        "geometry_contract": request["geometry"],
        "object_records": object_records,
        "poke_records": poke_records,
        "drop_records": drop_records,
        "quality_checks": checks,
        "information_boundary": {
            "source_and_calibration_poking_archive_open_count": 72,
            "source_and_calibration_dropping_archive_open_count": 36,
            "target_archive_open_count": 0,
            "opened_payload_member_count": total_payload_members,
            "image_member_open_count": 0,
            "point_cloud_member_open_count": 0,
            "camera_member_open_count": 0,
            "target_outcome_read": False,
        },
        "claim_boundary": [
            "This is a source/calibration feature panel, not a target result.",
            (
                "Poking take indices are file identities only; actions are "
                "continuous realized trajectories."
            ),
            (
                "Only first/final geometry is used; no online or same-state "
                "counterfactual claim is made."
            ),
        ],
        "content_sha256": "",
    }
    payload["content_sha256"] = content_sha256(payload)
    return payload


def write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    cache_path: Path | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    encoded = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (output_dir / "panel.json").write_bytes(encoded)
    if cache_path is not None:
        write_atomic(cache_path, encoded)
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
        "# PokeFlex source active-probe panel",
        "",
        f"- Objects: **{len(payload['object_records'])}**",
        f"- Poking responses: **{len(payload['poke_records'])}**",
        f"- Dropping outcomes: **{len(payload['drop_records'])}**",
        (
            "- Continuous action dimensions: "
            f"**{len(payload['feature_names']['action'])}**"
        ),
        (
            "- Probe-response dimensions: "
            f"**{len(payload['feature_names']['response'])}**"
        ),
        (
            "- Drop-query dimensions: "
            f"**{len(payload['feature_names']['drop_shape_query']) + len(payload['feature_names']['drop_pose_query'])}**"
        ),
        f"- Content SHA-256: `{payload['content_sha256']}`",
        "- Target archive opens: **0**",
        "",
        "This panel authorizes source-only policy development. It does not",
        "authorize opening the six registered target objects.",
        "",
    ]
    (output_dir / "panel.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    request = load_request(args.request)
    action_module = load_action_module(args.action_module)
    payload = run(args.root, request, action_module)
    write_outputs(args.output_dir, payload, args.cache_path)
    print(
        json.dumps(
            {
                "content_sha256": payload["content_sha256"],
                "objects": len(payload["object_records"]),
                "pokes": len(payload["poke_records"]),
                "drops": len(payload["drop_records"]),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
