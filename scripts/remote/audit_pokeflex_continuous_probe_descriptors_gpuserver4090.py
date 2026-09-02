#!/usr/bin/env python3
"""Audit continuous geometry-normalized PokeFlex probe descriptors."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from scipy.spatial import cKDTree

SCHEMA = "causal4d/pokeflex-continuous-probe-descriptor-audit"
SCHEMA_VERSION = 1
MESH_PATTERN = re.compile(r"/meshes/mesh-f(\d{5})\.obj$")
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
DESCRIPTOR_NAMES = (
    "contact_x_normalized",
    "contact_y_normalized",
    "contact_z_normalized",
    "approach_x",
    "approach_y",
    "approach_z",
    "local_stroke_x_normalized",
    "local_stroke_y_normalized",
    "local_stroke_z_normalized",
    "log_local_path_over_diagonal",
    "log_duration_frames",
)


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
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def load_request(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    require(
        payload.get("schema")
        == "causal4d/pokeflex-continuous-probe-descriptor-audit-request",
        "unexpected request schema",
    )
    require(payload.get("schema_version") == 1, "unexpected request version")
    objects = payload.get("source_and_calibration_objects")
    require(isinstance(objects, list) and len(objects) == 12, "expected 12 objects")
    require(len(set(objects)) == 12, "source object roster repeats")
    require(not (set(objects) & TARGET_OBJECTS), "target object entered audit")
    require(payload.get("take_indices") == [1, 2, 3, 4, 5, 6], "take roster changed")
    prior = payload.get("prior_action_class_audit")
    require(isinstance(prior, dict), "prior action-class audit binding is absent")
    require(prior.get("gate_passed") is False, "prior negative gate changed")
    require(
        prior.get("content_sha256")
        == "12450e44d7f32330d2d7608f3b13535e6373427e3bc60c0815038ccd10979247",
        "prior action-class result changed",
    )
    return payload


def locate_archive(root: Path, object_id: str, take_index: int) -> Path:
    expected = root / "poking" / object_id / f"{object_id}_T{take_index}.zip"
    if expected.is_file():
        return expected
    matches = sorted(root.rglob(f"{object_id}_T{take_index}.zip"))
    require(
        len(matches) == 1,
        f"expected one archive for {object_id} T{take_index}; found {len(matches)}",
    )
    path = matches[0]
    require("dropping" not in path.parts, "resolved a dropping archive")
    return path


def robot_member_name(archive: ZipFile, take_stem: str) -> str:
    exact = f"{take_stem}/robot_data.json"
    names = [info.filename for info in archive.infolist() if not info.is_dir()]
    if exact in names:
        return exact
    matches = [name for name in names if name.endswith("/robot_data.json")]
    require(len(matches) == 1, f"expected one robot member in {take_stem}")
    return matches[0]


def first_mesh_member(archive: ZipFile) -> str:
    candidates = []
    for info in archive.infolist():
        if info.is_dir():
            continue
        match = MESH_PATTERN.search(info.filename)
        if match is not None:
            candidates.append((int(match.group(1)), info.filename))
    require(bool(candidates), "archive contains no mesh OBJ member")
    return min(candidates)[1]


def parse_obj_vertices(payload: bytes) -> np.ndarray:
    vertices = []
    for line in payload.decode("utf-8", errors="strict").splitlines():
        if line.startswith("v "):
            fields = line.split()
            require(len(fields) >= 4, "invalid OBJ vertex")
            vertices.append(tuple(float(value) for value in fields[1:4]))
    require(len(vertices) >= 4, "initial mesh contains too few vertices")
    array = np.asarray(vertices, dtype=np.float64) / 1000.0
    require(np.all(np.isfinite(array)), "initial mesh contains non-finite vertices")
    return array


def parse_transform(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def parse_force(value: Any) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < 3 or not np.all(np.isfinite(vector[:3])):
        return None
    return vector[:3]


def parse_robot_records(
    payload: bytes,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    raw = json.loads(payload.decode("utf-8"))
    require(isinstance(raw, list), "robot_data.json is not a list")
    rows = []
    for index, record in enumerate(raw):
        if not isinstance(record, dict):
            continue
        transform = parse_transform(record.get("T_WT"))
        if transform is None:
            continue
        force = parse_force(record.get("forces"))
        try:
            frame = int(record.get("frame", index))
        except (TypeError, ValueError):
            continue
        rows.append((frame, transform, force))
    rows.sort(key=lambda item: item[0])
    require(len(rows) >= 10, "fewer than ten valid robot records")
    frames = np.asarray([item[0] for item in rows], dtype=np.int64)
    transforms = np.asarray([item[1] for item in rows], dtype=np.float64)
    forces = np.asarray(
        [item[2] if item[2] is not None else np.full(3, np.nan) for item in rows],
        dtype=np.float64,
    )
    valid_force = np.all(np.isfinite(forces), axis=1)
    return frames, transforms, forces, valid_force


def force_contact_index(
    forces: np.ndarray,
    valid_force: np.ndarray,
    threshold: float,
) -> int | None:
    active = valid_force & (forces[:, 1] > threshold)
    indices = np.flatnonzero(active)
    return int(indices[0]) if indices.size else None


def tool_points(transforms: np.ndarray, axis: int, offset: float) -> np.ndarray:
    local = np.zeros(3, dtype=np.float64)
    local[axis] = offset
    return transforms[:, :3, 3] + np.einsum(
        "nij,j->ni",
        transforms[:, :3, :3],
        local,
    )


def geometric_contact(
    transforms: np.ndarray,
    tree: cKDTree,
    axis: int,
    offset: float,
) -> tuple[int, float, np.ndarray]:
    points = tool_points(transforms, axis, offset)
    distances, _ = tree.query(points, k=1, workers=1)
    index = int(np.argmin(distances))
    return index, float(distances[index]), points


def normalized_descriptor(
    frames: np.ndarray,
    points: np.ndarray,
    contact_index: int,
    vertices: np.ndarray,
    window: int,
) -> np.ndarray:
    minimum = np.min(vertices, axis=0)
    maximum = np.max(vertices, axis=0)
    center = 0.5 * (minimum + maximum)
    extent = maximum - minimum
    diagonal = float(np.linalg.norm(extent))
    require(diagonal > 1e-9, "initial mesh diagonal is degenerate")
    safe_extent = np.where(extent > 1e-9, extent, diagonal)

    start = max(0, contact_index - window)
    end = min(len(points) - 1, contact_index + window)
    approach_start = max(0, contact_index - window)
    approach = points[contact_index] - points[approach_start]
    approach_norm = float(np.linalg.norm(approach))
    if approach_norm > 1e-12:
        approach = approach / approach_norm
    else:
        approach = np.zeros(3, dtype=np.float64)
    stroke = (points[end] - points[start]) / diagonal
    local_steps = np.linalg.norm(np.diff(points[start : end + 1], axis=0), axis=1)
    path = float(np.sum(local_steps)) / diagonal
    duration = int(frames[end] - frames[start] + 1)
    contact = (points[contact_index] - center) / safe_extent
    descriptor = np.asarray(
        [
            *contact.tolist(),
            *approach.tolist(),
            *stroke.tolist(),
            math.log1p(path),
            math.log1p(duration),
        ],
        dtype=np.float64,
    )
    require(np.all(np.isfinite(descriptor)), "descriptor is non-finite")
    return descriptor


def choose_tip_model(
    episodes: list[dict[str, Any]],
    axes: list[int],
    offsets: list[float],
) -> dict[str, Any]:
    candidates = []
    for axis in axes:
        for offset in offsets:
            errors = []
            distances = []
            for episode in episodes:
                force_index = episode["force_contact_index"]
                if force_index is None:
                    continue
                predicted, distance, _ = geometric_contact(
                    episode["transforms"],
                    episode["tree"],
                    axis,
                    offset,
                )
                errors.append(abs(predicted - int(force_index)))
                distances.append(distance)
            require(bool(errors), "no force contacts available for tip calibration")
            candidates.append(
                {
                    "axis": axis,
                    "offset_m": offset,
                    "median_absolute_index_error": float(np.median(errors)),
                    "p90_absolute_index_error": float(np.quantile(errors, 0.9)),
                    "median_surface_distance_m": float(np.median(distances)),
                }
            )
    return min(
        candidates,
        key=lambda item: (
            item["median_absolute_index_error"],
            item["p90_absolute_index_error"],
            item["median_surface_distance_m"],
            abs(item["offset_m"]),
            item["axis"],
        ),
    )


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def intrinsic_rank(matrix: np.ndarray, tolerance: float) -> int:
    singular = np.linalg.svd(matrix, compute_uv=False)
    if singular.size == 0 or singular[0] <= 0.0:
        return 0
    return int(np.sum(singular / singular[0] >= tolerance))


def object_action_spread(
    standardized: np.ndarray,
    objects: list[str],
) -> dict[str, Any]:
    per_object = {}
    medians = []
    for object_id in sorted(set(objects)):
        indices = [index for index, value in enumerate(objects) if value == object_id]
        rows = standardized[indices]
        distances = []
        for left in range(len(rows)):
            for right in range(left + 1, len(rows)):
                distances.append(float(np.linalg.norm(rows[left] - rows[right])))
        median = float(np.median(distances))
        medians.append(median)
        per_object[object_id] = {
            "pair_count": len(distances),
            "median_pairwise_distance": median,
            "minimum_pairwise_distance": float(np.min(distances)),
            "maximum_pairwise_distance": float(np.max(distances)),
        }
    return {
        "minimum_object_median_pairwise_distance": float(np.min(medians)),
        "median_object_median_pairwise_distance": float(np.median(medians)),
        "per_object": per_object,
    }


def cross_object_coverage(
    standardized: np.ndarray,
    objects: list[str],
) -> dict[str, Any]:
    nearest = []
    records = []
    for index, object_id in enumerate(objects):
        candidates = [
            other for other, value in enumerate(objects) if value != object_id
        ]
        distances = np.linalg.norm(
            standardized[candidates] - standardized[index],
            axis=1,
        )
        best_position = int(np.argmin(distances))
        best_index = candidates[best_position]
        nearest.append(float(distances[best_position]))
        records.append(
            {
                "object_id": object_id,
                "nearest_object_id": objects[best_index],
                "distance": float(distances[best_position]),
            }
        )
    return {
        "median_nearest_other_object_distance": float(np.median(nearest)),
        "p90_nearest_other_object_distance": float(np.quantile(nearest, 0.9)),
        "maximum_nearest_other_object_distance": float(np.max(nearest)),
        "records": records,
    }


def run(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), "PokeFlex root does not exist")
    objects = [str(value) for value in request["source_and_calibration_objects"]]
    takes = [int(value) for value in request["take_indices"]]
    threshold = float(request["contact_model"]["force_threshold_n"])
    window = int(request["contact_model"]["descriptor_window_frames"])
    axes = [int(value) for value in request["contact_model"]["candidate_tool_axes"]]
    offsets = [
        float(value) for value in request["contact_model"]["candidate_tip_offsets_m"]
    ]

    episodes = []
    payload_bytes = 0
    for object_id in objects:
        require(object_id not in TARGET_OBJECTS, "target object entered audit")
        for take_index in takes:
            archive_path = locate_archive(root, object_id, take_index)
            take_stem = f"{object_id}_T{take_index}"
            with ZipFile(archive_path) as archive:
                robot_member = robot_member_name(archive, take_stem)
                mesh_member = first_mesh_member(archive)
                robot_payload = archive.read(robot_member)
                mesh_payload = archive.read(mesh_member)
            payload_bytes += len(robot_payload) + len(mesh_payload)
            frames, transforms, forces, valid_force = parse_robot_records(
                robot_payload
            )
            vertices = parse_obj_vertices(mesh_payload)
            episodes.append(
                {
                    "object_id": object_id,
                    "take_index": take_index,
                    "archive_relative_path": str(archive_path.relative_to(root)),
                    "robot_member": robot_member,
                    "robot_sha256": hashlib.sha256(robot_payload).hexdigest(),
                    "initial_mesh_member": mesh_member,
                    "initial_mesh_sha256": hashlib.sha256(mesh_payload).hexdigest(),
                    "frames": frames,
                    "transforms": transforms,
                    "forces": forces,
                    "valid_force": valid_force,
                    "force_contact_index": force_contact_index(
                        forces,
                        valid_force,
                        threshold,
                    ),
                    "vertices": vertices,
                    "tree": cKDTree(vertices),
                }
            )

    tip_model = choose_tip_model(episodes, axes, offsets)
    axis = int(tip_model["axis"])
    offset = float(tip_model["offset_m"])
    records = []
    descriptors = []
    object_rows = []
    contact_errors = []
    surface_distances = []
    descriptor_discrepancies = []
    force_contact_count = 0
    for episode in episodes:
        predicted, surface_distance, points = geometric_contact(
            episode["transforms"],
            episode["tree"],
            axis,
            offset,
        )
        predicted_descriptor = normalized_descriptor(
            episode["frames"],
            points,
            predicted,
            episode["vertices"],
            window,
        )
        force_index = episode["force_contact_index"]
        error = None
        force_descriptor = None
        discrepancy = None
        if force_index is not None:
            force_contact_count += 1
            error = abs(predicted - int(force_index))
            contact_errors.append(error)
            force_descriptor = normalized_descriptor(
                episode["frames"],
                points,
                int(force_index),
                episode["vertices"],
                window,
            )
            discrepancy = float(np.linalg.norm(predicted_descriptor - force_descriptor))
            descriptor_discrepancies.append(discrepancy)
        surface_distances.append(surface_distance)
        descriptors.append(predicted_descriptor)
        object_rows.append(episode["object_id"])
        records.append(
            {
                "object_id": episode["object_id"],
                "take_index": episode["take_index"],
                "archive_relative_path": episode["archive_relative_path"],
                "robot_member": episode["robot_member"],
                "robot_sha256": episode["robot_sha256"],
                "initial_mesh_member": episode["initial_mesh_member"],
                "initial_mesh_sha256": episode["initial_mesh_sha256"],
                "force_contact_index": force_index,
                "geometry_contact_index": predicted,
                "absolute_contact_index_error": error,
                "geometry_contact_surface_distance_m": surface_distance,
                "geometry_descriptor": dict(
                    zip(DESCRIPTOR_NAMES, predicted_descriptor.tolist(), strict=True)
                ),
                "force_contact_descriptor": (
                    dict(zip(DESCRIPTOR_NAMES, force_descriptor.tolist(), strict=True))
                    if force_descriptor is not None
                    else None
                ),
                "descriptor_discrepancy": discrepancy,
            }
        )

    matrix = np.asarray(descriptors, dtype=np.float64)
    standardized, mean, scale = standardize(matrix)
    centered = standardized - np.mean(standardized, axis=0)
    rank = intrinsic_rank(
        centered,
        float(request["gate_thresholds"]["intrinsic_rank_relative_tolerance"]),
    )
    spread = object_action_spread(standardized, object_rows)
    coverage = cross_object_coverage(standardized, object_rows)
    contact_fraction = force_contact_count / len(episodes)
    median_error = float(np.median(contact_errors)) if contact_errors else math.inf
    p90_error = (
        float(np.quantile(contact_errors, 0.9)) if contact_errors else math.inf
    )
    median_surface = float(np.median(surface_distances))
    median_discrepancy = (
        float(np.median(descriptor_discrepancies))
        if descriptor_discrepancies
        else math.inf
    )

    thresholds = request["gate_thresholds"]
    checks = {
        "complete_12_by_6_panel": len(records) == 72,
        "minimum_force_contact_fraction": contact_fraction
        >= float(thresholds["minimum_force_contact_fraction"]),
        "maximum_median_contact_index_error": median_error
        <= float(thresholds["maximum_median_contact_index_error"]),
        "maximum_p90_contact_index_error": p90_error
        <= float(thresholds["maximum_p90_contact_index_error"]),
        "maximum_median_surface_distance_m": median_surface
        <= float(thresholds["maximum_median_surface_distance_m"]),
        "minimum_intrinsic_rank": rank >= int(thresholds["minimum_intrinsic_rank"]),
        "minimum_object_action_spread": spread[
            "minimum_object_median_pairwise_distance"
        ]
        >= float(thresholds["minimum_object_median_pairwise_distance"]),
        "maximum_cross_object_coverage_distance": coverage[
            "p90_nearest_other_object_distance"
        ]
        <= float(thresholds["maximum_p90_nearest_other_object_distance"]),
        "target_objects_absent": not (set(object_rows) & TARGET_OBJECTS),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "status": (
            "continuous-probe-interface-qualified"
            if passed
            else "continuous-probe-interface-not-qualified"
        ),
        "prior_action_class_audit": request["prior_action_class_audit"],
        "source_and_calibration_objects": objects,
        "target_objects_excluded": sorted(TARGET_OBJECTS),
        "take_indices": takes,
        "descriptor_names": list(DESCRIPTOR_NAMES),
        "tip_model": tip_model,
        "records": records,
        "normalization": {
            "mean": mean.tolist(),
            "scale": scale.tolist(),
        },
        "contact_validation": {
            "force_contact_count": force_contact_count,
            "record_count": len(records),
            "force_contact_fraction": contact_fraction,
            "median_absolute_contact_index_error": median_error,
            "p90_absolute_contact_index_error": p90_error,
            "median_geometry_contact_surface_distance_m": median_surface,
            "median_geometry_vs_force_descriptor_discrepancy": median_discrepancy,
        },
        "descriptor_geometry": {
            "intrinsic_rank": rank,
            "object_action_spread": spread,
            "cross_object_coverage": coverage,
        },
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": passed,
            "next_stage": (
                "run-source-only-continuous-sequential-probe-qualification"
                if passed
                else "retain-continuous-actions-but-revise-contact-localization"
            ),
        },
        "information_boundary": {
            "source_and_calibration_poking_archive_open_count": len(episodes),
            "target_archive_open_count": 0,
            "drop_archive_open_count": 0,
            "robot_member_open_count": len(episodes),
            "initial_mesh_member_open_count": len(episodes),
            "other_member_open_count": 0,
            "payload_bytes_read": payload_bytes,
            "challenge_outcome_read": False,
        },
        "claim_boundary": [
            (
                "A pass establishes a source-validated continuous probe interface "
                "that does not rely on nominal take identity."
            ),
            (
                "It does not establish probe value, target transfer, drop-query "
                "prediction, online execution, or safety."
            ),
        ],
        "content_sha256": "",
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload["content_sha256"] = content_sha256(payload)
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    validation = payload["contact_validation"]
    geometry = payload["descriptor_geometry"]
    lines = [
        "# PokeFlex continuous probe-descriptor audit",
        "",
        f"- Status: **{payload['status']}**",
        (
            "- Tip model: local axis "
            f"**{payload['tip_model']['axis']}**, offset "
            f"**{payload['tip_model']['offset_m']:.4f} m**"
        ),
        (
            "- Force-contact coverage: "
            f"**{100.0 * validation['force_contact_fraction']:.1f}%**"
        ),
        (
            "- Median / p90 contact-index error: "
            f"**{validation['median_absolute_contact_index_error']:.1f} / "
            f"{validation['p90_absolute_contact_index_error']:.1f} frames**"
        ),
        (
            "- Median surface distance: "
            f"**{validation['median_geometry_contact_surface_distance_m']:.4f} m**"
        ),
        f"- Descriptor intrinsic rank: **{geometry['intrinsic_rank']}**",
        (
            "- Minimum object median action spread: "
            f"**{geometry['object_action_spread']['minimum_object_median_pairwise_distance']:.3f}**"
        ),
        (
            "- p90 nearest other-object action distance: "
            f"**{geometry['cross_object_coverage']['p90_nearest_other_object_distance']:.3f}**"
        ),
        f"- Gate passed: **{payload['gate']['passed']}**",
        "- Target archive opens: **0**",
        "- Drop archive opens: **0**",
        "",
        "This audit replaces invalid nominal take classes with a continuous,",
        "geometry-normalized action representation. It is not a held-target result.",
        "",
    ]
    (output_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    request = load_request(args.request)
    payload = run(args.root, request)
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["contact_validation"], indent=2, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
