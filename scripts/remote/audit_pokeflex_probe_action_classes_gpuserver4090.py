#!/usr/bin/env python3
"""Audit source-only PokeFlex probe action classes and response diversity."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np

SCHEMA = "causal4d/pokeflex-probe-action-class-audit"
SCHEMA_VERSION = 1
ACTION_FEATURE_NAMES = (
    "direction_x",
    "direction_y",
    "direction_z",
    "log_path_length_m",
    "straightness",
    "log_duration_frames",
    "log_mean_step_mm",
)
RESPONSE_FEATURE_NAMES = (
    "log_peak_force_norm_n",
    "log_peak_abs_axis1_n",
    "log_force_norm_sum",
    "log_active_frame_count",
    "log_mean_active_force_norm_n",
)
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
        == "causal4d/pokeflex-probe-action-class-audit-request",
        "unexpected request schema",
    )
    require(payload.get("schema_version") == 1, "unexpected request version")
    objects = payload.get("source_and_calibration_objects")
    require(isinstance(objects, list) and len(objects) == 12, "expected 12 objects")
    require(len(set(objects)) == len(objects), "object roster repeats")
    require(not (set(objects) & TARGET_OBJECTS), "target object entered source audit")
    require(
        payload.get("take_indices") == [1, 2, 3, 4, 5, 6],
        "take-index roster changed",
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
    require("dropping" not in path.parts, f"resolved dropping archive for {object_id}")
    return path


def parse_transform(value: Any) -> np.ndarray | None:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    return matrix


def finite_vector(value: Any, minimum_size: int) -> np.ndarray | None:
    try:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if vector.size < minimum_size or not np.all(np.isfinite(vector)):
        return None
    return vector


def robot_member_name(archive: ZipFile, take_stem: str) -> str:
    exact = f"{take_stem}/robot_data.json"
    names = [info.filename for info in archive.infolist() if not info.is_dir()]
    if exact in names:
        return exact
    matches = [name for name in names if name.endswith("/robot_data.json")]
    require(len(matches) == 1, f"expected one robot member in {take_stem}")
    return matches[0]


def action_response_features(
    records: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    rows: list[tuple[int, np.ndarray, np.ndarray | None]] = []
    for index, record in enumerate(records):
        require(isinstance(record, dict), "robot record is not a mapping")
        transform = parse_transform(record.get("T_WT"))
        if transform is None:
            continue
        raw_frame = record.get("frame", index)
        try:
            frame = int(raw_frame)
        except (TypeError, ValueError):
            continue
        force = finite_vector(record.get("forces"), 3)
        rows.append((frame, transform[:3, 3].copy(), force))
    rows.sort(key=lambda item: item[0])
    require(len(rows) >= 3, "fewer than three valid tool poses")
    frames = np.asarray([item[0] for item in rows], dtype=np.int64)
    positions = np.asarray([item[1] for item in rows], dtype=np.float64)
    require(np.all(np.diff(frames) >= 0), "robot frames are unordered")
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    displacement = positions[-1] - positions[0]
    displacement_norm = float(np.linalg.norm(displacement))
    path_length = float(np.sum(steps))
    direction = (
        displacement / displacement_norm
        if displacement_norm > 1e-12
        else np.zeros(3, dtype=np.float64)
    )
    duration = int(frames[-1] - frames[0] + 1)
    mean_step_mm = float(np.mean(steps) * 1000.0) if steps.size else 0.0
    straightness = displacement_norm / max(path_length, 1e-12)
    action = np.asarray(
        [
            *direction.tolist(),
            math.log1p(path_length),
            straightness,
            math.log1p(duration),
            math.log1p(mean_step_mm),
        ],
        dtype=np.float64,
    )

    forces = np.asarray(
        [
            item[2][:3] if item[2] is not None else np.full(3, np.nan)
            for item in rows
        ],
        dtype=np.float64,
    )
    valid = np.all(np.isfinite(forces), axis=1)
    require(int(np.sum(valid)) >= 3, "fewer than three valid force records")
    forces = forces[valid]
    norms = np.linalg.norm(forces, axis=1)
    axis1 = np.abs(forces[:, 1])
    active = axis1 > 3.0
    mean_active = float(np.mean(norms[active])) if np.any(active) else 0.0
    response = np.asarray(
        [
            math.log1p(float(np.max(norms))),
            math.log1p(float(np.max(axis1))),
            math.log1p(float(np.sum(norms))),
            math.log1p(int(np.sum(active))),
            math.log1p(mean_active),
        ],
        dtype=np.float64,
    )
    require(np.all(np.isfinite(action)), "non-finite action features")
    require(np.all(np.isfinite(response)), "non-finite response features")
    metadata = {
        "valid_pose_count": int(len(rows)),
        "valid_force_count": int(forces.shape[0]),
        "first_frame": int(frames[0]),
        "last_frame": int(frames[-1]),
        "path_length_m": path_length,
        "displacement_m": displacement_norm,
        "active_force_frame_count": int(np.sum(active)),
    }
    return action, response, metadata


def standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(matrix, axis=0)
    scale = np.std(matrix, axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (matrix - mean) / scale, mean, scale


def leave_one_object_out_accuracy(
    features: np.ndarray,
    objects: list[str],
    labels: np.ndarray,
) -> tuple[float, list[list[int]], list[dict[str, Any]]]:
    unique_labels = sorted(set(int(value) for value in labels))
    confusion = np.zeros((len(unique_labels), len(unique_labels)), dtype=np.int64)
    label_to_row = {label: index for index, label in enumerate(unique_labels)}
    predictions: list[dict[str, Any]] = []
    correct = 0
    for held_object in sorted(set(objects)):
        train = np.asarray(
            [object_id != held_object for object_id in objects], dtype=bool
        )
        test_indices = [
            index for index, object_id in enumerate(objects) if object_id == held_object
        ]
        centroids = {
            label: np.mean(features[train & (labels == label)], axis=0)
            for label in unique_labels
        }
        for index in test_indices:
            distances = {
                label: float(np.linalg.norm(features[index] - centroid))
                for label, centroid in centroids.items()
            }
            predicted = min(distances, key=lambda label: (distances[label], label))
            actual = int(labels[index])
            confusion[label_to_row[actual], label_to_row[predicted]] += 1
            correct += int(predicted == actual)
            predictions.append(
                {
                    "object_id": held_object,
                    "take_index": actual,
                    "predicted_take_index": int(predicted),
                    "nearest_distance": distances[predicted],
                    "correct": bool(predicted == actual),
                }
            )
    return correct / len(objects), confusion.tolist(), predictions


def pairwise_distances(rows: np.ndarray) -> list[float]:
    values: list[float] = []
    for left in range(len(rows)):
        for right in range(left + 1, len(rows)):
            values.append(float(np.linalg.norm(rows[left] - rows[right])))
    return values


def separation_statistics(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    unique_labels = sorted(set(int(value) for value in labels))
    within: list[float] = []
    centroids = []
    for label in unique_labels:
        rows = features[labels == label]
        within.extend(pairwise_distances(rows))
        centroids.append(np.mean(rows, axis=0))
    between = pairwise_distances(np.asarray(centroids, dtype=np.float64))
    median_within = float(np.median(within))
    median_between = float(np.median(between))
    return {
        "median_within_take_distance": median_within,
        "median_between_take_centroid_distance": median_between,
        "between_to_within_ratio": median_between / max(median_within, 1e-12),
    }


def response_diversity(
    features: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    per_take = {}
    nonzero = 0
    for label in sorted(set(int(value) for value in labels)):
        rows = features[labels == label]
        variance = np.var(rows, axis=0, ddof=0)
        total = float(np.sum(variance))
        nonzero += int(total > 1e-8)
        per_take[str(label)] = {
            "feature_variance": variance.tolist(),
            "total_standardized_variance": total,
        }
    return {
        "take_classes_with_nonzero_response_variance": nonzero,
        "per_take": per_take,
    }


def write_outputs(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    payload["content_sha256"] = content_sha256(payload)
    (output_dir / "result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    gate = payload["gate"]
    summary = payload["summary"]
    lines = [
        "# PokeFlex source-only probe action-class audit",
        "",
        f"- Status: **{payload['status']}**",
        f"- Source/calibration objects: **{summary['object_count']}**",
        f"- Poking records: **{summary['record_count']}**",
        f"- LOO take-label accuracy: **{summary['loo_take_accuracy']:.3f}**",
        (
            "- Between/within action separation: "
            f"**{summary['between_to_within_ratio']:.3f}**"
        ),
        (
            "- Take classes with nonzero response variance: "
            f"**{summary['response_variable_take_count']}/6**"
        ),
        f"- Gate passed: **{gate['passed']}**",
        (
            "- Target-object archive opens: "
            f"**{payload['information_boundary']['target_archive_open_count']}**"
        ),
        (
            "- Mesh/image member opens: "
            f"**{payload['information_boundary']['non_robot_member_open_count']}**"
        ),
        "",
        "A pass authorizes treating the six common take indices as source-supported",
        "action classes for the subsequent source-only probe-to-drop study. It is",
        "not evidence that probing improves a held challenge.",
        "",
    ]
    (output_dir / "result.md").write_text("\n".join(lines), encoding="utf-8")


def run(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    require(root.is_dir(), "PokeFlex root does not exist")
    objects = [str(value) for value in request["source_and_calibration_objects"]]
    take_indices = [int(value) for value in request["take_indices"]]
    records = []
    action_rows = []
    response_rows = []
    object_rows: list[str] = []
    labels = []
    bytes_read = 0
    opened_archives = []
    for object_id in objects:
        require(object_id not in TARGET_OBJECTS, "target object entered source audit")
        for take_index in take_indices:
            archive_path = locate_archive(root, object_id, take_index)
            take_stem = f"{object_id}_T{take_index}"
            with ZipFile(archive_path) as archive:
                member = robot_member_name(archive, take_stem)
                payload = archive.read(member)
            bytes_read += len(payload)
            opened_archives.append(str(archive_path.relative_to(root)))
            raw_records = json.loads(payload.decode("utf-8"))
            require(isinstance(raw_records, list), "robot_data.json is not a list")
            action, response, metadata = action_response_features(raw_records)
            records.append(
                {
                    "object_id": object_id,
                    "take_index": take_index,
                    "action_qualified_take_id": f"poking:{take_stem}",
                    "archive_relative_path": str(archive_path.relative_to(root)),
                    "robot_member": member,
                    "robot_member_sha256": hashlib.sha256(payload).hexdigest(),
                    "robot_member_bytes": len(payload),
                    "action_features": dict(
                        zip(ACTION_FEATURE_NAMES, action.tolist(), strict=True)
                    ),
                    "response_features": dict(
                        zip(RESPONSE_FEATURE_NAMES, response.tolist(), strict=True)
                    ),
                    "metadata": metadata,
                }
            )
            action_rows.append(action)
            response_rows.append(response)
            object_rows.append(object_id)
            labels.append(take_index)

    action_matrix = np.asarray(action_rows, dtype=np.float64)
    response_matrix = np.asarray(response_rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    standardized_actions, action_mean, action_scale = standardize(action_matrix)
    standardized_responses, response_mean, response_scale = standardize(
        response_matrix
    )
    accuracy, confusion, predictions = leave_one_object_out_accuracy(
        standardized_actions,
        object_rows,
        label_array,
    )
    separation = separation_statistics(standardized_actions, label_array)
    diversity = response_diversity(standardized_responses, label_array)

    thresholds = request["gate_thresholds"]
    checks = {
        "complete_12_by_6_panel": len(records) == 72,
        "minimum_take_label_accuracy": accuracy
        >= float(thresholds["minimum_loo_take_accuracy"]),
        "minimum_between_within_ratio": separation["between_to_within_ratio"]
        >= float(thresholds["minimum_between_to_within_ratio"]),
        "minimum_variable_response_classes": diversity[
            "take_classes_with_nonzero_response_variance"
        ]
        >= int(thresholds["minimum_variable_response_take_classes"]),
        "target_objects_absent": not (set(object_rows) & TARGET_OBJECTS),
    }
    passed = all(checks.values())
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "status": (
            "source-action-classes-qualified"
            if passed
            else "source-action-classes-not-qualified"
        ),
        "dataset": {
            "root": str(root),
            "root_not_uploaded": True,
        },
        "source_and_calibration_objects": objects,
        "target_objects_excluded": sorted(TARGET_OBJECTS),
        "take_indices": take_indices,
        "action_feature_names": list(ACTION_FEATURE_NAMES),
        "response_feature_names": list(RESPONSE_FEATURE_NAMES),
        "records": records,
        "normalization": {
            "action_mean": action_mean.tolist(),
            "action_scale": action_scale.tolist(),
            "response_mean": response_mean.tolist(),
            "response_scale": response_scale.tolist(),
        },
        "classification": {
            "leave_one_object_out_accuracy": accuracy,
            "confusion_matrix_rows_actual_columns_predicted": confusion,
            "predictions": predictions,
        },
        "action_separation": separation,
        "response_diversity": diversity,
        "gate": {
            "thresholds": thresholds,
            "checks": checks,
            "passed": passed,
            "next_stage": (
                "run-source-only-sequential-probe-to-drop-qualification"
                if passed
                else "do-not-treat-take-index-as-action-class"
            ),
        },
        "summary": {
            "object_count": len(set(object_rows)),
            "record_count": len(records),
            "loo_take_accuracy": accuracy,
            "between_to_within_ratio": separation["between_to_within_ratio"],
            "response_variable_take_count": diversity[
                "take_classes_with_nonzero_response_variance"
            ],
        },
        "information_boundary": {
            "source_and_calibration_archive_open_count": len(opened_archives),
            "target_archive_open_count": 0,
            "robot_member_open_count": len(opened_archives),
            "non_robot_member_open_count": 0,
            "robot_member_payload_bytes_read": bytes_read,
            "mesh_payload_bytes_read": 0,
            "image_payload_bytes_read": 0,
            "drop_payload_bytes_read": 0,
            "challenge_outcome_read": False,
        },
        "claim_boundary": [
            "A pass establishes source-supported action-class comparability only.",
            (
                "It does not establish probe value, drop prediction, target transfer, "
                "online execution, or safety."
            ),
        ],
        "content_sha256": "",
    }


def main() -> int:
    args = parse_args()
    request = load_request(args.request)
    payload = run(args.root, request)
    write_outputs(args.output_dir, payload)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
