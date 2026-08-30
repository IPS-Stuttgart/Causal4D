#!/usr/bin/env python3
"""Run the frozen PokeFlex five-bout source/calibration feasibility gate.

The source phase opens only the five prelocked source object meshes. Calibration
meshes are opened only when every source requirement passes. The three sealed
target directories are never traversed or stat'ed by this script. The response
functional and policies are fixed in the supplied prelock JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

EXPECTED_BOUTS = 5
SMOOTH_RADIUS = 2
PEAK_MIN_SEPARATION = 18
LOCAL_RADIUS = 3
MAX_VERTICES = 2048
CHAMFER_CHUNK = 128
LOG_EPSILON = 1e-9
OBSERVATION_NOISE_RATIO = 0.25
TARGET_FRAME_PATTERN = re.compile(r"mesh-f(\d+)\.(?:obj|ply)$", re.IGNORECASE)
FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class PolicyFit:
    mean_query: float
    mean_probe: tuple[float, ...]
    variances: tuple[float, ...]
    covariances: tuple[float, ...]
    task_values: tuple[float, ...]
    slopes: tuple[float, ...]
    selected_task: int
    selected_generic: int


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def force_vector(record: dict[str, object]) -> tuple[float, float, float] | None:
    value = record.get("forces")
    if not isinstance(value, list) or len(value) < 3:
        return None
    parsed = [finite(item) for item in value[:3]]
    if any(item is None for item in parsed):
        return None
    return float(parsed[0]), float(parsed[1]), float(parsed[2])


def moving_average(values: list[float], radius: int) -> list[float]:
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    result = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        result.append((prefix[stop] - prefix[start]) / (stop - start))
    return result


def select_peaks(smoothed: list[float]) -> list[int]:
    candidates = []
    for index, value in enumerate(smoothed):
        start = max(0, index - LOCAL_RADIUS)
        stop = min(len(smoothed), index + LOCAL_RADIUS + 1)
        if value >= max(smoothed[start:stop]):
            candidates.append(index)
    selected: list[int] = []
    for index in sorted(candidates, key=lambda item: (-smoothed[item], item)):
        if all(abs(index - prior) >= PEAK_MIN_SEPARATION for prior in selected):
            selected.append(index)
        if len(selected) == EXPECTED_BOUTS:
            break
    if len(selected) < EXPECTED_BOUTS:
        for index in sorted(range(len(smoothed)), key=lambda item: (-smoothed[item], item)):
            if all(abs(index - prior) >= PEAK_MIN_SEPARATION for prior in selected):
                selected.append(index)
            if len(selected) == EXPECTED_BOUTS:
                break
    return sorted(selected)


def record_frame(record: dict[str, object], index: int) -> int:
    value = record.get("frame")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return index + 1


def load_peak_frames(input_root: Path) -> tuple[list[int], list[float]]:
    robot_path = input_root / "robot_data.json"
    payload = json.loads(robot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError(f"{robot_path}: expected a nonempty list")
    records = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"{robot_path}: non-mapping record")
        records.append(item)
    force_vectors = [force_vector(record) for record in records]
    if any(item is None for item in force_vectors):
        raise ValueError(f"{robot_path}: missing finite force vectors")
    norms = [
        math.sqrt(first * first + second * second + third * third)
        for first, second, third in force_vectors
        if first is not None
    ]
    peaks = select_peaks(moving_average(norms, SMOOTH_RADIUS))
    if len(peaks) != EXPECTED_BOUTS:
        raise ValueError(f"{robot_path}: expected {EXPECTED_BOUTS} peaks, got {len(peaks)}")
    return [record_frame(records[index], index) for index in peaks], [norms[index] for index in peaks]


def obj_vertices(path: Path) -> FloatArray:
    vertices: list[tuple[float, float, float]] = []
    with path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if not line.startswith("v "):
                continue
            fields = line.split()
            if len(fields) < 4:
                continue
            values = tuple(float(field) for field in fields[1:4])
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{path}: nonfinite vertex")
            vertices.append(values)
    if len(vertices) < 4:
        raise ValueError(f"{path}: fewer than four valid vertices")
    result = np.asarray(vertices, dtype=np.float64)
    count = min(result.shape[0], MAX_VERTICES)
    indices = np.linspace(0, result.shape[0] - 1, count, dtype=np.int64)
    return np.ascontiguousarray(result[indices])


def bounding_box_diagonal(vertices: FloatArray) -> float:
    extent = np.max(vertices, axis=0) - np.min(vertices, axis=0)
    result = float(np.linalg.norm(extent))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("template bounding-box diagonal must be finite and positive")
    return result


def mean_nearest_distance(source: FloatArray, target: FloatArray) -> float:
    total = 0.0
    count = 0
    for start in range(0, source.shape[0], CHAMFER_CHUNK):
        batch = source[start : start + CHAMFER_CHUNK]
        difference = batch[:, None, :] - target[None, :, :]
        squared = np.einsum("bij,bij->bi", difference, difference, optimize=True)
        nearest = np.sqrt(np.min(squared, axis=1))
        total += float(np.sum(nearest))
        count += int(nearest.size)
    return total / count


def symmetric_chamfer(source: FloatArray, target: FloatArray) -> float:
    return 0.5 * (
        mean_nearest_distance(source, target) + mean_nearest_distance(target, source)
    )


def template_mesh(input_root: Path) -> Path:
    candidates = sorted((input_root / "meshes").glob("*.obj"))
    if len(candidates) != 1:
        raise ValueError(f"{input_root}: expected one input template OBJ, found {len(candidates)}")
    return candidates[0]


def target_meshes_by_frame(target_root: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    mesh_root = target_root / "meshes"
    for path in mesh_root.iterdir():
        match = TARGET_FRAME_PATTERN.fullmatch(path.name)
        if match is None:
            continue
        frame = int(match.group(1))
        if frame in result:
            raise ValueError(f"{mesh_root}: duplicate target mesh frame {frame}")
        result[frame] = path
    if not result:
        raise ValueError(f"{mesh_root}: no target mesh frames")
    return result


def nearest_mesh(meshes: dict[int, Path], frame: int) -> tuple[int, Path, int]:
    selected = min(meshes, key=lambda value: (abs(value - frame), value))
    return selected, meshes[selected], abs(selected - frame)


def extract_responses(
    *,
    names: Iterable[str],
    inputs: Path,
    targets: Path,
    role: str,
) -> tuple[dict[str, list[float]], list[dict[str, Any]], list[str]]:
    responses: dict[str, list[float]] = {}
    records: list[dict[str, Any]] = []
    opened: list[str] = []
    for name in names:
        input_root = inputs / name
        target_root = targets / name
        peaks, peak_forces = load_peak_frames(input_root)
        template_path = template_mesh(input_root)
        template = obj_vertices(template_path)
        diagonal = bounding_box_diagonal(template)
        meshes = target_meshes_by_frame(target_root)
        values: list[float] = []
        bout_records = []
        for order, (peak_frame, peak_force) in enumerate(zip(peaks, peak_forces), start=1):
            target_frame, target_path, frame_delta = nearest_mesh(meshes, peak_frame)
            if frame_delta > 3:
                raise ValueError(
                    f"{name} bout {order}: nearest target mesh is {frame_delta} frames away"
                )
            target = obj_vertices(target_path)
            native_chamfer = symmetric_chamfer(template, target)
            normalized = native_chamfer / diagonal
            if not math.isfinite(normalized) or normalized <= 0.0:
                raise ValueError(f"{name} bout {order}: invalid normalized Chamfer")
            log_response = math.log(normalized + LOG_EPSILON)
            values.append(log_response)
            opened.append(str(target_path))
            bout_records.append(
                {
                    "order": order,
                    "peak_frame": peak_frame,
                    "target_frame": target_frame,
                    "target_frame_delta": frame_delta,
                    "peak_force_norm_n": peak_force,
                    "template_vertex_sample_count": int(template.shape[0]),
                    "target_vertex_sample_count": int(target.shape[0]),
                    "native_symmetric_chamfer": native_chamfer,
                    "template_bbox_diagonal_native": diagonal,
                    "normalized_symmetric_chamfer": normalized,
                    "log_normalized_response": log_response,
                    "target_mesh_sha256": sha256_file(target_path),
                }
            )
        responses[name] = values
        records.append(
            {
                "name": name,
                "role": role,
                "template_mesh": str(template_path),
                "template_mesh_sha256": sha256_file(template_path),
                "template_bbox_diagonal_native": diagonal,
                "bouts": bout_records,
            }
        )
    return responses, records, opened


def sample_variance(values: FloatArray) -> float:
    if values.size < 2:
        raise ValueError("at least two training values are required")
    return float(np.var(values, ddof=1))


def sample_covariance(first: FloatArray, second: FloatArray) -> float:
    if first.size != second.size or first.size < 2:
        raise ValueError("matched vectors with at least two values are required")
    return float(np.cov(first, second, ddof=1)[0, 1])


def fit_policy(matrix: FloatArray, query_override: FloatArray | None = None) -> PolicyFit:
    if matrix.ndim != 2 or matrix.shape[1] != EXPECTED_BOUTS or matrix.shape[0] < 3:
        raise ValueError("response matrix must have shape (N>=3, 5)")
    probes = matrix[:, :4]
    query = matrix[:, 4] if query_override is None else query_override
    if query.shape != (matrix.shape[0],):
        raise ValueError("query override has the wrong shape")
    means = np.mean(probes, axis=0)
    mean_query = float(np.mean(query))
    variances = np.array([sample_variance(probes[:, index]) for index in range(4)])
    covariances = np.array(
        [sample_covariance(probes[:, index], query) for index in range(4)]
    )
    denominators = (1.0 + OBSERVATION_NOISE_RATIO) * variances + 1e-12
    task_values = covariances * covariances / denominators
    slopes = covariances / denominators
    selected_task = int(np.argmax(task_values))
    selected_generic = int(np.argmax(variances))
    return PolicyFit(
        mean_query=mean_query,
        mean_probe=tuple(float(value) for value in means),
        variances=tuple(float(value) for value in variances),
        covariances=tuple(float(value) for value in covariances),
        task_values=tuple(float(value) for value in task_values),
        slopes=tuple(float(value) for value in slopes),
        selected_task=selected_task,
        selected_generic=selected_generic,
    )


def predict(fit: PolicyFit, row: FloatArray, candidate: int) -> float:
    return fit.mean_query + fit.slopes[candidate] * (
        float(row[candidate]) - fit.mean_probe[candidate]
    )


def fit_json(fit: PolicyFit) -> dict[str, Any]:
    return {
        "mean_query": fit.mean_query,
        "mean_probe": list(fit.mean_probe),
        "variances": list(fit.variances),
        "covariances": list(fit.covariances),
        "task_values": list(fit.task_values),
        "slopes": list(fit.slopes),
        "selected_task_bout": fit.selected_task + 1,
        "selected_generic_bout": fit.selected_generic + 1,
    }


def source_evaluation(names: list[str], responses: dict[str, list[float]]) -> dict[str, Any]:
    matrix = np.asarray([responses[name] for name in names], dtype=np.float64)
    folds = []
    task_errors = []
    generic_errors = []
    no_probe_errors = []
    random_errors = []
    for held_index, held_name in enumerate(names):
        training = np.delete(matrix, held_index, axis=0)
        held = matrix[held_index]
        fit = fit_policy(training)
        truth = float(held[4])
        task_prediction = predict(fit, held, fit.selected_task)
        generic_prediction = predict(fit, held, fit.selected_generic)
        no_probe_prediction = fit.mean_query
        candidate_predictions = [predict(fit, held, index) for index in range(4)]
        task_error = (task_prediction - truth) ** 2
        generic_error = (generic_prediction - truth) ** 2
        no_probe_error = (no_probe_prediction - truth) ** 2
        random_error = float(np.mean([(value - truth) ** 2 for value in candidate_predictions]))
        task_errors.append(task_error)
        generic_errors.append(generic_error)
        no_probe_errors.append(no_probe_error)
        random_errors.append(random_error)
        folds.append(
            {
                "held_out": held_name,
                "fit": fit_json(fit),
                "truth_log_response": truth,
                "predictions": {
                    "task_conditioned": task_prediction,
                    "generic_information": generic_prediction,
                    "no_probe": no_probe_prediction,
                    "all_candidate_bouts": candidate_predictions,
                },
                "squared_log_errors": {
                    "task_conditioned": task_error,
                    "generic_information": generic_error,
                    "no_probe": no_probe_error,
                    "random_cost_matched_expectation": random_error,
                },
            }
        )
    full_fit = fit_policy(matrix)
    destroyed_query = np.roll(matrix[:, 4], 1)
    destroyed_fit = fit_policy(matrix, destroyed_query)
    aggregates = {
        "task_conditioned_mean_squared_log_error": float(np.mean(task_errors)),
        "generic_information_mean_squared_log_error": float(np.mean(generic_errors)),
        "no_probe_mean_squared_log_error": float(np.mean(no_probe_errors)),
        "random_cost_matched_expected_mean_squared_log_error": float(
            np.mean(random_errors)
        ),
        "task_conditioned_wins_vs_generic": sum(
            first < second for first, second in zip(task_errors, generic_errors)
        ),
        "task_conditioned_wins_vs_no_probe": sum(
            first < second for first, second in zip(task_errors, no_probe_errors)
        ),
    }
    checks = {
        "full_source_policies_select_different_bouts": (
            full_fit.selected_task != full_fit.selected_generic
        ),
        "task_loo_better_than_generic": (
            aggregates["task_conditioned_mean_squared_log_error"]
            < aggregates["generic_information_mean_squared_log_error"]
        ),
        "task_loo_better_than_no_probe": (
            aggregates["task_conditioned_mean_squared_log_error"]
            < aggregates["no_probe_mean_squared_log_error"]
        ),
        "matched_task_value_greater_than_destroyed": (
            max(full_fit.task_values) > max(destroyed_fit.task_values)
        ),
    }
    return {
        "names": names,
        "folds": folds,
        "full_source_fit": fit_json(full_fit),
        "dependence_destroyed_fit": fit_json(destroyed_fit),
        "aggregates": aggregates,
        "checks": checks,
        "passed": all(checks.values()),
    }


def calibration_evaluation(
    names: list[str],
    responses: dict[str, list[float]],
    source_fit: PolicyFit,
) -> dict[str, Any]:
    records = []
    totals = {
        "task_conditioned": 0.0,
        "generic_information": 0.0,
        "no_probe": 0.0,
        "random_cost_matched_expectation": 0.0,
    }
    no_catastrophic = True
    for name in names:
        row = np.asarray(responses[name], dtype=np.float64)
        truth = float(row[4])
        task_prediction = predict(source_fit, row, source_fit.selected_task)
        generic_prediction = predict(source_fit, row, source_fit.selected_generic)
        no_probe_prediction = source_fit.mean_query
        candidate_predictions = [predict(source_fit, row, index) for index in range(4)]
        errors = {
            "task_conditioned": (task_prediction - truth) ** 2,
            "generic_information": (generic_prediction - truth) ** 2,
            "no_probe": (no_probe_prediction - truth) ** 2,
            "random_cost_matched_expectation": float(
                np.mean([(value - truth) ** 2 for value in candidate_predictions])
            ),
        }
        for key, value in errors.items():
            totals[key] += value
        no_catastrophic = no_catastrophic and (
            errors["task_conditioned"] <= 2.0 * errors["no_probe"] + 1e-12
        )
        records.append(
            {
                "name": name,
                "truth_log_response": truth,
                "predictions": {
                    "task_conditioned": task_prediction,
                    "generic_information": generic_prediction,
                    "no_probe": no_probe_prediction,
                    "all_candidate_bouts": candidate_predictions,
                },
                "squared_log_errors": errors,
            }
        )
    checks = {
        "task_total_better_than_generic": (
            totals["task_conditioned"] < totals["generic_information"]
        ),
        "task_total_better_than_no_probe": (
            totals["task_conditioned"] < totals["no_probe"]
        ),
        "no_object_above_twice_no_probe_error": no_catastrophic,
    }
    return {
        "names": names,
        "source_fit": fit_json(source_fit),
        "records": records,
        "total_squared_log_errors": totals,
        "checks": checks,
        "passed": all(checks.values()),
    }


def write_result(path: Path, report: dict[str, Any]) -> None:
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prelock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite existing result")
    prelock = json.loads(args.prelock.read_text(encoding="utf-8"))
    if prelock.get("schema") != "causal4d.pokeflex-five-bout-probe-prelock-v1":
        parser.error("unexpected prelock schema")
    dataset = prelock["dataset"]
    split = prelock["split_rule"]
    inputs = Path(dataset["input_root"])
    targets = Path(dataset["target_root"])
    source_names = list(split["source"])
    calibration_names = list(split["calibration"])
    sealed_target_names = list(split["sealed_target"])

    source_responses, source_records, source_opened = extract_responses(
        names=source_names,
        inputs=inputs,
        targets=targets,
        role="source",
    )
    source = source_evaluation(source_names, source_responses)
    report: dict[str, Any] = {
        "schema": "causal4d.pokeflex-five-bout-source-calibration-result-v1",
        "protocol_id": prelock["protocol_id"],
        "prelock_sha256": sha256_file(args.prelock),
        "prelock_canonical_sha256": canonical_sha256(prelock),
        "source_revision": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
        "method": {
            "max_vertices_per_mesh": MAX_VERTICES,
            "chamfer_chunk": CHAMFER_CHUNK,
            "log_epsilon": LOG_EPSILON,
            "observation_noise_ratio": OBSERVATION_NOISE_RATIO,
            "source_gate_precedes_calibration_access": True,
        },
        "source": source,
        "source_response_records": source_records,
        "opened_target_meshes": source_opened,
        "calibration": {
            "state": "closed-pending-source-gate",
            "passed": False,
        },
        "sealed_target": {
            "names": sealed_target_names,
            "directories_traversed": False,
            "files_stat_ed": False,
            "geometry_opened": False,
        },
        "claim_boundary": prelock["claim_boundary"],
    }
    if not source["passed"]:
        report["decision"] = "source-negative-calibration-and-target-remain-closed"
        write_result(args.output, report)
        print(json.dumps({"decision": report["decision"], "output": str(args.output)}))
        return 0

    calibration_responses, calibration_records, calibration_opened = extract_responses(
        names=calibration_names,
        inputs=inputs,
        targets=targets,
        role="calibration",
    )
    source_matrix = np.asarray(
        [source_responses[name] for name in source_names], dtype=np.float64
    )
    full_source_fit = fit_policy(source_matrix)
    calibration = calibration_evaluation(
        calibration_names,
        calibration_responses,
        full_source_fit,
    )
    report["calibration"] = calibration
    report["calibration_response_records"] = calibration_records
    report["opened_target_meshes"].extend(calibration_opened)
    report["decision"] = (
        "source-and-calibration-positive-target-still-closed"
        if calibration["passed"]
        else "calibration-negative-target-remains-closed"
    )
    write_result(args.output, report)
    print(
        json.dumps(
            {
                "decision": report["decision"],
                "output": str(args.output),
                "opened_target_mesh_count": len(report["opened_target_meshes"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
