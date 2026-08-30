#!/usr/bin/env python3
"""Test whether each readable PokeFlex take contains five separable poke bouts.

Only source robot records and target mesh *filenames* are read. Target geometry
bytes remain unopened. The fixed segmentation is intended as a structural gate
for a retrospective diagnostic-poke study, not as an outcome-optimized split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
from pathlib import Path
from typing import Any

SCHEMA = "causal4d.pokeflex-five-bout-structure-v1"
EXPECTED_BOUTS = 5
SMOOTH_RADIUS = 2
PEAK_MIN_SEPARATION = 18
LOCAL_RADIUS = 3
TARGET_FRAME_PATTERN = re.compile(r"mesh-f(\d+)\.(?:obj|ply)$", re.IGNORECASE)


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _force(record: dict[str, object]) -> list[float] | None:
    value = record.get("forces")
    if not isinstance(value, list) or len(value) < 3:
        return None
    result = [_finite(item) for item in value[:3]]
    if any(item is None for item in result):
        return None
    return [float(item) for item in result]


def _position(record: dict[str, object]) -> list[float] | None:
    value = record.get("T_WT")
    if not isinstance(value, list) or len(value) < 3:
        return None
    result: list[float] = []
    for row in value[:3]:
        if not isinstance(row, list) or len(row) < 4:
            return None
        item = _finite(row[3])
        if item is None:
            return None
        result.append(item)
    return result


def _frame(record: dict[str, object], index: int) -> int:
    value = record.get("frame")
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    if isinstance(value, str) and value.isascii() and value.isdigit():
        return int(value)
    return index + 1


def _moving_average(values: list[float], radius: int) -> list[float]:
    prefix = [0.0]
    for value in values:
        prefix.append(prefix[-1] + value)
    result = []
    for index in range(len(values)):
        start = max(0, index - radius)
        stop = min(len(values), index + radius + 1)
        result.append((prefix[stop] - prefix[start]) / (stop - start))
    return result


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _select_peaks(smoothed: list[float]) -> list[int]:
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


def _valley(smoothed: list[float], left: int, right: int) -> int:
    if right <= left + 1:
        return left
    return min(range(left + 1, right), key=lambda index: (smoothed[index], index))


def _distance(first: list[float], second: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(first, second)))


def _target_frames(root: Path) -> list[int]:
    result = []
    mesh_root = root / "meshes"
    if not mesh_root.is_dir():
        return result
    for path in mesh_root.iterdir():
        match = TARGET_FRAME_PATTERN.fullmatch(path.name)
        if match is not None:
            result.append(int(match.group(1)))
    return sorted(set(result))


def _nearest(values: list[int], target: int) -> tuple[int | None, int | None]:
    if not values:
        return None, None
    value = min(values, key=lambda item: (abs(item - target), item))
    return value, abs(value - target)


def analyze_take(input_root: Path, target_root: Path) -> dict[str, Any]:
    robot_path = input_root / "robot_data.json"
    result: dict[str, Any] = {
        "name": input_root.name,
        "robot_path": str(robot_path),
        "target_mesh_directory": str(target_root / "meshes"),
        "target_geometry_read": False,
    }
    try:
        payload = json.loads(robot_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        result["error"] = repr(exc)
        return result
    if not isinstance(payload, list) or not payload:
        result["error"] = "robot_data.json is not a nonempty list"
        return result
    records = [item for item in payload if isinstance(item, dict)]
    if len(records) != len(payload):
        result["error"] = "robot_data.json contains non-mapping records"
        return result

    forces = [_force(record) for record in records]
    positions = [_position(record) for record in records]
    if any(value is None for value in forces) or any(value is None for value in positions):
        result["error"] = "missing finite force or T_WT translation"
        return result
    force_vectors = [[float(item) for item in value] for value in forces if value is not None]
    tool_positions = [[float(item) for item in value] for value in positions if value is not None]
    frame_ids = [_frame(record, index) for index, record in enumerate(records)]
    norms = [math.sqrt(sum(item * item for item in vector)) for vector in force_vectors]
    smoothed = _moving_average(norms, SMOOTH_RADIUS)
    peaks = _select_peaks(smoothed)
    target_frames = _target_frames(target_root)

    boundaries = [0]
    for first, second in zip(peaks, peaks[1:]):
        boundaries.append(_valley(smoothed, first, second))
    boundaries.append(len(records) - 1)

    baseline = _quantile(norms, 0.10)
    bouts = []
    for order, peak in enumerate(peaks):
        outer_start = boundaries[order]
        outer_stop = boundaries[order + 1]
        adaptive = baseline + 0.20 * max(norms[peak] - baseline, 0.0)
        active_start = peak
        while active_start > outer_start and smoothed[active_start - 1] > adaptive:
            active_start -= 1
        active_stop = peak
        while active_stop < outer_stop and smoothed[active_stop + 1] > adaptive:
            active_stop += 1
        peak_frame = frame_ids[peak]
        nearest_target, nearest_delta = _nearest(target_frames, peak_frame)
        target_in_outer = [
            value
            for value in target_frames
            if frame_ids[outer_start] <= value <= frame_ids[outer_stop]
        ]
        bouts.append(
            {
                "order": order + 1,
                "outer_start_record": outer_start,
                "outer_stop_record": outer_stop,
                "outer_start_frame": frame_ids[outer_start],
                "outer_stop_frame": frame_ids[outer_stop],
                "active_start_record": active_start,
                "active_stop_record": active_stop,
                "active_start_frame": frame_ids[active_start],
                "active_stop_frame": frame_ids[active_stop],
                "peak_record": peak,
                "peak_frame": peak_frame,
                "peak_force_norm_n": norms[peak],
                "smoothed_peak_force_norm_n": smoothed[peak],
                "left_boundary_force_norm_n": norms[outer_start],
                "right_boundary_force_norm_n": norms[outer_stop],
                "left_boundary_to_peak_ratio": norms[outer_start] / max(norms[peak], 1e-12),
                "right_boundary_to_peak_ratio": norms[outer_stop] / max(norms[peak], 1e-12),
                "tool_position_at_peak": tool_positions[peak],
                "tool_position_at_outer_start": tool_positions[outer_start],
                "tool_position_at_outer_stop": tool_positions[outer_stop],
                "approach_distance_native_units": _distance(
                    tool_positions[outer_start], tool_positions[peak]
                ),
                "retraction_distance_native_units": _distance(
                    tool_positions[peak], tool_positions[outer_stop]
                ),
                "target_mesh_frame_count_in_outer_window": len(target_in_outer),
                "nearest_target_mesh_frame_to_peak": nearest_target,
                "nearest_target_mesh_frame_delta": nearest_delta,
            }
        )

    pairwise = []
    for first in range(len(peaks)):
        for second in range(first + 1, len(peaks)):
            pairwise.append(
                {
                    "first_order": first + 1,
                    "second_order": second + 1,
                    "distance_native_units": _distance(
                        tool_positions[peaks[first]], tool_positions[peaks[second]]
                    ),
                }
            )
    inter_bout_valleys = []
    for order, boundary in enumerate(boundaries[1:-1], start=1):
        adjacent_peak = min(norms[peaks[order - 1]], norms[peaks[order]])
        inter_bout_valleys.append(
            {
                "between_orders": [order, order + 1],
                "record": boundary,
                "frame": frame_ids[boundary],
                "force_norm_n": norms[boundary],
                "ratio_to_smaller_adjacent_peak": norms[boundary] / max(adjacent_peak, 1e-12),
            }
        )

    result.update(
        {
            "record_count": len(records),
            "target_mesh_frame_count": len(target_frames),
            "target_mesh_frame_minimum": min(target_frames) if target_frames else None,
            "target_mesh_frame_maximum": max(target_frames) if target_frames else None,
            "baseline_force_q10_n": baseline,
            "selected_peak_count": len(peaks),
            "bouts": bouts,
            "inter_bout_valleys": inter_bout_valleys,
            "peak_location_pairwise_distances": pairwise,
            "minimum_peak_location_distance_native_units": (
                min(item["distance_native_units"] for item in pairwise) if pairwise else None
            ),
            "maximum_peak_location_distance_native_units": (
                max(item["distance_native_units"] for item in pairwise) if pairwise else None
            ),
            "all_bouts_have_target_mesh_near_peak_within_3_frames": all(
                bout["nearest_target_mesh_frame_delta"] is not None
                and bout["nearest_target_mesh_frame_delta"] <= 3
                for bout in bouts
            ),
            "all_inter_bout_valley_ratios_below_0_35": all(
                item["ratio_to_smaller_adjacent_peak"] < 0.35
                for item in inter_bout_valleys
            ),
            "five_bout_structure_pass": (
                len(peaks) == EXPECTED_BOUTS
                and all(
                    bout["target_mesh_frame_count_in_outer_window"] > 0
                    for bout in bouts
                )
                and all(
                    item["ratio_to_smaller_adjacent_peak"] < 0.35
                    for item in inter_bout_valleys
                )
            ),
        }
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inputs",
        type=Path,
        default=Path("/mnt/lexar4tb/datasets/pokeflex/inputs"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path("/mnt/lexar4tb/datasets/pokeflex/targets"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("refusing to overwrite output")
    input_names = {path.name for path in args.inputs.iterdir() if path.is_dir()}
    target_names = {path.name for path in args.targets.iterdir() if path.is_dir()}
    paired = sorted(input_names & target_names)
    takes = [analyze_take(args.inputs / name, args.targets / name) for name in paired]
    report = {
        "schema": SCHEMA,
        "method": {
            "expected_bouts": EXPECTED_BOUTS,
            "smooth_radius": SMOOTH_RADIUS,
            "peak_min_separation_records": PEAK_MIN_SEPARATION,
            "local_peak_radius": LOCAL_RADIUS,
            "adaptive_active_threshold": "q10 + 0.20 * (peak - q10)",
            "reset_gate": "all inter-bout valleys < 0.35 of smaller adjacent peak",
            "target_availability_gate": "at least one target mesh filename in each outer bout window",
        },
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
        "boundary": (
            "Source robot trajectories and target mesh filenames only. No target geometry "
            "or future deformation value was read."
        ),
        "paired_take_count": len(paired),
        "takes": takes,
        "aggregate": {
            "five_bout_structure_pass_count": sum(
                item.get("five_bout_structure_pass") is True for item in takes
            ),
            "target_near_every_peak_pass_count": sum(
                item.get("all_bouts_have_target_mesh_near_peak_within_3_frames") is True
                for item in takes
            ),
            "low_valley_pass_count": sum(
                item.get("all_inter_bout_valley_ratios_below_0_35") is True
                for item in takes
            ),
            "error_count": sum("error" in item for item in takes),
        },
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                **report["aggregate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
