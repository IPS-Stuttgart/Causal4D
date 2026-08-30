#!/usr/bin/env python3
"""Inventory the readable 13-pair PokeFlex staging area on gpuserver6000.

The input side is source-authorized: this scanner may read robot records and
summarize candidate contact episodes. The target side remains outcome-closed:
only paths, byte sizes, and mesh/frame identifiers are inspected; no target mesh,
point-cloud, image, or numerical trajectory payload is parsed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

SCHEMA = "causal4d.pokeflex-public13-source-inventory-v1"
FRAME_PATTERN = re.compile(r"(\d{3,})")
MAX_FILES_PER_TAKE = 250_000
MAX_TEXT_BYTES = 1_000_000
FORCE_THRESHOLDS_N = (0.5, 1.0, 2.0, 5.0, 10.0)
MIN_EVENT_GAP_FRAMES = 8


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def extension(path: str) -> str:
    lowered = path.lower()
    for suffix in (".tar.gz", ".tar.xz", ".json.gz"):
        if lowered.endswith(suffix):
            return suffix
    return PurePosixPath(lowered).suffix


def frame_id(name: str) -> int | None:
    matches = FRAME_PATTERN.findall(PurePosixPath(name).stem)
    if not matches:
        return None
    return int(matches[-1])


def summarize_tree(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": str(root),
        "exists": root.is_dir(),
        "resolved": str(root.resolve(strict=False)),
        "files": 0,
        "bytes": 0,
        "extensions": {},
        "depth_components": {},
        "sample_paths": [],
        "frame_ranges": {},
        "truncated": False,
        "errors": [],
    }
    if not root.is_dir():
        return result
    ext = Counter()
    depth = defaultdict(Counter)
    frames: dict[str, list[int]] = defaultdict(list)
    samples: list[str] = []
    count = 0
    total_bytes = 0
    for current, _, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        for filename in filenames:
            path = current_path / filename
            count += 1
            if count > MAX_FILES_PER_TAKE:
                result["truncated"] = True
                break
            try:
                info = path.stat()
            except OSError as exc:
                result["errors"].append({"path": str(path), "error": repr(exc)})
                continue
            relative = path.relative_to(root).as_posix()
            total_bytes += info.st_size
            ext[extension(relative)] += 1
            parts = PurePosixPath(relative).parts
            for level in range(1, min(len(parts), 4)):
                depth[level]["/".join(parts[:level])] += 1
            if len(samples) < 120:
                samples.append(relative)
            identifier = frame_id(relative)
            if identifier is not None:
                key = "/".join(parts[:-1]) or "."
                frames[key].append(identifier)
        if result["truncated"]:
            break
    result.update(
        {
            "files": min(count, MAX_FILES_PER_TAKE),
            "bytes": total_bytes,
            "extensions": dict(ext.most_common(80)),
            "depth_components": {
                str(level): dict(counter.most_common(150))
                for level, counter in sorted(depth.items())
            },
            "sample_paths": samples,
            "frame_ranges": {
                key: {
                    "count": len(values),
                    "minimum": min(values),
                    "maximum": max(values),
                    "unique_count": len(set(values)),
                }
                for key, values in sorted(frames.items())
                if values
            },
        }
    )
    return result


def finite_vector(value: object, *, minimum_length: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < minimum_length:
        return None
    output: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        number = float(item)
        if not math.isfinite(number):
            return None
        output.append(number)
    return output


def force_vector(record: dict[str, object]) -> list[float] | None:
    for key in ("forces", "force", "wrench", "force_torque", "ft"):
        vector = finite_vector(record.get(key), minimum_length=3)
        if vector is not None:
            return vector[:3]
    return None


def transform_translation(record: dict[str, object]) -> list[float] | None:
    for key in ("T_WT", "tool_transform", "eef_pose", "end_effector_pose"):
        value = record.get(key)
        if not isinstance(value, list) or len(value) < 3:
            continue
        if all(isinstance(row, list) and len(row) >= 4 for row in value[:3]):
            translation: list[float] = []
            for row in value[:3]:
                item = row[3]
                if isinstance(item, bool) or not isinstance(item, (int, float)):
                    translation = []
                    break
                number = float(item)
                if not math.isfinite(number):
                    translation = []
                    break
                translation.append(number)
            if translation:
                return translation
    return None


def event_runs(mask: list[bool]) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    last_true: int | None = None
    for index, active in enumerate(mask):
        if active:
            if start is None:
                start = index
            last_true = index
            continue
        if start is not None and last_true is not None:
            runs.append((start, last_true))
        start = None
        last_true = None
    if start is not None and last_true is not None:
        runs.append((start, last_true))
    if not runs:
        return []
    merged = [runs[0]]
    for start, stop in runs[1:]:
        previous_start, previous_stop = merged[-1]
        if start - previous_stop <= MIN_EVENT_GAP_FRAMES:
            merged[-1] = (previous_start, stop)
        else:
            merged.append((start, stop))
    return merged


def quantile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    location = probability * (len(ordered) - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    if lower == upper:
        return ordered[lower]
    weight = location - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_robot_json(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.stat().st_size > MAX_TEXT_BYTES:
        report["error"] = "robot_data.json exceeds bounded source parser size"
        return report
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        report["error"] = repr(exc)
        return report
    if not isinstance(payload, list):
        report["payload_type"] = type(payload).__name__
        if isinstance(payload, dict):
            report["top_level_keys"] = sorted(map(str, payload))[:100]
        return report
    report["record_count"] = len(payload)
    records = [item for item in payload if isinstance(item, dict)]
    report["mapping_record_count"] = len(records)
    report["keys"] = sorted({str(key) for item in records for key in item})

    frame_values: list[int] = []
    force_norms: list[float] = []
    positions: list[list[float]] = []
    commands_present = Counter()
    timestamps_present = Counter()
    explicit_contact_present = Counter()
    for record in records:
        frame = record.get("frame")
        if isinstance(frame, int) and not isinstance(frame, bool) and frame >= 0:
            frame_values.append(frame)
        elif isinstance(frame, str) and frame.isascii() and frame.isdigit():
            frame_values.append(int(frame))
        vector = force_vector(record)
        if vector is not None:
            force_norms.append(math.sqrt(sum(component * component for component in vector)))
        translation = transform_translation(record)
        if translation is not None:
            positions.append(translation)
        for key in record:
            lowered = str(key).lower()
            if any(token in lowered for token in ("command", "desired", "target_action")):
                commands_present[str(key)] += 1
            if any(token in lowered for token in ("time", "stamp")):
                timestamps_present[str(key)] += 1
            if "contact" in lowered:
                explicit_contact_present[str(key)] += 1

    report["frame_ids"] = {
        "count": len(frame_values),
        "minimum": min(frame_values) if frame_values else None,
        "maximum": max(frame_values) if frame_values else None,
        "unique_count": len(set(frame_values)),
        "monotone_nondecreasing": all(
            first <= second for first, second in zip(frame_values, frame_values[1:])
        ),
    }
    report["command_like_keys"] = dict(commands_present)
    report["timestamp_like_keys"] = dict(timestamps_present)
    report["explicit_contact_keys"] = dict(explicit_contact_present)
    if force_norms:
        report["force_norm_n"] = {
            "count": len(force_norms),
            "minimum": min(force_norms),
            "q50": quantile(force_norms, 0.50),
            "q90": quantile(force_norms, 0.90),
            "q95": quantile(force_norms, 0.95),
            "q99": quantile(force_norms, 0.99),
            "maximum": max(force_norms),
        }
        report["candidate_contact_runs"] = {
            str(threshold): [
                {"start_record": start, "stop_record": stop, "length": stop - start + 1}
                for start, stop in event_runs([value > threshold for value in force_norms])
            ]
            for threshold in FORCE_THRESHOLDS_N
        }
    else:
        report["force_norm_n"] = None
        report["candidate_contact_runs"] = {}
    if positions:
        path_length = 0.0
        maximum_step = 0.0
        for first, second in zip(positions, positions[1:]):
            distance = math.sqrt(sum((b - a) ** 2 for a, b in zip(first, second)))
            path_length += distance
            maximum_step = max(maximum_step, distance)
        report["tool_translation"] = {
            "count": len(positions),
            "coordinate_minimum": [min(row[index] for row in positions) for index in range(3)],
            "coordinate_maximum": [max(row[index] for row in positions) for index in range(3)],
            "path_length_native_units": path_length,
            "maximum_step_native_units": maximum_step,
        }
    else:
        report["tool_translation"] = None
    return report


def find_robot_files(root: Path) -> list[Path]:
    matches = []
    if not root.is_dir():
        return matches
    for name in ("robot_data.json", "robot.json"):
        matches.extend(root.rglob(name))
    return sorted(set(matches))


def paired_names(inputs: Path, targets: Path) -> tuple[list[str], list[str], list[str]]:
    def children(root: Path) -> set[str]:
        if not root.is_dir():
            return set()
        return {path.name for path in root.iterdir() if path.is_dir()}

    input_names = children(inputs)
    target_names = children(targets)
    return sorted(input_names), sorted(target_names), sorted(input_names & target_names)


def summarize_take(inputs: Path, targets: Path, name: str) -> dict[str, Any]:
    input_root = inputs / name
    target_root = targets / name
    robot_files = find_robot_files(input_root)
    object_name, separator, take = name.rpartition("_T")
    return {
        "name": name,
        "object_id": object_name if separator else name,
        "take_id": f"T{take}" if separator else None,
        "input": summarize_tree(input_root),
        "target_structure_only": summarize_tree(target_root),
        "source_robot_records": [summarize_robot_json(path) for path in robot_files],
        "robot_file_count": len(robot_files),
        "target_numerical_payload_read": False,
    }


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

    input_names, target_names, paired = paired_names(args.inputs, args.targets)
    report = {
        "schema": SCHEMA,
        "runtime": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "uid": os.getuid(),
            "gid": os.getgid(),
        },
        "roots": {
            "inputs": str(args.inputs),
            "inputs_resolved": str(args.inputs.resolve(strict=False)),
            "targets": str(args.targets),
            "targets_resolved": str(args.targets.resolve(strict=False)),
        },
        "boundary": (
            "Input robot records are source-authorized. Target content remains closed: "
            "only filenames, sizes, and frame identifiers are inventoried."
        ),
        "input_names": input_names,
        "target_names": target_names,
        "paired_names": paired,
        "unpaired_inputs": sorted(set(input_names) - set(target_names)),
        "unpaired_targets": sorted(set(target_names) - set(input_names)),
        "takes": [summarize_take(args.inputs, args.targets, name) for name in paired],
    }
    encoded = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "paired_takes": len(paired),
                "sha256": hashlib.sha256(encoded.encode()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
