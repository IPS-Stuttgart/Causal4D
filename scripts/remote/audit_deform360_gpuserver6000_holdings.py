#!/usr/bin/env python3
"""Audit locally mounted public Deform360 holdings without decoding media."""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

_OBJECT_RE = re.compile(r"^[0-9]{3}-.+")
_EPISODE_RE = re.compile(r"^episode_([0-9]{4})$")
_CALIBRATION_FILES = ("intrinsics.npy", "extrinsics.npy", "dist.npy")


@dataclass(frozen=True)
class Qualification:
    expected_episode_count: int
    exact_raw_camera_count: int
    minimum_ten_episode_camera_count: int
    minimum_single_episode_camera_count: int
    expected_tactile_sensor_count: int
    minimum_processed_camera_count: int

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Qualification:
        fields = {
            name: int(value[name])
            for name in (
                "expected_episode_count",
                "exact_raw_camera_count",
                "minimum_ten_episode_camera_count",
                "minimum_single_episode_camera_count",
                "expected_tactile_sensor_count",
                "minimum_processed_camera_count",
            )
        }
        result = cls(**fields)
        _require(result.expected_episode_count > 0, "episode count must be positive")
        _require(
            result.exact_raw_camera_count
            >= result.minimum_ten_episode_camera_count
            >= result.minimum_single_episode_camera_count
            >= 1,
            "camera thresholds are inconsistent",
        )
        _require(
            result.expected_tactile_sensor_count > 0,
            "tactile sensor count must be positive",
        )
        _require(
            result.minimum_processed_camera_count > 0,
            "processed camera threshold must be positive",
        )
        return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_config(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "config must be a JSON object")
    _require(value.get("schema_version") == 1, "unsupported config schema")
    _require(
        value.get("protocol_id") == "causal4d-deform360-gpuserver6000-holdings-v1",
        "unexpected protocol id",
    )
    _require(value.get("runner_label") == "gpuserver6000", "unexpected runner")
    roots = value.get("roots")
    _require(isinstance(roots, list) and roots, "config roots must be nonempty")
    expected = value.get("expected_gpuserver6000_object_ids")
    _require(isinstance(expected, list) and expected, "expected IDs must be nonempty")
    _require(len(expected) == len(set(expected)), "expected IDs contain duplicates")
    exact = set(value["exact_reproduction_object_ids"])
    exploratory = set(value["exploratory_preprocessing_object_ids"])
    protected = set(value["protected_locked_cohort_object_ids"])
    _require(not exact & exploratory, "exact and exploratory objects overlap")
    _require(not exact & protected, "exact and protected objects overlap")
    _require(not exploratory & protected, "exploratory and protected objects overlap")
    _require(exact | exploratory | protected <= set(expected), "unknown object policy")
    Qualification.from_mapping(value["qualification"])
    return value


def _safe_children(path: Path) -> tuple[Path, ...]:
    try:
        return tuple(
            sorted((item for item in path.iterdir()), key=lambda item: item.name)
        )
    except OSError:
        return ()


def _looks_like_raw_object(path: Path) -> bool:
    if not path.is_dir() or _OBJECT_RE.fullmatch(path.name) is None:
        return False
    names = {child.name for child in _safe_children(path) if child.is_dir()}
    return any("_cam" in name or "_tactile" in name for name in names)


def _raw_object_directories(root: Path) -> tuple[Path, ...]:
    """Find object roots with a bounded two-level layout search."""

    candidates: set[Path] = set()
    if _looks_like_raw_object(root):
        candidates.add(root)
    containers = [root, root / "raw"]
    containers.extend(child for child in _safe_children(root) if child.is_dir())
    for container in containers:
        if not container.is_dir():
            continue
        if _looks_like_raw_object(container):
            candidates.add(container)
        for child in _safe_children(container):
            if not child.is_dir():
                continue
            if _looks_like_raw_object(child):
                candidates.add(child)
            raw_child = child / "raw"
            if _looks_like_raw_object(raw_child):
                candidates.add(raw_child)
    return tuple(sorted(candidates))


def _paired_stems(
    path: Path,
    data_suffix: str,
    *,
    excluded_prefixes: Sequence[str] = (),
) -> dict[str, Any]:
    data_paths = [
        item
        for item in _safe_children(path)
        if item.is_file()
        and item.suffix.lower() == data_suffix
        and not any(item.name.startswith(prefix) for prefix in excluded_prefixes)
    ]
    timestamp_paths = [
        item
        for item in _safe_children(path)
        if item.is_file() and item.suffix == ".txt"
    ]
    data_stems = {item.stem for item in data_paths}
    timestamp_stems = {item.stem for item in timestamp_paths}
    paired = sorted(data_stems & timestamp_stems)
    return {
        "stream": path.name,
        "data_count": len(data_paths),
        "timestamp_count": len(timestamp_paths),
        "paired_count": len(paired),
        "exact_stem_pairs": data_stems == timestamp_stems,
        "paired_stems": paired,
    }


def _raw_record(path: Path, thresholds: Qualification, role: str) -> dict[str, Any]:
    children = _safe_children(path)
    camera_dirs = tuple(
        item for item in children if item.is_dir() and "_cam" in item.name
    )
    tactile_dirs = tuple(
        item for item in children if item.is_dir() and "_tactile" in item.name
    )
    cameras = tuple(_paired_stems(item, ".mp4") for item in camera_dirs)
    tactile = tuple(
        _paired_stems(item, ".npy", excluded_prefixes=("median_",))
        for item in tactile_dirs
    )
    calibration_dir = path / "calibration_refined"
    calibration = {
        name: (calibration_dir / name).is_file() for name in _CALIBRATION_FILES
    }
    calibration_complete = all(calibration.values())
    expected = thresholds.expected_episode_count
    camera_ten = sum(item["paired_count"] >= expected for item in cameras)
    tactile_ten = sum(item["paired_count"] >= expected for item in tactile)
    camera_one = sum(item["paired_count"] >= 1 for item in cameras)
    tactile_one = sum(item["paired_count"] >= 1 for item in tactile)
    max_paired_episodes = max(
        (item["paired_count"] for item in (*cameras, *tactile)),
        default=0,
    )
    ten_episode_candidate = bool(
        calibration_complete
        and camera_ten >= thresholds.minimum_ten_episode_camera_count
        and tactile_ten >= thresholds.expected_tactile_sensor_count
    )
    exact_raw_candidate = bool(
        ten_episode_candidate
        and camera_ten >= thresholds.exact_raw_camera_count
        and len(camera_dirs) == thresholds.exact_raw_camera_count
        and len(tactile_dirs) == thresholds.expected_tactile_sensor_count
    )
    single_episode_candidate = bool(
        calibration_complete
        and camera_one >= thresholds.minimum_single_episode_camera_count
        and tactile_one >= thresholds.expected_tactile_sensor_count
    )
    if exact_raw_candidate:
        classification = "exact_ten_episode_raw_candidate"
    elif ten_episode_candidate:
        classification = "ten_episode_multiview_tactile_candidate"
    elif single_episode_candidate and max_paired_episodes == 1:
        classification = "single_episode_multiview_tactile_calibration"
    elif max_paired_episodes >= 2:
        classification = "partial_multiepisode_raw"
    else:
        classification = "incomplete_raw"
    return {
        "object_id": path.name,
        "role": role,
        "path": str(path.resolve()),
        "classification": classification,
        "camera_directory_count": len(camera_dirs),
        "camera_streams_with_ten_pairs": camera_ten,
        "camera_streams_with_one_pair": camera_one,
        "tactile_directory_count": len(tactile_dirs),
        "tactile_streams_with_ten_pairs": tactile_ten,
        "tactile_streams_with_one_pair": tactile_one,
        "maximum_paired_episode_count": max_paired_episodes,
        "calibration_files": calibration,
        "calibration_complete": calibration_complete,
        "ten_episode_candidate": ten_episode_candidate,
        "exact_raw_candidate": exact_raw_candidate,
        "single_episode_candidate": single_episode_candidate,
        "camera_streams": list(cameras),
        "tactile_streams": list(tactile),
    }


def _bounded_episode_directories(root: Path, max_depth: int = 4) -> tuple[Path, ...]:
    found: set[Path] = set()
    frontier = [(root, 0)]
    while frontier:
        directory, depth = frontier.pop()
        if depth > max_depth or not directory.is_dir():
            continue
        match = _EPISODE_RE.fullmatch(directory.name)
        if match is not None:
            found.add(directory)
            continue
        for child in reversed(_safe_children(directory)):
            if child.is_dir():
                frontier.append((child, depth + 1))
    return tuple(sorted(found))


def _object_id_for_episode(path: Path, root: Path) -> str | None:
    for ancestor in path.parents:
        if _OBJECT_RE.fullmatch(ancestor.name):
            return ancestor.name
        if ancestor == root.parent:
            break
    return None


def _processed_episode_record(path: Path) -> dict[str, Any]:
    camera_count = 0
    tactile_count = 0
    robot_available = False
    for child in _safe_children(path):
        if not child.is_dir():
            continue
        if (child / "undistorted.mp4").is_file() and (
            child / "aligned_timestamps.txt"
        ).is_file():
            camera_count += 1
        if (child / "synced_tactile.npy").is_file():
            tactile_count += 1
        if child.name == "robot" and (child / "robot.npz").is_file():
            robot_available = True
    return {
        "episode_id": path.name,
        "path": str(path.resolve()),
        "camera_count": camera_count,
        "tactile_sensor_count": tactile_count,
        "robot_available": robot_available,
    }


def _processed_records(
    roots: Iterable[tuple[str, Path]], thresholds: Qualification
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    locations: dict[str, set[str]] = defaultdict(set)
    roles: dict[str, set[str]] = defaultdict(set)
    for role, root in roots:
        for episode_dir in _bounded_episode_directories(root):
            object_id = _object_id_for_episode(episode_dir, root)
            if object_id is None:
                continue
            grouped[object_id].append(_processed_episode_record(episode_dir))
            locations[object_id].add(str(root.resolve()))
            roles[object_id].add(role)
    records: list[dict[str, Any]] = []
    for object_id, episodes in sorted(grouped.items()):
        unique = {item["path"]: item for item in episodes}
        episode_records = [unique[key] for key in sorted(unique)]
        ten_rgb = bool(
            len(episode_records) >= thresholds.expected_episode_count
            and sum(
                item["camera_count"] >= thresholds.minimum_processed_camera_count
                for item in episode_records
            )
            >= thresholds.expected_episode_count
        )
        visuotactile = sum(
            item["camera_count"] >= thresholds.minimum_processed_camera_count
            and item["tactile_sensor_count"] >= thresholds.expected_tactile_sensor_count
            for item in episode_records
        )
        if ten_rgb:
            classification = "ten_episode_processed_rgb_candidate"
        elif visuotactile >= 1:
            classification = "processed_visuotactile_calibration"
        else:
            classification = "partial_processed"
        records.append(
            {
                "object_id": object_id,
                "roles": sorted(roles[object_id]),
                "roots": sorted(locations[object_id]),
                "classification": classification,
                "episode_count": len(episode_records),
                "episodes_with_minimum_rgb_views": sum(
                    item["camera_count"] >= thresholds.minimum_processed_camera_count
                    for item in episode_records
                ),
                "episodes_with_complete_tactile": visuotactile,
                "ten_episode_processed_rgb_candidate": ten_rgb,
                "episodes": episode_records,
            }
        )
    return records


def build_report(config: Mapping[str, Any]) -> dict[str, Any]:
    thresholds = Qualification.from_mapping(config["qualification"])
    raw_records: list[dict[str, Any]] = []
    processed_roots: list[tuple[str, Path]] = []
    root_records: list[dict[str, Any]] = []
    for item in config["roots"]:
        _require(isinstance(item, Mapping), "root entry must be an object")
        role = str(item["role"])
        root = Path(str(item["path"])).expanduser().resolve()
        raw_dirs = _raw_object_directories(root)
        raw_records.extend(_raw_record(path, thresholds, role) for path in raw_dirs)
        processed_roots.append((role, root))
        root_records.append(
            {
                "role": role,
                "path": str(root),
                "exists": root.is_dir(),
                "raw_object_directories_found": [str(path) for path in raw_dirs],
            }
        )
    processed_records = _processed_records(processed_roots, thresholds)
    raw_ids = sorted({item["object_id"] for item in raw_records})
    processed_ids = sorted({item["object_id"] for item in processed_records})
    all_ids = sorted(set(raw_ids) | set(processed_ids))
    expected_ids = sorted(
        str(item) for item in config["expected_gpuserver6000_object_ids"]
    )
    exact_ids = sorted(
        {item["object_id"] for item in raw_records if item["exact_raw_candidate"]}
    )
    ten_raw_ids = sorted(
        {item["object_id"] for item in raw_records if item["ten_episode_candidate"]}
    )
    single_ids = sorted(
        {
            item["object_id"]
            for item in raw_records
            if item["classification"] == "single_episode_multiview_tactile_calibration"
        }
    )
    ten_processed_ids = sorted(
        item["object_id"]
        for item in processed_records
        if item["ten_episode_processed_rgb_candidate"]
    )
    processing_order = [
        object_id
        for object_id in (
            *config["exact_reproduction_object_ids"],
            *config["exploratory_preprocessing_object_ids"],
        )
        if object_id in ten_raw_ids
    ]
    raw_by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in raw_records:
        raw_by_id[record["object_id"]].append(record)
    processing_candidates = [
        {
            "object_id": object_id,
            "raw_path": sorted(raw_by_id[object_id], key=lambda item: item["path"])[0][
                "path"
            ],
            "mode": (
                "exact_completed_case_reproduction"
                if object_id in config["exact_reproduction_object_ids"]
                else "retrospective_public_preprocessing"
            ),
        }
        for object_id in processing_order
    ]
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DDeform360Gpuserver6000HoldingsAudit",
        "protocol_id": config["protocol_id"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset_repository": config["dataset_repository"],
        "dataset_revision": config["dataset_revision"],
        "runner_label": config["runner_label"],
        "roots": root_records,
        "raw_records": sorted(
            raw_records, key=lambda item: (item["object_id"], item["path"])
        ),
        "processed_records": processed_records,
        "summary": {
            "expected_gpuserver6000_object_count": len(expected_ids),
            "discovered_raw_object_count": len(raw_ids),
            "discovered_processed_object_count": len(processed_ids),
            "discovered_unique_object_count": len(all_ids),
            "expected_object_ids": expected_ids,
            "discovered_object_ids": all_ids,
            "missing_expected_object_ids": sorted(set(expected_ids) - set(all_ids)),
            "unexpected_object_ids": sorted(set(all_ids) - set(expected_ids)),
            "exact_ten_episode_raw_object_ids": exact_ids,
            "ten_episode_raw_candidate_ids": ten_raw_ids,
            "single_episode_calibration_ids": single_ids,
            "ten_episode_processed_rgb_candidate_ids": ten_processed_ids,
            "processing_candidates": processing_candidates,
            "uniform_26_object_benchmark_ready": len(ten_raw_ids) == len(expected_ids),
            "exact_001_rope_raw_candidate": "001-rope" in exact_ids,
        },
        "interpretation": {
            "enough_for_exact_001_reproduction": "001-rope" in exact_ids,
            "enough_for_multi_object_preprocessing": len(processing_candidates) >= 2,
            "enough_for_uniform_26_object_benchmark": (
                len(ten_raw_ids) == len(expected_ids)
            ),
            "qualification_is_a_new_paper_result": False,
        },
        "known_external_processed_only_object_ids": config[
            "known_external_processed_only_object_ids"
        ],
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "metadata_only": True,
            "file_payloads_read": False,
            "media_decoded": False,
            "raw_sources_modified": False,
            "symlinks_followed": False,
            "protected_locked_targets_opened": False,
            "new_paper_claim_authorized": False,
        },
    }


def main() -> None:
    args = _parse_args()
    config = load_config(args.config.resolve())
    report = build_report(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
