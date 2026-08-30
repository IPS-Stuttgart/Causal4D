#!/usr/bin/env python3
"""Read-only structural audit of fragmented Deform360 trees.

The audit intentionally does not decode video, images, tactile arrays, point
clouds, or model artifacts. It reads directory entries, file sizes, small JSON
metadata, and timestamp text line counts only. The output is source-only
feasibility evidence, not an experiment result.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

SCHEMA = "bayesian-phystwin-paper.deform360-fragment-audit/v1"
OBJECT_RE = re.compile(r"^\d{3}-")
EPISODE_RE = re.compile(r"^episode_?(\d+)$", re.IGNORECASE)
ACTION_KEY_RE = re.compile(
    r"(?:^|_)(?:action|primitive|interaction|task|motion|manipulation)(?:_|$)",
    re.IGNORECASE,
)
KNOWN_ACTIONS = (
    "bend",
    "close",
    "drag",
    "fold",
    "lift",
    "open",
    "poke",
    "press",
    "roll",
    "squeeze",
    "stretch",
    "twist",
    "wave",
)
RAW_CALIBRATION = ("intrinsics.npy", "extrinsics.npy", "dist.npy")
ALIGNED_CALIBRATION = ("undistorted_intrinsics.npy", "extrinsics.npy")
MAX_JSON_BYTES = 2_000_000
MAX_TIMESTAMP_BYTES = 20_000_000


def _safe_dirs(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    try:
        return sorted(
            (child for child in path.iterdir() if child.is_dir()),
            key=lambda child: child.name,
        )
    except OSError:
        return []


def _safe_files(path: Path, suffix: str | None = None) -> list[Path]:
    if not path.is_dir():
        return []
    try:
        values = [child for child in path.iterdir() if child.is_file()]
    except OSError:
        return []
    if suffix is not None:
        values = [child for child in values if child.suffix.lower() == suffix.lower()]
    return sorted(values, key=lambda child: child.name)


def _small_json(path: Path) -> tuple[Any | None, str | None]:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return None, f"stat:{type(exc).__name__}"
    if size > MAX_JSON_BYTES:
        return None, f"too-large:{size}"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return None, f"read:{type(exc).__name__}"


def _scalar_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        candidate = value.strip()
        return [candidate] if 0 < len(candidate) <= 160 else []
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [str(value)]
    if isinstance(value, list):
        result: list[str] = []
        for item in value[:100]:
            result.extend(_scalar_strings(item))
        return result
    return []


def _metadata_summary(paths: list[Path]) -> dict[str, Any]:
    action_records: list[dict[str, str]] = []
    errors: dict[str, str] = {}
    top_level_keys: set[str] = set()
    known_actions: set[str] = set()
    episode_indices: set[int] = set()

    def visit(value: Any, key_path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                key_text = str(key)
                next_path = (*key_path, key_text)
                match = EPISODE_RE.fullmatch(key_text)
                if match:
                    episode_indices.add(int(match.group(1)))
                if ACTION_KEY_RE.search(key_text):
                    for rendered in _scalar_strings(child):
                        action_records.append(
                            {"key_path": ".".join(next_path), "value": rendered}
                        )
                        lowered = rendered.lower()
                        for action in KNOWN_ACTIONS:
                            if re.search(rf"\b{re.escape(action)}\b", lowered):
                                known_actions.add(action)
                visit(child, next_path)
        elif isinstance(value, list):
            for index, child in enumerate(value[:1000]):
                visit(child, (*key_path, str(index)))

    for path in paths:
        payload, error = _small_json(path)
        if error is not None:
            errors[str(path)] = error
            continue
        if isinstance(payload, dict):
            top_level_keys.update(map(str, payload))
        visit(payload, (path.name,))

    unique_records: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for record in action_records:
        signature = (record["key_path"], record["value"])
        if signature not in seen:
            seen.add(signature)
            unique_records.append(record)
    return {
        "json_paths": [str(path) for path in paths],
        "read_errors": errors,
        "top_level_keys": sorted(top_level_keys),
        "action_records": unique_records[:200],
        "known_action_labels": sorted(known_actions),
        "episode_indices_mentioned": sorted(episode_indices),
    }


def _timestamp_line_count(path: Path) -> int | None:
    try:
        size = path.stat().st_size
        if size > MAX_TIMESTAMP_BYTES:
            return None
        with path.open("rb") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return None


def _raw_object(
    object_dir: Path,
    root_label: str,
    minimum_cameras: int,
) -> dict[str, Any]:
    child_dirs = _safe_dirs(object_dir)
    camera_dirs: list[Path] = []
    tactile_dirs: list[Path] = []
    for child in child_dirs:
        lowered = child.name.lower()
        if "tactile" in lowered:
            if any(
                path.suffix.lower() in {".npy", ".wav", ".flac"}
                and not path.name.startswith("median_")
                for path in _safe_files(child)
            ):
                tactile_dirs.append(child)
            continue
        if _safe_files(child, ".mp4"):
            camera_dirs.append(child)

    camera_recordings = {
        path.name: _safe_files(path, ".mp4") for path in camera_dirs
    }
    tactile_recordings = {
        path.name: [
            item
            for item in _safe_files(path)
            if item.suffix.lower() in {".npy", ".wav", ".flac"}
            and not item.name.startswith("median_")
        ]
        for path in tactile_dirs
    }
    episode_count = max(
        [len(values) for values in camera_recordings.values()]
        + [len(values) for values in tactile_recordings.values()]
        + [0]
    )
    calibration_dir = object_dir / "calibration_refined"
    calibration = {
        name: (calibration_dir / name).is_file() for name in RAW_CALIBRATION
    }
    calibration_complete = all(calibration.values())

    episodes: list[dict[str, Any]] = []
    for index in range(episode_count):
        camera_files = [
            values[index]
            for values in camera_recordings.values()
            if index < len(values)
        ]
        paired_timestamps = sum(
            1 for path in camera_files if path.with_suffix(".txt").is_file()
        )
        tactile_files = [
            values[index]
            for values in tactile_recordings.values()
            if index < len(values)
        ]
        camera_count = len(camera_files)
        timestamp_fraction = paired_timestamps / camera_count if camera_count else 0.0
        structurally_processible = (
            calibration_complete
            and camera_count >= minimum_cameras
            and timestamp_fraction >= 0.9
        )
        episodes.append(
            {
                "episode_rank": index,
                "camera_recording_count": camera_count,
                "camera_timestamp_pair_count": paired_timestamps,
                "camera_timestamp_pair_fraction": timestamp_fraction,
                "tactile_stream_recording_count": len(tactile_files),
                "structurally_processible": structurally_processible,
                "strong_360_multiview": (
                    calibration_complete
                    and camera_count >= 32
                    and timestamp_fraction >= 0.9
                ),
                "sample_camera_files": [path.name for path in camera_files[:3]],
            }
        )

    metadata_paths = [
        path
        for path in [object_dir / "metadata.json", *object_dir.glob("*.json")]
        if path.is_file()
    ]
    metadata_paths = sorted(set(metadata_paths), key=lambda path: str(path))
    metadata = _metadata_summary(metadata_paths)
    return {
        "object_id": object_dir.name,
        "source_kind": "raw",
        "root_label": root_label,
        "path": str(object_dir),
        "camera_stream_count": len(camera_dirs),
        "camera_recording_count_distribution": dict(
            sorted(Counter(map(len, camera_recordings.values())).items())
        ),
        "tactile_stream_count": len(tactile_dirs),
        "tactile_recording_count_distribution": dict(
            sorted(Counter(map(len, tactile_recordings.values())).items())
        ),
        "estimated_episode_count": episode_count,
        "calibration": calibration,
        "calibration_complete": calibration_complete,
        "structurally_processible_episode_count": sum(
            episode["structurally_processible"] for episode in episodes
        ),
        "strong_360_episode_count": sum(
            episode["strong_360_multiview"] for episode in episodes
        ),
        "episodes": episodes,
        "metadata": metadata,
    }


def _episode_directories(object_dir: Path) -> list[Path]:
    return [
        child
        for child in _safe_dirs(object_dir)
        if EPISODE_RE.fullmatch(child.name)
    ]


def _geometry_inventory(episode_dir: Path) -> dict[str, Any]:
    pcd_dirs = [
        episode_dir / "pcd_clean",
        episode_dir / "point_clouds",
        episode_dir / "control_points",
        episode_dir / "observations",
    ]
    pcd_like = 0
    for directory in pcd_dirs:
        if directory.is_dir():
            pcd_like += sum(
                1
                for path in directory.rglob("*")
                if path.is_file()
                and path.suffix.lower() in {".npz", ".npy", ".h5", ".ply"}
            )
    tracking_like = sum(
        1
        for path in episode_dir.rglob("*")
        if path.is_file()
        and (
            "track" in path.name.lower()
            or "control_point" in path.name.lower()
            or "sampled_hull" in path.name.lower()
        )
        and path.suffix.lower() in {".json", ".npz", ".npy", ".h5"}
    )
    robot_candidates = [
        episode_dir / "robot" / "robot.npz",
        episode_dir / "robot" / "robot.npy",
        episode_dir / "robot.npz",
        episode_dir / "robot.npy",
    ]
    robot_present = any(path.is_file() for path in robot_candidates)
    return {
        "pcd_or_control_artifact_count": pcd_like,
        "tracking_or_hull_artifact_count": tracking_like,
        "robot_state_present": robot_present,
        "query_target_ready": pcd_like >= 2 or tracking_like >= 2,
    }


def _aligned_object(
    object_dir: Path,
    root_label: str,
    minimum_cameras: int,
) -> dict[str, Any]:
    episodes: list[dict[str, Any]] = []
    object_jsons = sorted(
        (path for path in object_dir.glob("*.json") if path.is_file()),
        key=lambda path: str(path),
    )
    object_metadata = _metadata_summary(object_jsons)

    for episode_dir in _episode_directories(object_dir):
        camera_dirs: list[Path] = []
        tactile_dirs: list[Path] = []
        for child in _safe_dirs(episode_dir):
            lowered = child.name.lower()
            if "tactile" in lowered and (child / "synced_tactile.npy").is_file():
                tactile_dirs.append(child)
                continue
            if (child / "undistorted.mp4").is_file():
                camera_dirs.append(child)
        calibration = {
            name: (episode_dir / name).is_file() for name in ALIGNED_CALIBRATION
        }
        calibration_complete = all(calibration.values())
        timestamp_counts = [
            _timestamp_line_count(child / "aligned_timestamps.txt")
            for child in camera_dirs
            if (child / "aligned_timestamps.txt").is_file()
        ]
        finite_counts = [value for value in timestamp_counts if value is not None]
        timestamp_complete = (
            len(timestamp_counts) == len(camera_dirs)
            and bool(finite_counts)
            and len(set(finite_counts)) == 1
        )
        geometry = _geometry_inventory(episode_dir)
        episode_jsons = sorted(
            (path for path in episode_dir.glob("*.json") if path.is_file()),
            key=lambda path: str(path),
        )
        metadata = _metadata_summary(episode_jsons)
        match = EPISODE_RE.fullmatch(episode_dir.name)
        assert match is not None
        episodes.append(
            {
                "episode_index": int(match.group(1)),
                "episode_name": episode_dir.name,
                "path": str(episode_dir),
                "camera_count": len(camera_dirs),
                "timestamp_complete": timestamp_complete,
                "timestamp_frame_count": finite_counts[0]
                if timestamp_complete
                else None,
                "calibration": calibration,
                "calibration_complete": calibration_complete,
                "tactile_stream_count": len(tactile_dirs),
                "structurally_processible": (
                    calibration_complete
                    and len(camera_dirs) >= minimum_cameras
                    and timestamp_complete
                ),
                "strong_360_multiview": (
                    calibration_complete
                    and len(camera_dirs) >= 32
                    and timestamp_complete
                ),
                "geometry": geometry,
                "metadata": metadata,
            }
        )

    episodes.sort(key=lambda item: item["episode_index"])
    known_actions = set(object_metadata["known_action_labels"])
    for episode in episodes:
        known_actions.update(episode["metadata"]["known_action_labels"])
    return {
        "object_id": object_dir.name,
        "source_kind": "aligned",
        "root_label": root_label,
        "path": str(object_dir),
        "episode_count": len(episodes),
        "structurally_processible_episode_count": sum(
            episode["structurally_processible"] for episode in episodes
        ),
        "strong_360_episode_count": sum(
            episode["strong_360_multiview"] for episode in episodes
        ),
        "query_target_ready_episode_count": sum(
            episode["geometry"]["query_target_ready"] for episode in episodes
        ),
        "known_action_labels": sorted(known_actions),
        "metadata": object_metadata,
        "episodes": episodes,
    }


def _object_dirs(root: Path) -> list[Path]:
    return [path for path in _safe_dirs(root) if OBJECT_RE.match(path.name)]


def _aligned_roots(path: Path) -> list[Path]:
    candidates = [path, path / "aligned", path / "processed"]
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _object_dirs(candidate):
            result.append(candidate)
    return result


def _raw_roots(path: Path) -> list[Path]:
    candidates = [path, path / "raw"]
    result: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if _object_dirs(candidate):
            result.append(candidate)
    return result


def parse_root_spec(value: str) -> tuple[str, str, Path]:
    try:
        label, remainder = value.split("=", 1)
        kind, raw_path = remainder.split(":", 1)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "root must be LABEL=raw:/path, LABEL=aligned:/path, or LABEL=mixed:/path"
        ) from exc
    label = label.strip()
    kind = kind.strip().lower()
    if not label or kind not in {"raw", "aligned", "mixed"} or not raw_path.strip():
        raise argparse.ArgumentTypeError("invalid root specification")
    return label, kind, Path(raw_path).expanduser()


def run_audit(
    *,
    server_id: str,
    root_specs: list[tuple[str, str, Path]],
    minimum_raw_cameras: int,
    minimum_aligned_cameras: int,
) -> dict[str, Any]:
    roots: list[dict[str, Any]] = []
    objects: list[dict[str, Any]] = []
    seen_sources: set[tuple[str, str, str]] = set()

    for label, kind, requested_path in root_specs:
        requested = requested_path.resolve()
        root_entry: dict[str, Any] = {
            "label": label,
            "kind": kind,
            "requested_path": str(requested),
            "exists": requested.is_dir(),
            "raw_roots": [],
            "aligned_roots": [],
        }
        raw_roots = _raw_roots(requested) if kind in {"raw", "mixed"} else []
        aligned_roots = (
            _aligned_roots(requested) if kind in {"aligned", "mixed"} else []
        )
        for raw_root in raw_roots:
            root_entry["raw_roots"].append(str(raw_root.resolve()))
            for object_dir in _object_dirs(raw_root):
                signature = ("raw", str(raw_root.resolve()), object_dir.name)
                if signature in seen_sources:
                    continue
                seen_sources.add(signature)
                objects.append(_raw_object(object_dir, label, minimum_raw_cameras))
        for aligned_root in aligned_roots:
            root_entry["aligned_roots"].append(str(aligned_root.resolve()))
            for object_dir in _object_dirs(aligned_root):
                signature = ("aligned", str(aligned_root.resolve()), object_dir.name)
                if signature in seen_sources:
                    continue
                seen_sources.add(signature)
                objects.append(
                    _aligned_object(object_dir, label, minimum_aligned_cameras)
                )
        roots.append(root_entry)

    object_ids = sorted({entry["object_id"] for entry in objects})
    return {
        "schema": SCHEMA,
        "server_id": server_id,
        "source_only": True,
        "dataset_modified": False,
        "video_payload_decoded": False,
        "array_payload_loaded": False,
        "read_scope": [
            "directory entries",
            "file existence and sizes",
            "small JSON metadata",
            "aligned timestamp text line counts",
        ],
        "thresholds": {
            "minimum_raw_camera_recordings_per_episode": minimum_raw_cameras,
            "minimum_aligned_cameras_per_episode": minimum_aligned_cameras,
            "strong_360_camera_count": 32,
            "minimum_camera_timestamp_pair_fraction": 0.9,
        },
        "roots": roots,
        "object_ids": object_ids,
        "object_count": len(object_ids),
        "source_entry_count": len(objects),
        "objects": sorted(
            objects,
            key=lambda item: (
                item["object_id"],
                item["source_kind"],
                item["root_label"],
                item["path"],
            ),
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-id", required=True)
    parser.add_argument(
        "--root",
        action="append",
        type=parse_root_spec,
        default=[],
        help="LABEL=raw:/path, LABEL=aligned:/path, or LABEL=mixed:/path",
    )
    parser.add_argument("--minimum-raw-cameras", type=int, default=16)
    parser.add_argument("--minimum-aligned-cameras", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not args.root:
        parser.error("at least one --root is required")
    if args.minimum_raw_cameras < 2 or args.minimum_aligned_cameras < 2:
        parser.error("camera thresholds must be at least two")
    payload = run_audit(
        server_id=args.server_id,
        root_specs=args.root,
        minimum_raw_cameras=args.minimum_raw_cameras,
        minimum_aligned_cameras=args.minimum_aligned_cameras,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "server_id": payload["server_id"],
                "object_count": payload["object_count"],
                "source_entry_count": payload["source_entry_count"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
