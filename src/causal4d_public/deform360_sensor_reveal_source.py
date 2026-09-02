"""Source-only Deform360 sensor-reveal capability and routing audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tarfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias, cast

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[Any]
FloatArray: TypeAlias = npt.NDArray[np.float64]

AUDIT_SCHEMA_VERSION = 1
AUDIT_KIND = "Deform360SensorRevealSourceAudit"
CONFIG_KIND = "Deform360SensorRevealSourceAuditConfig"
_FRAME_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any], *, digest_field: str) -> str:
    unsigned = dict(payload)
    unsigned.pop(digest_field, None)
    return hashlib.sha256(_canonical_bytes(unsigned)).hexdigest()


def _sampled_file_sha256(path: Path, sample_bytes: int = 1024 * 1024) -> str:
    """Hash file size plus bounded first/last samples; not a full-file digest."""

    size = path.stat().st_size
    digest = hashlib.sha256()
    digest.update(str(size).encode("ascii"))
    with path.open("rb") as stream:
        digest.update(stream.read(sample_bytes))
        if size > sample_bytes:
            stream.seek(max(size - sample_bytes, 0))
            digest.update(stream.read(sample_bytes))
    return digest.hexdigest()


def _stat_file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.relative_to(root).as_posix(),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        **_stat_file_record(path, root),
        "sampled_sha256": _sampled_file_sha256(path),
        "sampled_sha256_semantics": "sha256(size || first_1MiB || last_1MiB)",
    }


def _load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    loaded: object = json.loads(config_path.read_text(encoding="utf-8"))
    _require(isinstance(loaded, dict), "config must be a JSON object")
    payload = cast(dict[str, Any], loaded)
    _require(
        payload.get("schema_version") == AUDIT_SCHEMA_VERSION, "config schema changed"
    )
    _require(payload.get("artifact_kind") == CONFIG_KIND, "config kind changed")
    _require(
        payload.get("config_sha256")
        == _payload_sha256(payload, digest_field="config_sha256"),
        "config checksum mismatch",
    )
    source_value = payload.get("source_episode_ids")
    if not isinstance(source_value, list):
        raise ValueError("source_episode_ids must be a list")
    source_ids = cast(list[object], source_value)
    _require(
        len(source_ids) >= 3
        and all(type(value) is int and value >= 0 for value in source_ids)
        and len(set(source_ids)) == len(source_ids),
        "source_episode_ids must contain at least three unique nonnegative integers",
    )
    forbidden_value = payload.get("forbidden_episode_ids")
    if not isinstance(forbidden_value, list):
        raise ValueError("forbidden_episode_ids must be a list")
    forbidden_ids = cast(list[object], forbidden_value)
    _require(
        all(type(value) is int and value >= 0 for value in forbidden_ids)
        and len(set(forbidden_ids)) == len(forbidden_ids),
        "forbidden_episode_ids must be unique nonnegative integers",
    )
    _require(
        not set(source_ids) & set(forbidden_ids),
        "source and forbidden episodes overlap",
    )
    boundary_value = payload.get("information_boundary")
    if not isinstance(boundary_value, Mapping):
        raise ValueError("information_boundary must be a mapping")
    boundary = cast(Mapping[str, Any], boundary_value)
    _require(
        boundary.get("source_only") is True
        and boundary.get("forbidden_episode_payloads_read") is False
        and boundary.get("held_target_payloads_read") is False
        and boundary.get("dataset_modified") is False
        and boundary.get("paper_claim_authorized") is False,
        "config opens a forbidden information or claim boundary",
    )
    for key in (
        "minimum_frames",
        "reset_count",
        "prefix_window_frames",
        "future_horizon_frames",
        "minimum_common_tactile_groups",
    ):
        _require(
            type(payload.get(key)) is int and payload[key] >= 1, f"{key} is invalid"
        )
    costs_value = payload.get("sensor_costs")
    if not isinstance(costs_value, Mapping):
        raise ValueError("sensor_costs are missing")
    costs = cast(Mapping[str, Any], costs_value)
    for key in ("robot_opening", "robot_translation", "tactile"):
        value = costs.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"sensor cost {key} is invalid")
        numeric = float(value)
        _require(
            math.isfinite(numeric) and numeric >= 0.0,
            f"sensor cost {key} is invalid",
        )
    return payload


def _find_episode_directory(
    root: Path,
    object_id: str,
    episode_id: int,
) -> Path | None:
    object_root = root / object_id
    candidates = (
        object_root / f"episode_{episode_id}",
        object_root / f"episode_{episode_id:04d}",
    )
    existing = [candidate for candidate in candidates if candidate.is_dir()]
    _require(
        len(existing) <= 1,
        f"episode {episode_id} resolves ambiguously below {root}",
    )
    return existing[0] if existing else None


def _episode_directory(root: Path, object_id: str, episode_id: int) -> Path:
    episode = _find_episode_directory(root, object_id, episode_id)
    if episode is None:
        raise ValueError(f"episode {episode_id} is missing below {root}")
    return episode


def _frame_id(name: str) -> int | None:
    match = _FRAME_PATTERN.search(Path(name).stem)
    return None if match is None else int(match.group(1))


def _point_cloud_inventory(episode: Path, root: Path) -> dict[str, Any]:
    """Inventory point-cloud carriers without reading any NPZ member payload."""

    tar_path = episode / "pcd_clean.tar"
    directory = episode / "pcd_clean"
    if tar_path.is_file():
        try:
            with tarfile.open(tar_path, mode="r:") as archive:
                members = [
                    member
                    for member in archive.getmembers()
                    if member.isfile() and member.name.lower().endswith(".npz")
                ]
        except tarfile.ReadError:
            return {
                "available": False,
                "container": "tar",
                "reason": "pcd_clean.tar is not an uncompressed tar archive",
                "file": _stat_file_record(tar_path, root),
                "member_header_metadata_read": False,
                "member_payloads_read": False,
            }
        indexed = [(_frame_id(member.name), member) for member in members]
        indexed.sort(
            key=lambda item: (
                item[0] is None,
                item[0] if item[0] is not None else 0,
                item[1].name,
            )
        )
        names = [member.name for _, member in indexed]
        frame_ids = [frame_id for frame_id, _ in indexed if frame_id is not None]
        descriptor = [
            {
                "name": member.name,
                "size_bytes": member.size,
                "offset": member.offset,
                "offset_data": member.offset_data,
            }
            for _, member in indexed
        ]
        return {
            "available": bool(names),
            "container": "uncompressed-tar",
            "frame_count": len(names),
            "frame_id_count": len(frame_ids),
            "first_member": names[0] if names else None,
            "last_member": names[-1] if names else None,
            "strictly_increasing_frame_ids": bool(frame_ids)
            and len(frame_ids) == len(names)
            and len(set(frame_ids)) == len(frame_ids)
            and all(right > left for left, right in zip(frame_ids, frame_ids[1:])),
            "member_header_descriptor_sha256": hashlib.sha256(
                _canonical_bytes(descriptor)
            ).hexdigest(),
            "file": _stat_file_record(tar_path, root),
            "member_header_metadata_read": True,
            "member_payloads_read": False,
        }
    if directory.is_dir():
        indexed_paths = [
            (_frame_id(path.name), path) for path in directory.glob("*.npz")
        ]
        indexed_paths.sort(
            key=lambda item: (
                item[0] is None,
                item[0] if item[0] is not None else 0,
                item[1].name,
            )
        )
        paths = [path for _, path in indexed_paths]
        frame_ids = [frame_id for frame_id, _ in indexed_paths if frame_id is not None]
        descriptor = [
            {
                "name": path.name,
                "size_bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for path in paths
        ]
        return {
            "available": bool(paths),
            "container": "directory",
            "frame_count": len(paths),
            "frame_id_count": len(frame_ids),
            "first_member": paths[0].name if paths else None,
            "last_member": paths[-1].name if paths else None,
            "strictly_increasing_frame_ids": bool(frame_ids)
            and len(frame_ids) == len(paths)
            and len(set(frame_ids)) == len(frame_ids)
            and all(right > left for left, right in zip(frame_ids, frame_ids[1:])),
            "member_metadata_descriptor_sha256": hashlib.sha256(
                _canonical_bytes(descriptor)
            ).hexdigest(),
            "directory": directory.relative_to(root).as_posix(),
            "member_header_metadata_read": True,
            "member_payloads_read": False,
        }
    return {
        "available": False,
        "frame_count": 0,
        "member_header_metadata_read": False,
        "member_payloads_read": False,
    }


def _leading_frame_count(value: Array) -> int | None:
    return int(value.shape[0]) if value.ndim >= 1 and value.shape[0] >= 1 else None


def _robot_inventory(
    episode: Path, root: Path
) -> tuple[dict[str, Any], dict[str, Array]]:
    candidates = (episode / "robot" / "robot.npz", episode / "robot.npz")
    paths = [path for path in candidates if path.is_file()]
    if not paths:
        paths = sorted(episode.glob("**/robot.npz"))
    if len(paths) != 1:
        return {
            "available": False,
            "reason": f"expected one robot.npz, found {len(paths)}",
        }, {}
    path = paths[0]
    arrays: dict[str, Array] = {}
    with np.load(path, allow_pickle=False) as payload:
        keys = sorted(payload.files)
        for key in keys:
            arrays[key] = np.asarray(payload[key])
    required_frame_counts = {
        count
        for key in ("openings", "T_worlds")
        if key in arrays
        if (count := _leading_frame_count(arrays[key])) is not None
    }
    all_frame_counts = {
        count
        for value in arrays.values()
        if (count := _leading_frame_count(value)) is not None
    }
    summary = {
        "available": True,
        "file": _file_record(path, root),
        "keys": keys,
        "arrays": {
            key: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for key, value in arrays.items()
        },
        "all_leading_frame_counts": sorted(all_frame_counts),
        "required_leading_frame_counts": sorted(required_frame_counts),
        "required_frame_counts_consistent": len(required_frame_counts) == 1,
        "frame_count": (
            next(iter(required_frame_counts))
            if len(required_frame_counts) == 1
            else None
        ),
        "required_keys_available": all(
            key in arrays for key in ("openings", "T_worlds")
        ),
        "values_read": True,
    }
    return summary, arrays


def _per_frame_mean_absolute(value: Array, chunk_size: int = 64) -> FloatArray:
    _require(value.ndim >= 1 and value.shape[0] >= 1, "sensor array lacks a frame axis")
    result: FloatArray = np.empty(value.shape[0], dtype=np.float64)
    for start in range(0, value.shape[0], chunk_size):
        stop = min(start + chunk_size, value.shape[0])
        chunk = np.asarray(value[start:stop], dtype=np.float64)
        _require(
            bool(np.all(np.isfinite(chunk))), "sensor array contains nonfinite values"
        )
        result[start:stop] = np.mean(np.abs(chunk).reshape(stop - start, -1), axis=1)
    return result


def _tactile_inventory(
    episode_roots: Sequence[tuple[Path, Path]],
) -> tuple[dict[str, Any], dict[str, FloatArray]]:
    paths: list[tuple[Path, Path]] = []
    seen_paths: set[Path] = set()
    for episode, root in episode_roots:
        for path in sorted(episode.glob("**/synced_tactile.npy")):
            resolved = path.resolve()
            if resolved not in seen_paths:
                paths.append((path, root))
                seen_paths.add(resolved)
    summaries: list[dict[str, Any]] = []
    energies: dict[str, FloatArray] = {}
    for path, root in paths:
        episode = next(
            candidate_episode
            for candidate_episode, candidate_root in episode_roots
            if candidate_root == root and path.is_relative_to(candidate_episode)
        )
        relative = path.relative_to(episode)
        sensor_name = relative.parent.as_posix()
        _require(
            sensor_name not in energies,
            f"duplicate tactile sensor {sensor_name!r} across source roots",
        )
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        energy = _per_frame_mean_absolute(value)
        energies[sensor_name] = energy
        summaries.append(
            {
                "sensor_name": sensor_name,
                "source_root": str(root),
                "file": _file_record(path, root),
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "frame_count": int(value.shape[0]),
                "frame_energy": {
                    "minimum": float(np.min(energy)),
                    "median": float(np.median(energy)),
                    "p95": float(np.quantile(energy, 0.95)),
                    "maximum": float(np.max(energy)),
                },
                "values_read": True,
            }
        )
    summaries.sort(key=lambda row: row["sensor_name"])
    return {
        "available": bool(summaries),
        "sensor_count": len(summaries),
        "sensor_names": [row["sensor_name"] for row in summaries],
        "sensors": summaries,
    }, energies


def _robot_feature_series(arrays: Mapping[str, Array]) -> dict[str, FloatArray]:
    features: dict[str, FloatArray] = {}
    openings = arrays.get("openings")
    if openings is not None and openings.ndim >= 1:
        reshaped = np.asarray(openings, dtype=np.float64).reshape(openings.shape[0], -1)
        _require(
            bool(np.all(np.isfinite(reshaped))),
            "robot openings contain nonfinite values",
        )
        features["robot-opening"] = np.mean(reshaped, axis=1)
    worlds = arrays.get("T_worlds")
    if worlds is not None and worlds.ndim >= 3 and worlds.shape[-2:] == (4, 4):
        translations = np.asarray(worlds[..., :3, 3], dtype=np.float64)
        translations = translations.reshape(translations.shape[0], -1, 3)
        _require(
            bool(np.all(np.isfinite(translations))),
            "robot transforms contain nonfinite values",
        )
        step = np.zeros(translations.shape[0], dtype=np.float64)
        if translations.shape[0] > 1:
            step[1:] = np.mean(
                np.linalg.norm(np.diff(translations, axis=0), axis=-1), axis=1
            )
        features["robot-translation"] = step
    return features


def _reset_positions(
    frame_count: int,
    *,
    reset_count: int,
    prefix_window: int,
    future_horizon: int,
) -> tuple[int, ...]:
    earliest = prefix_window
    latest = frame_count - future_horizon - 1
    _require(latest >= earliest, "episode is too short for the registered windows")
    if reset_count == 1:
        return (earliest,)
    raw = np.linspace(earliest, latest, reset_count)
    rounded = tuple(int(round(value)) for value in raw)
    _require(len(set(rounded)) == reset_count, "reset positions are not unique")
    return rounded


def _window_sum(series: FloatArray, end: int, width: int) -> float:
    start = max(end - width + 1, 0)
    return float(np.sum(series[start : end + 1]))


def _window_range(series: FloatArray, end: int, width: int) -> float:
    start = max(end - width + 1, 0)
    window = series[start : end + 1]
    return float(np.max(window) - np.min(window))


def _quantile_codebook(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    _require(
        array.ndim == 1 and array.size >= 3, "codebook needs at least three values"
    )
    lower = float(np.quantile(array, 1.0 / 3.0))
    upper = float(np.quantile(array, 2.0 / 3.0))
    if upper <= lower:
        spread = max(float(np.ptp(array)), 1.0) * 1e-12
        lower -= spread
        upper += spread
    return lower, upper


def _bin(value: float, thresholds: tuple[float, float]) -> int:
    return 0 if value <= thresholds[0] else 1 if value <= thresholds[1] else 2


def _entropy(labels: Sequence[int]) -> float:
    values = np.asarray(labels, dtype=np.int64)
    if values.size == 0:
        return 0.0
    counts = np.bincount(values)
    probabilities = counts[counts > 0] / values.size
    return float(-np.sum(probabilities * np.log(probabilities)))


def _mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    _require(len(x) == len(y), "mutual-information vectors differ in length")
    if not x:
        return 0.0
    joint: dict[tuple[int, int], int] = defaultdict(int)
    x_count: dict[int, int] = defaultdict(int)
    y_count: dict[int, int] = defaultdict(int)
    for left, right in zip(x, y, strict=True):
        joint[(left, right)] += 1
        x_count[left] += 1
        y_count[right] += 1
    total = float(len(x))
    result = 0.0
    for (left, right), count in joint.items():
        result += (count / total) * math.log(
            (count * total) / (x_count[left] * y_count[right])
        )
    return max(result, 0.0)


def _lookup_accuracy(keys: Sequence[tuple[int, ...]], labels: Sequence[int]) -> float:
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for key, label in zip(keys, labels, strict=True):
        groups[key].append(label)
    correct = 0
    for values in groups.values():
        counts = np.bincount(np.asarray(values, dtype=np.int64))
        correct += int(np.max(counts))
    return correct / len(labels) if labels else 0.0


def _conditional_mode_predictor(
    keys: Sequence[tuple[int, ...]],
    labels: Sequence[int],
) -> tuple[dict[tuple[int, ...], int], int]:
    _require(
        len(keys) == len(labels) and bool(labels), "predictor training data are invalid"
    )
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for key, label in zip(keys, labels, strict=True):
        groups[key].append(label)
    mapping = {
        key: int(np.argmax(np.bincount(np.asarray(values, dtype=np.int64))))
        for key, values in groups.items()
    }
    default = int(np.argmax(np.bincount(np.asarray(labels, dtype=np.int64))))
    return mapping, default


def _leave_one_episode_out_accuracy(
    rows: Sequence[Mapping[str, Any]],
    sensors: tuple[str, ...],
) -> dict[str, Any]:
    episode_ids = sorted({int(row["episode_id"]) for row in rows})
    _require(len(episode_ids) >= 3, "grouped routing needs at least three episodes")
    predictions: list[int] = []
    targets: list[int] = []
    baseline_predictions: list[int] = []
    episode_records: list[dict[str, Any]] = []
    for held_episode in episode_ids:
        training = [row for row in rows if int(row["episode_id"]) != held_episode]
        held = [row for row in rows if int(row["episode_id"]) == held_episode]
        _require(bool(training) and bool(held), "grouped routing fold is empty")
        target_thresholds = _quantile_codebook(
            [float(row["future_robot_translation_path_m"]) for row in training]
        )
        training_labels = [
            _bin(float(row["future_robot_translation_path_m"]), target_thresholds)
            for row in training
        ]
        held_labels = [
            _bin(float(row["future_robot_translation_path_m"]), target_thresholds)
            for row in held
        ]
        sensor_thresholds = {
            sensor: _quantile_codebook(
                [float(row["features"][sensor]) for row in training]
            )
            for sensor in sensors
        }
        training_keys = [
            tuple(
                _bin(float(row["features"][sensor]), sensor_thresholds[sensor])
                for sensor in sensors
            )
            for row in training
        ]
        held_keys = [
            tuple(
                _bin(float(row["features"][sensor]), sensor_thresholds[sensor])
                for sensor in sensors
            )
            for row in held
        ]
        mapping, default = _conditional_mode_predictor(training_keys, training_labels)
        fold_predictions = [mapping.get(key, default) for key in held_keys]
        predictions.extend(fold_predictions)
        targets.extend(held_labels)
        baseline_predictions.extend([default] * len(held_labels))
        episode_records.append(
            {
                "held_episode_id": held_episode,
                "case_count": len(held),
                "accuracy": float(
                    np.mean(np.asarray(fold_predictions) == np.asarray(held_labels))
                ),
                "training_majority_baseline_accuracy": float(
                    np.mean(np.asarray(held_labels) == default)
                ),
                "unseen_outcome_key_count": sum(
                    key not in mapping for key in held_keys
                ),
            }
        )
    return {
        "accuracy": float(np.mean(np.asarray(predictions) == np.asarray(targets))),
        "training_majority_baseline_accuracy": float(
            np.mean(np.asarray(baseline_predictions) == np.asarray(targets))
        ),
        "episode_count": len(episode_ids),
        "case_count": len(targets),
        "episode_records": episode_records,
    }


def _routing_diagnostic(
    rows: list[dict[str, Any]],
    sensor_costs: Mapping[str, float],
) -> dict[str, Any]:
    sensor_names = sorted(rows[0]["features"]) if rows else []
    target_values = [float(row["future_robot_translation_path_m"]) for row in rows]
    target_thresholds = _quantile_codebook(target_values)
    labels = [_bin(value, target_thresholds) for value in target_values]
    codebooks = {
        sensor: _quantile_codebook([float(row["features"][sensor]) for row in rows])
        for sensor in sensor_names
    }
    outcomes = {
        sensor: [
            _bin(float(row["features"][sensor]), codebooks[sensor]) for row in rows
        ]
        for sensor in sensor_names
    }
    single: list[dict[str, Any]] = []
    for sensor in sensor_names:
        accuracy = _lookup_accuracy([(value,) for value in outcomes[sensor]], labels)
        grouped = _leave_one_episode_out_accuracy(rows, (sensor,))
        single.append(
            {
                "sensor_name": sensor,
                "cost": float(sensor_costs[sensor]),
                "mutual_information_nats": _mutual_information(
                    outcomes[sensor], labels
                ),
                "same_support_lookup_accuracy": accuracy,
                "leave_one_episode_out": grouped,
                "outcome_counts": np.bincount(
                    np.asarray(outcomes[sensor], dtype=np.int64), minlength=3
                ).tolist(),
            }
        )
    pairs: list[dict[str, Any]] = []
    for left_index, left in enumerate(sensor_names):
        for right in sensor_names[left_index + 1 :]:
            keys = list(zip(outcomes[left], outcomes[right], strict=True))
            grouped = _leave_one_episode_out_accuracy(rows, (left, right))
            pairs.append(
                {
                    "sensor_names": [left, right],
                    "cost": float(sensor_costs[left] + sensor_costs[right]),
                    "same_support_lookup_accuracy": _lookup_accuracy(keys, labels),
                    "leave_one_episode_out": grouped,
                }
            )
    pairs.sort(
        key=lambda row: (
            -row["leave_one_episode_out"]["accuracy"],
            row["cost"],
            row["sensor_names"],
        )
    )
    baseline = float(
        np.max(np.bincount(np.asarray(labels, dtype=np.int64))) / len(labels)
    )
    single.sort(
        key=lambda row: (
            -row["leave_one_episode_out"]["accuracy"],
            row["cost"],
            row["sensor_name"],
        )
    )
    return {
        "target": "future mean end-effector translation path over registered horizon",
        "target_thresholds_m": list(target_thresholds),
        "target_outcome_counts": np.bincount(
            np.asarray(labels, dtype=np.int64), minlength=3
        ).tolist(),
        "target_entropy_nats": _entropy(labels),
        "majority_baseline_accuracy": baseline,
        "sensor_codebooks": {
            sensor: {"thresholds": list(thresholds), "outcomes": ["low", "mid", "high"]}
            for sensor, thresholds in codebooks.items()
        },
        "single_sensor_diagnostics": single,
        "best_fixed_pairs": pairs[:10],
        "interpretation": (
            "This is a same-support routing diagnostic over source episodes. It tests "
            "whether registered prefix features carry differentiated information about "
            "a future physical-motion carrier. Leave-one-episode-out rows are grouped by "
            "complete episode, but this remains source-only carrier evidence rather than "
            "held-target prediction evidence."
        ),
    }


def run_source_audit(
    point_cloud_root: str | Path,
    aligned_root: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    *,
    tactile_roots: Sequence[str | Path] = (),
) -> dict[str, Any]:
    config = _load_config(config_path)
    pcd_root = Path(point_cloud_root).resolve()
    robot_root = Path(aligned_root).resolve()
    extra_tactile_roots = tuple(Path(root).resolve() for root in tactile_roots)
    _require(pcd_root.is_dir(), f"point-cloud root is missing: {pcd_root}")
    _require(robot_root.is_dir(), f"aligned root is missing: {robot_root}")
    for root in extra_tactile_roots:
        _require(root.is_dir(), f"tactile root is missing: {root}")
    source_ids = tuple(config["source_episode_ids"])
    forbidden_ids = tuple(config["forbidden_episode_ids"])
    object_id = str(config["object_id"])
    episode_records: list[dict[str, Any]] = []
    feature_state: dict[int, dict[str, FloatArray]] = {}
    common_tactile: set[str] | None = None

    roots_for_tactile = tuple(
        dict.fromkeys((robot_root, pcd_root, *extra_tactile_roots))
    )
    for episode_id in source_ids:
        pcd_episode = _episode_directory(pcd_root, object_id, episode_id)
        robot_episode = _episode_directory(robot_root, object_id, episode_id)
        _require(
            pcd_episode.is_relative_to(pcd_root / object_id),
            "point-cloud episode escaped the object root",
        )
        _require(
            robot_episode.is_relative_to(robot_root / object_id),
            "aligned episode escaped the object root",
        )
        tactile_episode_roots = tuple(
            (episode, root)
            for root in roots_for_tactile
            if (episode := _find_episode_directory(root, object_id, episode_id))
            is not None
        )
        pcd = _point_cloud_inventory(pcd_episode, pcd_root)
        robot, robot_arrays = _robot_inventory(robot_episode, robot_root)
        tactile, tactile_energy = _tactile_inventory(tactile_episode_roots)
        feature_series = _robot_feature_series(robot_arrays)
        feature_series.update(
            {f"tactile:{name}": value for name, value in tactile_energy.items()}
        )
        feature_state[episode_id] = feature_series
        names = set(tactile_energy)
        common_tactile = names if common_tactile is None else common_tactile & names
        episode_records.append(
            {
                "episode_id": episode_id,
                "point_cloud_episode_directory": pcd_episode.relative_to(
                    pcd_root
                ).as_posix(),
                "aligned_episode_directory": robot_episode.relative_to(
                    robot_root
                ).as_posix(),
                "point_cloud": pcd,
                "robot": robot,
                "tactile": tactile,
            }
        )

    common_tactile = common_tactile or set()
    sensor_names = ["robot-opening", "robot-translation"] + [
        f"tactile:{name}" for name in sorted(common_tactile)
    ]
    source_costs = config["sensor_costs"]
    sensor_costs = {
        name: float(
            source_costs["robot_opening"]
            if name == "robot-opening"
            else source_costs["robot_translation"]
            if name == "robot-translation"
            else source_costs["tactile"]
        )
        for name in sensor_names
    }
    rows: list[dict[str, Any]] = []
    episode_readiness: list[dict[str, Any]] = []
    for record in episode_records:
        episode_id = int(record["episode_id"])
        series = feature_state[episode_id]
        available = [name for name in sensor_names if name in series]
        frame_counts = [len(series[name]) for name in available]
        pcd_frames = int(record["point_cloud"].get("frame_count") or 0)
        if pcd_frames:
            frame_counts.append(pcd_frames)
        common_frames = min(frame_counts) if frame_counts else 0
        ready = bool(
            record["point_cloud"].get("available")
            and record["robot"].get("available")
            and set(sensor_names) <= set(series)
            and common_frames >= int(config["minimum_frames"])
        )
        episode_readiness.append(
            {
                "episode_id": episode_id,
                "common_frame_count": common_frames,
                "available_sensor_groups": available,
                "ready": ready,
            }
        )
        if not ready:
            continue
        resets = _reset_positions(
            common_frames,
            reset_count=int(config["reset_count"]),
            prefix_window=int(config["prefix_window_frames"]),
            future_horizon=int(config["future_horizon_frames"]),
        )
        translation = series["robot-translation"]
        for reset in resets:
            features: dict[str, float] = {}
            for sensor in sensor_names:
                values = series[sensor]
                if sensor == "robot-opening":
                    features[sensor] = _window_range(
                        values, reset, int(config["prefix_window_frames"])
                    )
                else:
                    features[sensor] = _window_sum(
                        values, reset, int(config["prefix_window_frames"])
                    )
            horizon = int(config["future_horizon_frames"])
            rows.append(
                {
                    "case_id": f"{object_id}/episode-{episode_id}/reset-{reset}",
                    "episode_id": episode_id,
                    "reset_frame": reset,
                    "features": features,
                    "future_robot_translation_path_m": float(
                        np.sum(translation[reset + 1 : reset + horizon + 1])
                    ),
                }
            )

    pcd_ready = all(
        record["point_cloud"].get("available") for record in episode_records
    )
    robot_ready = all(
        record["robot"].get("required_keys_available") for record in episode_records
    )
    tactile_ready = len(common_tactile) >= int(config["minimum_common_tactile_groups"])
    cases_ready = len(rows) == len(source_ids) * int(config["reset_count"])
    ready = bool(pcd_ready and robot_ready and tactile_ready and cases_ready)
    routing = _routing_diagnostic(rows, sensor_costs) if rows and sensor_names else None
    payload: dict[str, Any] = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "artifact_kind": AUDIT_KIND,
        "protocol_id": config["protocol_id"],
        "dataset": {
            "repository": config["dataset_repository"],
            "revision": config["dataset_revision"],
            "point_cloud_root": str(pcd_root),
            "aligned_root": str(robot_root),
            "tactile_roots": [str(root) for root in roots_for_tactile],
            "object_id": object_id,
            "source_episode_ids": list(source_ids),
            "forbidden_episode_ids": list(forbidden_ids),
            "forbidden_episode_paths_not_opened": [
                str(root / object_id / f"episode_{episode_id:04d}")
                for root in dict.fromkeys((pcd_root, robot_root, *roots_for_tactile))
                for episode_id in forbidden_ids
            ],
        },
        "episode_records": episode_records,
        "episode_readiness": episode_readiness,
        "sensor_group_roster": [
            {
                "sensor_name": name,
                "registered_cost": sensor_costs[name],
                "finite_outcome_adapter": "source-tercile scalar-prefix codebook",
            }
            for name in sensor_names
        ],
        "feature_case_count": len(rows),
        "feature_case_descriptor_sha256": hashlib.sha256(
            _canonical_bytes(rows)
        ).hexdigest(),
        "routing_diagnostic": routing,
        "capability_gates": {
            "point_cloud_carriers_complete": pcd_ready,
            "robot_prefix_carriers_complete": robot_ready,
            "common_tactile_group_count": len(common_tactile),
            "minimum_common_tactile_group_count": int(
                config["minimum_common_tactile_groups"]
            ),
            "tactile_prefix_carriers_complete": tactile_ready,
            "registered_feature_cases_complete": cases_ready,
            "ready_for_source_only_sensor_reveal_pilot": ready,
        },
        "information_boundary": {
            **config["information_boundary"],
            "source_episode_payloads_read": True,
            "point_cloud_member_payloads_read": False,
            "robot_source_values_read": True,
            "tactile_source_values_read": True,
            "raw_camera_video_decoded": False,
            "forbidden_episode_paths_opened": False,
        },
        "claim_boundary": (
            "This audit establishes source-side carrier availability, synchronized "
            "prefix feature feasibility, finite codebook occupancy, and a same-support "
            "routing diagnostic. It does not establish held-out target performance, "
            "object-disjoint transport, causal sensor value, online acquisition, "
            "deployment authorization, or safety."
        ),
    }
    payload["result_sha256"] = _payload_sha256(payload, digest_field="result_sha256")
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_source_audit(payload: Mapping[str, Any]) -> None:
    _require(
        payload.get("schema_version") == AUDIT_SCHEMA_VERSION, "result schema changed"
    )
    _require(payload.get("artifact_kind") == AUDIT_KIND, "result kind changed")
    _require(
        payload.get("result_sha256")
        == _payload_sha256(payload, digest_field="result_sha256"),
        "result checksum mismatch",
    )
    boundary_value = payload.get("information_boundary")
    if not isinstance(boundary_value, Mapping):
        raise ValueError("information boundary is missing")
    boundary = cast(Mapping[str, Any], boundary_value)
    _require(
        boundary.get("source_only") is True
        and boundary.get("forbidden_episode_payloads_read") is False
        and boundary.get("held_target_payloads_read") is False
        and boundary.get("forbidden_episode_paths_opened") is False
        and boundary.get("dataset_modified") is False
        and boundary.get("paper_claim_authorized") is False,
        "result crossed a source or claim boundary",
    )
    dataset_value = payload.get("dataset")
    if not isinstance(dataset_value, Mapping):
        raise ValueError("dataset record is missing")
    dataset = cast(Mapping[str, Any], dataset_value)
    source_ids = dataset.get("source_episode_ids")
    forbidden_ids = dataset.get("forbidden_episode_ids")
    if not isinstance(source_ids, list) or not isinstance(forbidden_ids, list):
        raise ValueError("result episode rosters are missing")
    _require(
        not set(source_ids) & set(forbidden_ids),
        "result source and forbidden episode rosters overlap",
    )
    records_value = payload.get("episode_records")
    if not isinstance(records_value, list) or not all(
        isinstance(record, Mapping) for record in records_value
    ):
        raise ValueError("result episode records are missing")
    records = cast(list[Mapping[str, Any]], records_value)
    _require(
        [record.get("episode_id") for record in records] == source_ids,
        "result episode roster changed",
    )


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--point-cloud-root", type=Path)
    parser.add_argument("--aligned-root", type=Path)
    parser.add_argument("--tactile-root", type=Path, action="append", default=[])
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        validate_source_audit(json.loads(args.verify.read_text(encoding="utf-8")))
        return 0
    point_cloud_root = args.point_cloud_root
    aligned_root = args.aligned_root
    config = args.config
    output = args.output
    if (
        point_cloud_root is None
        or aligned_root is None
        or config is None
        or output is None
    ):
        raise ValueError(
            "--point-cloud-root, --aligned-root, --config, and --output are required"
        )
    run_source_audit(
        point_cloud_root,
        aligned_root,
        config,
        output,
        tactile_roots=args.tactile_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
