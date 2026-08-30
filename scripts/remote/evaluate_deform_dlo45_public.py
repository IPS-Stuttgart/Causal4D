#!/usr/bin/env python3
"""Target-closed public-data evaluation on DEFORM DLO4 and DLO5.

The evaluator is deliberately fail-closed. It only marks the result claim-eligible
when it can recover a repeated-action grouping from the released file identities.
Target suffixes are not passed to the abduction routine. Predictions are serialized
and hashed before suffix scoring.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

SUPPORTED_SUFFIXES = {
    ".npy",
    ".npz",
    ".mat",
    ".csv",
    ".txt",
    ".json",
    ".h5",
    ".hdf5",
}


@dataclass(frozen=True)
class LoadedTrajectory:
    object_id: str
    path: Path
    relative_path: str
    values: np.ndarray
    source_key: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix-fraction", type=float, default=0.30)
    parser.add_argument("--bootstrap-replicates", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--request-id", default="unspecified")
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def numeric_arrays(value: Any, prefix: str = "root") -> list[tuple[str, np.ndarray]]:
    arrays: list[tuple[str, np.ndarray]] = []
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and value.size:
            arrays.append((prefix, value))
        return arrays
    if isinstance(value, Mapping):
        for key, child in value.items():
            arrays.extend(numeric_arrays(child, f"{prefix}.{key}"))
        return arrays
    if isinstance(value, (list, tuple)):
        try:
            array = np.asarray(value)
        except (TypeError, ValueError):
            array = None
        if (
            array is not None
            and array.size
            and np.issubdtype(array.dtype, np.number)
        ):
            arrays.append((prefix, array))
        else:
            for index, child in enumerate(value[:200]):
                arrays.extend(numeric_arrays(child, f"{prefix}[{index}]"))
    return arrays


def choose_array(arrays: Iterable[tuple[str, np.ndarray]]) -> tuple[str, np.ndarray]:
    candidates = []
    for key, array in arrays:
        squeezed = np.asarray(array).squeeze()
        if squeezed.ndim < 2 or squeezed.size < 24:
            continue
        numeric_fraction = float(np.isfinite(squeezed.astype(float, copy=False)).mean())
        candidates.append((squeezed.size * numeric_fraction, key, squeezed))
    require(bool(candidates), "no usable numeric trajectory array")
    _, key, array = max(candidates, key=lambda item: item[0])
    return key, array


def load_raw_array(path: Path) -> tuple[str, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        return choose_array([("npy", np.load(path, allow_pickle=False))])
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            return choose_array((key, archive[key]) for key in archive.files)
    if suffix == ".mat":
        try:
            from scipy.io import loadmat

            payload = loadmat(path, simplify_cells=True)
            return choose_array(numeric_arrays(payload))
        except NotImplementedError:
            pass
        import h5py

        arrays: list[tuple[str, np.ndarray]] = []
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and len(arrays) < 200:
                    try:
                        arrays.append((name, np.asarray(obj)))
                    except (OSError, TypeError, ValueError):
                        return

            handle.visititems(visitor)
        return choose_array(arrays)
    if suffix in {".h5", ".hdf5"}:
        import h5py

        arrays = []
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset) and len(arrays) < 200:
                    try:
                        arrays.append((name, np.asarray(obj)))
                    except (OSError, TypeError, ValueError):
                        return

            handle.visititems(visitor)
        return choose_array(arrays)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return choose_array(numeric_arrays(payload))
    if suffix in {".csv", ".txt"}:
        delimiter = "," if suffix == ".csv" else None
        array = np.genfromtxt(
            path,
            delimiter=delimiter,
            comments="#",
            invalid_raise=False,
        )
        return choose_array([(suffix[1:], array)])
    raise ValueError(f"unsupported suffix {suffix}")


def interpolate_columns(values: np.ndarray) -> np.ndarray:
    output = values.astype(float, copy=True)
    x = np.arange(output.shape[0], dtype=float)
    keep = np.isfinite(output).mean(axis=0) >= 0.80
    output = output[:, keep]
    require(output.shape[1] >= 2, "fewer than two sufficiently finite features")
    for column in range(output.shape[1]):
        finite = np.isfinite(output[:, column])
        require(int(finite.sum()) >= 2, "feature has fewer than two finite values")
        if not finite.all():
            output[:, column] = np.interp(
                x,
                x[finite],
                output[finite, column],
            )
    return output


def canonicalize(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array).squeeze()
    require(value.ndim >= 2, "trajectory array must have at least two dimensions")
    shape = value.shape
    plausible = [index for index, length in enumerate(shape) if 12 <= length <= 10000]
    require(bool(plausible), f"no plausible time axis in shape {shape}")
    if 0 in plausible:
        time_axis = 0
    else:
        time_axis = max(plausible, key=lambda index: shape[index])
    value = np.moveaxis(value, time_axis, 0)
    value = value.reshape(value.shape[0], -1)
    require(value.shape[0] >= 12, "trajectory has fewer than 12 time steps")
    require(value.shape[1] <= 20000, "trajectory feature dimension is implausibly large")
    return interpolate_columns(value)


def discover_files(root: Path, object_id: str) -> list[Path]:
    token = object_id.lower()
    files = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_SUFFIXES
        and token in path.as_posix().lower()
    ]
    if not files:
        candidate_dirs = [
            path for path in root.rglob("*") if path.is_dir() and path.name.lower() == token
        ]
        for directory in candidate_dirs:
            files.extend(
                path
                for path in directory.rglob("*")
                if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
            )
    return sorted(set(files), key=lambda path: natural_key(path.as_posix()))


def natural_key(text: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def load_object(root: Path, object_id: str) -> tuple[list[LoadedTrajectory], list[dict[str, str]]]:
    records: list[LoadedTrajectory] = []
    failures: list[dict[str, str]] = []
    for path in discover_files(root, object_id):
        try:
            key, raw = load_raw_array(path)
            values = canonicalize(raw)
            records.append(
                LoadedTrajectory(
                    object_id=object_id,
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    values=values,
                    source_key=key,
                )
            )
        except Exception as error:  # noqa: BLE001
            failures.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "error": f"{type(error).__name__}: {error}",
                }
            )
    return records, failures


def grouping_quality(labels: Sequence[str]) -> tuple[int, list[int]]:
    counts = Counter(labels)
    return len(counts), sorted(counts.values())


def infer_grouping(records: Sequence[LoadedTrajectory]) -> dict[str, Any]:
    require(bool(records), "cannot group an empty record list")
    paths = [Path(record.relative_path) for record in records]
    candidates: list[dict[str, Any]] = []

    parent_labels = [path.parent.as_posix() for path in paths]
    candidates.append({"method": "parent", "labels": parent_labels, "verified": True})

    stems = [path.stem for path in paths]
    stripped = [
        re.sub(r"(?:[_-](?:trial|repeat|rep|take|run)?\d+)$", "", stem, flags=re.I)
        for stem in stems
    ]
    candidates.append(
        {"method": "stem_without_repeat_suffix", "labels": stripped, "verified": True}
    )

    number_lists = [re.findall(r"\d+", stem) for stem in stems]
    max_tokens = max((len(tokens) for tokens in number_lists), default=0)
    for index in range(max_tokens):
        if all(len(tokens) > index for tokens in number_lists):
            candidates.append(
                {
                    "method": f"numeric_token_from_left_{index}",
                    "labels": [tokens[index] for tokens in number_lists],
                    "verified": True,
                }
            )
    for index in range(1, max_tokens + 1):
        if all(len(tokens) >= index for tokens in number_lists):
            candidates.append(
                {
                    "method": f"numeric_token_from_right_{index}",
                    "labels": [tokens[-index] for tokens in number_lists],
                    "verified": True,
                }
            )

    ideal: list[dict[str, Any]] = []
    for candidate in candidates:
        group_count, sizes = grouping_quality(candidate["labels"])
        candidate["group_count"] = group_count
        candidate["group_sizes"] = sizes
        if group_count == 14 and sizes == [5] * 14:
            ideal.append(candidate)
    if ideal:
        preference = {
            "stem_without_repeat_suffix": 0,
            "parent": 1,
        }
        selected = min(
            ideal,
            key=lambda item: (
                preference.get(item["method"], 2),
                item["method"],
            ),
        )
        return selected

    if len(records) == 70:
        labels = [f"block_{index // 5:02d}" for index in range(70)]
        return {
            "method": "natural_order_blocks_of_five",
            "labels": labels,
            "verified": False,
            "group_count": 14,
            "group_sizes": [5] * 14,
            "reason": "No released identity field independently established the blocks.",
            "candidate_diagnostics": [
                {
                    "method": item["method"],
                    "group_count": item["group_count"],
                    "group_sizes": item["group_sizes"],
                }
                for item in candidates
            ],
        }

    return {
        "method": "unresolved",
        "labels": [f"unresolved_{index:03d}" for index in range(len(records))],
        "verified": False,
        "group_count": len(records),
        "group_sizes": [1] * len(records),
        "reason": f"Expected 14 repeated actions x 5 recordings, found {len(records)} usable files.",
        "candidate_diagnostics": [
            {
                "method": item["method"],
                "group_count": item["group_count"],
                "group_sizes": item["group_sizes"],
            }
            for item in candidates
        ],
    }


def mode(values: Sequence[int]) -> int:
    counts = Counter(values)
    return max(counts, key=lambda value: (counts[value], -value))


def resample(values: np.ndarray, length: int) -> np.ndarray:
    if values.shape[0] == length:
        return values.copy()
    source = np.linspace(0.0, 1.0, values.shape[0])
    target = np.linspace(0.0, 1.0, length)
    output = np.empty((length, values.shape[1]), dtype=float)
    for column in range(values.shape[1]):
        output[:, column] = np.interp(target, source, values[:, column])
    return output


def harmonize(
    records: Sequence[LoadedTrajectory], labels: Sequence[str]
) -> tuple[list[LoadedTrajectory], list[str], dict[str, Any]]:
    feature_dim = mode([record.values.shape[1] for record in records])
    retained = [record for record in records if record.values.shape[1] == feature_dim]
    retained_labels = [
        label
        for record, label in zip(records, labels, strict=True)
        if record.values.shape[1] == feature_dim
    ]
    require(len(retained) >= 10, "too few records share a common feature dimension")
    lengths = [record.values.shape[0] for record in retained]
    target_length = int(np.median(lengths))
    target_length = min(max(target_length, 20), 600)
    harmonized = [
        LoadedTrajectory(
            object_id=record.object_id,
            path=record.path,
            relative_path=record.relative_path,
            values=resample(record.values, target_length),
            source_key=record.source_key,
        )
        for record in retained
    ]
    return harmonized, retained_labels, {
        "feature_dimension": feature_dim,
        "target_length": target_length,
        "retained_count": len(retained),
        "discarded_dimension_mismatch": len(records) - len(retained),
        "original_lengths": lengths,
    }


def shift_trajectory(values: np.ndarray, delay: int) -> np.ndarray:
    indices = np.arange(values.shape[0]) - delay
    indices = np.clip(indices, 0, values.shape[0] - 1)
    return values[indices]


def robust_scale(source_values: np.ndarray, nominal: np.ndarray, prefix: int) -> np.ndarray:
    residuals = source_values[:, :prefix] - nominal[None, :prefix]
    scale = 1.4826 * np.median(np.abs(residuals), axis=(0, 1))
    nominal_motion = np.diff(nominal[:prefix], axis=0)
    floor = np.median(np.abs(nominal_motion), axis=0)
    global_floor = max(float(np.median(np.abs(nominal_motion))), 1e-9)
    scale = np.maximum(scale, np.maximum(floor * 0.10, global_floor * 0.02))
    return scale


def student_log_likelihood(residual: np.ndarray, scale: np.ndarray, nu: float = 4.0) -> float:
    standardized = residual / scale[None, :]
    terms = -0.5 * (nu + 1.0) * np.log1p((standardized**2) / nu)
    terms -= np.log(scale[None, :])
    return float(np.mean(terms) * residual.shape[0])


def softmax(log_weights: np.ndarray) -> np.ndarray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    return weights / weights.sum()


def abduct_and_predict(
    target_prefix: np.ndarray,
    source_values: np.ndarray,
    prefix: int,
) -> dict[str, Any]:
    """Build predictions without receiving the target suffix."""
    nominal = source_values.mean(axis=0)
    source_variance = source_values.var(axis=0, ddof=1)
    scale = robust_scale(source_values, nominal, prefix)
    delays = range(-4, 5)
    gains = np.linspace(0.75, 1.25, 11)
    candidate_predictions: list[np.ndarray] = []
    candidate_variances: list[np.ndarray] = []
    candidate_meta: list[dict[str, Any]] = []
    log_weights: list[float] = []

    for delay in delays:
        shifted = shift_trajectory(nominal, delay)
        shifted_variance = shift_trajectory(source_variance, delay)
        for gain in gains:
            offset = np.median(
                target_prefix - gain * shifted[:prefix],
                axis=0,
            )
            prediction = gain * shifted + offset[None, :]
            residual = target_prefix - prediction[:prefix]
            log_likelihood = student_log_likelihood(residual, scale)
            log_prior = -0.5 * ((gain - 1.0) / 0.15) ** 2
            log_prior += -0.5 * (delay / 2.0) ** 2
            candidate_predictions.append(prediction)
            candidate_variances.append((gain**2) * shifted_variance)
            candidate_meta.append(
                {
                    "delay_frames": delay,
                    "gain": float(gain),
                    "offset_norm": float(np.linalg.norm(offset)),
                }
            )
            log_weights.append(log_likelihood + log_prior)

    log_weight_array = np.asarray(log_weights)
    weights = softmax(log_weight_array)
    predictions = np.stack(candidate_predictions)
    variances = np.stack(candidate_variances)
    posterior_mean = np.einsum("c,ctd->td", weights, predictions)
    mixture_variance = np.einsum(
        "c,ctd->td",
        weights,
        variances + (predictions - posterior_mean[None, :, :]) ** 2,
    )
    variance_floor = np.maximum(scale**2, 1e-12)
    mixture_variance = np.maximum(mixture_variance, variance_floor[None, :])
    map_index = int(np.argmax(weights))
    residual_at_boundary = target_prefix[-1] - nominal[prefix - 1]
    last_residual = nominal + residual_at_boundary[None, :]
    persistence = np.repeat(target_prefix[-1][None, :], nominal.shape[0], axis=0)

    return {
        "nominal": nominal,
        "posterior_mean": posterior_mean,
        "posterior_variance": mixture_variance,
        "map": predictions[map_index],
        "last_residual": last_residual,
        "persistence": persistence,
        "posterior_weights": weights,
        "candidate_meta": candidate_meta,
        "map_meta": candidate_meta[map_index],
        "posterior_entropy": float(-np.sum(weights * np.log(weights + 1e-300))),
        "effective_candidate_count": float(1.0 / np.sum(weights**2)),
    }


def infer_coordinate_dimension(feature_dimension: int) -> int | None:
    if feature_dimension % 3 == 0:
        return 3
    if feature_dimension % 2 == 0:
        return 2
    return None


def centered(values: np.ndarray, coordinate_dimension: int | None) -> np.ndarray:
    if coordinate_dimension is None:
        return values
    points = values.reshape(values.shape[0], -1, coordinate_dimension)
    points = points - points.mean(axis=1, keepdims=True)
    return points.reshape(values.shape)


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    return float(np.sqrt(np.mean((prediction - target) ** 2)))


def normalized_rmse(prediction: np.ndarray, target: np.ndarray, scale: float) -> float:
    return rmse(prediction, target) / max(scale, 1e-12)


def gaussian_nll(prediction: np.ndarray, variance: np.ndarray, target: np.ndarray) -> float:
    safe_variance = np.maximum(variance, 1e-12)
    value = 0.5 * (
        np.log(2.0 * math.pi * safe_variance)
        + ((target - prediction) ** 2) / safe_variance
    )
    return float(np.mean(value))


def coverage90(prediction: np.ndarray, variance: np.ndarray, target: np.ndarray) -> float:
    half_width = 1.6448536269514722 * np.sqrt(np.maximum(variance, 1e-12))
    return float(np.mean(np.abs(target - prediction) <= half_width))


def prediction_hash(payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = payload[name]
        if isinstance(value, np.ndarray):
            digest.update(name.encode("utf-8"))
            digest.update(str(value.shape).encode("ascii"))
            digest.update(str(value.dtype).encode("ascii"))
            digest.update(np.ascontiguousarray(value).tobytes())
    return digest.hexdigest()


def evaluate_object(
    object_id: str,
    records: Sequence[LoadedTrajectory],
    labels: Sequence[str],
    prefix_fraction: float,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    require(all(len(indices) >= 2 for indices in grouped.values()), "singleton action group")
    length = records[0].values.shape[0]
    prefix = int(round(length * prefix_fraction))
    prefix = min(max(prefix, 6), length - 6)
    coordinate_dimension = infer_coordinate_dimension(records[0].values.shape[1])
    rows: list[dict[str, Any]] = []
    sealed_arrays: dict[str, np.ndarray] = {}

    for group_label in sorted(grouped, key=natural_key):
        indices = grouped[group_label]
        for target_index in indices:
            source_indices = [index for index in indices if index != target_index]
            target = records[target_index].values
            sources = np.stack([records[index].values for index in source_indices])
            result = abduct_and_predict(target[:prefix].copy(), sources, prefix)
            seal_payload = {
                "nominal": result["nominal"],
                "posterior_mean": result["posterior_mean"],
                "posterior_variance": result["posterior_variance"],
                "map": result["map"],
                "last_residual": result["last_residual"],
                "persistence": result["persistence"],
            }
            seal = prediction_hash(seal_payload)
            key = f"{object_id}_{group_label}_{target_index:03d}"
            for method_name, array in seal_payload.items():
                sealed_arrays[f"{key}__{method_name}"] = array

            suffix = slice(prefix, None)
            target_suffix = target[suffix]
            trajectory_extent = float(
                np.sqrt(np.mean((target - target.mean(axis=0, keepdims=True)) ** 2))
            )
            posterior_variance = result["posterior_variance"][suffix]
            base_variance = np.maximum(
                sources.var(axis=0, ddof=1)[suffix],
                np.median(posterior_variance) * 0.05 + 1e-12,
            )
            metrics: dict[str, dict[str, float]] = {}
            for method_name in (
                "nominal",
                "posterior_mean",
                "map",
                "last_residual",
                "persistence",
            ):
                prediction = result[method_name][suffix]
                variance = (
                    posterior_variance
                    if method_name == "posterior_mean"
                    else base_variance
                )
                metrics[method_name] = {
                    "rmse": rmse(prediction, target_suffix),
                    "normalized_rmse": normalized_rmse(
                        prediction,
                        target_suffix,
                        trajectory_extent,
                    ),
                    "centered_rmse": rmse(
                        centered(prediction, coordinate_dimension),
                        centered(target_suffix, coordinate_dimension),
                    ),
                    "gaussian_nll": gaussian_nll(
                        prediction,
                        variance,
                        target_suffix,
                    ),
                    "coverage90": coverage90(
                        prediction,
                        variance,
                        target_suffix,
                    ),
                }
            rows.append(
                {
                    "object_id": object_id,
                    "group_label": group_label,
                    "target_index": target_index,
                    "target_path": records[target_index].relative_path,
                    "source_paths": [records[index].relative_path for index in source_indices],
                    "prefix_steps": prefix,
                    "total_steps": length,
                    "feature_dimension": records[target_index].values.shape[1],
                    "coordinate_dimension": coordinate_dimension,
                    "prediction_sha256": seal,
                    "posterior_entropy": result["posterior_entropy"],
                    "effective_candidate_count": result["effective_candidate_count"],
                    "map_intervention": result["map_meta"],
                    "metrics": metrics,
                }
            )
    np.savez_compressed(output_dir / f"sealed_predictions_{object_id}.npz", **sealed_arrays)
    return rows, {
        "prefix_steps": prefix,
        "total_steps": length,
        "feature_dimension": records[0].values.shape[1],
        "coordinate_dimension": coordinate_dimension,
        "action_group_count": len(grouped),
        "recording_count": len(records),
    }


def flatten_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    flat: list[dict[str, Any]] = []
    for row in rows:
        base = {
            key: value
            for key, value in row.items()
            if key not in {"metrics", "source_paths", "map_intervention"}
        }
        base["source_paths"] = "|".join(row["source_paths"])
        base["map_gain"] = row["map_intervention"]["gain"]
        base["map_delay_frames"] = row["map_intervention"]["delay_frames"]
        for method, metrics in row["metrics"].items():
            for metric, value in metrics.items():
                base[f"{method}__{metric}"] = value
        flat.append(base)
    return flat


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(rows), "cannot write empty results CSV")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def grouped_differences(
    rows: Sequence[dict[str, Any]], method: str, baseline: str
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        key = f"{row['object_id']}::{row['group_label']}"
        delta = (
            row["metrics"][baseline]["rmse"]
            - row["metrics"][method]["rmse"]
        )
        grouped[key].append(float(delta))
    return {key: float(np.mean(values)) for key, values in grouped.items()}


def bootstrap_mean_ci(
    values: Sequence[float], replicates: int, rng: np.random.Generator
) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    require(array.size >= 2, "bootstrap requires at least two independent groups")
    indices = rng.integers(0, array.size, size=(replicates, array.size))
    means = array[indices].mean(axis=1)
    return {
        "mean": float(array.mean()),
        "lower95": float(np.quantile(means, 0.025)),
        "upper95": float(np.quantile(means, 0.975)),
    }


def one_sided_sign_pvalue(wins: int, non_ties: int) -> float:
    if non_ties == 0:
        return 1.0
    numerator = sum(math.comb(non_ties, value) for value in range(wins, non_ties + 1))
    return float(numerator / (2**non_ties))


def aggregate(
    rows: Sequence[dict[str, Any]],
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    methods = ["nominal", "posterior_mean", "map", "last_residual", "persistence"]
    metrics = ["rmse", "normalized_rmse", "centered_rmse", "gaussian_nll", "coverage90"]
    summary: dict[str, Any] = {"recording_count": len(rows), "methods": {}}
    for method in methods:
        summary["methods"][method] = {}
        for metric in metrics:
            values = np.asarray([row["metrics"][method][metric] for row in rows])
            summary["methods"][method][metric] = {
                "mean": float(values.mean()),
                "median": float(np.median(values)),
                "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            }

    comparisons = {}
    for baseline in ("nominal", "last_residual", "persistence"):
        differences = grouped_differences(rows, "posterior_mean", baseline)
        values = list(differences.values())
        wins = sum(value > 0.0 for value in values)
        losses = sum(value < 0.0 for value in values)
        non_ties = wins + losses
        comparisons[f"posterior_mean_vs_{baseline}"] = {
            "positive_delta_means_posterior_is_better": True,
            "independent_unit": "object_action_group",
            "group_count": len(values),
            "group_wins": wins,
            "group_losses": losses,
            "group_ties": len(values) - non_ties,
            "win_fraction": wins / len(values) if values else 0.0,
            "one_sided_sign_pvalue": one_sided_sign_pvalue(wins, non_ties),
            "mean_delta_bootstrap": bootstrap_mean_ci(
                values,
                bootstrap_replicates,
                rng,
            ),
            "per_group_delta": differences,
        }
    summary["comparisons"] = comparisons
    return summary


def markdown_report(evidence: Mapping[str, Any]) -> str:
    lines = [
        "# DEFORM DLO4/DLO5 public-data evaluation",
        "",
        f"Request: `{evidence['request_id']}`",
        "",
        f"Decision: **{evidence['decision']}**",
        "",
        "## Scope",
        "",
        "- Existing public DLO4 and DLO5 files only.",
        "- Leave-one-recording-out within released action groups.",
        "- Target prefix used for intervention abduction; suffix used only after prediction sealing.",
        "- Independent inference unit: object-action group.",
        "",
    ]
    if evidence.get("aggregate"):
        aggregate_result = evidence["aggregate"]
        lines.extend(
            [
                "## Aggregate suffix performance",
                "",
                "| Method | RMSE | normalized RMSE | centered RMSE | 90% coverage |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, result in aggregate_result["methods"].items():
            lines.append(
                "| {method} | {rmse:.6g} | {nrmse:.6g} | {crmse:.6g} | {coverage:.3f} |".format(
                    method=method,
                    rmse=result["rmse"]["mean"],
                    nrmse=result["normalized_rmse"]["mean"],
                    crmse=result["centered_rmse"]["mean"],
                    coverage=result["coverage90"]["mean"],
                )
            )
        lines.extend(["", "## Clustered comparisons", ""])
        for name, result in aggregate_result["comparisons"].items():
            ci = result["mean_delta_bootstrap"]
            lines.append(
                f"- `{name}`: {result['group_wins']}/{result['group_count']} group wins; "
                f"mean RMSE reduction {ci['mean']:.6g}, 95% bootstrap CI "
                f"[{ci['lower95']:.6g}, {ci['upper95']:.6g}]; "
                f"one-sided sign p={result['one_sided_sign_pvalue']:.6g}."
            )
        lines.append("")
    lines.extend(
        [
            "## Scientific boundary",
            "",
            f"- Claim eligible: `{evidence['claim_eligible']}`.",
            "- No new physical data were collected.",
            "- No result is promoted when released action grouping is unresolved.",
            "- The oracle candidate grid is not reported as a method result.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    require(0.15 <= args.prefix_fraction <= 0.60, "prefix fraction outside [0.15, 0.60]")
    require(args.bootstrap_replicates >= 1000, "too few bootstrap replicates")
    root = args.data_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    require(root.is_dir(), f"missing DEFORM root: {root}")

    all_rows: list[dict[str, Any]] = []
    object_reports: dict[str, Any] = {}
    all_groupings_verified = True
    for object_id in ("DLO4", "DLO5"):
        loaded, failures = load_object(root, object_id)
        grouping = infer_grouping(loaded)
        harmonized, labels, harmonization = harmonize(loaded, grouping["labels"])
        post_group_count, post_group_sizes = grouping_quality(labels)
        grouping_after_harmonization = {
            "group_count": post_group_count,
            "group_sizes": post_group_sizes,
        }
        eligible_object = bool(
            grouping["verified"]
            and len(harmonized) == 70
            and post_group_count == 14
            and post_group_sizes == [5] * 14
        )
        all_groupings_verified = all_groupings_verified and eligible_object
        object_report: dict[str, Any] = {
            "discovered_supported_file_count": len(loaded) + len(failures),
            "loaded_trajectory_count": len(loaded),
            "load_failures": failures,
            "source_arrays": [
                {
                    "path": record.relative_path,
                    "source_key": record.source_key,
                    "shape": list(record.values.shape),
                }
                for record in loaded
            ],
            "grouping": {
                key: value for key, value in grouping.items() if key != "labels"
            },
            "grouping_after_harmonization": grouping_after_harmonization,
            "harmonization": harmonization,
            "claim_eligible_object": eligible_object,
        }
        if post_group_count == 14 and all(size == 5 for size in post_group_sizes):
            rows, evaluation_shape = evaluate_object(
                object_id,
                harmonized,
                labels,
                args.prefix_fraction,
                output_dir,
            )
            all_rows.extend(rows)
            object_report["evaluation_shape"] = evaluation_shape
        else:
            object_report["evaluation_skipped_reason"] = (
                "The harmonized files do not form 14 groups of five recordings."
            )
        object_reports[object_id] = object_report

    aggregate_result = None
    if all_rows:
        write_csv(output_dir / "per_recording_results.csv", flatten_rows(all_rows))
        aggregate_result = aggregate(
            all_rows,
            args.bootstrap_replicates,
            args.seed,
        )

    claim_eligible = bool(all_groupings_verified and len(all_rows) == 140)
    positive = False
    if claim_eligible and aggregate_result is not None:
        nominal = aggregate_result["comparisons"]["posterior_mean_vs_nominal"]
        residual = aggregate_result["comparisons"]["posterior_mean_vs_last_residual"]
        positive = bool(
            nominal["mean_delta_bootstrap"]["lower95"] > 0.0
            and residual["mean_delta_bootstrap"]["lower95"] > 0.0
        )
    if not claim_eligible:
        decision = "not_claim_eligible_schema_or_grouping"
    elif positive:
        decision = "positive_public_realworld_result"
    else:
        decision = "claim_eligible_negative_or_inconclusive_result"

    evidence = {
        "schema_version": 1,
        "artifact_kind": "Causal4DDeformDLO45PublicEvaluationV1",
        "request_id": args.request_id,
        "repository_revision": os.environ.get("GITHUB_SHA"),
        "workflow_run_id": os.environ.get("GITHUB_RUN_ID"),
        "data_root": str(root),
        "prefix_fraction": args.prefix_fraction,
        "bootstrap_replicates": args.bootstrap_replicates,
        "seed": args.seed,
        "objects": object_reports,
        "aggregate": aggregate_result,
        "claim_eligible": claim_eligible,
        "decision": decision,
        "information_boundary": {
            "public_data_only": True,
            "new_physical_data_collected": False,
            "target_suffix_passed_to_abduction": False,
            "predictions_hashed_before_suffix_scoring": True,
            "source_files_modified": False,
            "unverified_grouping_can_authorize_claim": False,
        },
    }
    write_json(output_dir / "evidence.json", evidence)
    (output_dir / "report.md").write_text(markdown_report(evidence), encoding="utf-8")
    print(
        json.dumps(
            {
                "decision": decision,
                "claim_eligible": claim_eligible,
                "evaluated_recordings": len(all_rows),
                "groupings": {
                    object_id: report["grouping"]
                    for object_id, report in object_reports.items()
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
