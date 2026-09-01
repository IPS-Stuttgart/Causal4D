"""Verified DEFORM trajectory loading, grouping, and harmonization."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import pickle
import re
from typing import Any

import numpy as np

from .deform_dlo45_decision_common import require


@dataclass(frozen=True)
class LoadedTrajectory:
    """One released DEFORM trajectory after numeric canonicalization."""

    object_id: str
    path: Path
    relative_path: str
    values: np.ndarray
    source_kind: str


def natural_key(text: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def numeric_candidates(
    value: Any,
    prefix: str = "root",
) -> list[tuple[str, np.ndarray]]:
    """Extract plausible numeric arrays from common pickle payload containers."""
    candidates: list[tuple[str, np.ndarray]] = []
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.number) and value.size:
            candidates.append((prefix, value))
            return candidates
        if value.dtype == object and value.size:
            try:
                converted = np.asarray(value.tolist())
            except (TypeError, ValueError):
                converted = None
            if converted is not None and np.issubdtype(converted.dtype, np.number):
                candidates.append((prefix + ".tolist", converted))
            return candidates
    if hasattr(value, "to_numpy"):
        try:
            converted = value.to_numpy()
        except (AttributeError, TypeError, ValueError):
            converted = None
        if converted is not None:
            candidates.extend(numeric_candidates(converted, prefix + ".to_numpy"))
        if hasattr(value, "values"):
            try:
                candidates.extend(numeric_candidates(value.values, prefix + ".values"))
            except (AttributeError, TypeError, ValueError):
                pass
        return candidates
    if isinstance(value, Mapping):
        for key, child in list(value.items())[:200]:
            candidates.extend(numeric_candidates(child, f"{prefix}.{key}"))
        return candidates
    if isinstance(value, (list, tuple)):
        if value:
            try:
                converted = np.asarray(value)
            except (TypeError, ValueError):
                converted = None
            if converted is not None and np.issubdtype(converted.dtype, np.number):
                candidates.append((prefix, converted))
            elif len(value) <= 10_000:
                try:
                    stacked = np.stack([np.asarray(child) for child in value])
                except (TypeError, ValueError):
                    stacked = None
                if stacked is not None and np.issubdtype(stacked.dtype, np.number):
                    candidates.append((prefix + ".stack", stacked))
        return candidates
    return candidates


def choose_trajectory_array(
    candidates: Iterable[tuple[str, np.ndarray]],
) -> tuple[str, np.ndarray]:
    scored: list[tuple[tuple[int, int, int], str, np.ndarray]] = []
    for key, raw in candidates:
        array = np.asarray(raw).squeeze()
        if array.ndim < 2 or array.size < 72:
            continue
        if not np.issubdtype(array.dtype, np.number):
            continue
        try:
            finite_fraction = float(np.isfinite(array.astype(float, copy=False)).mean())
        except (TypeError, ValueError):
            continue
        if finite_fraction < 0.80:
            continue
        dimensions = tuple(int(length) for length in array.shape)
        time_length = max(dimensions)
        coordinate_hint = int(3 in dimensions)
        scored.append(
            (
                (coordinate_hint, time_length, int(array.size)),
                key,
                array,
            )
        )
    require(bool(scored), "no usable numeric trajectory array")
    _, key, array = max(scored, key=lambda item: item[0])
    return key, array


def load_pickle_payload(path: Path) -> Any:
    try:
        import pandas as pd

        return pd.read_pickle(path)
    except ImportError:
        with path.open("rb") as handle:
            return pickle.load(handle)  # noqa: S301 - checksum-verified official data


def load_raw_payload(
    path: Path,
    *,
    trusted_official_pickle: bool,
) -> tuple[str, np.ndarray]:
    suffix = path.suffix.lower()
    header = path.read_bytes()[:8]
    if suffix == ".npy" or header.startswith(b"\x93NUMPY"):
        return choose_trajectory_array([("npy", np.load(path, allow_pickle=False))])
    if suffix == ".npz" or header.startswith(b"PK\x03\x04"):
        with np.load(path, allow_pickle=False) as archive:
            return choose_trajectory_array(
                (f"npz.{key}", archive[key]) for key in archive.files
            )
    pickle_like = suffix in {"", ".p", ".pkl", ".pickle"} or header[:1] == b"\x80"
    if pickle_like:
        require(
            trusted_official_pickle,
            "pickle loading requires --trusted-official-pickle",
        )
        payload = load_pickle_payload(path)
        key, array = choose_trajectory_array(numeric_candidates(payload))
        return "pickle." + key, array
    raise ValueError(f"unsupported DEFORM trajectory format: {suffix or '<none>'}")


def interpolate_columns(values: np.ndarray) -> np.ndarray:
    output = np.asarray(values, dtype=float).copy()
    keep = np.isfinite(output).mean(axis=0) >= 0.80
    output = output[:, keep]
    require(output.shape[1] >= 6, "fewer than six sufficiently finite features")
    positions = np.arange(output.shape[0], dtype=float)
    for column in range(output.shape[1]):
        finite = np.isfinite(output[:, column])
        require(int(finite.sum()) >= 2, "feature has fewer than two finite values")
        if not finite.all():
            output[:, column] = np.interp(
                positions,
                positions[finite],
                output[finite, column],
            )
    return output


def canonicalize_trajectory(raw: np.ndarray) -> np.ndarray:
    value = np.asarray(raw).squeeze()
    require(value.ndim >= 2, "trajectory must have at least two dimensions")
    time_axis = int(np.argmax(value.shape))
    value = np.moveaxis(value, time_axis, 0)
    value = value.reshape(value.shape[0], -1)
    require(value.shape[0] >= 20, "trajectory has fewer than twenty frames")
    require(value.shape[1] <= 2_000, "trajectory feature dimension is implausible")
    return interpolate_columns(value)


def object_directory(root: Path, object_id: str) -> Path:
    direct = root / object_id
    if direct.is_dir():
        return direct
    matches = [
        path
        for path in root.rglob("*")
        if path.is_dir() and path.name.lower() == object_id.lower()
    ]
    require(len(matches) == 1, f"could not resolve one directory for {object_id}")
    return matches[0]


def discover_files(root: Path, object_id: str) -> list[Path]:
    directory = object_directory(root, object_id)
    return sorted(
        [
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.stat().st_size > 0
            and not path.name.startswith(".")
        ],
        key=lambda path: natural_key(path.relative_to(root).as_posix()),
    )


def load_object(
    root: Path,
    object_id: str,
    *,
    trusted_official_pickle: bool,
) -> tuple[list[LoadedTrajectory], list[dict[str, str]]]:
    records: list[LoadedTrajectory] = []
    failures: list[dict[str, str]] = []
    for path in discover_files(root, object_id):
        try:
            source_kind, raw = load_raw_payload(
                path,
                trusted_official_pickle=trusted_official_pickle,
            )
            records.append(
                LoadedTrajectory(
                    object_id=object_id,
                    path=path,
                    relative_path=path.relative_to(root).as_posix(),
                    values=canonicalize_trajectory(raw),
                    source_kind=source_kind,
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
    stems = [path.stem for path in paths]
    candidates: list[dict[str, Any]] = [
        {
            "method": "parent",
            "labels": [path.parent.as_posix() for path in paths],
            "priority": 1,
        },
        {
            "method": "stem_without_repeat_suffix",
            "labels": [
                re.sub(
                    r"(?:[_-](?:trial|repeat|rep|take|run)?\d+)$",
                    "",
                    stem,
                    flags=re.IGNORECASE,
                )
                for stem in stems
            ],
            "priority": 0,
        },
    ]
    number_lists = [re.findall(r"\d+", stem) for stem in stems]
    max_tokens = max((len(tokens) for tokens in number_lists), default=0)
    for index in range(max_tokens):
        if all(len(tokens) > index for tokens in number_lists):
            candidates.append(
                {
                    "method": f"numeric_token_from_left_{index}",
                    "labels": [tokens[index] for tokens in number_lists],
                    "priority": 10 + index,
                }
            )
    for index in range(1, max_tokens + 1):
        if all(len(tokens) >= index for tokens in number_lists):
            candidates.append(
                {
                    "method": f"numeric_token_from_right_{index}",
                    "labels": [tokens[-index] for tokens in number_lists],
                    "priority": 30 + index,
                }
            )

    diagnostics: list[dict[str, Any]] = []
    valid: list[dict[str, Any]] = []
    for candidate in candidates:
        group_count, sizes = grouping_quality(candidate["labels"])
        diagnostics.append(
            {
                "method": candidate["method"],
                "group_count": group_count,
                "group_sizes": sizes,
            }
        )
        if group_count == 14 and sizes == [5] * 14:
            valid.append(candidate)
    if valid:
        selected = min(valid, key=lambda item: (item["priority"], item["method"]))
        return {
            "method": selected["method"],
            "labels": selected["labels"],
            "verified": True,
            "group_count": 14,
            "group_sizes": [5] * 14,
            "candidate_diagnostics": diagnostics,
        }
    if len(records) == 70:
        return {
            "method": "natural_order_blocks_of_five",
            "labels": [f"block_{index // 5:02d}" for index in range(70)],
            "verified": False,
            "group_count": 14,
            "group_sizes": [5] * 14,
            "reason": "Released identities did not independently establish groups.",
            "candidate_diagnostics": diagnostics,
        }
    return {
        "method": "unresolved",
        "labels": [f"unresolved_{index:03d}" for index in range(len(records))],
        "verified": False,
        "group_count": len(records),
        "group_sizes": [1] * len(records),
        "reason": f"Expected 70 usable trajectories, found {len(records)}.",
        "candidate_diagnostics": diagnostics,
    }


def statistical_mode(values: Sequence[int]) -> int:
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
    records: Sequence[LoadedTrajectory],
    labels: Sequence[str],
) -> tuple[list[LoadedTrajectory], list[str], dict[str, Any]]:
    feature_dimension = statistical_mode([record.values.shape[1] for record in records])
    retained: list[LoadedTrajectory] = []
    retained_labels: list[str] = []
    for record, label in zip(records, labels, strict=True):
        if record.values.shape[1] == feature_dimension:
            retained.append(record)
            retained_labels.append(label)
    require(len(retained) >= 10, "too few trajectories share a feature dimension")
    original_lengths = [record.values.shape[0] for record in retained]
    target_length = min(max(int(np.median(original_lengths)), 20), 600)
    output = [
        LoadedTrajectory(
            object_id=record.object_id,
            path=record.path,
            relative_path=record.relative_path,
            values=resample(record.values, target_length),
            source_kind=record.source_kind,
        )
        for record in retained
    ]
    return (
        output,
        retained_labels,
        {
            "feature_dimension": feature_dimension,
            "target_length": target_length,
            "retained_count": len(retained),
            "discarded_dimension_mismatch": len(records) - len(retained),
            "original_lengths": original_lengths,
        },
    )
