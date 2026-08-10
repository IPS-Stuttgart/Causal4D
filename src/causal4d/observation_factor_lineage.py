"""Validate and bind exact Prob4D observation-factor bundle lineage."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from .artifact_io import (
    load_strict_json_object,
    read_regular_file_beneath,
    read_regular_file_no_symlinks,
)
from .contracts import TwinBelief
from .numpy_archive import load_numpy_archive_snapshot

OBSERVATION_FACTOR_SCHEMA = "prob4d.observation-factor-bundle"
PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION = 3
OBSERVATION_FACTOR_SCHEMA_VERSION = 4
SUPPORTED_OBSERVATION_FACTOR_SCHEMA_VERSIONS = frozenset(
    {
        PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
        OBSERVATION_FACTOR_SCHEMA_VERSION,
    }
)
GAUGE_PARAMETERIZATION = "log-scale-rotvec-translation-v1"
MARGINAL_GAUGE_COVARIANCE = "marginal-blocks-only"
JOINT_GAUGE_COVARIANCE = "joint-cross-window"
GAUGE_COVARIANCE_SEMANTICS = frozenset(
    {MARGINAL_GAUGE_COVARIANCE, JOINT_GAUGE_COVARIANCE}
)
_REQUIRED_FACTOR_ARRAYS = frozenset(
    {
        "point_ids",
        "points_local_m",
        "valid_mask",
        "local_covariance_m2",
        "association_probability",
        "prior_reliability",
    }
)
_TOP_LEVEL_FIELDS_V3 = frozenset(
    {
        "schema",
        "schema_version",
        "gauge_parameterization",
        "sequence_id",
        "case_id",
        "stream_id",
        "source_repository",
        "source_revision",
        "causal_frame_stop",
        "causal_frame_stop_convention",
        "metadata",
        "payload",
        "gauges",
        "factors",
    }
)
_TOP_LEVEL_FIELDS_V4 = _TOP_LEVEL_FIELDS_V3 | {"gauge_covariance"}
_PAYLOAD_FIELDS = frozenset({"path", "sha256", "allow_pickle"})
_GAUGE_FIELDS = frozenset({"gauge_id", "mean_key", "covariance_key"})
_FACTOR_FIELDS = frozenset(
    {
        "factor_id",
        "frame_index",
        "view_id",
        "window_id",
        "gauge_id",
        "correlation_group_id",
        "causal_frame_stop",
        "prior_nominal_probability",
        "composite_weight",
        "arrays",
        "ray_directions_local_key",
    }
)
_GAUGE_COVARIANCE_FIELDS = frozenset(
    {
        "semantics",
        "joint_covariance_key",
        "ordered_gauge_ids",
        "cross_window_covariance_preserved",
        "diagonal_blocks_match_gauge_marginals",
    }
)


def file_sha256(path: str | Path) -> str:
    """Hash one ordinary artifact without following any path symlink."""

    return read_regular_file_no_symlinks(
        path,
        name="observation-factor artifact",
    ).sha256


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - value.keys())
    extra = sorted(value.keys() - expected)
    if missing or extra:
        raise ValueError(f"{name} fields changed; missing={missing}, extra={extra}")


def _validate_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _schema_version(value: Any) -> int:
    if type(value) is not int:
        raise ValueError("observation-factor schema version must be an integer")
    if value not in SUPPORTED_OBSERVATION_FACTOR_SCHEMA_VERSIONS:
        raise ValueError("unsupported observation-factor schema version")
    return value


def compute_observation_factor_bundle_id(
    manifest_sha256: str,
    payload_sha256: str,
    *,
    schema_version: int = PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION,
) -> str:
    """Content-address the exact manifest and payload byte pair."""

    manifest_digest = _validate_sha256(manifest_sha256, name="manifest_sha256")
    payload_digest = _validate_sha256(payload_sha256, name="payload_sha256")
    version = _schema_version(schema_version)
    digest = hashlib.sha256()
    digest.update(f"{OBSERVATION_FACTOR_SCHEMA}\0".encode("utf-8"))
    digest.update(str(version).encode("ascii"))
    digest.update(b"\0")
    digest.update(manifest_digest.encode("ascii"))
    digest.update(b"\0")
    digest.update(payload_digest.encode("ascii"))
    return digest.hexdigest()


def _nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _nonnegative_integer(value: Any, *, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    result = _nonnegative_integer(value, name=name)
    if result < 1:
        raise ValueError(f"{name} must be positive")
    return result


def _bounded_probability(value: Any, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a JSON number")
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _composite_weight(value: Any) -> float:
    result = _bounded_probability(value, name="factor composite_weight")
    if result <= 0.0:
        raise ValueError("factor composite_weight must lie in (0, 1]")
    return result


def _require_float64_array(values: np.ndarray, *, name: str) -> np.ndarray:
    result = np.asarray(values)
    if result.dtype != np.dtype(np.float64):
        raise ValueError(f"{name} must use float64")
    return result


def _probability_vector(values: np.ndarray, *, name: str) -> np.ndarray:
    result = _require_float64_array(values, name=name)
    if not np.all(np.isfinite(result)) or np.any((result < 0.0) | (result > 1.0)):
        raise ValueError(f"{name} must lie in [0, 1]")
    return result


def _require_psd(
    values: np.ndarray,
    *,
    name: str,
    tolerance: float = 1e-12,
) -> None:
    symmetric = 0.5 * (values + np.swapaxes(values, -1, -2))
    if not np.allclose(values, symmetric, atol=1e-12, rtol=1e-10):
        raise ValueError(f"{name} must be symmetric")
    eigenvalues = np.linalg.eigvalsh(symmetric)
    if np.any(eigenvalues < -tolerance):
        raise ValueError(f"{name} must be positive semidefinite")


def _block_diagonal(values: list[np.ndarray]) -> np.ndarray:
    dimension = sum(value.shape[0] for value in values)
    result = np.zeros((dimension, dimension), dtype=np.float64)
    offset = 0
    for value in values:
        width = value.shape[0]
        result[offset : offset + width, offset : offset + width] = value
        offset += width
    return result


@dataclass(frozen=True)
class ObservationFactorLineage:
    """Immutable summary of a validated Prob4D schema-v3/v4 factor bundle."""

    artifact_id: str
    manifest_sha256: str
    payload_sha256: str
    case_id: str
    stream_id: str
    sequence_id: str
    causal_frame_stop: int
    minimum_frame_id: int
    maximum_frame_id: int
    factor_count: int
    observation_count: int
    active_observation_count: int
    gauge_count: int
    correlation_group_count: int
    source_repository: str
    source_revision: str
    schema_version: int = PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION
    gauge_covariance_semantics: str = MARGINAL_GAUGE_COVARIANCE
    cross_window_gauge_covariance_preserved: bool = False

    def __post_init__(self) -> None:
        _validate_sha256(self.artifact_id, name="artifact_id")
        _validate_sha256(self.manifest_sha256, name="manifest_sha256")
        _validate_sha256(self.payload_sha256, name="payload_sha256")
        version = _schema_version(self.schema_version)
        for name, value in (
            ("case_id", self.case_id),
            ("stream_id", self.stream_id),
            ("sequence_id", self.sequence_id),
            ("source_repository", self.source_repository),
            ("source_revision", self.source_revision),
        ):
            _nonempty_string(value, name=name)
        causal_stop = _positive_integer(
            self.causal_frame_stop,
            name="factor-bundle causal frame stop",
        )
        minimum_frame = _nonnegative_integer(
            self.minimum_frame_id,
            name="factor-bundle minimum frame",
        )
        maximum_frame = _nonnegative_integer(
            self.maximum_frame_id,
            name="factor-bundle maximum frame",
        )
        if minimum_frame > maximum_frame:
            raise ValueError("factor-bundle frame range is invalid")
        if maximum_frame >= causal_stop:
            raise ValueError("factor-bundle lineage crosses its causal stop")
        for name, value in (
            ("factor_count", self.factor_count),
            ("observation_count", self.observation_count),
            ("active_observation_count", self.active_observation_count),
            ("gauge_count", self.gauge_count),
            ("correlation_group_count", self.correlation_group_count),
        ):
            _positive_integer(value, name=name)
        if self.active_observation_count > self.observation_count:
            raise ValueError("active observation count exceeds total observations")
        semantics = _nonempty_string(
            self.gauge_covariance_semantics,
            name="gauge_covariance_semantics",
        )
        if semantics not in GAUGE_COVARIANCE_SEMANTICS:
            raise ValueError("unsupported gauge covariance semantics")
        if type(self.cross_window_gauge_covariance_preserved) is not bool:
            raise ValueError("cross-window gauge covariance flag must be Boolean")
        preserved = self.cross_window_gauge_covariance_preserved
        if preserved != (semantics == JOINT_GAUGE_COVARIANCE):
            raise ValueError(
                "gauge covariance semantics contradict the preservation flag"
            )
        if version == PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION and preserved:
            raise ValueError("schema-v3 lineage cannot preserve joint gauge covariance")
        object.__setattr__(self, "schema_version", version)
        object.__setattr__(self, "causal_frame_stop", causal_stop)
        object.__setattr__(self, "minimum_frame_id", minimum_frame)
        object.__setattr__(self, "maximum_frame_id", maximum_frame)
        object.__setattr__(self, "gauge_covariance_semantics", semantics)

    def metadata(self) -> dict[str, Any]:
        result = {
            "source_observation_factor_bundle_id": self.artifact_id,
            "source_observation_factor_schema": OBSERVATION_FACTOR_SCHEMA,
            "source_observation_factor_schema_version": self.schema_version,
            "source_observation_factor_case_id": self.case_id,
            "source_observation_factor_stream_id": self.stream_id,
            "source_observation_factor_sequence_id": self.sequence_id,
            "source_observation_factor_causal_frame_stop": self.causal_frame_stop,
            "source_observation_factor_repository": self.source_repository,
            "source_observation_factor_revision": self.source_revision,
            "source_observation_factor_manifest_sha256": self.manifest_sha256,
            "source_observation_factor_payload_sha256": self.payload_sha256,
        }
        if self.schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION:
            result.update(
                {
                    "source_observation_factor_gauge_covariance_semantics": (
                        self.gauge_covariance_semantics
                    ),
                    "source_observation_factor_cross_window_gauge_covariance_preserved": (
                        self.cross_window_gauge_covariance_preserved
                    ),
                }
            )
        return result


def load_observation_factor_lineage(
    manifest_path: str | Path,
) -> ObservationFactorLineage:
    """Validate a Prob4D schema-v3/v4 bundle without importing its producer."""

    manifest_snapshot = read_regular_file_no_symlinks(
        manifest_path,
        name="factor-bundle manifest",
    )
    manifest = manifest_snapshot.path
    manifest_sha = manifest_snapshot.sha256
    record = load_strict_json_object(
        manifest_snapshot.payload,
        name="factor-bundle manifest",
    )
    if record.get("schema") != OBSERVATION_FACTOR_SCHEMA:
        raise ValueError("unsupported observation-factor schema")
    schema_version = _schema_version(record.get("schema_version"))
    _require_exact_fields(
        record,
        _TOP_LEVEL_FIELDS_V4
        if schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION
        else _TOP_LEVEL_FIELDS_V3,
        name="factor-bundle manifest",
    )
    if record.get("gauge_parameterization") != GAUGE_PARAMETERIZATION:
        raise ValueError("unsupported observation-factor gauge parameterization")
    if record.get("causal_frame_stop_convention") != "exclusive":
        raise ValueError("factor-bundle causal frame stop must be exclusive")

    sequence_id = _nonempty_string(record["sequence_id"], name="sequence_id")
    case_id = _nonempty_string(record["case_id"], name="case_id")
    stream_id = _nonempty_string(record["stream_id"], name="stream_id")
    source_repository = _nonempty_string(
        record["source_repository"],
        name="source_repository",
    )
    source_revision = _nonempty_string(
        record["source_revision"],
        name="source_revision",
    )
    causal_stop = _positive_integer(
        record["causal_frame_stop"],
        name="factor-bundle causal frame stop",
    )
    _require_mapping(record["metadata"], name="factor-bundle metadata")

    payload_record = _require_mapping(
        record["payload"],
        name="factor-bundle payload descriptor",
    )
    _require_exact_fields(payload_record, _PAYLOAD_FIELDS, name="payload descriptor")
    if payload_record["allow_pickle"] is not False:
        raise ValueError("factor-bundle payload must disable pickle")
    declared_payload_sha = _validate_sha256(
        payload_record["sha256"],
        name="payload sha256",
    )
    payload_snapshot = read_regular_file_beneath(
        manifest.parent,
        payload_record["path"],
        name="factor-bundle payload",
    )
    actual_payload_sha = payload_snapshot.sha256
    if actual_payload_sha != declared_payload_sha:
        raise ValueError("factor-bundle payload checksum mismatch")
    payload_archive = load_numpy_archive_snapshot(
        payload_snapshot,
        expected_sha256=declared_payload_sha,
        name="factor-bundle payload",
    )

    gauge_records = record["gauges"]
    factor_records = record["factors"]
    if type(gauge_records) is not list or not gauge_records:
        raise ValueError("factor bundle must contain gauge records")
    if type(factor_records) is not list or not factor_records:
        raise ValueError("factor bundle must contain factor records")

    expected_array_names: list[str] = []
    gauge_ids: list[str] = []
    gauge_covariances: list[np.ndarray] = []
    factor_ids: set[str] = set()
    group_parameters: dict[str, tuple[float, float]] = {}
    frame_ids: list[int] = []
    total_observations = 0
    active_observations = 0
    covariance_semantics = MARGINAL_GAUGE_COVARIANCE
    cross_window_preserved = False

    with nullcontext(payload_archive.arrays) as arrays:
        for position, raw_gauge_record in enumerate(gauge_records):
            gauge_record = _require_mapping(
                raw_gauge_record,
                name=f"gauge record {position}",
            )
            _require_exact_fields(gauge_record, _GAUGE_FIELDS, name=f"gauge {position}")
            gauge_id = _nonempty_string(
                gauge_record["gauge_id"],
                name=f"gauge {position} ID",
            )
            if gauge_id in gauge_ids:
                raise ValueError("factor-bundle gauge IDs must be unique")
            gauge_ids.append(gauge_id)
            mean_key = _nonempty_string(
                gauge_record["mean_key"],
                name=f"gauge {position} mean key",
            )
            covariance_key = _nonempty_string(
                gauge_record["covariance_key"],
                name=f"gauge {position} covariance key",
            )
            expected_array_names.extend((mean_key, covariance_key))
            if mean_key not in arrays or covariance_key not in arrays:
                raise ValueError("factor-bundle gauge payload arrays are missing")
            mean = _require_float64_array(
                arrays[mean_key],
                name="factor-bundle gauge mean",
            )
            covariance = _require_float64_array(
                arrays[covariance_key],
                name="factor-bundle gauge covariance",
            )
            if mean.shape != (7,) or not np.all(np.isfinite(mean)):
                raise ValueError("factor-bundle gauge mean must have shape (7,)")
            if covariance.shape != (7, 7) or not np.all(np.isfinite(covariance)):
                raise ValueError(
                    "factor-bundle gauge covariance must have shape (7, 7)"
                )
            _require_psd(covariance, name="factor-bundle gauge covariance")
            gauge_covariances.append(covariance)

        if schema_version == OBSERVATION_FACTOR_SCHEMA_VERSION:
            covariance_record = _require_mapping(
                record["gauge_covariance"],
                name="gauge_covariance",
            )
            _require_exact_fields(
                covariance_record,
                _GAUGE_COVARIANCE_FIELDS,
                name="gauge_covariance",
            )
            covariance_semantics = _nonempty_string(
                covariance_record["semantics"],
                name="gauge covariance semantics",
            )
            if covariance_semantics not in GAUGE_COVARIANCE_SEMANTICS:
                raise ValueError("unsupported gauge covariance semantics")
            preserved = covariance_record["cross_window_covariance_preserved"]
            if type(preserved) is not bool:
                raise ValueError(
                    "schema-v4 cross_window_covariance_preserved must be Boolean"
                )
            if preserved != (covariance_semantics == JOINT_GAUGE_COVARIANCE):
                raise ValueError(
                    "schema-v4 gauge covariance semantics contradict the preservation flag"
                )
            if covariance_record["diagonal_blocks_match_gauge_marginals"] is not True:
                raise ValueError(
                    "schema-v4 gauge covariance must declare matching marginal blocks"
                )
            ordered_gauge_ids = covariance_record["ordered_gauge_ids"]
            if type(ordered_gauge_ids) is not list or any(
                type(value) is not str or not value for value in ordered_gauge_ids
            ):
                raise ValueError("schema-v4 ordered_gauge_ids must contain strings")
            if ordered_gauge_ids != gauge_ids:
                raise ValueError(
                    "schema-v4 joint gauge covariance order differs from gauge records"
                )
            joint_key = _nonempty_string(
                covariance_record["joint_covariance_key"],
                name="schema-v4 joint_covariance_key",
            )
            expected_array_names.append(joint_key)
            if joint_key not in arrays:
                raise ValueError("schema-v4 joint gauge covariance array is missing")
            joint_covariance = _require_float64_array(
                arrays[joint_key],
                name="schema-v4 joint gauge covariance",
            )
            dimension = 7 * len(gauge_ids)
            if joint_covariance.shape != (dimension, dimension):
                raise ValueError("schema-v4 joint gauge covariance has the wrong shape")
            if not np.all(np.isfinite(joint_covariance)):
                raise ValueError("schema-v4 joint gauge covariance must be finite")
            _require_psd(
                joint_covariance,
                name="schema-v4 joint gauge covariance",
            )
            for index, gauge_covariance in enumerate(gauge_covariances):
                block = joint_covariance[
                    7 * index : 7 * (index + 1),
                    7 * index : 7 * (index + 1),
                ]
                if not np.allclose(
                    block,
                    gauge_covariance,
                    atol=1e-12,
                    rtol=1e-10,
                ):
                    raise ValueError(
                        "schema-v4 joint gauge covariance diagonal blocks differ "
                        "from gauge marginals"
                    )
            if covariance_semantics == MARGINAL_GAUGE_COVARIANCE and not np.allclose(
                joint_covariance,
                _block_diagonal(gauge_covariances),
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(
                    "marginal-blocks-only semantics require zero cross-window covariance"
                )
            cross_window_preserved = preserved

        for position, raw_factor_record in enumerate(factor_records):
            factor_record = _require_mapping(
                raw_factor_record,
                name=f"factor record {position}",
            )
            _require_exact_fields(
                factor_record, _FACTOR_FIELDS, name=f"factor {position}"
            )
            identifiers = {
                name: _nonempty_string(
                    factor_record[name],
                    name=f"factor {position} {name}",
                )
                for name in (
                    "factor_id",
                    "view_id",
                    "window_id",
                    "gauge_id",
                    "correlation_group_id",
                )
            }
            factor_id = identifiers["factor_id"]
            if factor_id in factor_ids:
                raise ValueError("factor-bundle factor IDs must be unique")
            factor_ids.add(factor_id)
            if identifiers["gauge_id"] not in gauge_ids:
                raise ValueError("factor references an unavailable gauge")
            frame_index = _nonnegative_integer(
                factor_record["frame_index"],
                name="factor frame_index",
            )
            factor_stop = _positive_integer(
                factor_record["causal_frame_stop"],
                name="factor causal_frame_stop",
            )
            if frame_index >= causal_stop:
                raise ValueError("factor frame crosses the bundle causal stop")
            if factor_stop != causal_stop:
                raise ValueError("factor and bundle causal frame stops differ")
            frame_ids.append(frame_index)

            nominal_probability = _bounded_probability(
                factor_record["prior_nominal_probability"],
                name="factor prior_nominal_probability",
            )
            composite_weight = _composite_weight(factor_record["composite_weight"])
            group_id = identifiers["correlation_group_id"]
            parameters = (nominal_probability, composite_weight)
            previous = group_parameters.setdefault(group_id, parameters)
            if previous != parameters:
                raise ValueError(
                    "one correlation group has inconsistent factor metadata"
                )

            key_record = _require_mapping(
                factor_record["arrays"],
                name="factor payload array mapping",
            )
            _require_exact_fields(
                key_record,
                _REQUIRED_FACTOR_ARRAYS,
                name="factor payload array mapping",
            )
            factor_keys = {
                name: _nonempty_string(
                    key_record[name],
                    name=f"factor {position} {name} key",
                )
                for name in _REQUIRED_FACTOR_ARRAYS
            }
            expected_array_names.extend(factor_keys.values())
            ray_key_value = factor_record["ray_directions_local_key"]
            ray_key = None
            if ray_key_value is not None:
                ray_key = _nonempty_string(
                    ray_key_value,
                    name=f"factor {position} ray key",
                )
                expected_array_names.append(ray_key)
            if any(key not in arrays for key in factor_keys.values()):
                raise ValueError("factor payload arrays are missing")
            if ray_key is not None and ray_key not in arrays:
                raise ValueError("factor ray payload array is missing")

            point_ids = np.asarray(arrays[factor_keys["point_ids"]])
            points = _require_float64_array(
                arrays[factor_keys["points_local_m"]],
                name="factor points",
            )
            valid = np.asarray(arrays[factor_keys["valid_mask"]])
            covariance = _require_float64_array(
                arrays[factor_keys["local_covariance_m2"]],
                name="factor local covariance",
            )
            association = _probability_vector(
                arrays[factor_keys["association_probability"]],
                name="factor association probability",
            )
            reliability = _probability_vector(
                arrays[factor_keys["prior_reliability"]],
                name="factor prior reliability",
            )
            if point_ids.dtype != np.dtype(np.int64):
                raise ValueError("factor point IDs must use int64")
            if point_ids.ndim != 1 or len(point_ids) == 0:
                raise ValueError("factor point IDs must be a nonempty vector")
            if np.any(point_ids < 0) or len(np.unique(point_ids)) != len(point_ids):
                raise ValueError("factor point IDs must be nonnegative and unique")
            count = len(point_ids)
            if points.shape != (count, 3):
                raise ValueError("factor points must have shape (N, 3)")
            if valid.dtype != np.dtype(bool) or valid.shape != (count,):
                raise ValueError("factor valid mask must be a Boolean vector")
            if covariance.shape != (count, 3, 3):
                raise ValueError("factor local covariance must have shape (N, 3, 3)")
            if association.shape != (count,) or reliability.shape != (count,):
                raise ValueError("factor probability vectors must identify every point")
            active = valid & (association > 0.0) & (reliability > 0.0)
            if not np.all(np.isfinite(points[active])):
                raise ValueError("active factor points must be finite")
            if not np.all(np.isfinite(covariance[active])):
                raise ValueError("active factor covariance must be finite")
            if np.any(active):
                _require_psd(
                    covariance[active],
                    name="active factor covariance",
                )
            if ray_key is not None:
                rays = _require_float64_array(
                    arrays[ray_key],
                    name="factor rays",
                )
                if rays.shape != (count, 3):
                    raise ValueError("factor rays must have shape (N, 3)")
                if not np.all(np.isfinite(rays[active])):
                    raise ValueError("active factor rays must be finite")
                if np.any(
                    active & (np.linalg.norm(rays, axis=1) <= np.finfo(np.float64).eps)
                ):
                    raise ValueError("active factor rays must be nonzero")
            total_observations += count
            active_observations += int(np.sum(active))

        if len(expected_array_names) != len(set(expected_array_names)):
            raise ValueError("factor-bundle payload array keys are reused")
        expected_arrays = set(expected_array_names)
        actual_arrays = set(arrays)
        missing_arrays = expected_arrays - actual_arrays
        extra_arrays = actual_arrays - expected_arrays
        if missing_arrays or extra_arrays:
            raise ValueError(
                "factor-bundle payload array set changed; "
                f"missing={sorted(missing_arrays)}, extra={sorted(extra_arrays)}"
            )

    if active_observations < 1:
        raise ValueError("factor bundle has no active observation rows")
    artifact_id = compute_observation_factor_bundle_id(
        manifest_sha,
        actual_payload_sha,
        schema_version=schema_version,
    )
    return ObservationFactorLineage(
        artifact_id=artifact_id,
        manifest_sha256=manifest_sha,
        payload_sha256=actual_payload_sha,
        case_id=case_id,
        stream_id=stream_id,
        sequence_id=sequence_id,
        causal_frame_stop=causal_stop,
        minimum_frame_id=min(frame_ids),
        maximum_frame_id=max(frame_ids),
        factor_count=len(factor_records),
        observation_count=total_observations,
        active_observation_count=active_observations,
        gauge_count=len(gauge_records),
        correlation_group_count=len(group_parameters),
        source_repository=source_repository,
        source_revision=source_revision,
        schema_version=schema_version,
        gauge_covariance_semantics=covariance_semantics,
        cross_window_gauge_covariance_preserved=cross_window_preserved,
    )


def validate_twin_belief_observation_factor_lineage(
    twin_belief: TwinBelief,
    lineage: ObservationFactorLineage,
    *,
    require_bound: bool = True,
) -> dict[str, Any]:
    """Check case, O-minus containment, and exact factor-bundle binding."""

    if twin_belief.context.case_id != lineage.case_id:
        raise ValueError("factor bundle and TwinBelief identify different cases")
    if lineage.minimum_frame_id < twin_belief.context.o_minus.frame_start:
        raise ValueError("factor bundle begins before the TwinBelief O- boundary")
    if lineage.causal_frame_stop > twin_belief.context.o_minus.frame_stop:
        raise ValueError("factor bundle extends beyond the TwinBelief O- boundary")

    expected_metadata = lineage.metadata()
    metadata = twin_belief.metadata
    bound_id = metadata.get("source_observation_factor_bundle_id")
    if bound_id is not None and bound_id != lineage.artifact_id:
        raise ValueError("TwinBelief is bound to a different factor bundle")
    if require_bound and bound_id is None:
        raise ValueError("TwinBelief has no source factor-bundle binding")
    if bound_id is not None:
        for key, expected in expected_metadata.items():
            actual = metadata.get(key)
            if actual != expected:
                raise ValueError(
                    f"TwinBelief factor-bundle metadata mismatch for {key}"
                )
    return {
        "status": "valid",
        "twin_belief_id": twin_belief.artifact_id,
        "observation_factor_bundle_id": lineage.artifact_id,
        "observation_factor_schema_version": lineage.schema_version,
        "gauge_covariance_semantics": lineage.gauge_covariance_semantics,
        "cross_window_gauge_covariance_preserved": (
            lineage.cross_window_gauge_covariance_preserved
        ),
        "case_id": lineage.case_id,
        "stream_id": lineage.stream_id,
        "sequence_id": lineage.sequence_id,
        "lineage_bound": bound_id == lineage.artifact_id,
        "observation_frame_range": [
            lineage.minimum_frame_id,
            lineage.maximum_frame_id,
        ],
        "observation_causal_frame_stop": lineage.causal_frame_stop,
        "twin_o_minus_frame_range": [
            twin_belief.context.o_minus.frame_start,
            twin_belief.context.o_minus.frame_stop,
        ],
        "factor_count": lineage.factor_count,
        "observation_count": lineage.observation_count,
        "active_observation_count": lineage.active_observation_count,
        "gauge_count": lineage.gauge_count,
        "correlation_group_count": lineage.correlation_group_count,
        "manifest_sha256": lineage.manifest_sha256,
        "payload_sha256": lineage.payload_sha256,
    }


def bind_twin_belief_observation_factor_lineage(
    twin_belief: TwinBelief,
    lineage: ObservationFactorLineage,
) -> TwinBelief:
    """Bind the exact factor bundle actually consumed by the estimator."""

    validate_twin_belief_observation_factor_lineage(
        twin_belief,
        lineage,
        require_bound=False,
    )
    metadata = dict(twin_belief.metadata)
    existing = metadata.get("source_observation_factor_bundle_id")
    if existing is not None and existing != lineage.artifact_id:
        raise ValueError("TwinBelief already has incompatible factor lineage")
    metadata.update(lineage.metadata())
    return replace(twin_belief, metadata=metadata)


__all__ = [
    "GAUGE_COVARIANCE_SEMANTICS",
    "GAUGE_PARAMETERIZATION",
    "JOINT_GAUGE_COVARIANCE",
    "MARGINAL_GAUGE_COVARIANCE",
    "OBSERVATION_FACTOR_SCHEMA",
    "OBSERVATION_FACTOR_SCHEMA_VERSION",
    "PREVIOUS_OBSERVATION_FACTOR_SCHEMA_VERSION",
    "SUPPORTED_OBSERVATION_FACTOR_SCHEMA_VERSIONS",
    "ObservationFactorLineage",
    "bind_twin_belief_observation_factor_lineage",
    "compute_observation_factor_bundle_id",
    "file_sha256",
    "load_observation_factor_lineage",
    "validate_twin_belief_observation_factor_lineage",
]
