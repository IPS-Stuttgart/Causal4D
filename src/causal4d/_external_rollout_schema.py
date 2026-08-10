"""Portable import boundary for external finite physical rollout banks."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.rollout_bank import JointRolloutBank

EXTERNAL_ROLLOUT_IMPORT_SCHEMA = "causal4d.external_rollout_import"
EXTERNAL_ROLLOUT_IMPORT_SCHEMA_VERSION = 1
EXTERNAL_ROLLOUT_BANK_SCHEMA = "causal4d.external_rollout_bank"
EXTERNAL_ROLLOUT_BANK_SCHEMA_VERSION = 1

_IMPORT_REQUIRED_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "case_id",
        "source",
        "arrays",
        "layout",
        "coordinate_frame",
        "position_unit",
        "anchor_time_s",
    }
)
_IMPORT_OPTIONAL_FIELDS = frozenset(
    {
        "parameter_names",
        "variance_floor_m2",
        "confidence_level",
        "metadata",
    }
)
_IMPORT_ARRAY_REQUIRED_FIELDS = frozenset(
    {"node_ids", "trajectories", "frame_times_s", "rollout_weights"}
)
_IMPORT_ARRAY_OPTIONAL_FIELDS = frozenset(
    {"rollout_ids", "parameter_values", "camera_to_world"}
)
_SOURCE_REQUIRED_FIELDS = frozenset({"simulator"})
_SOURCE_OPTIONAL_FIELDS = frozenset({"revision", "artifact_id"})
_BANK_DESCRIPTOR_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_id",
        "case_id",
        "source",
        "node_ids",
        "frame_times_s",
        "anchor_time_s",
        "anchor_frame_index",
        "parameter_names",
        "source_npz_sha256",
        "import_manifest_sha256",
        "bank_artifact_id",
        "metadata",
    }
)
_BANK_SOURCE_FIELDS = frozenset({"simulator", "revision", "artifact_id"})
_UNIT_SCALE_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
_COORDINATE_FRAMES = frozenset({"world", "camera"})
_SUPPORTED_LAYOUTS = frozenset({"RTNC"})


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - optional)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_optional_string(value: Any, *, name: str) -> str | None:
    if value is None:
        return None
    return _require_nonempty_string(value, name=name)


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_finite_number(value: Any, *, name: str) -> float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite JSON number")
    return float(value)


def _require_positive_number(value: Any, *, name: str) -> float:
    result = _require_finite_number(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _validated_string_tuple(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_nonempty_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _reject_nonfinite_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant {value!r} is not allowed")


def _load_json_mapping(path: str | Path, *, name: str) -> Mapping[str, Any]:
    try:
        parsed = json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_json_keys,
            parse_constant=_reject_nonfinite_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    return _require_mapping(parsed, name=name)


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _source_array(
    archive: np.lib.npyio.NpzFile,
    key: Any,
    *,
    name: str,
) -> np.ndarray:
    key_text = _require_nonempty_string(key, name=f"arrays.{name}")
    if key_text not in archive.files:
        raise ValueError(f"source NPZ does not contain array {key_text!r} for {name}")
    try:
        return np.asarray(archive[key_text])
    except ValueError as error:
        raise ValueError(
            f"source array {key_text!r} cannot be loaded without pickle"
        ) from error


def _text_vector(values: np.ndarray, *, name: str) -> tuple[str, ...]:
    array = np.asarray(values)
    if array.ndim != 1 or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a one-dimensional text array")
    result: list[str] = []
    for index, raw in enumerate(array):
        value: Any = raw.item() if isinstance(raw, np.generic) else raw
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except UnicodeDecodeError as error:
                raise ValueError(f"{name}[{index}] is not valid UTF-8") from error
        result.append(_require_nonempty_string(value, name=f"{name}[{index}]"))
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def _camera_to_world(points_m: np.ndarray, transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError("camera_to_world must have finite shape (4, 4)")
    if not np.allclose(matrix[3], np.asarray([0.0, 0.0, 0.0, 1.0])):
        raise ValueError("camera_to_world must be a homogeneous rigid transform")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6, rtol=1e-6):
        raise ValueError("camera_to_world rotation must be orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=1e-6):
        raise ValueError("camera_to_world rotation must have determinant one")
    flat = points_m.reshape(-1, 3)
    homogeneous = np.column_stack((flat, np.ones(len(flat))))
    transformed = homogeneous @ matrix.T
    return transformed[:, :3].reshape(points_m.shape)


@dataclass(frozen=True)
class ExternalRolloutBundle:
    """Canonical finite rollout support with explicit node and time identities."""

    bank: JointRolloutBank
    case_id: str
    source_simulator: str
    node_ids: np.ndarray
    frame_times_s: np.ndarray
    anchor_time_s: float
    source_npz_sha256: str
    import_manifest_sha256: str
    source_revision: str | None = None
    source_artifact_id: str | None = None
    parameter_names: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.bank, JointRolloutBank):
            raise TypeError("bank must be a JointRolloutBank")
        case_id = _require_nonempty_string(self.case_id, name="case_id")
        source_simulator = _require_nonempty_string(
            self.source_simulator,
            name="source_simulator",
        )
        source_revision = _require_optional_string(
            self.source_revision,
            name="source_revision",
        )
        source_artifact_id = _require_optional_string(
            self.source_artifact_id,
            name="source_artifact_id",
        )
        source_npz_sha256 = _validate_sha256(
            self.source_npz_sha256,
            name="source_npz_sha256",
        )
        import_manifest_sha256 = _validate_sha256(
            self.import_manifest_sha256,
            name="import_manifest_sha256",
        )
        nodes = readonly_integer_array(self.node_ids, name="node_ids")
        if nodes.ndim != 1 or len(nodes) != self.bank.node_count:
            raise ValueError("node_ids must identify every rollout-bank node")
        if np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes):
            raise ValueError("node_ids must be unique and nonnegative")
        times = readonly_array(self.frame_times_s, dtype=np.float64)
        if times.shape != (self.bank.frame_count,) or not np.all(np.isfinite(times)):
            raise ValueError("frame_times_s must be finite and match rollout frames")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("frame_times_s must be strictly increasing")
        anchor_time = _require_finite_number(self.anchor_time_s, name="anchor_time_s")
        matches = np.flatnonzero(np.isclose(times, anchor_time, atol=1e-12, rtol=0.0))
        if len(matches) != 1:
            raise ValueError("anchor_time_s must match exactly one rollout frame")
        parameter_names = _validated_string_tuple(
            self.parameter_names,
            name="parameter_names",
            allow_empty=True,
        )
        metadata = validated_json_mapping(
            self.metadata,
            error_message="external rollout metadata must be finite JSON data",
        )
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "source_simulator", source_simulator)
        object.__setattr__(self, "source_revision", source_revision)
        object.__setattr__(self, "source_artifact_id", source_artifact_id)
        object.__setattr__(self, "source_npz_sha256", source_npz_sha256)
        object.__setattr__(self, "import_manifest_sha256", import_manifest_sha256)
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "frame_times_s", times)
        object.__setattr__(self, "anchor_time_s", anchor_time)
        object.__setattr__(self, "parameter_names", parameter_names)
        object.__setattr__(self, "metadata", metadata)

    @property
    def anchor_frame_index(self) -> int:
        matches = np.flatnonzero(
            np.isclose(self.frame_times_s, self.anchor_time_s, atol=1e-12, rtol=0.0)
        )
        return int(matches[0])

    def _descriptor_without_id(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_ROLLOUT_BANK_SCHEMA,
            "schema_version": EXTERNAL_ROLLOUT_BANK_SCHEMA_VERSION,
            "case_id": self.case_id,
            "source": {
                "simulator": self.source_simulator,
                "revision": self.source_revision,
                "artifact_id": self.source_artifact_id,
            },
            "node_ids": self.node_ids.tolist(),
            "frame_times_s": self.frame_times_s.tolist(),
            "anchor_time_s": self.anchor_time_s,
            "anchor_frame_index": self.anchor_frame_index,
            "parameter_names": list(self.parameter_names),
            "source_npz_sha256": self.source_npz_sha256,
            "import_manifest_sha256": self.import_manifest_sha256,
            "bank_artifact_id": self.bank.artifact_id,
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        encoded = _canonical_json(self._descriptor_without_id()).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        descriptor = self._descriptor_without_id()
        descriptor["artifact_id"] = self.artifact_id
        return descriptor

    def summary(self) -> dict[str, Any]:
        return {
            "artifact_id": self.artifact_id,
            "bank_artifact_id": self.bank.artifact_id,
            "case_id": self.case_id,
            "source_simulator": self.source_simulator,
            "source_revision": self.source_revision,
            "rollout_count": len(self.bank.hypothesis_ids),
            "frame_count": self.bank.frame_count,
            "node_count": self.bank.node_count,
            "coordinate_count": self.bank.coordinate_count,
            "anchor_time_s": self.anchor_time_s,
            "anchor_frame_index": self.anchor_frame_index,
            "frame_time_start_s": float(self.frame_times_s[0]),
            "frame_time_stop_s": float(self.frame_times_s[-1]),
            "parameter_names": list(self.parameter_names),
        }
