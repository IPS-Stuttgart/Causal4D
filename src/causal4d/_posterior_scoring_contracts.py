"""Content-addressed specifications and shared posterior-score helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Sequence

import numpy as np


POSTERIOR_SCORE_SCHEMA_VERSION = 1
POSTERIOR_SCORE_CLAIM_BOUNDARY = (
    "These scores evaluate an already-produced posterior. They do not change the "
    "frozen estimator, admit new evidence, establish calibration on physical target "
    "executions, or authorize model selection."
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _canonical_id(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(values))
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(_canonical_bytes(list(array.shape)))
    digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _readonly_array(values: Any, *, dtype: Any) -> np.ndarray:
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def _require_nonempty_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    result = _require_nonempty_string(value, name=name)
    if len(result) != 64 or any(
        character not in "0123456789abcdef" for character in result
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return result


def _require_finite_nonnegative(value: Any, *, name: str) -> float:
    if type(value) not in {int, float}:
        raise ValueError(f"{name} must be a finite nonnegative number")
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _validated_weights(values: Any, *, expected_count: int | None = None) -> np.ndarray:
    weights = _readonly_array(values, dtype=float)
    if weights.ndim != 1 or len(weights) == 0:
        raise ValueError("posterior weights must be a nonempty vector")
    if expected_count is not None and len(weights) != expected_count:
        raise ValueError("posterior weights do not match the component count")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("posterior weights must be finite and nonnegative")
    if not np.isclose(np.sum(weights), 1.0, atol=1e-10, rtol=1e-10):
        raise ValueError("posterior weights must sum to one")
    return weights


def _coordinate_mask(mask: np.ndarray, trajectory_shape: tuple[int, ...]) -> np.ndarray:
    supplied = np.asarray(mask, dtype=bool)
    if supplied.shape == trajectory_shape:
        result = supplied.copy()
    elif supplied.shape == trajectory_shape[:-1]:
        result = np.repeat(supplied[..., None], trajectory_shape[-1], axis=-1)
    else:
        raise ValueError(
            "valid_mask must match the trajectory or omit only its coordinate axis"
        )
    if not np.any(result):
        raise ValueError("valid_mask must select at least one coordinate")
    return result


def _as_component_matrix(values: Any, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim < 2 or len(array) == 0:
        raise ValueError(f"{name} must have shape (K, ...)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array.reshape(len(array), -1)


def trajectory_coordinate_index(
    frame: int,
    node: int,
    coordinate: int,
    trajectory_shape: Sequence[int],
) -> int:
    """Return the canonical flattened index for one ``(T, N, 3)`` coordinate."""

    shape = tuple(trajectory_shape)
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError("trajectory_shape must be (T, N, 3)")
    values = (frame, node, coordinate)
    names = ("frame", "node", "coordinate")
    for value, bound, name in zip(values, shape, names, strict=True):
        if type(value) is not int or not 0 <= value < bound:
            raise ValueError(f"{name} must be an integer inside trajectory_shape")
    return int(np.ravel_multi_index(values, shape))


@dataclass(frozen=True)
class TrajectoryScoreSpecificationV1:
    """Content-addressed selection, variogram, and query definition."""

    name: str
    valid_mask: np.ndarray
    variogram_pairs: np.ndarray = field(
        default_factory=lambda: np.empty((0, 2), dtype=np.int64)
    )
    variogram_pair_weights: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=float)
    )
    variogram_order: float = 0.5
    query_matrix: np.ndarray | None = None
    query_labels: tuple[str, ...] = ()
    query_units: tuple[str, ...] = ()
    query_covariance_floor_m2: float = 1.0e-12

    def __post_init__(self) -> None:
        name = _require_nonempty_string(self.name, name="name")
        raw_valid_mask = np.asarray(self.valid_mask)
        if raw_valid_mask.dtype.kind != "b":
            raise ValueError("valid_mask must contain exact Boolean values")
        valid_mask = _readonly_array(raw_valid_mask, dtype=bool)
        if valid_mask.ndim < 1 or not np.any(valid_mask):
            raise ValueError("valid_mask must be nonempty and select coordinates")

        raw_pairs = np.asarray(self.variogram_pairs)
        if raw_pairs.dtype.kind not in {"i", "u"}:
            raise ValueError("variogram_pairs must contain exact integer indices")
        pairs = _readonly_array(raw_pairs, dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("variogram_pairs must have shape (P, 2)")
        if np.any(pairs < 0) or np.any(pairs[:, 0] >= pairs[:, 1]):
            raise ValueError(
                "variogram pairs must be canonical distinct indices with left < right"
            )
        if len({tuple(row) for row in pairs.tolist()}) != len(pairs):
            raise ValueError("variogram pairs must be unique")

        pair_weights = _readonly_array(self.variogram_pair_weights, dtype=float)
        if pair_weights.shape != (len(pairs),):
            raise ValueError("variogram_pair_weights must match variogram_pairs")
        if len(pairs):
            if not np.all(np.isfinite(pair_weights)) or np.any(pair_weights < 0.0):
                raise ValueError(
                    "variogram pair weights must be finite and nonnegative"
                )
            if not np.isclose(
                np.sum(pair_weights),
                1.0,
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError("variogram pair weights must sum to one")
        elif len(pair_weights):
            raise ValueError("empty variogram pairs require empty pair weights")

        order = float(self.variogram_order)
        if not np.isfinite(order) or not 0.0 < order <= 2.0:
            raise ValueError("variogram_order must lie in (0, 2]")

        query = None
        if isinstance(self.query_labels, (str, bytes)):
            raise ValueError("query_labels must be a sequence of labels")
        if isinstance(self.query_units, (str, bytes)):
            raise ValueError("query_units must be a sequence of units")
        labels = tuple(self.query_labels)
        units = tuple(self.query_units)
        if self.query_matrix is None:
            if labels or units:
                raise ValueError("query labels and units require a query_matrix")
        else:
            query = _readonly_array(self.query_matrix, dtype=float)
            if query.ndim != 2 or query.shape[0] == 0 or query.shape[1] == 0:
                raise ValueError("query_matrix must have nonempty shape (Q, D)")
            if not np.all(np.isfinite(query)):
                raise ValueError("query_matrix must be finite")
            if np.any(np.linalg.norm(query, axis=1) == 0.0):
                raise ValueError("query_matrix rows must be nonzero")
            if len(labels) != query.shape[0]:
                raise ValueError("query_labels must identify every query row")
            if not units:
                units = ("m",) * query.shape[0]
            if len(units) != query.shape[0]:
                raise ValueError("query_units must identify every query row")
            if len(set(labels)) != len(labels):
                raise ValueError("query_labels must be unique")
            for index, label in enumerate(labels):
                _require_nonempty_string(label, name=f"query_labels[{index}]")
            for index, unit in enumerate(units):
                if _require_nonempty_string(unit, name=f"query_units[{index}]") != "m":
                    raise ValueError("trajectory query units must be metres ('m')")

        floor = _require_finite_nonnegative(
            self.query_covariance_floor_m2,
            name="query_covariance_floor_m2",
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "valid_mask", valid_mask)
        object.__setattr__(self, "variogram_pairs", pairs)
        object.__setattr__(self, "variogram_pair_weights", pair_weights)
        object.__setattr__(self, "variogram_order", order)
        object.__setattr__(self, "query_matrix", query)
        object.__setattr__(self, "query_labels", labels)
        object.__setattr__(self, "query_units", units)
        object.__setattr__(self, "query_covariance_floor_m2", floor)

    def descriptor(self) -> dict[str, Any]:
        query_descriptor = None
        if self.query_matrix is not None:
            query_descriptor = {
                "shape": list(self.query_matrix.shape),
                "sha256": _array_sha256(self.query_matrix),
                "labels": list(self.query_labels),
                "units": list(self.query_units),
                "covariance_floor_m2": self.query_covariance_floor_m2,
            }
        return {
            "schema_version": POSTERIOR_SCORE_SCHEMA_VERSION,
            "artifact_kind": "TrajectoryScoreSpecification",
            "name": self.name,
            "valid_mask": {
                "shape": list(self.valid_mask.shape),
                "sha256": _array_sha256(self.valid_mask),
            },
            "variogram": {
                "pairs_shape": list(self.variogram_pairs.shape),
                "pairs_sha256": _array_sha256(self.variogram_pairs),
                "pair_weights_sha256": _array_sha256(
                    self.variogram_pair_weights
                ),
                "order": self.variogram_order,
                "score_unit_power_m": 2.0 * self.variogram_order,
            },
            "query": query_descriptor,
        }

    @property
    def specification_id(self) -> str:
        return _canonical_id(self.descriptor())

