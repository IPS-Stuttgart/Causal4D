"""Strict evaluation-only reference trajectories for external bridge studies."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_array, readonly_integer_array
from causal4d.numpy_archive import load_numpy_archive

EXTERNAL_REFERENCE_SCHEMA = "causal4d.external_reference_trajectory"
EXTERNAL_REFERENCE_SCHEMA_VERSION = 1

_REQUIRED_MEMBERS = frozenset(
    {"case_id", "node_ids", "positions_world_m", "frame_times_s"}
)
_OPTIONAL_MEMBERS = frozenset({"validity_mask"})


def _scalar_text(value: np.ndarray, *, name: str) -> str:
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in {"U", "S"}:
        raise ValueError(f"{name} must be a scalar text member")
    item: Any = array.item()
    if isinstance(item, bytes):
        try:
            item = item.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(f"{name} must be valid UTF-8") from error
    if type(item) is not str or not item:
        raise ValueError(f"{name} must be a nonempty string")
    return item


@dataclass(frozen=True)
class ExternalReferenceTrajectory:
    """A content-addressed trajectory used only for evaluation and baselines."""

    case_id: str
    node_ids: np.ndarray
    positions_m: np.ndarray
    frame_times_s: np.ndarray
    validity: np.ndarray
    source_npz_sha256: str

    def __post_init__(self) -> None:
        if type(self.case_id) is not str or not self.case_id:
            raise ValueError("reference case_id must be a nonempty string")
        nodes = readonly_integer_array(self.node_ids, name="reference node_ids")
        if nodes.ndim != 1 or not len(nodes):
            raise ValueError("reference node_ids must be a nonempty vector")
        if np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes):
            raise ValueError("reference node_ids must be unique and nonnegative")
        positions = np.asarray(self.positions_m, dtype=np.float64).copy()
        if positions.ndim != 3 or positions.shape[1:] != (len(nodes), 3):
            raise ValueError("reference positions must have shape (T, N, 3)")
        if positions.shape[0] < 2:
            raise ValueError("reference trajectory must contain at least two frames")
        times = readonly_array(self.frame_times_s, dtype=np.float64)
        if times.shape != (positions.shape[0],) or not np.all(np.isfinite(times)):
            raise ValueError("reference frame_times_s must be finite and match T")
        if np.any(np.diff(times) <= 0.0):
            raise ValueError("reference frame_times_s must be strictly increasing")
        valid = np.asarray(self.validity, dtype=bool).copy()
        if valid.shape == positions.shape[:2]:
            valid = np.repeat(valid[..., None], 3, axis=2)
        if valid.shape != positions.shape:
            raise ValueError(
                "reference validity must have shape (T, N) or (T, N, 3)"
            )
        if np.any(valid & ~np.isfinite(positions)):
            raise ValueError("valid reference coordinates must be finite")
        if not np.any(valid):
            raise ValueError("reference trajectory has no valid coordinates")
        positions[~valid] = np.nan
        positions = readonly_array(positions, dtype=np.float64)
        valid = readonly_array(valid, dtype=bool)
        digest = self.source_npz_sha256
        if (
            type(digest) is not str
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("source_npz_sha256 must be a lowercase SHA-256 digest")
        object.__setattr__(self, "node_ids", nodes)
        object.__setattr__(self, "positions_m", positions)
        object.__setattr__(self, "frame_times_s", times)
        object.__setattr__(self, "validity", valid)

    @property
    def artifact_id(self) -> str:
        descriptor = {
            "schema": EXTERNAL_REFERENCE_SCHEMA,
            "schema_version": EXTERNAL_REFERENCE_SCHEMA_VERSION,
            "case_id": self.case_id,
            "source_npz_sha256": self.source_npz_sha256,
            "arrays": {
                "node_ids": array_sha256(self.node_ids),
                "positions_m": array_sha256(self.positions_m),
                "frame_times_s": array_sha256(self.frame_times_s),
                "validity": array_sha256(self.validity),
            },
        }
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_external_reference(path: str | Path) -> ExternalReferenceTrajectory:
    """Load a closed, non-pickled evaluation-reference NPZ."""

    snapshot = load_numpy_archive(path, name="external reference trajectory")
    archive = snapshot.arrays
    members = frozenset(archive)
    missing = sorted(_REQUIRED_MEMBERS - members)
    unexpected = sorted(members - _REQUIRED_MEMBERS - _OPTIONAL_MEMBERS)
    if missing or unexpected:
        raise ValueError(
            "external reference archive members changed; "
            f"missing={missing}, unexpected={unexpected}"
        )
    case_id = _scalar_text(archive["case_id"], name="case_id")
    nodes = np.asarray(archive["node_ids"])
    positions = np.asarray(archive["positions_world_m"], dtype=np.float64)
    times = np.asarray(archive["frame_times_s"], dtype=np.float64)
    validity = (
        np.asarray(archive["validity_mask"], dtype=bool)
        if "validity_mask" in members
        else np.isfinite(positions)
    )
    source_hash = snapshot.snapshot.sha256
    return ExternalReferenceTrajectory(
        case_id=case_id,
        node_ids=nodes,
        positions_m=positions,
        frame_times_s=times,
        validity=validity,
        source_npz_sha256=source_hash,
    )


__all__ = [
    "EXTERNAL_REFERENCE_SCHEMA",
    "EXTERNAL_REFERENCE_SCHEMA_VERSION",
    "ExternalReferenceTrajectory",
    "load_external_reference",
]
