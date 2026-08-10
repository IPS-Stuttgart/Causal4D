"""Audited one-to-one geometric mapping from query points to simulator nodes."""

from __future__ import annotations

import hashlib
import html
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from causal4d.atomic_io import atomic_write_binary, atomic_write_json, atomic_write_text
from causal4d.contracts import array_sha256
from causal4d.numpy_archive import load_numpy_archive

EXTERNAL_NODE_MAPPING_SCHEMA = "causal4d.external_query_node_mapping"
EXTERNAL_NODE_MAPPING_SCHEMA_VERSION = 1


def _load_array(
    path: str | Path,
    key: str,
    *,
    name: str,
) -> tuple[np.ndarray, str]:
    snapshot = load_numpy_archive(path, name=f"{name} NPZ")
    if key not in snapshot.arrays:
        raise ValueError(f"{name} NPZ is missing array {key!r}")
    return np.asarray(snapshot.arrays[key]), snapshot.snapshot.sha256


def _text_ids(values: np.ndarray, *, name: str) -> tuple[str, ...]:
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
        if type(value) is not str or not value:
            raise ValueError(f"{name}[{index}] must be a nonempty string")
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return tuple(result)


def build_external_node_mapping(
    query_positions_m: np.ndarray,
    node_positions_m: np.ndarray,
    node_ids: np.ndarray,
    *,
    query_ids: Sequence[str] | None = None,
    maximum_distance_m: float = 0.005,
    query_source_sha256: str | None = None,
    node_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a minimum-total-distance one-to-one assignment.

    This is an audited geometric convenience, not proof of material identity.
    """

    queries = np.asarray(query_positions_m, dtype=np.float64)
    nodes = np.asarray(node_positions_m, dtype=np.float64)
    raw_node_ids = np.asarray(node_ids)
    if queries.ndim != 2 or queries.shape[1] not in {2, 3} or not len(queries):
        raise ValueError("query_positions_m must have nonempty shape (Q, 2|3)")
    if nodes.ndim != 2 or nodes.shape[1] != queries.shape[1] or not len(nodes):
        raise ValueError("node_positions_m must have shape (N, C) matching queries")
    if len(queries) > len(nodes):
        raise ValueError(
            "one-to-one mapping requires at least as many nodes as queries"
        )
    if not np.all(np.isfinite(queries)) or not np.all(np.isfinite(nodes)):
        raise ValueError("query and node positions must be finite")
    if raw_node_ids.ndim != 1 or len(raw_node_ids) != len(nodes):
        raise ValueError("node_ids must identify every simulator node")
    if raw_node_ids.dtype.kind not in {"i", "u"}:
        raise ValueError("node_ids must use an integer dtype")
    if raw_node_ids.dtype.kind == "u" and raw_node_ids.size:
        if int(np.max(raw_node_ids)) > np.iinfo(np.int64).max:
            raise ValueError("node_ids contain values outside int64 range")
    normalized_node_ids = raw_node_ids.astype(np.int64, copy=True)
    if np.any(normalized_node_ids < 0) or len(np.unique(normalized_node_ids)) != len(
        normalized_node_ids
    ):
        raise ValueError("node_ids must be unique and nonnegative")
    if not np.isfinite(maximum_distance_m) or maximum_distance_m <= 0.0:
        raise ValueError("maximum_distance_m must be finite and positive")
    normalized_query_ids = (
        tuple(f"query_{index:04d}" for index in range(len(queries)))
        if query_ids is None
        else tuple(query_ids)
    )
    if len(normalized_query_ids) != len(queries) or any(
        type(value) is not str or not value for value in normalized_query_ids
    ):
        raise ValueError("query_ids must identify every query with nonempty strings")
    if len(set(normalized_query_ids)) != len(normalized_query_ids):
        raise ValueError("query_ids must not contain duplicates")

    distance_matrix = np.linalg.norm(queries[:, None] - nodes[None], axis=2)
    query_rows, node_columns = linear_sum_assignment(distance_matrix)
    if not np.array_equal(query_rows, np.arange(len(queries))):
        raise RuntimeError("assignment did not preserve every query row")
    distances = distance_matrix[query_rows, node_columns]
    assigned_node_ids = normalized_node_ids[node_columns]
    accepted = bool(np.all(distances <= maximum_distance_m))
    entries = [
        {
            "query_index": int(index),
            "query_id": normalized_query_ids[index],
            "node_index": int(node_columns[index]),
            "node_id": int(assigned_node_ids[index]),
            "distance_m": float(distances[index]),
            "query_position_m": queries[index].tolist(),
            "node_position_m": nodes[node_columns[index]].tolist(),
            "within_tolerance": bool(distances[index] <= maximum_distance_m),
        }
        for index in range(len(queries))
    ]
    descriptor: dict[str, Any] = {
        "schema": EXTERNAL_NODE_MAPPING_SCHEMA,
        "schema_version": EXTERNAL_NODE_MAPPING_SCHEMA_VERSION,
        "method": "minimum_total_distance_one_to_one",
        "accepted": accepted,
        "maximum_distance_m": float(maximum_distance_m),
        "maximum_assigned_distance_m": float(np.max(distances)),
        "mean_assigned_distance_m": float(np.mean(distances)),
        "query_count": len(queries),
        "node_count": len(nodes),
        "coordinate_count": queries.shape[1],
        "query_source_sha256": query_source_sha256,
        "node_source_sha256": node_source_sha256,
        "array_hashes": {
            "query_positions_m": array_sha256(queries),
            "node_positions_m": array_sha256(nodes),
            "node_ids": array_sha256(normalized_node_ids),
            "assigned_node_indices": array_sha256(node_columns.astype(np.int64)),
            "assigned_node_ids": array_sha256(assigned_node_ids),
            "assigned_distances_m": array_sha256(distances),
        },
        "entries": entries,
        "warnings": [
            "geometric proximity does not prove persistent material correspondence"
        ],
    }
    identity_payload = json.dumps(
        descriptor,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    descriptor["mapping_id"] = hashlib.sha256(identity_payload).hexdigest()
    return descriptor


def _projection_indices(projection: str, coordinate_count: int) -> tuple[int, int]:
    choices = {"xy": (0, 1), "xz": (0, 2), "yz": (1, 2)}
    if projection not in choices:
        raise ValueError("projection must be one of 'xy', 'xz', or 'yz'")
    indices = choices[projection]
    if max(indices) >= coordinate_count:
        if projection != "xy" or coordinate_count != 2:
            raise ValueError("requested projection is unavailable for 2-D positions")
    return indices


def render_external_node_mapping_svg(
    report: dict[str, Any],
    *,
    projection: str = "xy",
    width: int = 900,
    height: int = 650,
) -> str:
    """Render an auditable 2-D projection without plotting dependencies."""

    entries = report.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("mapping report has no entries")
    coordinate_count = int(report["coordinate_count"])
    first_axis, second_axis = _projection_indices(projection, coordinate_count)
    query = np.asarray([entry["query_position_m"] for entry in entries], dtype=float)
    nodes = np.asarray([entry["node_position_m"] for entry in entries], dtype=float)
    combined = np.concatenate((query, nodes), axis=0)[:, [first_axis, second_axis]]
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    span = np.maximum(maximum - minimum, 1e-9)
    margin = 60.0

    def transform(position: np.ndarray) -> tuple[float, float]:
        normalized = (position[[first_axis, second_axis]] - minimum) / span
        x = margin + normalized[0] * (width - 2.0 * margin)
        y = height - margin - normalized[1] * (height - 2.0 * margin)
        return float(x), float(y)

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="20" y="30" font-family="sans-serif" font-size="18">'
        f"External query-node mapping ({html.escape(projection)})</text>",
        f'<text x="20" y="52" font-family="sans-serif" font-size="12">'
        f"accepted={str(report['accepted']).lower()}, "
        f"max distance={report['maximum_assigned_distance_m']:.6g} m</text>",
    ]
    for entry in entries:
        query_position = np.asarray(entry["query_position_m"], dtype=float)
        node_position = np.asarray(entry["node_position_m"], dtype=float)
        qx, qy = transform(query_position)
        nx, ny = transform(node_position)
        status = "#333" if entry["within_tolerance"] else "#a00"
        elements.extend(
            (
                f'<line x1="{qx:.3f}" y1="{qy:.3f}" x2="{nx:.3f}" '
                f'y2="{ny:.3f}" stroke="{status}" stroke-width="1"/>',
                f'<circle cx="{qx:.3f}" cy="{qy:.3f}" r="5" fill="white" '
                f'stroke="{status}" stroke-width="2"/>',
                f'<rect x="{nx - 4:.3f}" y="{ny - 4:.3f}" width="8" height="8" '
                f'fill="{status}"/>',
                f'<text x="{qx + 7:.3f}" y="{qy - 7:.3f}" '
                f'font-family="sans-serif" font-size="10">'
                f"{html.escape(str(entry['query_id']))} → {entry['node_id']}</text>",
            )
        )
    elements.append("</svg>")
    return "\n".join(elements) + "\n"


def save_external_node_mapping(
    report: dict[str, Any],
    output_json: str | Path,
    *,
    output_npz: str | Path | None = None,
    output_svg: str | Path | None = None,
    projection: str = "xy",
    overwrite: bool = False,
) -> None:
    """Publish mapping evidence and optional machine/visual artifacts atomically."""

    targets = [Path(output_json)]
    if output_npz is not None:
        targets.append(Path(output_npz))
    if output_svg is not None:
        targets.append(Path(output_svg))
    if not overwrite:
        existing = [str(target) for target in targets if target.exists()]
        if existing:
            raise FileExistsError(
                "external node-mapping outputs already exist: " + repr(existing)
            )
    entries = report["entries"]
    if output_npz is not None:
        query_positions = np.asarray(
            [entry["query_position_m"] for entry in entries], dtype=np.float64
        )
        node_positions = np.asarray(
            [entry["node_position_m"] for entry in entries], dtype=np.float64
        )
        node_ids = np.asarray([entry["node_id"] for entry in entries], dtype=np.int64)
        node_indices = np.asarray(
            [entry["node_index"] for entry in entries], dtype=np.int64
        )
        distances = np.asarray(
            [entry["distance_m"] for entry in entries], dtype=np.float64
        )

        def writer(handle) -> None:
            np.savez_compressed(
                handle,
                mapping_id=np.asarray(report["mapping_id"]),
                accepted=np.asarray(report["accepted"], dtype=bool),
                query_ids=np.asarray([entry["query_id"] for entry in entries]),
                query_positions_m=query_positions,
                assigned_node_indices=node_indices,
                assigned_node_ids=node_ids,
                assigned_node_positions_m=node_positions,
                assigned_distances_m=distances,
            )

        atomic_write_binary(output_npz, writer, overwrite=overwrite)
    if output_svg is not None:
        atomic_write_text(
            output_svg,
            render_external_node_mapping_svg(report, projection=projection),
            overwrite=overwrite,
        )
    # JSON is the completion marker for the optional companion artifacts.
    atomic_write_json(output_json, report, overwrite=overwrite)


__all__ = [
    "EXTERNAL_NODE_MAPPING_SCHEMA",
    "EXTERNAL_NODE_MAPPING_SCHEMA_VERSION",
    "build_external_node_mapping",
    "render_external_node_mapping_svg",
    "save_external_node_mapping",
    "_load_array",
    "_text_ids",
]
