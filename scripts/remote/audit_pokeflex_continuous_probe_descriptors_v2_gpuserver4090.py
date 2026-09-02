#!/usr/bin/env python3
"""Run the revised source-only PokeFlex continuous probe audit.

Revision 2 leaves the descriptor, cohort, thresholds, and information boundary
unchanged. It replaces two contact-localization details after the first
source-only diagnostic:

* the force reference is the highest-excess-force contiguous active run rather
  than the first threshold crossing, which may belong to an initial preload;
* the geometry contact is the earliest trajectory point within a frozen
  tolerance of the minimum mesh distance, avoiding selection of a later return
  path that reaches the same location.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "audit_pokeflex_continuous_probe_descriptors_gpuserver4090.py"
)
TOLERANCE_GRID_M = (0.0, 0.001, 0.0025, 0.005, 0.01, 0.02)
_SELECTED_TOLERANCE_M = 0.0


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "pokeflex_continuous_probe_descriptor_audit_v1",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load continuous descriptor audit module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def dominant_force_contact_index(
    forces: np.ndarray,
    valid_force: np.ndarray,
    threshold: float,
) -> int | None:
    active = valid_force & (forces[:, 1] > threshold)
    runs: list[tuple[int, int]] = []
    start = None
    for index, value in enumerate(active.tolist()):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(active) - 1):
            end = index if value and index == len(active) - 1 else index - 1
            runs.append((start, end))
            start = None
    if not runs:
        return None

    def score(run: tuple[int, int]) -> tuple[float, int, float, int]:
        first, last = run
        values = forces[first : last + 1, 1]
        excess = float(np.sum(np.maximum(values - threshold, 0.0)))
        length = last - first + 1
        peak = float(np.max(values))
        return excess, length, peak, -first

    return int(max(runs, key=score)[0])


def near_minimum_geometric_contact(
    transforms: np.ndarray,
    tree: Any,
    axis: int,
    offset: float,
) -> tuple[int, float, np.ndarray]:
    points = BASE.tool_points(transforms, axis, offset)
    distances, _ = tree.query(points, k=1, workers=1)
    minimum = float(np.min(distances))
    candidates = np.flatnonzero(distances <= minimum + _SELECTED_TOLERANCE_M)
    index = int(candidates[0])
    return index, float(distances[index]), points


def choose_revised_tip_model(
    episodes: list[dict[str, Any]],
    axes: list[int],
    offsets: list[float],
) -> dict[str, Any]:
    global _SELECTED_TOLERANCE_M
    candidates = []
    for axis in axes:
        for offset in offsets:
            points_by_episode = []
            for episode in episodes:
                points = BASE.tool_points(episode["transforms"], axis, offset)
                distances, _ = episode["tree"].query(points, k=1, workers=1)
                points_by_episode.append((points, np.asarray(distances, dtype=float)))
            for tolerance in TOLERANCE_GRID_M:
                errors = []
                selected_distances = []
                for episode, (_, distances) in zip(
                    episodes,
                    points_by_episode,
                    strict=True,
                ):
                    force_index = episode["force_contact_index"]
                    if force_index is None:
                        continue
                    minimum = float(np.min(distances))
                    indices = np.flatnonzero(distances <= minimum + tolerance)
                    predicted = int(indices[0])
                    errors.append(abs(predicted - int(force_index)))
                    selected_distances.append(float(distances[predicted]))
                if not errors:
                    raise ValueError("no force contacts available for tip calibration")
                candidates.append(
                    {
                        "axis": axis,
                        "offset_m": offset,
                        "distance_tolerance_m": tolerance,
                        "median_absolute_index_error": float(np.median(errors)),
                        "p90_absolute_index_error": float(np.quantile(errors, 0.9)),
                        "median_surface_distance_m": float(
                            np.median(selected_distances)
                        ),
                    }
                )
    selected = min(
        candidates,
        key=lambda item: (
            item["median_absolute_index_error"],
            item["p90_absolute_index_error"],
            item["median_surface_distance_m"],
            item["distance_tolerance_m"],
            abs(item["offset_m"]),
            item["axis"],
        ),
    )
    _SELECTED_TOLERANCE_M = float(selected["distance_tolerance_m"])
    return selected


def main() -> int:
    BASE.force_contact_index = dominant_force_contact_index
    BASE.geometric_contact = near_minimum_geometric_contact
    BASE.choose_tip_model = choose_revised_tip_model
    return BASE.main()


if __name__ == "__main__":
    raise SystemExit(main())
