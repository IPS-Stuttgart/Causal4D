"""Source-gated task-conditioned sparse-marker evaluation on Tracking Cloth.

Protocol V2 compares query-conditioned marker selection against a stronger,
query-agnostic global-state reconstruction policy under an equal two-marker
budget. Cotton recordings are the only method-development data. The fitted
cotton models and policy choices are frozen before an independent denim source
gate. Wool and polyester payloads remain unopened unless every gate criterion
passes.

The current cloth state is hidden except for the selected two markers. Every arm
receives the same two earlier complete marker frames and the exact same
constant-velocity forecast. Complete recordings are the operational source-gate
units. Held-out target summaries instead use material--size physical specimens,
with horizons nested within recordings and recordings nested within specimens.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, TypeAlias

import numpy as np
from numpy.typing import NDArray

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int64]

PILOT_KIND = "TrackingClothTaskConditionedObservationV2"
RESULT_SCHEMA = "causal4d.tracking-cloth-query-observation-result"
REQUEST_SCHEMA = "causal4d.tracking-cloth-query-observation-request"
EXPECTED_ROOT = Path(
    "/home/github-runner/.cache/datasets/tracking-cloth-deformation-v1-zenodo-14644526"
)
MATERIALS = ("cotton", "denim", "wool", "polyester")
CANDIDATES = ("upper", "lower", "central", "lateral", "fast")
PRIMARY_SCENARIOS = ("shake", "twist", "table", "self_collision")
ARMS = (
    "constant_velocity",
    "source_mean_residual",
    "task_conditioned",
    "global_state_conditioned",
    "fixed_upper",
    "random_cost_matched",
    "dependence_destroyed",
)


def canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_request(path: Path) -> dict[str, Any]:
    request = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(request, dict):
        raise TypeError("request must be a JSON object")
    supplied = request.get("request_id")
    unhashed = dict(request)
    unhashed.pop("request_id", None)
    if supplied != canonical_sha256(unhashed):
        raise ValueError("request_id does not match canonical request contents")
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != 2:
        raise ValueError("unsupported request schema")
    if request.get("stage") != "source-gated-evaluation-v2":
        raise ValueError("unsupported stage")
    if Path(str(request.get("dataset_root"))) != EXPECTED_ROOT:
        raise ValueError(f"dataset_root must be exactly {EXPECTED_ROOT}")
    exact_lists = {
        "source_fit_materials": ["cotton"],
        "source_gate_materials": ["denim"],
        "target_materials": ["wool", "polyester"],
        "target_model_training_materials": ["cotton"],
        "candidate_groups": list(CANDIDATES),
        "primary_scenarios": list(PRIMARY_SCENARIOS),
    }
    for field, expected in exact_lists.items():
        if request.get(field) != expected:
            raise ValueError(f"{field} changed from the frozen V2 value")
    if request.get("primary_statistical_unit") != "material_size_physical_specimen":
        raise ValueError(
            "primary_statistical_unit must be material_size_physical_specimen"
        )
    if (
        request.get("source_gate_statistical_unit")
        != "complete_recording_after_averaging_horizons"
    ):
        raise ValueError("source_gate_statistical_unit changed")
    if (
        request.get("target_nesting")
        != "horizons_within_recordings_within_material_size_specimens"
    ):
        raise ValueError("target_nesting changed")
    if request.get("target_specimen_count_if_opened") != 4:
        raise ValueError("target_specimen_count_if_opened must be four")
    if request.get("generic_policy_target") != "current_global_affine_residual_field":
        raise ValueError("generic_policy_target changed")
    if request.get("constant_velocity_residual_prediction") != "exact_zero":
        raise ValueError("constant-velocity baseline is not exact zero residual")
    expected_boundary = {
        "target_file_contents_opened_before_source_gate": False,
        "rgb_or_depth_data_used": False,
        "physical_command_sent": False,
        "dataset_modified": False,
        "paper_claim_authorized": False,
        "population_generalization_claim_authorized": False,
        "target_models_refit_after_gate": False,
    }
    if request.get("information_boundary") != expected_boundary:
        raise ValueError("information boundary changed")
    return request


@dataclass(frozen=True)
class RecordingKey:
    path: Path
    relative_path: str
    material: str
    scenario: str
    size: str


@dataclass(frozen=True)
class Recording:
    key: RecordingKey
    times: FloatArray
    points_m: FloatArray
    unit_scale_to_m: float


@dataclass(frozen=True)
class WindowBatch:
    recording: RecordingKey
    horizon_seconds: float
    x_by_candidate: dict[str, FloatArray]
    query_target: FloatArray
    global_state_target: FloatArray
    baseline_query: FloatArray
    actual_query: FloatArray


@dataclass(frozen=True)
class LinearGaussianModel:
    x_mean: FloatArray
    x_scale: FloatArray
    y_mean: FloatArray
    coefficient: FloatArray
    covariance: FloatArray

    def predict(self, x: FloatArray) -> FloatArray:
        standardized = (x - self.x_mean) / self.x_scale
        return self.y_mean + standardized @ self.coefficient


@dataclass(frozen=True)
class PredictionModelSet:
    candidate_models: dict[str, LinearGaussianModel]
    source_mean: FloatArray
    source_mean_covariance: FloatArray
    constant_velocity_covariance: FloatArray
    dependence_destroyed: LinearGaussianModel


def classify_recording(path: Path, root: Path) -> RecordingKey | None:
    if path.suffix.lower() != ".csv":
        return None
    relative = path.relative_to(root).as_posix()
    text = relative.lower().replace("-", "_")
    material = next((name for name in MATERIALS if name in text), None)
    if material is None:
        return None
    if "shake" in text:
        scenario = "shake"
    elif "twist" in text:
        scenario = "twist"
    elif "half_lay" in text or "full_lay" in text or "tablecloth" in text:
        scenario = "table"
    elif "hitting" in text or "hit" in path.stem.lower():
        scenario = "hitting"
    elif any(
        token in text
        for token in ("self_collision", "selfcollision", "rep1", "rep2", "rep3")
    ):
        scenario = "self_collision"
    elif "collision" in text and "table" not in text:
        scenario = "self_collision"
    else:
        return None
    if "a3" in text:
        size = "A3"
    elif "a2" in text:
        size = "A2"
    else:
        size = "unknown"
    return RecordingKey(path, relative, material, scenario, size)


def discover_recordings(root: Path) -> list[RecordingKey]:
    if not root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {root}")
    keys = [
        key
        for path in sorted(root.rglob("*.csv"))
        if (key := classify_recording(path, root)) is not None
    ]
    relative = [key.relative_path for key in keys]
    if len(relative) != len(set(relative)):
        raise ValueError("duplicate relative recording paths")
    return keys


def _numeric(value: str) -> float:
    text = value.strip()
    return math.nan if not text else float(text)


def _candidate_data_row(row: list[str]) -> tuple[int, int] | None:
    trimmed = list(row)
    while trimmed and not trimmed[-1].strip():
        trimmed.pop()
    if len(trimmed) < 2 + 3 * 8:
        return None
    try:
        float(trimmed[0])
        float(trimmed[1])
    except ValueError:
        return None
    coordinates = len(trimmed) - 2
    if coordinates % 3:
        return None
    markers = coordinates // 3
    if markers not in (12, 20, 22):
        return None
    return len(trimmed), markers


def read_recording(key: RecordingKey) -> Recording:
    with key.path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    start: int | None = None
    width: int | None = None
    markers: int | None = None
    for index, row in enumerate(rows):
        candidate = _candidate_data_row(row)
        if candidate is not None:
            start = index
            width, markers = candidate
            break
    if start is None or width is None or markers is None:
        raise ValueError(f"could not locate numeric marker rows in {key.relative_path}")
    parsed: list[list[float]] = []
    for row in rows[start:]:
        trimmed = list(row)
        while trimmed and not trimmed[-1].strip():
            trimmed.pop()
        if len(trimmed) != width:
            continue
        try:
            values = [_numeric(value) for value in trimmed]
        except ValueError:
            continue
        if not math.isfinite(values[0]) or not math.isfinite(values[1]):
            continue
        parsed.append(values)
    if len(parsed) < 50:
        raise ValueError(
            f"too few numeric frames in {key.relative_path}: {len(parsed)}"
        )
    matrix = np.asarray(parsed, dtype=np.float64)
    times = matrix[:, 1]
    if not np.all(np.diff(times) > 0.0):
        raise ValueError(
            f"timestamps are not strictly increasing in {key.relative_path}"
        )
    points = matrix[:, 2:].reshape(matrix.shape[0], markers, 3)
    finite_frames = np.flatnonzero(
        np.sum(np.isfinite(points), axis=(1, 2)) >= 3 * min(markers, 12)
    )
    if not finite_frames.size:
        raise ValueError(f"no sufficiently complete frame in {key.relative_path}")
    reference = points[int(finite_frames[0])]
    valid = reference[np.all(np.isfinite(reference), axis=1)]
    extent = float(np.linalg.norm(np.nanmax(valid, axis=0) - np.nanmin(valid, axis=0)))
    if 0.05 <= extent <= 3.0:
        scale = 1.0
    elif 50.0 <= extent <= 3000.0:
        scale = 1e-3
    else:
        raise ValueError(
            f"cannot infer metric unit in {key.relative_path}; reference extent={extent}"
        )
    return Recording(
        key=key,
        times=np.asarray(times, dtype=np.float64),
        points_m=np.asarray(points * scale, dtype=np.float64),
        unit_scale_to_m=scale,
    )


def _two_indices(values: FloatArray, *, largest: bool) -> IntArray:
    order = np.argsort(values, kind="stable")
    selected = order[-2:] if largest else order[:2]
    return np.asarray(np.sort(selected), dtype=np.int64)


def _oriented_axes(points: FloatArray) -> FloatArray:
    centered = points - np.mean(points, axis=0)
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    axes = np.asarray(right, dtype=np.float64).copy()
    for row in range(axes.shape[0]):
        pivot = int(np.argmax(np.abs(axes[row])))
        if axes[row, pivot] < 0.0:
            axes[row] *= -1.0
    return axes


def marker_groups(
    previous: FloatArray,
    older: FloatArray,
    usable_indices: IntArray,
) -> dict[str, IntArray]:
    if usable_indices.size < 8:
        raise ValueError("fewer than eight usable markers")
    p = previous[usable_indices]
    o = older[usable_indices]
    centroid = np.mean(p, axis=0)
    speed = np.linalg.norm(p - o, axis=1)
    centered = p - centroid
    lateral_coordinate = centered @ _oriented_axes(p)[0]
    local: dict[str, IntArray] = {
        "upper": _two_indices(p[:, 2], largest=True),
        "lower": _two_indices(p[:, 2], largest=False),
        "central": _two_indices(np.linalg.norm(centered, axis=1), largest=False),
        "fast": _two_indices(speed, largest=True),
    }
    lateral_order = np.argsort(lateral_coordinate, kind="stable")
    local["lateral"] = np.asarray(
        sorted((int(lateral_order[0]), int(lateral_order[-1]))),
        dtype=np.int64,
    )
    return {
        name: np.asarray(usable_indices[group], dtype=np.int64)
        for name, group in local.items()
    }


def lower_query_indices(previous: FloatArray, usable_indices: IntArray) -> IntArray:
    if usable_indices.size < 8:
        raise ValueError("fewer than eight usable markers")
    count = max(4, int(math.ceil(usable_indices.size / 3.0)))
    order = np.argsort(previous[usable_indices, 2], kind="stable")
    return np.asarray(np.sort(usable_indices[order[:count]]), dtype=np.int64)


def group_feature(
    actual: FloatArray,
    predicted: FloatArray,
    previous: FloatArray,
    indices: IntArray,
) -> FloatArray:
    residual = actual[indices] - predicted[indices]
    if not np.all(np.isfinite(residual)):
        raise ValueError("candidate observation contains missing values")
    mean = np.mean(residual, axis=0)
    separation = previous[indices[1]] - previous[indices[0]]
    norm = float(np.linalg.norm(separation))
    direction = separation / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    differential = float((residual[1] - residual[0]) @ direction)
    return np.asarray([mean[0], mean[1], mean[2], differential], dtype=np.float64)


def query_vector(points: FloatArray, indices: IntArray) -> FloatArray:
    selected = points[indices]
    centroid = np.mean(selected, axis=0)
    radius = float(np.sqrt(np.mean(np.sum((selected - centroid) ** 2, axis=1))))
    return np.asarray([centroid[0], centroid[1], centroid[2], radius], dtype=np.float64)


def global_affine_residual_descriptor(
    actual: FloatArray,
    predicted: FloatArray,
    previous: FloatArray,
    indices: IntArray,
) -> FloatArray:
    """Describe the complete current residual field without using the query.

    A spatially affine residual field is fitted over the full usable marker set.
    The three-vector intercept and two three-vector gradients form a 9-D global
    state target, all in metres. A tenth component records the non-affine RMS.
    """

    residual = actual[indices] - predicted[indices]
    geometry = previous[indices]
    centered = geometry - np.mean(geometry, axis=0)
    axes = _oriented_axes(geometry)[:2]
    coordinates = centered @ axes.T
    scale = np.sqrt(np.mean(coordinates**2, axis=0))
    scale = np.maximum(scale, 1e-9)
    design = np.column_stack((np.ones(indices.size), coordinates / scale))
    coefficient, *_ = np.linalg.lstsq(design, residual, rcond=None)
    fitted = design @ coefficient
    non_affine_rms = float(np.sqrt(np.mean((residual - fitted) ** 2)))
    return np.concatenate((coefficient.reshape(-1), [non_affine_rms])).astype(
        np.float64,
        copy=False,
    )


def extract_windows(
    recording: Recording,
    *,
    horizon_seconds: float,
    lag_seconds: float,
    stride_seconds: float,
) -> WindowBatch:
    times = recording.times
    points = recording.points_m
    dt = float(np.median(np.diff(times)))
    if not 1.0 / 500.0 <= dt <= 1.0 / 20.0:
        raise ValueError(
            f"unexpected sampling interval in {recording.key.relative_path}: {dt}"
        )
    lag = max(1, int(round(lag_seconds / dt)))
    horizon = max(1, int(round(horizon_seconds / dt)))
    stride = max(1, int(round(stride_seconds / dt)))
    x_rows: dict[str, list[FloatArray]] = {name: [] for name in CANDIDATES}
    query_rows: list[FloatArray] = []
    state_rows: list[FloatArray] = []
    baseline_rows: list[FloatArray] = []
    actual_rows: list[FloatArray] = []
    for current in range(2 * lag, points.shape[0] - horizon, stride):
        older = points[current - 2 * lag]
        previous = points[current - lag]
        now = points[current]
        future = points[current + horizon]
        finite = (
            np.all(np.isfinite(older), axis=1)
            & np.all(np.isfinite(previous), axis=1)
            & np.all(np.isfinite(now), axis=1)
            & np.all(np.isfinite(future), axis=1)
        )
        usable = np.asarray(np.flatnonzero(finite), dtype=np.int64)
        if usable.size < 8:
            continue
        groups = marker_groups(previous, older, usable)
        query_indices = lower_query_indices(previous, usable)
        velocity_per_lag = previous - older
        predicted_now = previous + velocity_per_lag
        horizon_multiplier = 1.0 + horizon / lag
        predicted_future = previous + horizon_multiplier * velocity_per_lag
        baseline_query = query_vector(predicted_future, query_indices)
        actual_query = query_vector(future, query_indices)
        for name, indices in groups.items():
            x_rows[name].append(group_feature(now, predicted_now, previous, indices))
        query_rows.append(actual_query - baseline_query)
        state_rows.append(
            global_affine_residual_descriptor(now, predicted_now, previous, usable)
        )
        baseline_rows.append(baseline_query)
        actual_rows.append(actual_query)
    if len(query_rows) < 12:
        raise ValueError(
            f"too few valid windows in {recording.key.relative_path} at "
            f"{horizon_seconds}s: {len(query_rows)}"
        )
    return WindowBatch(
        recording=recording.key,
        horizon_seconds=float(horizon_seconds),
        x_by_candidate={name: np.stack(rows) for name, rows in x_rows.items()},
        query_target=np.stack(query_rows),
        global_state_target=np.stack(state_rows),
        baseline_query=np.stack(baseline_rows),
        actual_query=np.stack(actual_rows),
    )


def _weighted_mean(values: FloatArray, weights: FloatArray) -> FloatArray:
    return np.sum(values * weights[:, None], axis=0) / float(np.sum(weights))


def _recording_equal_arrays(
    batches: Sequence[WindowBatch],
    candidate: str,
    target: str,
    *,
    permute_target: bool = False,
    token_prefix: str = "",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    if not batches:
        raise ValueError("no batches to stack")
    xs: list[FloatArray] = []
    ys: list[FloatArray] = []
    weights: list[FloatArray] = []
    for batch in batches:
        x = batch.x_by_candidate[candidate]
        y = batch.query_target if target == "query" else batch.global_state_target
        if permute_target:
            seed_text = f"{token_prefix}/{batch.recording.relative_path}/{batch.horizon_seconds}"
            seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
            permutation = np.random.default_rng(seed).permutation(y.shape[0])
            y = y[permutation]
        xs.append(x)
        ys.append(y)
        weights.append(np.full(x.shape[0], 1.0 / x.shape[0], dtype=np.float64))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(weights)


def _covariance_from_error_batches(errors: Sequence[FloatArray]) -> FloatArray:
    if not errors:
        raise ValueError("no errors for covariance")
    dimension = errors[0].shape[1]
    second = np.zeros((dimension, dimension), dtype=np.float64)
    total = 0.0
    for error in errors:
        if error.ndim != 2 or error.shape[1] != dimension:
            raise ValueError("incompatible error batches")
        weight = 1.0 / error.shape[0]
        second += weight * (error.T @ error)
        total += 1.0
    covariance = second / total
    covariance = 0.5 * (covariance + covariance.T)
    average_variance = max(float(np.trace(covariance)) / dimension, 1e-12)
    shrinkage = 1e-3
    covariance = (1.0 - shrinkage) * covariance + shrinkage * average_variance * np.eye(
        dimension
    )
    covariance += average_variance * 1e-8 * np.eye(dimension)
    return covariance


def fit_weighted_model(
    batches: Sequence[WindowBatch],
    candidate: str,
    target: str,
    *,
    ridge: float,
    covariance: FloatArray | None = None,
    permute_target: bool = False,
    token_prefix: str = "",
) -> LinearGaussianModel:
    x, y, weights = _recording_equal_arrays(
        batches,
        candidate,
        target,
        permute_target=permute_target,
        token_prefix=token_prefix,
    )
    if x.shape[0] < 8 or x.shape[0] != y.shape[0]:
        raise ValueError("invalid regression arrays")
    x_mean = _weighted_mean(x, weights)
    centered_x = x - x_mean
    x_variance = _weighted_mean(centered_x**2, weights)
    x_scale = np.sqrt(np.maximum(x_variance, 1e-18))
    standardized = centered_x / x_scale
    y_mean = _weighted_mean(y, weights)
    centered_y = y - y_mean
    weighted_x = standardized * weights[:, None]
    penalty = float(ridge) * np.eye(standardized.shape[1])
    coefficient = np.linalg.solve(
        standardized.T @ weighted_x + penalty,
        weighted_x.T @ centered_y,
    )
    if covariance is None:
        residual = y - (y_mean + standardized @ coefficient)
        covariance = _covariance_from_error_batches([residual])
    return LinearGaussianModel(
        x_mean=np.asarray(x_mean, dtype=np.float64),
        x_scale=np.asarray(x_scale, dtype=np.float64),
        y_mean=np.asarray(y_mean, dtype=np.float64),
        coefficient=np.asarray(coefficient, dtype=np.float64),
        covariance=np.asarray(covariance, dtype=np.float64),
    )


def _recording_mse(error: FloatArray) -> float:
    return float(np.mean(np.sum(error**2, axis=1) / error.shape[1]))


def _normalized_recording_mse(error: FloatArray, scale: FloatArray) -> float:
    standardized = error / np.maximum(scale[None, :], 1e-12)
    return float(np.mean(np.sum(standardized**2, axis=1) / error.shape[1]))


def _leave_one_recording_errors(
    batches: Sequence[WindowBatch],
    candidate: str,
    target: str,
    *,
    ridge: float,
    permute_target: bool = False,
    token_prefix: str = "",
) -> list[FloatArray]:
    if len(batches) < 2:
        raise ValueError("leave-one-recording-out requires at least two recordings")
    errors: list[FloatArray] = []
    for held_out, batch in enumerate(batches):
        training = [item for index, item in enumerate(batches) if index != held_out]
        model = fit_weighted_model(
            training,
            candidate,
            target,
            ridge=ridge,
            permute_target=permute_target,
            token_prefix=token_prefix,
        )
        truth = batch.query_target if target == "query" else batch.global_state_target
        errors.append(truth - model.predict(batch.x_by_candidate[candidate]))
    return errors


def _equal_recording_target_scale(
    batches: Sequence[WindowBatch],
    target: str,
) -> FloatArray:
    """Scale target coordinates while giving every recording equal total weight."""
    values: list[FloatArray] = []
    weights: list[FloatArray] = []
    for batch in batches:
        value = batch.query_target if target == "query" else batch.global_state_target
        values.append(value)
        weights.append(np.full(value.shape[0], 1.0 / value.shape[0], dtype=np.float64))
    stacked = np.concatenate(values)
    stacked_weights = np.concatenate(weights)
    mean = _weighted_mean(stacked, stacked_weights)
    variance = _weighted_mean((stacked - mean) ** 2, stacked_weights)
    return np.sqrt(np.maximum(variance, 1e-18))


def choose_policies(
    source_batches: Mapping[tuple[str, float], Sequence[WindowBatch]],
    *,
    ridge: float,
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for (scenario, horizon), batches_raw in sorted(source_batches.items()):
        batches = list(batches_raw)
        if len(batches) < 2:
            continue
        state_scale = np.maximum(
            _equal_recording_target_scale(batches, "global"),
            1e-9,
        )
        query_scores: dict[str, float] = {}
        global_scores: dict[str, float] = {}
        for candidate in CANDIDATES:
            query_errors = _leave_one_recording_errors(
                batches,
                candidate,
                "query",
                ridge=ridge,
            )
            state_errors = _leave_one_recording_errors(
                batches,
                candidate,
                "global",
                ridge=ridge,
            )
            query_scores[candidate] = float(
                np.mean([_recording_mse(error) for error in query_errors])
            )
            global_scores[candidate] = float(
                np.mean(
                    [
                        _normalized_recording_mse(error, state_scale)
                        for error in state_errors
                    ]
                )
            )
        task = min(CANDIDATES, key=lambda name: (query_scores[name], name))
        global_state = min(CANDIDATES, key=lambda name: (global_scores[name], name))
        key = f"{scenario}/{horizon:.6g}"
        decisions[key] = {
            "scenario": scenario,
            "horizon_seconds": horizon,
            "task_selected": task,
            "global_state_selected": global_state,
            "task_query_cv_mse_by_candidate_m2": query_scores,
            "global_state_cv_normalized_mse_by_candidate": global_scores,
        }
    return decisions


def _equal_recording_mean_target(batches: Sequence[WindowBatch]) -> FloatArray:
    recording_means = [np.mean(batch.query_target, axis=0) for batch in batches]
    return np.mean(np.stack(recording_means), axis=0)


def _source_mean_loo_errors(batches: Sequence[WindowBatch]) -> list[FloatArray]:
    errors: list[FloatArray] = []
    for held_out, batch in enumerate(batches):
        training = [item for index, item in enumerate(batches) if index != held_out]
        mean = _equal_recording_mean_target(training)
        errors.append(batch.query_target - mean)
    return errors


def fit_prediction_models(
    source_batches: Mapping[tuple[str, float], Sequence[WindowBatch]],
    decisions: Mapping[str, Any],
    *,
    ridge: float,
) -> dict[str, PredictionModelSet]:
    result: dict[str, PredictionModelSet] = {}
    for (scenario, horizon), batches_raw in sorted(source_batches.items()):
        key = f"{scenario}/{horizon:.6g}"
        if key not in decisions:
            continue
        batches = list(batches_raw)
        candidate_models: dict[str, LinearGaussianModel] = {}
        for candidate in CANDIDATES:
            loo_errors = _leave_one_recording_errors(
                batches,
                candidate,
                "query",
                ridge=ridge,
            )
            covariance = _covariance_from_error_batches(loo_errors)
            candidate_models[candidate] = fit_weighted_model(
                batches,
                candidate,
                "query",
                ridge=ridge,
                covariance=covariance,
            )
        source_mean = _equal_recording_mean_target(batches)
        source_mean_covariance = _covariance_from_error_batches(
            _source_mean_loo_errors(batches)
        )
        constant_velocity_covariance = _covariance_from_error_batches(
            [batch.query_target for batch in batches]
        )
        task_candidate = str(decisions[key]["task_selected"])
        destroyed_loo = _leave_one_recording_errors(
            batches,
            task_candidate,
            "query",
            ridge=ridge,
            permute_target=True,
            token_prefix=f"destroy/{key}",
        )
        destroyed = fit_weighted_model(
            batches,
            task_candidate,
            "query",
            ridge=ridge,
            covariance=_covariance_from_error_batches(destroyed_loo),
            permute_target=True,
            token_prefix=f"destroy/{key}",
        )
        result[key] = PredictionModelSet(
            candidate_models=candidate_models,
            source_mean=source_mean,
            source_mean_covariance=source_mean_covariance,
            constant_velocity_covariance=constant_velocity_covariance,
            dependence_destroyed=destroyed,
        )
    return result


def _gaussian_metrics(
    error: FloatArray,
    covariance: FloatArray,
) -> tuple[float, float, float]:
    root = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(root, error.T).T
    quadratic = np.sum(whitened**2, axis=1)
    nll = 0.5 * (
        error.shape[1] * math.log(2.0 * math.pi)
        + 2.0 * float(np.sum(np.log(np.diag(root))))
        + quadratic
    )
    std = np.sqrt(np.diag(covariance))
    coverage = np.mean(np.abs(error) <= 1.6448536269514722 * std[None, :])
    normalized_nees = float(np.mean(quadratic) / error.shape[1])
    return float(np.mean(nll)), float(coverage), normalized_nees


def score_batches(
    batches: Mapping[tuple[str, float], Sequence[WindowBatch]],
    decisions: Mapping[str, Any],
    models: Mapping[str, PredictionModelSet],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scenario, horizon), group in sorted(batches.items()):
        key = f"{scenario}/{horizon:.6g}"
        if key not in models:
            continue
        decision = decisions[key]
        fitted = models[key]
        task_group = str(decision["task_selected"])
        global_group = str(decision["global_state_selected"])
        for batch in group:
            token = f"random/{batch.recording.relative_path}/{key}"
            seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
            random_group = CANDIDATES[seed % len(CANDIDATES)]
            zeros = np.zeros_like(batch.query_target)
            arms: dict[str, tuple[FloatArray, FloatArray]] = {
                "constant_velocity": (zeros, fitted.constant_velocity_covariance),
                "source_mean_residual": (
                    np.broadcast_to(fitted.source_mean, batch.query_target.shape),
                    fitted.source_mean_covariance,
                ),
                "task_conditioned": (
                    fitted.candidate_models[task_group].predict(
                        batch.x_by_candidate[task_group]
                    ),
                    fitted.candidate_models[task_group].covariance,
                ),
                "global_state_conditioned": (
                    fitted.candidate_models[global_group].predict(
                        batch.x_by_candidate[global_group]
                    ),
                    fitted.candidate_models[global_group].covariance,
                ),
                "fixed_upper": (
                    fitted.candidate_models["upper"].predict(
                        batch.x_by_candidate["upper"]
                    ),
                    fitted.candidate_models["upper"].covariance,
                ),
                "random_cost_matched": (
                    fitted.candidate_models[random_group].predict(
                        batch.x_by_candidate[random_group]
                    ),
                    fitted.candidate_models[random_group].covariance,
                ),
                "dependence_destroyed": (
                    fitted.dependence_destroyed.predict(
                        batch.x_by_candidate[task_group]
                    ),
                    fitted.dependence_destroyed.covariance,
                ),
            }
            arm_rows: dict[str, Any] = {}
            for name in ARMS:
                predicted_residual, covariance = arms[name]
                error = batch.query_target - predicted_residual
                nll, coverage, normalized_nees = _gaussian_metrics(error, covariance)
                mse = _recording_mse(error)
                arm_rows[name] = {
                    "mse_m2": mse,
                    "rmse_mm": 1000.0 * math.sqrt(mse),
                    "gaussian_nll": nll,
                    "marginal_90_coverage": coverage,
                    "normalized_joint_nees": normalized_nees,
                }
            rows.append(
                {
                    "recording": batch.recording.relative_path,
                    "material": batch.recording.material,
                    "scenario": scenario,
                    "size": batch.recording.size,
                    "horizon_seconds": horizon,
                    "windows": int(batch.query_target.shape[0]),
                    "task_selected": task_group,
                    "global_state_selected": global_group,
                    "random_selected": random_group,
                    "arms": arm_rows,
                }
            )
    return rows


def recording_level_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["recording"]), []).append(row)
    result: list[dict[str, Any]] = []
    for recording, group in sorted(grouped.items()):
        first = group[0]
        if any(
            item["material"] != first["material"]
            or item["scenario"] != first["scenario"]
            or item["size"] != first["size"]
            for item in group
        ):
            raise ValueError(f"recording metadata changed across horizons: {recording}")
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arms[arm] = {
                metric: float(np.mean([item["arms"][arm][metric] for item in group]))
                for metric in (
                    "mse_m2",
                    "gaussian_nll",
                    "marginal_90_coverage",
                    "normalized_joint_nees",
                )
            }
            arms[arm]["rmse_mm"] = 1000.0 * math.sqrt(arms[arm]["mse_m2"])
        result.append(
            {
                "recording": recording,
                "material": first["material"],
                "scenario": first["scenario"],
                "size": first["size"],
                "horizons": sorted(float(item["horizon_seconds"]) for item in group),
                "arms": arms,
            }
        )
    return result


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    recording_rows = recording_level_rows(rows)
    if not recording_rows:
        raise ValueError("no scored recording rows")
    aggregate: dict[str, Any] = {
        "recordings": len(recording_rows),
        "recording_horizon_rows": len(rows),
        "arms": {},
    }
    for arm in ARMS:
        mse = np.asarray(
            [row["arms"][arm]["mse_m2"] for row in recording_rows],
            dtype=np.float64,
        )
        aggregate["arms"][arm] = {
            "equal_recording_mse_m2": float(np.mean(mse)),
            "equal_recording_rmse_mm": 1000.0 * math.sqrt(float(np.mean(mse))),
            "equal_recording_gaussian_nll": float(
                np.mean([row["arms"][arm]["gaussian_nll"] for row in recording_rows])
            ),
            "equal_recording_marginal_90_coverage": float(
                np.mean(
                    [row["arms"][arm]["marginal_90_coverage"] for row in recording_rows]
                )
            ),
            "equal_recording_normalized_joint_nees": float(
                np.mean(
                    [
                        row["arms"][arm]["normalized_joint_nees"]
                        for row in recording_rows
                    ]
                )
            ),
        }
    for reference in ("constant_velocity", "source_mean_residual"):
        baseline = aggregate["arms"][reference]["equal_recording_mse_m2"]
        for arm in ARMS:
            value = aggregate["arms"][arm]["equal_recording_mse_m2"]
            aggregate["arms"][arm][f"relative_mse_improvement_vs_{reference}"] = (
                float((baseline - value) / baseline) if baseline > 0.0 else 0.0
            )
    return aggregate


def _relative_improvement(reference: float, value: float) -> float:
    return float((reference - value) / reference) if reference > 0.0 else 0.0


def source_gate(
    rows: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_rows(rows)
    task = aggregate["arms"]["task_conditioned"]["equal_recording_mse_m2"]
    global_state = aggregate["arms"]["global_state_conditioned"][
        "equal_recording_mse_m2"
    ]
    exact_cv = aggregate["arms"]["constant_velocity"]["equal_recording_mse_m2"]
    destroyed = aggregate["arms"]["dependence_destroyed"]["equal_recording_mse_m2"]
    recording_rows = recording_level_rows(rows)
    wins = [
        row["arms"]["task_conditioned"]["mse_m2"]
        < row["arms"]["global_state_conditioned"]["mse_m2"]
        for row in recording_rows
    ]
    win_fraction = float(np.mean(wins))
    scenario_ratios: dict[str, float] = {}
    for scenario in PRIMARY_SCENARIOS:
        subset = [row for row in recording_rows if row["scenario"] == scenario]
        if not subset:
            continue
        task_s = float(
            np.mean([row["arms"]["task_conditioned"]["mse_m2"] for row in subset])
        )
        generic_s = float(
            np.mean(
                [row["arms"]["global_state_conditioned"]["mse_m2"] for row in subset]
            )
        )
        scenario_ratios[scenario] = task_s / generic_s if generic_s > 0.0 else math.inf
    distinct_cells = sum(
        item["task_selected"] != item["global_state_selected"]
        for item in decisions.values()
    )
    thresholds = request["source_gate"]
    improvements = {
        "vs_global_state": _relative_improvement(global_state, task),
        "vs_constant_velocity": _relative_improvement(exact_cv, task),
        "vs_dependence_destroyed": _relative_improvement(destroyed, task),
    }
    checks = {
        "minimum_improvement_vs_global_state": improvements["vs_global_state"]
        >= float(thresholds["minimum_improvement_vs_global_state"]),
        "minimum_improvement_vs_constant_velocity": improvements["vs_constant_velocity"]
        >= float(thresholds["minimum_improvement_vs_constant_velocity"]),
        "minimum_improvement_vs_dependence_destroyed": improvements[
            "vs_dependence_destroyed"
        ]
        >= float(thresholds["minimum_improvement_vs_dependence_destroyed"]),
        "minimum_recording_win_fraction": win_fraction
        >= float(thresholds["minimum_recording_win_fraction"]),
        "maximum_worst_scenario_ratio": max(
            scenario_ratios.values(),
            default=math.inf,
        )
        <= float(thresholds["maximum_worst_scenario_ratio"]),
        "minimum_distinct_selection_cells": distinct_cells
        >= int(thresholds["minimum_distinct_selection_cells"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "relative_mse_improvements": improvements,
        "task_vs_global_state_recording_win_fraction": win_fraction,
        "task_to_global_state_scenario_mse_ratios": scenario_ratios,
        "distinct_task_vs_global_state_selection_cells": int(distinct_cells),
        "win_fraction_unit": "complete_recording_after_averaging_horizons",
        "inferential_boundary": (
            "Operational repeated-recording gate only; it is not a population "
            "confidence statement over physical cloth specimens."
        ),
        "aggregate": aggregate,
        "recording_rows": recording_rows,
    }


def specimen_level_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average horizons within recordings and recordings within specimens."""
    recording_rows = recording_level_rows(rows)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in recording_rows:
        material = str(row["material"])
        size = str(row["size"])
        if size not in {"A2", "A3"}:
            raise ValueError(f"unknown physical specimen size: {size}")
        grouped.setdefault(f"{material}/{size}", []).append(row)
    if not grouped:
        raise ValueError("no physical specimen rows")
    result: list[dict[str, Any]] = []
    for specimen, group in sorted(grouped.items()):
        first = group[0]
        arms: dict[str, Any] = {}
        for arm in ARMS:
            arms[arm] = {
                metric: float(np.mean([row["arms"][arm][metric] for row in group]))
                for metric in (
                    "mse_m2",
                    "gaussian_nll",
                    "marginal_90_coverage",
                    "normalized_joint_nees",
                )
            }
            arms[arm]["rmse_mm"] = 1000.0 * math.sqrt(arms[arm]["mse_m2"])
        result.append(
            {
                "specimen": specimen,
                "material": first["material"],
                "size": first["size"],
                "recordings": len(group),
                "scenarios": sorted({str(row["scenario"]) for row in group}),
                "arms": arms,
            }
        )
    return result


def aggregate_specimens(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    specimen_rows = specimen_level_rows(rows)
    aggregate: dict[str, Any] = {
        "unit": "material_size_physical_specimen",
        "specimens": len(specimen_rows),
        "recordings": len(recording_level_rows(rows)),
        "recording_horizon_rows": len(rows),
        "by_specimen": {row["specimen"]: row for row in specimen_rows},
        "arms": {},
    }
    for arm in ARMS:
        mse = np.asarray(
            [row["arms"][arm]["mse_m2"] for row in specimen_rows],
            dtype=np.float64,
        )
        aggregate["arms"][arm] = {
            "equal_specimen_mse_m2": float(np.mean(mse)),
            "equal_specimen_rmse_mm": 1000.0 * math.sqrt(float(np.mean(mse))),
            "equal_specimen_gaussian_nll": float(
                np.mean([row["arms"][arm]["gaussian_nll"] for row in specimen_rows])
            ),
            "equal_specimen_marginal_90_coverage": float(
                np.mean(
                    [row["arms"][arm]["marginal_90_coverage"] for row in specimen_rows]
                )
            ),
            "equal_specimen_normalized_joint_nees": float(
                np.mean(
                    [row["arms"][arm]["normalized_joint_nees"] for row in specimen_rows]
                )
            ),
        }
    for reference in ("constant_velocity", "source_mean_residual"):
        baseline = aggregate["arms"][reference]["equal_specimen_mse_m2"]
        for arm in ARMS:
            value = aggregate["arms"][arm]["equal_specimen_mse_m2"]
            aggregate["arms"][arm][f"relative_mse_improvement_vs_{reference}"] = (
                float((baseline - value) / baseline) if baseline > 0.0 else 0.0
            )
    return aggregate


def _bootstrap_specimen_difference(
    rows: Sequence[Mapping[str, Any]],
    arm_a: str,
    arm_b: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, Any]:
    specimen_rows = specimen_level_rows(rows)
    by_material: dict[str, list[float]] = {}
    observed: dict[str, float] = {}
    for row in specimen_rows:
        difference = float(row["arms"][arm_a]["mse_m2"] - row["arms"][arm_b]["mse_m2"])
        observed[str(row["specimen"])] = difference
        by_material.setdefault(str(row["material"]), []).append(difference)
    rng = np.random.default_rng(seed)
    simulated = np.empty(draws, dtype=np.float64)
    materials = sorted(by_material)
    for index in range(draws):
        material_means: list[float] = []
        for material in materials:
            values = np.asarray(by_material[material], dtype=np.float64)
            sample = rng.choice(values, size=values.size, replace=True)
            material_means.append(float(np.mean(sample)))
        simulated[index] = float(np.mean(material_means))
    point = float(np.mean([np.mean(by_material[name]) for name in materials]))
    lower, upper = np.quantile(simulated, [0.025, 0.975])
    return {
        "analysis_unit": "material_size_physical_specimen",
        "nesting": "horizons_within_recordings_within_specimens",
        "stratification": "material",
        "specimens": len(specimen_rows),
        "observed_by_specimen_m2": observed,
        "arm_a_specimen_win_fraction": float(
            np.mean(np.asarray(list(observed.values()), dtype=np.float64) < 0.0)
        ),
        "mean_m2": point,
        "lower_m2": float(lower),
        "upper_m2": float(upper),
        "draws": int(draws),
        "interpretation": (
            "Descriptive released-specimen interval only; four specimens do not "
            "support broad cloth-population inference."
        ),
    }


def batches_by_key(
    recordings: Sequence[Recording],
    request: Mapping[str, Any],
) -> dict[tuple[str, float], list[WindowBatch]]:
    result: dict[tuple[str, float], list[WindowBatch]] = {}
    for recording in recordings:
        if recording.key.scenario not in PRIMARY_SCENARIOS:
            continue
        for horizon in request["horizon_seconds"]:
            batch = extract_windows(
                recording,
                horizon_seconds=float(horizon),
                lag_seconds=float(request["lag_seconds"]),
                stride_seconds=float(request["stride_seconds"]),
            )
            result.setdefault((recording.key.scenario, float(horizon)), []).append(
                batch
            )
    return result


def read_materials(
    keys: Sequence[RecordingKey],
    materials: set[str],
) -> tuple[list[Recording], list[dict[str, Any]]]:
    recordings: list[Recording] = []
    manifest: list[dict[str, Any]] = []
    for key in keys:
        if key.material not in materials or key.scenario not in PRIMARY_SCENARIOS:
            continue
        recording = read_recording(key)
        recordings.append(recording)
        manifest.append(
            {
                "path": key.relative_path,
                "bytes": key.path.stat().st_size,
                "sha256": sha256_file(key.path),
                "material": key.material,
                "scenario": key.scenario,
                "size": key.size,
                "frames": int(recording.points_m.shape[0]),
                "markers": int(recording.points_m.shape[1]),
                "unit_scale_to_m": recording.unit_scale_to_m,
            }
        )
    return recordings, manifest


def _subset_aggregate(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    values: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        subset = [row for row in rows if row[field] == value]
        if subset:
            result[value] = aggregate_rows(subset)
    return result


def _subset_specimen_aggregate(
    rows: Sequence[Mapping[str, Any]],
    field: str,
    values: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for value in values:
        subset = [row for row in rows if row[field] == value]
        if subset:
            result[value] = aggregate_specimens(subset)
    return result


def run_evaluation(root: Path, request: dict[str, Any]) -> dict[str, Any]:
    if root.resolve() != EXPECTED_ROOT.resolve():
        raise ValueError(f"root must resolve exactly to {EXPECTED_ROOT}")
    keys = discover_recordings(root)
    census: dict[str, Any] = {
        "classified_csv_count": len(keys),
        "by_material": {
            material: sum(key.material == material for key in keys)
            for material in MATERIALS
        },
        "by_scenario": {
            scenario: sum(key.scenario == scenario for key in keys)
            for scenario in (*PRIMARY_SCENARIOS, "hitting")
        },
        "target_paths_classified_from_names_only": [
            key.relative_path for key in keys if key.material in {"wool", "polyester"}
        ],
    }
    expected = request["expected_census"]
    if census["classified_csv_count"] != int(expected["classified_csv_count"]):
        raise ValueError(f"classified recording census changed: {census}")
    for material, count in expected["by_material"].items():
        if census["by_material"].get(material) != int(count):
            raise ValueError(f"material census changed: {census}")

    source_recordings, source_manifest = read_materials(keys, {"cotton"})
    source_batches = batches_by_key(source_recordings, request)
    decisions = choose_policies(source_batches, ridge=float(request["ridge"]))
    source_models = fit_prediction_models(
        source_batches,
        decisions,
        ridge=float(request["ridge"]),
    )

    gate_recordings, gate_manifest = read_materials(keys, {"denim"})
    gate_batches = batches_by_key(gate_recordings, request)
    gate_rows = score_batches(gate_batches, decisions, source_models)
    gate = source_gate(gate_rows, decisions, request)

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 2,
        "pilot_kind": PILOT_KIND,
        "request_id": request["request_id"],
        "dataset": {
            "name": "Tracking Cloth Deformation",
            "root": str(root),
            "census": census,
            "source_fit_materials": ["cotton"],
            "source_gate_materials": ["denim"],
            "target_materials": ["wool", "polyester"],
            "source_manifest": source_manifest,
            "gate_manifest": gate_manifest,
        },
        "design": {
            "candidate_groups": list(CANDIDATES),
            "primary_scenarios": list(PRIMARY_SCENARIOS),
            "horizon_seconds": request["horizon_seconds"],
            "lag_seconds": request["lag_seconds"],
            "stride_seconds": request["stride_seconds"],
            "query": (
                "future lower-third centroid xyz and RMS radius residual beyond "
                "exact constant velocity"
            ),
            "observation": "current two-marker acceleration-residual summary",
            "generic_policy_target": "current global affine residual field",
            "model_weighting": "each source recording has equal total regression weight",
            "covariance_estimation": (
                "source leave-one-recording-out predictive residual second moment"
            ),
            "source_gate_statistical_unit": (
                "complete recording after averaging registered horizons"
            ),
            "primary_target_statistical_unit": "material-size physical specimen",
            "target_nesting": "horizons within recordings within specimens",
            "target_specimen_count_if_opened": 4,
            "target_model_policy": "cotton-fitted models frozen; no post-gate refit",
        },
        "source_selection": decisions,
        "source_gate": gate,
        "source_gate_rows": gate_rows,
        "target_contents_opened": False,
        "target": None,
        "claim_boundary": [
            (
                "This is task-conditioned sparse marker observation, not physical "
                "intervention selection."
            ),
            "Cotton alone determines policy choices and fitted prediction models.",
            "Denim is an independent frozen source gate for those cotton-fitted models.",
            (
                "Wool and polyester contents remain unopened unless every denim gate "
                "criterion passes."
            ),
            (
                "The denim source gate uses complete recordings after averaging horizons; "
                "it is an operational gate, not a population confidence statement."
            ),
            (
                "Held-out target summaries use four material-size physical specimens, "
                "with horizons nested within recordings and recordings within specimens."
            ),
            (
                "Specimen bootstrap intervals are descriptive for the released specimens "
                "and do not establish population-level cloth-material generalization."
            ),
            "No paper claim is authorized automatically by this workflow.",
        ],
    }
    if not gate["passed"]:
        result["decision"] = "source-gate-failed-target-closed-v2"
        result["result_id"] = canonical_sha256(result)
        return result

    # Target payloads are opened only after the source gate above is fixed true.
    # The cotton-fitted model set is not refit with denim or target observations.
    target_recordings, target_manifest = read_materials(
        keys,
        {"wool", "polyester"},
    )
    target_batches = batches_by_key(target_recordings, request)
    target_rows = score_batches(target_batches, decisions, source_models)
    target = {
        "manifest": target_manifest,
        "rows": target_rows,
        "recording_rows": recording_level_rows(target_rows),
        "specimen_rows": specimen_level_rows(target_rows),
        "recording_aggregate": aggregate_rows(target_rows),
        "primary_statistical_unit": "material_size_physical_specimen",
        "primary_specimen_aggregate": aggregate_specimens(target_rows),
        "paired_task_minus_global_state_mse": _bootstrap_specimen_difference(
            target_rows,
            "task_conditioned",
            "global_state_conditioned",
            seed=20260831,
            draws=10000,
        ),
        "paired_task_minus_constant_velocity_mse": _bootstrap_specimen_difference(
            target_rows,
            "task_conditioned",
            "constant_velocity",
            seed=20260832,
            draws=10000,
        ),
        "paired_task_minus_dependence_destroyed_mse": _bootstrap_specimen_difference(
            target_rows,
            "task_conditioned",
            "dependence_destroyed",
            seed=20260833,
            draws=10000,
        ),
        "by_material": _subset_specimen_aggregate(
            target_rows,
            "material",
            ("wool", "polyester"),
        ),
        "by_size": _subset_specimen_aggregate(target_rows, "size", ("A2", "A3")),
    }
    result["dataset"]["target_manifest"] = target_manifest
    result["target_contents_opened"] = True
    result["target"] = target
    result["decision"] = "source-gate-passed-target-scored-review-required-v2"
    result["result_id"] = canonical_sha256(result)
    return result


def validate_result(result: Mapping[str, Any]) -> None:
    if result.get("schema") != RESULT_SCHEMA or result.get("schema_version") != 2:
        raise ValueError("unexpected result schema")
    if result.get("pilot_kind") != PILOT_KIND:
        raise ValueError("unexpected pilot kind")
    gate_passed = bool(result["source_gate"]["passed"])
    if bool(result["target_contents_opened"]) != gate_passed:
        raise ValueError("target access does not match source gate")
    if gate_passed and result["target"] is None:
        raise ValueError("passed source gate lacks target result")
    if not gate_passed and result["target"] is not None:
        raise ValueError("failed source gate contains target result")
    if gate_passed:
        target = result["target"]
        if target["primary_statistical_unit"] != "material_size_physical_specimen":
            raise ValueError("unexpected primary target statistical unit")
        if target["primary_specimen_aggregate"]["specimens"] != 4:
            raise ValueError("held-out target must contain exactly four specimens")
    supplied = result["result_id"]
    unhashed = dict(result)
    unhashed.pop("result_id")
    if supplied != canonical_sha256(unhashed):
        raise ValueError("result_id mismatch")


def write_summary(result: Mapping[str, Any], path: Path) -> None:
    gate = result["source_gate"]
    improvements = gate["relative_mse_improvements"]
    lines = [
        "# Tracking Cloth task-conditioned sparse-marker evaluation V2",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Decision: `{result['decision']}`",
        f"- Source gate passed: `{gate['passed']}`",
        f"- Target contents opened: `{result['target_contents_opened']}`",
        "",
        "## Denim source gate",
        "",
        (
            "- Task vs global-state policy MSE improvement: "
            f"`{improvements['vs_global_state']:.3%}`"
        ),
        (
            "- Task vs exact constant velocity MSE improvement: "
            f"`{improvements['vs_constant_velocity']:.3%}`"
        ),
        (
            "- Task vs dependence-destroyed MSE improvement: "
            f"`{improvements['vs_dependence_destroyed']:.3%}`"
        ),
        (
            "- Complete-recording win fraction vs global-state policy: "
            f"`{gate['task_vs_global_state_recording_win_fraction']:.3%}`"
        ),
        "",
    ]
    if result["target"] is not None:
        aggregate = result["target"]["primary_specimen_aggregate"]["arms"]
        lines.extend(
            [
                "## Held-out wool and polyester",
                "",
                "Primary unit: four material-size physical specimens.",
                "",
            ]
        )
        for arm in ARMS:
            row = aggregate[arm]
            lines.append(
                f"- **{arm}:** RMSE={row['equal_specimen_rmse_mm']:.6f} mm, "
                f"NLL={row['equal_specimen_gaussian_nll']:.6f}, "
                f"coverage={row['equal_specimen_marginal_90_coverage']:.3%}, "
                f"normalized joint NEES={row['equal_specimen_normalized_joint_nees']:.6f}"
            )
        difference = result["target"]["paired_task_minus_global_state_mse"]
        lines.extend(
            [
                "",
                "### Paired physical-specimen bootstrap contrast",
                "",
                (
                    "- Task minus global-state MSE: "
                    f"`{difference['mean_m2']:.9g}` m^2 "
                    f"[`{difference['lower_m2']:.9g}`, "
                    f"`{difference['upper_m2']:.9g}`]"
                ),
            ]
        )
    lines.extend(["", "## Claim boundary", ""])
    lines.extend(f"- {item}" for item in result["claim_boundary"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    request = load_request(args.request)
    result = run_evaluation(EXPECTED_ROOT, request)
    validate_result(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite {result_path}")
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_summary(result, args.output_dir / "summary.md")
    print(
        json.dumps(
            {
                "result_id": result["result_id"],
                "decision": result["decision"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
