"""Source-gated task-conditioned sparse-marker evaluation on Tracking Cloth.

The experiment hides the current cloth state except for two selected markers.
Every arm shares the same two earlier full marker frames and constant-velocity
forecast. Candidate observations are valued on cotton recordings only. Denim is
an independent source gate. Wool and polyester file contents remain unopened
unless the gate passes.

This is a real-data task-conditioned *observation* experiment, not a robot probe,
not a cloth simulator, and not a claim of independent physical-cloth population
generalization. Complete recordings are the analysis units; windows are nested.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

PILOT_KIND = "TrackingClothTaskConditionedObservationV1"
RESULT_SCHEMA = "causal4d.tracking-cloth-query-observation-result"
REQUEST_SCHEMA = "causal4d.tracking-cloth-query-observation-request"
EXPECTED_ROOT = Path(
    "/home/github-runner/.cache/datasets/"
    "tracking-cloth-deformation-v1-zenodo-14644526"
)
MATERIALS = ("cotton", "denim", "wool", "polyester")
CANDIDATES = ("upper", "lower", "central", "lateral", "fast")
PRIMARY_SCENARIOS = ("shake", "twist", "table", "self_collision")


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
    if request.get("schema") != REQUEST_SCHEMA or request.get("schema_version") != 1:
        raise ValueError("unsupported request schema")
    if request.get("stage") != "source-gated-evaluation":
        raise ValueError("unsupported stage")
    if Path(str(request.get("dataset_root"))) != EXPECTED_ROOT:
        raise ValueError(f"dataset_root must be exactly {EXPECTED_ROOT}")
    if request.get("source_fit_materials") != ["cotton"]:
        raise ValueError("source_fit_materials must be exactly cotton")
    if request.get("source_gate_materials") != ["denim"]:
        raise ValueError("source_gate_materials must be exactly denim")
    if request.get("target_materials") != ["wool", "polyester"]:
        raise ValueError("target_materials must be exactly wool and polyester")
    if request.get("candidate_groups") != list(CANDIDATES):
        raise ValueError("candidate marker groups changed")
    if request.get("primary_scenarios") != list(PRIMARY_SCENARIOS):
        raise ValueError("primary scenario roster changed")
    boundary = request.get("information_boundary")
    expected_boundary = {
        "target_file_contents_opened_before_source_gate": False,
        "rgb_or_depth_data_used": False,
        "physical_command_sent": False,
        "dataset_modified": False,
        "paper_claim_authorized": False,
    }
    if boundary != expected_boundary:
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
    y: FloatArray
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
    return RecordingKey(
        path=path,
        relative_path=relative,
        material=material,
        scenario=scenario,
        size=size,
    )


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
    if not text:
        return math.nan
    return float(text)


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
    start = None
    width = None
    markers = None
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
        raise ValueError(f"too few numeric frames in {key.relative_path}: {len(parsed)}")
    matrix = np.asarray(parsed, dtype=np.float64)
    times = matrix[:, 1]
    if not np.all(np.diff(times) > 0.0):
        raise ValueError(f"timestamps are not strictly increasing in {key.relative_path}")
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


def marker_groups(previous: FloatArray, older: FloatArray) -> dict[str, IntArray]:
    finite = np.all(np.isfinite(previous), axis=1) & np.all(np.isfinite(older), axis=1)
    indices = np.flatnonzero(finite)
    if indices.size < 8:
        raise ValueError("fewer than eight usable markers")
    p = previous[indices]
    o = older[indices]
    centroid = np.mean(p, axis=0)
    speed = np.linalg.norm(p - o, axis=1)
    centered = p - centroid
    _, _, right = np.linalg.svd(centered, full_matrices=False)
    lateral_coordinate = centered @ right[0]
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
        name: np.asarray(indices[group], dtype=np.int64)
        for name, group in local.items()
    }


def lower_query_indices(previous: FloatArray, older: FloatArray) -> IntArray:
    finite = np.all(np.isfinite(previous), axis=1) & np.all(np.isfinite(older), axis=1)
    indices = np.flatnonzero(finite)
    if indices.size < 8:
        raise ValueError("fewer than eight usable markers")
    count = max(4, int(math.ceil(indices.size / 3.0)))
    order = np.argsort(previous[indices, 2], kind="stable")
    return np.asarray(np.sort(indices[order[:count]]), dtype=np.int64)


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
    direction = (
        separation / norm if norm > 1e-12 else np.array([1.0, 0.0, 0.0])
    )
    differential = float((residual[1] - residual[0]) @ direction)
    return np.asarray(
        [mean[0], mean[1], mean[2], differential],
        dtype=np.float64,
    )


def query_vector(points: FloatArray, indices: IntArray) -> FloatArray:
    selected = points[indices]
    centroid = np.mean(selected, axis=0)
    radius = float(np.sqrt(np.mean(np.sum((selected - centroid) ** 2, axis=1))))
    return np.asarray(
        [centroid[0], centroid[1], centroid[2], radius],
        dtype=np.float64,
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
    y_rows: list[FloatArray] = []
    baseline_rows: list[FloatArray] = []
    actual_rows: list[FloatArray] = []
    for current in range(2 * lag, points.shape[0] - horizon, stride):
        older = points[current - 2 * lag]
        previous = points[current - lag]
        now = points[current]
        future = points[current + horizon]
        try:
            groups = marker_groups(previous, older)
            query_indices = lower_query_indices(previous, older)
        except ValueError:
            continue
        candidate_indices = np.unique(np.concatenate(tuple(groups.values())))
        required = np.unique(np.concatenate((query_indices, candidate_indices)))
        if not (
            np.all(np.isfinite(older[required]))
            and np.all(np.isfinite(previous[required]))
            and np.all(np.isfinite(now[candidate_indices]))
            and np.all(np.isfinite(future[query_indices]))
        ):
            continue
        velocity_per_lag = previous - older
        predicted_now = previous + velocity_per_lag
        horizon_multiplier = 1.0 + horizon / lag
        predicted_future = previous + horizon_multiplier * velocity_per_lag
        baseline_query = query_vector(predicted_future, query_indices)
        actual_query = query_vector(future, query_indices)
        for name, indices in groups.items():
            x_rows[name].append(
                group_feature(now, predicted_now, previous, indices)
            )
        y_rows.append(actual_query - baseline_query)
        baseline_rows.append(baseline_query)
        actual_rows.append(actual_query)
    if len(y_rows) < 12:
        raise ValueError(
            f"too few valid windows in {recording.key.relative_path} at "
            f"{horizon_seconds}s: {len(y_rows)}"
        )
    return WindowBatch(
        recording=recording.key,
        horizon_seconds=float(horizon_seconds),
        x_by_candidate={name: np.stack(rows) for name, rows in x_rows.items()},
        y=np.stack(y_rows),
        baseline_query=np.stack(baseline_rows),
        actual_query=np.stack(actual_rows),
    )


def fit_model(x: FloatArray, y: FloatArray, *, ridge: float) -> LinearGaussianModel:
    if (
        x.ndim != 2
        or y.ndim != 2
        or x.shape[0] != y.shape[0]
        or x.shape[0] < 8
    ):
        raise ValueError("invalid regression arrays")
    x_mean = np.mean(x, axis=0)
    x_scale = np.std(x, axis=0, ddof=1)
    x_scale = np.maximum(x_scale, 1e-9)
    y_mean = np.mean(y, axis=0)
    standardized = (x - x_mean) / x_scale
    centered_y = y - y_mean
    penalty = float(ridge) * np.eye(standardized.shape[1])
    coefficient = np.linalg.solve(
        standardized.T @ standardized + penalty,
        standardized.T @ centered_y,
    )
    residual = centered_y - standardized @ coefficient
    covariance = residual.T @ residual / max(residual.shape[0] - 1, 1)
    floor = (
        max(
            float(np.trace(covariance)) / max(covariance.shape[0], 1),
            1e-12,
        )
        * 1e-6
    )
    covariance = (
        0.5 * (covariance + covariance.T)
        + floor * np.eye(covariance.shape[0])
    )
    return LinearGaussianModel(
        x_mean,
        x_scale,
        y_mean,
        coefficient,
        covariance,
    )


def fit_no_observation(y: FloatArray) -> tuple[FloatArray, FloatArray]:
    mean = np.mean(y, axis=0)
    residual = y - mean
    covariance = residual.T @ residual / max(y.shape[0] - 1, 1)
    floor = (
        max(
            float(np.trace(covariance)) / max(covariance.shape[0], 1),
            1e-12,
        )
        * 1e-6
    )
    covariance = (
        0.5 * (covariance + covariance.T)
        + floor * np.eye(covariance.shape[0])
    )
    return mean, covariance


def _stack_batches(
    batches: Iterable[WindowBatch],
    candidate: str,
) -> tuple[FloatArray, FloatArray]:
    selected = list(batches)
    if not selected:
        raise ValueError("no batches to stack")
    return (
        np.concatenate(
            [batch.x_by_candidate[candidate] for batch in selected],
            axis=0,
        ),
        np.concatenate([batch.y for batch in selected], axis=0),
    )


def _recording_mse(error: FloatArray) -> float:
    return float(np.mean(np.sum(error**2, axis=1) / error.shape[1]))


def _entropy_score(x: FloatArray) -> float:
    covariance = np.cov(x, rowvar=False)
    scale = max(float(np.trace(covariance)) / covariance.shape[0], 1e-12)
    sign, logdet = np.linalg.slogdet(
        covariance + scale * 1e-6 * np.eye(covariance.shape[0])
    )
    return float(logdet) if sign > 0 else -math.inf


def choose_groups(
    source_batches: dict[tuple[str, float], list[WindowBatch]],
    *,
    ridge: float,
) -> dict[str, Any]:
    decisions: dict[str, Any] = {}
    for (scenario, horizon), batches in sorted(source_batches.items()):
        if len(batches) < 2:
            continue
        task_scores: dict[str, float] = {}
        entropy_scores: dict[str, float] = {}
        for candidate in CANDIDATES:
            per_recording: list[float] = []
            for held_out in range(len(batches)):
                training = [
                    batch
                    for index, batch in enumerate(batches)
                    if index != held_out
                ]
                x_train, y_train = _stack_batches(training, candidate)
                model = fit_model(x_train, y_train, ridge=ridge)
                batch = batches[held_out]
                error = batch.y - model.predict(
                    batch.x_by_candidate[candidate]
                )
                per_recording.append(_recording_mse(error))
            task_scores[candidate] = float(np.mean(per_recording))
            x_all, _ = _stack_batches(batches, candidate)
            entropy_scores[candidate] = _entropy_score(x_all)
        task = min(CANDIDATES, key=lambda name: (task_scores[name], name))
        generic = max(
            CANDIDATES,
            key=lambda name: (entropy_scores[name], name),
        )
        decisions[f"{scenario}/{horizon:.6g}"] = {
            "scenario": scenario,
            "horizon_seconds": horizon,
            "task_selected": task,
            "generic_selected": generic,
            "task_cv_mse_by_candidate_m2": task_scores,
            "generic_logdet_by_candidate": entropy_scores,
        }
    return decisions


def _permutation_indices(count: int, token: str) -> IntArray:
    seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    return np.asarray(rng.permutation(count), dtype=np.int64)


def build_models(
    training_batches: dict[tuple[str, float], list[WindowBatch]],
    decisions: dict[str, Any],
    *,
    ridge: float,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for (scenario, horizon), batches in sorted(training_batches.items()):
        key = f"{scenario}/{horizon:.6g}"
        if key not in decisions:
            continue
        y_all = np.concatenate([batch.y for batch in batches], axis=0)
        no_mean, no_covariance = fit_no_observation(y_all)
        candidate_models: dict[str, LinearGaussianModel] = {}
        for candidate in CANDIDATES:
            x_all, _ = _stack_batches(batches, candidate)
            candidate_models[candidate] = fit_model(
                x_all,
                y_all,
                ridge=ridge,
            )
        destroyed_candidate = decisions[key]["task_selected"]
        x_destroyed, _ = _stack_batches(batches, destroyed_candidate)
        permutation = _permutation_indices(y_all.shape[0], f"destroy/{key}")
        destroyed = fit_model(
            x_destroyed,
            y_all[permutation],
            ridge=ridge,
        )
        models[key] = {
            "no_mean": no_mean,
            "no_covariance": no_covariance,
            "candidate_models": candidate_models,
            "destroyed": destroyed,
        }
    return models


def _gaussian_metrics(
    error: FloatArray,
    covariance: FloatArray,
) -> tuple[float, float]:
    root = np.linalg.cholesky(covariance)
    whitened = np.linalg.solve(root, error.T).T
    nll = 0.5 * (
        error.shape[1] * math.log(2.0 * math.pi)
        + 2.0 * float(np.sum(np.log(np.diag(root))))
        + np.sum(whitened**2, axis=1)
    )
    std = np.sqrt(np.diag(covariance))
    coverage = np.mean(
        np.abs(error) <= 1.6448536269514722 * std[None, :]
    )
    return float(np.mean(nll)), float(coverage)


def score_batches(
    batches: dict[tuple[str, float], list[WindowBatch]],
    decisions: dict[str, Any],
    models: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (scenario, horizon), group in sorted(batches.items()):
        key = f"{scenario}/{horizon:.6g}"
        if key not in models:
            continue
        decision = decisions[key]
        fitted = models[key]
        task_group = str(decision["task_selected"])
        generic_group = str(decision["generic_selected"])
        for batch in group:
            token = f"random/{batch.recording.relative_path}/{key}"
            seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:16], 16)
            random_group = CANDIDATES[seed % len(CANDIDATES)]
            arms: dict[str, tuple[FloatArray, FloatArray]] = {
                "constant_velocity": (
                    np.broadcast_to(fitted["no_mean"], batch.y.shape),
                    fitted["no_covariance"],
                ),
                "task_conditioned": (
                    fitted["candidate_models"][task_group].predict(
                        batch.x_by_candidate[task_group]
                    ),
                    fitted["candidate_models"][task_group].covariance,
                ),
                "generic_information": (
                    fitted["candidate_models"][generic_group].predict(
                        batch.x_by_candidate[generic_group]
                    ),
                    fitted["candidate_models"][generic_group].covariance,
                ),
                "fixed_upper": (
                    fitted["candidate_models"]["upper"].predict(
                        batch.x_by_candidate["upper"]
                    ),
                    fitted["candidate_models"]["upper"].covariance,
                ),
                "random_cost_matched": (
                    fitted["candidate_models"][random_group].predict(
                        batch.x_by_candidate[random_group]
                    ),
                    fitted["candidate_models"][random_group].covariance,
                ),
                "dependence_destroyed": (
                    fitted["destroyed"].predict(
                        batch.x_by_candidate[task_group]
                    ),
                    fitted["destroyed"].covariance,
                ),
            }
            arm_rows: dict[str, Any] = {}
            for name, (predicted_residual, covariance) in arms.items():
                error = batch.y - predicted_residual
                nll, coverage = _gaussian_metrics(error, covariance)
                mse = _recording_mse(error)
                arm_rows[name] = {
                    "mse_m2": mse,
                    "rmse_mm": 1000.0 * math.sqrt(mse),
                    "gaussian_nll": nll,
                    "marginal_90_coverage": coverage,
                }
            rows.append(
                {
                    "recording": batch.recording.relative_path,
                    "material": batch.recording.material,
                    "scenario": scenario,
                    "size": batch.recording.size,
                    "horizon_seconds": horizon,
                    "windows": int(batch.y.shape[0]),
                    "task_selected": task_group,
                    "generic_selected": generic_group,
                    "random_selected": random_group,
                    "arms": arm_rows,
                }
            )
    return rows


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("no scored recording rows")
    arms = sorted(rows[0]["arms"])
    aggregate: dict[str, Any] = {"recordings": len(rows), "arms": {}}
    for arm in arms:
        mse = np.asarray(
            [row["arms"][arm]["mse_m2"] for row in rows],
            dtype=np.float64,
        )
        nll = np.asarray(
            [row["arms"][arm]["gaussian_nll"] for row in rows],
            dtype=np.float64,
        )
        coverage = np.asarray(
            [row["arms"][arm]["marginal_90_coverage"] for row in rows],
            dtype=np.float64,
        )
        aggregate["arms"][arm] = {
            "equal_recording_mse_m2": float(np.mean(mse)),
            "equal_recording_rmse_mm": 1000.0
            * math.sqrt(float(np.mean(mse))),
            "equal_recording_gaussian_nll": float(np.mean(nll)),
            "equal_recording_marginal_90_coverage": float(
                np.mean(coverage)
            ),
        }
    baseline = aggregate["arms"]["constant_velocity"][
        "equal_recording_mse_m2"
    ]
    for arm in arms:
        value = aggregate["arms"][arm]["equal_recording_mse_m2"]
        aggregate["arms"][arm][
            "relative_mse_improvement_vs_constant_velocity"
        ] = float((baseline - value) / baseline) if baseline > 0.0 else 0.0
    return aggregate


def source_gate(
    rows: list[dict[str, Any]],
    request: dict[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_rows(rows)
    task = aggregate["arms"]["task_conditioned"][
        "equal_recording_mse_m2"
    ]
    generic = aggregate["arms"]["generic_information"][
        "equal_recording_mse_m2"
    ]
    baseline = aggregate["arms"]["constant_velocity"][
        "equal_recording_mse_m2"
    ]
    task_vs_generic = (generic - task) / generic
    task_vs_baseline = (baseline - task) / baseline
    recording_differences = np.asarray(
        [
            row["arms"]["generic_information"]["mse_m2"]
            - row["arms"]["task_conditioned"]["mse_m2"]
            for row in rows
        ],
        dtype=np.float64,
    )
    win_fraction = float(np.mean(recording_differences > 0.0))
    ratios: dict[str, float] = {}
    for scenario in PRIMARY_SCENARIOS:
        subset = [row for row in rows if row["scenario"] == scenario]
        if not subset:
            continue
        task_s = float(
            np.mean(
                [
                    row["arms"]["task_conditioned"]["mse_m2"]
                    for row in subset
                ]
            )
        )
        generic_s = float(
            np.mean(
                [
                    row["arms"]["generic_information"]["mse_m2"]
                    for row in subset
                ]
            )
        )
        ratios[scenario] = task_s / generic_s if generic_s > 0 else math.inf
    distinct = sum(
        row["task_selected"] != row["generic_selected"] for row in rows
    )
    thresholds = request["source_gate"]
    checks = {
        "minimum_improvement_vs_generic": task_vs_generic
        >= float(thresholds["minimum_improvement_vs_generic"]),
        "minimum_improvement_vs_constant_velocity": task_vs_baseline
        >= float(thresholds["minimum_improvement_vs_constant_velocity"]),
        "minimum_recording_win_fraction": win_fraction
        >= float(thresholds["minimum_recording_win_fraction"]),
        "maximum_worst_scenario_ratio": max(
            ratios.values(),
            default=math.inf,
        )
        <= float(thresholds["maximum_worst_scenario_ratio"]),
        "minimum_distinct_selection_rows": distinct
        >= int(thresholds["minimum_distinct_selection_rows"]),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "task_vs_generic_relative_mse_improvement": float(task_vs_generic),
        "task_vs_constant_velocity_relative_mse_improvement": float(
            task_vs_baseline
        ),
        "task_vs_generic_recording_win_fraction": win_fraction,
        "task_to_generic_scenario_mse_ratios": ratios,
        "distinct_task_vs_generic_selection_rows": int(distinct),
        "aggregate": aggregate,
    }


def _bootstrap_difference(
    rows: list[dict[str, Any]],
    arm_a: str,
    arm_b: str,
    *,
    seed: int,
    draws: int,
) -> dict[str, float]:
    by_material: dict[str, list[float]] = {}
    for row in rows:
        difference = (
            row["arms"][arm_a]["mse_m2"]
            - row["arms"][arm_b]["mse_m2"]
        )
        by_material.setdefault(row["material"], []).append(float(difference))
    rng = np.random.default_rng(seed)
    simulated = np.empty(draws, dtype=np.float64)
    materials = sorted(by_material)
    for index in range(draws):
        material_means = []
        for material in materials:
            values = np.asarray(by_material[material], dtype=np.float64)
            sample = rng.choice(values, size=values.size, replace=True)
            material_means.append(float(np.mean(sample)))
        simulated[index] = float(np.mean(material_means))
    point = float(np.mean([np.mean(by_material[name]) for name in materials]))
    lower, upper = np.quantile(simulated, [0.025, 0.975])
    return {
        "mean_m2": point,
        "lower_m2": float(lower),
        "upper_m2": float(upper),
    }


def batches_by_key(
    recordings: list[Recording],
    request: dict[str, Any],
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
            result.setdefault(
                (recording.key.scenario, float(horizon)),
                [],
            ).append(batch)
    return result


def read_materials(
    keys: list[RecordingKey],
    materials: set[str],
) -> tuple[list[Recording], list[dict[str, Any]]]:
    recordings: list[Recording] = []
    manifest: list[dict[str, Any]] = []
    for key in keys:
        if (
            key.material not in materials
            or key.scenario not in PRIMARY_SCENARIOS
        ):
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
            key.relative_path
            for key in keys
            if key.material in {"wool", "polyester"}
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
    decisions = choose_groups(source_batches, ridge=float(request["ridge"]))
    source_models = build_models(
        source_batches,
        decisions,
        ridge=float(request["ridge"]),
    )

    gate_recordings, gate_manifest = read_materials(keys, {"denim"})
    gate_batches = batches_by_key(gate_recordings, request)
    gate_rows = score_batches(gate_batches, decisions, source_models)
    gate = source_gate(gate_rows, request)

    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "schema_version": 1,
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
                "future lower-third centroid xyz and RMS radius residual "
                "beyond constant velocity"
            ),
            "observation": "current two-marker acceleration residual summary",
            "statistical_unit": "complete recording; windows are nested",
        },
        "source_selection": decisions,
        "source_gate": gate,
        "source_gate_rows": gate_rows,
        "target_contents_opened": False,
        "target": None,
        "claim_boundary": [
            (
                "This is task-conditioned sparse marker observation, not "
                "physical intervention selection."
            ),
            "Cotton is method-development data and denim is the frozen source gate.",
            (
                "Wool and polyester contents remain unopened unless the denim "
                "gate passes."
            ),
            "Recordings, not windows or marker coordinates, are the analysis units.",
            (
                "The target contains two held-out materials but only a small "
                "number of physical cloth specimens."
            ),
            "No paper claim is authorized automatically by this workflow.",
        ],
    }
    if not gate["passed"]:
        result["decision"] = "source-gate-failed-target-closed"
        result["result_id"] = canonical_sha256(result)
        return result

    combined_recordings = source_recordings + gate_recordings
    combined_batches = batches_by_key(combined_recordings, request)
    target_models = build_models(
        combined_batches,
        decisions,
        ridge=float(request["ridge"]),
    )
    # Target payloads are opened only after the source gate above is fixed true.
    target_recordings, target_manifest = read_materials(
        keys,
        {"wool", "polyester"},
    )
    target_batches = batches_by_key(target_recordings, request)
    target_rows = score_batches(target_batches, decisions, target_models)
    target_aggregate = aggregate_rows(target_rows)
    target = {
        "manifest": target_manifest,
        "rows": target_rows,
        "aggregate": target_aggregate,
        "paired_task_minus_generic_mse": _bootstrap_difference(
            target_rows,
            "task_conditioned",
            "generic_information",
            seed=20260831,
            draws=10000,
        ),
        "paired_task_minus_constant_velocity_mse": _bootstrap_difference(
            target_rows,
            "task_conditioned",
            "constant_velocity",
            seed=20260832,
            draws=10000,
        ),
        "by_material": {
            material: aggregate_rows(
                [row for row in target_rows if row["material"] == material]
            )
            for material in ("wool", "polyester")
        },
    }
    result["dataset"]["target_manifest"] = target_manifest
    result["target_contents_opened"] = True
    result["target"] = target
    result["decision"] = "source-gate-passed-target-scored-review-required"
    result["result_id"] = canonical_sha256(result)
    return result


def validate_result(result: dict[str, Any]) -> None:
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("schema_version") != 1
    ):
        raise ValueError("unexpected result schema")
    gate_passed = bool(result["source_gate"]["passed"])
    if bool(result["target_contents_opened"]) != gate_passed:
        raise ValueError("target access does not match source gate")
    if gate_passed and result["target"] is None:
        raise ValueError("passed source gate lacks target result")
    if not gate_passed and result["target"] is not None:
        raise ValueError("failed source gate contains target result")
    supplied = result["result_id"]
    unhashed = dict(result)
    unhashed.pop("result_id")
    if supplied != canonical_sha256(unhashed):
        raise ValueError("result_id mismatch")


def write_summary(result: dict[str, Any], path: Path) -> None:
    gate = result["source_gate"]
    lines = [
        "# Tracking Cloth task-conditioned sparse-marker evaluation",
        "",
        f"- Result ID: `{result['result_id']}`",
        f"- Decision: `{result['decision']}`",
        f"- Source gate passed: `{gate['passed']}`",
        f"- Target contents opened: `{result['target_contents_opened']}`",
        "",
        "## Denim source gate",
        "",
        (
            "- Task vs generic MSE improvement: "
            f"`{gate['task_vs_generic_relative_mse_improvement']:.3%}`"
        ),
        (
            "- Task vs constant velocity MSE improvement: "
            f"`{gate['task_vs_constant_velocity_relative_mse_improvement']:.3%}`"
        ),
        (
            "- Recording win fraction vs generic: "
            f"`{gate['task_vs_generic_recording_win_fraction']:.3%}`"
        ),
        "",
    ]
    if result["target"] is not None:
        aggregate = result["target"]["aggregate"]["arms"]
        lines.extend(["## Held-out wool and polyester", ""])
        for arm in (
            "task_conditioned",
            "generic_information",
            "constant_velocity",
            "fixed_upper",
            "random_cost_matched",
            "dependence_destroyed",
        ):
            row = aggregate[arm]
            lines.append(
                f"- **{arm}:** RMSE={row['equal_recording_rmse_mm']:.6f} mm, "
                f"NLL={row['equal_recording_gaussian_nll']:.6f}, "
                f"coverage={row['equal_recording_marginal_90_coverage']:.3%}"
            )
        difference = result["target"]["paired_task_minus_generic_mse"]
        lines.extend(
            [
                "",
                "### Paired recording-bootstrap contrast",
                "",
                (
                    "- Task minus generic MSE: "
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
