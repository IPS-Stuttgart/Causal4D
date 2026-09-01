"""Forecast construction for PokeFlex realized-load prefix abduction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

from causal4d_public._pokeflex_realized_load_common import (
    BoolArray,
    FloatArray,
    PokeFlexRealizedLoadSourceConfig,
    _require,
    _sha256_file,
)


@dataclass(frozen=True)
class RealizedLoadTake:
    take_id: str
    root: Path
    robot_sha256: str
    frame_ids: NDArray[np.int64]
    force_axis_n: FloatArray
    tool_positions_m: FloatArray
    onset_index: int
    window_force_n: FloatArray
    window_tool_positions_m: FloatArray
    window_phase: FloatArray
    window_speed_m_per_frame: FloatArray


@dataclass(frozen=True)
class TargetKinematicConditioning:
    """Target-side information exposed to the forecast constructor."""

    take_id: str
    window_phase: FloatArray
    window_speed_m_per_frame: FloatArray


@dataclass(frozen=True)
class ForecastBundle:
    means: Mapping[str, FloatArray]
    variances: Mapping[str, FloatArray]
    contact_probabilities: Mapping[str, FloatArray]
    metadata: Mapping[str, Any]


def _valid_transform(value: Any) -> FloatArray:
    matrix = np.asarray(value, dtype=np.float64)
    _require(matrix.shape == (4, 4), "tool transform must be 4 x 4")
    _require(np.all(np.isfinite(matrix)), "tool transform is non-finite")
    _require(
        np.allclose(matrix[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6, rtol=0.0),
        "tool transform has an invalid homogeneous row",
    )
    rotation = matrix[:3, :3]
    _require(
        np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4, rtol=1e-4)
        and np.linalg.det(rotation) > 0.0,
        "tool transform has an invalid rotation",
    )
    return matrix


def _discover_take_root(dataset_root: Path, take_id: str) -> Path:
    direct = dataset_root / take_id
    candidates: list[Path] = []
    if (direct / "robot_data.json").is_file():
        candidates.append(direct)
    object_direct = dataset_root / "3dPrintedBunny" / take_id
    if (object_direct / "robot_data.json").is_file():
        candidates.append(object_direct)
    if not candidates:
        candidates = [
            path.parent
            for path in dataset_root.rglob("robot_data.json")
            if path.parent.name == take_id
        ]
    unique: dict[str, Path] = {}
    root_resolved = dataset_root.resolve()
    for candidate in candidates:
        resolved = candidate.resolve()
        _require(
            resolved == root_resolved or root_resolved in resolved.parents,
            f"take escapes dataset root: {take_id}",
        )
        _require(not candidate.is_symlink(), f"take root is a symlink: {take_id}")
        unique[str(resolved)] = resolved
    _require(len(unique) == 1, f"expected exactly one root for {take_id}, found {len(unique)}")
    return next(iter(unique.values()))


def _first_persistent_true(values: BoolArray, count: int) -> int:
    for start in range(0, len(values) - count + 1):
        if bool(np.all(values[start : start + count])):
            return start
    raise ValueError("no persistent contact onset found")


def _phase_and_speed(positions: FloatArray, path_phase_weight: float) -> tuple[FloatArray, FloatArray]:
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    speed = np.concatenate(([0.0], steps))
    cumulative = np.concatenate(([0.0], np.cumsum(steps)))
    if cumulative[-1] <= np.finfo(np.float64).eps:
        path_phase = np.linspace(0.0, 1.0, len(positions))
    else:
        path_phase = cumulative / cumulative[-1]
    time_phase = np.linspace(0.0, 1.0, len(positions))
    phase = path_phase_weight * path_phase + (1.0 - path_phase_weight) * time_phase
    return phase.astype(np.float64), speed.astype(np.float64)


def load_realized_load_take(
    dataset_root: str | Path,
    take_id: str,
    config: PokeFlexRealizedLoadSourceConfig,
) -> RealizedLoadTake:
    root = _discover_take_root(Path(dataset_root).resolve(), take_id)
    robot_path = root / "robot_data.json"
    payload = json.loads(robot_path.read_text(encoding="utf-8"))
    _require(isinstance(payload, list) and payload, "robot log is empty")
    records: dict[int, Mapping[str, Any]] = {}
    for item in payload:
        _require(isinstance(item, Mapping), "robot record is not an object")
        frame_raw = item.get("frame")
        _require(
            isinstance(frame_raw, (int, str)) and not isinstance(frame_raw, bool),
            "robot frame id is invalid",
        )
        frame = int(frame_raw)
        _require(frame >= 0 and frame not in records, "robot frame ids repeat")
        records[frame] = item
    frame_ids = np.asarray(sorted(records), dtype=np.int64)
    forces = []
    positions = []
    for frame in frame_ids:
        item = records[int(frame)]
        force = np.asarray(item.get("forces"), dtype=np.float64).reshape(-1)
        _require(
            len(force) > config.force_axis_index and np.all(np.isfinite(force)),
            "force vector is invalid",
        )
        forces.append(float(force[config.force_axis_index]))
        transform = _valid_transform(item.get("T_WT"))
        positions.append(transform[:3, 3].copy())
    force_axis = np.asarray(forces, dtype=np.float64)
    tool_positions = np.asarray(positions, dtype=np.float64)
    active = force_axis > config.contact_threshold_n
    onset = _first_persistent_true(active, config.onset_consecutive_frames)
    window_length = config.prefix_frame_count + config.forecast_horizon_frames
    stop = onset + window_length
    _require(stop <= len(force_axis), f"{take_id} has insufficient post-onset horizon")
    window_force = force_axis[onset:stop].copy()
    window_tool = tool_positions[onset:stop].copy()
    phase, speed = _phase_and_speed(window_tool, config.path_phase_weight)
    return RealizedLoadTake(
        take_id=take_id,
        root=root,
        robot_sha256=_sha256_file(robot_path),
        frame_ids=frame_ids,
        force_axis_n=force_axis,
        tool_positions_m=tool_positions,
        onset_index=onset,
        window_force_n=window_force,
        window_tool_positions_m=window_tool,
        window_phase=phase,
        window_speed_m_per_frame=speed,
    )


def _warp_profile(source: RealizedLoadTake, target_phase: FloatArray) -> FloatArray:
    phase = source.window_phase
    force = source.window_force_n
    keep = np.concatenate(([True], np.diff(phase) > 1e-12))
    phase_unique = phase[keep]
    force_unique = force[keep]
    if len(phase_unique) < 2:
        phase_unique = np.linspace(0.0, 1.0, len(force))
        force_unique = force
    return np.interp(target_phase, phase_unique, force_unique).astype(np.float64)


def _shift(values: FloatArray, delay_frames: int) -> FloatArray:
    indices = np.arange(len(values), dtype=np.int64) - int(delay_frames)
    indices = np.clip(indices, 0, len(values) - 1)
    return values[indices]


def _softmax(log_weights: FloatArray) -> FloatArray:
    shifted = log_weights - np.max(log_weights)
    weights = np.exp(shifted)
    total = float(np.sum(weights))
    _require(np.isfinite(total) and total > 0.0, "posterior weights are invalid")
    return weights / total


def _robust_scale(profiles: FloatArray, floor: float) -> FloatArray:
    center = np.median(profiles, axis=0)
    scale = 1.4826 * np.median(np.abs(profiles - center[None, :]), axis=0)
    return np.maximum(scale, floor).astype(np.float64)


def _student_log_likelihood(residual: FloatArray, scale: FloatArray, degrees: float) -> float:
    standardized = residual / scale
    terms = -0.5 * (degrees + 1.0) * np.log1p((standardized**2) / degrees)
    terms -= np.log(scale)
    return float(np.sum(terms))


def _normal_cdf(values: FloatArray) -> FloatArray:
    erf = np.vectorize(math.erf, otypes=[np.float64])
    return 0.5 * (1.0 + erf(values / math.sqrt(2.0)))


def _contact_probability(mean: FloatArray, variance: FloatArray, threshold: float) -> FloatArray:
    standard_deviation = np.sqrt(np.maximum(variance, 1e-12))
    probability = 1.0 - _normal_cdf((threshold - mean) / standard_deviation)
    return np.clip(probability, 1e-6, 1.0 - 1e-6)


def _kinematic_features(
    phase: FloatArray,
    speed_m_per_frame: FloatArray,
) -> FloatArray:
    length = len(phase)
    _require(speed_m_per_frame.shape == (length,), "kinematic speed has the wrong shape")
    time = np.linspace(0.0, 1.0, length)
    speed_scale = max(float(np.median(speed_m_per_frame)), 1e-8)
    speed = speed_m_per_frame / speed_scale
    return np.column_stack(
        (
            np.ones(length),
            time,
            time**2,
            phase,
            phase**2,
            speed,
        )
    ).astype(np.float64)


def _ridge_prediction(
    target: TargetKinematicConditioning,
    sources: Sequence[RealizedLoadTake],
    target_prefix: FloatArray,
    config: PokeFlexRealizedLoadSourceConfig,
) -> FloatArray:
    matrices = [
        _kinematic_features(
            source.window_phase,
            source.window_speed_m_per_frame,
        )
        for source in sources
    ]
    observations = [source.window_force_n for source in sources]
    design = np.concatenate(matrices, axis=0)
    response = np.concatenate(observations, axis=0)
    penalty = config.ridge_penalty * np.eye(design.shape[1])
    penalty[0, 0] = 0.0
    beta = np.linalg.solve(design.T @ design + penalty, design.T @ response)
    prediction = _kinematic_features(
        target.window_phase,
        target.window_speed_m_per_frame,
    ) @ beta
    prefix = config.prefix_frame_count
    prediction += float(np.median(target_prefix - prediction[:prefix]))
    return prediction.astype(np.float64)


def _linear_prediction(target_prefix: FloatArray, total_length: int) -> FloatArray:
    x_prefix = np.arange(len(target_prefix), dtype=np.float64)
    design = np.column_stack((np.ones(len(target_prefix)), x_prefix))
    beta = np.linalg.lstsq(design, target_prefix, rcond=None)[0]
    x = np.arange(total_length, dtype=np.float64)
    return (beta[0] + beta[1] * x).astype(np.float64)


def _posterior_from_profiles(
    target_prefix: FloatArray,
    evidence_profiles: FloatArray,
    outcome_profiles: FloatArray,
    template_variance: FloatArray,
    config: PokeFlexRealizedLoadSourceConfig,
) -> tuple[FloatArray, FloatArray, FloatArray, dict[str, Any]]:
    prefix = config.prefix_frame_count
    scale = _robust_scale(evidence_profiles[:, :prefix], config.likelihood_scale_floor_n)
    predictions: list[FloatArray] = []
    probabilities: list[FloatArray] = []
    metadata: list[dict[str, Any]] = []
    log_weights = []
    for template_index in range(len(evidence_profiles)):
        for delay in config.delay_grid_frames:
            evidence = _shift(evidence_profiles[template_index], int(delay))
            outcome = _shift(outcome_profiles[template_index], int(delay))
            for gain in config.gain_grid:
                offset = float(np.median(target_prefix - float(gain) * evidence[:prefix]))
                evidence_prediction = float(gain) * evidence + offset
                outcome_prediction = float(gain) * outcome + offset
                residual = target_prefix - evidence_prediction[:prefix]
                log_likelihood = _student_log_likelihood(
                    residual,
                    scale,
                    config.student_t_degrees_of_freedom,
                )
                log_prior = -0.5 * ((float(gain) - 1.0) / config.gain_prior_std) ** 2
                log_prior -= 0.5 * (float(delay) / config.delay_prior_std_frames) ** 2
                predictions.append(outcome_prediction)
                component_variance = (float(gain) ** 2) * template_variance
                probabilities.append(
                    _contact_probability(
                        outcome_prediction,
                        component_variance,
                        config.contact_threshold_n,
                    )
                )
                metadata.append(
                    {
                        "template_index": template_index,
                        "delay_frames": int(delay),
                        "gain": float(gain),
                        "offset_n": offset,
                    }
                )
                log_weights.append(log_likelihood + log_prior)
    weights = _softmax(np.asarray(log_weights, dtype=np.float64))
    prediction_array = np.stack(predictions)
    mean = np.einsum("c,ct->t", weights, prediction_array)
    component_variance = np.asarray(
        [(item["gain"] ** 2) * template_variance for item in metadata],
        dtype=np.float64,
    )
    variance = np.einsum(
        "c,ct->t",
        weights,
        component_variance + (prediction_array - mean[None, :]) ** 2,
    )
    variance = np.maximum(variance, config.predictive_variance_floor_n2)
    contact_probability = np.einsum("c,ct->t", weights, np.stack(probabilities))
    map_index = int(np.argmax(weights))
    summary = {
        "candidate_count": len(metadata),
        "posterior_entropy": float(-np.sum(weights * np.log(weights + 1e-300))),
        "effective_candidate_count": float(1.0 / np.sum(weights**2)),
        "map_candidate": metadata[map_index],
        "map_prediction": prediction_array[map_index],
        "map_variance": np.maximum(
            component_variance[map_index],
            config.predictive_variance_floor_n2,
        ),
        "map_contact_probability": probabilities[map_index],
    }
    return mean, variance, np.clip(contact_probability, 1e-6, 1.0 - 1e-6), summary


def build_forecast_bundle(
    target_prefix: FloatArray,
    target: TargetKinematicConditioning,
    sources: Sequence[RealizedLoadTake],
    config: PokeFlexRealizedLoadSourceConfig,
) -> ForecastBundle:
    """Construct predictions without receiving the target force suffix."""

    prefix = config.prefix_frame_count
    total_length = prefix + config.forecast_horizon_frames
    _require(target_prefix.shape == (prefix,), "target prefix has the wrong shape")
    _require(
        len(target.window_phase) == total_length,
        "target window has the wrong length",
    )
    _require(
        target.window_speed_m_per_frame.shape == (total_length,),
        "target kinematic speed has the wrong length",
    )
    _require(len(sources) >= 3, "at least three source takes are required")
    source_ids = tuple(source.take_id for source in sources)
    _require(target.take_id not in source_ids, "target appears in source roster")
    _require(len(set(source_ids)) == len(source_ids), "source take ids repeat")

    profiles = np.stack([_warp_profile(source, target.window_phase) for source in sources])
    template_variance = np.var(profiles, axis=0, ddof=1)
    template_variance = np.maximum(
        template_variance,
        config.predictive_variance_floor_n2,
    )
    proposed_mean, proposed_variance, proposed_contact, posterior = _posterior_from_profiles(
        target_prefix,
        profiles,
        profiles,
        template_variance,
        config,
    )
    permutation = np.roll(np.arange(len(profiles)), -1)
    destroyed_profiles = profiles[permutation]
    destroyed_mean, destroyed_variance, destroyed_contact, destroyed = (
        _posterior_from_profiles(
            target_prefix,
            profiles,
            destroyed_profiles,
            template_variance,
            config,
        )
    )
    mean_profile = np.mean(profiles, axis=0)
    mean_profile += float(np.median(target_prefix - mean_profile[:prefix]))
    persistence = np.repeat(float(target_prefix[-1]), total_length)
    linear = _linear_prediction(target_prefix, total_length)
    ridge = _ridge_prediction(target, sources, target_prefix, config)
    map_prediction = np.asarray(posterior.pop("map_prediction"), dtype=np.float64)
    map_variance = np.asarray(posterior.pop("map_variance"), dtype=np.float64)
    map_contact = np.asarray(posterior.pop("map_contact_probability"), dtype=np.float64)
    destroyed.pop("map_prediction")
    destroyed.pop("map_variance")
    destroyed.pop("map_contact_probability")

    means = {
        "persistence": persistence,
        "linear_extrapolation": linear,
        "mean_prefix_offset": mean_profile,
        "kinematic_ridge": ridge,
        "posterior_map": map_prediction,
        "posterior_mean": proposed_mean,
        "dependence_destroyed": destroyed_mean,
    }
    variances = {
        name: template_variance.copy()
        for name in (
            "persistence",
            "linear_extrapolation",
            "mean_prefix_offset",
            "kinematic_ridge",
        )
    }
    variances.update(
        {
            "posterior_map": map_variance,
            "posterior_mean": proposed_variance,
            "dependence_destroyed": destroyed_variance,
        }
    )
    contact_probabilities = {
        name: _contact_probability(
            mean,
            variances[name],
            config.contact_threshold_n,
        )
        for name, mean in means.items()
        if name not in {"posterior_map", "posterior_mean", "dependence_destroyed"}
    }
    contact_probabilities.update(
        {
            "posterior_map": map_contact,
            "posterior_mean": proposed_contact,
            "dependence_destroyed": destroyed_contact,
        }
    )
    return ForecastBundle(
        means=means,
        variances=variances,
        contact_probabilities=contact_probabilities,
        metadata={
            "source_take_ids": list(source_ids),
            "known_future_tool_kinematics_used": True,
            "target_force_suffix_used": False,
            "posterior": posterior,
            "dependence_destroyed": destroyed,
            "dependence_control_permutation": permutation.tolist(),
            "dependence_control_prefix_marginal_preserved": True,
            "dependence_control_suffix_marginal_preserved": True,
        },
    )
