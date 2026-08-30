"""Source-only Bayesian transport pilot for official Deform360 point clouds."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
import math
from pathlib import Path
import re
import tarfile
from typing import Any, cast

import numpy as np


PILOT_SCHEMA_VERSION = 1
PILOT_KIND = "Deform360OfficialPointCloudSourcePilot"
PILOT_CONFIG_KIND = "Deform360OfficialPointCloudSourcePilotConfig"
_FRAME_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _payload_sha256(payload: Mapping[str, Any], *, digest_field: str) -> str:
    canonical = dict(payload)
    canonical.pop(digest_field, None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _finite_float(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float} and type(value) is not bool and np.isfinite(value),
        f"{name} must be a finite number",
    )
    return float(value)


def _positive_int(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 1, f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class PilotConfig:
    """Locked controls for the official point-cloud source pilot."""

    protocol_id: str
    object_id: str
    source_episode_ids: tuple[int, ...]
    forbidden_episode_ids: tuple[int, ...]
    episode_actions: Mapping[int, str]
    horizon_frames: tuple[int, ...]
    reset_count: int
    maximum_points: int
    minimum_frames: int
    rho_grid_min: float
    rho_grid_max: float
    rho_grid_count: int
    prior_variance_floor: float
    likelihood_information_cap: int
    predictive_variance_floor_m2: float
    guard_minimum_win_fraction: float
    guard_minimum_relative_improvement: float
    guard_maximum_worst_episode_ratio: float
    decision_minimum_relative_improvement: float
    decision_minimum_episode_win_fraction: float
    decision_maximum_worst_episode_ratio: float

    def __post_init__(self) -> None:
        _require(bool(self.protocol_id), "protocol_id must be nonempty")
        _require(bool(self.object_id), "object_id must be nonempty")
        _require(
            len(self.source_episode_ids) >= 3
            and len(set(self.source_episode_ids)) == len(self.source_episode_ids)
            and all(
                type(value) is int and value >= 0
                for value in self.source_episode_ids
            ),
            "source_episode_ids must contain at least three unique "
            "nonnegative integers",
        )
        _require(
            len(set(self.forbidden_episode_ids)) == len(self.forbidden_episode_ids)
            and all(
                type(value) is int and value >= 0
                for value in self.forbidden_episode_ids
            ),
            "forbidden_episode_ids must be unique nonnegative integers",
        )
        _require(
            not set(self.source_episode_ids) & set(self.forbidden_episode_ids),
            "source and forbidden episode rosters overlap",
        )
        _require(
            set(self.episode_actions) == set(self.source_episode_ids)
            and all(bool(value) for value in self.episode_actions.values()),
            "episode_actions must exactly describe the source roster",
        )
        _require(
            len(self.horizon_frames) >= 1
            and tuple(sorted(set(self.horizon_frames))) == self.horizon_frames
            and all(type(value) is int and value >= 1 for value in self.horizon_frames),
            "horizon_frames must be strictly increasing positive integers",
        )
        _positive_int(self.reset_count, name="reset_count")
        _positive_int(self.maximum_points, name="maximum_points")
        _positive_int(self.minimum_frames, name="minimum_frames")
        _positive_int(self.rho_grid_count, name="rho_grid_count")
        _positive_int(
            self.likelihood_information_cap,
            name="likelihood_information_cap",
        )
        _require(self.rho_grid_count >= 3, "rho_grid_count must be at least three")
        _require(
            np.isfinite(self.rho_grid_min)
            and np.isfinite(self.rho_grid_max)
            and self.rho_grid_min < self.rho_grid_max,
            "rho grid bounds are invalid",
        )
        for name in (
            "prior_variance_floor",
            "predictive_variance_floor_m2",
            "guard_minimum_win_fraction",
            "guard_minimum_relative_improvement",
            "guard_maximum_worst_episode_ratio",
            "decision_minimum_relative_improvement",
            "decision_minimum_episode_win_fraction",
            "decision_maximum_worst_episode_ratio",
        ):
            value = _finite_float(getattr(self, name), name=name)
            _require(value >= 0.0, f"{name} must be nonnegative")
        _require(
            0.0 <= self.guard_minimum_win_fraction <= 1.0,
            "guard_minimum_win_fraction must be in [0, 1]",
        )
        _require(
            0.0 <= self.decision_minimum_episode_win_fraction <= 1.0,
            "decision_minimum_episode_win_fraction must be in [0, 1]",
        )
        _require(
            self.guard_maximum_worst_episode_ratio >= 1.0,
            "guard_maximum_worst_episode_ratio must be at least one",
        )
        _require(
            self.decision_maximum_worst_episode_ratio >= 1.0,
            "decision_maximum_worst_episode_ratio must be at least one",
        )
        _require(
            self.minimum_frames
            >= self.horizon_frames[-1] + self.reset_count + 3,
            "minimum_frames cannot support the reset and horizon ladder",
        )

    @property
    def rho_grid(self) -> np.ndarray:
        return np.linspace(
            self.rho_grid_min,
            self.rho_grid_max,
            self.rho_grid_count,
            dtype=np.float64,
        )


@dataclass(frozen=True)
class EpisodeSequence:
    """Deterministically subsampled persistent points for one source episode."""

    episode_id: int
    action: str
    frame_ids: np.ndarray
    positions_m: np.ndarray
    archive_record: Mapping[str, Any]

    def __post_init__(self) -> None:
        frames = np.asarray(self.frame_ids)
        positions = np.asarray(self.positions_m)
        _require(
            frames.ndim == 1
            and np.issubdtype(frames.dtype, np.integer)
            and len(frames) >= 1
            and np.all(np.diff(frames) > 0),
            "frame_ids must be a strictly increasing integer vector",
        )
        _require(
            positions.ndim == 3
            and positions.shape[0] == len(frames)
            and positions.shape[2] == 3
            and positions.shape[1] >= 1
            and np.all(np.isfinite(positions)),
            "positions_m must be finite with shape (T, N, 3)",
        )


def load_pilot_config(path: str | Path) -> tuple[PilotConfig, dict[str, Any]]:
    """Load and validate the content-addressed source-pilot lock."""

    config_path = Path(path).resolve()
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    _require(payload.get("schema_version") == 1, "pilot config schema changed")
    _require(
        payload.get("artifact_kind") == PILOT_CONFIG_KIND,
        "pilot config kind changed",
    )
    _require(
        payload.get("config_sha256")
        == _payload_sha256(payload, digest_field="config_sha256"),
        "pilot config checksum mismatch",
    )
    boundary = cast(Mapping[str, Any], payload.get("information_boundary"))
    _require(
        isinstance(boundary, Mapping)
        and boundary.get("source_only") is True
        and boundary.get("source_future_positions_allowed_for_scoring") is True
        and boundary.get("official_velocity_arrays_used") is False
        and boundary.get("forbidden_episode_payloads_read") is False
        and boundary.get("dataset_modified") is False
        and boundary.get("new_physical_data_collected") is False
        and boundary.get("paper_claim_authorized") is False,
        "pilot config opens a forbidden information or claim boundary",
    )
    actions_raw = payload.get("episode_actions")
    _require(isinstance(actions_raw, Mapping), "episode_actions are missing")
    actions: dict[int, str] = {}
    for raw_key, raw_value in actions_raw.items():
        _require(
            type(raw_key) is str
            and raw_key.isdigit()
            and type(raw_value) is str
            and bool(raw_value),
            "episode_actions contain an invalid entry",
        )
        actions[int(raw_key)] = raw_value
    decision_raw = payload.get("decision")
    _require(isinstance(decision_raw, Mapping), "decision controls are missing")
    _require(
        decision_raw.get("primary_horizon_frames")
        == max(cast(Sequence[int], payload["horizon_frames"])),
        "primary decision horizon differs from the maximum registered horizon",
    )
    config = PilotConfig(
        protocol_id=str(payload["protocol_id"]),
        object_id=str(payload["object_id"]),
        source_episode_ids=tuple(payload["source_episode_ids"]),
        forbidden_episode_ids=tuple(payload["forbidden_episode_ids"]),
        episode_actions=actions,
        horizon_frames=tuple(payload["horizon_frames"]),
        reset_count=payload["reset_count"],
        maximum_points=payload["maximum_points"],
        minimum_frames=payload["minimum_frames"],
        rho_grid_min=payload["rho_grid_min"],
        rho_grid_max=payload["rho_grid_max"],
        rho_grid_count=payload["rho_grid_count"],
        prior_variance_floor=payload["prior_variance_floor"],
        likelihood_information_cap=payload["likelihood_information_cap"],
        predictive_variance_floor_m2=payload["predictive_variance_floor_m2"],
        guard_minimum_win_fraction=payload["guard"]["minimum_win_fraction"],
        guard_minimum_relative_improvement=payload["guard"][
            "minimum_relative_improvement"
        ],
        guard_maximum_worst_episode_ratio=payload["guard"][
            "maximum_worst_episode_ratio"
        ],
        decision_minimum_relative_improvement=payload["decision"][
            "minimum_relative_improvement"
        ],
        decision_minimum_episode_win_fraction=payload["decision"][
            "minimum_episode_win_fraction"
        ],
        decision_maximum_worst_episode_ratio=payload["decision"][
            "maximum_worst_episode_ratio"
        ],
    )
    return config, payload


def _member_frame_id(name: str) -> int:
    match = _FRAME_PATTERN.search(Path(name).stem)
    _require(match is not None, f"point-cloud member lacks a frame index: {name}")
    return int(match.group(1))


def _point_indices(point_count: int, maximum_points: int) -> np.ndarray:
    _require(point_count >= 1, "point cloud is empty")
    count = min(point_count, maximum_points)
    if count == point_count:
        return np.arange(point_count, dtype=np.int64)
    indices = np.linspace(0, point_count - 1, count, dtype=np.int64)
    _require(len(np.unique(indices)) == count, "point subsample contains duplicates")
    return indices


def load_episode_archive(
    archive_path: str | Path,
    *,
    episode_id: int,
    action: str,
    maximum_points: int,
    minimum_frames: int,
) -> EpisodeSequence:
    """Read only persistent point positions from one official ``pcd_clean.tar``."""

    path = Path(archive_path).resolve()
    _require(path.is_file() and not path.is_symlink(), f"archive is missing: {path}")
    before = path.stat()
    archive_sha256 = _sha256_file(path)
    frame_ids: list[int] = []
    positions: list[np.ndarray] = []
    selected_indices: np.ndarray | None = None
    point_count: int | None = None
    available_keys: tuple[str, ...] | None = None

    with tarfile.open(path, mode="r:*") as archive:
        members = [
            member
            for member in archive.getmembers()
            if member.isfile() and member.name.lower().endswith(".npz")
        ]
        members.sort(key=lambda member: (_member_frame_id(member.name), member.name))
        _require(members, f"archive contains no NPZ point-cloud frames: {path}")
        parsed_ids = [_member_frame_id(member.name) for member in members]
        _require(
            len(set(parsed_ids)) == len(parsed_ids)
            and np.all(np.diff(parsed_ids) > 0),
            "point-cloud archive frame identities are duplicated or unordered",
        )
        for member, frame_id in zip(members, parsed_ids, strict=True):
            extracted = archive.extractfile(member)
            _require(
                extracted is not None,
                f"failed to read archive member: {member.name}",
            )
            raw = extracted.read()
            with np.load(BytesIO(raw), allow_pickle=False) as payload:
                keys = tuple(sorted(payload.files))
                _require("pts" in keys, f"point-cloud frame lacks 'pts': {member.name}")
                if available_keys is None:
                    available_keys = keys
                else:
                    _require(keys == available_keys, "point-cloud NPZ key set changed")
                points = np.asarray(payload["pts"], dtype=np.float64)
            _require(
                points.ndim == 2
                and points.shape[1] == 3
                and np.all(np.isfinite(points)),
                f"point positions are malformed: {member.name}",
            )
            if point_count is None:
                point_count = len(points)
                selected_indices = _point_indices(point_count, maximum_points)
            else:
                _require(len(points) == point_count, "persistent point count changed")
            assert selected_indices is not None
            frame_ids.append(frame_id)
            positions.append(points[selected_indices])

    after = path.stat()
    _require(
        (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
        "point-cloud archive changed while being read",
    )
    _require(len(frame_ids) >= minimum_frames, "point-cloud episode is too short")
    assert point_count is not None
    assert selected_indices is not None
    assert available_keys is not None
    record = {
        "path": str(path),
        "size_bytes": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256": archive_sha256,
        "frame_count": len(frame_ids),
        "first_frame_id": frame_ids[0],
        "last_frame_id": frame_ids[-1],
        "full_point_count": point_count,
        "selected_point_count": len(selected_indices),
        "selected_point_index_sha256": hashlib.sha256(
            selected_indices.astype("<i8", copy=False).tobytes()
        ).hexdigest(),
        "available_npz_keys": list(available_keys),
        "used_npz_keys": ["pts"],
        "official_velocity_arrays_used": False,
        "archive_extracted_to_dataset": False,
    }
    return EpisodeSequence(
        episode_id=episode_id,
        action=action,
        frame_ids=np.asarray(frame_ids, dtype=np.int64),
        positions_m=np.stack(positions).astype(np.float64, copy=False),
        archive_record=record,
    )


def select_reset_positions(
    frame_count: int,
    *,
    reset_count: int,
    maximum_horizon: int,
) -> tuple[int, ...]:
    """Choose reset positions from availability alone."""

    _positive_int(frame_count, name="frame_count")
    count = _positive_int(reset_count, name="reset_count")
    horizon = _positive_int(maximum_horizon, name="maximum_horizon")
    earliest = 3
    latest = frame_count - horizon - 1
    _require(latest >= earliest + count - 1, "episode cannot support reset ladder")
    if count == 1:
        return (earliest,)
    positions = tuple(
        earliest + (index * (latest - earliest)) // (count - 1)
        for index in range(count)
    )
    _require(
        len(set(positions)) == count
        and positions[0] == earliest
        and positions[-1] == latest,
        "availability-only reset selection failed",
    )
    return positions


def _episode_rho_and_residual(sequence: EpisodeSequence) -> tuple[float, float]:
    velocity = np.diff(sequence.positions_m, axis=0)
    _require(len(velocity) >= 2, "episode has too few velocity transitions")
    previous = velocity[:-1]
    following = velocity[1:]
    denominator = float(np.sum(previous * previous))
    if denominator <= np.finfo(np.float64).tiny:
        rho = 0.0
    else:
        rho = float(np.sum(previous * following) / denominator)
    residual = following - rho * previous
    return rho, float(np.mean(residual * residual))


def fit_training_prior(
    sequences: Sequence[EpisodeSequence],
    *,
    config: PilotConfig,
) -> dict[str, Any]:
    """Fit an equal-episode Gaussian prior over the damping coefficient."""

    _require(len(sequences) >= 2, "at least two training episodes are required")
    estimates = [_episode_rho_and_residual(sequence) for sequence in sequences]
    raw_rhos = np.asarray([row[0] for row in estimates], dtype=np.float64)
    rhos = np.clip(raw_rhos, config.rho_grid_min, config.rho_grid_max)
    residuals = np.asarray([row[1] for row in estimates], dtype=np.float64)
    prior_mean = float(np.mean(rhos))
    prior_variance = float(
        max(
            np.var(rhos, ddof=1) if len(rhos) > 1 else 0.0,
            config.prior_variance_floor,
        )
    )
    residual_variance = float(
        max(float(np.median(residuals)), config.predictive_variance_floor_m2)
    )
    grid = config.rho_grid
    log_weights = -0.5 * (grid - prior_mean) ** 2 / prior_variance
    log_weights -= float(np.max(log_weights))
    weights = np.exp(log_weights)
    weights /= float(np.sum(weights))
    return {
        "episode_ids": [sequence.episode_id for sequence in sequences],
        "episode_rho_estimates": raw_rhos.tolist(),
        "clipped_episode_rho_estimates": rhos.tolist(),
        "episode_residual_variance_m2": residuals.tolist(),
        "prior_mean": prior_mean,
        "prior_variance": prior_variance,
        "residual_variance_m2": residual_variance,
        "rho_grid": grid,
        "prior_weights": weights,
    }


def _posterior_weights(
    sequence: EpisodeSequence,
    reset_position: int,
    *,
    training_prior: Mapping[str, Any],
    config: PilotConfig,
) -> np.ndarray:
    grid = np.asarray(training_prior["rho_grid"], dtype=np.float64)
    prior = np.asarray(training_prior["prior_weights"], dtype=np.float64)
    prefix = sequence.positions_m[: reset_position + 1]
    velocity = np.diff(prefix, axis=0)
    _require(len(velocity) >= 2, "reset prefix has too few velocity transitions")
    previous = velocity[:-1]
    following = velocity[1:]
    residual = following[None] - grid[:, None, None, None] * previous[None]
    mean_square = np.mean(residual * residual, axis=(1, 2, 3))
    effective_count = min(len(previous), config.likelihood_information_cap)
    variance = float(training_prior["residual_variance_m2"])
    log_weights = np.log(np.maximum(prior, np.finfo(np.float64).tiny))
    log_weights -= 0.5 * effective_count * mean_square / variance
    log_weights -= float(np.max(log_weights))
    posterior = np.exp(log_weights)
    total = float(np.sum(posterior))
    _require(np.isfinite(total) and total > 0.0, "posterior normalization failed")
    return posterior / total


def _transport_factors(grid: np.ndarray, horizon: int) -> np.ndarray:
    powers = np.arange(horizon, dtype=np.int64)
    return np.sum(grid[:, None] ** powers[None], axis=1)


def _symmetric_chamfer_m(reference: np.ndarray, prediction: np.ndarray) -> float:
    delta = reference[:, None, :] - prediction[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    return 0.5 * float(
        np.mean(np.min(distances, axis=1)) + np.mean(np.min(distances, axis=0))
    )


def _point_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - reference
    return {
        "identity_rmse_m": float(np.sqrt(np.mean(error * error))),
        "symmetric_chamfer_m": _symmetric_chamfer_m(reference, prediction),
    }


def _probabilistic_metrics(
    reference: np.ndarray,
    mean: np.ndarray,
    variance: np.ndarray,
) -> dict[str, float]:
    safe_variance = np.maximum(variance, np.finfo(np.float64).tiny)
    error = reference - mean
    nll = 0.5 * (np.log(2.0 * math.pi * safe_variance) + error * error / safe_variance)
    radius = 1.6448536269514722 * np.sqrt(safe_variance)
    covered = np.abs(error) <= radius
    return {
        "marginal_gaussian_nll": float(np.mean(nll)),
        "marginal_90_coverage": float(np.mean(covered)),
        "mean_full_90_interval_width_m": float(np.mean(2.0 * radius)),
    }


def _forecast_reset(
    sequence: EpisodeSequence,
    reset_position: int,
    horizon: int,
    *,
    training_prior: Mapping[str, Any],
    config: PilotConfig,
) -> dict[str, Any]:
    posterior = _posterior_weights(
        sequence,
        reset_position,
        training_prior=training_prior,
        config=config,
    )
    grid = np.asarray(training_prior["rho_grid"], dtype=np.float64)
    factors = _transport_factors(grid, horizon)
    current = sequence.positions_m[reset_position]
    velocity = current - sequence.positions_m[reset_position - 1]
    component_predictions = current[None] + factors[:, None, None] * velocity[None]
    posterior_mean = np.tensordot(posterior, component_predictions, axes=(0, 0))
    centered = component_predictions - posterior_mean[None]
    between_variance = np.tensordot(posterior, centered * centered, axes=(0, 0))
    process_variance = horizon * float(training_prior["residual_variance_m2"])
    posterior_variance = np.maximum(
        between_variance + process_variance,
        config.predictive_variance_floor_m2,
    )
    map_index = int(np.argmax(posterior))
    map_prediction = component_predictions[map_index]
    constant_velocity = current + horizon * velocity
    persistence = current
    reference = sequence.positions_m[reset_position + horizon]
    posterior_metrics = {
        **_point_metrics(reference, posterior_mean),
        **_probabilistic_metrics(reference, posterior_mean, posterior_variance),
    }
    return {
        "reset_position": reset_position,
        "reset_frame_id": int(sequence.frame_ids[reset_position]),
        "horizon_frames": horizon,
        "evaluation_frame_id": int(sequence.frame_ids[reset_position + horizon]),
        "posterior_rho_mean": float(np.sum(posterior * grid)),
        "posterior_rho_standard_deviation": float(
            np.sqrt(np.sum(posterior * (grid - np.sum(posterior * grid)) ** 2))
        ),
        "posterior_map_rho": float(grid[map_index]),
        "posterior": posterior_metrics,
        "map": _point_metrics(reference, map_prediction),
        "constant_velocity": _point_metrics(reference, constant_velocity),
        "persistence": _point_metrics(reference, persistence),
    }


def evaluate_episode_candidate(
    sequence: EpisodeSequence,
    *,
    training_prior: Mapping[str, Any],
    config: PilotConfig,
) -> dict[int, dict[str, Any]]:
    """Evaluate one held episode using only prefixes at registered resets."""

    reset_positions = select_reset_positions(
        len(sequence.frame_ids),
        reset_count=config.reset_count,
        maximum_horizon=config.horizon_frames[-1],
    )
    rows: dict[int, dict[str, Any]] = {}
    for horizon in config.horizon_frames:
        reset_rows = [
            _forecast_reset(
                sequence,
                reset_position,
                horizon,
                training_prior=training_prior,
                config=config,
            )
            for reset_position in reset_positions
        ]
        models = ("posterior", "map", "constant_velocity", "persistence")
        model_metrics: dict[str, dict[str, float]] = {}
        for model in models:
            fields = tuple(cast(Mapping[str, float], reset_rows[0][model]))
            model_metrics[model] = {
                field: float(
                    np.mean(
                        [
                            cast(Mapping[str, float], row[model])[field]
                            for row in reset_rows
                        ]
                    )
                )
                for field in fields
            }
        rows[horizon] = {
            "horizon_frames": horizon,
            "reset_positions": list(reset_positions),
            "reset_frame_ids": [
                int(sequence.frame_ids[position]) for position in reset_positions
            ],
            "metrics": model_metrics,
            "mean_posterior_rho": float(
                np.mean([row["posterior_rho_mean"] for row in reset_rows])
            ),
            "mean_posterior_rho_standard_deviation": float(
                np.mean(
                    [row["posterior_rho_standard_deviation"] for row in reset_rows]
                )
            ),
            "reset_records": reset_rows,
        }
    return rows


def _fit_guard(
    training_sequences: Sequence[EpisodeSequence],
    horizon: int,
    *,
    config: PilotConfig,
) -> dict[str, Any]:
    rows = []
    for held_index, held in enumerate(training_sequences):
        inner_training = [
            sequence
            for index, sequence in enumerate(training_sequences)
            if index != held_index
        ]
        prior = fit_training_prior(inner_training, config=config)
        metrics = evaluate_episode_candidate(
            held,
            training_prior=prior,
            config=config,
        )[horizon]["metrics"]
        candidate = float(metrics["posterior"]["identity_rmse_m"])
        persistence = float(metrics["persistence"]["identity_rmse_m"])
        ratio = candidate / max(persistence, np.finfo(np.float64).tiny)
        rows.append(
            {
                "episode_id": held.episode_id,
                "candidate_identity_rmse_m": candidate,
                "persistence_identity_rmse_m": persistence,
                "candidate_to_persistence_ratio": ratio,
                "win": candidate < persistence,
            }
        )
    candidate_mean = float(np.mean([row["candidate_identity_rmse_m"] for row in rows]))
    persistence_mean = float(
        np.mean([row["persistence_identity_rmse_m"] for row in rows])
    )
    relative_improvement = (
        persistence_mean - candidate_mean
    ) / max(persistence_mean, np.finfo(np.float64).tiny)
    win_fraction = float(np.mean([row["win"] for row in rows]))
    worst_ratio = float(max(row["candidate_to_persistence_ratio"] for row in rows))
    accepted = bool(
        relative_improvement >= config.guard_minimum_relative_improvement
        and win_fraction >= config.guard_minimum_win_fraction
        and worst_ratio <= config.guard_maximum_worst_episode_ratio
    )
    return {
        "horizon_frames": horizon,
        "training_episode_count": len(rows),
        "relative_improvement": relative_improvement,
        "win_fraction": win_fraction,
        "worst_episode_ratio": worst_ratio,
        "minimum_relative_improvement": config.guard_minimum_relative_improvement,
        "minimum_win_fraction": config.guard_minimum_win_fraction,
        "maximum_worst_episode_ratio": config.guard_maximum_worst_episode_ratio,
        "accepted": accepted,
        "episode_records": rows,
    }


def _aggregate_horizon(
    episode_records: Sequence[Mapping[str, Any]],
    horizon: int,
    *,
    config: PilotConfig,
) -> dict[str, Any]:
    rows = [
        cast(Mapping[str, Any], record["horizons"][str(horizon)])
        for record in episode_records
    ]
    model_names = (
        "posterior",
        "guarded",
        "map",
        "constant_velocity",
        "persistence",
    )
    model_metrics: dict[str, dict[str, float]] = {}
    for model in model_names:
        metric_names = tuple(cast(Mapping[str, float], rows[0]["metrics"][model]))
        model_metrics[model] = {
            metric: float(
                np.mean(
                    [
                        cast(Mapping[str, float], row["metrics"][model])[metric]
                        for row in rows
                    ]
                )
            )
            for metric in metric_names
        }
    persistence = model_metrics["persistence"]["identity_rmse_m"]
    comparisons: dict[str, dict[str, float]] = {}
    for model in model_names[:-1]:
        candidate = model_metrics[model]["identity_rmse_m"]
        ratios = [
            cast(Mapping[str, float], row["metrics"][model])["identity_rmse_m"]
            / max(
                cast(Mapping[str, float], row["metrics"]["persistence"])[
                    "identity_rmse_m"
                ],
                np.finfo(np.float64).tiny,
            )
            for row in rows
        ]
        comparisons[model] = {
            "relative_improvement_vs_persistence": (persistence - candidate)
            / max(persistence, np.finfo(np.float64).tiny),
            "episode_win_fraction_vs_persistence": float(
                np.mean(np.asarray(ratios) < 1.0)
            ),
            "worst_episode_ratio_vs_persistence": float(max(ratios)),
        }
    return {
        "horizon_frames": horizon,
        "episode_count": len(rows),
        "metrics": model_metrics,
        "comparisons": comparisons,
        "guard_accept_fraction": float(
            np.mean([row["guard_accepted"] for row in rows])
        ),
        "exact_fallback_fraction": float(
            np.mean([row["exact_fallback"] for row in rows])
        ),
        "episode_records": [
            {
                "episode_id": int(record["episode_id"]),
                "action": str(record["action"]),
                "guard_accepted": bool(row["guard_accepted"]),
                "exact_fallback": bool(row["exact_fallback"]),
                "guarded_identity_rmse_m": float(
                    row["metrics"]["guarded"]["identity_rmse_m"]
                ),
                "persistence_identity_rmse_m": float(
                    row["metrics"]["persistence"]["identity_rmse_m"]
                ),
            }
            for record, row in zip(episode_records, rows, strict=True)
        ],
    }


def _build_decision(
    horizon_summaries: Mapping[str, Mapping[str, Any]],
    *,
    config: PilotConfig,
) -> dict[str, Any]:
    primary_key = str(config.horizon_frames[-1])
    primary = horizon_summaries[primary_key]
    guarded = cast(Mapping[str, float], primary["comparisons"]["guarded"])
    passed = bool(
        guarded["relative_improvement_vs_persistence"]
        >= config.decision_minimum_relative_improvement
        and guarded["episode_win_fraction_vs_persistence"]
        >= config.decision_minimum_episode_win_fraction
        and guarded["worst_episode_ratio_vs_persistence"]
        <= config.decision_maximum_worst_episode_ratio
    )
    return {
        "primary_horizon_frames": config.horizon_frames[-1],
        "minimum_relative_improvement": config.decision_minimum_relative_improvement,
        "minimum_episode_win_fraction": config.decision_minimum_episode_win_fraction,
        "maximum_worst_episode_ratio": config.decision_maximum_worst_episode_ratio,
        "passed": passed,
        "classification": (
            "source-only-real-point-cloud-positive-pilot"
            if passed
            else "source-only-real-point-cloud-bounded-or-negative-pilot"
        ),
        "paper_claim_authorized": False,
        "interpretation": (
            "A positive decision supports only a same-object, cross-action, "
            "source-only observed-reset pilot on released reconstructed point clouds. "
            "It does not establish held-out target transfer, unseen-object "
            "generalization, calibrated "
            "physical uncertainty, Prob4D provider competence, or deployment safety."
        ),
    }


def run_official_point_cloud_source_pilot(
    processed_root: str | Path,
    config_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """Run the locked leave-one-source-action-out real-data pilot."""

    config, config_payload = load_pilot_config(config_path)
    root = Path(processed_root).resolve()
    _require(root.is_dir(), f"processed Deform360 root is missing: {root}")
    object_root = root / config.object_id
    _require(object_root.is_dir(), f"processed object is missing: {object_root}")

    sequences = []
    for episode_id in config.source_episode_ids:
        archive_path = object_root / f"episode_{episode_id}" / "pcd_clean.tar"
        sequences.append(
            load_episode_archive(
                archive_path,
                episode_id=episode_id,
                action=config.episode_actions[episode_id],
                maximum_points=config.maximum_points,
                minimum_frames=config.minimum_frames,
            )
        )

    forbidden_paths = [
        str(object_root / f"episode_{episode_id}" / "pcd_clean.tar")
        for episode_id in config.forbidden_episode_ids
    ]
    episode_records = []
    for held_index, held in enumerate(sequences):
        training = [
            sequence
            for index, sequence in enumerate(sequences)
            if index != held_index
        ]
        prior = fit_training_prior(training, config=config)
        candidate_horizons = evaluate_episode_candidate(
            held,
            training_prior=prior,
            config=config,
        )
        guards = {
            horizon: _fit_guard(training, horizon, config=config)
            for horizon in config.horizon_frames
        }
        horizon_records: dict[str, Any] = {}
        for horizon in config.horizon_frames:
            candidate_row = candidate_horizons[horizon]
            metrics = dict(candidate_row["metrics"])
            accepted = bool(guards[horizon]["accepted"])
            if accepted:
                guarded_metrics = dict(metrics["posterior"])
                exact_fallback = False
            else:
                guarded_metrics = dict(metrics["persistence"])
                exact_fallback = True
                _require(
                    guarded_metrics == metrics["persistence"],
                    "exact fallback metrics changed",
                )
            metrics["guarded"] = guarded_metrics
            horizon_records[str(horizon)] = {
                **candidate_row,
                "metrics": metrics,
                "guard": guards[horizon],
                "guard_accepted": accepted,
                "exact_fallback": exact_fallback,
            }
        episode_records.append(
            {
                "episode_id": held.episode_id,
                "action": held.action,
                "archive": dict(held.archive_record),
                "training_episode_ids": [sequence.episode_id for sequence in training],
                "training_prior": {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in prior.items()
                },
                "horizons": horizon_records,
            }
        )

    summaries = {
        str(horizon): _aggregate_horizon(episode_records, horizon, config=config)
        for horizon in config.horizon_frames
    }
    decision = _build_decision(summaries, config=config)
    output = Path(output_path).resolve()
    payload: dict[str, Any] = {
        "schema_version": PILOT_SCHEMA_VERSION,
        "artifact_kind": PILOT_KIND,
        "protocol_id": config.protocol_id,
        "config": {
            "path": str(Path(config_path).resolve()),
            "file_sha256": _sha256_file(Path(config_path).resolve()),
            "config_sha256": config_payload["config_sha256"],
        },
        "dataset": {
            "processed_root": str(root),
            "object_id": config.object_id,
            "source_episode_ids": list(config.source_episode_ids),
            "source_actions": {
                str(episode_id): config.episode_actions[episode_id]
                for episode_id in config.source_episode_ids
            },
            "forbidden_episode_ids": list(config.forbidden_episode_ids),
            "forbidden_archive_paths_not_opened": forbidden_paths,
            "official_velocity_arrays_used": False,
            "persistent_point_identity_assumption": (
                "The released pcd_clean contract seeds one fixed point set and advects "
                "it through each episode; the pilot verifies constant point count."
            ),
        },
        "method": {
            "name": "generalized-Bayesian damped local transport",
            "state": (
                "current persistent point positions and causal finite-difference "
                "velocity"
            ),
            "latent_parameter": "scalar velocity persistence rho",
            "training": "equal-episode prior fitted on other registered source actions",
            "adaptation": (
                "prefix-only generalized likelihood with capped temporal information"
            ),
            "comparators": [
                "persistence",
                "constant_velocity",
                "posterior_map",
                "posterior_mean",
                "source-guarded posterior-or-exact-persistence",
            ],
            "statistical_unit": "source episode/action",
            "points_frames_and_coordinates_are_nested": True,
        },
        "episode_records": episode_records,
        "horizon_summaries": summaries,
        "decision": decision,
        "information_boundary": {
            "source_only": True,
            "source_episode_payloads_read": True,
            "forbidden_episode_payloads_read": False,
            "official_velocity_arrays_used": False,
            "only_causal_prefix_used_for_held_episode_adaptation": True,
            "dataset_modified": False,
            "new_physical_data_collected": False,
            "paper_claim_authorized": False,
        },
    }
    payload["result_sha256"] = _payload_sha256(payload, digest_field="result_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return payload


def validate_official_point_cloud_source_pilot(payload: Mapping[str, Any]) -> None:
    """Validate result identity and source-only evidence boundaries."""

    _require(
        payload.get("schema_version") == PILOT_SCHEMA_VERSION,
        "result schema changed",
    )
    _require(payload.get("artifact_kind") == PILOT_KIND, "result kind changed")
    _require(
        payload.get("result_sha256")
        == _payload_sha256(payload, digest_field="result_sha256"),
        "result checksum mismatch",
    )
    boundary = payload.get("information_boundary")
    _require(isinstance(boundary, Mapping), "result boundary is missing")
    _require(
        boundary.get("source_only") is True
        and boundary.get("forbidden_episode_payloads_read") is False
        and boundary.get("official_velocity_arrays_used") is False
        and boundary.get("only_causal_prefix_used_for_held_episode_adaptation") is True
        and boundary.get("dataset_modified") is False
        and boundary.get("new_physical_data_collected") is False
        and boundary.get("paper_claim_authorized") is False,
        "result crossed a source, data, or claim boundary",
    )
    episodes = payload.get("episode_records")
    _require(
        isinstance(episodes, list) and len(episodes) >= 3,
        "episode records are missing",
    )
    episode_ids = [
        record.get("episode_id")
        for record in episodes
        if isinstance(record, Mapping)
    ]
    _require(len(episode_ids) == len(episodes), "episode record is malformed")
    _require(
        len(set(episode_ids)) == len(episode_ids),
        "episode records are duplicated",
    )
    dataset = payload.get("dataset")
    _require(isinstance(dataset, Mapping), "dataset record is missing")
    _require(
        episode_ids == dataset.get("source_episode_ids"),
        "result episode roster differs from the source roster",
    )
    _require(
        not set(episode_ids)
        & set(cast(Sequence[int], dataset.get("forbidden_episode_ids"))),
        "result includes a forbidden episode",
    )
    _require(
        payload.get("decision", {}).get("paper_claim_authorized") is False,
        "pilot result authorized a paper claim",
    )


__all__ = [
    "EpisodeSequence",
    "PILOT_CONFIG_KIND",
    "PILOT_KIND",
    "PILOT_SCHEMA_VERSION",
    "PilotConfig",
    "evaluate_episode_candidate",
    "fit_training_prior",
    "load_episode_archive",
    "load_pilot_config",
    "run_official_point_cloud_source_pilot",
    "select_reset_positions",
    "validate_official_point_cloud_source_pilot",
]
