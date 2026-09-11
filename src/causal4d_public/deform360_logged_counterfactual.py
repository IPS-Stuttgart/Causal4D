"""Source-only logged cross-intervention abduction on public Deform360 rope data.

This module does not claim individual-level real counterfactual ground truth.  It
uses repeated real interactions of the same physical object to test whether a
persistent dynamics posterior abducted from factual interaction A improves
prediction under a distinct recorded intervention B.  Event-specific contact is
provided by B and is never transferred from A.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_rope_dynamics import RopeDynamicsObservation, rollout_rope_dynamics
from .deform360_rope_fit import (
    RopeForwardFitConfig,
    _candidate_parameters,
    _mean_chamfer_m,
    _source_forecast_case,
)
from .deform360_rope_observations import (
    DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
    load_source_rope_dynamics_observation,
    rope_source_observation_artifact_sha256,
)


DEFORM360_LOGGED_COUNTERFACTUAL_SCHEMA_VERSION = 1


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def logged_counterfactual_artifact_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("result_sha256", None)
    return hashlib.sha256(_canonical_bytes(canonical)).hexdigest()


@dataclass(frozen=True)
class LoggedCounterfactualConfig:
    """Frozen source-only generalized-Bayes and promotion settings."""

    prefix_frame_count: int = 6
    minimum_temperature_m: float = 5.0e-4
    posterior_power: float = 1.0
    minimum_mean_improvement_vs_uniform_fraction: float = 0.02
    minimum_mean_improvement_vs_permuted_fraction: float = 0.02
    minimum_primary_win_fraction: float = 0.60
    maximum_primary_pair_ratio: float = 1.25

    def __post_init__(self) -> None:
        _require(
            self.prefix_frame_count >= 2, "prefix_frame_count must be at least two"
        )
        _require(
            np.isfinite(self.minimum_temperature_m)
            and self.minimum_temperature_m > 0.0,
            "minimum_temperature_m must be finite and positive",
        )
        _require(
            np.isfinite(self.posterior_power) and self.posterior_power > 0.0,
            "posterior_power must be finite and positive",
        )
        for name in (
            "minimum_mean_improvement_vs_uniform_fraction",
            "minimum_mean_improvement_vs_permuted_fraction",
        ):
            value = float(getattr(self, name))
            _require(0.0 <= value < 1.0, f"invalid {name}")
        _require(
            0.0 <= self.minimum_primary_win_fraction <= 1.0,
            "minimum_primary_win_fraction must lie in [0,1]",
        )
        _require(
            np.isfinite(self.maximum_primary_pair_ratio)
            and self.maximum_primary_pair_ratio >= 1.0,
            "maximum_primary_pair_ratio must be finite and at least one",
        )


def _prediction_metrics(
    reference: np.ndarray, prediction: np.ndarray
) -> dict[str, float]:
    _require(reference.shape == prediction.shape, "reference/prediction shape mismatch")
    return {
        "chamfer_distance_m": _mean_chamfer_m(reference, prediction),
        "track_error_m": float(np.mean(np.linalg.norm(reference - prediction, axis=2))),
    }


def _candidate_bank(
    observation: RopeDynamicsObservation,
    *,
    forward_config: RopeForwardFitConfig,
) -> dict[str, Any]:
    case = _source_forecast_case(observation, forward_config)
    candidates = _candidate_parameters(forward_config)
    rollouts = []
    scores = []
    reference = np.asarray(case["reference_positions_m"], dtype=np.float64)[1:]
    for parameters in candidates:
        rollout = rollout_rope_dynamics(
            case["initial_positions_m"],
            case["initial_velocities_m_s"],
            case["controller_positions_m"],
            case["contact_active"],
            case["contact_node_indices"],
            case["contact_offsets_m"],
            case["rest_lengths_m"],
            parameters,
            dt_seconds=float(case["dt_seconds"]),
            gravity_m_s2=np.zeros(3),
            substeps=forward_config.substeps,
            constraint_iterations=forward_config.constraint_iterations,
        )[1:]
        rollouts.append(rollout)
        scores.append(_mean_chamfer_m(reference, rollout))
    persistence = np.repeat(
        np.asarray(case["initial_positions_m"], dtype=np.float64)[None],
        len(reference),
        axis=0,
    )
    return {
        "episode_id": observation.episode_id,
        "reference": reference,
        "candidate_rollouts": np.asarray(rollouts, dtype=np.float64),
        "candidate_chamfer_m": np.asarray(scores, dtype=np.float64),
        "persistence": persistence,
    }


def _factual_posterior(
    candidate_chamfer_m: np.ndarray,
    *,
    config: LoggedCounterfactualConfig,
) -> tuple[np.ndarray, dict[str, float]]:
    scores = np.asarray(candidate_chamfer_m, dtype=np.float64)
    _require(
        scores.ndim == 1 and len(scores) >= 2,
        "candidate scores must be one-dimensional",
    )
    _require(
        np.all(np.isfinite(scores)) and np.all(scores >= 0.0),
        "invalid candidate scores",
    )
    median = float(np.median(scores))
    mad = float(np.median(np.abs(scores - median)))
    temperature = max(config.minimum_temperature_m, mad)
    log_weight = (
        -config.posterior_power * (scores - float(np.min(scores))) / temperature
    )
    log_weight -= float(np.max(log_weight))
    weights = np.exp(log_weight)
    weights /= np.sum(weights)
    entropy = float(-np.sum(weights * np.log(np.maximum(weights, 1.0e-300))))
    return weights, {
        "temperature_m": temperature,
        "score_median_m": median,
        "score_mad_m": mad,
        "posterior_entropy_nats": entropy,
        "effective_candidate_count": float(np.exp(entropy)),
        "map_candidate_index": int(np.argmax(weights)),
        "map_probability": float(np.max(weights)),
    }


def _permutation_shift(protocol_id: str, factual_episode_id: str, count: int) -> int:
    _require(count >= 2, "candidate permutation needs at least two candidates")
    digest = hashlib.sha256(
        f"{protocol_id}:candidate-permutation:{factual_episode_id}".encode("utf-8")
    ).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (count - 1)


def _factual_partner_map(
    protocol_id: str, episode_ids: Sequence[str]
) -> dict[str, str]:
    ids = tuple(sorted(map(str, episode_ids)))
    _require(len(ids) >= 3, "primary pairing needs at least three source episodes")
    partners: dict[str, str] = {}
    for challenge in ids:
        candidates = [factual for factual in ids if factual != challenge]
        partners[challenge] = min(
            candidates,
            key=lambda factual: hashlib.sha256(
                f"{protocol_id}:primary-pair:{challenge}:{factual}".encode("utf-8")
            ).hexdigest(),
        )
    return partners


def _mixture_prediction(weights: np.ndarray, rollouts: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(weights, dtype=np.float64)
    trajectories = np.asarray(rollouts, dtype=np.float64)
    _require(
        trajectories.ndim == 4 and trajectories.shape[0] == len(probabilities),
        "candidate rollout bank shape mismatch",
    )
    _require(
        np.all(probabilities >= 0.0) and np.isclose(np.sum(probabilities), 1.0),
        "mixture weights must be nonnegative and sum to one",
    )
    return np.tensordot(probabilities, trajectories, axes=(0, 0))


def _pair_result(
    protocol_id: str,
    factual: Mapping[str, Any],
    challenge: Mapping[str, Any],
    *,
    config: LoggedCounterfactualConfig,
) -> dict[str, Any]:
    factual_weights, posterior = _factual_posterior(
        np.asarray(factual["candidate_chamfer_m"]), config=config
    )
    count = len(factual_weights)
    uniform_weights = np.full(count, 1.0 / count, dtype=np.float64)
    shift = _permutation_shift(protocol_id, str(factual["episode_id"]), count)
    permuted_weights = np.roll(factual_weights, shift)
    rollouts = np.asarray(challenge["candidate_rollouts"], dtype=np.float64)
    reference = np.asarray(challenge["reference"], dtype=np.float64)
    predictions = {
        "factual_abduction": _mixture_prediction(factual_weights, rollouts),
        "uniform_physics": _mixture_prediction(uniform_weights, rollouts),
        "candidate_id_permuted": _mixture_prediction(permuted_weights, rollouts),
        "persistence": np.asarray(challenge["persistence"], dtype=np.float64),
    }
    metrics = {
        name: _prediction_metrics(reference, prediction)
        for name, prediction in predictions.items()
    }
    return {
        "factual_episode_id": str(factual["episode_id"]),
        "challenge_episode_id": str(challenge["episode_id"]),
        "same_episode": bool(factual["episode_id"] == challenge["episode_id"]),
        "posterior": posterior,
        "permutation_shift": shift,
        "permutation_preserves_weight_multiset": bool(
            np.array_equal(np.sort(factual_weights), np.sort(permuted_weights))
        ),
        "metrics": metrics,
    }


def _aggregate_pairs(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    _require(bool(rows), "cannot aggregate an empty pair panel")
    methods = (
        "factual_abduction",
        "uniform_physics",
        "candidate_id_permuted",
        "persistence",
    )
    mean_cd = {
        method: float(
            np.mean([row["metrics"][method]["chamfer_distance_m"] for row in rows])
        )
        for method in methods
    }
    mean_track = {
        method: float(
            np.mean([row["metrics"][method]["track_error_m"] for row in rows])
        )
        for method in methods
    }
    abduced = np.asarray(
        [row["metrics"]["factual_abduction"]["chamfer_distance_m"] for row in rows]
    )
    comparisons: dict[str, Any] = {}
    for control in ("uniform_physics", "candidate_id_permuted", "persistence"):
        baseline = np.asarray(
            [row["metrics"][control]["chamfer_distance_m"] for row in rows]
        )
        comparisons[control] = {
            "mean_relative_improvement_fraction": float(
                (np.mean(baseline) - np.mean(abduced)) / np.mean(baseline)
            ),
            "win_fraction": float(np.mean(abduced < baseline)),
            "wins": int(np.count_nonzero(abduced < baseline)),
            "ties": int(np.count_nonzero(abduced == baseline)),
            "losses": int(np.count_nonzero(abduced > baseline)),
            "worst_abduced_to_control_ratio": float(np.max(abduced / baseline)),
        }
    return {
        "pair_count": len(rows),
        "mean_chamfer_distance_m": mean_cd,
        "mean_track_error_m": mean_track,
        "comparisons": comparisons,
    }


def evaluate_logged_cross_intervention_abduction(
    observations: Sequence[RopeDynamicsObservation],
    *,
    protocol_id: str,
    config: LoggedCounterfactualConfig = LoggedCounterfactualConfig(),
    forward_config: RopeForwardFitConfig | None = None,
) -> dict[str, Any]:
    """Evaluate factual-A -> intervention-B transfer on source-only real episodes."""

    _require(bool(protocol_id), "protocol_id must be nonempty")
    _require(len(observations) >= 4, "logged counterfactual panel needs four episodes")
    episode_ids = [observation.episode_id for observation in observations]
    _require(len(episode_ids) == len(set(episode_ids)), "source episode ids repeat")
    forward = forward_config or RopeForwardFitConfig(
        prefix_frame_count=config.prefix_frame_count
    )
    _require(
        forward.prefix_frame_count == config.prefix_frame_count,
        "forward and counterfactual prefix lengths differ",
    )
    banks = {
        observation.episode_id: _candidate_bank(observation, forward_config=forward)
        for observation in observations
    }
    ordered_pairs = []
    for factual_id in sorted(banks):
        for challenge_id in sorted(banks):
            if factual_id == challenge_id:
                continue
            ordered_pairs.append(
                _pair_result(
                    protocol_id,
                    banks[factual_id],
                    banks[challenge_id],
                    config=config,
                )
            )
    partner_map = _factual_partner_map(protocol_id, episode_ids)
    primary_pairs = [
        _pair_result(
            protocol_id,
            banks[partner_map[challenge_id]],
            banks[challenge_id],
            config=config,
        )
        for challenge_id in sorted(banks)
    ]
    primary = _aggregate_pairs(primary_pairs)
    all_pairs = _aggregate_pairs(ordered_pairs)
    uniform = primary["comparisons"]["uniform_physics"]
    permuted = primary["comparisons"]["candidate_id_permuted"]
    gate = {
        "minimum_mean_improvement_vs_uniform_fraction": (
            config.minimum_mean_improvement_vs_uniform_fraction
        ),
        "observed_mean_improvement_vs_uniform_fraction": uniform[
            "mean_relative_improvement_fraction"
        ],
        "minimum_mean_improvement_vs_permuted_fraction": (
            config.minimum_mean_improvement_vs_permuted_fraction
        ),
        "observed_mean_improvement_vs_permuted_fraction": permuted[
            "mean_relative_improvement_fraction"
        ],
        "minimum_primary_win_fraction": config.minimum_primary_win_fraction,
        "observed_win_fraction_vs_uniform": uniform["win_fraction"],
        "observed_win_fraction_vs_permuted": permuted["win_fraction"],
        "maximum_primary_pair_ratio": config.maximum_primary_pair_ratio,
        "observed_worst_ratio_vs_uniform": uniform["worst_abduced_to_control_ratio"],
        "observed_worst_ratio_vs_permuted": permuted["worst_abduced_to_control_ratio"],
    }
    gate["passed"] = bool(
        gate["observed_mean_improvement_vs_uniform_fraction"]
        >= gate["minimum_mean_improvement_vs_uniform_fraction"]
        and gate["observed_mean_improvement_vs_permuted_fraction"]
        >= gate["minimum_mean_improvement_vs_permuted_fraction"]
        and gate["observed_win_fraction_vs_uniform"]
        >= gate["minimum_primary_win_fraction"]
        and gate["observed_win_fraction_vs_permuted"]
        >= gate["minimum_primary_win_fraction"]
        and gate["observed_worst_ratio_vs_uniform"]
        <= gate["maximum_primary_pair_ratio"]
        and gate["observed_worst_ratio_vs_permuted"]
        <= gate["maximum_primary_pair_ratio"]
    )
    return {
        "config": asdict(config),
        "forward_config": asdict(forward),
        "episode_ids": sorted(episode_ids),
        "candidate_count": len(_candidate_parameters(forward)),
        "persistent_latent": "shared reduced-rope dynamics candidate index",
        "event_specific_latent_policy": (
            "challenge contact schedule/node/offset comes from challenge episode; "
            "factual contact is never transferred"
        ),
        "factual_abduction": (
            "generalized Bayes over the finite dynamics bank using only the full "
            "source factual interaction"
        ),
        "primary_partner_map": partner_map,
        "primary_pairs": primary_pairs,
        "primary_summary": primary,
        "all_ordered_pairs": ordered_pairs,
        "all_pairs_summary": all_pairs,
        "source_gate": gate,
    }


def build_logged_counterfactual_source_artifact(
    observation_payloads: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str,
    config: LoggedCounterfactualConfig = LoggedCounterfactualConfig(),
) -> dict[str, Any]:
    """Bind the real-data source-only test to immutable observation artifacts."""

    accepted = []
    inputs = []
    for payload in observation_payloads:
        _require(
            payload.get("schema_version") == DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
            "source observation schema mismatch",
        )
        _require(
            payload.get("result_sha256")
            == rope_source_observation_artifact_sha256(payload),
            "source observation checksum mismatch",
        )
        _require(payload.get("split") == "source", "counterfactual input is not source")
        boundary = payload.get("information_boundary", {})
        _require(
            boundary.get("target_files_read") is False,
            "source counterfactual input read target files",
        )
        inputs.append(
            {
                "episode_index": int(payload["episode_index"]),
                "episode_id": str(payload["episode_id"]),
                "result_sha256": str(payload["result_sha256"]),
                "quality_passed": bool(payload["quality"]["passed"]),
            }
        )
        if payload["quality"]["passed"]:
            accepted.append(load_source_rope_dynamics_observation(payload))
    _require(len(accepted) >= 4, "too few quality-passing source observations")
    result = evaluate_logged_cross_intervention_abduction(
        accepted,
        protocol_id=protocol_id,
        config=config,
    )
    artifact: dict[str, Any] = {
        "schema_version": DEFORM360_LOGGED_COUNTERFACTUAL_SCHEMA_VERSION,
        "artifact_kind": "Deform360LoggedCrossInterventionAbductionSourceV1",
        "protocol_id": protocol_id,
        "source_inputs": sorted(inputs, key=lambda row: row["episode_index"]),
        **result,
        "information_boundary": {
            "public_real_data": True,
            "source_episode_futures_read": True,
            "target_prefix_read": False,
            "target_future_read": False,
            "target_tactile_read": False,
            "target_scores_read": False,
            "candidate_permutation_uses_outcomes": False,
        },
        "claim_boundary": (
            "This is a source-only logged cross-intervention transfer test on "
            "repeated real interactions of one physical object. It tests whether "
            "persistent physics abducted from factual interaction A predicts a "
            "different recorded intervention B. It is not individual-level real "
            "counterfactual ground truth because the two executions are separate "
            "physical trials."
        ),
        "target_promotion_authorized": False,
    }
    artifact["result_sha256"] = logged_counterfactual_artifact_sha256(artifact)
    return artifact
