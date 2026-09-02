"""Causally strict logged cross-intervention evaluation for Deform360 rope.

Factual interaction A may be consumed completely for abduction. Challenge
interaction B contributes only its observed prefix geometry/contact state and
its released future controller trajectory to prediction. B future geometry is
reserved for scoring, and future tactile/contact annotations are never consumed
by the predictive branch.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import numpy as np

from .deform360_logged_counterfactual import (
    DEFORM360_LOGGED_COUNTERFACTUAL_SCHEMA_VERSION,
    LoggedCounterfactualConfig,
    _aggregate_pairs,
    _candidate_bank,
    _factual_partner_map,
    _pair_result,
    logged_counterfactual_artifact_sha256,
)
from .deform360_rope_dynamics import RopeDynamicsObservation, rollout_rope_dynamics
from .deform360_rope_fit import RopeForwardFitConfig, _candidate_parameters
from .deform360_rope_observations import (
    DEFORM360_ROPE_OBSERVATION_SCHEMA_VERSION,
    load_source_rope_dynamics_observation,
    rope_source_observation_artifact_sha256,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _prefix_only_challenge_case(
    observation: RopeDynamicsObservation,
    *,
    prefix_frame_count: int,
) -> dict[str, Any]:
    """Construct B intervention state without B future contact annotations."""

    positions = np.asarray(observation.positions_m, dtype=np.float64)
    controllers = np.asarray(observation.controller_positions_m, dtype=np.float64)
    active = np.asarray(observation.contact_active, dtype=bool)
    contact_frames = np.flatnonzero(np.any(active, axis=1))
    _require(len(contact_frames) > 0, "challenge contains no observed contact onset")
    prefix_start = int(contact_frames[0])
    prefix_end = prefix_start + prefix_frame_count
    _require(prefix_end < len(positions), "challenge prefix has no held-out future")

    prefix_positions = positions[prefix_start:prefix_end]
    prefix_controllers = controllers[prefix_start:prefix_end]
    prefix_active = active[prefix_start:prefix_end]
    active_at_endpoint = np.asarray(prefix_active[-1], dtype=bool)
    _require(
        np.any(active_at_endpoint),
        "no controller remains in contact at the challenge prefix endpoint",
    )

    contact_nodes: list[int] = []
    contact_offsets = np.zeros((controllers.shape[1], 3), dtype=np.float64)
    diagnostics = []
    for controller in range(controllers.shape[1]):
        observed = np.flatnonzero(prefix_active[:, controller])
        if not active_at_endpoint[controller]:
            contact_nodes.append(0)
            diagnostics.append(
                {
                    "controller_index": controller,
                    "active_at_prefix_endpoint": False,
                    "contact_node_index": 0,
                    "prefix_active_frame_count": int(len(observed)),
                }
            )
            continue
        _require(
            len(observed) >= 1,
            "active endpoint controller has no observed prefix contact frame",
        )
        distances = np.linalg.norm(
            prefix_controllers[observed, controller, None, :]
            - prefix_positions[observed],
            axis=2,
        )
        node = int(np.argmin(np.median(distances, axis=0)))
        offset = np.median(
            prefix_positions[observed, node] - prefix_controllers[observed, controller],
            axis=0,
        )
        contact_nodes.append(node)
        contact_offsets[controller] = offset
        diagnostics.append(
            {
                "controller_index": controller,
                "active_at_prefix_endpoint": True,
                "contact_node_index": node,
                "prefix_active_frame_count": int(len(observed)),
                "contact_offset_m": offset.tolist(),
            }
        )

    future_length = len(positions) - prefix_end + 1
    future_active = np.repeat(active_at_endpoint[None], future_length, axis=0)
    initial = positions[prefix_end - 1]
    reference = positions[prefix_end - 1 :]
    return {
        "episode_id": observation.episode_id,
        "prefix_start_index": prefix_start,
        "prefix_end_index_exclusive": prefix_end,
        "initial_positions_m": initial,
        "initial_velocities_m_s": np.zeros_like(initial),
        "controller_positions_m": controllers[prefix_end - 1 :],
        "contact_active": future_active,
        "contact_node_indices": tuple(contact_nodes),
        "contact_offsets_m": contact_offsets,
        "rest_lengths_m": np.linalg.norm(np.diff(initial, axis=0), axis=1),
        "reference_positions_m": reference,
        "dt_seconds": observation.dt_seconds,
        "contact_realization_policy": (
            "contact onset/activity is observed only through the B prefix; controllers "
            "active at the prefix endpoint are held active for the forecast; contact "
            "node and offset are estimated only from B prefix geometry"
        ),
        "prefix_contact_diagnostics": diagnostics,
        "future_contact_annotations_read_for_prediction": False,
    }


def _strict_challenge_bank(
    observation: RopeDynamicsObservation,
    *,
    forward_config: RopeForwardFitConfig,
) -> dict[str, Any]:
    case = _prefix_only_challenge_case(
        observation,
        prefix_frame_count=forward_config.prefix_frame_count,
    )
    candidates = _candidate_parameters(forward_config)
    rollouts = []
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
    persistence = np.repeat(
        np.asarray(case["initial_positions_m"], dtype=np.float64)[None],
        len(reference),
        axis=0,
    )
    return {
        "episode_id": observation.episode_id,
        "reference": reference,
        "candidate_rollouts": np.asarray(rollouts, dtype=np.float64),
        "persistence": persistence,
        "challenge_contact_realization": {
            "prefix_start_index": case["prefix_start_index"],
            "prefix_end_index_exclusive": case["prefix_end_index_exclusive"],
            "contact_realization_policy": case["contact_realization_policy"],
            "prefix_contact_diagnostics": case["prefix_contact_diagnostics"],
            "future_contact_annotations_read_for_prediction": False,
        },
    }


def evaluate_logged_cross_intervention_abduction_strict(
    observations: Sequence[RopeDynamicsObservation],
    *,
    protocol_id: str,
    config: LoggedCounterfactualConfig = LoggedCounterfactualConfig(),
    forward_config: RopeForwardFitConfig | None = None,
) -> dict[str, Any]:
    """Evaluate full-A abduction followed by prefix-only B intervention prediction."""

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

    factual_banks = {
        observation.episode_id: _candidate_bank(observation, forward_config=forward)
        for observation in observations
    }
    challenge_banks = {
        observation.episode_id: _strict_challenge_bank(
            observation,
            forward_config=forward,
        )
        for observation in observations
    }

    ordered_pairs = []
    for factual_id in sorted(factual_banks):
        for challenge_id in sorted(challenge_banks):
            if factual_id == challenge_id:
                continue
            ordered_pairs.append(
                _pair_result(
                    protocol_id,
                    factual_banks[factual_id],
                    challenge_banks[challenge_id],
                    config=config,
                )
            )
    partner_map = _factual_partner_map(protocol_id, episode_ids)
    primary_pairs = [
        _pair_result(
            protocol_id,
            factual_banks[partner_map[challenge_id]],
            challenge_banks[challenge_id],
            config=config,
        )
        for challenge_id in sorted(challenge_banks)
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
        "factual_abduction": (
            "full factual A source outcome; generalized Bayes over the frozen "
            "finite dynamics bank"
        ),
        "challenge_intervention": (
            "B prefix geometry/contact realization plus released future controller "
            "trajectory; B future geometry is scoring-only and B future contact "
            "annotations are excluded from prediction"
        ),
        "primary_partner_map": partner_map,
        "primary_pairs": primary_pairs,
        "primary_summary": primary,
        "all_ordered_pairs": ordered_pairs,
        "all_pairs_summary": all_pairs,
        "source_gate": gate,
        "challenge_contact_diagnostics": {
            episode_id: challenge_banks[episode_id]["challenge_contact_realization"]
            for episode_id in sorted(challenge_banks)
        },
    }


def build_logged_counterfactual_source_artifact_strict(
    observation_payloads: Sequence[Mapping[str, Any]],
    *,
    protocol_id: str,
    config: LoggedCounterfactualConfig = LoggedCounterfactualConfig(),
) -> dict[str, Any]:
    """Bind the strict real-data source test to immutable source artifacts."""

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
    result = evaluate_logged_cross_intervention_abduction_strict(
        accepted,
        protocol_id=protocol_id,
        config=config,
    )
    artifact: dict[str, Any] = {
        "schema_version": DEFORM360_LOGGED_COUNTERFACTUAL_SCHEMA_VERSION,
        "artifact_kind": "Deform360LoggedCrossInterventionAbductionSourceStrictV1",
        "protocol_id": protocol_id,
        "source_inputs": sorted(inputs, key=lambda row: row["episode_index"]),
        **result,
        "information_boundary": {
            "public_real_data": True,
            "factual_source_episode_futures_read_for_abduction": True,
            "challenge_prefix_geometry_and_contact_read": True,
            "challenge_future_controller_trajectory_read_as_intervention": True,
            "challenge_future_geometry_read_for_prediction": False,
            "challenge_future_geometry_read_for_scoring": True,
            "challenge_future_contact_annotations_read_for_prediction": False,
            "target_prefix_read": False,
            "target_future_read": False,
            "target_tactile_read": False,
            "target_scores_read": False,
            "candidate_permutation_uses_challenge_outcome": False,
        },
        "claim_boundary": (
            "Source-only logged cross-intervention transfer on repeated real "
            "interactions of one physical object. The challenge prediction uses "
            "only B prefix state/contact plus the released future controller "
            "trajectory. Separate A and B executions are not individual-level "
            "counterfactual ground truth."
        ),
        "target_promotion_authorized": False,
    }
    artifact["result_sha256"] = logged_counterfactual_artifact_sha256(artifact)
    return artifact
