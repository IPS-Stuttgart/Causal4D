from __future__ import annotations

import numpy as np

from causal4d_public.deform360_logged_counterfactual import (
    LoggedCounterfactualConfig,
    _factual_partner_map,
    _factual_posterior,
    _permutation_shift,
    evaluate_logged_cross_intervention_abduction,
)
from causal4d_public.deform360_rope_dynamics import (
    RopeDynamicsObservation,
    SharedRopeDynamicsParameters,
    rollout_rope_dynamics,
)


def _synthetic_observation(
    episode: int, direction: np.ndarray
) -> RopeDynamicsObservation:
    frame_count = 14
    node_count = 5
    dt = 1.0 / 30.0
    initial = np.zeros((node_count, 3), dtype=np.float64)
    initial[:, 0] = np.linspace(0.0, 0.20, node_count)
    controllers = np.repeat(initial[-1][None, None, :], frame_count, axis=0)
    controllers = np.repeat(controllers, 1, axis=1)
    displacement = np.linspace(0.0, 0.035, frame_count)[:, None] * direction[None]
    controllers[:, 0] += displacement
    active = np.zeros((frame_count, 1), dtype=bool)
    active[1:, 0] = True
    positions = np.repeat(initial[None], frame_count, axis=0)
    prefix_end = 1 + 6
    parameters = SharedRopeDynamicsParameters(
        spring_acceleration_per_m_s2=0.0,
        edge_damping_per_s=0.0,
        bending_acceleration_per_m_s2=8.0,
        bending_damping_per_s=0.0,
        contact_acceleration_per_m_s2=30.0,
        contact_damping_per_s=5.0,
        drag_per_s=0.2,
    )
    rest = np.linalg.norm(np.diff(initial, axis=0), axis=1)
    future = rollout_rope_dynamics(
        initial,
        np.zeros_like(initial),
        controllers[prefix_end - 1 :],
        active[prefix_end - 1 :],
        (node_count - 1,),
        np.zeros((1, 3)),
        rest,
        parameters,
        dt_seconds=dt,
        gravity_m_s2=np.zeros(3),
        substeps=4,
        constraint_iterations=16,
    )
    positions[prefix_end - 1 :] = future
    return RopeDynamicsObservation(
        episode_id=f"synthetic/episode_{episode:04d}",
        positions_m=positions,
        controller_positions_m=controllers,
        contact_active=active,
        contact_node_indices=(node_count - 1,),
        contact_offsets_m=np.zeros((1, 3)),
        dt_seconds=dt,
    )


def test_factual_posterior_is_normalized_and_factual_only() -> None:
    scores = np.asarray([0.010, 0.020, 0.030, 0.040])
    weights, diagnostics = _factual_posterior(
        scores,
        config=LoggedCounterfactualConfig(minimum_temperature_m=1.0e-4),
    )
    assert np.isclose(weights.sum(), 1.0)
    assert np.all(weights >= 0.0)
    assert int(np.argmax(weights)) == 0
    assert diagnostics["map_candidate_index"] == 0
    assert diagnostics["effective_candidate_count"] > 1.0


def test_candidate_permutation_has_no_identity_shift() -> None:
    shift = _permutation_shift("protocol", "episode-a", 200)
    assert 1 <= shift < 200
    weights = np.arange(200)
    assert not np.array_equal(weights, np.roll(weights, shift))
    assert np.array_equal(np.sort(weights), np.sort(np.roll(weights, shift)))


def test_primary_pairing_never_uses_the_challenge_itself() -> None:
    episodes = [f"episode-{index}" for index in range(5)]
    mapping = _factual_partner_map("protocol", episodes)
    assert set(mapping) == set(episodes)
    assert all(challenge != factual for challenge, factual in mapping.items())


def test_source_evaluator_uses_distinct_logged_interventions() -> None:
    observations = [
        _synthetic_observation(0, np.asarray([0.0, 1.0, 0.0])),
        _synthetic_observation(1, np.asarray([0.0, 0.0, 1.0])),
        _synthetic_observation(2, np.asarray([0.0, -1.0, 0.0])),
        _synthetic_observation(3, np.asarray([0.0, 0.6, 0.8])),
    ]
    result = evaluate_logged_cross_intervention_abduction(
        observations,
        protocol_id="synthetic-logged-counterfactual-v1",
        config=LoggedCounterfactualConfig(
            minimum_mean_improvement_vs_uniform_fraction=0.0,
            minimum_mean_improvement_vs_permuted_fraction=0.0,
            minimum_primary_win_fraction=0.0,
            maximum_primary_pair_ratio=10.0,
        ),
    )
    assert result["candidate_count"] == 200
    assert len(result["primary_pairs"]) == 4
    assert len(result["all_ordered_pairs"]) == 12
    assert all(not row["same_episode"] for row in result["all_ordered_pairs"])
    assert all(
        row["permutation_preserves_weight_multiset"]
        for row in result["all_ordered_pairs"]
    )
    assert result["event_specific_latent_policy"].startswith("challenge contact")
