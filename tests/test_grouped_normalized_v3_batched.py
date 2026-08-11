from __future__ import annotations

import numpy as np

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
)
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.rollout_bank import JointRolloutBank


def _metadata(identifier: str, *, gain: float, shift: int) -> dict:
    return {
        "hypothesis_id": identifier,
        "action": {
            "proposal_id": "known",
            "future_action_observed": True,
            "provenance": "normalized-v3 batching unit test",
        },
        "contact": {
            "attachment_shifts": [shift],
            "gain_multiplier": gain,
            "delay_steps": 0,
            "slip_fraction": 0.0,
            "rotation_degrees": 0.0,
        },
    }


def _problem() -> tuple[
    JointRolloutBank,
    TwinBelief,
    np.ndarray,
    np.ndarray,
    GroupedObservationEvidence,
]:
    frame_count = 7
    trajectories = np.zeros((3, 2, frame_count, 1, 3), dtype=float)
    time = np.arange(frame_count, dtype=float)
    trajectories[1, :, :, 0, 0] = 0.01 * time
    trajectories[2, :, :, 0, 0] = -0.01 * time
    trajectories[:, 1, :, 0, 1] = 0.002 * time
    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "high_gain", "shifted"),
        hypothesis_metadata=(
            _metadata("nominal", gain=1.0, shift=0),
            _metadata("high_gain", gain=1.15, shift=0),
            _metadata("shifted", gain=1.0, shift=1),
        ),
        hypothesis_prior_weights=np.asarray([0.6, 0.2, 0.2]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.4, 0.6]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
    )
    full_frames = frame_count + 3
    intervention_frame = 4
    context = build_causal_context(
        protocol_id="normalized-v3-batching-unit",
        case_id="synthetic",
        observations=np.zeros((full_frames, 1, 3)),
        observed_actions=np.zeros((full_frames, 1, 3)),
        counterfactual_actions=np.zeros((full_frames, 1, 3)),
        intervention_frame=intervention_frame,
    )
    state_shape = (2, 1, 3)
    discrepancy = np.zeros(state_shape)
    discrepancy[1, 0, 1] = 0.001
    belief = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0", "p1"),
        theta_names=("spring",),
        endpoint_position_m=np.zeros(state_shape),
        endpoint_velocity_mps=np.zeros(state_shape),
        theta=bank.parameter_particles,
        discrepancy_mean_m=discrepancy,
        discrepancy_variance_m2=np.full(state_shape, 1e-8),
        weights=bank.parameter_weights,
    )
    observations = trajectories[1, 0].copy()
    mask = np.ones((frame_count, 1), dtype=bool)
    evidence = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        mask=mask,
        source_id="unit",
    )
    return bank, belief, observations, mask, evidence


def test_grouped_component_batching_preserves_normalized_v3_semantics() -> None:
    bank, belief, observations, mask, evidence = _problem()
    config = FactualAbductionConfig(
        observation_scale_m=0.001,
        grouped_likelihood_semantics="normalized_v3",
    )
    dense = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
    )
    batched = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        grouped_component_batch_size=1,
    )

    assert np.array_equal(batched.weights, dense.weights)
    assert batched.metadata == dense.metadata
    assert batched.artifact_id == dense.artifact_id
    diagnostics = batched.metadata["grouped_observation_evidence"]["diagnostics"]
    assert diagnostics["score_semantics"] == "normalized_coordinate_mean_v3"
    assert diagnostics["likelihood_power"] == config.likelihood_power
