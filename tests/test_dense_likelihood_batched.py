from __future__ import annotations

import numpy as np
import pytest

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    evaluate_factual_abduction,
)
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.rollout_bank import JointRolloutBank


def _metadata(identifier: str, *, gain: float, shift: int) -> dict:
    return {
        "hypothesis_id": identifier,
        "action": {
            "proposal_id": "known",
            "future_action_observed": True,
            "provenance": "dense batching unit test",
        },
        "contact": {
            "attachment_shifts": [shift],
            "gain_multiplier": gain,
            "delay_steps": 0,
            "slip_fraction": 0.0,
            "rotation_degrees": 0.0,
        },
    }


def _problem() -> tuple[JointRolloutBank, TwinBelief, np.ndarray, np.ndarray]:
    frame_count = 7
    trajectories = np.zeros((3, 2, frame_count, 2, 3), dtype=float)
    time = np.arange(frame_count, dtype=float)
    trajectories[0, :, :, 0, 0] = 0.003 * time
    trajectories[1, :, :, 0, 0] = 0.010 * time
    trajectories[2, :, :, 0, 0] = -0.008 * time
    trajectories[:, 1, :, 0, 1] = 0.002 * time
    trajectories[1, :, :, 1, 2] = 0.004 * np.square(time)

    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "high_gain", "shifted"),
        hypothesis_metadata=(
            _metadata("nominal", gain=1.0, shift=0),
            _metadata("high_gain", gain=1.15, shift=0),
            _metadata("shifted", gain=1.0, shift=1),
        ),
        hypothesis_prior_weights=np.asarray([0.6, 0.25, 0.15]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.4, 0.6]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
    )

    full_frames = frame_count + 3
    intervention_frame = 4
    context = build_causal_context(
        protocol_id="dense-batching-unit",
        case_id="synthetic",
        observations=np.zeros((full_frames, 2, 3)),
        observed_actions=np.zeros((full_frames, 1, 3)),
        counterfactual_actions=np.zeros((full_frames, 1, 3)),
        intervention_frame=intervention_frame,
    )
    state_shape = (2, 2, 3)
    discrepancy = np.zeros(state_shape)
    discrepancy[0, 1, 0] = 0.0007
    discrepancy[1, 0, 1] = 0.0011
    discrepancy_variance = np.full(state_shape, 1e-8)
    discrepancy_variance[1, 1, 2] = 4e-8
    belief = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0", "p1"),
        theta_names=("spring",),
        endpoint_position_m=np.zeros(state_shape),
        endpoint_velocity_mps=np.zeros(state_shape),
        theta=bank.parameter_particles,
        discrepancy_mean_m=discrepancy,
        discrepancy_variance_m2=discrepancy_variance,
        weights=bank.parameter_weights,
    )

    observations = trajectories[1, 0].astype(float)
    observations[:, 1, 1] = 0.001 * time
    observations[2, 1, 2] = np.nan
    mask = np.ones((frame_count, 2), dtype=bool)
    mask[1, 0] = False
    return bank, belief, observations, mask


@pytest.mark.parametrize("likelihood_semantics", ["legacy_v1", "normalized_v2"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 20])
def test_dense_component_batching_preserves_posterior_and_artifact_identity(
    likelihood_semantics: str,
    batch_size: int,
) -> None:
    bank, belief, observations, mask = _problem()
    config = FactualAbductionConfig(
        observation_scale_m=0.003,
        likelihood_power=7.0,
        dynamic_likelihood_weight=0.4,
        degrees_of_freedom=3.5,
        likelihood_semantics=likelihood_semantics,
        difference_correlation=(
            0.2 if likelihood_semantics == "normalized_v2" else 0.0
        ),
    )
    dense = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )
    batched = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        dense_component_batch_size=batch_size,
    )

    assert np.array_equal(batched.weights, dense.weights)
    assert batched.metadata == dense.metadata
    assert batched.artifact_id == dense.artifact_id


@pytest.mark.parametrize("likelihood_semantics", ["legacy_v1", "normalized_v2"])
def test_dense_component_batching_preserves_nominal_comparator(
    likelihood_semantics: str,
) -> None:
    bank, belief, observations, mask = _problem()
    config = FactualAbductionConfig(
        observation_scale_m=0.003,
        likelihood_power=7.0,
        dynamic_likelihood_weight=0.4,
        likelihood_semantics=likelihood_semantics,
        difference_correlation=(
            0.2 if likelihood_semantics == "normalized_v2" else 0.0
        ),
    )
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )
    dense = evaluate_factual_abduction(
        bank,
        belief,
        factual,
        observations,
        observation_mask=mask,
        prefix_frame_count=4,
        config=config,
    )
    batched = evaluate_factual_abduction(
        bank,
        belief,
        factual,
        observations,
        observation_mask=mask,
        prefix_frame_count=4,
        config=config,
        dense_component_batch_size=2,
    )

    assert batched == dense


@pytest.mark.parametrize("invalid_batch_size", [0, -1, True, 1.5])
def test_dense_component_batch_size_must_be_a_positive_integer(
    invalid_batch_size: object,
) -> None:
    bank, belief, observations, mask = _problem()

    with pytest.raises(
        ValueError,
        match="component_batch_size must be a positive integer",
    ):
        abduct_factual_intervention(
            bank,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            dense_component_batch_size=invalid_batch_size,  # type: ignore[arg-type]
        )


def test_dense_component_batching_rejects_grouped_evidence() -> None:
    bank, belief, observations, mask = _problem()
    evidence = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.003,
        mask=mask,
        source_id="unit",
    )

    with pytest.raises(
        ValueError,
        match="dense_component_batch_size cannot be combined",
    ):
        abduct_factual_intervention(
            bank,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            grouped_evidence=evidence,
            dense_component_batch_size=2,
        )
