from __future__ import annotations

import numpy as np
import pytest

from causal4d.bound_factual_abduction import (
    abduct_factual_intervention_bound,
    validate_factual_rollout_action_binding,
)
from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.intervention_abduction import abduct_factual_intervention
from causal4d.rollout_bank import JointRolloutBank


def _problem(*, proposal_id: str = "observed-pull", factual: bool = True):
    frame_count = 6
    trajectories = np.zeros((2, 1, frame_count, 1, 3), dtype=float)
    trajectories[1, 0, :, 0, 0] = np.arange(frame_count) * 0.01

    def metadata(identifier: str, gain: float) -> dict[str, object]:
        return {
            "hypothesis_id": identifier,
            "action": {
                "proposal_id": proposal_id,
                "future_action_observed": factual,
                "provenance": "unit factual action",
            },
            "contact": {
                "attachment_shifts": [0],
                "gain_multiplier": gain,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }

    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "high-gain"),
        hypothesis_metadata=(
            metadata("nominal", 1.0),
            metadata("high-gain", 1.2),
        ),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1.0e-8,
    )
    intervention_frame = 2
    complete_observations = np.zeros((7, 1, 3), dtype=float)
    actions = np.zeros((7, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="bound-abduction-v1",
        case_id="synthetic",
        observations=complete_observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
        observed_action_id="observed-pull",
    )
    belief = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0",),
        theta_names=("stiffness",),
        endpoint_position_m=np.zeros((1, 1, 3)),
        endpoint_velocity_mps=np.zeros((1, 1, 3)),
        theta=np.asarray([[1.0]]),
        discrepancy_mean_m=np.zeros((1, 1, 3)),
        discrepancy_variance_m2=np.zeros((1, 1, 3)),
        weights=np.asarray([1.0]),
    )
    observations = trajectories[1, 0].copy()
    return bank, belief, observations


def test_bound_abduction_preserves_posterior_and_adds_exact_binding() -> None:
    bank, belief, observations = _problem()
    ordinary = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=3,
    )
    bound = abduct_factual_intervention_bound(
        bank,
        belief,
        observations,
        prefix_frame_count=3,
    )

    assert np.array_equal(bound.weights, ordinary.weights)
    assert np.array_equal(bound.phi, ordinary.phi)
    assert np.array_equal(bound.kappa_obs, ordinary.kappa_obs)
    assert bound.artifact_id != ordinary.artifact_id
    binding = bound.metadata["factual_rollout_action_binding"]
    assert binding["rollout_bank_id"] == bank.artifact_id
    assert binding["source_twin_belief_id"] == belief.artifact_id
    assert binding["observed_action_id"] == belief.context.u_obs.action_id
    assert binding["hypothesis_action_ids"] == ["observed-pull", "observed-pull"]


def test_action_binding_rejects_relabelled_factual_bank() -> None:
    bank, belief, observations = _problem(proposal_id="different-action")
    with pytest.raises(ValueError, match="action identity"):
        abduct_factual_intervention_bound(
            bank,
            belief,
            observations,
            prefix_frame_count=3,
        )


def test_action_binding_rejects_counterfactual_bank() -> None:
    bank, belief, _ = _problem(factual=False)
    with pytest.raises(ValueError, match="factual observed action"):
        validate_factual_rollout_action_binding(bank, belief)


def test_bound_abduction_remains_future_suffix_invariant() -> None:
    bank, belief, observations = _problem()
    first = abduct_factual_intervention_bound(
        bank,
        belief,
        observations,
        prefix_frame_count=3,
    )
    changed = observations.copy()
    changed[3:] += 100.0
    second = abduct_factual_intervention_bound(
        bank,
        belief,
        changed,
        prefix_frame_count=3,
    )
    assert first.artifact_id == second.artifact_id
