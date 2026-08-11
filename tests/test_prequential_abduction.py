import numpy as np

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
)
from causal4d.observation_evidence import (
    GroupedObservationEvidence,
    ObservationGroup,
)
from causal4d.prequential_abduction import (
    build_prequential_abduction_path,
    grouped_observation_prefix,
)
from causal4d.rollout_bank import JointRolloutBank


def _problem():
    bank_frames = 8
    trajectories = np.zeros((3, 1, bank_frames, 1, 3), dtype=float)
    time = np.arange(bank_frames, dtype=float)
    trajectories[1, 0, :, 0, 0] = 0.01 * time
    trajectories[2, 0, :, 0, 0] = -0.01 * time

    def metadata(identifier: str, gain: float, shift: int) -> dict[str, object]:
        return {
            "hypothesis_id": identifier,
            "action": {
                "proposal_id": "known",
                "future_action_observed": True,
                "provenance": "unit factual action",
            },
            "contact": {
                "attachment_shifts": [shift],
                "gain_multiplier": gain,
                "delay_steps": 0,
                "slip_fraction": 0.0,
                "rotation_degrees": 0.0,
            },
        }

    bank = JointRolloutBank(
        hypothesis_ids=("nominal", "high_gain", "shifted"),
        hypothesis_metadata=(
            metadata("nominal", 1.0, 0),
            metadata("high_gain", 1.15, 0),
            metadata("shifted", 1.0, 1),
        ),
        hypothesis_prior_weights=np.asarray([0.6, 0.2, 0.2]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
    )
    full_frames = 11
    intervention_frame = 4
    full_observations = np.zeros((full_frames, 1, 3), dtype=float)
    full_observations[intervention_frame - 1 :] = trajectories[1, 0]
    actions = np.zeros((full_frames, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="prequential_abduction_unit",
        case_id="synthetic",
        observations=full_observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
    )
    belief = TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=("p0",),
        theta_names=("spring",),
        endpoint_position_m=np.zeros((1, 1, 3)),
        endpoint_velocity_mps=np.zeros((1, 1, 3)),
        theta=np.asarray([[0.0]]),
        discrepancy_mean_m=np.zeros((1, 1, 3)),
        discrepancy_variance_m2=np.zeros((1, 1, 3)),
        weights=np.asarray([1.0]),
    )
    observations = trajectories[1, 0].copy()
    mask = np.ones((bank_frames, 1), dtype=bool)
    config = FactualAbductionConfig(
        observation_scale_m=0.001,
        likelihood_power=60.0,
        dynamic_likelihood_weight=0.5,
    )
    return bank, belief, observations, mask, config


def test_final_path_step_is_exact_one_shot_abduction() -> None:
    bank, belief, observations, mask, config = _problem()
    result = build_prequential_abduction_path(
        bank,
        belief,
        observations,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
    )
    standalone = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )

    assert result.factual_interventions[-1].artifact_id == standalone.artifact_id
    assert result.path.factual_intervention_ids[-1] == standalone.artifact_id
    assert np.array_equal(result.path.posterior_weights[-1], standalone.weights)
    assert result.path.metadata["future_frames_read"] == 0
    assert not result.path.posterior_weights.flags.writeable


def test_future_changes_do_not_change_prequential_path() -> None:
    bank, belief, observations, mask, config = _problem()
    first = build_prequential_abduction_path(
        bank,
        belief,
        observations,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
    )
    changed = observations.copy()
    changed[4:] += 1000.0
    changed_mask = mask.copy()
    changed_mask[4:] = False
    second = build_prequential_abduction_path(
        bank,
        belief,
        changed,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=changed_mask,
        config=config,
    )

    assert first.path.artifact_id == second.path.artifact_id
    assert first.path.factual_intervention_ids == second.path.factual_intervention_ids
    assert np.array_equal(first.path.posterior_weights, second.path.posterior_weights)


def test_later_prefix_change_cannot_rewrite_earlier_steps() -> None:
    bank, belief, observations, mask, config = _problem()
    first = build_prequential_abduction_path(
        bank,
        belief,
        observations,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
    )
    changed = observations.copy()
    changed[3] += 0.25
    second = build_prequential_abduction_path(
        bank,
        belief,
        changed,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
    )

    assert (
        first.path.factual_intervention_ids[:2]
        == (second.path.factual_intervention_ids[:2])
    )
    assert (
        first.path.factual_intervention_ids[2]
        != (second.path.factual_intervention_ids[2])
    )


def test_grouped_prefix_uses_matching_covariance_principal_submatrix() -> None:
    group = ObservationGroup(
        group_id="cross-frame",
        values_m=np.asarray([0.1, 0.2, 0.3]),
        frame_indices=np.asarray([1, 2, 3]),
        node_indices=np.asarray([0, 0, 0]),
        coordinate_indices=np.asarray([0, 1, 2]),
        covariance_m2=np.asarray(
            [
                [4.0, 1.0, 0.5],
                [1.0, 3.0, 0.25],
                [0.5, 0.25, 2.0],
            ]
        ),
        contributor_ids=("camera-a",),
        source_id="camera",
    )
    evidence = GroupedObservationEvidence(
        groups=(group,),
        evidence_id="cross-frame-evidence",
    )

    prefix = grouped_observation_prefix(evidence, prefix_frame_count=3)

    assert prefix.groups[0].values_m.tolist() == [0.1, 0.2]
    assert np.array_equal(
        prefix.groups[0].covariance_m2,
        group.covariance_m2[:2, :2],
    )
    assert prefix.metadata["split_group_ids"] == ["cross-frame"]


def test_grouped_path_preserves_final_dense_and_batched_identity() -> None:
    bank, belief, observations, mask, config = _problem()
    evidence = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        mask=mask,
        source_id="prequential-test",
    )
    prefix = grouped_observation_prefix(evidence, prefix_frame_count=3)
    assert all(np.all(group.frame_indices < 3) for group in prefix.groups)
    assert prefix.metadata["future_frames_read"] == 0
    assert grouped_observation_prefix(evidence, prefix_frame_count=4) is evidence

    dense = build_prequential_abduction_path(
        bank,
        belief,
        observations,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
    )
    batched = build_prequential_abduction_path(
        bank,
        belief,
        observations,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        grouped_component_batch_size=1,
    )
    standalone = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        grouped_component_batch_size=1,
    )

    assert dense.path.artifact_id == batched.path.artifact_id
    assert dense.path.factual_intervention_ids[-1] == standalone.artifact_id
    assert np.array_equal(dense.path.posterior_weights, batched.path.posterior_weights)


def test_path_reports_entropy_ess_kl_and_total_variation() -> None:
    bank, belief, observations, mask, config = _problem()
    path = build_prequential_abduction_path(
        bank,
        belief,
        observations,
        prefix_frame_counts=(2, 3, 4),
        observation_mask=mask,
        config=config,
    ).path

    assert np.all(path.posterior_entropy >= 0.0)
    assert np.all(path.posterior_effective_sample_size >= 1.0)
    assert np.all(path.posterior_effective_sample_size <= len(path.component_ids))
    assert path.previous_step_kl[0] == 0.0
    assert path.previous_step_total_variation[0] == 0.0
    assert np.all(path.previous_step_kl[1:] >= 0.0)
    assert np.all(
        (path.previous_step_total_variation >= 0.0)
        & (path.previous_step_total_variation <= 1.0)
    )
    assert np.array_equal(
        path.map_component_indices,
        np.argmax(path.posterior_weights, axis=1),
    )
