import numpy as np
import pytest

from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.factual_abduction_uncertainty import (
    FactualAbductionUncertaintyV1,
    load_factual_abduction_uncertainty_npz,
    save_factual_abduction_uncertainty_npz,
)
from causal4d.identifiability import (
    IdentifiabilityConfig,
    assess_intervention_identifiability,
)
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    evaluate_factual_abduction,
    factual_joint_weights,
)
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.rollout_bank import JointRolloutBank


def _problem():
    bank_frames = 8
    trajectories = np.zeros((3, 1, bank_frames, 1, 3), dtype=float)
    time = np.arange(bank_frames, dtype=float)
    trajectories[1, 0, :, 0, 0] = 0.01 * time
    trajectories[2, 0, :, 0, 0] = -0.01 * time

    def metadata(identifier, gain, shift):
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
        protocol_id="grouped_abduction_unit",
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
    config = FactualAbductionConfig(observation_scale_m=0.001)
    evidence = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        mask=mask,
        prior_nominal_probability=0.99,
        source_id="unit_tracks",
    )
    return bank, belief, observations, mask, config, evidence


def test_grouped_abduction_recovers_matching_intervention() -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
    )
    joint = factual_joint_weights(factual, hypothesis_count=3, particle_count=1)
    assert np.argmax(joint[:, 0]) == 1
    assert factual.metadata["grouped_observation_evidence"]["group_count"] == 3
    result = evaluate_factual_abduction(
        bank,
        belief,
        factual,
        observations,
        observation_mask=mask,
        prefix_frame_count=4,
        config=config,
        grouped_evidence=evidence,
    )
    assert result["evidence_model"] == "grouped_robust_composite"
    assert result["map_hypothesis_id"] == "high_gain"


def test_unidentifiable_abduction_returns_exact_joint_prior() -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    identifiability = assess_intervention_identifiability(
        np.asarray([[1.0], [1.0]]),
        np.asarray([[1.0], [1.0]]),
    )
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        identifiability=identifiability,
        abstain_when_unidentifiable=True,
    )
    assert not identifiability.identifiable
    assert np.array_equal(factual.weights, bank.prior_joint_weights.reshape(-1))
    assert factual.metadata["abduction_abstained_unidentifiable"] is True


def _partially_identified_query(query_sensitivity: np.ndarray):
    return assess_intervention_identifiability(
        np.asarray([[1.0, 1.0], [0.0, 1.0]]),
        np.asarray([[0.0], [1.0]]),
        query_sensitivity=query_sensitivity,
        config=IdentifiabilityConfig(
            minimum_information_eigenvalue=1e-8,
            minimum_residualized_response_fraction=0.01,
            maximum_subspace_cosine=1.0,
            maximum_query_null_response_fraction=1e-8,
        ),
    )


def test_registered_query_policy_admits_query_despite_partial_parameter_rank() -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    identifiability = _partially_identified_query(np.asarray([[1.0, 1.0]]))
    assert not identifiability.identifiable
    assert identifiability.query_identifiable

    full_parameter = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        identifiability=identifiability,
        abstain_when_unidentifiable=True,
    )
    registered_query = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        identifiability=identifiability,
        abstain_when_unidentifiable=True,
        identifiability_policy="registered_query",
    )

    assert np.array_equal(
        full_parameter.weights,
        bank.prior_joint_weights.reshape(-1),
    )
    assert not np.array_equal(
        registered_query.weights,
        bank.prior_joint_weights.reshape(-1),
    )
    assert registered_query.metadata["identifiability_policy"] == "registered_query"
    assert registered_query.metadata["identifiability_policy_admitted"] is True


def test_registered_query_policy_falls_back_exactly_for_unresolved_query() -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    identifiability = _partially_identified_query(np.asarray([[1.0, -1.0]]))
    assert identifiability.query_identifiable is False
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        identifiability=identifiability,
        abstain_when_unidentifiable=True,
        identifiability_policy="registered_query",
    )
    assert np.array_equal(factual.weights, bank.prior_joint_weights.reshape(-1))
    assert factual.metadata["identifiability_policy_admitted"] is False


def test_registered_query_policy_requires_query_sensitivity() -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    identifiability = assess_intervention_identifiability(np.asarray([[1.0], [0.0]]))
    with pytest.raises(ValueError, match="requires query_sensitivity"):
        abduct_factual_intervention(
            bank,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            config=config,
            grouped_evidence=evidence,
            identifiability=identifiability,
            abstain_when_unidentifiable=True,
            identifiability_policy="registered_query",
        )


def test_structured_abduction_uncertainty_is_bound_recorded_and_round_trips(
    tmp_path,
) -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    group = evidence.groups[0]
    factor = np.zeros((group.coordinate_count, 1), dtype=float)
    factor[0, 0] = 0.01
    uncertainty = FactualAbductionUncertaintyV1(
        rollout_bank_id=bank.artifact_id,
        twin_belief_id=belief.artifact_id,
        grouped_evidence_id=evidence.evidence_id,
        source_artifact_ids=("prob4d-source-only-covariance",),
        source_only=True,
        disjoint_from_twin_belief_uncertainty=True,
        disjoint_from_grouped_observation_covariance=True,
        group_covariance_factor_m={group.group_id: factor},
    )
    path = tmp_path / "abduction-uncertainty.npz"
    save_factual_abduction_uncertainty_npz(path, uncertainty)
    restored = load_factual_abduction_uncertainty_npz(path)
    assert restored.artifact_id == uncertainty.artifact_id

    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=evidence,
        abduction_uncertainty=restored,
    )
    recorded = factual.metadata["factual_abduction_uncertainty"]
    assert recorded["artifact_id"] == uncertainty.artifact_id
    diagnostics = factual.metadata["grouped_observation_evidence"]["diagnostics"]
    assert diagnostics["low_rank_covariance_group_ids"] == [group.group_id]


def test_abduction_uncertainty_rejects_wrong_bindings() -> None:
    bank, belief, observations, mask, config, evidence = _problem()
    group = evidence.groups[0]
    uncertainty = FactualAbductionUncertaintyV1(
        rollout_bank_id="wrong-bank",
        twin_belief_id=belief.artifact_id,
        grouped_evidence_id=evidence.evidence_id,
        source_artifact_ids=("source-only-covariance",),
        source_only=True,
        disjoint_from_twin_belief_uncertainty=True,
        disjoint_from_grouped_observation_covariance=True,
        group_covariance_m2={group.group_id: np.eye(group.coordinate_count) * 1e-6},
    )
    with pytest.raises(ValueError, match="rollout_bank_id"):
        abduct_factual_intervention(
            bank,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            config=config,
            grouped_evidence=evidence,
            abduction_uncertainty=uncertainty,
        )


def test_abduction_uncertainty_requires_disjoint_source_declaration() -> None:
    bank, belief, _, _, _, evidence = _problem()
    group = evidence.groups[0]
    with pytest.raises(ValueError, match="disjoint-source declaration"):
        FactualAbductionUncertaintyV1(
            rollout_bank_id=bank.artifact_id,
            twin_belief_id=belief.artifact_id,
            grouped_evidence_id=evidence.evidence_id,
            source_artifact_ids=("combined-uncertainty",),
            source_only=True,
            disjoint_from_twin_belief_uncertainty=True,
            disjoint_from_grouped_observation_covariance=True,
            additional_independent_variance_m2=1e-6,
            group_covariance_factor_m={
                group.group_id: np.zeros((group.coordinate_count, 1))
            },
        )


def test_abduction_uncertainty_rejects_duplicate_dense_and_factor_routes() -> None:
    bank, belief, _, _, _, evidence = _problem()
    group = evidence.groups[0]
    with pytest.raises(ValueError, match="either dense or low-rank"):
        FactualAbductionUncertaintyV1(
            rollout_bank_id=bank.artifact_id,
            twin_belief_id=belief.artifact_id,
            grouped_evidence_id=evidence.evidence_id,
            source_artifact_ids=("duplicate-route",),
            source_only=True,
            disjoint_from_twin_belief_uncertainty=True,
            disjoint_from_grouped_observation_covariance=True,
            group_covariance_m2={group.group_id: np.eye(group.coordinate_count) * 1e-6},
            group_covariance_factor_m={
                group.group_id: np.zeros((group.coordinate_count, 1))
            },
        )
