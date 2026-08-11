from __future__ import annotations

import numpy as np
import pytest

from causal4d.cli.abduct_phystwin_intervention import build_parser
from causal4d.contracts import TwinBelief, build_causal_context
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    evaluate_factual_abduction,
)
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.rollout_bank import JointRolloutBank


def _metadata(identifier: str) -> dict:
    return {
        "hypothesis_id": identifier,
        "action": {
            "proposal_id": "known",
            "future_action_observed": True,
            "provenance": "likelihood semantics unit test",
        },
        "contact": {
            "attachment_shifts": [0],
            "gain_multiplier": 1.0,
            "delay_steps": 0,
            "slip_fraction": 0.0,
            "rotation_degrees": 0.0,
        },
    }


def _belief(
    bank: JointRolloutBank,
    *,
    variance: np.ndarray | None = None,
) -> TwinBelief:
    full_frames = bank.frame_count + 3
    intervention_frame = 4
    observations = np.zeros((full_frames, bank.node_count, bank.coordinate_count))
    actions = np.zeros((full_frames, 1, bank.coordinate_count))
    context = build_causal_context(
        protocol_id="likelihood-semantics-unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=intervention_frame,
    )
    particle_count = len(bank.parameter_weights)
    state_shape = (particle_count, bank.node_count, bank.coordinate_count)
    return TwinBelief(
        context=context,
        endpoint_frame=intervention_frame - 1,
        particle_ids=tuple(f"p{index}" for index in range(particle_count)),
        theta_names=("spring",),
        endpoint_position_m=np.zeros(state_shape),
        endpoint_velocity_mps=np.zeros(state_shape),
        theta=bank.parameter_particles,
        discrepancy_mean_m=np.zeros(state_shape),
        discrepancy_variance_m2=(
            np.zeros(state_shape) if variance is None else variance
        ),
        weights=bank.parameter_weights,
    )


def _registered_problem():
    bank_frames = 8
    trajectories = np.zeros((3, 1, bank_frames, 1, 3), dtype=float)
    time = np.arange(bank_frames, dtype=float)
    trajectories[1, 0, :, 0, 0] = 0.01 * time
    trajectories[2, 0, :, 0, 0] = -0.01 * time

    def metadata(identifier, gain, shift):
        result = _metadata(identifier)
        result["contact"] = {
            **result["contact"],
            "attachment_shifts": [shift],
            "gain_multiplier": gain,
        }
        return result

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
    belief = _belief(bank)
    observations = trajectories[1, 0].copy()
    mask = np.ones((bank_frames, 1), dtype=bool)
    config = FactualAbductionConfig(
        observation_scale_m=0.001,
        likelihood_power=60.0,
        dynamic_likelihood_weight=0.5,
    )
    return bank, belief, observations, mask, config


def test_legacy_v1_preserves_registered_artifact_identity() -> None:
    bank, belief, observations, mask, config = _registered_problem()

    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )

    assert factual.artifact_id == (
        "4b0d5227f9c561db217ef7f6a016d12bd5469e5e94ffcd877dfbeda26ebd491d"
    )
    assert factual.metadata["abduction_likelihood"] == {
        "degrees_of_freedom": 4.0,
        "dynamic_likelihood_weight": 0.5,
        "likelihood_power": 60.0,
        "observation_scale_m": 0.001,
    }
    assert "likelihood_semantics" not in factual.metadata["abduction_likelihood"]


def test_normalized_v2_pays_particle_specific_scale_normalization() -> None:
    trajectories = np.zeros((1, 2, 5, 1, 3), dtype=float)
    bank = JointRolloutBank(
        hypothesis_ids=("nominal",),
        hypothesis_metadata=(_metadata("nominal"),),
        hypothesis_prior_weights=np.asarray([1.0]),
        parameter_particles=np.asarray([[0.0], [1.0]]),
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
    )
    variance = np.zeros((2, 1, 3), dtype=float)
    variance[1] = 1.0
    belief = _belief(bank, variance=variance)
    observations = np.zeros((5, 1, 3), dtype=float)
    mask = np.ones((5, 1), dtype=bool)

    legacy = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=FactualAbductionConfig(
            observation_scale_m=0.1,
            likelihood_power=4.0,
            dynamic_likelihood_weight=0.0,
        ),
    )
    normalized = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=FactualAbductionConfig(
            observation_scale_m=0.1,
            likelihood_power=4.0,
            dynamic_likelihood_weight=0.0,
            likelihood_semantics="normalized_v2",
        ),
    )

    assert np.array_equal(legacy.weights, np.asarray([0.5, 0.5]))
    assert normalized.weights[0] > 0.999
    assert normalized.metadata["abduction_likelihood"] == {
        "observation_scale_m": 0.1,
        "likelihood_power": 4.0,
        "dynamic_likelihood_weight": 0.0,
        "degrees_of_freedom": 4.0,
        "likelihood_semantics": "normalized_v2",
        "difference_correlation": 0.0,
    }


def test_normalized_v2_includes_endpoint_to_first_response_increment() -> None:
    trajectories = np.zeros((2, 1, 6, 1, 3), dtype=float)
    trajectories[:, 0, 1:, 0, 0] = np.arange(1.0, 6.0)
    trajectories[0, 0, 0, 0, 0] = -2.0
    trajectories[1, 0, 0, 0, 0] = 0.0
    bank = JointRolloutBank(
        hypothesis_ids=("wrong_endpoint", "matching"),
        hypothesis_metadata=(
            _metadata("wrong_endpoint"),
            _metadata("matching"),
        ),
        hypothesis_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=np.asarray([[0.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )
    belief = _belief(bank)
    observations = trajectories[1, 0].copy()
    mask = np.ones((6, 1), dtype=bool)

    legacy = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=FactualAbductionConfig(
            observation_scale_m=0.1,
            likelihood_power=8.0,
            dynamic_likelihood_weight=1.0,
        ),
    )
    normalized = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=FactualAbductionConfig(
            observation_scale_m=0.1,
            likelihood_power=8.0,
            dynamic_likelihood_weight=1.0,
            likelihood_semantics="normalized_v2",
        ),
    )

    assert np.allclose(legacy.weights, [0.5, 0.5])
    assert normalized.weights[1] > 0.99


def test_normalized_v2_evaluation_records_effective_model() -> None:
    bank, belief, observations, mask, _ = _registered_problem()
    config = FactualAbductionConfig(
        observation_scale_m=0.001,
        likelihood_power=60.0,
        dynamic_likelihood_weight=0.5,
        likelihood_semantics="normalized_v2",
        difference_correlation=0.25,
    )
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
    )

    result = evaluate_factual_abduction(
        bank,
        belief,
        factual,
        observations,
        observation_mask=mask,
        prefix_frame_count=4,
        config=config,
    )

    assert result["evidence_model"] == "normalized_dense_v2"


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("observation_scale_m", np.nan),
        ("likelihood_power", np.inf),
        ("dynamic_likelihood_weight", np.nan),
        ("degrees_of_freedom", np.inf),
        ("difference_correlation", np.nan),
        ("difference_correlation", 1.0),
        ("grouped_covariance_condition_number_limit", np.nan),
        ("grouped_covariance_condition_number_limit", 0.5),
    ],
)
def test_factual_abduction_config_rejects_nonfinite_or_invalid_values(
    keyword: str,
    value: float,
) -> None:
    with pytest.raises(ValueError):
        FactualAbductionConfig(**{keyword: value})


def test_factual_abduction_config_rejects_unknown_grouped_semantics() -> None:
    with pytest.raises(ValueError, match="unsupported grouped likelihood semantics"):
        FactualAbductionConfig(grouped_likelihood_semantics="unknown")


def test_difference_correlation_requires_normalized_v2() -> None:
    with pytest.raises(ValueError, match="available only with normalized_v2"):
        FactualAbductionConfig(difference_correlation=0.25)


def test_normalized_v2_cannot_silently_bypass_grouped_evidence() -> None:
    bank, belief, observations, mask, _ = _registered_problem()
    grouped = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        mask=mask,
        source_id="unit",
    )

    with pytest.raises(ValueError, match="cannot be combined"):
        abduct_factual_intervention(
            bank,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            config=FactualAbductionConfig(likelihood_semantics="normalized_v2"),
            grouped_evidence=grouped,
        )


def test_cli_defaults_to_legacy_and_exposes_normalized_v2() -> None:
    parser = build_parser()
    default = parser.parse_args(["bank", "belief", "data", "factual", "evaluation"])
    normalized = parser.parse_args(
        [
            "bank",
            "belief",
            "data",
            "factual",
            "evaluation",
            "--likelihood-semantics",
            "normalized_v2",
            "--difference-correlation",
            "0.25",
        ]
    )

    assert default.likelihood_semantics == "legacy_v1"
    assert default.difference_correlation == 0.0
    assert normalized.likelihood_semantics == "normalized_v2"
    assert normalized.difference_correlation == 0.25


def test_grouped_normalized_v3_records_effective_model_and_diagnostics() -> None:
    bank, belief, observations, mask, _ = _registered_problem()
    grouped = GroupedObservationEvidence.from_dense_prefix(
        observations,
        prefix_frame_count=4,
        scale_m=0.001,
        mask=mask,
        source_id="unit",
    )
    config = FactualAbductionConfig(
        observation_scale_m=0.001,
        likelihood_power=12.0,
        grouped_likelihood_semantics="normalized_v3",
    )

    factual = abduct_factual_intervention(
        bank,
        belief,
        observations,
        prefix_frame_count=4,
        observation_mask=mask,
        config=config,
        grouped_evidence=grouped,
    )
    result = evaluate_factual_abduction(
        bank,
        belief,
        factual,
        observations,
        observation_mask=mask,
        prefix_frame_count=4,
        config=config,
        grouped_evidence=grouped,
    )

    assert result["evidence_model"] == "grouped_normalized_v3"
    assert factual.metadata["abduction_likelihood"] == {
        "observation_scale_m": 0.001,
        "likelihood_power": 12.0,
        "dynamic_likelihood_weight": 0.25,
        "degrees_of_freedom": 4.0,
        "grouped_likelihood_semantics": "normalized_v3",
        "grouped_covariance_condition_number_limit": 1.0e12,
    }
    diagnostics = factual.metadata["grouped_observation_evidence"]["diagnostics"]
    assert diagnostics["score_semantics"] == "normalized_coordinate_mean_v3"
    assert diagnostics["likelihood_power"] == 12.0
    assert np.isclose(sum(diagnostics["normalization_coordinate_fractions"]), 1.0)


def test_grouped_normalized_v3_requires_grouped_evidence() -> None:
    bank, belief, observations, mask, _ = _registered_problem()

    with pytest.raises(ValueError, match="requires grouped observation evidence"):
        abduct_factual_intervention(
            bank,
            belief,
            observations,
            prefix_frame_count=4,
            observation_mask=mask,
            config=FactualAbductionConfig(
                grouped_likelihood_semantics="normalized_v3"
            ),
        )


def test_cli_exposes_grouped_normalized_v3_separately() -> None:
    parsed = build_parser().parse_args(
        [
            "bank",
            "belief",
            "data",
            "factual",
            "evaluation",
            "--grouped-observation-likelihood",
            "--grouped-likelihood-semantics",
            "normalized_v3",
            "--grouped-covariance-condition-number-limit",
            "1000000",
        ]
    )

    assert parsed.likelihood_semantics == "legacy_v1"
    assert parsed.grouped_likelihood_semantics == "normalized_v3"
    assert parsed.grouped_covariance_condition_number_limit == 1.0e6
