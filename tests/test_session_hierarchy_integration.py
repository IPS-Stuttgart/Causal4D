import numpy as np

from causal4d.hierarchical_abduction import abduct_hierarchical_interventions
from causal4d.prefix_likelihood import PrefixLikelihoodConfig
from causal4d.rollout_bank import JointRolloutBank


def _bank() -> JointRolloutBank:
    trajectories = np.zeros((4, 1, 5, 1, 3), dtype=float)
    slopes = [0.0, 0.10, 0.12, 0.24]
    for index, slope in enumerate(slopes):
        trajectories[index, 0, :, 0, 0] = slope * np.arange(5, dtype=float)
    metadata = tuple(
        {
            "contact": {
                "gain_multiplier": gain,
                "delay_steps": 0,
                "rotation_degrees": 0,
                "attachment_shifts": [shift],
                "slip_fraction": 0.0,
            }
        }
        for gain, shift in ((0.8, -1), (0.8, 1), (1.2, -1), (1.2, 1))
    )
    return JointRolloutBank(
        hypothesis_ids=("low-left", "low-right", "high-left", "high-right"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.full(4, 0.25),
        parameter_particles=np.asarray([[1.0]]),
        parameter_weights=np.asarray([1.0]),
        trajectories=trajectories,
    )


def _prefix_config() -> PrefixLikelihoodConfig:
    return PrefixLikelihoodConfig(
        observation_scale_m=0.05,
        likelihood_power=2.0,
        dynamic_likelihood_weight=0.5,
    )


def test_identity_session_transition_is_exact_legacy_fallback() -> None:
    bank = _bank()
    observations = [
        bank.trajectories[1, 0].copy(),
        bank.trajectories[2, 0].copy(),
    ]
    arguments = {
        "prefix_frame_counts": [4, 4],
        "config": _prefix_config(),
        "session_ids": ["session-low", "session-high"],
    }
    legacy = abduct_hierarchical_interventions(
        [bank, bank],
        observations,
        **arguments,
    )
    hierarchical = abduct_hierarchical_interventions(
        [bank, bank],
        observations,
        session_phi_transition=np.eye(2),
        **arguments,
    )

    np.testing.assert_array_equal(hierarchical.shared_weights, legacy.shared_weights)
    for actual, expected in zip(
        hierarchical.execution_joint_weights,
        legacy.execution_joint_weights,
        strict=True,
    ):
        np.testing.assert_array_equal(actual, expected)
    assert "session_hierarchy_mode" not in legacy.metadata
    assert hierarchical.metadata["session_hierarchy_mode"] == (
        "zero_variance_identity"
    )
    assert hierarchical.session_hierarchy is not None
    assert hierarchical.session_hierarchy.mode == "zero_variance_identity"
    for session_weights in hierarchical.session_hierarchy.session_joint_weights:
        np.testing.assert_array_equal(session_weights, legacy.shared_weights)


def test_session_transition_retains_session_specific_persistent_state() -> None:
    bank = _bank()
    result = abduct_hierarchical_interventions(
        [bank, bank],
        [
            bank.trajectories[1, 0].copy(),
            bank.trajectories[2, 0].copy(),
        ],
        prefix_frame_counts=[4, 4],
        config=_prefix_config(),
        session_ids=["session-low", "session-high"],
        session_phi_transition=np.asarray([[0.9, 0.1], [0.1, 0.9]]),
    )

    assert result.session_hierarchy is not None
    assert result.session_hierarchy.mode == "finite_session_transition"
    low_session, high_session = result.session_phi_marginals
    assert low_session[0] > high_session[0]
    assert high_session[1] > low_session[1]
    np.testing.assert_allclose(
        np.sum(result.session_hierarchy.session_joint_weights[0], axis=0),
        result.parameter_marginal,
    )
