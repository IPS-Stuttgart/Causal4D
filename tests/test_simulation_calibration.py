import numpy as np
import pytest

from causal4d.contact_inference import ContactRolloutBank, ContactState
from causal4d.simulation_calibration import run_contact_rollout_sbc
from causal4d.simulator import Action, GraphObject, PhysicalParameters


def _bank() -> ContactRolloutBank:
    graph_object = GraphObject(
        name="sbc_graph",
        rest_positions=np.asarray([[0.0, 0.0], [0.1, 0.0]]),
        edges=((0, 1),),
        mass=1.0,
        support_stiffness=0.2,
        true_parameters=PhysicalParameters(1.0, 1.0, 1.0),
        sensor_nodes=(0, 1),
    )
    action = Action(
        action_id="probe",
        split="test",
        contact_nodes=(0,),
        commanded_forces=np.zeros((4, 1, 2), dtype=float),
    )
    states = (
        ContactState((0,), 0.8, 0, 0.0, 0.0),
        ContactState((1,), 1.2, 1, 0.0, 0.0),
    )
    particles = np.asarray(
        [
            [0.8, 0.9, 0.85],
            [1.2, 1.1, 1.15],
        ],
        dtype=float,
    )
    trajectories = np.zeros((2, 2, 5, 2, 2), dtype=float)
    time = np.arange(5, dtype=float)
    for contact_index in range(2):
        for parameter_index in range(2):
            component = 2 * contact_index + parameter_index
            slope = 0.018 * (component + 1)
            trajectories[contact_index, parameter_index, :, :, 0] = (
                slope * time[:, None]
            )
            trajectories[contact_index, parameter_index, :, 0, 1] = 0.35 * slope * time
            trajectories[contact_index, parameter_index, :, 1, 1] = -0.20 * slope * time
    return ContactRolloutBank(
        graph_object=graph_object,
        action=action,
        contact_states=states,
        contact_prior_weights=np.asarray([0.5, 0.5]),
        parameter_particles=particles,
        parameter_weights=np.asarray([0.5, 0.5]),
        trajectories=trajectories,
        variance_floor_m2=1e-8,
        confidence_level=0.9,
    )


def test_prior_only_sbc_has_uniform_randomized_ranks() -> None:
    result = run_contact_rollout_sbc(
        _bank(),
        trials=3000,
        prefix_frame_count=3,
        likelihood_scale_m=0.01,
        likelihood_power=0.0,
        observation_noise_std_m=0.0,
        seed=123,
        bin_count=10,
    )

    assert result.joint_rank_max_abs_frequency_error < 0.03
    assert result.contact_rank_max_abs_frequency_error < 0.03
    assert max(result.parameter_rank_max_abs_frequency_error) < 0.03
    assert np.isclose(result.mean_entropy_reduction, 0.0, atol=1e-12)
    assert np.isclose(result.mean_true_joint_posterior_mass, 0.25)
    assert np.isclose(result.mean_true_contact_posterior_mass, 0.5)


def test_exact_gaussian_self_model_is_rank_calibrated_and_informative() -> None:
    result = run_contact_rollout_sbc(
        _bank(),
        trials=2500,
        prefix_frame_count=4,
        likelihood_scale_m=0.006,
        likelihood_power=1.0,
        dynamic_likelihood_weight=0.0,
        observation_noise_std_m=0.006,
        seed=321,
        bin_count=10,
    )

    assert result.joint_rank_max_abs_frequency_error < 0.035
    assert result.contact_rank_max_abs_frequency_error < 0.035
    assert max(result.parameter_rank_max_abs_frequency_error) < 0.035
    assert result.mean_entropy_reduction > 0.2
    assert result.mean_true_joint_posterior_mass > 0.5
    assert result.mean_joint_effective_sample_size < 4.0


def test_sbc_serialization_keeps_real_calibration_boundary_explicit() -> None:
    result = run_contact_rollout_sbc(
        _bank(),
        trials=25,
        prefix_frame_count=3,
        likelihood_scale_m=0.01,
        likelihood_power=0.0,
        observation_noise_std_m=0.0,
        seed=7,
        bin_count=5,
    )
    payload = result.as_dict()
    assert payload["schema"] == "causal4d.simulation_calibration"
    assert payload["schema_version"] == 1
    assert "not evidence of real-data calibration" in payload["interpretation"]
    assert sum(payload["rank_histograms"]["joint"]) == 25


@pytest.mark.parametrize(
    ("trials", "bin_count", "noise"),
    [
        (0, 10, 0.01),
        (10, 1, 0.01),
        (10, 10, -0.01),
    ],
)
def test_sbc_rejects_invalid_run_settings(
    trials: int,
    bin_count: int,
    noise: float,
) -> None:
    with pytest.raises(ValueError):
        run_contact_rollout_sbc(
            _bank(),
            trials=trials,
            prefix_frame_count=3,
            likelihood_scale_m=0.01,
            observation_noise_std_m=noise,
            bin_count=bin_count,
        )
