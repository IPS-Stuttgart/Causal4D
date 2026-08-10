import json

import numpy as np
import pytest

from causal4d.artifact_io import ArtifactValidationError
from causal4d.contracts import FactualIntervention, array_sha256, build_causal_context
from causal4d.independent_sensor_ablation import (
    INDEPENDENT_SENSOR_ABLATION_ARMS,
    build_independent_sensor_ablation,
    load_independent_sensor_ablation_report,
    save_independent_sensor_ablation_report,
)
from causal4d.sensor_evidence import ActuatorEvidence, ContactWrenchEvidence


def _factual(weights: np.ndarray | None = None) -> FactualIntervention:
    observations = np.zeros((8, 2, 3), dtype=float)
    actions = np.zeros((8, 1, 3), dtype=float)
    context = build_causal_context(
        protocol_id="sensor_ablation_unit",
        case_id="unit_case",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=4,
    )
    return FactualIntervention(
        context=context,
        component_ids=("phi0-k0", "phi0-k1", "phi1-k0", "phi1-k1"),
        phi_names=("gain_multiplier", "delay_steps", "rotation_degrees"),
        kappa_names=("contact_node", "slip_fraction"),
        phi=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.8, 1.0, 0.0],
                [0.8, 1.0, 0.0],
            ]
        ),
        kappa_obs=np.asarray(
            [
                [0.0, 0.0],
                [1.0, 0.2],
                [0.0, 0.0],
                [1.0, 0.2],
            ]
        ),
        hypothesis_indices=np.arange(4),
        twin_particle_indices=np.zeros(4, dtype=np.int64),
        weights=(
            np.full(4, 0.25, dtype=float)
            if weights is None
            else np.asarray(weights, dtype=float)
        ),
        evidence_frame_stop=6,
        source_twin_belief_id=array_sha256(np.zeros(1)),
    )


def _identity() -> dict[str, str]:
    return {
        "protocol_id": "sensor_ablation_unit",
        "case_id": "unit_case",
        "observed_action_id": "u_obs",
    }


def _actuator() -> tuple[ActuatorEvidence, np.ndarray]:
    positions = np.zeros((2, 1, 3), dtype=float)
    evidence = ActuatorEvidence(
        **_identity(),
        stream_id="measured_end_effector",
        clock_id="robot_monotonic",
        provenance="encoder stream independent of object observations",
        sample_times_s=np.asarray([0.0, 1.0 / 30.0]),
        positions_m=positions,
        variance_m2=np.full_like(positions, 1.0e-4),
        evidence_frame_stop=6,
    )
    predictions = np.stack(
        (
            positions,
            positions,
            positions + 0.2,
            positions + 0.2,
        ),
        axis=0,
    )
    return evidence, predictions


def _wrench(
    *,
    clock_id: str = "robot_monotonic",
) -> tuple[ContactWrenchEvidence, np.ndarray]:
    wrench = np.asarray([[1.0, 0.0, 0.0]])
    evidence = ContactWrenchEvidence(
        **_identity(),
        stream_id="wrist_force",
        clock_id=clock_id,
        provenance="force stream independent of object observations",
        sample_times_s=np.asarray([0.0]),
        wrench=wrench,
        variance=np.full_like(wrench, 1.0e-3),
        quantity_names=("force_x_n", "force_y_n", "force_z_n"),
        evidence_frame_stop=6,
    )
    predictions = np.stack(
        (
            wrench,
            np.zeros_like(wrench),
            wrench,
            np.zeros_like(wrench),
        ),
        axis=0,
    )
    return evidence, predictions


def _build() -> object:
    factual = _factual()
    actuator, actuator_predictions = _actuator()
    wrench, wrench_predictions = _wrench()
    return build_independent_sensor_ablation(
        factual,
        actuator_evidence=actuator,
        predicted_actuator_positions_m=actuator_predictions,
        wrench_evidence=wrench,
        predicted_contact_wrench=wrench_predictions,
        component_metrics={"trajectory_error_mm": np.asarray([0.0, 5.0, 10.0, 15.0])},
        metric_units={"trajectory_error_mm": "mm"},
        metadata={"diagnostic_only": True},
    )


def test_ablation_separates_phi_and_kappa_information() -> None:
    result = _build()
    np.testing.assert_array_less(
        0.999,
        np.sum(result.actuator_only.weights[:2]),
    )
    np.testing.assert_array_less(
        0.999,
        np.sum(result.wrench_only.weights[[0, 2]]),
    )
    assert result.actuator_and_wrench.weights[0] > 0.999

    attribution = result.report.attribution
    phi = attribution["phi_entropy_reduction_nats"]
    kappa = attribution["kappa_entropy_reduction_nats"]
    assert phi["actuator_only"] > 0.5
    assert abs(phi["wrench_only"]) < 1.0e-10
    assert kappa["wrench_only"] > 0.5
    assert abs(kappa["actuator_only"]) < 1.0e-10
    assert phi["actuator_and_wrench"] > 0.5
    assert kappa["actuator_and_wrench"] > 0.5

    evidence = result.report.evidence
    assert evidence["object_observation_likelihood_reused"] is False
    assert evidence["future_object_frames_read"] == 0
    assert evidence["common_clock_verified"] is True


def test_component_metrics_are_bound_and_attributed() -> None:
    result = _build()
    arms = result.report.arm_summaries
    source = arms["object_prefix"]["component_metrics"]["trajectory_error_mm"]
    combined = arms["actuator_and_wrench"]["component_metrics"]["trajectory_error_mm"]
    assert source["posterior_expected_value"] == pytest.approx(7.5)
    assert combined["posterior_expected_value"] < 0.01
    assert len(source["component_values_sha256"]) == 64

    metric = result.report.attribution["component_metrics"]["trajectory_error_mm"]
    assert metric["unit"] == "mm"
    assert (
        metric["expected_improvement_over_object_prefix"]["actuator_and_wrench"] > 7.49
    )
    assert metric["combined_increment_over_best_single"] > 2.49


def test_absent_evidence_preserves_every_arm_exactly() -> None:
    factual = _factual()
    result = build_independent_sensor_ablation(factual)
    for arm_name in INDEPENDENT_SENSOR_ABLATION_ARMS:
        assert result.posterior(arm_name) is factual
        assert result.report.arm_summaries[arm_name]["exact_source_fallback"] is True
    with pytest.raises(KeyError, match="unknown independent-sensor"):
        result.posterior("not-an-arm")


def test_component_invariant_factors_preserve_exact_fallback() -> None:
    factual = _factual()
    actuator, _ = _actuator()
    wrench, _ = _wrench()
    actuator_predictions = np.broadcast_to(
        actuator.positions_m,
        (4,) + actuator.positions_m.shape,
    ).copy()
    wrench_predictions = np.broadcast_to(
        wrench.wrench,
        (4,) + wrench.wrench.shape,
    ).copy()
    result = build_independent_sensor_ablation(
        factual,
        actuator_evidence=actuator,
        predicted_actuator_positions_m=actuator_predictions,
        wrench_evidence=wrench,
        predicted_contact_wrench=wrench_predictions,
    )
    assert result.actuator_only is factual
    assert result.wrench_only is factual
    assert result.actuator_and_wrench is factual


def test_zero_prior_support_is_never_resurrected() -> None:
    factual = _factual(np.asarray([0.5, 0.5, 0.0, 0.0]))
    actuator, actuator_predictions = _actuator()
    result = build_independent_sensor_ablation(
        factual,
        actuator_evidence=actuator,
        predicted_actuator_positions_m=actuator_predictions,
    )
    np.testing.assert_array_equal(result.actuator_only.weights[2:], 0.0)
    np.testing.assert_array_equal(result.actuator_and_wrench.weights[2:], 0.0)


def test_report_round_trip_is_closed_and_exactly_once(tmp_path) -> None:
    result = _build()
    path = tmp_path / "independent-sensor-ablation.json"
    save_independent_sensor_ablation_report(path, result.report)
    restored = load_independent_sensor_ablation_report(path)
    assert restored.artifact_id == result.report.artifact_id
    assert restored.as_dict() == result.report.as_dict()
    with pytest.raises(FileExistsError):
        save_independent_sensor_ablation_report(path, result.report)


def test_report_rejects_tampering(tmp_path) -> None:
    result = _build()
    payload = result.report.as_dict()
    payload["arms"]["object_prefix"]["effective_sample_size"] = 999.0
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ArtifactValidationError, match="identity mismatch"):
        load_independent_sensor_ablation_report(path)


def test_report_metadata_is_deeply_immutable() -> None:
    result = _build()
    with pytest.raises(TypeError, match="immutable"):
        result.report.arm_summaries["object_prefix"]["map_component_id"] = "x"
    with pytest.raises(TypeError, match="immutable"):
        result.report.metadata["diagnostic_only"] = False


def test_component_metric_contract_fails_closed() -> None:
    factual = _factual()
    with pytest.raises(ValueError, match="must have shape"):
        build_independent_sensor_ablation(
            factual,
            component_metrics={"bad": np.ones(3)},
        )
    with pytest.raises(ValueError, match="finite and nonnegative"):
        build_independent_sensor_ablation(
            factual,
            component_metrics={"bad": np.asarray([0.0, 1.0, -1.0, 2.0])},
        )
    with pytest.raises(ValueError, match="identify every component metric"):
        build_independent_sensor_ablation(
            factual,
            component_metrics={"loss": np.arange(4, dtype=float)},
            metric_units={"other": "mm"},
        )


def test_combined_arm_rejects_mismatched_clocks() -> None:
    factual = _factual()
    actuator, actuator_predictions = _actuator()
    wrench, wrench_predictions = _wrench(clock_id="force_clock")
    with pytest.raises(ValueError, match="must use the same clock"):
        build_independent_sensor_ablation(
            factual,
            actuator_evidence=actuator,
            predicted_actuator_positions_m=actuator_predictions,
            wrench_evidence=wrench,
            predicted_contact_wrench=wrench_predictions,
        )


def test_report_binds_prediction_arrays() -> None:
    factual = _factual()
    actuator, predictions = _actuator()
    first = build_independent_sensor_ablation(
        factual,
        actuator_evidence=actuator,
        predicted_actuator_positions_m=predictions,
    )
    changed = predictions.copy()
    changed[2:] += 0.01
    second = build_independent_sensor_ablation(
        factual,
        actuator_evidence=actuator,
        predicted_actuator_positions_m=changed,
    )
    binding = first.report.evidence["predicted_actuator_positions_m"]
    assert list(binding["shape"]) == list(predictions.shape)
    assert len(binding["sha256"]) == 64
    assert first.report.artifact_id != second.report.artifact_id


def test_ablation_rejects_already_sensor_updated_source() -> None:
    factual = _factual()
    actuator, predictions = _actuator()
    first = build_independent_sensor_ablation(
        factual,
        actuator_evidence=actuator,
        predicted_actuator_positions_m=predictions,
    )
    with pytest.raises(ValueError, match="must precede independent-sensor"):
        build_independent_sensor_ablation(first.actuator_only)
