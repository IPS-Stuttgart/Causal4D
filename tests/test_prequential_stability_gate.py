from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.contracts import FactualIntervention, build_causal_context
from causal4d.prequential_abduction import (
    PrequentialAbductionPathV1,
    PrequentialAbductionResult,
)
from causal4d.prequential_query_stability import build_prequential_query_stability
from causal4d.prequential_stability_gate import (
    PREQUENTIAL_STABILITY_CRITERIA,
    PrequentialStabilityGateConfigV1,
    evaluate_prequential_stability,
    load_prequential_stability_decision,
    load_prequential_stability_gate_config,
    route_prequential_factual_intervention,
    write_prequential_stability_decision,
    write_prequential_stability_gate_config,
)


def _posterior_summaries(weights: np.ndarray) -> tuple[np.ndarray, ...]:
    positive = weights > 0.0
    entropy_terms = np.zeros_like(weights)
    entropy_terms[positive] = weights[positive] * np.log(weights[positive])
    entropy = -np.sum(entropy_terms, axis=1)
    effective_sample_size = 1.0 / np.sum(np.square(weights), axis=1)
    maximum_weight = np.max(weights, axis=1)
    map_component_indices = np.argmax(weights, axis=1).astype(np.int64)
    previous_step_kl = np.zeros(len(weights), dtype=float)
    previous_step_total_variation = np.zeros(len(weights), dtype=float)
    for index in range(1, len(weights)):
        current = weights[index]
        previous = weights[index - 1]
        selected = current > 0.0
        previous_safe = np.maximum(previous[selected], np.finfo(float).tiny)
        previous_step_kl[index] = float(
            np.sum(current[selected] * np.log(current[selected] / previous_safe))
        )
        previous_step_total_variation[index] = float(
            0.5 * np.sum(np.abs(current - previous))
        )
    return (
        entropy,
        effective_sample_size,
        maximum_weight,
        map_component_indices,
        previous_step_kl,
        previous_step_total_variation,
    )


def _result() -> PrequentialAbductionResult:
    weights = np.asarray(
        [
            [0.50, 0.50],
            [0.55, 0.45],
            [0.56, 0.44],
            [0.561, 0.439],
            [0.90, 0.10],
        ],
        dtype=float,
    )
    observations = np.zeros((9, 1, 3), dtype=float)
    actions = np.zeros_like(observations)
    context = build_causal_context(
        protocol_id="prequential-stability-unit",
        case_id="synthetic",
        observations=observations,
        observed_actions=actions,
        counterfactual_actions=actions,
        intervention_frame=2,
    )
    factual = tuple(
        FactualIntervention(
            context=context,
            component_ids=("component-a", "component-b"),
            phi_names=(),
            kappa_names=(),
            phi=np.zeros((2, 0), dtype=float),
            kappa_obs=np.zeros((2, 0), dtype=float),
            hypothesis_indices=np.asarray([0, 1], dtype=np.int64),
            twin_particle_indices=np.asarray([0, 0], dtype=np.int64),
            weights=row,
            evidence_frame_stop=stop,
            source_twin_belief_id="b" * 64,
            metadata={"prefix": prefix},
        )
        for row, prefix, stop in zip(
            weights,
            (2, 3, 4, 5, 6),
            (3, 4, 5, 6, 7),
            strict=True,
        )
    )
    summaries = _posterior_summaries(weights)
    path = PrequentialAbductionPathV1(
        source_rollout_bank_id="a" * 64,
        source_twin_belief_id="b" * 64,
        component_ids=("component-a", "component-b"),
        factual_intervention_ids=tuple(step.artifact_id for step in factual),
        step_evidence_ids=tuple(f"prefix:{value}" for value in (2, 3, 4, 5, 6)),
        prefix_frame_counts=np.asarray([2, 3, 4, 5, 6], dtype=np.int64),
        evidence_frame_stops=np.asarray([3, 4, 5, 6, 7], dtype=np.int64),
        posterior_weights=weights,
        posterior_entropy=summaries[0],
        posterior_effective_sample_size=summaries[1],
        posterior_maximum_weight=summaries[2],
        map_component_indices=summaries[3],
        previous_step_kl=summaries[4],
        previous_step_total_variation=summaries[5],
        metadata={"future_frames_read": 0},
    )
    return PrequentialAbductionResult(path=path, factual_interventions=factual)


def _stability(result: PrequentialAbductionResult):
    return build_prequential_query_stability(
        result.path,
        np.asarray([[0.0], [1.0]], dtype=float),
        query_id="registered-prefix-query-v1",
        query_labels=("endpoint-x",),
        query_units=("m",),
        query_scales=(1.0,),
        metadata={"registered_before_target_access": True},
    )


def _config(**overrides: object) -> PrequentialStabilityGateConfigV1:
    values: dict[str, object] = {
        "minimum_prefix_frame_count": 3,
        "required_consecutive_passes": 2,
        "maximum_previous_total_variation": 0.02,
        "maximum_previous_kl": 10.0,
        "minimum_effective_sample_size": 1.0,
        "maximum_query_mean_shift_standardized_l2": 10.0,
        "maximum_query_wasserstein_standardized": 10.0,
        "minimum_query_interval_overlap_fraction": 0.0,
        "source_artifact_ids": ("c" * 64,),
        "source_only": True,
        "registered_before_target_access": True,
        "metadata": {"selection_units": ["source-session-a", "source-session-b"]},
    }
    values.update(overrides)
    return PrequentialStabilityGateConfigV1(**values)


def test_stability_gate_selects_first_prospectively_stable_prefix() -> None:
    result = _result()
    decision = evaluate_prequential_stability(
        result.path,
        _stability(result),
        _config(),
    )

    assert decision.status == "accepted"
    assert decision.selected_step_index == 3
    assert decision.selected_prefix_frame_count == 5
    assert decision.selected_evidence_frame_stop == 6
    assert decision.consecutive_pass_counts.tolist() == [0, 0, 1, 2, 0]
    assert decision.step_pass.tolist() == [False, False, True, True, False]
    assert decision.metadata["uses_previous_prefix_metrics_only"] is True
    assert decision.metadata["uses_final_path_metrics"] is False
    assert tuple(decision.as_dict()["criteria"]) == PREQUENTIAL_STABILITY_CRITERIA

    selected = route_prequential_factual_intervention(
        result,
        decision,
        fallback=result.factual_interventions[0],
    )
    assert selected is result.factual_interventions[3]


def test_stability_gate_fallback_preserves_exact_object_identity() -> None:
    result = _result()
    decision = evaluate_prequential_stability(
        result.path,
        _stability(result),
        _config(maximum_previous_total_variation=0.0001),
    )
    fallback = result.factual_interventions[0]

    assert decision.status == "fallback"
    assert decision.selected_step_index is None
    assert (
        route_prequential_factual_intervention(
            result,
            decision,
            fallback=fallback,
        )
        is fallback
    )


def test_stability_gate_rejects_mismatched_path_and_noncausal_input() -> None:
    result = _result()
    other = _result()
    # Rebind the otherwise equal path by changing finite metadata.
    from dataclasses import replace

    other_path = replace(other.path, metadata={"future_frames_read": 0, "other": True})
    with pytest.raises(ValueError, match="does not bind"):
        evaluate_prequential_stability(
            other_path,
            _stability(result),
            _config(),
        )
    noncausal = replace(result.path, metadata={"future_frames_read": 1})
    with pytest.raises(ValueError, match="future_frames_read=0"):
        evaluate_prequential_stability(
            noncausal,
            _stability(result),
            _config(),
        )


def test_stability_config_and_decision_round_trip_without_overwrite(
    tmp_path: Path,
) -> None:
    result = _result()
    config = _config()
    decision = evaluate_prequential_stability(
        result.path,
        _stability(result),
        config,
    )
    config_path = tmp_path / "config.json"
    decision_path = tmp_path / "decision.json"

    write_prequential_stability_gate_config(config_path, config)
    write_prequential_stability_decision(decision_path, decision)
    assert (
        load_prequential_stability_gate_config(config_path).artifact_id
        == config.artifact_id
    )
    assert (
        load_prequential_stability_decision(decision_path).artifact_id
        == decision.artifact_id
    )
    with pytest.raises(FileExistsError):
        write_prequential_stability_gate_config(config_path, config)
    with pytest.raises(FileExistsError):
        write_prequential_stability_decision(decision_path, decision)


def test_stability_decision_loader_rejects_coercive_boolean_payload(
    tmp_path: Path,
) -> None:
    result = _result()
    decision = evaluate_prequential_stability(
        result.path,
        _stability(result),
        _config(),
    )
    payload = decision.as_dict()
    payload["criterion_pass"][0][0] = 1
    path = tmp_path / "coercive.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="must contain Booleans"):
        load_prequential_stability_decision(path)


def test_stability_config_rejects_target_or_nonfinite_thresholds() -> None:
    with pytest.raises(ValueError, match="source_only must be explicitly true"):
        _config(source_only=False)
    with pytest.raises(ValueError, match="finite JSON number"):
        _config(maximum_previous_kl=np.inf)
    with pytest.raises(ValueError, match="must not exceed one"):
        _config(minimum_query_interval_overlap_fraction=1.1)


def test_stability_decision_enforces_first_prefix_and_fallback_consistency() -> None:
    from dataclasses import replace

    result = _result()
    decision = evaluate_prequential_stability(
        result.path,
        _stability(result),
        _config(required_consecutive_passes=1),
    )
    assert decision.selected_step_index == 2

    with pytest.raises(ValueError, match="first passing prefix"):
        replace(
            decision,
            selected_step_index=3,
            selected_prefix_frame_count=5,
            selected_evidence_frame_stop=6,
            selected_factual_intervention_id=(
                result.factual_interventions[3].artifact_id
            ),
        )

    with pytest.raises(ValueError, match="cannot contain a passing prefix"):
        replace(
            decision,
            status="fallback",
            selected_step_index=None,
            selected_prefix_frame_count=None,
            selected_evidence_frame_stop=None,
            selected_factual_intervention_id=None,
            fallback_reason="forced-test-fallback",
        )

    with pytest.raises(ValueError, match="evidence_frame_stop does not match"):
        replace(decision, selected_evidence_frame_stop=999)
    with pytest.raises(ValueError, match="factual_intervention_id does not match"):
        replace(
            decision,
            selected_factual_intervention_id=(
                result.factual_interventions[4].artifact_id
            ),
        )


def test_stability_decision_allows_repeated_factual_artifact_ids() -> None:
    from dataclasses import replace

    result = _result()
    decision = evaluate_prequential_stability(
        result.path,
        _stability(result),
        _config(maximum_previous_total_variation=0.0001),
    )
    repeated = (decision.factual_intervention_ids[0],) * len(
        decision.factual_intervention_ids
    )

    restored = replace(decision, factual_intervention_ids=repeated)

    assert restored.status == "fallback"
    assert restored.factual_intervention_ids == repeated
