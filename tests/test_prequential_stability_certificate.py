from __future__ import annotations

import numpy as np
import pytest

from causal4d.prequential_abduction import PrequentialAbductionPathV1
from causal4d.prequential_query_stability import PrequentialQueryStabilityV1
from causal4d.prequential_stability_certificate import (
    PrequentialStabilityRuleV1,
    build_prequential_stability_certificate,
)


PREFIXES = np.asarray([2, 3, 4, 5], dtype=np.int64)
FACTUAL_IDS = tuple(character * 64 for character in "3456")
FALLBACK_ID = "f" * 64


def _summaries(weights: np.ndarray):
    positive = weights > 0.0
    terms = np.zeros_like(weights)
    terms[positive] = weights[positive] * np.log(weights[positive])
    entropy = -np.sum(terms, axis=1)
    ess = 1.0 / np.sum(np.square(weights), axis=1)
    maximum = np.max(weights, axis=1)
    maps = np.argmax(weights, axis=1).astype(np.int64)
    kl = np.zeros(len(weights))
    tv = np.zeros(len(weights))
    for index in range(1, len(weights)):
        current = weights[index]
        previous = np.maximum(weights[index - 1], np.finfo(float).tiny)
        kl[index] = float(np.sum(current * np.log(current / previous)))
        tv[index] = float(0.5 * np.sum(np.abs(current - weights[index - 1])))
    return entropy, ess, maximum, maps, kl, tv


def _path(*, unstable_last: bool = False, suffix: str = ""):
    weights = np.asarray(
        [
            [0.50, 0.50],
            [0.55, 0.45],
            [0.56, 0.44],
            [0.90, 0.10] if unstable_last else [0.565, 0.435],
        ]
    )
    values = _summaries(weights)
    return PrequentialAbductionPathV1(
        source_rollout_bank_id="1" * 64,
        source_twin_belief_id="2" * 64,
        component_ids=("component-0", "component-1"),
        factual_intervention_ids=FACTUAL_IDS,
        step_evidence_ids=tuple(character * 64 for character in "789a"),
        prefix_frame_counts=PREFIXES,
        evidence_frame_stops=np.asarray([3, 4, 5, 6], dtype=np.int64),
        posterior_weights=weights,
        posterior_entropy=values[0],
        posterior_effective_sample_size=values[1],
        posterior_maximum_weight=values[2],
        map_component_indices=values[3],
        previous_step_kl=values[4],
        previous_step_total_variation=values[5],
        metadata={"future_frames_read": 0, "suffix": suffix},
    )


def _stability(path: PrequentialAbductionPathV1):
    return PrequentialQueryStabilityV1(
        source_prequential_path_id=path.artifact_id,
        query_id="registered-query",
        query_labels=("query-output",),
        query_units=("m",),
        query_scales=np.asarray([1.0]),
        confidence_level=0.90,
        prefix_frame_counts=path.prefix_frame_counts,
        posterior_weights=path.posterior_weights,
        component_query_values=np.asarray([[0.0], [1.0]]),
        metadata={"future_frames_read": 0},
    )


def _rule(**changes):
    values = {
        "threshold_source_id": "b" * 64,
        "fallback_artifact_id": FALLBACK_ID,
        "maximum_previous_mean_shift_standardized_l2": 0.10,
        "maximum_previous_gaussian_wasserstein_standardized": 0.10,
        "minimum_previous_interval_overlap_fraction": 0.85,
        "maximum_previous_posterior_kl": 0.10,
        "maximum_previous_posterior_total_variation": 0.10,
        "minimum_effective_sample_size": 1.5,
        "required_consecutive_steps": 2,
        "maximum_prefix_frame_count": 5,
    }
    values.update(changes)
    return PrequentialStabilityRuleV1(**values)


def test_selects_earliest_consecutive_stable_prefix() -> None:
    path = _path()
    result = build_prequential_stability_certificate(_stability(path), path, _rule())
    assert result.stable
    assert result.accepted_step_index == 2
    assert result.accepted_prefix_frame_count == 4
    assert result.selected_posterior_id == FACTUAL_IDS[2]
    np.testing.assert_array_equal(result.step_passes, [False, True, True, True])


def test_unstable_path_routes_exact_fallback() -> None:
    path = _path(unstable_last=True)
    result = build_prequential_stability_certificate(
        _stability(path),
        path,
        _rule(required_consecutive_steps=3),
    )
    assert result.exact_fallback_required
    assert result.selected_posterior_id == FALLBACK_ID
    assert result.decision == "exact_fallback_no_stable_prefix"


def test_later_prefix_change_cannot_change_accepted_earlier_prefix() -> None:
    first = _path()
    second = _path(unstable_last=True)
    result_a = build_prequential_stability_certificate(
        _stability(first), first, _rule()
    )
    result_b = build_prequential_stability_certificate(
        _stability(second), second, _rule()
    )
    assert result_a.accepted_step_index == result_b.accepted_step_index == 2
    assert result_a.selected_posterior_id == result_b.selected_posterior_id


def test_path_ancestry_and_fallback_identity_fail_closed() -> None:
    path = _path()
    other = _path(suffix="different")
    with pytest.raises(ValueError, match="does not match"):
        build_prequential_stability_certificate(_stability(path), other, _rule())
    with pytest.raises(ValueError, match="distinct from every prefix"):
        build_prequential_stability_certificate(
            _stability(path),
            path,
            _rule(fallback_artifact_id=FACTUAL_IDS[0]),
        )
