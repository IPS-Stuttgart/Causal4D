"""Bounded-memory grouped-likelihood updates with streaming diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Any, Mapping, cast

import numpy as np

from causal4d.grouped_likelihood import (
    GroupLikelihoodDiagnostics,
    GroupedScoreSemantics,
    grouped_component_log_likelihoods,
)
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.weighting import log_weights_from_probabilities


@dataclass(frozen=True)
class GroupedLikelihoodSummaryDiagnostics:
    """Per-group diagnostics without a component-by-group matrix."""

    group_ids: tuple[str, ...]
    effective_group_weights: tuple[float, ...]
    mean_nominal_responsibility_by_group: tuple[float, ...]
    minimum_nominal_responsibility_by_group: tuple[float, ...]
    component_count: int
    full_covariance_group_ids: tuple[str, ...] = ()
    low_rank_covariance_group_ids: tuple[str, ...] = ()
    score_semantics: GroupedScoreSemantics = "legacy_sum_v1"
    likelihood_power: float = 1.0
    contributor_power_caps: tuple[float, ...] = ()
    group_coordinate_counts: tuple[int, ...] = ()
    normalization_coordinate_mass: float | None = None
    source_covariance_condition_numbers: tuple[float, ...] = ()
    normalization_coordinate_fractions: tuple[float, ...] = ()
    responsibility_storage: str = "streaming_summary"

    def as_dict(self) -> dict[str, Any]:
        """Return a finite JSON-compatible summary."""

        result: dict[str, Any] = {
            "group_ids": list(self.group_ids),
            "effective_group_weights": list(self.effective_group_weights),
            "mean_nominal_responsibility_by_group": list(
                self.mean_nominal_responsibility_by_group
            ),
            "minimum_nominal_responsibility_by_group": list(
                self.minimum_nominal_responsibility_by_group
            ),
            "component_count": self.component_count,
            "responsibility_storage": self.responsibility_storage,
        }
        if self.full_covariance_group_ids:
            result["full_covariance_group_ids"] = list(
                self.full_covariance_group_ids
            )
        if self.low_rank_covariance_group_ids:
            result["low_rank_covariance_group_ids"] = list(
                self.low_rank_covariance_group_ids
            )
        if self.score_semantics != "legacy_sum_v1":
            result.update(
                {
                    "score_semantics": self.score_semantics,
                    "likelihood_power": self.likelihood_power,
                    "contributor_power_caps": list(self.contributor_power_caps),
                    "group_coordinate_counts": list(self.group_coordinate_counts),
                    "normalization_coordinate_mass": (
                        self.normalization_coordinate_mass
                    ),
                    "source_covariance_condition_numbers": list(
                        self.source_covariance_condition_numbers
                    ),
                    "normalization_coordinate_fractions": list(
                        self.normalization_coordinate_fractions
                    ),
                }
            )
        return result


def _batch_indices(
    leading_shape: tuple[int, ...],
    start: int,
    stop: int,
) -> tuple[np.ndarray, ...] | None:
    if not leading_shape:
        return None
    flat = np.arange(start, stop, dtype=np.int64)
    return tuple(np.asarray(item) for item in np.unravel_index(flat, leading_shape))


def _batch_values(
    values: np.ndarray,
    *,
    leading_shape: tuple[int, ...],
    tail_shape: tuple[int, ...],
    start: int,
    stop: int,
    name: str,
) -> np.ndarray:
    try:
        broadcast = np.broadcast_to(
            np.asarray(values, dtype=float),
            (*leading_shape, *tail_shape),
        )
    except ValueError as error:
        raise ValueError(
            f"{name} must broadcast to component dimensions and {tail_shape}"
        ) from error
    indices = _batch_indices(leading_shape, start, stop)
    if indices is None:
        return np.asarray(broadcast, dtype=float)[None]
    return cast(np.ndarray, broadcast[indices])


def _batch_mapping(
    values: Mapping[str, np.ndarray] | None,
    evidence: GroupedObservationEvidence,
    *,
    leading_shape: tuple[int, ...],
    start: int,
    stop: int,
    factors: bool,
) -> dict[str, np.ndarray]:
    supplied = dict(values or {})
    groups = {group.group_id: group for group in evidence.groups}
    unknown = set(supplied) - set(groups)
    if unknown:
        raise ValueError(f"component covariance references unknown groups: {unknown}")
    result: dict[str, np.ndarray] = {}
    for group_id, raw_value in supplied.items():
        coordinate_count = groups[group_id].coordinate_count
        raw = np.asarray(raw_value, dtype=float)
        if factors:
            if raw.ndim < 2 or raw.shape[-2] != coordinate_count or raw.shape[-1] < 1:
                raise ValueError(
                    "covariance factors must end in "
                    "(group_coordinate, positive_rank)"
                )
            tail = (coordinate_count, raw.shape[-1])
        else:
            tail = (coordinate_count, coordinate_count)
        result[group_id] = _batch_values(
            raw,
            leading_shape=leading_shape,
            tail_shape=tail,
            start=start,
            stop=stop,
            name=f"component covariance for group {group_id!r}",
        )
    return result


def _signature(diagnostics: GroupLikelihoodDiagnostics) -> tuple[Any, ...]:
    return (
        diagnostics.group_ids,
        diagnostics.effective_group_weights,
        diagnostics.full_covariance_group_ids,
        diagnostics.low_rank_covariance_group_ids,
        diagnostics.score_semantics,
        diagnostics.likelihood_power,
        diagnostics.contributor_power_caps,
        diagnostics.group_coordinate_counts,
        diagnostics.normalization_coordinate_mass,
        diagnostics.source_covariance_condition_numbers,
        diagnostics.normalization_coordinate_fractions,
    )


def posterior_weights_from_grouped_evidence_batched(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
    component_batch_size: int,
    component_variance_m2: np.ndarray | None = None,
    component_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
    component_group_covariance_factor_m: Mapping[str, np.ndarray] | None = None,
    score_semantics: GroupedScoreSemantics = "legacy_sum_v1",
    likelihood_power: float = 1.0,
    max_source_covariance_condition_number: float = 1.0e12,
) -> tuple[np.ndarray, GroupedLikelihoodSummaryDiagnostics]:
    """Update finite support in batches and stream responsibility summaries."""

    if type(component_batch_size) is not int or component_batch_size < 1:
        raise ValueError("component_batch_size must be a positive integer")
    if not isinstance(evidence, GroupedObservationEvidence):
        raise TypeError("evidence must be GroupedObservationEvidence")
    components = np.asarray(predicted_components_m, dtype=float)
    if components.ndim < 4:
        raise ValueError("predicted_components_m must end in (frame, node, coordinate)")
    if not np.all(np.isfinite(components)):
        raise ValueError("predicted components must be finite")
    leading_shape = components.shape[:-3]
    tail_shape = components.shape[-3:]
    component_count = int(prod(leading_shape))
    if component_count < 1:
        raise ValueError("predicted components must contain finite support")

    prior = np.asarray(prior_weights, dtype=float)
    if prior.shape != leading_shape:
        raise ValueError("prior_weights must match the component leading dimensions")
    if (
        not np.all(np.isfinite(prior))
        or np.any(prior < 0.0)
        or not np.isclose(np.sum(prior), 1.0)
    ):
        raise ValueError("prior_weights must be finite, nonnegative, and sum to one")

    scores = np.empty(component_count, dtype=float)
    responsibility_sum: np.ndarray | None = None
    responsibility_minimum: np.ndarray | None = None
    signature: tuple[Any, ...] | None = None
    first: GroupLikelihoodDiagnostics | None = None

    for start in range(0, component_count, component_batch_size):
        stop = min(start + component_batch_size, component_count)
        batch_components = _batch_values(
            components,
            leading_shape=leading_shape,
            tail_shape=tail_shape,
            start=start,
            stop=stop,
            name="predicted_components_m",
        )
        batch_variance = None
        if component_variance_m2 is not None:
            batch_variance = _batch_values(
                component_variance_m2,
                leading_shape=leading_shape,
                tail_shape=tail_shape,
                start=start,
                stop=stop,
                name="component_variance_m2",
            )
        batch_scores, diagnostics = grouped_component_log_likelihoods(
            batch_components,
            evidence,
            prefix_frame_count=prefix_frame_count,
            component_variance_m2=batch_variance,
            component_group_covariance_m2=_batch_mapping(
                component_group_covariance_m2,
                evidence,
                leading_shape=leading_shape,
                start=start,
                stop=stop,
                factors=False,
            ),
            component_group_covariance_factor_m=_batch_mapping(
                component_group_covariance_factor_m,
                evidence,
                leading_shape=leading_shape,
                start=start,
                stop=stop,
                factors=True,
            ),
            score_semantics=score_semantics,
            likelihood_power=likelihood_power,
            max_source_covariance_condition_number=(
                max_source_covariance_condition_number
            ),
        )
        scores[start:stop] = np.asarray(batch_scores, dtype=float).reshape(-1)
        responsibilities = np.asarray(
            diagnostics.nominal_responsibilities,
            dtype=float,
        ).reshape(stop - start, -1)
        if signature is None:
            signature = _signature(diagnostics)
            first = diagnostics
            responsibility_sum = np.zeros(responsibilities.shape[1], dtype=float)
            responsibility_minimum = np.full(
                responsibilities.shape[1],
                np.inf,
            )
        elif _signature(diagnostics) != signature:
            raise RuntimeError("grouped diagnostics changed between component batches")
        assert responsibility_sum is not None
        assert responsibility_minimum is not None
        responsibility_sum += np.sum(responsibilities, axis=0)
        responsibility_minimum = np.minimum(
            responsibility_minimum,
            np.min(responsibilities, axis=0),
        )

    if first is None or responsibility_sum is None or responsibility_minimum is None:
        raise RuntimeError("grouped batched update produced no component scores")
    log_posterior = (
        log_weights_from_probabilities(prior.reshape(-1), name="prior_weights")
        + scores
    )
    maximum = float(np.max(log_posterior))
    posterior = np.exp(log_posterior - maximum)
    posterior /= np.sum(posterior)
    summary = GroupedLikelihoodSummaryDiagnostics(
        group_ids=first.group_ids,
        effective_group_weights=first.effective_group_weights,
        mean_nominal_responsibility_by_group=tuple(
            float(value) for value in responsibility_sum / component_count
        ),
        minimum_nominal_responsibility_by_group=tuple(
            float(value) for value in responsibility_minimum
        ),
        component_count=component_count,
        full_covariance_group_ids=first.full_covariance_group_ids,
        low_rank_covariance_group_ids=first.low_rank_covariance_group_ids,
        score_semantics=first.score_semantics,
        likelihood_power=first.likelihood_power,
        contributor_power_caps=first.contributor_power_caps,
        group_coordinate_counts=first.group_coordinate_counts,
        normalization_coordinate_mass=first.normalization_coordinate_mass,
        source_covariance_condition_numbers=(
            first.source_covariance_condition_numbers
        ),
        normalization_coordinate_fractions=first.normalization_coordinate_fractions,
    )
    return posterior.reshape(leading_shape), summary


__all__ = [
    "GroupedLikelihoodSummaryDiagnostics",
    "posterior_weights_from_grouped_evidence_batched",
]
