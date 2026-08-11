"""Leakage-safe posterior paths for factual intervention abduction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import re
from typing import Any

import numpy as np

from causal4d.contracts import FactualIntervention, TwinBelief, array_sha256
from causal4d.factual_abduction_uncertainty import FactualAbductionUncertaintyV1
from causal4d.identifiability import InterventionIdentifiabilityResult
from causal4d.immutable_array import readonly_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    IdentifiabilityPolicy,
    abduct_factual_intervention,
)
from causal4d.observation_evidence import (
    GroupedObservationEvidence,
    ObservationGroup,
)
from causal4d.rollout_bank import JointRolloutBank


PREQUENTIAL_ABDUCTION_PATH_SCHEMA_VERSION = 1
_PREQUENTIAL_ABDUCTION_PATH_KIND = "Causal4DPrequentialAbductionPathV1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _require_sha256(value: object, *, name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _string_tuple(
    values: Sequence[str],
    *,
    name: str,
    expected_count: int | None = None,
    unique: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(values)
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"{name} must contain {expected_count} entries")
    if not result or any(type(value) is not str or not value for value in result):
        raise ValueError(f"{name} must contain nonempty strings")
    if unique and len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _integer_vector(values: object, *, name: str) -> np.ndarray:
    raw = np.asarray(values)
    if raw.ndim != 1 or not len(raw) or not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"{name} must be a nonempty integer vector")
    result = readonly_array(raw, dtype=np.int64)
    if np.any(result < 0):
        raise ValueError(f"{name} must be nonnegative")
    return result


def _posterior_summaries(
    posterior_weights: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    positive = posterior_weights > 0.0
    entropy_terms = np.zeros_like(posterior_weights)
    entropy_terms[positive] = posterior_weights[positive] * np.log(
        posterior_weights[positive]
    )
    entropy = -np.sum(entropy_terms, axis=1)
    effective_sample_size = 1.0 / np.sum(np.square(posterior_weights), axis=1)
    maximum_weight = np.max(posterior_weights, axis=1)
    map_component_indices = np.argmax(posterior_weights, axis=1).astype(np.int64)
    previous_step_kl = np.zeros(len(posterior_weights), dtype=float)
    previous_step_total_variation = np.zeros(len(posterior_weights), dtype=float)
    for index in range(1, len(posterior_weights)):
        current = posterior_weights[index]
        previous = posterior_weights[index - 1]
        current_positive = current > 0.0
        previous_safe = np.maximum(
            previous[current_positive],
            np.finfo(float).tiny,
        )
        previous_step_kl[index] = max(
            0.0,
            float(
                np.sum(
                    current[current_positive]
                    * np.log(current[current_positive] / previous_safe)
                )
            ),
        )
        previous_step_total_variation[index] = float(
            0.5 * np.sum(np.abs(current - previous))
        )
    return (
        readonly_array(entropy, dtype=float),
        readonly_array(effective_sample_size, dtype=float),
        readonly_array(maximum_weight, dtype=float),
        readonly_array(map_component_indices, dtype=np.int64),
        readonly_array(previous_step_kl, dtype=float),
        readonly_array(previous_step_total_variation, dtype=float),
    )


@dataclass(frozen=True)
class PrequentialAbductionPathV1:
    """Content-addressed posterior evolution over nested causal prefixes."""

    source_rollout_bank_id: str
    source_twin_belief_id: str
    component_ids: tuple[str, ...]
    factual_intervention_ids: tuple[str, ...]
    step_evidence_ids: tuple[str, ...]
    prefix_frame_counts: np.ndarray
    evidence_frame_stops: np.ndarray
    posterior_weights: np.ndarray
    posterior_entropy: np.ndarray
    posterior_effective_sample_size: np.ndarray
    posterior_maximum_weight: np.ndarray
    map_component_indices: np.ndarray
    previous_step_kl: np.ndarray
    previous_step_total_variation: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        bank_id = _require_sha256(
            self.source_rollout_bank_id,
            name="source_rollout_bank_id",
        )
        belief_id = _require_sha256(
            self.source_twin_belief_id,
            name="source_twin_belief_id",
        )
        components = _string_tuple(
            self.component_ids,
            name="component_ids",
            unique=True,
        )
        prefixes = _integer_vector(
            self.prefix_frame_counts,
            name="prefix_frame_counts",
        )
        step_count = len(prefixes)
        if prefixes[0] < 2 or np.any(np.diff(prefixes) <= 0):
            raise ValueError(
                "prefix_frame_counts must be strictly increasing and start at two"
            )
        stops = _integer_vector(
            self.evidence_frame_stops,
            name="evidence_frame_stops",
        )
        if stops.shape != prefixes.shape or np.any(np.diff(stops) <= 0):
            raise ValueError(
                "evidence_frame_stops must increase with prefix_frame_counts"
            )
        factual_ids = _string_tuple(
            self.factual_intervention_ids,
            name="factual_intervention_ids",
            expected_count=step_count,
        )
        for index, value in enumerate(factual_ids):
            _require_sha256(value, name=f"factual_intervention_ids[{index}]")
        evidence_ids = _string_tuple(
            self.step_evidence_ids,
            name="step_evidence_ids",
            expected_count=step_count,
        )
        weights = readonly_array(self.posterior_weights, dtype=float)
        if weights.shape != (step_count, len(components)):
            raise ValueError("posterior_weights must have shape (step, component)")
        if (
            not np.all(np.isfinite(weights))
            or np.any(weights < 0.0)
            or not np.allclose(
                np.sum(weights, axis=1),
                1.0,
                atol=1e-12,
                rtol=1e-10,
            )
        ):
            raise ValueError(
                "posterior_weights must be finite, nonnegative, and row-normalized"
            )
        expected = _posterior_summaries(weights)
        supplied = (
            readonly_array(self.posterior_entropy, dtype=float),
            readonly_array(self.posterior_effective_sample_size, dtype=float),
            readonly_array(self.posterior_maximum_weight, dtype=float),
            _integer_vector(
                self.map_component_indices,
                name="map_component_indices",
            ),
            readonly_array(self.previous_step_kl, dtype=float),
            readonly_array(self.previous_step_total_variation, dtype=float),
        )
        names = (
            "posterior_entropy",
            "posterior_effective_sample_size",
            "posterior_maximum_weight",
            "map_component_indices",
            "previous_step_kl",
            "previous_step_total_variation",
        )
        for name, actual, wanted in zip(names, supplied, expected, strict=True):
            if actual.shape != (step_count,):
                raise ValueError(f"{name} must identify every prefix step")
            if not np.all(np.isfinite(actual)) or not np.allclose(
                actual,
                wanted,
                atol=1e-12,
                rtol=1e-10,
            ):
                raise ValueError(f"{name} does not match posterior_weights")
        if np.any(supplied[3] >= len(components)):
            raise ValueError("map_component_indices exceed component support")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="prequential metadata must contain finite JSON data",
        )
        object.__setattr__(self, "source_rollout_bank_id", bank_id)
        object.__setattr__(self, "source_twin_belief_id", belief_id)
        object.__setattr__(self, "component_ids", components)
        object.__setattr__(self, "factual_intervention_ids", factual_ids)
        object.__setattr__(self, "step_evidence_ids", evidence_ids)
        object.__setattr__(self, "prefix_frame_counts", prefixes)
        object.__setattr__(self, "evidence_frame_stops", stops)
        object.__setattr__(self, "posterior_weights", weights)
        for name, value in zip(names, supplied, strict=True):
            object.__setattr__(self, name, value)
        object.__setattr__(self, "metadata", metadata)

    def as_dict(self) -> dict[str, Any]:
        """Return a finite JSON-compatible diagnostic summary."""

        return {
            "schema_version": PREQUENTIAL_ABDUCTION_PATH_SCHEMA_VERSION,
            "artifact_kind": _PREQUENTIAL_ABDUCTION_PATH_KIND,
            "artifact_id": self.artifact_id,
            "source_rollout_bank_id": self.source_rollout_bank_id,
            "source_twin_belief_id": self.source_twin_belief_id,
            "component_ids": list(self.component_ids),
            "factual_intervention_ids": list(self.factual_intervention_ids),
            "step_evidence_ids": list(self.step_evidence_ids),
            "prefix_frame_counts": self.prefix_frame_counts.tolist(),
            "evidence_frame_stops": self.evidence_frame_stops.tolist(),
            "posterior_entropy": self.posterior_entropy.tolist(),
            "posterior_effective_sample_size": (
                self.posterior_effective_sample_size.tolist()
            ),
            "posterior_maximum_weight": self.posterior_maximum_weight.tolist(),
            "map_component_indices": self.map_component_indices.tolist(),
            "previous_step_kl": self.previous_step_kl.tolist(),
            "previous_step_total_variation": (
                self.previous_step_total_variation.tolist()
            ),
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        descriptor = {
            "schema_version": PREQUENTIAL_ABDUCTION_PATH_SCHEMA_VERSION,
            "artifact_kind": _PREQUENTIAL_ABDUCTION_PATH_KIND,
            "source_rollout_bank_id": self.source_rollout_bank_id,
            "source_twin_belief_id": self.source_twin_belief_id,
            "component_ids": list(self.component_ids),
            "factual_intervention_ids": list(self.factual_intervention_ids),
            "step_evidence_ids": list(self.step_evidence_ids),
            "metadata": plain_json(self.metadata),
            "arrays": {
                name: array_sha256(getattr(self, name))
                for name in (
                    "prefix_frame_counts",
                    "evidence_frame_stops",
                    "posterior_weights",
                    "posterior_entropy",
                    "posterior_effective_sample_size",
                    "posterior_maximum_weight",
                    "map_component_indices",
                    "previous_step_kl",
                    "previous_step_total_variation",
                )
            },
        }
        encoded = json.dumps(
            descriptor,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class PrequentialAbductionResult:
    """A path artifact paired with its immutable factual posterior steps."""

    path: PrequentialAbductionPathV1
    factual_interventions: tuple[FactualIntervention, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.path, PrequentialAbductionPathV1):
            raise TypeError("path must be PrequentialAbductionPathV1")
        steps = tuple(self.factual_interventions)
        if len(steps) != len(self.path.prefix_frame_counts) or any(
            not isinstance(step, FactualIntervention) for step in steps
        ):
            raise ValueError(
                "factual_interventions must identify every prequential path step"
            )
        for index, step in enumerate(steps):
            if step.artifact_id != self.path.factual_intervention_ids[index]:
                raise ValueError("factual intervention ID does not match path")
            if step.component_ids != self.path.component_ids or not np.array_equal(
                step.weights,
                self.path.posterior_weights[index],
            ):
                raise ValueError("factual intervention support does not match path")
            if step.evidence_frame_stop != self.path.evidence_frame_stops[index]:
                raise ValueError(
                    "factual intervention evidence stop does not match path"
                )
        object.__setattr__(self, "factual_interventions", steps)


def grouped_observation_prefix(
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
) -> GroupedObservationEvidence:
    """Restrict grouped evidence to coordinates strictly before one prefix stop."""

    if not isinstance(evidence, GroupedObservationEvidence):
        raise TypeError("evidence must be GroupedObservationEvidence")
    if type(prefix_frame_count) is not int or prefix_frame_count < 2:
        raise ValueError("prefix_frame_count must be an integer of at least two")
    if any(np.any(group.frame_indices <= 0) for group in evidence.groups):
        raise ValueError("grouped O-plus evidence may not reuse the endpoint frame")
    if all(
        np.all(group.frame_indices < prefix_frame_count) for group in evidence.groups
    ):
        return evidence

    groups: list[ObservationGroup] = []
    split_group_ids: list[str] = []
    omitted_group_ids: list[str] = []
    for group in evidence.groups:
        selected = group.frame_indices < prefix_frame_count
        if not np.any(selected):
            omitted_group_ids.append(group.group_id)
            continue
        indices = np.flatnonzero(selected)
        if len(indices) != group.coordinate_count:
            split_group_ids.append(group.group_id)
        groups.append(
            ObservationGroup(
                group_id=group.group_id,
                values_m=group.values_m[indices],
                frame_indices=group.frame_indices[indices],
                node_indices=group.node_indices[indices],
                coordinate_indices=group.coordinate_indices[indices],
                covariance_m2=group.covariance_m2[np.ix_(indices, indices)],
                contributor_ids=group.contributor_ids,
                prior_nominal_probability=group.prior_nominal_probability,
                outlier_scale_multiplier=group.outlier_scale_multiplier,
                degrees_of_freedom=group.degrees_of_freedom,
                composite_weight=group.composite_weight,
                source_id=group.source_id,
                view_id=group.view_id,
                metadata={
                    "source_group_metadata": plain_json(group.metadata),
                    "source_group_coordinate_count": group.coordinate_count,
                    "prequential_prefix_frame_count": prefix_frame_count,
                },
            )
        )
    if not groups:
        raise ValueError("grouped evidence contains no coordinates before the prefix")
    return GroupedObservationEvidence(
        groups=tuple(groups),
        evidence_id=f"{evidence.evidence_id}:prefix:{prefix_frame_count}",
        metadata={
            "source_evidence_id": evidence.evidence_id,
            "source_evidence_metadata": plain_json(evidence.metadata),
            "prefix_frame_count_including_endpoint": prefix_frame_count,
            "future_frames_read": 0,
            "split_group_ids": split_group_ids,
            "omitted_group_ids": omitted_group_ids,
        },
    )


def _prefixes(
    values: Sequence[int] | None,
    *,
    bank: JointRolloutBank,
    belief: TwinBelief,
) -> tuple[int, ...]:
    maximum_from_context = (
        belief.context.o_plus.frame_stop - belief.context.o_plus.frame_start + 1
    )
    maximum = min(bank.frame_count - 1, maximum_from_context)
    if values is None:
        result = tuple(range(2, maximum + 1))
    else:
        if isinstance(values, (str, bytes)):
            raise ValueError("prefix_frame_counts must be a sequence of integers")
        result = tuple(values)
    if not result or any(type(value) is not int for value in result):
        raise ValueError("prefix_frame_counts must contain integers")
    if result[0] < 2 or any(
        current >= following for current, following in zip(result, result[1:])
    ):
        raise ValueError(
            "prefix_frame_counts must be strictly increasing and start at two"
        )
    if result[-1] > maximum:
        raise ValueError("prefix_frame_counts exceed the causal O-plus boundary")
    return result


def _validated_prefix_mapping(
    values: Mapping[int, Any] | None,
    *,
    prefixes: tuple[int, ...],
    name: str,
) -> dict[int, Any]:
    result = dict(values or {})
    if any(type(key) is not int for key in result):
        raise ValueError(f"{name} keys must be integer prefix frame counts")
    unknown = set(result) - set(prefixes)
    if unknown:
        raise ValueError(f"{name} contains unused prefixes: {sorted(unknown)}")
    return result


def build_prequential_abduction_path(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    *,
    prefix_frame_counts: Sequence[int] | None = None,
    observation_mask: np.ndarray | None = None,
    config: FactualAbductionConfig | None = None,
    grouped_evidence: GroupedObservationEvidence | None = None,
    identifiability_by_prefix: Mapping[
        int,
        InterventionIdentifiabilityResult,
    ]
    | None = None,
    abstain_when_unidentifiable: bool = False,
    identifiability_policy: IdentifiabilityPolicy = "full_parameter",
    abduction_uncertainty_by_prefix: Mapping[
        int,
        FactualAbductionUncertaintyV1,
    ]
    | None = None,
    grouped_component_batch_size: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> PrequentialAbductionResult:
    """Run ordinary factual abduction over nested, leakage-safe prefix stops.

    Every path element is produced by :func:`abduct_factual_intervention` with
    the same estimator settings as a standalone call. The operation is diagnostic:
    it does not feed an earlier or later posterior into another update. The final
    step is therefore required to equal an ordinary one-shot abduction exactly.
    """

    if not isinstance(bank, JointRolloutBank):
        raise TypeError("bank must be JointRolloutBank")
    if not isinstance(belief, TwinBelief):
        raise TypeError("belief must be TwinBelief")
    prefixes = _prefixes(prefix_frame_counts, bank=bank, belief=belief)
    identifiability = _validated_prefix_mapping(
        identifiability_by_prefix,
        prefixes=prefixes,
        name="identifiability_by_prefix",
    )
    uncertainty = _validated_prefix_mapping(
        abduction_uncertainty_by_prefix,
        prefixes=prefixes,
        name="abduction_uncertainty_by_prefix",
    )
    if grouped_evidence is None and uncertainty:
        raise ValueError(
            "abduction_uncertainty_by_prefix requires grouped observation evidence"
        )

    factual_steps: list[FactualIntervention] = []
    evidence_ids: list[str] = []
    for prefix in prefixes:
        step_evidence = (
            None
            if grouped_evidence is None
            else grouped_observation_prefix(
                grouped_evidence,
                prefix_frame_count=prefix,
            )
        )
        factual = abduct_factual_intervention(
            bank,
            belief,
            observations_from_endpoint_m,
            prefix_frame_count=prefix,
            observation_mask=observation_mask,
            config=config,
            grouped_evidence=step_evidence,
            identifiability=identifiability.get(prefix),
            abstain_when_unidentifiable=abstain_when_unidentifiable,
            identifiability_policy=identifiability_policy,
            abduction_uncertainty=uncertainty.get(prefix),
            grouped_component_batch_size=grouped_component_batch_size,
        )
        factual_steps.append(factual)
        evidence_ids.append(
            f"dense-prefix:{prefix}"
            if step_evidence is None
            else step_evidence.evidence_id
        )

    weights = np.stack([step.weights for step in factual_steps], axis=0)
    summaries = _posterior_summaries(weights)
    path = PrequentialAbductionPathV1(
        source_rollout_bank_id=bank.artifact_id,
        source_twin_belief_id=belief.artifact_id,
        component_ids=factual_steps[0].component_ids,
        factual_intervention_ids=tuple(step.artifact_id for step in factual_steps),
        step_evidence_ids=tuple(evidence_ids),
        prefix_frame_counts=np.asarray(prefixes, dtype=np.int64),
        evidence_frame_stops=np.asarray(
            [step.evidence_frame_stop for step in factual_steps],
            dtype=np.int64,
        ),
        posterior_weights=weights,
        posterior_entropy=summaries[0],
        posterior_effective_sample_size=summaries[1],
        posterior_maximum_weight=summaries[2],
        map_component_indices=summaries[3],
        previous_step_kl=summaries[4],
        previous_step_total_variation=summaries[5],
        metadata={
            "operator": "prequential-factual-abduction",
            "diagnostic_only": True,
            "changes_estimator": False,
            "changes_registered_protocol": False,
            "future_frames_read": 0,
            "grouped_observation_evidence": grouped_evidence is not None,
            "identifiability_policy": identifiability_policy,
            "previous_step_kl_zero_mass_floor": np.finfo(float).tiny,
            "user_metadata": plain_json(metadata or {}),
        },
    )
    return PrequentialAbductionResult(
        path=path,
        factual_interventions=tuple(factual_steps),
    )


__all__ = [
    "PREQUENTIAL_ABDUCTION_PATH_SCHEMA_VERSION",
    "PrequentialAbductionPathV1",
    "PrequentialAbductionResult",
    "build_prequential_abduction_path",
    "grouped_observation_prefix",
]
