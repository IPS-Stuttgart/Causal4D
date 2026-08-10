"""Prepared, reusable full-joint inference for validated Prob4D evidence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from causal4d.joint_observation import LinearJointObservationEvidence
from causal4d.prepared_joint_observation import (
    PreparedJointGaussianLikelihoodDiagnostics,
    PreparedJointObservation,
    posterior_weights_from_prepared_joint_observation,
    prepare_joint_observation,
    prepared_joint_component_log_likelihoods,
)
from causal4d.prob4d_joint_observation import (
    Prob4DJointObservationDiagnostics,
    Prob4DReliabilityPolicy,
    joint_observation_from_prob4d,
)


@dataclass(frozen=True, slots=True)
class PreparedProb4DJointObservation:
    """One validated Prob4D artifact with a reusable exact inference plan."""

    evidence: LinearJointObservationEvidence
    adapter_diagnostics: Prob4DJointObservationDiagnostics
    prepared: PreparedJointObservation

    def __post_init__(self) -> None:
        if not isinstance(self.evidence, LinearJointObservationEvidence):
            raise TypeError("evidence must be LinearJointObservationEvidence")
        if not isinstance(self.adapter_diagnostics, Prob4DJointObservationDiagnostics):
            raise TypeError(
                "adapter_diagnostics must be Prob4DJointObservationDiagnostics"
            )
        if not isinstance(self.prepared, PreparedJointObservation):
            raise TypeError("prepared must be PreparedJointObservation")
        if self.prepared.evidence.artifact_id != self.evidence.artifact_id:
            raise ValueError(
                "prepared inference plan must bind the same Prob4D evidence artifact"
            )
        if (
            self.adapter_diagnostics.observation_count
            != self.evidence.observation_count
        ):
            raise ValueError(
                "Prob4D adapter diagnostics and prepared evidence dimensions differ"
            )

    @property
    def artifact_id(self) -> str:
        """Return the content identity of the consumed Prob4D evidence."""

        return self.evidence.artifact_id

    def log_likelihoods(
        self,
        predicted_components_m: np.ndarray,
        *,
        prefix_frame_count: int,
        component_independent_variance_m2: np.ndarray | None = None,
        component_joint_covariance_m2: np.ndarray | None = None,
        component_joint_covariance_factor_m: np.ndarray | None = None,
        component_chunk_size: int | None = None,
        maximum_working_bytes: int = 256 * 1024**2,
    ) -> tuple[np.ndarray, PreparedJointGaussianLikelihoodDiagnostics]:
        """Score finite rollout support with the cached Prob4D plan."""

        return prepared_joint_component_log_likelihoods(
            predicted_components_m,
            self.prepared,
            prefix_frame_count=prefix_frame_count,
            component_independent_variance_m2=component_independent_variance_m2,
            component_joint_covariance_m2=component_joint_covariance_m2,
            component_joint_covariance_factor_m=component_joint_covariance_factor_m,
            component_chunk_size=component_chunk_size,
            maximum_working_bytes=maximum_working_bytes,
        )

    def posterior_weights(
        self,
        prior_weights: np.ndarray,
        predicted_components_m: np.ndarray,
        *,
        prefix_frame_count: int,
        component_independent_variance_m2: np.ndarray | None = None,
        component_joint_covariance_m2: np.ndarray | None = None,
        component_joint_covariance_factor_m: np.ndarray | None = None,
        component_chunk_size: int | None = None,
        maximum_working_bytes: int = 256 * 1024**2,
    ) -> tuple[np.ndarray, PreparedJointGaussianLikelihoodDiagnostics]:
        """Update finite support while preserving exact zero prior mass."""

        return posterior_weights_from_prepared_joint_observation(
            prior_weights,
            predicted_components_m,
            self.prepared,
            prefix_frame_count=prefix_frame_count,
            component_independent_variance_m2=component_independent_variance_m2,
            component_joint_covariance_m2=component_joint_covariance_m2,
            component_joint_covariance_factor_m=component_joint_covariance_factor_m,
            component_chunk_size=component_chunk_size,
            maximum_working_bytes=maximum_working_bytes,
        )


def _descriptor_with_source_identity(
    descriptor: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve producer identity from the portable descriptor metadata.

    Historical adapter fixtures place ``source_revision`` and
    ``source_artifact_sha256`` at the top level, whereas portable strict Prob4D
    artifacts bind the same fields in ``metadata``. Accept both representations,
    reject disagreement, and pass one canonical mapping to the existing adapter.
    """

    normalized = dict(descriptor)
    metadata = descriptor.get("metadata")
    if metadata is None:
        return normalized
    if not isinstance(metadata, Mapping):
        return normalized

    for name in ("source_revision", "source_artifact_sha256"):
        direct = descriptor.get(name)
        nested = metadata.get(name)
        if direct is not None and nested is not None and direct != nested:
            raise ValueError(f"Prob4D {name} differs between descriptor and metadata")
        if direct is None and nested is not None:
            normalized[name] = nested
    return normalized


def prepare_prob4d_joint_observation(
    descriptor: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    *,
    rollout_frame_ids: Sequence[int],
    entity_to_node: Mapping[int, int],
    reliability_policy: Prob4DReliabilityPolicy = "require_neutral",
    evidence_id: str | None = None,
) -> PreparedProb4DJointObservation:
    """Validate, adapt, and compile one Prob4D observation exactly once."""

    normalized_descriptor = _descriptor_with_source_identity(descriptor)
    evidence, diagnostics = joint_observation_from_prob4d(
        normalized_descriptor,
        arrays,
        rollout_frame_ids=rollout_frame_ids,
        entity_to_node=entity_to_node,
        reliability_policy=reliability_policy,
        evidence_id=evidence_id,
    )
    return PreparedProb4DJointObservation(
        evidence=evidence,
        adapter_diagnostics=diagnostics,
        prepared=prepare_joint_observation(evidence),
    )


__all__ = [
    "PreparedProb4DJointObservation",
    "prepare_prob4d_joint_observation",
]
