"""Action-bound factual abduction for future claim-bearing integrations.

The historical factual-abduction path is preserved for frozen compatibility.
This additive wrapper binds the finite rollout bank to the exact observed action
identity before evaluating evidence and records that binding in the resulting
artifact metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

import numpy as np

from causal4d.contracts import FactualIntervention, TwinBelief
from causal4d.factual_abduction_uncertainty import FactualAbductionUncertaintyV1
from causal4d.identifiability import InterventionIdentifiabilityResult
from causal4d.intervention_abduction import (
    FactualAbductionConfig,
    abduct_factual_intervention,
)
from causal4d.observation_evidence import GroupedObservationEvidence
from causal4d.rollout_bank import JointRolloutBank


IdentifiabilityPolicy = Literal["full_parameter", "registered_query"]
ACTION_BINDING_SCHEMA_NAME = "causal4d.factual-rollout-action-binding"
ACTION_BINDING_SCHEMA_VERSION = 1


def validate_factual_rollout_action_binding(
    bank: JointRolloutBank,
    belief: TwinBelief,
) -> dict[str, Any]:
    """Require every factual hypothesis to represent the exact observed action."""

    expected_action_id = belief.context.u_obs.action_id
    observed_ids: list[str] = []
    for index, metadata in enumerate(bank.hypothesis_metadata):
        action = metadata.get("action")
        if not isinstance(action, Mapping):
            raise ValueError(f"hypothesis {index} has no action metadata mapping")
        proposal_id = action.get("proposal_id")
        if type(proposal_id) is not str or not proposal_id:
            raise ValueError(
                f"hypothesis {index} action proposal_id must be a nonempty string"
            )
        future_action_observed = action.get("future_action_observed")
        if type(future_action_observed) is not bool:
            raise ValueError(
                f"hypothesis {index} future_action_observed must be boolean"
            )
        if not future_action_observed:
            raise ValueError(
                f"hypothesis {index} does not represent the factual observed action"
            )
        if proposal_id != expected_action_id:
            raise ValueError(
                "factual rollout action identity differs from TwinBelief u_obs: "
                f"expected {expected_action_id!r}, observed {proposal_id!r}"
            )
        observed_ids.append(proposal_id)

    return {
        "schema_name": ACTION_BINDING_SCHEMA_NAME,
        "schema_version": ACTION_BINDING_SCHEMA_VERSION,
        "rollout_bank_id": bank.artifact_id,
        "source_twin_belief_id": belief.artifact_id,
        "protocol_id": belief.context.protocol_id,
        "case_id": belief.context.case_id,
        "observed_action_id": expected_action_id,
        "observed_action_trajectory_sha256": (
            belief.context.u_obs.trajectory_sha256
        ),
        "observed_action_frame_start": belief.context.u_obs.frame_start,
        "observed_action_frame_stop": belief.context.u_obs.frame_stop,
        "hypothesis_count": len(bank.hypothesis_ids),
        "hypothesis_action_ids": observed_ids,
        "all_hypotheses_marked_factual": True,
    }


def abduct_factual_intervention_bound(
    bank: JointRolloutBank,
    belief: TwinBelief,
    observations_from_endpoint_m: np.ndarray,
    *,
    prefix_frame_count: int,
    observation_mask: np.ndarray | None = None,
    config: FactualAbductionConfig | None = None,
    grouped_evidence: GroupedObservationEvidence | None = None,
    identifiability: InterventionIdentifiabilityResult | None = None,
    abstain_when_unidentifiable: bool = False,
    identifiability_policy: IdentifiabilityPolicy = "full_parameter",
    abduction_uncertainty: FactualAbductionUncertaintyV1 | None = None,
    dense_component_batch_size: int | None = None,
    grouped_component_batch_size: int | None = None,
) -> FactualIntervention:
    """Run factual abduction after exact observed-action identity validation."""

    binding = validate_factual_rollout_action_binding(bank, belief)
    factual = abduct_factual_intervention(
        bank,
        belief,
        observations_from_endpoint_m,
        prefix_frame_count=prefix_frame_count,
        observation_mask=observation_mask,
        config=config,
        grouped_evidence=grouped_evidence,
        identifiability=identifiability,
        abstain_when_unidentifiable=abstain_when_unidentifiable,
        identifiability_policy=identifiability_policy,
        abduction_uncertainty=abduction_uncertainty,
        dense_component_batch_size=dense_component_batch_size,
        grouped_component_batch_size=grouped_component_batch_size,
    )
    metadata = dict(factual.metadata)
    metadata["factual_rollout_action_binding"] = binding
    return FactualIntervention(
        context=factual.context,
        component_ids=factual.component_ids,
        phi_names=factual.phi_names,
        kappa_names=factual.kappa_names,
        phi=factual.phi,
        kappa_obs=factual.kappa_obs,
        hypothesis_indices=factual.hypothesis_indices,
        twin_particle_indices=factual.twin_particle_indices,
        weights=factual.weights,
        evidence_frame_stop=factual.evidence_frame_stop,
        source_twin_belief_id=factual.source_twin_belief_id,
        metadata=metadata,
    )


__all__ = [
    "ACTION_BINDING_SCHEMA_NAME",
    "ACTION_BINDING_SCHEMA_VERSION",
    "abduct_factual_intervention_bound",
    "validate_factual_rollout_action_binding",
]
