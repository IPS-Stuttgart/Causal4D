"""Version 1 of Causal4D's supported Python API.

Only names listed in ``__all__`` are covered by the v1 compatibility promise.
Research prototypes, registered-protocol internals, and command implementations
remain available from their owning modules but are not part of this surface.
"""

from __future__ import annotations

from typing import Final

from causal4d.contracts import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TaskPosterior,
    TwinBelief,
)
from causal4d.counterfactual import apply_counterfactual_operator
from causal4d.hierarchical_abduction import (
    HierarchicalAbductionResult,
    abduct_hierarchical_interventions,
)
from causal4d.joint_observation import (
    JOINT_OBSERVATION_SCHEMA_VERSION,
    CovarianceRepresentation,
    JointGaussianLikelihoodDiagnostics,
    LinearJointObservationEvidence,
    block_diagonalize_covariance,
    joint_component_log_likelihoods,
    posterior_weights_from_joint_observation,
)
from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig,
    prefix_component_log_likelihood,
    update_joint_weights_from_prefix,
)
from causal4d.provider_contract import (
    BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
    PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
    PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult,
    load_bayesian_phystwin_provider_manifest,
    require_bayesian_phystwin_provider,
    validate_bayesian_phystwin_provider,
    validate_provider_compatibility,
)
from causal4d.rollout_bank import JointRolloutBank, SparseTrajectoryEvidence


PUBLIC_API_NAME: Final = "causal4d.api.v1"
PUBLIC_API_VERSION: Final = 1

__all__ = [
    "BASE_CAUSAL4D_PROVIDER_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE",
    "BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES",
    "JOINT_OBSERVATION_SCHEMA_VERSION",
    "PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION",
    "PUBLIC_API_NAME",
    "PUBLIC_API_VERSION",
    "CounterfactualQuery",
    "CovarianceRepresentation",
    "FactualIntervention",
    "HierarchicalAbductionResult",
    "JointGaussianLikelihoodDiagnostics",
    "JointRolloutBank",
    "LinearJointObservationEvidence",
    "PhysicalBeliefProviderManifest",
    "PhysicalPosterior",
    "PrefixLikelihoodConfig",
    "ProviderCompatibilityResult",
    "SparseTrajectoryEvidence",
    "TaskPosterior",
    "TwinBelief",
    "abduct_hierarchical_interventions",
    "apply_counterfactual_operator",
    "block_diagonalize_covariance",
    "joint_component_log_likelihoods",
    "load_bayesian_phystwin_provider_manifest",
    "posterior_weights_from_joint_observation",
    "prefix_component_log_likelihood",
    "require_bayesian_phystwin_provider",
    "update_joint_weights_from_prefix",
    "validate_bayesian_phystwin_provider",
    "validate_provider_compatibility",
]
