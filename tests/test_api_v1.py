from __future__ import annotations

import causal4d

from causal4d import api
from causal4d.api import v1


EXPECTED_V1_EXPORTS = (
    "BASE_CAUSAL4D_PROVIDER_CAPABILITIES",
    "BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS",
    "BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE",
    "BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES",
    "INTERVENTIONAL_CONTRAST_SCHEMA_VERSION",
    "JOINT_OBSERVATION_SCHEMA_VERSION",
    "PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION",
    "PUBLIC_API_NAME",
    "PUBLIC_API_VERSION",
    "ContrastConditionalVariancePolicy",
    "ContrastCouplingPolicy",
    "CounterfactualQuery",
    "CovarianceRepresentation",
    "FactualIntervention",
    "HierarchicalAbductionResult",
    "InterventionalContrastPosteriorV1",
    "InterventionalContrastQueryV1",
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
    "build_interventional_contrast",
    "joint_component_log_likelihoods",
    "load_bayesian_phystwin_provider_manifest",
    "load_interventional_contrast",
    "posterior_weights_from_joint_observation",
    "prefix_component_log_likelihood",
    "require_bayesian_phystwin_provider",
    "save_interventional_contrast",
    "update_joint_weights_from_prefix",
    "validate_bayesian_phystwin_provider",
    "validate_provider_compatibility",
)


def test_api_package_exposes_the_versioned_namespace() -> None:
    assert api.__all__ == ["v1"]
    assert api.v1 is v1


def test_v1_surface_is_explicit_and_unique() -> None:
    assert tuple(v1.__all__) == EXPECTED_V1_EXPORTS
    assert len(v1.__all__) == len(set(v1.__all__))
    assert v1.PUBLIC_API_NAME == "causal4d.api.v1"
    assert v1.PUBLIC_API_VERSION == 1
    for name in v1.__all__:
        assert hasattr(v1, name)


def test_v1_preserves_existing_top_level_symbol_identity() -> None:
    for name in EXPECTED_V1_EXPORTS:
        if name.startswith("PUBLIC_API_"):
            continue
        assert getattr(v1, name) is getattr(causal4d, name)


def test_v1_does_not_admit_experimental_or_protocol_internals() -> None:
    forbidden = {
        "ContactPatchStateV2",
        "ProspectiveV2PromotionFreezeV1",
        "SemanticTimingMetadata",
        "build_registered_real_analysis_manifest",
    }
    assert forbidden.isdisjoint(v1.__all__)
