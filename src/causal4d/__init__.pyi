"""Typing surface for the compatibility-preserving package-root exports."""

from causal4d.action_conditioned_counterfactual import (
    ActionConditionedPhysicalPosterior as ActionConditionedPhysicalPosterior,
    apply_action_conditioned_counterfactual_operator as apply_action_conditioned_counterfactual_operator,
)
from causal4d.action_conditioned_discrepancy import (
    ActionConditionedDiscrepancyFeatures as ActionConditionedDiscrepancyFeatures,
    ActionConditionedDiscrepancyForecast as ActionConditionedDiscrepancyForecast,
    ActionConditionedDiscrepancyModel as ActionConditionedDiscrepancyModel,
    build_action_conditioned_features as build_action_conditioned_features,
    forecast_action_conditioned_persistence as forecast_action_conditioned_persistence,
)
from causal4d.action_support import (
    ACTION_SUPPORT_SCHEMA_VERSION as ACTION_SUPPORT_SCHEMA_VERSION,
    ActionSupportCalibration as ActionSupportCalibration,
    ActionSupportDecision as ActionSupportDecision,
    ActionSupportSelection as ActionSupportSelection,
    ActionSupportSourceCase as ActionSupportSourceCase,
    evaluate_action_support as evaluate_action_support,
    fit_action_support_calibration as fit_action_support_calibration,
    load_action_support_calibration as load_action_support_calibration,
    load_claim_bearing_action_support_calibration as load_claim_bearing_action_support_calibration,
    select_action_supported_candidate as select_action_supported_candidate,
    write_action_support_calibration as write_action_support_calibration,
)
from causal4d.action_support_counterfactual import (
    apply_guarded_action_conditioned_counterfactual_operator as apply_guarded_action_conditioned_counterfactual_operator,
)
from causal4d.counterfactual_regret import (
    COUNTERFACTUAL_REGRET_ENDPOINTS as COUNTERFACTUAL_REGRET_ENDPOINTS,
    COUNTERFACTUAL_REGRET_SCHEMA_VERSION as COUNTERFACTUAL_REGRET_SCHEMA_VERSION,
    CounterfactualRegretCertificate as CounterfactualRegretCertificate,
    CounterfactualRegretDecision as CounterfactualRegretDecision,
    CounterfactualRegretFeatures as CounterfactualRegretFeatures,
    CounterfactualRegretPrerequisite as CounterfactualRegretPrerequisite,
    CounterfactualRegretSelection as CounterfactualRegretSelection,
    CounterfactualRegretSourceCase as CounterfactualRegretSourceCase,
    CounterfactualRegretTarget as CounterfactualRegretTarget,
    evaluate_counterfactual_regret as evaluate_counterfactual_regret,
    fit_counterfactual_regret_certificate as fit_counterfactual_regret_certificate,
    load_claim_bearing_counterfactual_regret_certificate as load_claim_bearing_counterfactual_regret_certificate,
    load_counterfactual_regret_certificate as load_counterfactual_regret_certificate,
    select_counterfactual_regret_candidate as select_counterfactual_regret_candidate,
    write_counterfactual_regret_certificate as write_counterfactual_regret_certificate,
    write_counterfactual_regret_decision as write_counterfactual_regret_decision,
)
from causal4d.latent_contact_v2 import (
    LATENT_CONTACT_V2_SCHEMA_VERSION as LATENT_CONTACT_V2_SCHEMA_VERSION,
    ContactEffectPosteriorV2 as ContactEffectPosteriorV2,
    ContactEndpoint as ContactEndpoint,
    ContactLikelihoodV2Diagnostics as ContactLikelihoodV2Diagnostics,
    ContactObservationEvidenceV2 as ContactObservationEvidenceV2,
    ContactPatchHypothesisSupportV2 as ContactPatchHypothesisSupportV2,
    ContactPatchRolloutBankV2 as ContactPatchRolloutBankV2,
    ContactPatchStateV2 as ContactPatchStateV2,
    ContactV2Selection as ContactV2Selection,
    ContactV2SupportDecision as ContactV2SupportDecision,
    ContactV2SupportPolicy as ContactV2SupportPolicy,
    ContactV2SupportRejectedError as ContactV2SupportRejectedError,
    GraphContactPatchModelV2 as GraphContactPatchModelV2,
    LinearContactObservationGroup as LinearContactObservationGroup,
    SparseContactPatch as SparseContactPatch,
    build_contact_patch_rollout_bank_v2 as build_contact_patch_rollout_bank_v2,
    contact_component_log_likelihoods_v2 as contact_component_log_likelihoods_v2,
    evaluate_contact_v2_support as evaluate_contact_v2_support,
    gaussian_mixture_quantiles as gaussian_mixture_quantiles,
    posterior_weights_from_contact_evidence_v2 as posterior_weights_from_contact_evidence_v2,
    select_contact_v2_candidate as select_contact_v2_candidate,
)
from causal4d.decision_trace import (
    DECISION_TRACE_ENDPOINTS as DECISION_TRACE_ENDPOINTS,
    DECISION_TRACE_PIPELINE as DECISION_TRACE_PIPELINE,
    DECISION_TRACE_SCHEMA_NAME as DECISION_TRACE_SCHEMA_NAME,
    DECISION_TRACE_SCHEMA_VERSION as DECISION_TRACE_SCHEMA_VERSION,
    DECISION_TRACE_STAGE_KINDS as DECISION_TRACE_STAGE_KINDS,
    DecisionTraceArtifact as DecisionTraceArtifact,
    DecisionTraceBuildResult as DecisionTraceBuildResult,
    DecisionTraceDecision as DecisionTraceDecision,
    DecisionTraceSelection as DecisionTraceSelection,
    DecisionTraceStage as DecisionTraceStage,
    UnifiedDecisionTrace as UnifiedDecisionTrace,
    build_unified_decision_trace as build_unified_decision_trace,
    load_claim_bearing_decision_trace as load_claim_bearing_decision_trace,
    load_decision_trace as load_decision_trace,
    require_decision_trace_stack_lock as require_decision_trace_stack_lock,
    write_decision_trace as write_decision_trace,
)
from causal4d.benchmark import (
    CounterfactualBenchmarkConfig as CounterfactualBenchmarkConfig,
    build_protocol as build_protocol,
)
from causal4d.causal_sufficiency import (
    CausalSufficiencyResult as CausalSufficiencyResult,
    assess_command_residual_sufficiency as assess_command_residual_sufficiency,
)
from causal4d.claim_bearing_observation_lineage import (
    load_claim_bearing_prob4d_observation_lineage as load_claim_bearing_prob4d_observation_lineage,
    require_claim_bearing_prob4d_lineage as require_claim_bearing_prob4d_lineage,
)
from causal4d.contact_evaluation import (
    run_latent_contact_benchmark as run_latent_contact_benchmark,
)
from causal4d.contact_inference import (
    LatentContactConfig as LatentContactConfig,
)
from causal4d.contact_traction import (
    graph_traction_field as graph_traction_field,
    integrate_contact_wrench as integrate_contact_wrench,
)
from causal4d.contracts import (
    CounterfactualQuery as CounterfactualQuery,
    FactualIntervention as FactualIntervention,
    PhysicalPosterior as PhysicalPosterior,
    TaskPosterior as TaskPosterior,
    TwinBelief as TwinBelief,
)
from causal4d.counterfactual import (
    apply_counterfactual_operator as apply_counterfactual_operator,
    project_physical_posterior as project_physical_posterior,
)
from causal4d.discrepancy_belief import (
    GraphDiscrepancyBelief as GraphDiscrepancyBelief,
    graph_discrepancy_group_covariances as graph_discrepancy_group_covariances,
    load_graph_discrepancy_belief as load_graph_discrepancy_belief,
    write_graph_discrepancy_belief as write_graph_discrepancy_belief,
)
from causal4d.evaluation import (
    run_counterfactual_benchmark as run_counterfactual_benchmark,
)
from causal4d.factual_abduction_uncertainty import (
    FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION as FACTUAL_ABDUCTION_UNCERTAINTY_SCHEMA_VERSION,
    FactualAbductionUncertaintyV1 as FactualAbductionUncertaintyV1,
    load_factual_abduction_uncertainty_npz as load_factual_abduction_uncertainty_npz,
    save_factual_abduction_uncertainty_npz as save_factual_abduction_uncertainty_npz,
)
from causal4d.finite_query_ambiguity import (
    FiniteQueryAmbiguityConfig as FiniteQueryAmbiguityConfig,
    FiniteQueryAmbiguityResult as FiniteQueryAmbiguityResult,
    assess_finite_query_ambiguity as assess_finite_query_ambiguity,
)
from causal4d.graph_mode_abduction import (
    GraphModeAbductionConfig as GraphModeAbductionConfig,
    abduct_factual_intervention_graph_mode as abduct_factual_intervention_graph_mode,
    graph_mode_joint_weights as graph_mode_joint_weights,
)
from causal4d.grouped_likelihood import (
    GroupLikelihoodDiagnostics as GroupLikelihoodDiagnostics,
    GroupedScoreSemantics as GroupedScoreSemantics,
    grouped_component_log_likelihoods as grouped_component_log_likelihoods,
    posterior_weights_from_grouped_evidence as posterior_weights_from_grouped_evidence,
)
from causal4d.hierarchical_abduction import (
    HierarchicalAbductionResult as HierarchicalAbductionResult,
    abduct_hierarchical_interventions as abduct_hierarchical_interventions,
)
from causal4d.identifiability import (
    IdentifiabilityConfig as IdentifiabilityConfig,
    InterventionIdentifiabilityResult as InterventionIdentifiabilityResult,
    assess_intervention_identifiability as assess_intervention_identifiability,
    finite_response_sensitivity as finite_response_sensitivity,
    project_identifiable_intervention_update as project_identifiable_intervention_update,
)
from causal4d.interventional_contrast import (
    INTERVENTIONAL_CONTRAST_SCHEMA_VERSION as INTERVENTIONAL_CONTRAST_SCHEMA_VERSION,
    ContrastConditionalVariancePolicy as ContrastConditionalVariancePolicy,
    ContrastCouplingPolicy as ContrastCouplingPolicy,
    InterventionalContrastPosteriorV1 as InterventionalContrastPosteriorV1,
    InterventionalContrastQueryV1 as InterventionalContrastQueryV1,
    build_interventional_contrast as build_interventional_contrast,
    load_interventional_contrast as load_interventional_contrast,
    save_interventional_contrast as save_interventional_contrast,
)
from causal4d.joint_observation import (
    JOINT_OBSERVATION_SCHEMA_VERSION as JOINT_OBSERVATION_SCHEMA_VERSION,
    CovarianceRepresentation as CovarianceRepresentation,
    JointGaussianLikelihoodDiagnostics as JointGaussianLikelihoodDiagnostics,
    LinearJointObservationEvidence as LinearJointObservationEvidence,
    block_diagonalize_covariance as block_diagonalize_covariance,
    joint_component_log_likelihoods as joint_component_log_likelihoods,
    posterior_weights_from_joint_observation as posterior_weights_from_joint_observation,
)
from causal4d.observation_evidence import (
    GroupedObservationEvidence as GroupedObservationEvidence,
    ObservationGroup as ObservationGroup,
)
from causal4d.observation_factor_lineage import (
    OBSERVATION_FACTOR_SCHEMA as OBSERVATION_FACTOR_SCHEMA,
    OBSERVATION_FACTOR_SCHEMA_VERSION as OBSERVATION_FACTOR_SCHEMA_VERSION,
    ObservationFactorLineage as ObservationFactorLineage,
    bind_twin_belief_observation_factor_lineage as bind_twin_belief_observation_factor_lineage,
    load_observation_factor_lineage as load_observation_factor_lineage,
    validate_twin_belief_observation_factor_lineage as validate_twin_belief_observation_factor_lineage,
)
from causal4d.partial_identifiability import (
    preserve_prior_within_unidentified_subspace as preserve_prior_within_unidentified_subspace,
)
from causal4d.prefix_likelihood import (
    PrefixLikelihoodConfig as PrefixLikelihoodConfig,
    prefix_component_log_likelihood as prefix_component_log_likelihood,
    update_joint_weights_from_prefix as update_joint_weights_from_prefix,
)
from causal4d.prob4d_joint_observation import (
    PROB4D_JOINT_ADAPTER_SCHEMA_VERSION as PROB4D_JOINT_ADAPTER_SCHEMA_VERSION,
    Prob4DJointObservationDiagnostics as Prob4DJointObservationDiagnostics,
    Prob4DReliabilityPolicy as Prob4DReliabilityPolicy,
    joint_observation_from_prob4d as joint_observation_from_prob4d,
)
from causal4d.prob4d_observation_lineage import (
    validate_claim_bearing_prob4d_observation_metadata as validate_claim_bearing_prob4d_observation_metadata,
    validate_prob4d_causal_observation_metadata as validate_prob4d_causal_observation_metadata,
)
from causal4d.provider_contract import (
    BASE_CAUSAL4D_PROVIDER_CAPABILITIES as BASE_CAUSAL4D_PROVIDER_CAPABILITIES,
    BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS as BAYESIAN_PHYSTWIN_ARTIFACT_SCHEMA_VERSIONS,
    BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE as BAYESIAN_PHYSTWIN_COMPATIBILITY_RANGE,
    BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES as BAYESIAN_PHYSTWIN_PROVIDER_CAPABILITIES,
    PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION as PHYSICAL_BELIEF_PROVIDER_SCHEMA_VERSION,
    PhysicalBeliefProviderManifest as PhysicalBeliefProviderManifest,
    ProviderCompatibilityResult as ProviderCompatibilityResult,
    load_bayesian_phystwin_provider_manifest as load_bayesian_phystwin_provider_manifest,
    require_bayesian_phystwin_provider as require_bayesian_phystwin_provider,
    validate_bayesian_phystwin_provider as validate_bayesian_phystwin_provider,
    validate_provider_compatibility as validate_provider_compatibility,
)
from causal4d.rollout_bank import (
    JointRolloutBank as JointRolloutBank,
    SparseTrajectoryEvidence as SparseTrajectoryEvidence,
)
from causal4d.semantic_freshness import (
    SEMANTIC_TIMING_SCHEMA_VERSION as SEMANTIC_TIMING_SCHEMA_VERSION,
    SEMANTIC_TIMING_SCOPE as SEMANTIC_TIMING_SCOPE,
    SemanticFreshnessDecision as SemanticFreshnessDecision,
    SemanticFreshnessLimits as SemanticFreshnessLimits,
    SemanticTimingMetadata as SemanticTimingMetadata,
    apply_semantic_freshness_gate as apply_semantic_freshness_gate,
)
from causal4d.sensor_evidence import (
    INDEPENDENT_SENSOR_SCHEMA_VERSION as INDEPENDENT_SENSOR_SCHEMA_VERSION,
    ActuatorEvidence as ActuatorEvidence,
    ContactWrenchEvidence as ContactWrenchEvidence,
    load_independent_sensor_evidence as load_independent_sensor_evidence,
    save_independent_sensor_evidence as save_independent_sensor_evidence,
)
from causal4d.sensor_factorized_abduction import (
    IndependentSensorAbductionConfig as IndependentSensorAbductionConfig,
    predict_affine_actuator_realizations as predict_affine_actuator_realizations,
    reweight_factual_intervention_with_independent_sensors as reweight_factual_intervention_with_independent_sensors,
)
from causal4d.stable_discrepancy_dynamics import (
    StableDiscrepancyTransitionModel as StableDiscrepancyTransitionModel,
    forecast_action_conditioned_dynamics as forecast_action_conditioned_dynamics,
)

__version__: str
