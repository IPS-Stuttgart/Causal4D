# Causal4D documentation map

This directory contains conceptual references, integration contracts,
diagnostics, registered protocols, and operator runbooks. The shortest path for
a new software user is:

1. [End-to-end AIP demonstration](aip_end_to_end_demo.md)
2. [Public API and compatibility policy](public_api.md)
3. [Causal model and identification boundary](causal_model_and_identification.md)
4. [Abduction, intervention, and prediction](causal4d_abduction_intervention_prediction.md)
5. [Command-line interface](command_line.md)

## Concepts and contracts

- [Causal model and identification](causal_model_and_identification.md)
- [Abduction, intervention, and prediction](causal4d_abduction_intervention_prediction.md)
- [Action-conditioned counterfactuals](action_conditioned_counterfactual.md)
- [Action-conditioned discrepancy](action_conditioned_discrepancy.md)
- [Interventional contrasts](interventional_contrast.md)
- [Query-space variance decomposition](query_variance_decomposition.md)

## Integration

- [BayesianPhysTwin provider contract](bayesian_phystwin_provider.md)
- [BayesianPhysTwin recursive provider contract](bayesian_phystwin_recursive_provider_contract.md)
- [Belief handoff](bpt_belief_handoff.md)
- [Camera geometry contract](camera_geometry_contract.md)
- [Migration from BayesianPhysTwin](migration_from_bayesian_phystwin.md)

## Reproduction and diagnostics

- [Paper reproduction](paper_reproduction.md)
- [Controlled collaborator demonstration](controlled_collaborator_demo.md)
- [Automation integrity](automation_integrity.md)
- [Branch hygiene](branch_hygiene.md)
- [Exact-head validation](exact_head_validation.md)

## Registered physical experiment

The files in this section are operational and evidence-bearing. Use the exact
version associated with the frozen experiment rather than copying commands into
another document.

- [Physical-experiment milestone](causal4d_real_experiment_milestone.md)
- [Source-panel acquisition](causal4d_source_panel_acquisition.md)
- [Pre-acquisition readiness](causal4d_preacquisition_readiness.md)
- [Real-evidence status and accounting](causal4d_real_evidence_status.md)
- [Acquisition environment capsule](causal4d_acquisition_environment_capsule.md)
- [Acquisition flight recorder](causal4d_acquisition_flight_recorder.md)

## Governance and scope

- [Paper scope](causal4d_paper_scope.md)
- [Licensing](licensing.md)
- [Automation integrity](automation_integrity.md)
- [Branch hygiene](branch_hygiene.md)

The documentation distinguishes controlled software evidence, diagnostic public
data, and registered physical evidence. A successful software demonstration or
workflow run never increments the physical evidence count unless the registered
protocol explicitly admits it.
