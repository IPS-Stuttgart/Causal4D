# Causal4D documentation map

This directory contains conceptual references, integration contracts,
diagnostics, public-data protocols, and retained operator runbooks. The shortest
path for a new software user is:

1. [End-to-end AIP demonstration](aip_end_to_end_demo.md)
2. [Public API and compatibility policy](public_api.md)
3. [Causal model and identification boundary](causal_model_and_identification.md)
4. [Abduction, intervention, and prediction](causal4d_abduction_intervention_prediction.md)
5. [Command-line interface](command_line.md)

Operational document status is declared in the machine-readable
[lifecycle registry](lifecycle_registry.json) and checked in CI. `current`
documents are mutable references, `frozen` documents are byte-locked protocol
records, and `superseded` documents remain available for provenance.

## Concepts and contracts

- [Causal model and identification](causal_model_and_identification.md)
- [Abduction, intervention, and prediction](causal4d_abduction_intervention_prediction.md)
- [Action-conditioned counterfactuals](action_conditioned_counterfactual.md)
- [Action-conditioned discrepancy](action_conditioned_discrepancy.md)
- [Interventional contrasts](interventional_contrast.md)
- [Query-space variance decomposition](query_variance_decomposition.md)
- [Paper scope](causal4d_paper_scope.md)

## Integration

- [BayesianPhysTwin provider contract](bayesian_phystwin_provider.md)
- [BayesianPhysTwin recursive provider contract](bayesian_phystwin_recursive_provider_contract.md)
- [Belief handoff](bpt_belief_handoff.md)
- [Action-bound factual abduction](bound_factual_abduction.md)
- [Camera geometry contract](camera_geometry_contract.md)
- [Migration from BayesianPhysTwin](migration_from_bayesian_phystwin.md)

## Public-data evidence

The first paper is public-data-only. These documents define the active empirical
program:

- [Deform360 `001-rope` held-out-action protocol](causal4d_deform360_public_protocol.md)
- [PokeFlex public readiness and retained source-gate negative](causal4d_pokeflex_public_readiness.md)
- [Paper reproduction](paper_reproduction.md)
- [Controlled collaborator demonstration](controlled_collaborator_demo.md)
- [Core causal and numerical invariant certificate](core_invariant_certificate.md)

Positive and negative public-data results are both evidence. A rejected source
backend must remain rejected and does not create an obligation to collect a new
physical dataset.

## Reproduction and diagnostics

- [Target-free real design sensitivity](causal4d_real_design_sensitivity.md)
- [Query-space variance decomposition](query_variance_decomposition.md)
- [Automation integrity](automation_integrity.md)
- [Branch hygiene](branch_hygiene.md)
- [Exact-head validation](exact_head_validation.md)

## Optional future physical validation

The files in this section preserve the previously registered 18-session,
36-execution hardware protocol. They are **not required for the public-data-only
first paper**. Their `0/36` state is retained as provenance, not as a missing
submission result.

### Retained operator references

- [Optional physical-validation protocol](causal4d_real_experiment_milestone.md)
- [Source-panel acquisition](causal4d_source_panel_acquisition.md)
- [Pre-acquisition readiness](causal4d_preacquisition_readiness.md)
- [Real-evidence status and accounting](causal4d_real_evidence_status.md)
- [Acquisition environment capsule](causal4d_acquisition_environment_capsule.md)
- [Acquisition flight recorder](causal4d_acquisition_flight_recorder.md)
- [Pre-acquisition next-action derivation](causal4d_preacquisition_next_action.md)
- [Next-action packet](causal4d_preacquisition_next_action_packet.md)
- [Next-action validation](causal4d_preacquisition_next_action_validation.md)
- [Fresh-reset mode-0 cross-check](causal4d_reset_mode0_pilot.md)
- [Self-hosted pre-acquisition automation](self_hosted_preacquisition_automation.md)

### Frozen amendment lineage

- [Pre-acquisition governance amendment v5](causal4d_preacquisition_v5.md)
  permits disclosed single-operator self-attestation and makes no independent
  attestation claim.
- [Pre-acquisition amendment v4](causal4d_preacquisition_v4.md) is superseded by
  v5 and retained unchanged for provenance.
- [Pre-acquisition amendment v3](causal4d_preacquisition_v3.md) is superseded by
  v4 and retained unchanged for provenance.
- [Pre-acquisition amendment v2](causal4d_preacquisition_v2.md) is superseded by
  v3 and retained unchanged for provenance.

If a future collaborator elects to run the hardware study, the lifecycle
validator and exact frozen protocol still apply. No such execution is needed to
submit or interpret the public-data paper.

## Governance and scope

- [Paper scope](causal4d_paper_scope.md)
- [Licensing](licensing.md)
- [Automation integrity](automation_integrity.md)
- [Branch hygiene](branch_hygiene.md)

The documentation distinguishes controlled evidence, public-data held-out
evidence, released diagnostics, and optional physical acquisition. A software
workflow never creates physical evidence, and absence of optional physical
evidence does not invalidate the bounded public-data claims.
