# Causal4D documentation map

This directory contains conceptual references, integration contracts,
diagnostics, registered protocols, and operator runbooks. The shortest path for
a new software user is:

1. [End-to-end AIP demonstration](aip_end_to_end_demo.md)
2. [Public API and compatibility policy](public_api.md)
3. [Causal model and identification boundary](causal_model_and_identification.md)
4. [Abduction, intervention, and prediction](causal4d_abduction_intervention_prediction.md)
5. [Command-line interface](command_line.md)

Operational document status is declared in the machine-readable
[lifecycle registry](lifecycle_registry.json) and checked in CI. `current`
documents are the mutable operator references, `frozen` documents are
byte-locked protocol records, and `superseded` documents remain available only
for provenance and identify their successor. Do not infer operational status
from a filename or search result alone.

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
- [Action-bound factual abduction](bound_factual_abduction.md)
- [Camera geometry contract](camera_geometry_contract.md)
- [Migration from BayesianPhysTwin](migration_from_bayesian_phystwin.md)

## Reproduction and diagnostics

- [Paper reproduction](paper_reproduction.md)
- [Target-free real design sensitivity](causal4d_real_design_sensitivity.md)
- [Core causal and numerical invariant certificate](core_invariant_certificate.md)
- [Controlled collaborator demonstration](controlled_collaborator_demo.md)
- [Automation integrity](automation_integrity.md)
- [Branch hygiene](branch_hygiene.md)
- [Exact-head validation](exact_head_validation.md)

## Registered physical experiment

The files in this section are operational and evidence-bearing. Use the exact
version associated with the frozen experiment rather than copying commands into
another document.

### Active operator references

- [Physical-experiment milestone](causal4d_real_experiment_milestone.md)
- [Source-panel acquisition](causal4d_source_panel_acquisition.md)
- [Pre-acquisition readiness](causal4d_preacquisition_readiness.md)
- [Real-evidence status and accounting](causal4d_real_evidence_status.md)
- [Acquisition environment capsule](causal4d_acquisition_environment_capsule.md)
- [Acquisition flight recorder](causal4d_acquisition_flight_recorder.md)
- [Pre-acquisition next-action derivation](causal4d_preacquisition_next_action.md)
- [Next-action packet](causal4d_preacquisition_next_action_packet.md)
- [Next-action validation](causal4d_preacquisition_next_action_validation.md)
- [Self-hosted pre-acquisition automation](self_hosted_preacquisition_automation.md)

### Frozen amendment lineage

- [Pre-acquisition governance amendment v5](causal4d_preacquisition_v5.md)
  is the active, byte-locked amendment. It permits disclosed single-operator
  self-attestation and makes no independent-attestation claim.
- [Pre-acquisition amendment v4](causal4d_preacquisition_v4.md) is superseded by
  v5 and retained unchanged for provenance.
- [Pre-acquisition amendment v3](causal4d_preacquisition_v3.md) is superseded by
  v4 and retained unchanged for provenance.
- [Pre-acquisition amendment v2](causal4d_preacquisition_v2.md) is superseded by
  v3 and retained unchanged for provenance.

The lifecycle validator requires every active operational role to be unique and
linked here. It also verifies the Git blob identities of frozen and superseded
amendments and rejects an unregistered future `causal4d_preacquisition_v*.md`
document.

## Governance and scope

- [Paper scope](causal4d_paper_scope.md)
- [Licensing](licensing.md)
- [Automation integrity](automation_integrity.md)
- [Branch hygiene](branch_hygiene.md)

The documentation distinguishes controlled software evidence, diagnostic public
data, and registered physical evidence. A successful software demonstration or
workflow run never increments the physical evidence count unless the registered
protocol explicitly admits it.
