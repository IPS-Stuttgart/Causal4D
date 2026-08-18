# Guarded BayesianPhysTwin belief handoff v2

`causal4d.guarded_bpt_belief_handoff_v2` is the claim-bearing Causal4D consumer for the exact runtime and complete-belief selection contracts introduced by BayesianPhysTwin PR #703.

The v2 handoff separates four identities that must not be conflated:

1. the admitted Prob4D/BayesianPhysTwin inference update;
2. the exact independently observed Prob4D runtime commit;
3. the complete BayesianPhysTwin candidate construction and guard decision; and
4. the complete BayesianPhysTwin belief actually selected after guarding.

An inference-admissible update is not sufficient for Causal4D evidence admission. The frozen complete-belief guard may still reject it. In that case the v2 path requires the selected BayesianPhysTwin belief to equal the construction receipt's baseline belief, requires the Causal4D candidate to retain the exact baseline artifact, rejects query covariance, and appends no evidence-consumption entry.

When the guard accepts the candidate, the handoff requires a registered query covariance tied to the same update and tree-block result. The Causal4D evidence ledger records the exact Prob4D Git commit from `Prob4DRuntimeIdentityV1`; it never records the revision-evidence source label such as `installed_vcs_metadata` as though it were a revision.

The returned `BayesianPhysTwinGuardedBeliefHandoffReceiptV2` binds:

- update, admission, result, observation, and linearization identities;
- provider manifest and exact runtime identity;
- candidate-construction, guard-certificate, guard-decision, and selection identities;
- baseline, candidate, and selected BayesianPhysTwin belief identities;
- baseline and delivered Causal4D belief identities;
- query covariance and evidence-ledger identities; and
- accepted-candidate versus exact-fallback behavior.

This is an additive consumer. The historical v1 handoff remains available for frozen compatibility, but new claim-bearing cross-repository studies should use v2 after BayesianPhysTwin PR #703 is merged and released. A passing contract test establishes lineage and fail-closed behavior only; it does not establish provider competence, covariance calibration, physical-query benefit, intervention benefit, deployment safety, or state of the art.
