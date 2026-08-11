# Registered-query abduction and structured uncertainty

This document describes a **prospective** Causal4D path. It does not alter the
registered physical-study estimator, its artifact identity, its source/target
boundary, or the 36-execution protocol. Calls that omit the options below retain
the historical full-parameter identifiability gate and the existing diagonal
TwinBelief discrepancy variance exactly.

## Query-specific identifiability admission

The identifiability diagnostic already separates two questions:

1. Is the complete intervention parameter vector locally identifiable after
   nuisance projection?
2. Does the registered future query depend materially on unresolved intervention
   directions?

The abduction API and PhysTwin CLI now expose that distinction explicitly:

- `full_parameter` is the default and preserves the historical behavior.
- `registered_query` is opt-in and requires `query_sensitivity` in a source-only
  identifiability NPZ.

With `--abstain-when-unidentifiable`, a rejected policy returns the exact original
joint prior over intervention hypotheses and physical particles. It does not
renormalize a subset, inject an approximate floor, or create support that was not
present in the prior.

The identifiability NPZ accepts:

- required `intervention_sensitivity`;
- optional `nuisance_sensitivity`;
- optional positive diagonal or dense `covariance`;
- optional `covariance_factor`, interpreted as the exact positive-semidefinite
  update `U @ U.T`;
- optional positive `parameter_scales`; and
- optional `query_sensitivity` with shape `(query_coordinate, parameter)`.

Example:

```bash
causal4d experiment phystwin abduct-intervention \
  rollout_bank.npz twin_belief.npz final_data.pkl \
  factual_intervention.npz evaluation.json \
  --grouped-observation-likelihood \
  --identifiability-npz source_registered_identifiability.npz \
  --identifiability-policy registered_query \
  --maximum-query-null-response-fraction 0.10 \
  --abstain-when-unidentifiable
```

The query matrix, horizon, output semantics, units, and validity mask must be
registered before target-future access. The CLI does not infer or optimize a
query from target outcomes.

## FactualAbductionUncertaintyV1

`FactualAbductionUncertaintyV1` carries additional covariance through the
operational grouped-abduction path. The artifact is content-addressed and binds
to exactly one:

- `JointRolloutBank.artifact_id`;
- `TwinBelief.artifact_id`; and
- `GroupedObservationEvidence.evidence_id`.

It supports:

- additional independent component variance, broadcastable to
  `(hypothesis, particle, frame, node, coordinate)`;
- full covariance for selected observation groups; or
- low-rank factors in meters, evaluated as `U @ U.T` without materializing the
  dense update.

A group cannot use full and low-rank representations simultaneously. Combining
an independent term with correlated terms requires an explicit declaration that
they represent disjoint sources. Every artifact must also attest that it is
source-only, disjoint from the uncertainty already stored in `TwinBelief`, and
disjoint from the covariance already stored in the grouped observation evidence.
Mismatched bindings, unknown groups, invalid shapes, non-finite arrays,
non-positive-semidefinite covariance, missing declarations, or a changed content
hash fail closed before posterior scoring.

### NPZ interchange

Use `save_factual_abduction_uncertainty_npz` and
`load_factual_abduction_uncertainty_npz`; loading verifies the declared artifact
ID against all arrays and metadata. The interchange contains scalar binding and
provenance fields, an optional `additional_independent_variance_m2`, ordered
`dense_group_ids` and `factor_group_ids`, and one numbered array per group.
Pickle loading is disabled.

The CLI accepts the verified artifact only with grouped evidence:

```bash
causal4d experiment phystwin abduct-intervention \
  rollout_bank.npz twin_belief.npz final_data.pkl \
  factual_intervention.npz evaluation.json \
  --grouped-observation-likelihood \
  --factual-abduction-uncertainty-npz source_only_uncertainty.npz
```

The same uncertainty is used for the Causal4D intervention posterior and the
same-evidence nominal-contact comparator. The resulting factual artifact records
the uncertainty content address and whether full or low-rank covariance was used
for each group.

## Required prospective evaluation

Promotion requires fresh object/session groups disjoint from fitting and
calibration data. At minimum, report:

- point-error noninferiority against the unchanged mean comparator;
- Gaussian log score or energy score;
- nominal and worst-group coverage;
- interval width;
- the frequency and consequences of exact fallback; and
- separate ablations for diagonal, full, and low-rank uncertainty.

These results must remain separate from the frozen primary physical study. A
negative or inconclusive prospective result is reportable and must not trigger
post-target retuning of the registered query or admission threshold.
