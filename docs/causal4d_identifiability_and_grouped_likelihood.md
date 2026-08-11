# Intervention Identifiability and Grouped Observation Likelihood

## Status

This implementation adds an opt-in, backward-compatible inference path. The
frozen `v0.3.0-causal4d-aip` dense likelihood remains unchanged unless grouped
evidence or an identifiability result is explicitly supplied.

## Motivation

A response prefix can often be explained by several observationally similar
causes:

- realized gain, delay, controller-frame bias, contact, or slip;
- physical-state or parameter error;
- coherent camera or gauge bias;
- unresolved readout discrepancy.

Explaining the prefix is therefore insufficient. Causal4D should update the
realized-intervention posterior only when the intervention response remains
informative after nuisance response is removed.

## Grouped evidence

`ObservationGroup` selects a vector of scalar rollout coordinates and supplies:

- the complete metric covariance of that vector;
- a residual-independent prior nominal probability;
- a broad-component covariance multiplier;
- a Student-t degrees-of-freedom value;
- a frozen composite-likelihood weight;
- source, view, and contributor provenance.

The nominal and broad components use the same covariance orientation. If
`C_g` is the declared covariance and `nu > 2`, the Student-t scale is

```text
Psi_g = (nu - 2) / nu * C_g.
```

The broad component uses `lambda_out * Psi_g`. The residual changes the
posterior nominal responsibility but never changes the prior nominal
probability.

`GroupedObservationEvidence` caps repeated contributors automatically. If one
contributor is used in `m` groups, each affected group receives at most `1/m`
of its declared composite power. Duplicating an observation while retaining
its contributor identity therefore leaves the total evidence power unchanged.
Feeders should still construct scientifically meaningful groups instead of
relying on this cap as a substitute for covariance modeling.

`GroupedObservationEvidence.from_dense_prefix` is a convenience adapter that
creates one group per permitted O-plus frame with diagonal covariance. It is
mainly intended for controlled benchmarks and migration tests. Real feeders
should export full covariance, contributor identities, and view/source
provenance directly.

## Conditional identifiability

Let `J_z` contain whitened finite-response sensitivities for the proposed
intervention variables and let `J_n` contain whitened nuisance sensitivities.
The implementation projects intervention response onto the orthogonal
complement of the nuisance span:

```text
J_z|n = (I - P_n) J_z
I_z|n = J_z|n.T J_z|n.
```

`assess_intervention_identifiability` reports:

- the eigenvalues and effective rank of `I_z|n`;
- its minimum eigenvalue and condition number;
- the fraction of intervention-response energy remaining after nuisance
  projection;
- the largest cosine between the intervention and nuisance subspaces;
- explicit failure reasons.

Thresholds are supplied through `IdentifiabilityConfig` and must be frozen on
source or controlled data. They are not estimated from the target future.

`finite_response_sensitivity` converts predeclared simulator perturbations into
secant columns. It does not claim that the resulting local response is a global
transition model.

## Guarded factual abduction

`abduct_factual_intervention` now accepts three optional arguments:

```python
factual = abduct_factual_intervention(
    bank,
    belief,
    observations,
    prefix_frame_count=7,
    grouped_evidence=evidence,
    identifiability=diagnostic,
    abstain_when_unidentifiable=True,
)
```

When the diagnostic fails and guarded abduction is enabled, the returned
`FactualIntervention` uses the exact original joint prior over physical
particles and intervention hypotheses. The artifact records the diagnostic and
failure reasons. No target-future frame is read.

When grouped evidence is supplied, Causal4D scores the particle-specific
readout trajectories with the robust grouped likelihood and adds each
particle's discrepancy variance to the selected coordinate covariance.
The physical simulator trajectories remain immutable.

## Command-line use

The factual-abduction CLI can construct frame-grouped evidence directly from
the permitted object tracks:

```bash
causal4d experiment phystwin abduct-intervention \
  known.bank.npz belief.npz CASE/final_data.pkl \
  factual.npz factual_eval.json \
  --grouped-observation-likelihood \
  --prior-nominal-probability 0.95 \
  --outlier-scale-multiplier 100
```

A source-frozen identifiability artifact is a non-pickled NPZ containing
`intervention_sensitivity` and optional `nuisance_sensitivity` and `covariance`
arrays. Guarded use is:

```bash
causal4d experiment phystwin abduct-intervention \
  known.bank.npz belief.npz CASE/final_data.pkl \
  factual.npz factual_eval.json \
  --grouped-observation-likelihood \
  --identifiability-npz source_identifiability.npz \
  --abstain-when-unidentifiable
```

The sensitivity artifact must be fitted or generated without the target
continuation. Threshold flags are explicit CLI inputs so a protocol can freeze
them before target outcomes are opened.

## Required protocol usage

1. Build intervention sensitivities from source-only or controlled finite
   responses.
2. Build nuisance sensitivities for the bias, gauge, state, and discrepancy
   modes that are admitted by the experiment.
3. Freeze identifiability thresholds before opening a target continuation.
4. Construct grouped O-plus evidence using only the declared response prefix.
5. Abduce with exact-prior abstention enabled.
6. Report abstention rate, conditional-information diagnostics, posterior
   entropy reduction, factual continuation, and held-out interventional
   prediction.

A passed local diagnostic is necessary but not sufficient for a paper claim.
The same-object multi-action protocol and independent-execution calibration
remain the required real-evidence gates.

## Prospective registered-query and covariance bridge

For an opt-in query-specific gate and provenance-bound full or low-rank
conditional covariance in factual abduction, see
[`registered_query_abduction_and_structured_uncertainty.md`](
registered_query_abduction_and_structured_uncertainty.md
).
This extension is prospective and does not change the registered physical-study
path or its default artifact identity.
