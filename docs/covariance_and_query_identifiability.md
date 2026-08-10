# Correlation-Preserving Abduction and Query-Aware Identifiability

## Scope

This development path is opt-in and does not modify the frozen
`v0.3.0-causal4d-aip` milestone. It addresses two limitations of the current
finite-bank inference:

1. low-rank discrepancy, shared camera bias, and gauge uncertainty can be
   correlated across nodes, coordinates, and frames, whereas a diagonal variance
   loses that dependence;
2. complete recovery of every intervention parameter is stronger than the actual
   goal of identifying a particular held-out interventional prediction.

The legacy diagonal grouped likelihood and binary identifiability decision remain
unchanged when the new arguments are omitted.

## Full grouped component covariance

`posterior_weights_from_grouped_evidence` and
`grouped_component_log_likelihoods` accept
`component_group_covariance_m2`, a mapping from observation-group ID to a
component-specific covariance. Each value must be broadcastable to

```text
component_shape + (group_coordinate_count, group_coordinate_count).
```

The covariance is added to the observation covariance and to any legacy
component-wise diagonal variance before the robust Student-t mixture is scored.
Every supplied matrix must be finite, symmetric, and positive semidefinite.
Unknown group IDs fail closed.

`graph_discrepancy_group_covariances` maps a `GraphDiscrepancyBelief` and its
hash-locked graph basis into those grouped covariance matrices. The graph
coefficient state is treated as persistent over the scored prefix, so the same
mode observed at different frames remains correlated. The current belief stores
separate coefficient covariance per Cartesian coordinate; cross-coordinate
terms are therefore zero, while the declared projection variance remains a
coordinate-wise diagonal remainder.

Example:

```python
covariance_by_group = graph_discrepancy_group_covariances(
    discrepancy_belief,
    graph_basis,
    grouped_evidence,
    component_ids=particle_ids,
)
posterior, diagnostics = posterior_weights_from_grouped_evidence(
    prior_weights,
    predicted_components,
    grouped_evidence,
    prefix_frame_count=prefix_frame_count,
    component_group_covariance_m2=covariance_by_group,
)
```

A covariance with leading shape `(particle, d, d)` broadcasts over a rollout
bank with leading shape `(hypothesis, particle)`. This lets one discrepancy
belief per physical particle be shared across its intervention hypotheses.

## Structured covariance whitening

`assess_intervention_identifiability` accepts either a dense positive-definite
response covariance or a positive diagonal variance vector. The optional
`covariance_factor` adds a shared positive-semidefinite term:

```text
Sigma = B + U U.T
```

where `B` is the dense or diagonal value supplied through `covariance`, and `U`
is the response-by-rank `covariance_factor`. For a diagonal base, the diagnostic
uses storage proportional to the response count times the low rank rather than
forming a response-by-response covariance.

The implementation whitens by the base covariance and then applies the exact
inverse square root of the low-rank update through a thin singular-value
decomposition. Consequently, the conditional information, nuisance projection,
subspace angles, and query gate agree with an explicitly materialized dense
`B + U U.T` covariance up to floating-point precision.

```python
result = assess_intervention_identifiability(
    intervention_sensitivity,
    nuisance_sensitivity,
    covariance=response_variance,
    covariance_factor=shared_camera_and_gauge_factor,
    parameter_scales=parameter_scales,
    query_sensitivity=query_sensitivity,
)
```

The factor requires an explicit positive-definite base covariance. Nonpositive
diagonal entries, nonfinite factors, mismatched response dimensions, and empty
low-rank factors fail closed. This is an information-geometry diagnostic; richer
covariance does not by itself establish calibration and should remain source-fit
or preregistered before confirmatory evaluation.

## Standardized partial identifiability

`assess_intervention_identifiability` accepts positive
`parameter_scales`. For physical intervention coordinates `z` and standardized
coordinates `eta`, use

```text
z - z0 = diag(parameter_scales) eta.
```

The conditional information matrix and its identified/null bases are computed in
`eta` coordinates. Equivalent unit changes therefore give the same diagnostic
when the sensitivity columns and scales are transformed consistently.

The returned `identified_basis` and `null_basis` provide an explicit orthogonal
decomposition. `project_identifiable_intervention_update` converts a proposed
physical-unit update to standardized coordinates, removes its unresolved
component, and converts it back.

## Query-aware gate

An optional `query_sensitivity` describes the local response of the requested
future prediction to the intervention variables. The diagnostic reports

```text
query_null_response_fraction
```

as the fraction of squared query sensitivity lying in the unresolved
intervention subspace. `query_identifiable` passes when that fraction does not
exceed the preregistered
`maximum_query_null_response_fraction`.

This deliberately separates two statements:

- `identifiable`: every declared intervention coordinate is locally identified;
- `query_identifiable`: the requested prediction is insensitive enough to the
  unidentified combinations.

Thus a held-out prediction may remain admissible even when gain, delay, contact,
or frame-bias components cannot all be individually recovered. Conversely, a
query that amplifies an unresolved direction must widen uncertainty or abstain.
Neither local result establishes global recovery or held-out calibration.
