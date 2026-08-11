# Structured covariance in grouped likelihoods

`causal4d.grouped_likelihood` accepts component-specific correlated uncertainty
in two equivalent forms:

```text
dense covariance: C
low-rank factor:  U, with C = U U^T
```

The existing dense `component_group_covariance_m2` path remains unchanged. The
optional `component_group_covariance_factor_m` path is intended for shared gauge,
graph-discrepancy, camera-bias, or other low-rank uncertainty where forming one
full covariance matrix for every posterior component would be wasteful. The
factor API is additive and opt-in; no producer is silently converted from dense
to structured covariance.

## Numerical method

For a base covariance `B` and factor `U`, the Student-t likelihood needs the
log determinant and quadratic form of:

```text
B + U U^T
```

The structured path computes these quantities with:

1. a Cholesky factorization of `B`;
2. whitening of the residual and `U`;
3. the matrix determinant lemma for the log determinant; and
4. the Woodbury identity for the Mahalanobis term.

Only the rank-by-rank system `I + U^T B^-1 U` is factorized after the base
Cholesky. The implementation does not form `U U^T`, does not use a dense
inverse, and does not call the dense `slogdet` path.

## API

For one observation group:

```python
score, responsibility = group_log_likelihood(
    predicted_values_m,
    group,
    additive_variance_m2=component_diagonal_variance,
    additive_covariance_factor_m=component_factor_m,
)
```

For a complete grouped update:

```python
posterior, diagnostics = posterior_weights_from_grouped_evidence(
    prior_weights,
    predicted_components_m,
    evidence,
    prefix_frame_count=prefix_frame_count,
    component_group_covariance_factor_m={
        "graph-discrepancy": graph_factor_m,
        "shared-gauge": gauge_factor_m,
    },
)
```

Each factor has units of meters and must broadcast to:

```text
component_shape + (group_coordinate_count, rank)
```

The resulting covariance contribution is therefore measured in square meters.
The dense covariance, diagonal variance, and low-rank factor arguments may be
combined only when they represent distinct uncertainty sources. Supplying the
same uncertainty through more than one representation would double count it.

`GroupLikelihoodDiagnostics.low_rank_covariance_group_ids` records the groups
that used the structured path. `full_covariance_group_ids` continues to identify
groups with explicit dense component covariance.

## Compatibility and claim boundary

The default path is byte-for-byte API compatible: callers that do not provide a
factor use the original dense Student-t implementation. Existing frozen
experiments and result identities are unchanged.

Tests compare structured and explicitly materialized dense covariance updates
for mixture scores, nominal responsibilities, and normalized posterior weights.
They also verify broadcasting, fail-closed factor validation, diagnostics, and
that the structured path does not call dense determinant evaluation. These are
numerical and engineering guarantees, not new empirical accuracy or calibration
evidence.

## Coordinate-normalized development score

The structured covariance representation is also supported by the opt-in
[`normalized_coordinate_mean_v3`](grouped_normalized_likelihood_v3.md) grouped
score. That score preserves the dense/low-rank equivalence described here while
adding contributor-capped coordinate normalization, an explicit likelihood
power, conditioning diagnostics, and a fail-closed source-covariance threshold.
`legacy_sum_v1` remains the default and the registered physical estimator is
unchanged.
