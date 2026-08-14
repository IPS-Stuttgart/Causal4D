# Explicit cross-branch covariance for interventional contrasts

## Scope

`causal4d.interventional_cross_covariance` is an additive analysis layer for an
existing `InterventionalContrastPosteriorV1` whose conditional-variance policy is
`independent_readout`. It leaves both physical branch posteriors and the historical
contrast estimator unchanged.

The historical policy uses

```text
Cov(Q(R^a) - Q(R^b) | pair) = C_a + C_b.
```

A separately registered source-only or controlled-data model may instead supply
one query-space cross covariance `C_ab` for every coupled component pair. The
adjusted conditional contrast covariance is

```text
C_delta = C_a + C_b - C_ab - C_ab^T.
```

Every declared joint block must be positive semidefinite:

```text
[ C_a     C_ab  ]
[ C_ab^T  C_b   ] >= 0.
```

The implementation fails closed when a joint block or the resulting contrast
covariance is invalid.

## Exact fallback

Omitting `cross_branch_conditional_covariance` returns the exact source contrast
object, including object identity and artifact identity. Supplying a covariance
requires a lowercase SHA-256 `cross_covariance_model_id` and exact source branch
identities.

```python
from causal4d.interventional_contrast import (
    build_interventional_cross_covariance,
)

adjusted = build_interventional_cross_covariance(
    source_contrast,
    branch_a,
    branch_b,
    cross_branch_conditional_covariance=source_frozen_cross_covariance,
    cross_covariance_model_id=source_frozen_model_id,
)
```

The returned `InterventionalCrossCovarianceV1` binds the source contrast, model,
pair support, branch covariances, cross covariance, adjusted covariance, query
labels and units, and finite metadata into one content identity. It exposes the
same finite-mixture mean, covariance, positive-effect probability, and marginal
quantile summaries as the source contrast.

## Relationship to correlation sensitivity

The existing readout-correlation sensitivity artifact varies scalar marginal
correlations over a declared grid. This module is complementary: it consumes one
complete pair-specific, potentially cross-output covariance tensor whose identity
and source are already frozen. It does not estimate a covariance or select one
from target outcomes.

## Scientific boundary

This is query-space uncertainty analysis, not a new physical estimator. It does
not establish empirical calibration, identify a physical discrepancy mechanism,
provide individual-level real counterfactual ground truth, or change the frozen
36-execution protocol. A nonzero covariance must be fitted or preregistered using
source or controlled data before confirmatory target access.
