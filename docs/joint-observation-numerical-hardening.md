# Joint-observation numerical hardening

The full-joint Gaussian and grouped Student-t likelihoods use exact low-rank
covariance updates through Cholesky whitening, the matrix determinant lemma, and
the Woodbury identity.

## Fail-closed Woodbury subtraction

The corrected Mahalanobis term is analytically nonnegative. Earlier
implementations clipped every negative floating-point result to zero. That is
appropriate only for roundoff-scale cancellation; a larger negative value can
indicate an unstable factorization or an invalid numerical state.

`causal4d.low_rank_numerics.nonnegative_woodbury_quadratic` now applies one shared
rule to dense, block-diagonal, prepared, component-factor, and grouped Student-t
paths. It clips only a dimension- and scale-aware roundoff interval and raises
`FloatingPointError` beyond it. No covariance, likelihood, posterior-support, or
evidence schema changes.

## Grouped selector covariance propagation

A diagonal trajectory variance passed through a sparse linear operator induces
covariance only among output rows that reuse the same selected trajectory scalar.
The compatibility implementation previously compared every sparse term with
every other term.

Terms are now grouped in one pass by `(frame, node, coordinate)` and combined by
output row before the required small outer products are formed. For `K` sparse
terms and selector groups `g`, work changes from `O(K^2)` selector comparisons to
approximately

```text
O(K + sum_g |rows_g|^2).
```

Duplicate terms targeting the same output row are summed before advanced
indexing, preserving exact covariance accumulation. The block-diagonal path
retains its fail-closed rejection when one selected scalar would induce
covariance across different declared blocks.

Focused tests compare both compatibility paths with an explicitly materialized
sparse operator, compare them with the prepared operator, cover duplicate-row
aggregation, and verify every low-rank likelihood route uses the shared numerical
guard.

## Validation boundary

The focused suite exercises every modified numerical route and is followed by the
complete default test suite. Permanent pull-request gates additionally cover the
supported Python matrix, declared dependency floors, packaging, security scans,
and the pinned BayesianPhysTwin installed-wheel boundary. These checks establish
implementation and compatibility preservation; they do not create physical
evidence or alter the registered real-experiment method.
