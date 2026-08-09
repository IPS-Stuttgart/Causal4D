# Prepared full-joint observation inference

`causal4d.prepared_joint_observation` is an additive execution path for repeated
full-joint updates over finite Causal4D support. The historical
`causal4d.joint_observation` API remains unchanged.

## Motivation

The compatibility path now shares one base factorization across all rollout
components inside a single update. Repeated updates with the same static Prob4D
evidence still rebuild the sparse selections and covariance preparation on each
call. Independent trajectory variance propagation also retains the historical
quadratic term-pair scan. In addition, an additive dense covariance term must be
positive definite by itself, although a covariance contribution may legitimately
be singular and positive semidefinite.

The prepared path addresses those remaining boundaries without changing the
likelihood:

- compile duplicate sparse trajectory selectors into one deterministic sparse
  linear operator;
- retain the exact static dense or block-diagonal solver across repeated calls;
- retain its pre-whitened evidence-level low-rank factor;
- score components in an explicit bounded-memory chunk size;
- permit finite symmetric positive-semidefinite additive covariance terms;
- require the final total covariance to remain positive definite; and
- preserve exact zero prior support.

## Usage

```python
from causal4d.prepared_joint_observation import (
    posterior_weights_from_prepared_joint_observation,
    prepare_joint_observation,
)

prepared = prepare_joint_observation(evidence)
posterior, diagnostics = posterior_weights_from_prepared_joint_observation(
    prior_weights,
    predicted_components_m,
    prepared,
    prefix_frame_count=prefix_frame_count,
    component_chunk_size=32,
    maximum_working_bytes=256 * 1024**2,
)
```

Preparation validates the existing `LinearJointObservationEvidence` object and
stores its exact object identity. It does not reinterpret Prob4D reliability,
association, factor-group, causal-prefix, frame-mapping, or entity-mapping
semantics.

## Positive-semidefinite additive covariance

The base covariance remains strictly positive definite. An additive component
covariance is instead checked for:

- finite entries;
- symmetry; and
- nonnegative eigenvalues up to a scale-aware numerical tolerance.

Rank-deficient covariance is therefore accepted. Negative directions are
rejected even when the positive-definite base could numerically mask them. The
sum is then processed by the existing Cholesky-based likelihood, so the final
total covariance must still be positive definite.

Low-rank positive-semidefinite terms should normally remain in factor form. The
dense or block additive form is useful when an upstream component already owns
a covariance representation and its rank is not known cheaply.

## Memory boundary

The scorer estimates component-local working memory before the first chunk is
allocated. If one component cannot fit below `maximum_working_bytes`, it raises
`MemoryError` before constructing the dynamic covariance. Otherwise it selects
the minimum of:

- the number of components;
- the caller's optional `component_chunk_size`; and
- the budget-derived chunk size.

Diagnostics retain the selected chunk size, number of chunks, maximum budget,
estimated peak working bytes, sparse selector count, operator nonzero count, and
whether the static base factorization was reused.

The estimate is an execution guard, not a universal process-RSS bound. NumPy,
BLAS, SciPy, and the Python allocator may retain additional implementation
memory.

## Numerical parity

Regression coverage compares the prepared and historical paths for:

- dense and block-diagonal bases;
- evidence-level and component-level low-rank factors;
- chunked and unchunked scoring;
- posterior normalization and exact zero support;
- duplicate selector means and propagated covariance; and
- direct full-covariance calculations with singular additive covariance.

A dedicated self-hosted workflow exercises the installed wheel on `workstation2`
with a 2,048-row, rank-7 structured observation and publishes numerical parity,
chunking, and timing diagnostics. It is triggered only by an exact
registered-maintainer issue on reviewed `main`; the self-hosted job has
read-only repository permission and never consumes issue text as executable
input.

## Scientific boundary

This module changes execution efficiency and the admissible representation of a
mathematically valid covariance contribution. It does not change the registered
18-session/36-execution estimator, acquisition candidate, six-frame information
boundary, Prob4D used-or-unused declaration, target identity, calibration rule,
scientific result, or physical evidence count.
