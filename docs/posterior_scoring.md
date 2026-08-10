# Dependence-aware posterior scoring

`causal4d.posterior_scoring` evaluates an already-produced physical or task
posterior without changing its components, weights, estimator, evidence ledger,
or acquisition protocol. It is intended for registered secondary analysis and
post-freeze diagnostics.

The existing real-calibration path reports marginal coverage, coordinate NEES,
and diagonal Gaussian scores. Those quantities remain useful, but they cannot
show whether the posterior represents temporal and spatial dependence correctly.
Two posteriors may have identical coordinate means and variances while assigning
opposite probability to the joint trajectory that actually occurred.

## Scores

### Weighted trajectory energy score

For weighted trajectory samples `x_k`, weights `w_k`, and truth `y`, the module
computes

```text
sum_k w_k ||x_k - y|| - 1/2 sum_k sum_l w_k w_l ||x_k - x_l||.
```

The implementation uses the Euclidean norm divided by the square root of the
registered coordinate count. This is only a positive scale change, so propriety
is retained, while the result remains in metres and remains comparable across
registered query dimensions.

`exact_component_energy_score_m` scores the actual finite rollout support.
`sampled_mixture_energy_score_m` additionally includes the posterior's declared
component-wise diagonal readout variance through deterministic seeded,
antithetic Gaussian draws. The latter is a reproducible approximation and does
not invent cross-coordinate conditional covariance.

### Registered variogram score

The variogram score is sensitive to dependence along predeclared coordinate
pairs:

```text
sum_p a_p (|y_i - y_j|^q - E|X_i - X_j|^q)^2.
```

Pairs are canonical flattened indices into a `(T, N, 3)` trajectory. They must
be unique, satisfy `left < right`, and lie inside the registered validity mask.
The pair weights are nonnegative and sum to one. Typical inventories include:

- adjacent-frame pairs for the same node and coordinate;
- early-to-late pairs;
- within-contact-patch node pairs; and
- attachment-to-body pairs.

Use `trajectory_coordinate_index()` to construct indices without relying on a
hand-written flattening convention.

### Multivariate registered-query log score

A linear query `Q` maps the complete trajectory vector into a low-dimensional
registered task output. The implementation moment-matches the weighted rollout
mixture with

```text
Cov(QX) = Cov_k(Q mu_k) + Q diag(E_k sigma_k^2) Q^T
```

and then evaluates the complete multivariate Gaussian log score. Off-diagonal
covariance is retained. Query rows have unique semantic labels and explicit
metre units. A declared diagonal floor can make an otherwise rank-deficient
query covariance positive definite; the stored score records that floor.

`gaussian_log_score()` can also score an independently supplied mean and full
covariance, for example a validated registered BayesianPhysTwin query.

### Ordered variance attribution

The default physical-posterior attribution applies a nested law of total
variance in this explicit order:

1. physical twin particle;
2. persistent realization variables `phi`;
3. contact hypothesis;
4. execution-specific variables `kappa_cf`;
5. residual finite support; and
6. conditional readout discrepancy.

Each increment is nonnegative and all increments add to total mean coordinate
variance. The increments are order-dependent descriptive quantities, not causal
effects of uncertainty sources. The order and that warning are embedded in the
content-addressed result.

## Example

```python
import numpy as np

from causal4d.posterior_scoring import (
    TrajectoryScoreSpecificationV1,
    score_physical_posterior,
    trajectory_coordinate_index,
)

shape = truth_m.shape  # (T, N, 3)
first = trajectory_coordinate_index(1, 4, 0, shape)
second = trajectory_coordinate_index(2, 4, 0, shape)

query = np.zeros((1, int(np.prod(shape))), dtype=float)
query[0, trajectory_coordinate_index(shape[0] - 1, 4, 0, shape)] = 1.0

specification = TrajectoryScoreSpecificationV1(
    name="registered-endpoint-and-temporal-score-v1",
    valid_mask=valid_point_frames,
    variogram_pairs=np.asarray([[first, second]], dtype=np.int64),
    variogram_pair_weights=np.asarray([1.0]),
    variogram_order=0.5,
    query_matrix=query,
    query_labels=("final-node-4-x",),
    query_units=("m",),
    query_covariance_floor_m2=1.0e-12,
)

result = score_physical_posterior(
    physical_posterior,
    truth_m,
    specification,
    conditional_draws_per_component=8,
    random_seed=20260810,
)

payload = result.as_dict()
assert result.score_id
```

Passing a validated `TaskPosterior` uses its task weights only after checking
that it references the exact physical posterior, preserves the exact component
roster, and carries byte-identical physical weights. The result records whether
physical or task weights were scored and binds the corresponding artifact ID.

## Registration and interpretation

The validity mask, variogram inventory and weights, variogram order, query
matrix, labels, units, and covariance floor form a content-addressed
`TrajectoryScoreSpecificationV1`. They should be sealed before target outcomes
are accessed. The result separately binds the posterior artifact, weight
artifact, specification, truth bytes, sample count, and random seed.

These diagnostics do not change the registered 18-session/36-execution study,
the six-frame causal boundary, Prob4D admission, BayesianPhysTwin handoff,
physical evidence count, or the frozen estimator. Passing them is not by itself
evidence of real-data calibration, provider competence, physical benefit,
generalization, or state of the art.

## Computational cost

The exact energy score is quadratic in the number of support samples and linear
in the registered coordinate count. `sample_chunk_size` bounds the number of
left-hand sample rows materialized at once, and `coordinate_chunk_size` bounds
the coordinate block. The variogram score is linear in the number of registered
pairs. Registered query scoring is quadratic in the low query dimension rather
than the full trajectory dimension.
