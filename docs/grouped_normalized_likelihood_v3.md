# Grouped normalized likelihood v3

## Purpose and status

The robust grouped likelihood preserves full observation covariance, fixed prior
outlier reliability, component-specific discrepancy covariance, and
contributor-aware duplicate caps. Its original `legacy_sum_v1` score is retained
unchanged for frozen and historical analyses.

`normalized_coordinate_mean_v3` is an **opt-in development comparator** for
cases where the number and dimension of admitted groups vary across providers or
prefixes. It does not change the registered 18-session/36-execution physical
estimator, authorize confirmatory collection, or establish calibration or
physical benefit. The score is never selected implicitly from evidence shape,
covariance representation, or provider identity.

## Score

For group `g`, let:

- `ell_g(theta)` be the existing full-covariance robust multivariate Student-t
  mixture log density;
- `c_g` be the source-declared `composite_weight`;
- `m_g` be the largest multiplicity of any contributor named by the group;
- `a_g = 1 / m_g` be the duplicate-evidence power cap; and
- `d_g` be the number of jointly scored coordinates.

The normalized score is

```text
L_v3(theta) = beta * sum_g a_g c_g ell_g(theta)
                     / sum_g a_g d_g,
```

where `beta` is the explicit `likelihood_power`.

The denominator uses contributor caps but not `composite_weight`. Consequently:

- exact duplicated evidence carrying the same contributor identity cannot
  sharpen or weaken the posterior;
- source reliability temperatures remain multiplicative and do not cancel;
- raw coordinate count does not silently determine posterior concentration; and
- the declared likelihood power is applied explicitly on the grouped path.

Group boundaries still define shared robust-mixture states. Repartitioning one
multivariate Student-t group into several groups therefore changes the declared
outlier model unless those groups are explicitly intended as independent
reliability units. The implementation does not claim arbitrary split-group
invariance.

## Covariance handling

Each group continues to use its complete positive-definite covariance. Optional
component uncertainty can be supplied as diagonal variance, dense covariance, a
low-rank factor, or nonoverlapping combinations of those representations. Dense
and low-rank forms are required to agree numerically.

Normalized v3 additionally fails closed when a source covariance condition
number exceeds the preregistered limit. The default limit is `1e12`; a study may
choose a stricter value before target access. No hidden jitter, clipping, or
pseudoinverse is introduced.

## Diagnostics

`GroupLikelihoodDiagnostics` records:

- score semantics and applied likelihood power;
- contributor power caps and effective group weights;
- coordinate counts and normalization coordinate mass;
- source covariance condition numbers;
- effective information fractions; and
- nominal/outlier responsibilities plus dense/low-rank covariance usage.

Legacy artifact metadata remains unchanged. The additional fields are emitted
only when normalized v3 is selected.

For large finite supports,
`posterior_weights_from_grouped_evidence_batched` evaluates the same score in
deterministic component batches and streams only the per-group responsibility
mean and minimum. It therefore avoids materializing the complete
`component_count x group_count` responsibility matrix. The ordinary grouped
likelihood and factual-abduction APIs remain unchanged.

## API

```python
posterior, diagnostics = posterior_weights_from_grouped_evidence(
    prior_weights,
    predicted_components_m,
    evidence,
    prefix_frame_count=prefix_frame_count,
    component_group_covariance_factor_m=structured_covariance,
    score_semantics="normalized_coordinate_mean_v3",
    likelihood_power=12.0,
    max_source_covariance_condition_number=1.0e10,
)
```

The bounded-memory summary path is:

```python
from causal4d.grouped_likelihood_streaming import (
    posterior_weights_from_grouped_evidence_batched,
)

posterior, summary = posterior_weights_from_grouped_evidence_batched(
    prior_weights,
    predicted_components_m,
    evidence,
    prefix_frame_count=prefix_frame_count,
    component_batch_size=64,
    component_group_covariance_factor_m=structured_covariance,
    score_semantics="normalized_coordinate_mean_v3",
    likelihood_power=12.0,
    max_source_covariance_condition_number=1.0e10,
)
```

For factual intervention abduction:

```python
config = FactualAbductionConfig(
    grouped_likelihood_semantics="normalized_v3",
    likelihood_power=12.0,
    grouped_covariance_condition_number_limit=1.0e10,
)
```

The command-line development path is:

```bash
causal4d experiment phystwin abduct-intervention \
  bank.npz belief.npz final_data.pkl factual.npz evaluation.json \
  --grouped-observation-likelihood \
  --grouped-likelihood-semantics normalized_v3 \
  --grouped-covariance-condition-number-limit 1e10
```

## Tested invariants

The test suite verifies:

- exact duplicate-contributor power capping;
- preservation of source `composite_weight` as a reliability temperature;
- coordinate-permutation and group-order invariance;
- dense versus low-rank covariance equivalence;
- normalized information accounting;
- fail-closed handling of ill-conditioned covariance and invalid settings;
- exact preservation of legacy factual artifact identity; and
- explicit separation from dense `normalized_v2`.

These are numerical and provenance guarantees, not empirical evidence that v3
improves prediction, coverage, transfer, or causal identification. Any promotion
requires a separately frozen source-only comparison and held-out evaluation.
