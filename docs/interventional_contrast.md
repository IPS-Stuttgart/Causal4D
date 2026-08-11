# Interventional contrast posterior

## Purpose

`causal4d.interventional_contrast` constructs an analysis-only posterior for

```text
Q(left action) - Q(right action)
```

from two `PhysicalPosterior` artifacts that descend from the same `TwinBelief`
and `FactualIntervention`. The result keeps the two source posterior marginals
exactly and records the cross-world coupling needed to turn two marginal action
predictions into a posterior over their difference.

The module does not rerun abduction, alter weights, open target outcomes, or
change the registered 36-execution estimator.

## Minimal example

```python
import numpy as np

from causal4d.interventional_contrast import (
    InterventionalContrastSpecificationV1,
    build_interventional_contrast,
    save_interventional_contrast,
    validate_interventional_contrast_sources,
)
from causal4d.posterior_scoring import trajectory_coordinate_index

shape = left_posterior.readout_trajectories_m.shape[1:]  # (T, N, 3)
query = np.zeros((1, int(np.prod(shape))), dtype=float)
query[0, trajectory_coordinate_index(shape[0] - 1, 4, 0, shape)] = 1.0

specification = InterventionalContrastSpecificationV1(
    name="final-node-4-x-left-minus-right",
    query_matrix=query,
    query_labels=("final-node-4-x",),
    trajectory_source="readout",
    coupling_policy="auto",
    conditional_readout_correlation=None,
    confidence_level=0.90,
)

contrast = build_interventional_contrast(
    left_posterior,
    right_posterior,
    specification,
)

print(contrast.posterior_mean_m)
print(contrast.posterior_covariance_m2)
print(contrast.probability_positive)
print(contrast.credible_interval_m)

validate_interventional_contrast_sources(
    contrast,
    left_posterior,
    right_posterior,
)
save_interventional_contrast("contrast.npz", contrast, overwrite=False)
```

`load_interventional_contrast()` revalidates the closed NPZ inventory, exact
array dtypes, specification identity, coupling marginals, immutable arrays, and
content-derived artifact ID. `validate_interventional_contrast_sources()` then
rebuilds the deterministic query and coupling against the exact bound source
posteriors when those artifacts are available.

## Query definition

The query matrix has shape `(Q, T*N*3)` and uses C-order flattening of one
trajectory `(T, N, 3)`. Each row is a dimensionless linear readout with metre
output. Examples include:

- one endpoint coordinate;
- average motion of a registered contact patch;
- a difference between two nodes;
- early-to-late displacement; or
- a frozen task projection.

The result direction is always `left - right`. Swapping the two source
posteriors changes the sign and creates a different artifact identity.

## Coupling policies

The two source posteriors determine `p(Y_left)` and `p(Y_right)`, but not their
joint distribution. The coupling is therefore part of the estimand.

### `auto`

Automatic coupling reads the contact semantics embedded by the counterfactual
operator:

| Source semantics | Resolved policy | Shared variables |
| --- | --- | --- |
| fixed same grasp | `shared_theta_phi_kappa` | particle, `phi`, `kappa` |
| evolving slip | `shared_theta_phi_patch` | particle, `phi`, contact patch |
| new contact | `shared_theta_phi` | physical particle, `phi` |

Within every shared stratum, component choices not included in the shared key
are coupled by the product of their conditional distributions. This preserves
the source marginals while allowing shared uncertainty to cancel in the action
contrast.

Automatic coupling rejects mixed contact semantics. Such a comparison changes
both command and contact mechanism and is not a pure action contrast.

### Explicit shared policies

`shared_theta_phi_kappa`, `shared_theta_phi_patch`, and `shared_theta_phi` expose
the same mechanisms directly for preregistered sensitivity analyses. They fail
when the compared posteriors assign different marginal mass to a supposedly
shared latent stratum.

### `component_id`

This policy shares exact component identities. It is useful only when both
producers intentionally preserve one common component roster.

### `independent`

The independent product preserves both marginals but shares no latent variable.
It is a sensitivity analysis. It generally overstates uncertainty in a contrast
when the two actions descend from the same physical and intervention posterior.
A `maximum_pair_count` guard prevents accidental quadratic expansion.

## State and readout contrasts

### State target

`trajectory_source="state"` compares simulator state trajectories. Conditional
readout variance is not applicable.

### Readout target

`trajectory_source="readout"` compares discrepancy-aware predicted
observations. `readout_variance_m2` supplies only marginal conditional
variances. Its cross-world correlation is not identified automatically.

- `conditional_readout_correlation=None` excludes conditional readout variance
  and reports the finite-support component contrast only.
- `0` assumes conditionally independent readout errors.
- `1` assumes a perfectly shared standardized readout-error mode.
- values in `[-1, 1]` provide an explicit sensitivity analysis.

For correlation `rho`, each source coordinate contributes

```text
v_left + v_right - 2 rho sqrt(v_left v_right)
```

to the contrast variance. The query projection retains cross-query covariance.
The artifact reports both between-component covariance and expected conditional
covariance separately.

## Reported quantities

The result contains or derives:

- exact source posterior, query, action, and factual-lineage identities;
- requested and resolved coupling policies;
- sparse nonzero component-pair indices and weights;
- proof through validation that pair marginals equal both source posteriors;
- query-space contrast components;
- posterior mean and complete query covariance;
- `P(contrast > 0)` for every query row;
- equal-tail marginal credible intervals;
- pair count and effective pair count; and
- a fixed analysis-only claim boundary.

The probability and intervals are exact for the stored discrete support when
conditional readout variance is excluded. When a correlation is supplied, they
are computed from the resulting univariate Gaussian mixture for each query row.
The full covariance retains cross-query dependence, but the reported intervals
remain marginal rather than simultaneous.

## Interpretation

A positive mean or high `P(contrast > 0)` states how the two frozen action
posteriors differ under the recorded coupling assumptions. It does not establish
an individual-level real counterfactual, because only one action can occur in
one physical execution. Confirmatory real claims still require the registered
matched-execution design, session-level analysis, and independent-execution
calibration.

The formal SCM and identification assumptions are documented in
[`causal_model_and_identification.md`](causal_model_and_identification.md).
