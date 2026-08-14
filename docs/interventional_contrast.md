# Interventional contrast posterior

`causal4d.interventional_contrast` constructs a typed posterior for an explicit
linear difference between two already-produced `PhysicalPosterior` artifacts.
It is analysis-only: neither source posterior, its weights, the estimator, nor a
registered protocol is changed.

The sign convention is always

```text
Q(branch_a readout) - Q(branch_b readout).
```

The structural interpretation and identification assumptions are defined in
[`causal_model_and_identification.md`](causal_model_and_identification.md).

## Define a query

The query matrix multiplies a C-order flattened `(time, node, coordinate)`
readout trajectory. Each row has one unique label and one unit.

```python
import numpy as np

from causal4d.api.v1 import InterventionalContrastQueryV1

trajectory_shape = branch_a.readout_trajectories_m.shape[1:]
dimension = int(np.prod(trajectory_shape))
query_matrix = np.zeros((1, dimension), dtype=float)

# Final-frame x coordinate of node 4.
frame_count, node_count, coordinate_count = trajectory_shape
index = ((frame_count - 1) * node_count + 4) * coordinate_count
query_matrix[0, index] = 1.0

query = InterventionalContrastQueryV1(
    name="final-node-4-x",
    matrix=query_matrix,
    labels=("final-node-4-x",),
    units=("m",),
    metadata={"registered_before_target_access": True},
)
```

The query is content addressed. Changing a coefficient, label, unit, or metadata
value changes `query.query_id`.

## Build a contrast

```python
from causal4d.api.v1 import build_interventional_contrast

contrast = build_interventional_contrast(
    branch_a,
    branch_b,
    query,
    branch_a_label="do(lift_high)",
    branch_b_label="do(lateral_low)",
    coupling_policy="shared_twin_phi",
    shared_kappa_names=(
        "attachment_shift_hand_0",
        "attachment_shift_hand_1",
    ),
    conditional_variance_policy="component_means_only",
)
```

The builder requires both branches to share the factual protocol, `O-`, admitted
`O+` prefix, factual command, BayesianPhysTwin belief, factual-intervention
artifact, and variable schemas. It rejects a coupling that cannot preserve both
source posterior marginals.

### Coupling policies

- `shared_component`: pair identical complete finite-support components.
- `shared_twin_phi`: pair inside common physical-particle/`Phi` strata and draw
  remaining event variables conditionally in each branch. `shared_kappa_names`
  can additionally keep a registered contact patch fixed.
- `independent_product`: form an uncoupled product diagnostic. This is not a
  paired individual-level real counterfactual effect.

A `maximum_pair_count` guard is checked before product support is materialized.

### Conditional variance policies

- `component_means_only`: retain only finite-support component-mean contrasts.
- `independent_readout`: propagate each branch's declared diagonal readout
  variance through the query and add the two query covariances.

No current source artifact identifies cross-branch conditional discrepancy
covariance. The independent policy records zero cross-branch covariance rather
than inventing cancellation.

### Cross-branch correlation sensitivity

An `independent_readout` contrast can be passed to
`build_interventional_contrast_readout_correlation_sensitivity` together with
its two unchanged source branches. The additive artifact evaluates each query
marginal over a declared correlation grid and requires its zero-correlation row
to reproduce the source contrast exactly. It does not estimate correlation or
construct one joint cross-output covariance. See
[`interventional_contrast_readout_correlation.md`](interventional_contrast_readout_correlation.md).

## Posterior summaries

```python
mean = contrast.mean
covariance = contrast.covariance
standard_deviation = contrast.standard_deviation
probability_a_exceeds_b = contrast.probability_positive
lower, upper = contrast.central_interval(0.90)
quantiles = contrast.marginal_quantiles((0.05, 0.5, 0.95))
summary = contrast.as_dict()
```

`central_interval` and `marginal_quantiles` evaluate the declared finite Gaussian
mixture, including exact point masses when conditional variance is zero. They
are posterior credible summaries, not an empirical coverage guarantee.

## Strict archive

```python
import hashlib
from pathlib import Path

from causal4d.api.v1 import (
    load_interventional_contrast,
    save_interventional_contrast,
)

archive_path = Path("contrast.npz")
save_interventional_contrast(
    archive_path,
    contrast,
    overwrite=False,
)
archive_sha256 = hashlib.sha256(archive_path.read_bytes()).hexdigest()
restored = load_interventional_contrast(
    archive_path,
    expected_sha256=archive_sha256,
)
assert restored.artifact_id == contrast.artifact_id
```

The archive is non-pickled, atomically published, loaded from one bounded and
symlink-free exact-byte snapshot, strictly shaped and typed, and content
addresses:

- both source posterior IDs and source query IDs;
- the branch orientation and trajectory shape;
- coupling and conditional-variance policies;
- any shared event-coordinate names;
- the complete query contract;
- pair indices, weights, component contrasts, and conditional covariance; and
- finite immutable metadata and the analysis claim boundary.

## Scientific boundary

The result does not modify source weights or produce a new physical posterior.
It does not authorize target-informed query selection, individual-level real
counterfactual ground truth, calibrated uncertainty, contact recovery, or a
positive physical-effect claim. Paper-facing queries and coupling policies must
be frozen before confirmatory target access.
