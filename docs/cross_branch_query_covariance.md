# Registered cross-branch query covariance

## Scope

`RegisteredCrossBranchQueryCovarianceV1` supplies an explicitly registered
conditional covariance between the same query evaluated on two counterfactual
branches. It closes a narrow uncertainty gap in the interventional-contrast API
without changing either `PhysicalPosterior`, its weights, the factual
intervention, or a registered physical protocol.

For a coupled component pair `k`, the artifact stores

```text
C_ab[k] = Cov(Q(branch_a), Q(branch_b) | pair k).
```

The marginal query covariances remain owned by the two physical posteriors. The
contrast builder forms

```text
Cov(Q_a - Q_b | k) = C_a[k] + C_b[k] - C_ab[k] - C_ab[k]^T.
```

No cross-branch cancellation is inferred when the artifact is absent.

## Construction

The covariance must be fitted on source or calibration executions and registered
before target access. It binds both source posterior identities, both source
query identities, the registered contrast query, the coupling policy, any shared
`kappa` coordinates, the exact ordered pair support, and every source artifact
used to estimate the covariance.

```python
import numpy as np

from causal4d.interventional_contrast import (
    RegisteredCrossBranchQueryCovarianceV1,
    build_interventional_contrast,
)

cross_covariance = RegisteredCrossBranchQueryCovarianceV1(
    source_branch_a_posterior_id=branch_a.artifact_id,
    source_branch_b_posterior_id=branch_b.artifact_id,
    source_branch_a_query_id=branch_a.source_query_id,
    source_branch_b_query_id=branch_b.source_query_id,
    query_id=query.query_id,
    branch_a_component_count=len(branch_a.weights),
    branch_b_component_count=len(branch_b.weights),
    coupling_policy="shared_twin_phi",
    shared_kappa_names=("attachment_patch",),
    pair_indices=registered_pair_indices,
    cross_covariance=source_fitted_cross_covariance,
    source_artifact_ids=(source_fit_report_sha256,),
    source_only=True,
    registered_before_target_access=True,
    metadata={"independent_unit": "source_session"},
)

contrast = build_interventional_contrast(
    branch_a,
    branch_b,
    query,
    branch_a_label="do(lift_high)",
    branch_b_label="do(lateral_low)",
    coupling_policy="shared_twin_phi",
    shared_kappa_names=("attachment_patch",),
    conditional_variance_policy="registered_cross_branch",
    cross_branch_query_covariance=cross_covariance,
)
```

The ordered `pair_indices` must equal the pair support produced by the declared
coupling exactly. Reordering, dropping, or adding a pair changes the estimand and
is rejected.

## Joint covariance validation

For every pair, Causal4D reconstructs

```text
[ C_a    C_ab ]
[ C_ab^T C_b  ]
```

and requires the complete block matrix to be positive semidefinite. It then
requires the derived contrast covariance to be positive semidefinite as well.
This catches cross-covariance magnitudes or directions that are incompatible
with the two branch marginals. Validation is scale-aware and fail closed.

The cross-covariance matrix itself need not be symmetric. A cross-covariance is
an oriented relation from branch A to branch B; its transpose occupies the other
block of the joint covariance.

## Strict archive

```python
from causal4d.interventional_contrast import (
    load_registered_cross_branch_query_covariance,
    save_registered_cross_branch_query_covariance,
)

save_registered_cross_branch_query_covariance(
    "cross-branch-query-covariance.npz",
    cross_covariance,
)
restored = load_registered_cross_branch_query_covariance(
    "cross-branch-query-covariance.npz"
)
assert restored.artifact_id == cross_covariance.artifact_id
```

The archive is non-pickled, atomically published without overwrite by default,
loaded from one symlink-free exact-byte snapshot, and revalidated against a
closed descriptor and array inventory.

## Interpretation boundary

A registered cross-branch covariance changes conditional contrast uncertainty,
not branch means, coupling weights, or source posterior marginals. It does not
establish empirical calibration, identify a physical mechanism, justify the
allowed pair support, authorize target-informed fitting, or provide
individual-level real counterfactual ground truth. Report it alongside the
independent-readout result and, where coupling ambiguity remains, the
coupling-robust contrast bounds.
