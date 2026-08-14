# Cross-branch readout-correlation sensitivity

`PhysicalPosterior` currently provides branch-marginal conditional readout
variance but no identified cross-branch conditional discrepancy covariance. An
`independent_readout` interventional contrast therefore adds the two branch
query covariances and sets the unobserved cross term to zero.

`causal4d.interventional_contrast` now provides an additive sensitivity artifact
for the uncertainty in that assumption. It does not replace the source contrast
or construct a new physical posterior.

## Marginal model

For one paired finite-support component and one registered scalar query output,
let

```text
v_a = Var(Q(R_a) | component)
v_b = Var(Q(R_b) | component).
```

At declared conditional correlation `rho`, the marginal contrast variance is

```text
v_delta(rho) = v_a + v_b - 2 rho sqrt(v_a v_b),
-1 <= rho <= 1.
```

This is the exact scalar Cauchy-Schwarz range. The same scalar grid is evaluated
for every query output, but the result remains coordinatewise. It does not claim
that all output-wise extrema belong to one simultaneous multivariate
cross-branch covariance.

## Build the sensitivity artifact

Start with an ordinary contrast whose conditional-variance policy is
`independent_readout`:

```python
from causal4d.interventional_contrast import (
    build_interventional_contrast,
    build_interventional_contrast_readout_correlation_sensitivity,
)

contrast = build_interventional_contrast(
    branch_a,
    branch_b,
    query,
    branch_a_label="do(lift_high)",
    branch_b_label="do(lateral_low)",
    coupling_policy="shared_twin_phi",
    conditional_variance_policy="independent_readout",
)

sensitivity = (
    build_interventional_contrast_readout_correlation_sensitivity(
        branch_a,
        branch_b,
        contrast,
        correlations=(-1.0, -0.5, 0.0, 0.5, 1.0),
        metadata={"registered_before_target_access": True},
    )
)
```

The builder binds both branch posterior IDs, the complete source contrast, its
query and coupling, and the exact correlation grid. It independently
reconstructs each branch's query-marginal variance and requires the zero-
correlation row to reproduce the source contrast's variance and
`P(Q_a-Q_b>0)`.

The artifact reports, for every grid value and query output:

- conditional variance;
- total mixture variance;
- posterior probability that branch A exceeds branch B; and
- the grid envelopes of total variance and probability.

The finite-support component means, coupling weights, and posterior mean remain
unchanged.

## Strict archive

```python
import hashlib
from pathlib import Path

from causal4d.interventional_contrast import (
    load_interventional_contrast_readout_correlation_sensitivity,
    save_interventional_contrast_readout_correlation_sensitivity,
)

path = Path("contrast-readout-correlation.npz")
save_interventional_contrast_readout_correlation_sensitivity(
    path,
    sensitivity,
    overwrite=False,
)
archive_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
restored = load_interventional_contrast_readout_correlation_sensitivity(
    path,
    expected_sha256=archive_sha256,
)
assert restored.artifact_id == sensitivity.artifact_id
```

The archive is atomically published, non-pickled, strictly typed, loaded from one
bounded symlink-free snapshot, and content addressed.

## Interpretation boundary

The correlation grid is an assumption or source-frozen sensitivity range. The
artifact does not estimate correlation from branch marginals, establish
calibration, provide individual-level real counterfactual ground truth, or
select a favorable correlation after target access. A source-estimated covariance
can instead use `RegisteredCrossBranchQueryCovarianceV1`, documented in
[`cross_branch_query_covariance.md`](cross_branch_query_covariance.md), but still
requires independent source or controlled paired evidence.
