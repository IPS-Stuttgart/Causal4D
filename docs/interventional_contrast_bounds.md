# Coupling-robust interventional contrast bounds

## Purpose and status

`InterventionalContrastPosteriorV1` reports a contrast under one explicit
cross-world coupling. Branch marginals alone do not identify that coupling.
`InterventionalContrastBoundsV1` is an analysis-only sensitivity artifact that
shows which marginal contrast conclusions remain valid when the pair weights are
allowed to vary.

The bounds do not modify either physical posterior, the factual intervention,
the estimator, or a registered physical protocol. They do not use target truth
and do not create individual-level real counterfactual ground truth.

## Bound set

The builder starts from one verified `InterventionalContrastPosteriorV1`. It
keeps:

- both source component marginals exactly fixed;
- the source artifact's allowed pair support fixed;
- the declared component contrast means; and
- the declared conditional-variance policy fixed.

Only the nonnegative weights assigned to allowed pairs may change. Therefore:

- an `independent_product` source exposes the full marginal Frechet class;
- a `shared_twin_phi` source gives bounds restricted to its declared shared
  strata and optional shared event coordinates; and
- a `shared_component` source has only one feasible coupling and therefore
  produces collapsed bounds.

The implementation never adds a pair that the source coupling declared
structurally impossible.

## Reported quantities

For each query output, the artifact reports:

- the coupling-invariant posterior mean;
- lower and upper variance;
- lower and upper `P(Q(branch_a) - Q(branch_b) > 0)`; and
- lower and upper CDF values at registered thresholds.

For a threshold `t`, the CDF objective for allowed pair `k` is the conditional
probability

```text
P(Delta_k <= t).
```

For deterministic component means this is an indicator. Under
`independent_readout`, it is the corresponding Gaussian probability using the
already-declared pair covariance. Variance bounds optimize the conditional
second moment and then subtract the squared coupling-invariant mean.

Each coordinate and threshold is optimized separately. Different endpoints may
therefore use different extremal couplings; the artifact is not one joint
worst-case posterior across every reported output simultaneously.

## Example

```python
from causal4d.interventional_contrast import (
    build_interventional_contrast,
    build_interventional_contrast_bounds,
    save_interventional_contrast_bounds,
)

contrast = build_interventional_contrast(
    branch_a,
    branch_b,
    query,
    branch_a_label="do(lift_high)",
    branch_b_label="do(lateral_low)",
    coupling_policy="independent_product",
    conditional_variance_policy="independent_readout",
)

bounds = build_interventional_contrast_bounds(
    contrast,
    cdf_thresholds=(-0.01, 0.0, 0.01),
    metadata={"registered_before_target_access": True},
)

print(bounds.probability_positive_lower)
print(bounds.probability_positive_upper)
save_interventional_contrast_bounds("contrast-bounds.npz", bounds)
```

For multiple query outputs, pass thresholds with shape
`(threshold, query_output)`. A scalar threshold is broadcast to every output.

## Numerical and provenance contract

The transport problems are linear programs solved with SciPy HiGHS. The source
pair weights are verified as a feasible coupling before optimization. A
configurable pair-count guard prevents accidental materialization of an
unbounded analysis problem. Solver failure, malformed thresholds, non-finite
objectives, source-marginal mismatch, and invalid result bounds fail closed.

The result is content addressed and supports strict atomic non-pickled NPZ I/O.
Its identity binds the source contrast and query identities, coupling and
conditional-variance semantics, every threshold and bound, source-coupling
summaries, and finite metadata.

## Interpretation

A narrow interval supports a conclusion that is insensitive to the remaining
coupling ambiguity inside the declared structural support. A wide interval says
that the branch marginals and current structural restrictions do not determine
that conclusion. The selected-coupling estimate should then be reported next to,
not instead of, the sensitivity interval.

These are posterior identification bounds. They are not empirical calibration,
physical-effect confirmation, deployment authorization, or evidence that the
chosen allowed pair support is causally complete.
