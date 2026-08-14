# Prequential registered-query stability

## Scope

`causal4d.prequential_query_stability` projects an unchanged
`PrequentialAbductionPathV1` into one caller-registered finite query. It answers a
different question from posterior-weight stability:

```text
Did the posterior move between prefixes?
```

versus:

```text
Did that movement materially change the registered physical query?
```

A large component-space KL divergence can be harmless when the affected
components make nearly identical query predictions. Conversely, a small change
in posterior mass can matter when it moves a low-dimensional safety- or
paper-relevant query. The diagnostic keeps those interpretations separate.

It does not change factual abduction, feed one prefix posterior into another,
select a prefix, read held-out future observations, or alter the frozen
36-execution method.

## Construction

Supply one deterministic query vector for every complete prequential support
component, together with registered coordinate labels, units, and positive
characteristic scales:

```python
from causal4d.prequential_query_stability import (
    build_prequential_query_stability,
)

stability = build_prequential_query_stability(
    prequential.path,
    component_query_values_m,
    query_id=registered_query.query_id,
    query_labels=("late-track-x", "late-track-y", "late-track-z"),
    query_units=("m", "m", "m"),
    query_scales=(0.01, 0.01, 0.01),
    confidence_level=0.90,
    metadata={"registered_before_target_access": True},
)

summary = stability.summary_arrays()
print(summary["final_mean_shift_standardized_l2"])
print(summary["final_gaussian_wasserstein_standardized"])
```

`component_query_values_m[c, q]` must use the same component order as the source
prequential path. The content identity binds the source path, query definition,
scales, confidence level, complete component-query matrix, posterior weights,
prefix stops, and diagnostic metadata.

## Reported quantities

For every causal prefix, the artifact reports:

- posterior query mean and covariance in the declared native units;
- coordinatewise equal-tail credible intervals;
- coordinatewise absolute mean movement from the preceding prefix;
- coordinatewise absolute mean movement from the final declared prefix;
- scale-normalized Euclidean mean movement from the preceding and final prefixes;
- scale-normalized moment-matched Gaussian 2-Wasserstein distance from the
  preceding and final prefixes; and
- mean coordinatewise credible-interval overlap with the preceding and final
  prefixes.

Characteristic query scales are mandatory. They make multivariate distances
meaningful when coordinates differ in magnitude or units and make the
standardized diagnostics invariant to a consistent conversion such as metres to
millimetres. Native-unit summaries remain available for physical interpretation.

The Wasserstein quantity compares Gaussian distributions with the exact mixture
mean and covariance at each prefix. It is a moment diagnostic, not a claim that
the finite posterior query is Gaussian. The credible intervals are instead
computed directly from the weighted finite support, coordinate by coordinate.

## Invariances and validation

Focused tests establish that:

- a consistent unit conversion changes native values but not standardized drift;
- a simultaneous permutation of component identities, posterior columns, and
  query rows leaves every reported diagnostic unchanged;
- the first previous-prefix and final final-prefix distances are exact zeros;
- the corresponding interval-overlap controls are exact ones;
- nonpositive scales, duplicate query labels, nonfinite values, and component
  count mismatches fail closed; and
- changing either the source posterior path or any component query value changes
  the content identity.

## Interpretation boundary

This artifact can diagnose the earliest prefix at which a registered query
appears stable, but it must not choose a new confirmatory prefix from target
outcomes. Prefix selection, tolerances, and any stability threshold must be
frozen on source or calibration executions in a separately versioned protocol.

A stable query does not establish provider competence, intervention
identifiability, empirical calibration, counterfactual accuracy, physical
contact truth, deployment safety, or state of the art. It should be interpreted
alongside held-out execution-level proper scores, coverage, interval width,
finite-support adequacy, exact fallback accounting, and the registered physical
experiment.

## Prospective source-frozen gate

A separately versioned future protocol may apply the source-frozen thresholds in
[`prequential_stability_gate.md`](prequential_stability_gate.md). That gate uses
only preceding-prefix movement, requires a preregistered number of consecutive
passes, selects the first passing causal prefix, and returns an exact caller-owned
fallback when no prefix is admitted. The diagnostic in this document remains
unchanged and does not itself select a prefix.
