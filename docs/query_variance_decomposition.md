# Registered-query variance decomposition

## Scope

`causal4d diagnostic uncertainty decompose-query` attributes the covariance of
one fixed finite-posterior query to caller-declared support factors and additive
conditional covariance sources. It is an analysis diagnostic. It does not modify
the posterior or infer that a named support factor is the true physical cause of
an error.

Let posterior component `c` have weight `w_c`, query mean `mu_c`, and declared
conditional covariance sources `Sigma_{c,s}`. The total query covariance is

```text
Cov(Q) = Cov_c(mu_c) + sum_s E_c[Sigma_{c,s}].
```

The first term is finite-support uncertainty. The second contains within-component
uncertainty sources such as observation, discrepancy, or numerical covariance,
provided by the caller under an explicitly additive interpretation.

## Exact factor attribution

Suppose every component has labels for factors such as physical particle,
actuator realization, contact patch, or slip regime. For a subset `S` of factor
names, define

```text
V(S) = Cov(E[mu_c | labels in S]).
```

For factor `j`, Causal4D reports the exact Shapley covariance

```text
Phi_j = sum_{S subset F without j}
        |S|! (|F|-|S|-1)! / |F|!
        * (V(S union {j}) - V(S)).
```

The declared factor contributions satisfy

```text
sum_j Phi_j = V(F).
```

Any remaining finite-component variation is retained as

```text
unresolved_component = Cov_c(mu_c) - V(F).
```

It is never silently assigned to the nearest named factor. The complete
reconstruction is therefore

```text
Cov(Q)
  = sum_j Phi_j
  + unresolved_component
  + sum_s E_c[Sigma_{c,s}].
```

Exact attribution is intentionally limited to eight factors because its cost is
exponential in the declared factor count.

## Input archive

The NPZ archive is non-pickled and must contain exactly the arrays named by the
specification. The defaults are:

```text
component_weights       shape (component,)
component_query_means   shape (component, query)
```

Every conditional covariance source has shape
`(component, query, query)`. Unexpected arrays, duplicate array assignments,
object arrays, Boolean/string numerical coercion, nonfinite values, and
non-positive-semidefinite covariance matrices fail closed.

Example specification:

```json
{
  "schema_version": 1,
  "artifact_kind": "Causal4DQueryVarianceDecompositionInputV1",
  "query_id": "late-endpoint-position",
  "query_labels": ["x", "y", "z"],
  "query_units": ["m", "m", "m"],
  "query_scales": [0.01, 0.01, 0.01],
  "factor_values": {
    "physical_particle": ["p0", "p0", "p1", "p1"],
    "contact": ["left", "right", "left", "right"]
  },
  "conditional_covariance_arrays": {
    "readout_discrepancy": "readout_covariance"
  },
  "metadata": {
    "registered_before_target_access": true
  }
}
```

The number of labels in every factor array must equal the component count. The
factor order in JSON does not affect the result identity.

## Build and validate

```bash
causal4d diagnostic uncertainty decompose-query build \
  posterior-query-input.npz \
  query-decomposition-spec.json \
  query-variance-decomposition.json

causal4d diagnostic uncertainty decompose-query validate \
  query-variance-decomposition.json
```

The output is content-addressed over:

- the query definition and characteristic scales;
- normalized posterior-weight and component-query identities;
- complete factor labels;
- conditional covariance archive identities;
- every reported covariance, share, and diagnostic; and
- finite JSON metadata, including verified input NPZ and specification hashes
  when the CLI is used.

Portable validation checks the schema, content identity, positive-semidefinite
matrices, support inventories, between/total covariance reconstruction,
coordinatewise variance shares, and scale-standardized trace shares.

## Interpretation

The artifact reports both native-coordinate variance shares and a scalar trace
share after division by the registered query scales. The latter is useful when
query coordinates have different units or characteristic magnitudes.

Several boundaries remain essential:

- Shapley attribution is relative to the declared factor set. Adding or removing
  a correlated factor can redistribute attribution.
- A support label is not an independently measured physical cause.
- Conditional covariance sources are summed because the caller declares them
  additive; the artifact does not infer independence or non-overlap.
- Small posterior variance does not establish empirical calibration.
- The diagnostic cannot select a confirmatory query, prefix, model, threshold,
  exclusion, or optional branch from target outcomes.

For paper-facing use, report this decomposition beside held-out execution-level
proper scores, coverage, interval width, exact fallback, and oracle attribution.
