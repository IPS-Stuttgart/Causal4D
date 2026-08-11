# Stable causal-core typing boundary

## Scope

Causal4D applies an additive strict MyPy ratchet to the small reusable core that
owns versioned artifacts, factual intervention abduction, grouped likelihoods,
and the counterfactual operator. Historical diagnostics, acquisition tooling,
and fast-moving research modules remain on the repository's incremental typing
policy until they are promoted in separate reviewed tranches.

The single source of truth is:

```bash
python scripts/ci/run_stable_core_mypy.py
```

It runs MyPy with:

```text
--python-version 3.12
--disallow-untyped-defs
--warn-return-any
--no-implicit-reexport
```

against exactly:

```text
src/causal4d/api/v1.py
src/causal4d/contracts.py
src/causal4d/counterfactual.py
src/causal4d/grouped_likelihood.py
src/causal4d/intervention_abduction.py
src/causal4d/observation_evidence.py
```

Both ordinary CI and the required merge gate invoke that same runner. The policy
test locks the exact options, target inventory, command construction, and one
workflow invocation per required workflow.

## Repair boundary

The initial ratchet required only explicit NumPy return casts, generated-array
annotations, one local-variable rename, and a typed wrapper around NumPy's
`savez_compressed` stub mismatch. These changes do not alter array values,
posterior weights, archive members, causal timing, artifact identity, estimator
semantics, registered protocols, target access, or evidence counts.

Expansion requires an intentional change to the runner inventory and policy
test. Strict mode is not enabled repository-wide merely to create unrelated
migration work.

## Acquisition and merge-base independence

The strict tranche does not own pre-acquisition readiness or operator identity.
Those paths continue to inherit the current fail-closed behavior from `main`,
including the requirement for a genuinely distinct independent verifier. A
typing-only pull request must therefore pass the complete merge-result suite
against the current base; it cannot freeze, bypass, or reinterpret an older
acquisition fixture merely to satisfy static analysis.
