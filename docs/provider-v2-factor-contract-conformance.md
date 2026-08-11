# Independent Prob4D provider-v2 factor conformance

Causal4D carries a byte-identical copy of Prob4D's advanced provider-v2 factor
corpus under:

```text
causal4d/contract_data/provider_v2_factors_v1
```

The corpus fixes one valid schema-v4 explicit-gauge factor bundle, its causal
square-root gauge tree, and ten adversarial mutations. Its aggregate identity is:

```text
fe0374f46319287e3709497de9cbb73f7497286cf4f157f246096f2c352e4446
```

Causal4D validates these neutral bytes independently. The validator does not
import Prob4D, BayesianPhysTwin, or their implementation modules. It:

- verifies every copied member digest and the aggregate corpus identity;
- validates provider API version 2, factor API version 2, factor schema 4, and
  tree-sparse schema 1;
- validates all factor identifiers, probabilities, causal cutoffs, local point
  covariances, correlation-group settings, and complete joint gauge covariance;
- reconstructs the selected world-space rows directly from the stored `Sim(3)`
  vectors;
- reconstructs the complete joint gauge covariance from the causal tree;
- independently recomputes the tree-prior identity;
- independently recomputes the portable stack-semantic identity; and
- requires all ten declared invalid mutations to fail closed with the registered
  rejection class.

The fixed portable identities are:

```text
minimal prior:
ddb97db5c953635eaa881c4d1b1fbe3e9508a72d0c0fb13a5d2a7f5727021dee

minimal stack semantics:
58621710b5b22a64163c47b4756f200cea13e56491d85a3852af96ec1cb0f4fb
```

Dense/tree and row-materialization comparisons use explicit tolerances:

```text
absolute: 1e-12
relative: 1e-10
```

The historical exact digest of derived floating-point stack bytes remains in the
corpus as a reference-runtime diagnostic. It is not a cross-runtime conformance
condition, because supported NumPy and BLAS implementations may differ in final
bits while preserving every contract identity and numerical relation.

## Verification

Verify the installed independent copy with:

```bash
python -m causal4d.provider_v2_factor_contract_bundle --compact
```

The command reports the copied corpus identity, provider and schema versions,
valid and invalid vector counts, observation count, prior identity, semantic
identity, and numerical tolerances. Verification is local and deterministic: it
reads only the installed package data and performs no network access or remote
artifact discovery.

## Ownership boundary

Prob4D owns construction and calibration of probabilistic 4-D observations.
BayesianPhysTwin owns physical-prior fusion, guarded updates, and exact fallback.
Causal4D consumes only admitted factual evidence and owns intervention abduction
and counterfactual prediction.

This corpus prevents those repositories from silently assigning different
meanings to gauge order, causal cutoffs, row probabilities, covariance sharing,
or tree-sparse factorization. It does not make Causal4D a Prob4D consumer by
Python import, and it does not permit Causal4D to reinterpret or recalibrate
Prob4D uncertainty.

Passing this verifier establishes software-contract interoperability only. It
does not establish observation accuracy, covariance calibration, physical-query
benefit, Causal4D intervention benefit, deployment safety, or state of the art.
