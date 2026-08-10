# Bayesian-PhysTwin dynamic belief provider v3

Causal4D recognizes the additive public module
`bayesian_phystwin.causal4d_belief_provider_v3` through
`causal4d.belief_provider_v3_contract`.

The provider adds a source-frozen dynamic endpoint family while retaining the
provider-v2 model-averaged endpoint and recursive Prob4D stream surface. It
exposes:

- an exact last-residual persistence component;
- robust local-level components;
- robust damped-trend components with horizon-dependent predictive means;
- per-track or explicitly selected object-pooled component evidence;
- fail-closed dynamic covariance; and
- immutable dynamic configuration, posterior, and prediction artifacts.

Causal4D validates the exact provider API identity, API/schema version 3,
required capabilities, inherited provider-v2 artifact schemas, dynamic artifact
schema version 2, package compatibility, and all scientific-boundary metadata.
A missing capability, schema drift, metadata drift, provider-name mismatch, or
requested-revision mismatch fails before residual inputs are opened.

## Development check

The contract can be checked against an installed BayesianPhysTwin wheel with:

```bash
CAUSAL4D_REQUIRE_BPT_BELIEF_PROVIDER_V3=1 python -m pytest -q \
  tests/test_belief_provider_v3_contract.py
```

The dedicated installed-wheel workflow builds Causal4D and the exact pinned
BayesianPhysTwin revision as separate wheels, installs them into an isolated
virtual environment, and runs the same contract test without editable imports.

## Scientific boundary

Provider compatibility is software evidence only. It does not establish that a
dynamic endpoint component improves a physical query, that raw predictive
covariance is calibrated, or that object-pooled evidence transfers. Component
families, priors, evidence pooling, and horizon choices require source-only
freezing and independent grouped evaluation.

Provider v3 is a post-freeze diagnostic boundary. It does not alter the frozen
36-execution estimator, provider-v1 or provider-v2 historical experiments,
graph-persistence fallback, Prob4D admission decision, physical evidence count,
or method-freeze rules.
