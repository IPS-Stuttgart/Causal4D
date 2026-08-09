# Versioned Python API

Causal4D contains stable contracts, registered evidence machinery, diagnostics, and
fast-moving research prototypes. Importing every convenient top-level name made it
unclear which interfaces downstream projects could rely on.

The supported version-1 surface is now explicit:

```python
from causal4d.api.v1 import (
    CounterfactualQuery,
    PhysicalPosterior,
    PrefixLikelihoodConfig,
    apply_counterfactual_operator,
)
```

Only names listed in `causal4d.api.v1.__all__` are covered by the v1 compatibility
promise. Existing imports such as `from causal4d import PhysicalPosterior` remain
compatible, but new downstream code should prefer the versioned path.

## What v1 contains

The initial surface is deliberately conservative:

- causal query and posterior contracts;
- rollout-bank and prefix-likelihood interfaces;
- the counterfactual operator;
- hierarchical intervention abduction;
- full-joint Gaussian observation contracts and updates; and
- the versioned BayesianPhysTwin provider boundary.

Registered acquisition internals, CLI implementations, paper-specific report
builders, prospective promotion machinery, semantic branches, and experimental
contact models remain in their owning modules. They can evolve without expanding a
stable compatibility promise accidentally.

## Compatibility policy

Within `causal4d.api.v1`:

- existing names retain their import path and meaning;
- additive names may be introduced when they are mature;
- incompatible semantics require a new contract or API version;
- provider schema changes remain governed by their own versioned contracts; and
- deprecations should include a documented migration path before removal.

A future `causal4d.api.v2` may coexist with v1. The unversioned top-level package is
kept for backward compatibility and discovery, not as an automatic promise that
every re-export is stable.

## Checking the surface

The repository locks the exact v1 inventory and verifies that each versioned symbol
is the same object as its historical top-level counterpart. This prevents silent
wrappers, accidental copies, and unreviewed expansion of the stable namespace.
