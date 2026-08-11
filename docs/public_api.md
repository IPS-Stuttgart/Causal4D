# Versioned Python API

Causal4D contains stable contracts, registered evidence machinery, diagnostics, and
fast-moving research prototypes. Importing every convenient top-level name made it
unclear which interfaces downstream projects could rely on and forced even a simple
package import to initialize unrelated research modules.

The supported version-1 surface is explicit:

```python
from causal4d.api.v1 import (
    CounterfactualQuery,
    PhysicalPosterior,
    PrefixLikelihoodConfig,
    apply_counterfactual_operator,
    project_physical_posterior,
)
```

Only names listed in `causal4d.api.v1.__all__` are covered by the v1 compatibility
promise. Existing imports such as `from causal4d import PhysicalPosterior` remain
compatible, but new downstream code should prefer the versioned path.

## Import behavior

The unversioned package root retains its historical export inventory through lazy
attribute loading. `import causal4d` now defines the version, export table, and lazy
resolver without importing every experimental, protocol, semantic, or provider
module. Accessing a historical export loads its owning module once and caches the
exact object at the package root:

```python
import causal4d

posterior_type = causal4d.PhysicalPosterior
```

This changes import cost, not object identity or public names. A wildcard import
still resolves every name in `causal4d.__all__` and therefore deliberately loads the
complete historical surface. Importing `causal4d.api.v1` loads the modules required
by the supported API but not unrelated research families such as action-support,
latent-contact-v2, semantic-freshness, or decision-trace machinery.

## What v1 contains

The surface is deliberately conservative:

- causal query and posterior contracts;
- rollout-bank and prefix-likelihood interfaces;
- the counterfactual operator and explicit physical-posterior query projection;
- hierarchical intervention abduction;
- full-joint Gaussian observation contracts and updates;
- explicit interventional-contrast query and posterior contracts; and
- the versioned BayesianPhysTwin provider boundary.

Registered acquisition internals, CLI implementations, paper-specific report
builders, prospective promotion machinery, semantic branches, and experimental
contact models remain in their owning modules. They can evolve without expanding a
stable compatibility promise accidentally.

## Counterfactual query projection

A counterfactual rollout bank contains one factual intervention-endpoint frame and
then exactly `query.horizon_frames` future frames. The dense counterfactual operator
retains that endpoint for compatibility. Use the explicit projection helper for a
query-facing artifact:

```python
dense = apply_counterfactual_operator(
    bank,
    manifest,
    twin_belief,
    factual_intervention,
    query,
)
future = project_physical_posterior(dense, query)
```

The default projection removes the endpoint and preserves the registered
`query_node_indices` order. Set `include_endpoint=True` only when the endpoint is
part of the intended output. The projected artifact retains component weights,
causal ancestry, the source posterior identity, and the complete frame/node policy
in its content-addressed metadata.

## Execution-only grouped batching

Large grouped-abduction studies may evaluate finite support in bounded component
batches without changing the estimator:

```python
from causal4d.intervention_abduction import abduct_factual_intervention

factual = abduct_factual_intervention(
    bank,
    twin_belief,
    observations,
    prefix_frame_count=6,
    grouped_evidence=evidence,
    grouped_component_batch_size=64,
)
```

`grouped_component_batch_size` is an execution-only memory bound. It is not written
to scientific metadata, and the batched path is required to reproduce the dense
posterior weights, diagnostics, and artifact identity exactly. The dense path
remains the default. This research entry point is documented here for resource
control but is not thereby added to the v1 compatibility surface.

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
is the same object as its historical top-level counterpart. Subprocess import probes
also lock the lazy root boundary and prevent unrelated research modules from
silently re-entering the stable import path.
