# Public API and compatibility policy

Causal4D contains portable contracts, supported estimator operations, registered
evidence machinery, diagnostics, and fast-moving research prototypes. The public
API is split so artifact-only consumers do not need to import estimator
implementations and so new symbols do not enter a broad compatibility promise by
accident.

## Recommended versioned namespaces

### `causal4d.artifacts.v1`

This is the artifact-only surface. It exports the content-addressed AIP contracts
and the safe non-pickled codec:

```python
from causal4d.artifacts.v1 import (
    CounterfactualQuery,
    FactualIntervention,
    PhysicalPosterior,
    TwinBelief,
    load_contract,
    save_contract,
)
```

Use it in storage, transport, inspection, and provider-boundary code.

### `causal4d.inference.v1`

This is the estimator surface for the AIP pipeline:

```python
from causal4d.inference.v1 import (
    abduct_factual_intervention,
    apply_counterfactual_operator,
    project_physical_posterior,
)
```

Use it when code performs factual abduction, applies a counterfactual action, or
projects a physical posterior into a registered query.

### `causal4d.api.v1`

The aggregate v1 surface remains supported for existing controlled-benchmark and
integration code:

```python
from causal4d.api.v1 import (
    CounterfactualBenchmarkConfig,
    CounterfactualQuery,
    PhysicalPosterior,
    build_protocol,
)
```

It is not removed or silently redirected. New AIP integrations should prefer the
split artifact and inference namespaces because they make dependency direction
and ownership explicit.

Only names listed in the relevant namespace's `__all__` are covered by that
version's compatibility promise.

## Package-root compatibility

Historical imports such as `from causal4d import PhysicalPosterior` remain
compatible through lazy attribute loading. `import causal4d` defines the version,
export table, and lazy resolver without importing every experimental, protocol,
semantic, or provider module. Accessing a historical export loads its owning
module once and caches the exact object at the package root:

```python
import causal4d

posterior_type = causal4d.PhysicalPosterior
```

This changes import cost, not object identity or public names. A wildcard import
still resolves every name in `causal4d.__all__` and therefore deliberately loads
the complete historical surface. New public symbols must enter an owning module
and a reviewed versioned namespace rather than being added to the package root by
default.

## What the aggregate v1 surface contains

`causal4d.api.v1` is deliberately conservative:

- controlled benchmark protocol construction;
- causal query and posterior contracts;
- rollout-bank and prefix-likelihood interfaces;
- the counterfactual operator and explicit posterior projection;
- hierarchical intervention abduction;
- full-joint Gaussian observation contracts and updates;
- interventional-contrast query and posterior contracts; and
- the versioned BayesianPhysTwin provider boundary.

Registered acquisition internals, CLI implementations, paper-specific report
builders, prospective promotion machinery, semantic branches, and experimental
contact models remain in their owning modules. They can evolve without expanding
a stable compatibility promise accidentally.

## Counterfactual query projection

A counterfactual rollout bank contains one factual intervention-endpoint frame
and then exactly `query.horizon_frames` future frames. The dense counterfactual
operator retains that endpoint for compatibility. Use the explicit projection
helper for a query-facing artifact:

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
causal ancestry, source-posterior identity, and the complete frame/node policy in
its content-addressed metadata.

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

`grouped_component_batch_size` is an execution-only memory bound. It is not
written to scientific metadata, and the batched path must reproduce dense
posterior weights, diagnostics, and artifact identity exactly. This research
entry point remains in its owning module rather than entering a versioned public
namespace merely because it is documented.

## Compatibility rules

Within one numbered namespace:

- existing exported names retain their import path, meaning, and owning object;
- signatures do not change incompatibly;
- serialized contracts retain their schema and content-identity rules;
- additions require explicit API review and regression coverage;
- provider schemas remain governed by their own versioned contracts; and
- removals or incompatible semantics require a new numbered namespace.

The exact split-namespace `__all__` inventories are regression-tested. Re-exported
functions and classes must be the same objects as those in their owning modules,
not wrappers with subtly different behavior. A future v2 namespace may coexist
with v1.

Versioned Python compatibility does not override a frozen scientific protocol. A
software-compatible implementation can still be inadmissible for a registered
experiment when its commit, provider lock, configuration, or evidence boundary
differs from the frozen method.

## Executable example

Run:

```bash
python -m causal4d.demo.aip --output-dir build/aip-demo
```

The demonstration exercises both split namespaces, writes and reloads every
contract, and checks the held-out-suffix causal cutoff. See
[aip_end_to_end_demo.md](aip_end_to_end_demo.md).
