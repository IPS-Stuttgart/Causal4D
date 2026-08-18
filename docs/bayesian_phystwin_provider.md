# Bayesian-PhysTwin provider boundary

Causal4D consumes Bayesian-PhysTwin only through explicit versioned public
modules:

- `bayesian_phystwin.causal4d_provider_v1` for frozen scientific and diagnostic
  compatibility names;
- `bayesian_phystwin.causal4d_provider_v2` for all production initial and restart
  replay execution through immutable request-complete contracts;
- `bayesian_phystwin.causal4d_belief_provider_v1` for NumPy-only fixed-anchor
  endpoint inference and immutable endpoint posteriors;
- `bayesian_phystwin.causal4d_belief_provider_v2` for additive model-averaged
  endpoints and source-calibrated horizon discrepancy moments;
- `bayesian_phystwin.causal4d_guarded_belief_provider_v1` for exact Prob4D
  runtime, candidate-construction, complete-belief guard, and selected-belief
  receipt identities;
- `bayesian_phystwin.causal4d_tree_block_provider_v1` for strict claim-bearing
  tree-block posterior linear-query covariance without dense joint covariance;
- `bayesian_phystwin.causal4d_graph_provider_v1` for the NumPy-only spring-graph
  value type, graph construction, and released controller grouping semantics;
- `bayesian_phystwin.causal4d_artifacts_v1` for hash-locked released pickle
  inputs and immutable raw-track correspondence;
- `bayesian_phystwin.causal4d_artifacts_v2` for versioned released visual and
  correspondence artifacts without importing experiment internals;
- `bayesian_phystwin.causal4d_public_provider_v1` for source-locked public-data
  diagnostics that still reuse BPT experiment semantics.

The graph module is explicitly parented to Bayesian-PhysTwin's immutable
`causal4d_provider_v2` contract. Causal4D's belief exporter validates the
separate belief-provider manifest and invokes only its fixed-anchor operation.
Registered tree-block covariance queries use their own additive provider and
local validation contract. Guarded handoff v2 imports runtime and complete-belief
selection contracts only through its dedicated versioned facade; an older pinned
wheel may omit that additive module unless the paired compatibility lane sets
`CAUSAL4D_REQUIRE_GUARDED_BPT_PROVIDER=1`. The rollout-bank backend and resumable
cache execute
replay exclusively through provider v2. Provider v1 remains only for frozen
scientific and diagnostic compatibility operations.

Production source and scripts no longer import any unversioned
Bayesian-PhysTwin implementation module. The canonical module inventory is
[`ci/bayesian_phystwin_provider_registry.json`](../ci/bayesian_phystwin_provider_registry.json).
The AST boundary test derives its allowlist from that registry, while a separate
schema test checks unique roles, exact API-version suffixes, local contract
modules, and documentation coverage. Adding a provider therefore requires one
reviewable registry entry rather than synchronized hand-written inventories.

## Compatibility contract

Normal development accepts Bayesian-PhysTwin versions in the range
`>=0.4,<0.5`. Compatibility is not inferred from the package version alone.
Causal4D validates six deliberately separate provider manifests:

- scientific provider API/schema version 1 for frozen compatibility names and
  migrated diagnostics;
- replay provider API/schema version 2 for typed initial/restart requests,
  immutable position/velocity trajectories, frame provenance, and stateless
  replay execution;
- belief provider API/schema version 1 for the fixed robust discrepancy endpoint,
  immutable configuration, causal-prefix validation, and immutable posterior;
- additive belief provider API/schema version 2 for evidence-weighted endpoint
  model averaging and source-frozen horizon discrepancy prediction;
- tree-block query provider API/schema version 1 for strict claim-bearing update
  validation and exact factorized linear-query covariance; and
- graph provider API/schema version 1 for graph and controller grouping values.

The scientific manifest requires its existing `TwinBelief` and `GraphBelief`
artifact schemas. The replay manifest additionally requires `ReplayRequest` and
`ReplayTrajectory` schema version 1 and every provider-v2 replay capability.

The belief provider is checked separately for:

- the exact `bayesian_phystwin.causal4d_belief_provider_v1` API identity;
- causal-prefix endpoint inference and finite-residual preflight;
- immutable fixed-anchor configuration and endpoint posterior capabilities;
- `FixedBayesianAnchorConfig` and `RobustEndpointPosterior` schema version 1;
- the fixed readout/model-discrepancy inference role; and
- the supported Bayesian-PhysTwin package range.

The additive belief provider is checked separately for:

- the exact `bayesian_phystwin.causal4d_belief_provider_v2` API identity and
  manifest schema version 2;
- evidence-weighted endpoint model averaging, per-track component evidence, and
  causal-prefix finite-residual preflight;
- source-calibrated mean retention and horizon-dependent predictive covariance;
- immutable endpoint, prediction, and calibration artifact schemas; and
- an explicit boundary that keeps raw model covariance distinct from interval
  calibration and target-side coverage claims.

The tree-block query provider is checked separately for:

- the exact `bayesian_phystwin.causal4d_tree_block_provider_v1` API identity and
  manifest schema version 1;
- strict claim-bearing update and admission validation;
- factorized linear-query covariance and query-identity binding;
- immutable query covariance and absence of dense joint materialization;
- the exact tree-block update, result, covariance, operator, and query artifact
  schema versions; and
- the explicit boundary that keeps the admitted working Gauss-Newton/IRLS
  covariance distinct from empirical calibration and target-side coverage.

The graph provider is checked separately for:

- graph-provider API/schema version 1;
- `phystwin_spring_graph` and `controller_grouping` capabilities;
- `PhysTwinSpringGraph` artifact schema version 1;
- the exact public graph-provider identity; and
- the exact parent `bayesian_phystwin.causal4d_provider_v2` identity and API
  version 2.

The scientific manifest is loaded with
`load_bayesian_phystwin_provider_manifest()` and checked with
`validate_bayesian_phystwin_provider()`. The replay manifest is loaded with
`load_bayesian_phystwin_replay_provider_manifest()` and checked with
`validate_bayesian_phystwin_replay_provider()`. The belief manifest is loaded
with `load_bayesian_phystwin_belief_provider_manifest()` and checked with
`validate_bayesian_phystwin_belief_provider()`. The additive belief manifest
is loaded with `load_bayesian_phystwin_belief_provider_v2_manifest()` and checked
with `validate_bayesian_phystwin_belief_provider_v2()`. The tree-block query
manifest is loaded with
`load_bayesian_phystwin_tree_block_query_provider_manifest()` and checked with
`validate_bayesian_phystwin_tree_block_query_provider()`. The graph manifest is
loaded with `load_bayesian_phystwin_graph_provider_manifest()` and checked with
`validate_bayesian_phystwin_graph_provider()`. A version, capability, artifact,
provider-identity, inference-role, graph-provider, or parent-provider mismatch
fails closed and is reported explicitly.

## Execution API

Production simulation uses `PhysTwinReplayProvider` from the explicitly versioned
`causal4d_provider_v2` module. Each invocation is one immutable
`InitialReplayRequestV1` or `RestartReplayRequestV1` containing:

- a content-addressed request identifier;
- the exact simulator-configuration and initial-state identifiers;
- grouped spring log-scales and the complete controller trajectory;
- for restarts, the particle-specific endpoint position and velocity; and
- the complete requested frame interval.

Causal4D independently validates every `ReplayTrajectoryV1` response against the
request ID, configuration ID, state ID, frame IDs, timestep, shapes, and finite
position/velocity values. The resumable cache stores and hashes positions,
velocities, frame provenance, timestep, and all three identities. A cache hit can
therefore reconstruct a complete provider-v2 response without instantiating Warp.

Fixed robust endpoint inference uses
`infer_fixed_bayesian_anchor_endpoint()` from the belief provider. Causal4D
validates the installed belief manifest before reading residuals, passes the
exclusive causal cutoff explicitly, and consumes the immutable
`RobustEndpointPosteriorV1` fields. The historical broad provider is no longer the
belief exporter's endpoint-inference dependency.

Registered factorized covariance queries use `RegisteredTreeBlockQueryV1` and
`evaluate_registered_tree_block_query()`. The query binds its name, semantic row
labels, output units, coefficient dimension, matrix digest, and finite JSON
metadata to one content ID. Before execution, Causal4D validates the dedicated
provider manifest. It then independently reconstructs the returned provider
artifact, checks the update/result/query identities and matrix digest, and copies
its covariance into an irreversibly read-only
`ValidatedTreeBlockQueryCovarianceV1`. Accepted and rejected strict updates keep
their original status and reason; a rejected fallback is never relabelled as
accepted evidence.

Provider v1 is not the production replay or endpoint-inference boundary. It
remains a versioned compatibility facade for frozen diagnostics and scientific
operations that have no request-complete replay role. Graph and controller
geometry remain in `causal4d_graph_provider_v1`, which is NumPy-only and declares
replay provider v2 as its parent contract.

The official rollout manifest records the scientific provider, replay-provider-v2,
and graph-provider manifests separately. It also records source-artifact hashes,
simulator/state identifiers, every request ID, exact frame provenance, and position
and velocity digests. Public-data studies additionally record the public-study
provider manifest, and Molmo query preparation requires trusted SHA-256 identities
for both `final_data.pkl` and `calibrate.pkl`. This provenance separation improves
upgrade auditability; it is not an empirical accuracy, calibration, or
causal-prediction claim.

## Development installation

For sibling checkouts, install Bayesian-PhysTwin first and then Causal4D:

```bash
python -m pip install -e "../Bayesian-PhysTwin[graph]"
python -m pip install -e ".[dev]"
CAUSAL4D_REQUIRE_BPT_PROVIDER=1 python -m pytest -q \
  tests/test_bpt_provider_integration.py \
  tests/test_bpt_graph_provider_integration.py \
  tests/test_belief_provider_contract.py \
  tests/test_bpt_belief_provider_usage.py
CAUSAL4D_REQUIRE_TREE_BLOCK_QUERY_PROVIDER=1 python -m pytest -q \
  tests/test_bpt_tree_block_query_provider_integration.py \
  tests/test_tree_block_query_provider_contract.py \
  tests/test_tree_block_belief_query.py
CAUSAL4D_REQUIRE_GUARDED_BPT_PROVIDER=1 python -m pytest -q \
  tests/test_guarded_bpt_belief_handoff_v2.py \
  tests/test_bpt_provider_import_boundary.py
```

Package-based installations may use `python -m pip install ".[phystwin]"`;
the extra encodes the supported `>=0.4,<0.5` range rather than one Git commit.
The cross-repository workflows test the current development branches against
each other's public contracts. The additive tree-block query workflow uses its
own exact provider revision file, leaving the historical fixed provider pin
unchanged. An AST boundary test rejects every BPT import in production source and
scripts unless its exact module is present in the machine-readable registry.
`tests/test_bpt_provider_registry.py` prevents the registry, local contract
modules, and this document from drifting independently.

## Frozen experiments

A frozen experiment must lock the complete two-repository stack, not combine
current Causal4D with an old provider snapshot. The historical pre-provider-API
stack remains available as:

```bash
python -m pip install -r requirements/frozen/causal4d-0.3.0.txt
```

That file locks Causal4D and Bayesian-PhysTwin to exact Git commits. Existing
milestone tags and recorded environments remain unchanged. New experiments
should record the exact BPT revision in every provider manifest in addition to
using the normal compatibility range during development.
