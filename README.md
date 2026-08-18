# Causal4D

Causal4D is a research framework for **Bayesian abduction of realized
interventions and held-out interventional prediction for deformable-object
dynamics**.

```text
commanded action u
        |
        v
realized intervention z = (actuation realization phi, contact state kappa)
        |
        v
uncertain physical rollout and observable future
```

The software makes the causal sequence explicit:

1. start from an uncertain physical-twin belief;
2. infer how an observed command was physically realized from an allowed
   response prefix;
3. branch at the prefix endpoint under a held-out action;
4. propagate physical, intervention, and unresolved-discrepancy uncertainty;
5. optionally reweight safe rollouts through a separately gated semantic prior.

This repository is the canonical home of Causal4D. It was extracted with
history from
[BayesianPhysTwin](https://github.com/IPS-Stuttgart/BayesianPhysTwin); the
migration boundary is recorded in
[docs/migration_from_bayesian_phystwin.md](docs/migration_from_bayesian_phystwin.md).

## Five-minute start

Install the core package and run the deterministic CPU-only
abduction-intervention-prediction demonstration:

```bash
python -m pip install -e .
python -m causal4d.demo.aip --output-dir build/aip-demo
```

The demonstration writes non-pickled, content-addressed contracts for a
`TwinBelief`, factual intervention, counterfactual query, physical posterior,
and projected posterior. It reloads every artifact and verifies that replacing
the held-out suffix cannot change factual abduction.

The output is a controlled software demonstration, not physical evidence or a
scientific result. See
[the end-to-end AIP guide](docs/aip_end_to_end_demo.md).

### Recommended public imports

New artifact consumers should use the artifact-only namespace:

```python
from causal4d.artifacts.v1 import TwinBelief, load_contract, save_contract
```

Estimator consumers should import AIP operations separately:

```python
from causal4d.inference.v1 import (
    abduct_factual_intervention,
    apply_counterfactual_operator,
    project_physical_posterior,
)
```

`causal4d.api.v1` remains supported for the existing controlled-benchmark API,
and historical package-root exports remain available for compatibility. New AIP
integrations should prefer the split namespaces so artifact-only code does not
implicitly depend on estimator implementations. The stability policy is
specified in [docs/public_api.md](docs/public_api.md).

## Installation

Core and public protocol code:

```bash
python -m pip install -e .
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

BayesianPhysTwin adapters:

```bash
python -m pip install -e ".[phystwin]"
```

The `phystwin` extra accepts BayesianPhysTwin `>=0.4,<0.5` and validates the
provider manifest at runtime. Frozen experiments instead lock both repositories
through the corresponding frozen requirements file. Visual, Warp-runtime, and
actuator-calibration dependencies remain separate so the controlled benchmark
stays lightweight.

## Command-line interface

Causal4D 0.5 installs one executable with typed grouped routes:

```bash
causal4d --help
causal4d commands list
causal4d commands describe protocol/real
causal4d commands migrate causal4d-real-protocol
causal4d commands validate --require-installed
```

The 67 historical `causal4d-*` console scripts are no longer installed. Their
successor routes remain available, while frozen tags and milestone environments
retain the original executables. See
[the command-line guide](docs/command_line.md) and
[the 0.5 migration table](docs/command_migration_0_5.md).

### Controlled benchmark construction

The aggregate compatibility API remains useful for controlled protocol
construction:

```python
from causal4d.api.v1 import CounterfactualBenchmarkConfig, build_protocol

config = CounterfactualBenchmarkConfig(
    frame_count=18,
    training_repeats=1,
    parameter_grid_count=3,
)
protocol = build_protocol(config)

for object_protocol in protocol:
    print(
        object_protocol.graph_object.name,
        object_protocol.validation_action.action_id,
        object_protocol.test_action.action_id,
    )
```

The runnable example is
[`examples/python_api_quickstart.py`](examples/python_api_quickstart.py).

Run the controlled benchmarks with:

```bash
causal4d benchmark counterfactual \
  --output-dir runs/causal4d-counterfactual-v1

causal4d benchmark latent-contact \
  --output-dir runs/causal4d-latent-contact-v1
```

## Project map

### Causal4D core

`src/causal4d/` owns typed posterior contracts, intervention abduction,
counterfactual operators, discrepancy transfer, semantic trust gates, physical
validation, and prospective mechanism gates. The structural model, causal
timing, transport assumptions, and real-data identification boundary are
formalized in
[docs/causal_model_and_identification.md](docs/causal_model_and_identification.md).

Analysis-only action comparisons use the typed, content-addressed
`InterventionalContrastPosteriorV1` contract documented in
[docs/interventional_contrast.md](docs/interventional_contrast.md). They do not
change a source posterior or the frozen physical protocol.

### BayesianPhysTwin integration

[BayesianPhysTwin](https://github.com/IPS-Stuttgart/BayesianPhysTwin) supplies
the uncertain deformable-object twin: state and parameter particles, graph
geometry, PhysTwin/Warp replay, and perception/discrepancy artifacts. Causal4D
consumes those artifacts and owns intervention and counterfactual inference.
Production source uses versioned provider facades rather than importing
BayesianPhysTwin internals. See
[docs/bayesian_phystwin_provider.md](docs/bayesian_phystwin_provider.md).

### Public-data studies

`src/causal4d_public/` contains source-locked Deform360 and PokeFlex adapters,
preflight checks, technical-failure accounting, shared-physics controls, and
frozen public-data protocols. These studies are evidence about specific model
classes and information boundaries; they are not all positive confirmations.

### Prob4D

[Prob4D](https://github.com/IPS-Stuttgart/Prob4D) is a separate probabilistic 4D
observation and calibration feeder. It is not assumed prior literature and is
not part of Causal4D's core causal claim. Causal4D consumes versioned Prob4D
observation artifacts only through narrow, source-calibrated gates with exact
fallback.

## Reproduction and diagnostics

Create and independently verify an immutable, target-free paper reproduction
bundle with the `causal4d paper reproduce` route. It regenerates the registered
report shell, source-verifies supplied tables and gate decisions, and records
exact hashes without copying raw sensor data. See
[docs/paper_reproduction.md](docs/paper_reproduction.md).

Query-space uncertainty attribution is available through
`causal4d diagnostic uncertainty decompose-query`. It uses exact Shapley
attribution over declared finite factors and verifies numerical additivity and
content identity. It is diagnostic-only and cannot select or change the frozen
physical method. See
[docs/query_variance_decomposition.md](docs/query_variance_decomposition.md).

## Next scientific milestone

The controlled result has passed. The decisive first-paper milestone is the
locked same-object physical experiment: 18 grasp sessions, 36 command
executions, and independent-execution calibration. Primary-method development
is frozen for this result; another discrepancy mechanism, semantic component,
planner, or public-data branch cannot replace it.

The experiment must report either successful transfer/calibration or a
well-powered negative result without target-informed method selection. Operator
instructions and exact evidence boundaries live in:

- [source-panel acquisition](docs/causal4d_source_panel_acquisition.md);
- [pre-acquisition readiness](docs/causal4d_preacquisition_readiness.md);
- [real-evidence accounting](docs/causal4d_real_evidence_status.md); and
- [the physical-experiment milestone](docs/causal4d_real_experiment_milestone.md).

Keeping the operational command sequence in those versioned runbooks avoids
turning the repository landing page into a second, potentially divergent copy
of the registered protocol.

## Evidence boundary

- `milestones/v0.3.0-causal4d-aip/` is the frozen controlled and first real AIP
  milestone.
- Released PhysTwin interactions are diagnostic-only after their recorded
  audits and must not be reused for further model selection.
- Deform360 and PokeFlex preserve source/target access boundaries, retained
  technical failures, and unsealable cases separately.
- Graph persistence remains the unresolved-discrepancy fallback unless a
  physical mechanism passes prospective held-out gates.
- A semantic prior has zero influence unless its locked trust gate passes;
  rejection gives exact physical-posterior fallback.
- The 36-execution same-object protocol is the decisive pending milestone;
  optional branches cannot alter or rescue its primary result.

See [docs/causal4d_paper_scope.md](docs/causal4d_paper_scope.md) for the narrow
paper claim and [the documentation map](docs/README.md) for the full set of
conceptual, integration, diagnostic, and operator guides.

## Repository layout

```text
src/causal4d/          core inference, contracts, and PhysTwin adapters
src/causal4d_public/   Deform360 and PokeFlex public-data studies
configs/               locked protocol and registration artifacts
docs/                  formulation, protocols, diagnostics, and claim limits
milestones/            immutable research milestones and evidence manifests
runs/                  small checked-in diagnostic result bundles
scripts/remote/        reproducible remote execution wrappers
tests/                 unit, protocol, parity, and artifact-boundary tests
```

## License and citation

Causal4D software and project documentation are available under the
[MIT License](LICENSE). Third-party datasets, model checkpoints, provider
repositories, and externally sourced artifacts retain their own terms and are
not relicensed by this repository. See [docs/licensing.md](docs/licensing.md)
and [`CITATION.cff`](CITATION.cff).
