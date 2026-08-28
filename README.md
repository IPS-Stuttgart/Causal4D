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
5. return exact fallback when the registered support or evidence is inadequate.

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

The output is a controlled software demonstration, not an empirical result. See
[the end-to-end AIP guide](docs/aip_end_to_end_demo.md).

### Recommended public imports

Artifact consumers should use:

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
and historical package-root exports remain available for compatibility. See
[docs/public_api.md](docs/public_api.md).

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
provider manifest at runtime. Frozen studies instead lock exact repository and
wheel identities.

## Command-line interface

Causal4D installs one executable with typed grouped routes:

```bash
causal4d --help
causal4d commands list
causal4d commands describe protocol/real
causal4d commands validate --require-installed
```

See [the command-line guide](docs/command_line.md).

### Controlled benchmarks

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
timing, transport assumptions, and identification boundary are formalized in
[docs/causal_model_and_identification.md](docs/causal_model_and_identification.md).

Analysis-only action comparisons use the typed, content-addressed
`InterventionalContrastPosteriorV1` contract documented in
[docs/interventional_contrast.md](docs/interventional_contrast.md).

### BayesianPhysTwin integration

[BayesianPhysTwin](https://github.com/IPS-Stuttgart/BayesianPhysTwin) supplies
the uncertain deformable-object twin: state and parameter particles, graph
geometry, physical replay, and perception/discrepancy artifacts. Causal4D
consumes those artifacts and owns intervention and interventional inference. See
[docs/bayesian_phystwin_provider.md](docs/bayesian_phystwin_provider.md).

### Public-data studies

`src/causal4d_public/` contains source-locked Deform360 and PokeFlex adapters,
preflight checks, technical-failure accounting, shared-physics controls, and
frozen public-data protocols. Positive and negative public-data results are both
part of the evidence program.

### Prob4D

[Prob4D](https://github.com/IPS-Stuttgart/Prob4D) is a separate probabilistic 4-D
observation and calibration feeder. It is optional and is not required for the
first Causal4D paper.

## Public-data-only paper program

The first Causal4D paper no longer requires a new hardware acquisition. Its
bounded empirical spine is:

1. **Controlled causal validation.** Joint latent-contact abduction changes
   shifted-contact RMSE from `4.132 mm` to `0.805 mm` and nominal 90% coverage
   from `77.9%` to `90.8%` under shared simulator exogenous conditions.
2. **Topology and identifiability boundary.** On an independent controlled
   panel, exact contact-node recovery is `75%`, one-hop recovery is `100%`, and
   every exact-node miss still improves trajectory prediction. This supports
   predictive intervention equivalence, not physical contact equivalence.
3. **Public Deform360 held-out action.** A source-fitted physical forward model
   is sealed before opening the public target future. On the held-out
   `move both edges` action, visual-only prediction reaches `47.58 mm` Chamfer
   distance versus `71.84 mm` for persistence. The six-frame tactile state is
   worse (`59.74 mm`), while the full-tactile oracle reaches `46.70 mm`.
4. **Public PokeFlex negative control.** The first sparse official-Warp backend
   fails its source gate: pooled leave-one-take-out selection wins `0/5` takes
   and obtains `23.771 mm` mean Chamfer versus `10.093 mm` for persistence. The
   sealed target remains unopened for that rejected backend.
5. **Released real diagnostic.** The already released single-interaction audit
   localizes undercoverage and model-discrepancy headroom without being promoted
   to independent confirmation.

Together these studies support a paper about causal formulation, Bayesian
intervention abduction, held-out public-data prediction, and explicit
identification/failure boundaries. They do **not** establish individual-level
real counterfactual ground truth, physical contact recovery, calibrated real
uncertainty, arbitrary-object generalization, or robot-control safety.

The detailed claim hierarchy is in
[docs/causal4d_paper_scope.md](docs/causal4d_paper_scope.md). Additional public
benchmarks may strengthen the paper, but they are improvements rather than
submission blockers.

## Optional future hardware validation

The previously registered 18-session/36-execution same-object protocol is
retained for provenance and for a possible future collaborator-led validation.
It is not required for the public-data-only first paper, and its `0/36` evidence
state must not be described as a missing result in that paper. The archived
operational boundary is summarized in
[docs/causal4d_real_experiment_milestone.md](docs/causal4d_real_experiment_milestone.md).

## Reproduction and diagnostics

Create and independently verify immutable paper bundles with the
`causal4d paper reproduce` route. Query-space uncertainty attribution is
available through `causal4d diagnostic uncertainty decompose-query`. See
[docs/paper_reproduction.md](docs/paper_reproduction.md) and
[docs/query_variance_decomposition.md](docs/query_variance_decomposition.md).

## Evidence boundary

- `milestones/v0.3.0-causal4d-aip/` is the frozen controlled and first released
  AIP milestone.
- Deform360 and PokeFlex preserve source/target access boundaries, retained
  technical failures, exclusions, and rejected methods separately.
- Public held-out actions support interventional prediction for the exact
  released records; they do not provide both potential outcomes for one
  physical execution.
- Graph persistence remains the unresolved-discrepancy fallback unless a
  mechanism passes its declared source and held-out gates.
- A semantic prior has zero influence unless its locked trust gate passes;
  rejection gives exact physical-posterior fallback.
- The historical 36-execution protocol is optional future validation and cannot
  block or rescue the public-data-only paper.

See [the documentation map](docs/README.md) for conceptual, integration,
public-data, diagnostic, and historical acquisition documents.

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
