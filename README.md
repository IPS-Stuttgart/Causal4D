# Causal4D

Causal4D is a research framework for **Bayesian abduction of realized
interventions and held-out interventional prediction for deformable-object
dynamics**.

The central distinction is:

```text
commanded action u
        |
        v
realized intervention z = (actuation realization phi, contact state kappa)
        |
        v
uncertain physical rollout and observable future
```

The software implements explicit abduction, intervention, and prediction:

1. begin from an uncertain physical-twin belief;
2. infer how an observed command was physically realized from an allowed
   response prefix;
3. branch at the prefix endpoint under a held-out action;
4. propagate physical, intervention, and unresolved discrepancy uncertainty;
5. optionally reweight safe rollouts with a separately gated semantic prior.

This repository is the canonical home of Causal4D. It was extracted with
history from
[Bayesian-PhysTwin](https://github.com/IPS-Stuttgart/BayesianPhysTwin);
the migration boundary is recorded in
[docs/migration_from_bayesian_phystwin.md](docs/migration_from_bayesian_phystwin.md).

## Project Map

### Causal4D core

`src/causal4d/` owns the typed posterior contracts, controlled benchmark,
latent-contact inference, intervention abduction, counterfactual operators,
discrepancy transfer, semantic trust gates, physical validation, and
prospective mechanism gates.

The structural model, causal timing, transport assumptions, and real-data
identification boundary are formalized in
[docs/causal_model_and_identification.md](docs/causal_model_and_identification.md).
Analysis-only action comparisons use the typed, content-addressed
`InterventionalContrastPosteriorV1` API documented in
[docs/interventional_contrast.md](docs/interventional_contrast.md); they do not
change a source posterior or the frozen physical protocol.

### Bayesian-PhysTwin integration

[Bayesian-PhysTwin](https://github.com/IPS-Stuttgart/BayesianPhysTwin) supplies
the uncertain deformable-object twin: state and parameter particles, graph
geometry, PhysTwin/Warp replay, and perception/discrepancy artifacts. Causal4D
consumes those artifacts and owns the intervention and counterfactual
inference. Frozen scientific operations use the versioned
`causal4d_provider_v1` facade, while production replay uses
`causal4d_provider_v2` through the `PhysTwinReplayProvider` protocol. Graph and
released visual artifacts use separately versioned provider facades; production
source does not import Bayesian-PhysTwin internals.

Install the `phystwin` extra for these adapters. Core controlled benchmarks do
not require Warp or the PhysTwin checkout. See
[docs/bayesian_phystwin_provider.md](docs/bayesian_phystwin_provider.md) for
the compatibility and frozen-lock policies.

### Public-data studies

`src/causal4d_public/` contains source-locked Deform360 and PokeFlex adapters,
preflight checks, technical-failure accounting, shared-physics controls, and
the frozen public-data protocols. These studies are evidence about specific
model classes and information boundaries; they are not all positive
confirmations.

### Prob4D

[Prob4D](https://github.com/IPS-Stuttgart/Prob4D) is a separate, newly developed
probabilistic 4D observation and calibration feeder. It is not assumed prior
literature and is not part of Causal4D's core causal claim. Causal4D may consume
versioned Prob4D observation artifacts through a narrow interface, but camera
evidence is admitted only through source-calibrated gates with exact fallback.

## Installation

Core and public protocol code:

```bash
python -m pip install -e .
```

Development tools:

```bash
python -m pip install -e ".[dev]"
```

Bayesian-PhysTwin adapters:

```bash
python -m pip install -e ".[phystwin]"
```

The `phystwin` extra accepts Bayesian-PhysTwin `>=0.4,<0.5` and validates the
provider manifest at runtime. Frozen experiments instead lock both repositories
with `requirements/frozen/causal4d-0.3.0.txt`. Visual, Warp-runtime, and
actuator-calibration dependencies remain separate so the controlled benchmark
stays lightweight.

## Command-line interface

Causal4D 0.5 installs exactly one executable. All stable, diagnostic,
experimental, public-study, and archived operations are typed grouped routes:

```bash
causal4d --help
causal4d commands list
causal4d commands describe protocol/real
causal4d commands migrate causal4d-real-protocol
causal4d commands validate --require-installed
```

The 67 historical `causal4d-*` console scripts are no longer installed. Their
successor routes remain available, while frozen tags and milestone environments
retain the original executables. See [the command-line interface](docs/command_line.md)
and [the complete 0.5 migration table](docs/command_migration_0_5.md).

## Quick Start

Run the controlled counterfactual benchmark:

```bash
causal4d benchmark counterfactual \
  --output-dir runs/causal4d-counterfactual-v1
```

Run the latent-contact benchmark:

```bash
causal4d benchmark latent-contact \
  --output-dir runs/causal4d-latent-contact-v1
```

Generate the controlled collaborator video through the
[`Controlled collaborator demo video`](.github/workflows/controlled-demo-video.yml)
workflow. The uploaded bundle contains an MP4, GIF, poster, summary, and
checksums. Its presentation and claim boundary are documented in
[docs/controlled_collaborator_demo.md](docs/controlled_collaborator_demo.md).

Validate the locked same-object real protocol:

```bash
causal4d protocol real validate-protocol \
  configs/causal4d/sloth_multi_action_v1.json
```

The full PhysTwin abduction chain is documented in
[docs/causal4d_abduction_intervention_prediction.md](docs/causal4d_abduction_intervention_prediction.md).

## Reviewer-facing paper reproduction

Create an immutable target-free bundle from the exact registered protocol,
method freeze, and analysis manifest:

```bash
causal4d paper reproduce \
  --protocol configs/causal4d/sloth_multi_action_v1.json \
  --analysis-manifest /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --method-freeze /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --output-dir /data/causal4d-sloth-multi-action-v1/paper-reproduction-plan
```

The bundle copies no raw sensor data and changes no method. It regenerates the
registered report shell, source-verifies supplied effect tables and gate
decisions, records exact hashes and byte counts, and can be independently
reopened with `causal4d paper reproduce --verify <bundle-dir>`. See
[the paper reproduction guide](docs/paper_reproduction.md).

## Next Scientific Milestone

The controlled result has passed. The next first-paper milestone is the locked
same-object physical experiment: 18 grasp sessions, 36 command executions, and
independent-execution calibration. Primary-method development is frozen for
this result; another discrepancy mechanism, semantic component, planner, or
public-data branch cannot replace it.

The v4 amendment itself remains immutable. After scaffolding the acquisition
dataset, create its separate operational evidence records:

```bash
causal4d protocol readiness scaffold \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

Drive the required 12-execution physical source panel through its registered
order and exactly-once evidence boundary:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes

causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json
```

The status identifies the next execution and its exact command profile. The
publisher admits only that execution, verifies every bound artifact, and never
overwrites a final manifest. See
[docs/causal4d_source_panel_acquisition.md](docs/causal4d_source_panel_acquisition.md)
for the operator workflow and evidence boundary.

Complete and seal the source-panel, actuator, support/gravity, and
nonconfirmatory dry-run gates. Then seal the exact clean Causal4D commit,
Bayesian-PhysTwin pin, protocol files, analysis boundary, and reporting contract:

```bash
causal4d protocol freeze seal \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --frozen-by "<operator-or-principal-investigator>"
```

Independently validate and attest that exact freeze before collection:

```bash
causal4d protocol freeze attest \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
  --verified-by "<independent-verifier>"
```

Seal the software-environment gate after that attestation, then require a
hash-verified readiness decision before confirmatory execution 1:

```bash
causal4d protocol readiness status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-ready \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition-readiness.json
```

The readiness gate binds the 12-run source panel, actuator synchronization,
support/gravity registration, the nonconfirmatory end-to-end dry run, the
method freeze, exact Causal4D/Bayesian-PhysTwin package artifacts, and an
explicit Prob4D used-or-unused declaration. It refuses readiness when any
confirmatory manifest already exists. See
[docs/causal4d_preacquisition_readiness.md](docs/causal4d_preacquisition_readiness.md)
for its evidence and exit-code contracts.

Track acquisition progress without counting templates as evidence:

```bash
causal4d protocol real status \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1 \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/evidence-status.json
```

Before analysis, add
`--repository-root <frozen-checkout> --verify-file-hashes --require-complete`.
The version-2 gate remains closed until the approved timebase, schema-3 physical
contact registration, sealed and independently attested method freeze, all 18
same-grasp session manifests, all 36 execution manifests, and every registered
artifact hash validate. The gate also proves that the timebase, contact
approval, method freeze, and independent attestation do not postdate the first
valid execution. See
[docs/causal4d_real_evidence_status.md](docs/causal4d_real_evidence_status.md)
for the exact accounting and exit-code contract.

The experiment must report either successful transfer/calibration or a
well-powered negative result without target-informed method selection. See
[docs/causal4d_real_experiment_milestone.md](docs/causal4d_real_experiment_milestone.md)
for the freeze, acquisition, and reporting workflow.

## Evidence Boundary

- `milestones/v0.3.0-causal4d-aip/` is the frozen controlled and first real
  abduction-intervention-prediction milestone.
- The released PhysTwin interactions are diagnostic-only after their recorded
  audits; they must not be reused for further model selection.
- Deform360 and PokeFlex artifacts preserve source/target access boundaries,
  retained technical failures, and unsealable cases separately.
- Graph persistence remains the unresolved-discrepancy fallback unless a
  physical mechanism passes the prospective held-out shrinkage, prediction,
  plausibility, transfer, and calibration gates.
- MolmoMotion or another semantic prior has zero influence unless its locked
  trust gate passes; rejection gives exact physical-posterior fallback.
- The 36-execution same-object real protocol is now the decisive pending
  milestone; optional branches cannot alter or rescue its primary result.

See [docs/causal4d_paper_scope.md](docs/causal4d_paper_scope.md) for the narrow
paper claim and the other documents in `docs/` for protocol-specific details.

## Repository Layout

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

## License

Causal4D software and its associated project documentation are available under the
[MIT License](LICENSE). Third-party datasets, model checkpoints, provider
repositories, and externally sourced artifacts retain their own terms and are not
relicensed by this repository. See [docs/licensing.md](docs/licensing.md) for the
scope, historical-version policy, contribution terms, and citation boundary.
