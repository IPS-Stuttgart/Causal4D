# Command-line interface

Causal4D 0.5 installs exactly one executable:

```bash
causal4d --help
causal4d --version
```

Every operation is a typed, lazily imported grouped route. Root help, command
inventory, migration lookup, and metadata validation do not import optional
BayesianPhysTwin, Warp, vision, or GPU dependencies.

## Focused discovery

The default root help intentionally shows only `stable` routes. This gives new
users the supported reproduction, evidence, calibration, and protocol surface
without mixing it with diagnostics, prospective experiments, public-study
utilities, or archived paths:

```bash
causal4d --help
```

The complete catalog remains available without importing command modules:

```bash
causal4d --help-all
causal4d commands list
causal4d commands list --lifecycle stable
causal4d commands list --lifecycle diagnostic --lifecycle experimental
causal4d commands list --claim-bearing --json
```

`--help-all` includes an explicit lifecycle label after every summary. Lifecycle
filters are repeatable and combine as a union. `--claim-bearing` can be combined
with lifecycle and removed-executable filters. An unfiltered
`causal4d commands list --json` remains the complete machine-readable inventory,
so the focused help view does not hide or delete any route.

## Stable workflows

```bash
causal4d benchmark counterfactual --output-dir runs/counterfactual
causal4d benchmark latent-contact --output-dir runs/latent-contact
causal4d protocol real validate-protocol configs/causal4d/sloth_multi_action_v1.json
causal4d protocol freeze validate method_freeze.json protocol.json checkout/
causal4d protocol readiness status checkout/ dataset/ --verify-file-hashes
causal4d protocol acquisition doctor protocol.json checkout/ dataset/
causal4d evidence observation-lineage validate observation.npz twin_belief.npz
causal4d paper reproduce --verify paper-reproduction-v1/
causal4d calibration execution-block --help
```

`protocol readiness` is the fail-closed gate before confirmatory collection.
`protocol acquisition` provides the method-neutral pre-session doctor, health
snapshot evaluator, and append-only session journal.

## Authoritative registry

The registry records, for every supported route:

- the grouped route and Python target;
- lifecycle: stable, diagnostic, experimental, public-study, or archive;
- optional extras and external providers;
- owner and claim-bearing status;
- the removed historical executable, when one existed; and
- the release in which that executable was removed.

Inspect it without importing command modules:

```bash
causal4d commands list
causal4d commands list --json
causal4d commands list --removed-only
causal4d commands describe diagnostic/real/oracle-gap --json
causal4d commands validate --json --require-installed
```

`commands validate` requires installed package metadata to expose only:

```text
causal4d = causal4d.cli.root:main
```

Any installed `causal4d-*` wrapper, missing primary executable, duplicate route,
or target mismatch fails validation.

## Removed executables

Version 0.5 removed all 67 historical `causal4d-*` console scripts. Removed names
are migration metadata, not runnable aliases:

```bash
causal4d commands migrate causal4d-real-protocol
# causal4d-real-protocol -> causal4d protocol real
```

See [the complete 0.5 migration table](command_migration_0_5.md). Frozen tags and
the exact environments retained under `milestones/` preserve historical command
surfaces; current releases do not recreate them.

## Contribution rule

New command functionality must be added to the grouped registry and assigned a
lifecycle. Adding another `[project.scripts]` entry is a packaging-policy
violation. Installed wheel and source-distribution CI invokes `--help` for every
registered grouped route, including commands that the focused root help does not
show by default.
