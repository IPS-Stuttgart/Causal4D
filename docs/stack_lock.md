# Three-repository stack locks

Causal4D can publish a content-addressed compatibility lock for the exact
Prob4D, BayesianPhysTwin, and Causal4D wheels exercised together. The lock is a
portable release and CI artifact: it binds the three wheel contents to their
reported package versions, exact tested source revisions, required provider
modules, and the fixed `Prob4D -> BayesianPhysTwin -> Causal4D` pipeline.

## Create a lock

Build all three wheels first, then provide one wheel and one exact 40-character
source revision for every distribution:

```bash
causal4d stack create \
  --wheel wheelhouse/prob4d-0.3.0-py3-none-any.whl \
  --wheel wheelhouse/bayesian_phystwin-0.4.0-py3-none-any.whl \
  --wheel wheelhouse/causal4d-0.5.0-py3-none-any.whl \
  --revision prob4d=<PROB4D_COMMIT> \
  --revision bayesian-phystwin=<BPT_COMMIT> \
  --revision causal4d=<CAUSAL4D_COMMIT> \
  --output causal4d-stack-lock.json
```

Creation fails unless the wheel metadata contains exactly the three canonical
distributions and every source revision is present. The output is written
atomically and is not replaced unless `--force` is supplied.

The `lock_id` is the SHA-256 digest of the canonical JSON payload excluding the
`lock_id` field itself. Distribution order, source repositories, required
provider-v2 modules, and compatibility declarations are schema-locked, so a
consumer cannot silently reinterpret the artifact.

## Verify exact wheel artifacts

Pass the lock and all three wheel files:

```bash
causal4d stack verify \
  --lock causal4d-stack-lock.json \
  --wheel wheelhouse/prob4d-0.3.0-py3-none-any.whl \
  --wheel wheelhouse/bayesian_phystwin-0.4.0-py3-none-any.whl \
  --wheel wheelhouse/causal4d-0.5.0-py3-none-any.whl \
  --json
```

Verification re-reads wheel metadata and checks each distribution's canonical
name, version, byte size, and SHA-256 digest. Renaming a downloaded wheel does
not invalidate it; changing its contents does. Duplicate distributions,
missing wheels, unexpected wheels, malformed metadata, duplicate JSON keys,
and a mismatched `lock_id` fail closed.

For diagnostics that only need to establish that the lock document itself is
untampered, use the explicit weaker mode:

```bash
causal4d stack verify \
  --lock causal4d-stack-lock.json \
  --lock-only
```

`--lock-only` does **not** verify the wheel artifacts and cannot be combined
with `--wheel`.

## Verify the installed interface surface

Add `--installed` to compare the active Python environment with the lock:

```bash
causal4d stack verify \
  --lock causal4d-stack-lock.json \
  --wheel wheelhouse/prob4d-0.3.0-py3-none-any.whl \
  --wheel wheelhouse/bayesian_phystwin-0.4.0-py3-none-any.whl \
  --wheel wheelhouse/causal4d-0.5.0-py3-none-any.whl \
  --installed \
  --json
```

The installed check verifies all of the following without opening experiment
artifacts or target outcomes:

- the installed distribution versions exactly match the locked wheel versions;
- every required provider and lineage module recorded in the lock imports;
- `prob4d.api.v2` reports API version 2;
- `bayesian_phystwin.causal4d_provider_v2` reports provider API version 2; and
- `causal4d.api.v1` reports public API version 1.

Failures are emitted as typed issues such as `distribution_missing`,
`distribution_version_mismatch`, `required_module_import_failed`, and
`public_api_version_mismatch`. The original stack-verification payload remains
unchanged when `--installed` is omitted.

`--lock-only --installed` is allowed as a weaker environment diagnostic. It
checks the lock structure and installed interface surface, but not the three
wheel files. Even when wheel and installed checks both pass, the report keeps
`claim_bearing_ready=false`: Python package metadata and successful imports do
not prove that every installed file is byte-identical to the submitted wheels.
That stronger binding belongs in the build and deployment attestations.

## Evidence boundary

A stack lock proves artifact identity and records the source revisions that the
producer reports for those artifacts. It does not independently prove that the
wheels were built from those revisions; that provenance remains the
responsibility of the build workflow and its checkout, clean-tree, and build
attestations. The installed check proves version and interface compatibility,
not installed-file identity.

The three-repository installed-wheel workflow supplies the stronger context: it
checks out immutable source revisions, builds the wheels, records their hashes,
creates and re-verifies the stack lock with the installed Causal4D CLI, then
runs the isolated golden path, strict claim-bearing provider-v2 attestation,
and cross-repository contract tests. The lock and verification report are
published alongside those diagnostics.

A passing stack lock or installed-stack diagnostic is packaging and
compatibility evidence only. It does not change `claim_ready`, replace
prospective calibration, or count as one of the frozen confirmatory physical
executions.
