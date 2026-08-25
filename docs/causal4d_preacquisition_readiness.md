# Pre-acquisition readiness attestation

The registered v5 amendment defines the order of work before the 36-run
confirmatory experiment and supersedes v4 only for human-separation governance.
Its Boolean collection gate is intentionally frozen at `false`; it is a
preregistration artifact, not a mutable checklist.

`causal4d protocol readiness` adds a separate, evidence-derived decision layer.
It never rewrites the v2, v3, v4, or v5 protocol files. The first confirmatory
execution is permitted only when every prerequisite and operational gate below
validates from immutable, checksummed evidence.

## Decision contract

The readiness status distinguishes three states:

- **ready**: all evidence validates, all file hashes were checked, the registered
  chronology is respected, and no confirmatory manifest exists;
- **valid but incomplete**: the evidence tree is internally consistent, but one
  or more required artifacts are absent or still templates;
- **invalid**: present evidence is malformed or contradictory, an operational
  approval postdates the method freeze, or confirmatory collection already
  started before the gate opened.

With `--require-ready`, these states use exit codes `0`, `3`, and `2`,
respectively. This lets an acquisition launcher fail closed without confusing an
ordinary incomplete setup with corrupt evidence.

## Filesystem contract

Scaffolding creates five non-overwriting templates below the dataset root:

```text
preacquisition/
├── signature_panel.json
├── actuator_sync.json
├── support_registration.json
├── end_to_end_dry_run.json
├── software_environment.json
└── source_panel/executions/<execution-id>/manifest.template.json
```

Each gate record binds the locked protocol and active v5 amendment, its underlying
files, completion and approval timestamps, the no-target-outcomes boundary, and
a canonical SHA-256 digest. The scaffold also writes one source-panel manifest
template per registered execution. Operators complete a separate staging copy;
`source-panel-publish` validates all bound files and atomically creates the final
`manifest.json` without overwriting an existing record. `seal-gate` verifies all
bound files before replacing the completed gate template atomically. A sealed
gate cannot be resealed. Evidence descriptors use the same relative-path
contract throughout:

```json
{"path": "relative/file", "sha256": "<64 lowercase hex>", "bytes": 123}
```

During hash verification, descriptor paths must resolve to ordinary files below
the dataset root. Symlinked files and symlinked intermediate directories are
rejected, including links that would escape to an otherwise valid external file.

### Signature panel

The signature gate requires the exact 12 v2 source-panel execution IDs in their
registered order and 12 independent reset/grasp sessions. Every bound
`SourcePanelExecutionManifest` must state that it is complete, source-only,
outside all confirmatory folds, included without quality-gate failures, and did
not use target outcomes. Its artifact descriptors are hash-verified.

Use the source-panel status before every physical source session. It validates
that completed manifests form the exact registered prefix and reports the next
execution together with its complete registered command profile:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes
```

Publish a completed staging manifest only through the exactly-once path:

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json
```

The publisher admits only the reported next execution, recursively rejects
target-outcome fields, verifies every artifact checksum and byte count, and
validates the temporary manifest before the final path can exist. See
[physical source-panel acquisition](causal4d_source_panel_acquisition.md) for the
complete operator procedure and failure boundary.

### Actuator synchronization

The actuator gate requires one checksummed
`ActuatorRealizationCalibration` artifact for every source-panel execution. It
checks the locked PyRecEst version, the source/dry-run information boundary,
hardware-timestamp authority, artifact identity, and the registered maximum
RGB-D/actuator synchronization error.

### Support and gravity registration

The support gate records the locked world frame, measured gravity vector,
support geometry, registration closure error, and the threshold chosen by the
pre-acquisition calibration procedure. The registration file must be included in
the gate's evidence descriptors.

### End-to-end dry run

The dry-run gate requires a nonconfirmatory execution ID and successful exercise
of all registered stages:

```text
synchronized_acquisition
observation_prefix_build
intervention_abduction
held_out_prediction
artifact_hash_validation
status_generation
```

The dry run may not reuse a confirmatory execution ID or use target outcomes.

### Software lineage

The software-environment gate is sealed after the method freeze. It binds:

- the exact method-freeze and registered self-attestation file hashes;
- the frozen Causal4D and Bayesian-PhysTwin commits;
- package versions and SHA-256 descriptors for the installed wheel or equivalent
  immutable distribution artifact;
- an explicit Prob4D declaration. When Prob4D supplies claim-bearing
  observations, its commit, version, distribution, and observation-contract
  version are required. When it is not used, the record must say so and give a
  reason;
- the actual observation producer and Python runtime identity; and
- a checksummed resolved dependency report, the selected numerical backend,
  NumPy/SciPy and applicable Torch/Warp/OpenCV versions, CUDA runtime and driver
  identity when used, and an immutable container-image digest when containerized.

This makes the executed bytes, transitive numerical runtime, and observation
producer explicit without changing the registered scientific method.

## Workflow

Create templates after scaffolding the registered dataset:

```bash
causal4d protocol readiness scaffold \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

Complete and publish all 12 source execution manifests through the ordered
source-panel control. Then complete each operational gate JSON and its referenced
evidence and seal it. Operational gates must be completed and approved before
the method freeze. For example:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-complete

causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  signature_panel_complete \
  --approved-by "<reviewer>"
```

Under v5, seal the software environment after `method_freeze.json` and its
registered self-attestation validate. The same registered operator may approve
this gate, and no independent attestation is claimed:

```bash
causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  software_environment_locked \
  --approved-by "florianpfaff"
```

Finally, require a hash-verified ready decision before execution 1:

```bash
causal4d protocol readiness status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-ready \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition-readiness.json
```

The resulting status includes the derived collection flags, prerequisite and gate
validation details, confirmatory-manifest counts, blockers, a portable
`evidence_sha256` that excludes mount-local paths, and an exact host-local
`status_sha256`. Use `evidence_sha256` for archive relocation and cross-host
comparison, and `status_sha256` to identify the exact emitted snapshot. Only
`ready=true` and `first_confirmatory_execution_allowed=true` authorize the first
confirmatory execution.
