# Physical source-panel acquisition

The registered pre-acquisition amendment requires 12 source-only physical
executions before the method freeze and before any confirmatory target execution.
They use four command profiles with three independent reset/grasp sessions per
profile at the registered upper-torso contact.

This panel estimates repeatability and source-only mechanism signatures. It is
not part of a confirmatory fold and does not increment the `0/36` confirmatory
evidence count.

## Safety boundary

The source-panel control surface enforces these invariants:

- the protocol, v2, v3, and v4 registration chain must validate;
- execution and session identities come from that chain, not from operator input;
- completed manifests must form the exact registered prefix;
- the next execution includes its complete registered command profile;
- staging copies the immutable registered worksheet and fills only completion
  values plus descriptors computed from the actual artifact bytes;
- every artifact must be an ordinary, symlink-free file below the exact registered
  execution directory;
- artifact SHA-256 values and byte counts are recomputed before and after staging;
- staging never creates or replaces a final evidence manifest;
- publication creates `manifest.json` exactly once and never overwrites it;
- target-outcome fields are forbidden recursively;
- templates do not count as completed evidence; and
- source-panel completion requires all 12 intact templates, all 12 final
  manifests, and file-hash verification.

An invalid or missing template, an out-of-order manifest, an unexpected execution
entry, a stale digest, a concurrent artifact change, or a malformed completed
manifest fails closed.

## Scaffold once

First create the registered confirmatory dataset structure and the separate
pre-acquisition evidence templates:

```bash
causal4d protocol real scaffold \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1

causal4d protocol readiness scaffold \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

The readiness scaffold writes one immutable worksheet template per registered
source execution:

```text
preacquisition/source_panel/executions/<execution-id>/manifest.template.json
```

Do not rename, edit in place, delete, or promote that worksheet by a filesystem
move. Retain every worksheet through source-panel gate sealing.

## Inspect the next execution

Before every source session, derive the current status from disk:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition/source-panel-status.json
```

The report identifies `next_execution` and includes the exact registered profile,
contact, realization condition, replicate, execution ID, and independent session
ID. Use that record as the operator instruction. Do not choose another profile or
repair the ordering manually.

A valid incomplete panel is expected. With `--require-complete`, the status command
returns `0` when all 12 executions validate, `3` when the panel is valid but
incomplete, and `2` when present evidence is malformed or contradictory.

## Acquire one source execution

Use a fresh reset and fresh grasp for the displayed execution. Store the raw sensor,
controller, timing, registration, gripper, contact, and technical-quality files
below its registered execution directory:

```text
preacquisition/source_panel/executions/<execution-id>/
```

Preserve technical failures; do not silently replace a registered execution. The
staging command below constructs a publishable successful manifest, so run it only
after the preregistered technical quality gates have passed. When a gate fails,
retain the raw files and failure record for review and do not stage or publish a
manifest that claims inclusion.

## Build the staging manifest safely

Use the exact start and end times and repeat `--artifact` for every artifact file:

```bash
causal4d protocol readiness source-panel-stage \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --started-at-utc 2026-08-10T08:00:00Z \
  --ended-at-utc 2026-08-10T08:01:00Z \
  --artifact \
    preacquisition/source_panel/executions/<execution-id>/rgbd.npz \
  --artifact \
    preacquisition/source_panel/executions/<execution-id>/controller.csv \
  --artifact \
    preacquisition/source_panel/executions/<execution-id>/timing.json
```

The builder derives the exact next execution, reads and validates its immutable
worksheet, copies all registered identities, validates UTC chronology, computes
SHA-256 and byte counts, and atomically creates:

```text
staging/<execution-id>.json
```

It refuses replacement. Paths may be dataset-relative or absolute, but after
resolution every file must remain below the dataset root and below the exact
registered execution directory. Absolute escapes, `..` escapes, symlinks,
directories, duplicate paths, and the worksheet/final manifest files are rejected.
The builder rehashes the worksheet and artifacts after writing and removes the
new staging file if any concurrent change is detected.

The resulting manifest has the registered schema:

```json
{
  "status": "complete",
  "fresh_reset_and_fresh_grasp": true,
  "confirmatory_fold_member": false,
  "target_outcomes_used": false,
  "included": true,
  "quality_gate_failures": [],
  "started_at_utc": "<UTC ISO-8601 timestamp>",
  "ended_at_utc": "<UTC ISO-8601 timestamp>",
  "artifacts": [
    {
      "path": "preacquisition/source_panel/executions/<id>/<artifact>",
      "sha256": "<computed 64-character lowercase digest>",
      "bytes": 123
    }
  ]
}
```

Staging is not verification, approval, or publication. It does not reserve the
execution slot and does not change claim-bearing evidence.

## Verify the staged bytes

Hash-verify the exact staging manifest and all referenced files, and persist the
content-addressed preflight report:

```bash
causal4d protocol readiness source-panel-verify-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/<execution-id>-preflight.json
```

The preflight is read-only. It requires that the staging file is directly below
`dataset_root/staging`, has the exact next execution filename and schema, and still
matches every artifact byte. It also confirms that the final manifest does not
exist and that source-panel status stayed unchanged during verification.

## Obtain independent review

A registered reviewer inspects the staged manifest, artifacts, and exact preflight
and seals a review receipt:

```bash
causal4d protocol readiness source-panel-review-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --reviewed-by "<registered-reviewer-id>"
```

This creates the registered receipt under:

```text
staging/reviews/<execution-id>.json
```

Review remains separate from publication. A stale or changed staging manifest or
preflight must be verified and reviewed again.

## Publish exactly once

A distinct registered publisher performs the irreversible claim-bearing step:

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --review-receipt \
    /data/causal4d-sloth-multi-action-v1/staging/reviews/<execution-id>.json \
  --published-by "<registered-publisher-id>"
```

The publisher revalidates the registered operator roles, review receipt, completed
JSON, exact next-execution identity, artifact hashes, and source-panel status. It
atomically creates:

```text
preacquisition/source_panel/executions/<execution-id>/manifest.json
```

The destination must not already exist. Repeated publication, out-of-order
publication, stale review, or failed validation leaves the final path untouched.
After success, recompute `next-action`; do not manually increment the execution.

## Complete and seal the source-panel gate

After the twelfth publication, require terminal validation:

```bash
causal4d protocol readiness source-panel-status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-complete \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition/source-panel-status.json
```

Only then complete the `signature_panel.json` gate record, bind all 12 final
manifest descriptors as evidence, and obtain the registered independent approval:

```bash
causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  signature_panel_complete \
  --approved-by "<registered-gate-approver-id>"
```

The actuator synchronization, support/gravity registration, and nonconfirmatory
end-to-end dry-run gates remain separate prerequisites. All four operational gates
and the operator registry must predate the method freeze. The software-environment
gate must follow the freeze and independent attestation.

## Evidence interpretation

A green source-panel status means only that the registered 12 source executions
exist in order and their bound files validate. It is pre-acquisition evidence, not
confirmatory performance evidence. It cannot change a method, threshold, exclusion,
target identity, or paper claim, and it cannot authorize execution 1 by itself.
Confirmatory collection remains forbidden until the final readiness status reports:

```text
ready=true
first_confirmatory_execution_allowed=true
```
