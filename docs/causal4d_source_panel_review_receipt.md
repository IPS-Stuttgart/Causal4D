# Governance-bound source-panel publication receipt

A successful staged preflight proves that the next source manifest and all
referenced artifacts are valid at one instant. The review-receipt flow adds a
registered human review before the existing exactly-once publisher.

Under active v5 governance, the same registered operator may review and publish.
The receipt records that fact and must not claim independent review. Historical
v4 evidence retains its two-person requirement.

## V5 operator sequence

```text
acquire registered source execution
        ↓
hash-verify staged manifest and artifacts
        ↓
registered operator seals a content-addressed review receipt
        ↓
the same registered operator reruns validation and publishes exactly once
        ↓
recompute readiness and the next action
```

The reviewer must be active in the sealed operator registry with either the
`gate_approver` or `independent_verifier` role. A one-person v5 project uses
the registered `gate_approver`; it does not assign or claim an independent role.

## Review command

After staged preflight succeeds:

```bash
causal4d protocol readiness source-panel-review-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --reviewed-by florianpfaff
```

The command reruns the complete preflight and writes exactly once to:

```text
staging/reviews/<execution-id>.json
```

The review must follow execution completion, and the operator registry must
predate it. The receipt binds protocol and amendment identities, execution and
session identities, staged manifest bytes and digest, source completion time,
preflight identities, source-panel state before publication, registry identity,
operator identity and roles, review time, and the no-target/no-method-change
boundary.

## Publication command

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/<execution-id>.json \
  --review-receipt \
  /data/causal4d-sloth-multi-action-v1/staging/reviews/<execution-id>.json \
  --published-by florianpfaff
```

Before publication the CLI reruns preflight, validates the canonical receipt,
revalidates the operator and roles, checks receipt stability, and invokes the
exactly-once publisher. Under v5 the result records:

```text
governance_mode=single_operator_self_attested
independent_people=false
independent_preacquisition_attestation_claimed=false
```

Under historical v4 governance, reviewer and publisher must still resolve to
different person digests and `independent_people=true`.

## Failure handling

A receipt becomes stale if the staged manifest, a referenced artifact, the
source-panel prefix, or the operator registry changes. Review receipts are
non-overwriting; preserve a failed receipt and create a newly versioned
pre-acquisition process rather than editing it in place.

Unknown or inactive operators, unsupported roles, review before execution
completion, field drift, target-outcome fields, symlinks, and noncanonical paths
fail closed. V5 permits same-person review and publication only when the exact
validated v5 governance policy is active.

## Scientific boundary

The receipt does not make an execution valid, increment evidence count, reserve
the next source slot, alter the estimator, or authorize confirmatory collection.
It records a checksummed self-review before publication. It is not independent
reproduction or independent attestation.
