# Source-panel staging preflight

Physical source-panel acquisition is expensive, and the final manifest is
published exactly once. A malformed manifest, wrong artifact checksum, or
concurrently changing source file should therefore be detected before the
non-overwriting publication step.

`causal4d protocol readiness source-panel-verify-staged` performs the same
claim-boundary checks needed for admission while leaving the claim-bearing source
registry unchanged. It verifies that the staging file:

- is an ordinary file directly below `dataset_root/staging`, with no symlinked
  path component;
- is named exactly `<next-execution-id>.json`;
- is exactly the next registered source execution and session;
- has the exact schema of the registered manifest template;
- contains no target-outcome field, including nested fields;
- marks the source execution complete, included, and free of quality-gate failures;
- retains the source-only and nonconfirmatory information boundary;
- references only admissible artifacts whose byte counts and SHA-256 values match;
- remains byte-identical throughout validation;
- retains byte-identical referenced artifacts throughout validation;
- leaves the hash-verified source-panel status unchanged; and
- does not collide with an already published final manifest.

## Operator flow

First construct the staging manifest from the immutable registered worksheet and
the actual artifact bytes:

```bash
causal4d protocol readiness source-panel-stage \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --started-at-utc 2026-08-10T08:00:00Z \
  --ended-at-utc 2026-08-10T08:01:00Z \
  --artifact \
    preacquisition/source_panel/executions/source-lift_high-r1/rgbd.npz \
  --artifact \
    preacquisition/source_panel/executions/source-lift_high-r1/controller.csv
```

Repeat `--artifact` for every ordinary file below the registered execution
directory. The builder computes descriptors, refuses replacement, and rechecks the
worksheet, artifacts, and source-panel status after atomically creating the staging
file. It does not approve or publish evidence.

Then run the independent preflight:

```bash
causal4d protocol readiness source-panel-verify-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/source-lift_high-r1.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/source-lift_high-r1-preflight.json
```

A successful result has:

```text
safe_to_publish=true
published=false
claim_bearing_evidence_mutated=false
changes_registered_method=false
source_panel_status_stable=true
target_outcomes_used=false
```

It also records the exact staged-manifest checksum, a complete post-validation
snapshot of every referenced artifact, the stable source-panel status identity,
and the expected one-step progress. The `publication_command_argv` field identifies
the base publication target; the claim-bearing command still requires the separately
sealed review receipt and publisher identity described below. The report is derived
operator evidence. It does not count as a physical execution and does not reserve
the next execution slot.

## Independent review and publication

A registered reviewer inspects the staging manifest, artifact inventory, and exact
preflight and seals the review receipt:

```bash
causal4d protocol readiness source-panel-review-staged \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/source-lift_high-r1.json \
  --reviewed-by "<registered-reviewer-id>"
```

A distinct registered publisher then supplies that receipt explicitly:

```bash
causal4d protocol readiness source-panel-publish \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/staging/source-lift_high-r1.json \
  --review-receipt \
    /data/causal4d-sloth-multi-action-v1/staging/reviews/source-lift_high-r1.json \
  --published-by "<registered-publisher-id>"
```

Publication reruns all validation. This matters because a preflight report is a
snapshot: a staging file or referenced artifact changed afterward is rejected, and
concurrent publication remains protected by the exactly-once final write.

## Mutation and concurrency boundary

The preflight hashes the staging manifest before parsing and after complete
validation. It also snapshots every referenced artifact before and after the
registered validator runs. Any changed byte count, SHA-256 value, artifact path, or
source-panel status identity fails the preflight.

These checks close mutations that occur while the report is being constructed. They
cannot reserve files after the command returns. The exactly-once publisher therefore
remains authoritative and reruns every check immediately before publication.

## Exit behavior

The command returns `0` only after complete read-only validation. Invalid,
out-of-order, target-informed, symlinked, misnamed, missing, changing, or
checksum-inconsistent staging evidence returns `2` through the readiness CLI's
normal fail-closed error contract.

## Scientific boundary

The builder and preflight do not modify the estimator, intervention posterior,
source or target split, six-frame information boundary, quality gates, exclusion
rules, physical evidence count, or registered analysis. Neither authorizes
confirmatory collection nor converts a staged file into claim-bearing evidence.
