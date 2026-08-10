# Pre-acquisition next-action decision

The registered physical experiment has independent prerequisite, source-panel,
approval, freeze, and software-lineage gates. The authoritative status commands
remain the source of truth, but their complete blocker lists are not an operator
runbook.

`causal4d protocol readiness next-action` derives exactly one admissible next step
from the current hash-verified readiness and source-panel status. It does not create
physical evidence, repair invalid artifacts, choose scientific settings, or inspect
target outcomes.

## Usage

```bash
causal4d protocol readiness next-action \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/next-action.json \
  --output-markdown \
  /data/causal4d-sloth-multi-action-v1/operator/next-action.md
```

File hashes are verified by default. `--skip-file-hashes` is available for a
structure-only inspection, but such a run can never authorize confirmatory
collection. When all logical evidence is present, the resulting action remains
`run_final_hash_verified_readiness_gate` until a hash-verified decision passes.

The command returns:

- exit code `0` only when readiness already authorizes the first confirmatory
  execution;
- exit code `3` for a valid but incomplete state with one prescribed next action;
- exit code `2` when present evidence is malformed, contradictory, out of order,
  or otherwise invalid.

## Deterministic action order

The decision follows the registered information order and emits the first applicable
action from this sequence:

1. create the registered real-dataset scaffold when the dataset root is absent or
   empty;
2. create the non-overwriting pre-acquisition gate and source-panel templates;
3. scaffold and seal the operator identity registry;
4. stop on malformed evidence, chronology violations, unexpected source-panel
   entries, or confirmatory collection that began before readiness;
5. complete the fixed object registration, slip pilot, shared timebase, and
   independently reviewed contact registration;
6. acquire, safely stage, verify, independently review, and publish exactly the
   next registered source-panel execution;
7. seal source-panel completion, actuator synchronization, support/gravity, and
   nonconfirmatory end-to-end dry-run gates;
8. seal the exact clean method freeze;
9. obtain an independent freeze attestation;
10. seal the deployed software environment;
11. run the final hash-verified readiness gate; and
12. validate the freeze and begin only the first registered confirmatory session.

Operator-registry scaffolding is proposed only when both the sealed registry and its
registered draft template are absent. When the sealed registry is missing but a
valid unsealed template is already present, the action advances to
`seal_operator_registry` and remains nonautomatable. A present malformed or
symlinked template yields `stop_and_repair_invalid_evidence`; it is never silently
replaced or treated as a fresh scaffold target. The draft itself is operational
state, not a governed approval.

The source-panel action includes the exact execution ID, session ID, command profile,
manifest template, fixed staging destination, preflight path, review-receipt path,
and exactly-once publication command. It never skips ahead or selects a different
source execution.

## Source-panel publication sequence

A physical source execution has an irreversible claim-bearing publication step.
The next-action artifact therefore represents six explicit phases:

```text
acquire registered execution
        ↓
build the staging manifest from the registered template and actual artifact bytes
        ↓
verify the staged manifest and every referenced artifact
        ↓
independently review the content-addressed preflight report
        ↓
publish the manifest exactly once
        ↓
recompute the next action
```

The staging command is emitted under `staged_manifest_build_argv` and
`staged_manifest_build_text`. It contains placeholders for the exact UTC start/end
times and one artifact path. Repeat `--artifact` for every ordinary file below the
registered execution directory. The fixed destination is recorded under
`staged_manifest_path`.

The builder derives the current `next_execution`, copies the immutable registered
worksheet, computes artifact SHA-256 values and byte counts, and refuses replacement.
It does not verify, review, publish, or mutate the final claim-bearing manifest.

The read-only preflight command is emitted under
`post_acquisition_verification_argv` and writes the path in
`preflight_report_path`. Independent review is emitted under `staged_review_argv`
and produces `review_receipt_path`. Publication is separate under
`claim_bearing_publication_argv`; it is never presented as the immediate
post-acquisition command.

These fields make the human boundary explicit:

```json
{
  "two_person_publication_required": true,
  "independent_review_required_before_publication": true,
  "changes_registered_method": false,
  "target_outcomes_permitted": false
}
```

Publication reruns all registered identity, role, review, staging, and file-hash
checks. Neither a staging file nor a preflight report reserves the next execution
slot or counts as physical evidence.

## Persisted-action freshness

Before executing a persisted decision, validate it against the current filesystem:

```bash
causal4d protocol readiness next-action-validate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/operator/next-action.json \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/operator/next-action-validation.json
```

Validation requires the current schema, exact content hashes, current repository and
dataset mounts, the same logical evidence digest, and the same action/execution
identity. Any intervening publication or evidence mutation makes the decision stale.

## Output contract

The JSON report is content addressed and contains:

- protocol, design, pre-acquisition plan, and amendment identities;
- readiness and source-panel evidence/status identities;
- one `action` object with a stable `action_id` and category;
- argv and shell-rendered forms for executable commands;
- required inputs, outputs, and operator role;
- explicit staging, verification, review, and claim-bearing publication commands
  when applicable;
- the completion check that follows the complete action sequence;
- whether physical acquisition is required;
- whether the action is mechanically automatable;
- a portable `evidence_sha256` that normalizes repository and dataset mount points;
  and
- an exact host-local `status_sha256`.

The Markdown report renders the same sequence for an acquisition operator. Both
formats are derived status artifacts and may be overwritten by a newer snapshot;
they are never counted as experimental evidence.

## Invalid evidence

An invalid state always yields `stop_and_repair_invalid_evidence`. The command lists
the detected malformed prerequisites, invalid gates, chronology blockers,
source-panel blockers, or premature-collection condition. It deliberately does not
synthesize a repair command because a method-affecting defect may require a new
protocol version rather than mutation of the current registration.

Resolve the first invalid boundary under the applicable runbook and independent
review policy, then rerun the decision command.

## Scientific boundary

This interface is operational provenance only. It does not modify:

- the frozen estimator or intervention posterior;
- the six-frame causal information boundary;
- the 12-source-execution or 36-confirmatory-execution order;
- any fit, calibration, or target split;
- a quality gate, exclusion rule, conformal threshold, or analysis decision;
- the physical evidence count; or
- the distinction between controlled counterfactual evidence and real held-out
  interventional prediction.

A `begin_first_confirmatory_session` action is emitted only when the existing
readiness status already has `ready=true`, all requested hashes were verified, and
`first_confirmatory_execution_allowed=true` follows from the registered collection
gate.
