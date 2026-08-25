# Causal4D Real-Experiment Milestone

Status: **next scientific milestone; primary-method development frozen**.

Decision date: 2026-07-26.

The controlled counterfactual result has passed. The next result that can
materially change the first-paper claim is the preregistered same-object,
multi-action physical experiment: 18 grasp sessions, 36 command executions,
and independent-execution calibration. Another discrepancy mechanism, semantic
component, planner, or public-data branch cannot substitute for this evidence.

## Milestone outcome

The milestone is complete only when the locked protocol has been acquired and
reported as one of two scientifically useful outcomes:

1. successful factual prediction, held-out action/contact transfer, and
   independent-execution calibration under the registered gates; or
2. a well-powered negative result that quantifies where transfer or calibration
   fails, with all registered executions, exclusions, uncertainty intervals,
   and replay/reset variance retained.

A negative result is not a reason to tune on the target executions. It is the
reported empirical boundary of the frozen method.

## Method-development freeze

The following rules apply to the primary real-experiment analysis:

- No new intervention, discrepancy, semantic, reconstruction, or planning
  mechanism may enter the primary comparison before the 36-execution result is
  reported.
- The exact Causal4D commit, clean-worktree state, protocol, acquisition
  schedule, exact locked Bayesian-PhysTwin revision, final v5 pre-acquisition
  governance amendment, mechanism-gate control evidence, scope document, and protocol
  document are sealed before the first confirmatory execution.
- The confirmatory uncertainty path is
  `causal4d calibration execution-block`: one preregistered execution per
  independent session, nine calibration units per outer fold, and the locked
  rank-9-of-9 threshold at nominal 90% coverage.
- `causal4d calibration real` remains an explicitly diagnostic coordinate-level
  affine-calibration path and cannot produce the confirmatory calibration claim.
- Target outcomes may not select methods, hyperparameters, exclusions,
  calibration transforms, thresholds, or optional branches.
- Optional semantic and public-data results cannot rescue a failed factual,
  transfer, new-contact, or calibration gate.
- A method-affecting defect found before target outcomes are inspected requires
  collection to stop and a new protocol/version to be issued. A method-affecting
  defect found after inspection is reported as a limitation; it is not repaired
  and rerun under the same registration.

Acquisition-only tooling may still improve when it does not alter the recorded
signals, quality gates, target IDs, information boundary, or analysis outputs.
Every such change must be logged with its commit and reason.

## Freeze and acquisition workflow

V5 permits Florian Pfaff to perform these checks as the single registered
self-attesting operator. Every report must state that no independent
pre-acquisition attestation is claimed.

Start from a clean checkout of the commit intended for acquisition. Scaffold the
non-overwriting dataset and readiness evidence first:

```bash
causal4d protocol real scaffold \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1

causal4d protocol readiness scaffold \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

Complete and seal the source-panel, actuator, support/gravity, and
nonconfirmatory dry-run gates. These operational approvals must predate the
method freeze. Then write and self-attest the freeze manifest under v5:

```bash
causal4d protocol freeze seal \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --frozen-by "florianpfaff"

causal4d protocol freeze attest \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
  --verified-by "florianpfaff"
```

After the self-attestation, seal the software-environment gate. It binds the
exact freeze and attestation hashes, Causal4D and Bayesian-PhysTwin package artifacts,
the observation producer, and an explicit Prob4D used-or-unused declaration.
Finally, require the hash-verified readiness decision:

```bash
causal4d protocol readiness status \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  --verify-file-hashes \
  --require-ready \
  --output-json \
  /data/causal4d-sloth-multi-action-v1/preacquisition-readiness.json
```

Before every acquisition session, verify that the checkout still matches the
sealed commit and that no locked file has drifted:

```bash
causal4d protocol freeze validate \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  /opt/causal4d-frozen \
  --expected-causal4d-commit "$(git -C /opt/causal4d-frozen rev-parse HEAD)"
```

The seal command refuses a dirty Git worktree. Freeze schema v2 records file
checksums, the protocol design digest, the exact Bayesian-PhysTwin commit from
`requirements/ci/bayesian-phystwin-provider-v1.sha`, the final v5 amendment
digest, its mechanism-gate control digest, the six-frame observation boundary,
and the registered analysis entrypoints. It also freezes the confirmatory
execution-block score, calibration unit, fold count, finite rank, and
non-claims. A manifest that substitutes the older coordinate-pooled calibration
command fails validation.

The separate readiness status does not mutate that freeze or the v5 amendment.
It derives the collection decision from the registered prerequisite validators,
checksummed operational records, software lineage, chronology, and zero
confirmatory manifests. See
[causal4d_preacquisition_readiness.md](causal4d_preacquisition_readiness.md).

## Execution checkpoints

### 1. Readiness gate

Before confirmatory execution 1:

- pass the versioned pre-acquisition amendment;
- register the physical object and all three canonical contact-node sets;
- pass the slip go/no-go pilot;
- complete the exact 12-execution, 12-independent-session source panel;
- validate commanded versus measured actuation;
- validate support, gravity, camera/controller, and shared-clock registration;
- pass one nonconfirmatory end-to-end dry run;
- seal and self-attest `method_freeze.json` with the registered operator;
- bind the exact software distributions and observation producer;
- confirm that no confirmatory manifest or target outcome exists; and
- obtain `first_confirmatory_execution_allowed=true` from
  `causal4d protocol readiness status --verify-file-hashes --require-ready`.

### 2. Confirmatory acquisition

Collect all 36 executions in the locked order. Record failures and exclusions
rather than silently replacing runs. Stop on a failed protocol-level gate; do
not redesign the method in response to target behavior.

### 3. Blind validation

Before analysis, run:

```bash
causal4d protocol real validate-dataset \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1
```

Validation must cover the protocol copy, schedule, object registration, slip
pilot, all execution manifests, required streams, and checksums.

### 4. Registered analysis

Report the preregistered primary comparisons:

- nominal PhysTwin;
- Bayesian-PhysTwin with nominal realized intervention;
- frozen Causal4D;
- intervention oracle as a diagnostic only.

The primary targets are factual continuation, chronological same-grasp transfer,
new-contact transfer with fresh `kappa_cf`, and the 12 locked cross-action/contact
calibration folds. Every fold must be fitted and evaluated through
`causal4d calibration execution-block`; the target evaluation may not revise the
source-frozen threshold.

## Mandatory report

The first report produced from the confirmatory dataset must include:

- results for all 36 executions or every preregistered exclusion and reason;
- execution-level effect estimates and uncertainty intervals;
- factual, same-grasp, new-contact, and calibration results separately;
- coverage, interval width, NLL or energy score, NEES, and worst-group coverage;
- replay/reset variance and sensitivity to the finite calibration sample;
- the largest and second-largest calibration scores, maximum-to-median ratio,
  leave-one-calibration-session-out thresholds, and fold-wise interval width;
- oracle diagnostics that separate inference, proposal, and model discrepancy;
- either the successful transfer/calibration claim or a precise negative bound.

No architecture extension is the next milestone after sealing. The next decision
is made from the real result.
