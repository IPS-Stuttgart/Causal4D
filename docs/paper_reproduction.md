# Reviewer-facing paper reproduction bundle

## Purpose

`causal4d paper reproduce` composes the existing registered protocol, sealed
method freeze, registered analysis, report shell, effect-reporting code, and
real-result interpretation into one immutable flat bundle. It is an
orchestration and verification layer; it does not introduce another estimator,
analysis rule, threshold, exclusion rule, calibration method, or scientific
claim.

The command copies only finite registered and derived JSON/Markdown artifacts.
It does not copy raw RGB-D, controller, force/torque, tracking, or other sensor
streams. Every copied and regenerated file is bound by the bundle's exact
SHA-256/byte-count manifest.

## Target-free bundle

Before confirmatory evidence exists, create a portable reproduction plan from
the exact registered sources:

```bash
causal4d paper reproduce \
  --protocol configs/causal4d/sloth_multi_action_v1.json \
  --analysis-manifest /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --method-freeze /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --output-dir /data/causal4d-sloth-multi-action-v1/paper-reproduction-plan
```

The resulting status is `target-free-plan`. The report shell contains no target
values, all result slots remain empty, and the bundle records a physical-evidence
increment of zero.

## Result bundle

After registered analysis, add each complete endpoint effect table and the
source-verified gate summary:

```bash
causal4d paper reproduce \
  --protocol configs/causal4d/sloth_multi_action_v1.json \
  --analysis-manifest /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --method-freeze /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --effect-table /data/causal4d-sloth-multi-action-v1/factual-effects.json \
  --effect-table /data/causal4d-sloth-multi-action-v1/same-grasp-effects.json \
  --effect-table /data/causal4d-sloth-multi-action-v1/new-contact-effects.json \
  --gate-summary /data/causal4d-sloth-multi-action-v1/real-result-gates.json \
  --require-complete \
  --output-dir /data/causal4d-sloth-multi-action-v1/paper-reproduction-v1
```

When the gate summary says the evidence registry is complete, the command
requires at least one source-verified effect table for every endpoint in the
registered analysis. `--require-complete` additionally rejects a missing or
incomplete gate summary and rejects target-informed selection.

An incomplete gate summary is retained as `incomplete-result` when
`--require-complete` is absent. It is interpreted through the preregistered
decision tree rather than silently promoted.

## Independent verification

A reviewer can reopen the bundle without the original working directory:

```bash
causal4d paper reproduce --verify /path/to/paper-reproduction-v1
```

Verification performs two layers:

1. the generic result-bundle verifier checks the exact flat inventory, byte
   counts, SHA-256 values, ordinary-file requirement, and absence of symlinks;
2. the paper verifier reopens the bundled sources and regenerates the report
   shell, Markdown, every session-clustered effect report, source-verification
   record, result interpretation, semantic-conformance report, and reviewer
   README byte for byte.

Use `--require-complete` during verification to reject a target-free or
incomplete bundle.

## Bundle contents

A target-free bundle contains:

```text
manifest.json
paper-reproduction.json
semantic-conformance.json
source-protocol.json
source-method-freeze.json
source-registered-analysis.json
report-shell.json
report-shell.md
README.md
```

Result bundles additionally contain exact `source-effect-table-*.json` copies,
regenerated `effect-report-*.json` products, and—when supplied—the gate summary,
source-verification record, and preregistered interpretation.

`manifest.json` is written last and the whole directory is published by an
exactly-once atomic rename. Existing destinations are never replaced.

## Semantic boundary

The semantic-conformance report explicitly checks that:

- the protocol and registered analysis identify the same semantic design;
- the method-freeze bytes match the digest bound by the registered analysis;
- the registered analysis is locked before target access;
- target outcomes cannot select methods or hyperparameters;
- optional branches cannot rescue or change the primary analysis;
- Prob4D cannot change the frozen primary analysis;
- the report shell is a deterministic projection of the registered analysis;
- every supplied effect table and gate summary is source-verified; and
- complete evidence is not inferred from green tests or interface compatibility.

The bundle is non-claim-bearing. A `complete-result` bundle may contain a
complete registered scientific result, but the act of bundling or verifying it
does not create evidence, establish provider competence, or enlarge the claim
beyond the registered same-object protocol.
