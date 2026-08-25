# Causal4D pre-acquisition governance amendment v5

Status: **locked before any physical execution**

Plan ID: `causal4d-sloth-preacquisition-v5-single-operator`

Canonical amendment SHA-256: `c0128865c7b527304dc7a6177d7f935d753bfdbc1e4469243f1acaeae6ce8e93`

## Decision

Causal4D is currently operated by one person. Version 4 required a second,
person-level independent verifier and therefore made the registered physical
study impossible without inventing an identity or recruiting someone who does
not exist. Neither is acceptable.

Version 5 uses the alternative governance path recorded before acquisition:
one registered operator may perform and attest the pre-acquisition checks.
Every artifact must name that operator honestly. The project makes **no claim
of independent pre-acquisition attestation**.

## What changed

Only human-separation governance changed:

- the method freezer may self-attest the immutable freeze;
- the software-environment approver may be the same registered person;
- source-panel review and publication may be performed by the same registered
  person, with the review receipt and byte checks still required;
- contact registration uses schema 4 with two chronological review passes by the registered
  operator rather than two purportedly independent people;
- reports must state that these are self-attested checks.

The operator registry, chronology checks, cryptographic hashes, write-once
artifacts, exact fallback behavior, complete failure accounting, and target
boundary remain mandatory.

## What did not change

The complete v4 scientific and acquisition state is copied byte-for-value into
v5 and validated at load time. In particular, v5 does not change:

- the 18-session, 36-execution acquisition design;
- the 12-execution source panel or its three 8-fit/4-held-out folds;
- contact regions, commands, reversal/speed/hold contrasts, or calibration
  sessions;
- graph basis, rank, mechanism ladder, shrinkage threshold, transfer gates, or
  calibration arithmetic;
- the target split, target-outcome prohibition, no-replacement rule, or
  execution-level reporting;
- the controlled mechanism-gate evidence or any released-case result.

The old v4 artifact remains immutable in the evidence chain. Version 5
supersedes it only for governance and has a new plan identity.

## Required disclosure

Every paper, report, and evidence summary using this study must say:

> One registered operator performed the pre-acquisition checks and
> self-attested the freeze; no independent pre-acquisition attestation is
> claimed.

“Independent verifier,” “independent review,” and equivalent wording must not
be used for evidence produced under this policy unless a genuinely distinct
person later performs a separately registered review.

## Review semantics

Self-attestation is not a waiver of checks. The same operator must still:

1. seal the operator registry before governed evidence;
2. complete each review after the underlying artifact is finalized;
3. preserve the review receipt and all source hashes;
4. perform two chronological contact-registration review passes;
5. run the final hash-verified readiness command;
6. retain every failed or aborted execution with an explicit disposition;
7. keep all target outcomes closed until the registered barrier opens.

This amendment permits the study to proceed honestly. It does not make the
result independently reproduced.
