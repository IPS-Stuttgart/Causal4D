# Registered real-analysis manifest

The confirmatory Causal4D analysis is sealed as one self-contained,
content-addressed manifest after the method freeze and before target access. The
manifest does not change the estimator, protocol units, target split, calibration
threshold, or exclusion policy. It makes the already registered analysis and
reporting rules independently inspectable and fail-closed.

## Seal the manifest

Run this from the exact clean checkout referenced by `method_freeze.json`:

```bash
causal4d protocol real analysis-manifest-seal \
  /opt/causal4d-frozen \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  /data/causal4d-sloth-multi-action-v1/registered-analysis.json \
  --registered-by "<independent-registrar>"
```

The command revalidates the protocol, every method-freeze file digest, exact
Causal4D and BayesianPhysTwin revisions, and the checked-in real-analysis
interval amendment. Publication is atomic and non-overwriting. The output binds:

- `analysis_id`, the canonical SHA-256 of the logical manifest excluding its own
  identity field;
- exact manifest SHA-256 and byte count;
- exact method-freeze SHA-256; and
- interval-amendment ID, SHA-256, byte count, path, and complete contract.

A second publication to the same path fails. A changed method freeze, protocol,
interval rule, comparison arm, calibration rule, endpoint inventory, or claim
boundary changes the identity or fails validation.

## Revalidate retained bytes

```bash
causal4d protocol real analysis-manifest-validate \
  /opt/causal4d-frozen \
  configs/causal4d/sloth_multi_action_v1.json \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  /data/causal4d-sloth-multi-action-v1/registered-analysis.json
```

Validation reopens the protocol, freeze, analysis manifest, and interval
amendment and verifies their exact identities. The result is target-free and
cannot authorize acquisition by itself.

## Readiness admission

The canonical `causal4d protocol readiness status` path requires
`registered-analysis.json` as a first-class prerequisite. It verifies the exact
method-freeze SHA-256, protocol and amendment identities, software revisions,
manifest identity, and registration chronology. The collection gate exposes:

```text
primary_analysis_registered=true
```

A missing, malformed, consistently re-addressed policy change, pre-freeze
registration, interval-amendment mismatch, or software-identity mismatch keeps
`first_confirmatory_execution_allowed=false`.

## Bound analysis contract

Schema version 3 closes and content-addresses:

- the protocol, v4 operational amendment, interval amendment, method freeze,
  Causal4D revision, and pinned BayesianPhysTwin revision;
- the six-frame causal prefix and zero-future-frame selection boundary;
- the exact primary and diagnostic command entrypoints;
- nominal PhysTwin, BayesianPhysTwin with nominal realized intervention, frozen
  Causal4D, MAP joint-component Causal4D, Causal4D with prior twin weights, and
  the intervention oracle as distinct arms;
- complete factual, same-grasp, and new-contact endpoint inventories;
- equal-target-session effects with primary bootstrap-t, required Student-t
  robustness, and historical percentile sensitivity;
- the rule that both primary and robustness lower bounds must be positive for a
  positive interval claim;
- the 12-fold execution-block calibration contract with nine independent
  calibration units, rank 9 of 9, and no target threshold reselection;
- complete failure and preregistered-exclusion accounting;
- the obligation to report success or a well-powered negative result; and
- the same-object, non-SOTA, non-safety, and non-raw-covariance-calibration claim
  boundary.

The MAP and prior-twin arms are diagnostic only. They cannot replace the frozen
Causal4D primary candidate or rescue a failed primary endpoint. The target-free
report shell binds the same interval amendment, exposes separate primary,
robustness, and historical-sensitivity columns, and requires the two diagnostic
arms to remain visibly non-primary.

## Compatibility

The result-source verifier continues to accept historical schema-version-1
manifests for already frozen consumers. New confirmatory acquisition uses the
schema-version-3 sealing command. Schema 3 is strict: consistently re-addressing
a changed interval, comparison arm, or reporting policy is rejected.

## Scientific boundary

The manifest is preregistration infrastructure, not empirical evidence. It reads
no physical target outcome, does not increment the `0/36` acquisition count, and
cannot rescue a failed factual, transfer, new-contact, or calibration gate.
