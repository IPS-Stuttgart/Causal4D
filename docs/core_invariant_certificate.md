# Core causal and numerical invariant certificate

## Purpose

`causal4d.core_invariant_certificate` runs a deterministic generated-input audit
of the central abduction–intervention–prediction path. It collects several
metamorphic and numerical-differential checks in one content-addressed JSON
certificate that can be executed from a source checkout, wheel, or source
distribution.

The certificate is software evidence only. It does not read physical data,
access target outcomes, estimate a real effect, change the frozen method, or
increment the physical evidence count.

## Checks

The current certificate requires all of the following:

1. modifying every held-out suffix coordinate leaves factual abduction
   byte-identical;
2. dense component batching leaves the factual artifact identity unchanged;
3. permuting factual rollout hypotheses preserves the component-aligned
   posterior distribution;
4. failed intervention identifiability returns the exact joint prior;
5. a relabelled factual rollout action is rejected before likelihood evaluation;
6. permuting counterfactual rollout hypotheses preserves the complete physical
   posterior distribution;
7. permuting registered query-node order produces the exact corresponding output
   permutation;
8. every core contract survives a non-pickled save/load round trip with the same
   artifact identity;
9. diagonal-plus-low-rank and explicitly dense covariance representations give
   the same intervention-identifiability information; and
10. equivalent intervention-parameter unit conversions leave the standardized
    identifiability result unchanged.

The generated problem is the same deterministic CPU-only AIP fixture used by the
public demonstration. The certificate additionally uses the action-bound factual
abduction path, so the finite factual bank must match the exact observed-action
identity.

## Run

```bash
python -m causal4d.core_invariant_certificate \
  --output-json build/causal4d-core-invariants.json
```

The command exits with status 0 only when every check passes. The JSON artifact
is finite, key-sorted, content-addressed, and non-overwriting by default. Use
`--overwrite` only for a non-evidence local convenience run.

## Interpretation boundary

A passing certificate establishes parity and fail-closed behavior for the exact
generated checks. It does not prove correctness on all inputs, physical-model
adequacy, observation competence, empirical calibration, counterfactual causal
validity, deployment safety, or state of the art.

The certificate complements rather than replaces the complete test suite,
installed-wheel integration, security scanning, the target-free design
sensitivity audit, and the registered physical experiment.
