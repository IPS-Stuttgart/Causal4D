# Source-frozen prequential stability gate

## Scope

`causal4d.prequential_stability_gate` converts the existing leakage-safe
prequential diagnostics into a separate, prospective admission decision for a
future or separately versioned protocol. It does not alter factual abduction or
the frozen 36-execution method.

The gate uses only changes from the immediately preceding causal prefix. It
never uses a comparison with the final declared prefix, a held-out future frame,
or a target outcome to select a stopping point.

## Source-frozen configuration

```python
from causal4d.prequential_stability_gate import (
    PrequentialStabilityGateConfigV1,
)

config = PrequentialStabilityGateConfigV1(
    minimum_prefix_frame_count=4,
    required_consecutive_passes=2,
    maximum_previous_total_variation=0.02,
    maximum_previous_kl=0.01,
    minimum_effective_sample_size=4.0,
    maximum_query_mean_shift_standardized_l2=0.10,
    maximum_query_wasserstein_standardized=0.15,
    minimum_query_interval_overlap_fraction=0.80,
    source_artifact_ids=(source_selection_report_sha256,),
    source_only=True,
    registered_before_target_access=True,
    metadata={"independent_unit": "source_session"},
)
```

Thresholds and the required consecutive-pass count must be selected on source or
calibration sessions. The content identity changes when any threshold,
provenance record, or metadata value changes.

## Decision

```python
from causal4d.prequential_stability_gate import (
    evaluate_prequential_stability,
    route_prequential_factual_intervention,
)

decision = evaluate_prequential_stability(
    prequential.path,
    registered_query_stability,
    config,
)

selected = route_prequential_factual_intervention(
    prequential,
    decision,
    fallback=nominal_or_prior_factual_intervention,
)
```

A prefix passes only when all of the following hold:

- a preceding prefix exists;
- the minimum prefix length is reached;
- posterior total variation and KL movement are below their limits;
- posterior effective sample size is above its limit;
- registered-query mean movement and moment-matched Wasserstein movement are
  below their limits; and
- registered-query interval overlap is above its limit.

The first prefix completing the required number of consecutive passes is
selected. Later prefixes cannot rewrite that decision. If no prefix passes, the
router returns the caller-supplied fallback object by exact Python object
identity.

## Content and I/O

Both the configuration and decision are content-addressed strict JSON artifacts.
The decision binds the source prequential path, registered-query stability
artifact, configuration, criterion matrix, consecutive-pass path, selected
factual-intervention identity or fallback reason, and the declaration that only
previous-prefix metrics were used. Publication is atomic and no-overwrite by
default; loaders reject symlinks, duplicate keys, non-finite JSON, coercive array
types, stale identities, and altered array digests.

## Interpretation boundary

Passing the gate means only that one source-frozen definition of posterior and
query stability was met on the admitted causal prefix. It does not establish
intervention identifiability, physical accuracy, empirical calibration,
provider competence, deployment safety, or a positive causal effect. Those
claims still require independent execution-level evaluation with proper scores,
coverage, interval width, harmful-update accounting, and exact fallback.
