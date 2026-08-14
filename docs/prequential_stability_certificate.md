# Source-frozen prequential stability routing

## Scope

`causal4d.prequential_stability_certificate` turns an existing leakage-safe
`PrequentialAbductionPathV1` and registered-query stability diagnostic into a
deterministic routing certificate for a separately versioned future protocol.
It does not alter the registered 36-execution estimator.

The rule is frozen from source or controlled sessions and binds:

- the threshold-source artifact;
- one exact fallback artifact;
- limits on previous-step standardized mean movement;
- limits on previous-step Gaussian Wasserstein movement;
- a minimum credible-interval overlap fraction;
- limits on posterior KL divergence and total variation;
- a minimum posterior effective sample size;
- the required number of consecutive passing transitions; and
- the latest causal prefix that may be admitted.

## Chronological decision

Only previous-step quantities enter the decision. The certificate scans causal
prefixes in chronological order and selects the earliest prefix that completes
the required run of passing transitions. Later prefixes cannot alter an already
accepted earlier decision.

When no prefix qualifies, the certificate selects the source-frozen fallback
artifact exactly:

```text
accept_stable_prefix
    -> selected_posterior_id = factual_intervention_ids[accepted_step]

exact_fallback_no_stable_prefix
    -> selected_posterior_id = fallback_artifact_id
```

The fallback must be distinct from every prefix posterior. The supplied query
stability artifact must bind the exact prequential path, prefix inventory, and
posterior weights. Both sources must declare `future_frames_read=0`.

```python
from causal4d.prequential_stability_certificate import (
    PrequentialStabilityRuleV1,
    build_prequential_stability_certificate,
)

rule = PrequentialStabilityRuleV1(
    threshold_source_id=source_threshold_report_id,
    fallback_artifact_id=nominal_factual_posterior_id,
    maximum_previous_mean_shift_standardized_l2=0.10,
    maximum_previous_gaussian_wasserstein_standardized=0.10,
    minimum_previous_interval_overlap_fraction=0.85,
    maximum_previous_posterior_kl=0.10,
    maximum_previous_posterior_total_variation=0.10,
    minimum_effective_sample_size=8.0,
    required_consecutive_steps=2,
    maximum_prefix_frame_count=6,
)

certificate = build_prequential_stability_certificate(
    query_stability,
    prequential_path,
    rule,
)
```

`PrequentialStabilityRuleV1` and `PrequentialStabilityCertificateV1` are immutable,
content-addressed values. The certificate records every step decision, accepted
step and prefix, selected posterior identity, fallback identity, source artifact
identities, and the explicit claim boundary.

## Scientific boundary

Passing the rule is a source-frozen stability decision, not evidence of physical
correctness or calibration. Thresholds, query scales, fallback identity,
consecutive-step policy, and maximum prefix must be selected before target access.
A future promotion study must still report held-out trajectory accuracy, proper
scores, coverage, interval width, worst-session regret, and exact fallback
frequency. A stable but inaccurate posterior remains a failed scientific result.
