# Registered real-analysis reporting

This layer reports paired Causal4D effects without treating the 36 command
executions as 36 independent experiments. It is additive reporting code for the
frozen real protocol. It does not change the estimator, target split,
calibration threshold, exclusion policy, or gate decisions.

## Why sessions are the resampling unit

The protocol contains 18 grasp sessions and two executions per grasp. Executions
within one session share the grasp, reset, registration, object state, and other
nuisance variables. The primary effect report therefore:

1. computes the paired candidate-versus-baseline effect for every included
   registered evaluation unit;
2. averages those effects within each target grasp session;
3. weights the resulting session effects equally; and
4. obtains a deterministic 95% bootstrap-t interval by resampling sessions;
5. computes a Student-t interval as a required veto-only robustness check; and
6. retains the historical percentile interval as non-decision-making sensitivity.

The fixed reporting configuration is:

```text
resampling unit: target grasp session
primary interval: bootstrap-t
required robustness interval: Student-t
historical sensitivity interval: percentile bootstrap
bootstrap replicates: 20,000
bootstrap seed: 20,260,726
confidence level: 95%
```

A positive interval claim requires strictly positive lower bounds from both the
bootstrap-t and Student-t intervals. Student-t may veto but cannot rescue a
bootstrap-t failure. A non-estimable primary interval yields no positive claim;
the negative or bounded result remains reportable. An unweighted execution mean
is diagnostic only and cannot override the equal-session estimate.

## Complete accounting

A report consumes a content-addressed
`Causal4DRealAnalysisEffectTable`. The table must account for every target unit
in exactly one registered endpoint:

| Endpoint | Registered units | Target sessions |
| --- | ---: | ---: |
| factual continuation | 36 | 18 |
| same-grasp transfer | 18 | 18 |
| new-contact transfer | 12 | 12 |

Every record is matched against the locked protocol by source execution, target
execution, target session, acquisition index, action, contact, and realization
condition. Missing, additional, reordered, or relabelled target identities fail
closed.

A preregistered exclusion remains in the table with:

```json
{
  "included": false,
  "exclusion_reason": "<registered reason>",
  "baseline_value": null,
  "candidate_value": null
}
```

Excluded units cannot carry target metric values. Included units cannot carry an
exclusion reason.

## Effect-table contract

The top-level fields are closed:

```json
{
  "schema_version": 1,
  "artifact_kind": "Causal4DRealAnalysisEffectTable",
  "effect_table_id": "<canonical SHA-256>",
  "protocol_id": "causal4d-sloth-multi-action-v1",
  "protocol_design_sha256":
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968",
  "preacquisition_amendment_sha256":
    "0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f",
  "method_freeze_sha256": "<64 lowercase hex>",
  "analysis_manifest_sha256": "<64 lowercase hex>",
  "endpoint": "factual_continuation",
  "metric_id": "track_error_m",
  "metric_unit": "m",
  "lower_is_better": true,
  "target_outcomes_used": true,
  "target_informed_selection": false,
  "object_id": "sloth_plush_instance_1",
  "records": []
}
```

For factual continuation, `unit_id` equals `target_execution_id` and
`source_execution_id` is null. For transfer endpoints, `unit_id` is exactly:

```text
<source_execution_id>-><target_execution_id>
```

The table identity is the canonical SHA-256 of every field except
`effect_table_id` itself. Use
`causal4d.real_analysis_reporting.effect_table_id_for_payload` to derive it.

## Source verification

The report reopens and verifies the concrete sealed method freeze and registered
analysis manifest. Their SHA-256 values must match the effect table, and their
internal protocol, amendment, target-access, and optional-branch restrictions
must pass the same source-verification boundary used by the final interpretation
artifact.

The verifier reads, hashes, and parses the same exact bytes. Duplicate JSON keys,
non-finite JSON values, symbolic links, and concurrent file replacement cannot
silently separate the retained digest from the validated payload.
The registered protocol is also passed through the repository's complete
`validate_protocol` contract. Its design SHA-256 is recomputed from the full
content, so a modified split, execution label, acquisition order, or balance
cannot be admitted by retaining the old embedded digest.

## Reported diagnostics

The output contains:

- the equal-session primary effect and deterministic bootstrap-t interval;
- the required Student-t veto-only robustness interval;
- the historical percentile-bootstrap sensitivity interval;
- the registered two-interval positive-claim decision;
- candidate-better, tied, and worse session counts;
- complete inclusion and exclusion accounting;
- an unweighted execution diagnostic;
- a non-decision-making acquisition-order slope and early/late contrast;
- secondary action, contact, and realization-condition summaries;
- the complete registered action-condition support matrix; and
- explicit same-object and non-safety claim boundaries.

The acquisition-order diagnostic cannot select exclusions or revise the primary
result. Realization-condition summaries are descriptive because all condition/action
cells exist but are unequally replicated, and conditions occupy different
acquisition-time ranges.

## Calibration utility

`summarize_execution_block_utility` combines an existing frozen
`ExecutionBlockConformalCalibration` with its target-fold evaluation. It reports:

- execution-block coverage;
- target coordinate count;
- mean, median, and maximum interval width;
- the frozen threshold;
- largest, second-largest, and median calibration scores;
- maximum-to-median score ratio; and
- leave-one-calibration-session-out diagnostics.

Coverage without interval width is explicitly marked insufficient. Fragility
outputs cannot select or modify the threshold. Pointwise, pooled-coordinate, and
worst-group conformal guarantees remain prohibited.

## Python and module CLI

```python
from causal4d.real_analysis_reporting import (
    build_real_analysis_effect_report,
    write_real_analysis_effect_report,
)

report = build_real_analysis_effect_report(
    "factual-track-effects.json",
    "configs/causal4d/sloth_multi_action_v1.json",
    method_freeze_path="method_freeze.json",
    analysis_manifest_path="registered-analysis.json",
)
write_real_analysis_effect_report(
    "factual-track-session-report.json",
    report,
)
```

The packaged module CLI exposes the same operation:

```bash
python -m causal4d.cli.real_analysis_reporting \
  factual-track-effects.json \
  configs/causal4d/sloth_multi_action_v1.json \
  factual-track-session-report.json \
  --method-freeze method_freeze.json \
  --analysis-manifest registered-analysis.json \
  --require-estimable
```

Run it separately for each registered endpoint and primary metric. A return code
of 3 with `--require-estimable` means complete accounting was retained but fewer
than two target sessions remained estimable.

## Claim boundary

Even a positive report remains bounded to:

- the single registered physical sloth instance;
- the three registered contact regions;
- the four registered action profiles;
- the registered realization conditions; and
- the deployed perception, actuator, and physical-twin configuration.

It does not establish object-class generalization, individual-level real
counterfactual ground truth, raw-covariance calibration, or general robot safety.
Subgroup and drift diagnostics cannot rescue a failed registered endpoint.
