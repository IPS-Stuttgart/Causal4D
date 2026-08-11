# Registered real-analysis intervals

The real Causal4D endpoints are aggregated by independent target grasp session.
Before any confirmatory physical execution, a target-free operating-characteristic
audit showed material undercoverage of the previously registered unstudentized
percentile interval at the exact endpoint sample sizes of 12 and 18 sessions.
The content-addressed amendment in
`configs/causal4d/real_analysis_interval_amendment_v1.json` therefore changes the
reporting rule without changing the estimator, protocol units, target split,
exclusion policy, or target-access boundary.

The registered interval policy is:

1. **Primary:** deterministic session-clustered bootstrap-t, with 20,000
   resamples, seed `20260726`, and 95% confidence.
2. **Required robustness:** Student-t interval for the same equal-session mean.
   It may veto a positive claim but can never rescue a primary failure.
3. **Historical sensitivity:** the original session percentile interval. It is
   retained for continuity and cannot change the primary decision.

A positive interval claim requires strictly positive lower bounds from both the
bootstrap-t and Student-t intervals. A non-estimable bootstrap-t interval, a
bootstrap-t interval containing zero, or a Student-t interval containing zero
precludes a positive claim. The complete negative or bounded result remains
reportable.

## Target-free evidence

All interval-selection studies evaluated the immutable implementation at
`fa6a64b2442474321e453e9e8fdccd591e0a282d` and used no physical target outcome.
The compact, content-addressed evidence record is retained at
`runs/causal4d_real_analysis_interval_v1/operating_characteristics.json` and is
bound by both the amendment and method freeze.

### Percentile-bootstrap operating characteristics

- workflow run: `31091137654`;
- audit ID:
  `7dbea2a9b99cbc98acd03fa28af9583f0e95d4d0772e58853af4f05d0584267a`;
- ten distribution/sample-size scenarios;
- 2,000 synthetic session panels per scenario;
- 20,000 bootstrap resamples with seed `20260726` per panel; and
- exact production-implementation parity.

For Gaussian session effects, nominal 95% percentile coverage was approximately
90.9% at 12 sessions and 93.1% at 18 sessions. Coverage was lower under strong
right skew.

### Interval-method comparison

- workflow run: `31091652355`;
- audit ID:
  `5a13c416d7efd522f5123f98afacaacd218838583d78256d463eeb5e1d478576`;
- 15,000 common synthetic panels; and
- percentile, basic, Student-t, BCa, and bootstrap-t intervals evaluated on the
  same panels and resamples.

Bootstrap-t had the smallest mean absolute coverage error, `0.019`, and the best
worst-case absolute coverage error, `0.042`. Student-t had the smallest maximum
favorable one-sided type-I error, approximately `0.0267`. These target-free
results justify the registered primary-plus-veto rule; they are not physical
evidence.

## Content binding

The method-freeze manifest records the amendment contract and exact file bytes.
The schema-3 registered-analysis manifest independently binds:

- the repository-relative amendment path;
- amendment ID;
- exact SHA-256 and byte count; and
- the complete closed amendment contract.

Sealing or validating the analysis reopens the checked-in amendment file. A
changed method, seed, confidence level, role, evidence identifier, target-use
flag, or consistently re-addressed policy fails closed.

## Build the verification artifact

```bash
python -m causal4d.cli.real_analysis_interval_diagnostics \
  effects.json \
  configs/causal4d/sloth_multi_action_v1.json \
  interval-diagnostics.json \
  --method-freeze method-freeze.json \
  --analysis-manifest registered-analysis.json
```

The command rebuilds the complete source-verified primary report and independently
recomputes all three intervals through the shared interval implementation. It
fails when the report and recomputation differ. The output records the registered
intervals, decision gate, amendment identity, workflow evidence, and unchanged
same-object claim boundary.

Use `--overwrite` only to regenerate the same derived output path intentionally.
The verification artifact cannot revise the registered interval policy or select
another method after target access.
