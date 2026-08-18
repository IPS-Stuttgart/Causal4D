# One-way production belief path v1

## Canonical claim-bearing path

The production scientific path is fixed as

```text
Prob4D observation artifact
  -> BayesianPhysTwin candidate inference and guard
  -> selected complete BayesianPhysTwin belief or exact baseline fallback
  -> Causal4D factual abduction
  -> explicit intervention
  -> counterfactual prediction.
```

Prob4D owns observation means, identities, gauge/dependence structure,
conditional and shared covariance, reliability, and causal source lineage.
BayesianPhysTwin owns whether those observations change the physical belief.
Causal4D consumes only the selected complete belief and the content-bound handoff
receipt for claim-bearing inference.

## Prohibited production shortcuts

The supported `causal4d.inference.v1` path and the BayesianPhysTwin handoff
modules must not import the external `prob4d` package or Causal4D's historical
`causal4d.prob4d_*` adapters. Those modules remain available for artifact
validation, compatibility studies, diagnostics, and frozen historical
reproduction, but they are not an alternate claim-bearing inference route.

In particular, Causal4D must not:

- recover raw Prob4D factors after BayesianPhysTwin rejected the update;
- independently recalibrate or reinterpret provider covariance to rescue a
  rejected belief;
- count a rejected observation as consumed evidence;
- let a downstream counterfactual result retroactively establish provider
  competence; or
- combine fields from the candidate with fields from the exact fallback belief.

## Exact fallback consequence

For an all-fallback handoff, Causal4D retains the exact caller-owned baseline
`TwinBelief` object and the prior evidence ledger. The receipt records zero
accepted observation-evidence consumption and
`raw_prob4d_reinterpreted=false`.

For a mixed recursive stream, only accepted BayesianPhysTwin steps enter the
Causal4D evidence ledger. Rejected steps remain explicit fallback events and do
not contribute state-update evidence.

## Executable policy

`tests/test_bpt_belief_handoff_production_path_policy.py` parses the supported
inference and handoff modules and fails if they import raw Prob4D packages or
provider-specific adapters. It also locks the provider-neutral export surface
of `causal4d.inference.v1` and the explicit no-reinterpretation receipt field.

Existing behavioral tests remain authoritative for exact object identity,
causal-prefix intervals, provider-manifest binding, duplicate-evidence
rejection, recursive accepted-step accounting, and covariance-query handoff.

Run the focused policy with:

```bash
pytest -q tests/test_bpt_belief_handoff_production_path_policy.py
```

## Scope boundary

This policy establishes ownership and software routing only. It does not prove
Prob4D provider competence, BayesianPhysTwin physical benefit, Causal4D
counterfactual benefit, uncertainty calibration, deployment safety, or state of
the art. Those remain separate empirical decisions on independently grouped
physical evidence.
