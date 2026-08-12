# Recursive BayesianPhysTwin belief handoff

## Scope

`causal4d.recursive_bpt_belief_handoff` is an additive ownership boundary for a
completed BayesianPhysTwin recursive Prob4D stream run. It consumes the selected
complete BayesianPhysTwin belief; it does not reopen or reinterpret the raw
Prob4D factor bundles.

The existing `causal4d.belief_provider_v2_contract` remains the narrow contract
for model-averaged endpoint and horizon-discrepancy use. The separate
`causal4d.belief_provider_v2_recursive_contract` requires the complete recursive
provider-v2 surface:

- append-only Prob4D stream validation;
- complete-belief candidate or exact-prior routing;
- persistent recursive nuisance policy;
- fixed provider, calibration, and runtime identities;
- explicit posterior-covariance semantics; and
- immutable stream-step and stream-run records.

No existing provider or handoff API changes behavior.

## Handoff

A claim-bearing caller first completes a BayesianPhysTwin
`ClaimBearingProb4DStreamRunV1`. It then converts the selected complete belief to
one Causal4D `TwinBelief` with the same causal context, particle identities,
physical-parameter support, and state shape.

```python
from causal4d.recursive_bpt_belief_handoff import (
    bind_recursive_bayesian_phystwin_belief_handoff,
)

bound = bind_recursive_bayesian_phystwin_belief_handoff(
    stream_run,
    selected_bpt_belief,
    baseline_belief=causal4d_baseline,
    candidate_belief=causal4d_candidate,
    prob4d_source_revision=prob4d_revision,
)
```

The selected BayesianPhysTwin artifact identity must equal
`stream_run.final_belief_id`. The run and every step must bind the installed
recursive provider manifest, one recursive nuisance policy, one covariance
policy, one calibration-artifact set, and independently verified runtime
revision evidence.

The Causal4D candidate may contain updated state, discrepancy, and particle
weights. It may not relabel particles, change physical-parameter values, change
the endpoint, change the causal context, or change the state shape.

## Evidence ownership

Each accepted stream member appends exactly one `state_update` record to the
existing `ConsumedEvidenceLedgerV1`. The record binds:

- the stream, run, step, and stream-update identities;
- the claim-bearing update and underlying observation-factor identities;
- the observation binding and physical linearization;
- guard, selection, covariance-semantics, covariance-policy, and nuisance-policy
  identities; and
- the member's half-open causal frame interval.

Rejected members append nothing. Their exact fallback status remains visible in
the recursive run and handoff receipt, but a factor that did not enter the final
posterior is not recorded as consumed posterior evidence.

The ledger rejects duplicate updates, duplicate raw factors, relabelled source
bytes, and incompatible cross-stage use. Several accepted recursive members may
share one correlation group because BayesianPhysTwin has already evaluated them
through one explicitly persistent nuisance policy.

## Exact fallback

When every stream member falls back, all of the following are required:

- the selected BayesianPhysTwin belief equals the stream's initial belief;
- the supplied Causal4D candidate is the exact baseline artifact;
- the prior evidence ledger is unchanged;
- zero Prob4D factors are consumed; and
- the returned Causal4D belief is the original baseline object.

When at least one member is accepted, the returned belief embeds the complete
ownership ledger and a compact recursive handoff descriptor. The
content-addressed receipt separately lists accepted and fallback step IDs and
binds the selected BayesianPhysTwin and delivered Causal4D belief identities.

## Causal interval

Every admitted member must lie inside the Causal4D pre-intervention observation
window. Member intervals must be ordered and non-overlapping. A member that
begins before the declared prefix, crosses its exclusive stop, or overlaps a
preceding member fails before the Causal4D belief or ledger is changed.

## Scientific boundary

The recursive provider contract and handoff establish software compatibility,
identity, causal-prefix integrity, complete-belief routing, covariance-policy
provenance, exact fallback, and evidence ownership. They do not establish:

- real Prob4D provider competence;
- calibrated predictive uncertainty;
- physical-state or intervention identifiability;
- improvement on a fresh object or acquisition session;
- Causal4D counterfactual benefit;
- deployment safety; or
- state of the art.

This path is prospective infrastructure. It does not modify the frozen
18-session/36-execution estimator, admit Prob4D into that primary method, open a
target outcome, or increment physical evidence.
