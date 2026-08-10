# Strict BayesianPhysTwin belief handoff

`causal4d.bpt_belief_handoff` is the versioned ownership boundary between a
strict claim-bearing BayesianPhysTwin update and a Causal4D `TwinBelief`.

The boundary is additive. Existing fixed-anchor, horizon-discrepancy, replay,
graph, registered-query, and physical-evaluation providers remain unchanged.

## Accepted update

An accepted handoff requires all of the following:

1. a `ClaimBearingTreeBlockProb4DUpdateV1`;
2. a baseline Causal4D belief;
3. a candidate belief with the same causal context, endpoint, particle roster,
   physics parameters, and particle weights;
4. a `ValidatedTreeBlockQueryCovarianceV1` whose update, tree-block result,
   inference status, and reason exactly match the claim-bearing update; and
5. an evidence ledger whose protocol, case, and causal prefix match the belief.

The handoff then:

- appends exactly one `state_update` consumption to the ledger;
- binds the Prob4D observation, linearization, provider-manifest, calibration,
  admission, update, and tree-block-result identities;
- consumes the registered query covariance exactly once;
- embeds the resulting evidence ledger in the delivered belief;
- embeds a compact handoff descriptor in the belief metadata; and
- creates a content-addressed
  `BayesianPhysTwinBeliefHandoffReceiptV1`.

Causal4D receives only the registered posterior query and identities. It does not
receive or reinterpret raw Prob4D observation rows, gauge factors, or a complete
joint covariance.

## Rejected update

A rejected handoff is stricter:

- no registered observation covariance may be supplied;
- no Prob4D evidence is appended to the ledger;
- the candidate must be the exact baseline belief artifact;
- the returned belief is the original baseline object; and
- the receipt records zero evidence consumption, zero covariance consumption,
  and exact baseline retention.

This preserves the fail-closed BayesianPhysTwin semantics through the Causal4D
boundary rather than merely recording a rejection reason.

## Evidence ownership

The accepted state update uses the existing `ConsumedEvidenceLedgerV1`. The
claim-bearing update identity is the evidence identity and the underlying
Prob4D observation artifact is the raw-factor identity. Existing ledger checks
therefore reject:

- reuse of the same update;
- reuse of the same observation factor under a different update label;
- relabelling identical observation bytes;
- multiplication of correlated state and intervention evidence; and
- evidence that crosses the declared causal prefix.

Use:

```python
from causal4d.bpt_belief_handoff import (
    bind_bayesian_phystwin_belief_handoff,
)

bound = bind_bayesian_phystwin_belief_handoff(
    update,
    baseline_belief=baseline,
    candidate_belief=bayesian_phystwin_belief,
    query_covariance=validated_registered_query,
    prob4d_source_revision=prob4d_revision,
)
```

The returned `BoundBayesianPhysTwinBeliefV1` contains:

- `belief`: the delivered Causal4D belief;
- `evidence_ledger`: the complete ownership ledger; and
- `receipt`: the content-addressed handoff receipt.

`consumed_evidence_ledger_from_twin_belief()` recovers the exact embedded ledger.
For an unbound belief it returns the corresponding empty ledger.

## Separate uncertainty reductions

The receipt carries two separately named quantities:

- `bpt_truncation_mass`: probability mass removed or summarized inside the
  BayesianPhysTwin belief representation;
- `causal4d_support_reduction_mass`: support mass removed later by a Causal4D
  intervention or query policy.

Both must be finite values in `[0, 1]`. They are never combined or relabelled as
one quantity.

## Optional-provider boundary

The handoff module is intentionally not imported by the package root. Importing
core Causal4D therefore does not import or require BayesianPhysTwin. The provider
type is resolved only when `bind_bayesian_phystwin_belief_handoff()` is invoked,
so core-only installations retain the existing dependency boundary while an
installed BayesianPhysTwin wheel is still checked at the actual handoff point.

## Scientific boundary

The receipt establishes implementation, content identity, causal-prefix
integrity, evidence ownership, registered-query covariance ownership, and exact
fallback behavior. It does not establish observation competence, empirical
uncertainty calibration, physical-query benefit, Causal4D intervention benefit,
deployment safety, generalization, or state of the art.
