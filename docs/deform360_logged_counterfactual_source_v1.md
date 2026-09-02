# Deform360 logged cross-intervention abduction v1

## Question

Can one real interaction of a deformable object identify persistent latent
physics that improves prediction under a *different* recorded intervention on
the same object?

This is the public-real-data bridge between Causal4D's exact controlled
abduction--intervention--prediction operator and its existing one-action
Deform360 forecast.  It deliberately does **not** rename separate logged trials
as individual-level counterfactual ground truth.

## Real-data semantics

For a factual source interaction `A`, the existing frozen 200-member reduced
rope-dynamics bank is scored on the complete source outcome.  These factual-only
scores produce a generalized-Bayes posterior over the persistent dynamics-bank
index.  For a distinct challenge source interaction `B`:

- the persistent posterior from `A` is retained;
- `B` supplies its own measured controller trajectory;
- `B` supplies its own registered contact schedule, contact node, and offset;
- no event-specific contact quantity from `A` is copied into `B`; and
- the candidate rollouts under `B` are mixed using the abducted posterior.

Thus the causal structure is

```text
real A outcome -> abduct persistent physics -> do(action B) -> predict real B future
```

while the event-specific intervention realization is fresh for `B`.

## Attribution controls

The primary controls are designed to isolate information carried by the
*factual-to-physics identity mapping*.

1. `uniform_physics` evaluates the exact same candidate rollout bank under `B`
   but ignores `A`.
2. `candidate_id_permuted` keeps the complete factual posterior weight multiset
   unchanged and keeps the complete `B` rollout bank unchanged, but applies the
   weights to different candidate identities using a deterministic nonzero
   cyclic shift.
3. `persistence` repeats the challenge prefix endpoint.

If factual abduction helps but the candidate-ID permutation removes the gain,
the improvement cannot be attributed merely to posterior concentration or to
the challenge rollout marginal.

## Statistical unit and frozen source gate

The five quality-passing source episodes from the published `001-rope` protocol
are used.  For the primary panel, each challenge episode receives exactly one
distinct factual partner selected by a SHA-256 rule that does not inspect any
trajectory outcome.  The challenge episode is therefore the primary source
unit.  All 20 ordered distinct pairs are retained as a secondary mechanism
matrix, not treated as 20 independent physical units.

The source gate requires, on the five primary challenge units:

- at least 2% mean Chamfer improvement over `uniform_physics`;
- at least 2% mean Chamfer improvement over `candidate_id_permuted`;
- at least 60% challenge wins against each control; and
- no challenge worse than 1.25x either control.

The exact configuration is frozen in
`protocols/deform360_logged_counterfactual_source_v1.json`.

## Evidence boundary

This first stage is source-only.  It may read complete futures of the five
already-open source interactions, but it does not reuse the historical
`001-rope` target as a fresh confirmation and it does not inspect the protected
six-object replication cohort.  A positive source gate authorizes only the
*registration* of a separate target-closed study on additional public
multi-interaction objects.

A later positive multi-object result could support the bounded claim that
persistent physical information abducted from one real interaction transfers
across an intervention change.  Even then, repeated logged executions are not
literal individual-level counterfactual ground truth unless identical exogenous
conditions are independently established.
