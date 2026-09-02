# Counterexample-guided continuous decision certification

The finite decision-identification layers are exact only for a supplied finite
hypothesis roster. A grid of material, contact, or alignment parameters is not a
certificate: a decision-breaking physical world may lie between sampled nodes.
This module replaces the finite roster by a compact continuous parameter box and
verified loss-gap envelopes.

## Registered interface

Let `theta` lie in one axis-aligned box `Theta`, let `loss(theta, a)` be the
registered terminal loss of action `a`, and let `K[a]` be a valid global
Lipschitz constant in the parameter `L-infinity` metric:

```text
|loss(theta, a) - loss(theta', a)| <= K[a] ||theta - theta'||_infinity.
```

For a box with center `c` and radius `r`, every pairwise action-loss gap obeys

```text
loss(theta, a) - loss(theta, b)
<= loss(c, a) - loss(c, b) + (K[a] + K[b]) r.
```

Taking the maximum over competitors gives a sound upper bound on the worst-case
regret of action `a` throughout that box. Evaluating the center gives a concrete
lower-bound witness. The implementation never promotes a finite grid into
continuous support.

## Counterexample-guided branch and bound

`certify_continuous_decision` starts from the complete registered parameter box
and repeatedly splits the region carrying the largest unresolved regret excess.
For every action it maintains:

- a witnessed lower bound from evaluated physical parameter points; and
- a verified upper bound covering every still-active parameter box.

The routine terminates in one of four states:

1. `certified`: exactly one action has a verified upper regret below tolerance
   and every other action has a concrete violating parameter;
2. `no-admissible-action`: every action has a concrete counterexample;
3. `multiple-admissible-actions`: at least two actions are uniformly within the
   registered tolerance; or
4. `inconclusive`: the evaluation, depth, or resolution budget is exhausted.

Only the first state emits an action. Every other state routes to the caller's
exact fallback. Search failure is therefore not confused with physical
admissibility.

## Controlled strict-separation study

The deterministic mechanism uses a continuous physical coordinate `x` in
`[-1, 1]`. Action 0 is preferable almost everywhere but has a narrow
high-regret pocket centered at `x = 0.375`; action 1 has a small constant loss.
A three-node grid at `{-1, 0, 1}` misses the pocket, reports zero empirical
worst-case regret for action 0, and selects it.

The continuous certificate instead:

- finds the decision-breaking world at `x = 0.375`;
- witnesses regret `0.9` for action 0;
- verifies that action 1 has worst-case regret below the registered `0.15`
  tolerance; and
- certifies action 1.

A source-qualified observation then restricts the relevant coordinate to
`[-1, 0.2]` while leaving a full nuisance coordinate in `[-1, 1]` unresolved.
Using a source-qualified local Lipschitz envelope on that restricted support, the
certificate selects action 0 at tolerance `0.05`. Thus additional evidence
changes the justified decision without identifying the complete physical state.

A one-evaluation control returns `inconclusive` and exact fallback rather than a
partial certificate.

This establishes:

```text
finite-grid success != continuous-support certification
counterexample elimination != complete-state identification
optimizer exhaustion != action admissibility
```

## Relation to controlled interventions

The state-changing intervention planner currently operates on finite physical
support. The continuous certificate is the terminal support oracle required for
a counterexample-guided dual-control extension:

1. search the continuous physical model class for a world that breaks the
   current terminal action;
2. if none exists under verified envelopes, act;
3. otherwise choose an intervention whose possible outcomes eliminate or
   separate the surviving counterexamples;
4. recompute the support certificate; and
5. fall back whenever verification remains inconclusive.

The present module implements steps 1, 2, 4, and 5. Coupling verified continuous
support refinement to the merged controlled transition-observation planner is
the next algorithmic layer.

## Scientific boundary

Soundness is conditional on the supplied parameter domain, deterministic loss
oracle, and valid global or source-qualified local Lipschitz constants. The
module does not validate those physical ingredients, establish that a learned
simulator bounds reality, prove target-domain transport, authorize deployment,
or provide a safety guarantee.
