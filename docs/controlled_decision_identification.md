# Controlled decision identification

The passive sequential layer treats a diagnostic as an observation channel that
leaves the physical hypothesis fixed.  This module covers the dual-control case:
a diagnostic intervention can both change the physical state and emit an
observation.

For intervention `e`, the registered finite kernel is

```text
K_e[h, h_next, y] = P(H_next=h_next, Y=y | H=h, e).
```

The solver propagates the complete physical belief through this joint kernel.
`factorized_controlled_intervention` directly combines an action-conditioned
state-transition matrix with a next-state observation channel.  This provides a
bridge to the existing dynamic-contact transition model without treating contact
regime changes as passive sensor noise.  A
terminal action is emitted only when its regret is below the registered
tolerance for every state with positive posterior support.  Otherwise the solver
selects another intervention or returns exact fallback.

## Exact finite-horizon policy

The Bellman state contains the complete posterior belief, remaining intervention
roster, remaining horizon, and remaining additive risk budget.  Every candidate
intervention is retained only when all positive-probability observation branches
reach a certified terminal action.  Each intervention can be used at most once.
The objective can minimize expected or worst-case intervention cost.

This differs from treating a physical poke as a passive sensor.  An observation
may identify the pre-intervention state while the same intervention changes the
state that the terminal action will encounter.

## Coarsest controlled decision quotient

Terminal action-loss equivalence alone is insufficient for active control.  Let
`Pi` be a partition of the physical state space.  It is stable for the registered
controlled interface when states in the same class have:

1. identical normalized terminal action-loss vectors; and
2. for every intervention, outcome, and next class, identical probability mass
   assigned to that `(next class, outcome)` pair.

The implementation starts from terminal decision equivalence and repeatedly
refines classes until this controlled lumpability condition holds.  The fixed
point is the coarsest registered controlled decision quotient.  Aggregating
belief mass over its classes preserves every terminal certificate, controlled
belief update, finite-horizon policy, expected and worst-case cost, and additive
risk charge.

A passive decision quotient is sufficient for active acquisition if and only if
it is already stable under every registered transition-observation kernel.  The
returned witnesses identify concrete terminal-decision-equivalent states whose
controlled consequences differ.

## Controlled strict-separation study

The deterministic study contains 64 complete states.  The initially supported 32
states encode a task bit, a route bit, two nuisance bits, and one duplicated
microstate bit.  A cheap routing intervention reveals the route but also toggles
a physical polarity that changes the correct terminal action.

| Policy | First intervention | Guaranteed cost | Actual terminal loss |
|---|---|---:|---:|
| Generic mutual information | nuisance four-way | not decision sufficient | -- |
| One-step controlled value | global effective-action test | 0.50 | 0 |
| Cheapest fixed sufficient sequence | global effective-action test | 0.50 | 0 |
| **Two-step controlled policy** | **route toggle** | **0.35** | **0** |
| Static observation approximation | route toggle | 0.35 | **1 on every supported state** |

The route toggle has no immediate decision value.  It becomes useful because its
outcome selects one of two cheap local tests.  The exact adaptive controlled
policy reduces guaranteed cost by 30% relative to both one-step and fixed
acquisition.

The static approximation retains exactly the intervention outcome marginals but
ignores the state transition.  It chooses the same acquisition tree and then
selects the opposite terminal action on every initially supported state.  This
isolates why state-changing physical probes require controlled belief dynamics.

The passive terminal decision quotient has two classes and fails controlled
lumpability.  The controlled quotient has 32 classes, removes the duplicated
microstate bit, and reproduces the complete adaptive policy and costs exactly.
Every terminal action is certified while eight complete physical states remain
compatible.

## Scientific boundary

The result is exact only for the supplied finite physical-state roster, terminal
losses, transition-observation kernels, costs, additive risk charges, regret
tolerance, and horizon.  It does not validate a real physical support, learned
transition model, sensor likelihood, target-domain transport, deployment
competence, or safety.

The decisive empirical successor is a prospective hidden-contact-topology task
where a gentle first interaction changes contact state and determines which
branch-specific second diagnostic is needed before a consequential pull or
release action.
