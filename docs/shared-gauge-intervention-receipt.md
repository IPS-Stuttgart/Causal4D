# Shared-gauge intervention receipts

## Purpose

A symmetry-complete state belief can leave an absolute physical gauge unresolved
while a downstream action remains well defined in the same moving or object
frame. The decision is only valid when the state orbit and the action transform
refer to **one shared group element**. Similar numerical rotations are not enough;
the transformation paths need one auditable contract and provenance chain.

This module provides that Causal4D-facing contract. It validates a finite group,
two metric representations of that group, the commanded action orbit, and a
realized-action envelope. It then emits a content-addressed receipt that can be
consumed by Prob4D's gauge-coupled regret certificate and BayesianPhysTwin's
act-or-fallback policy.

## Finite shared-gauge contract

Let `G={g_0,...,g_{K-1}}` be a registered finite group with multiplication table
`m(i,j)`. The contract validates:

1. a two-sided identity;
2. closure and valid indices;
3. one occurrence of every group element in every multiplication-table row and
   column;
4. associativity;
5. a state-space representation `R_x(g)`;
6. an action-space representation `R_u(g)`;
7. the homomorphism equations

   \[
   R_x(g_i)R_x(g_j)=R_x(g_{m(i,j)}),
   \qquad
   R_u(g_i)R_u(g_j)=R_u(g_{m(i,j)});
   \]

8. isometry in the declared state and action metrics.

The exact element ordering is part of the contract identity. Reordering the
action transformations while retaining the same set of matrices is rejected,
because it can silently couple the state to a different group element.

## Commanded action orbit

For action templates `a_1,...,a_A`, the commanded orbit must satisfy

\[
u^{\rm cmd}_{ga}=R_u(g)a_a
\]

for every registered group element and action. The receipt recomputes this orbit
from the template bank and representation. It does not trust a producer's claim
that the transform was applied correctly.

A `transform_instance_id` and separate source-evidence, action-template,
commanded-intervention, realized-intervention, loss, fallback, and radius-
provenance identifiers are hashed into the receipt. This prevents a certificate
for one state frame or one command bank from being replayed against another.

## Realized intervention envelope

Let `u_real(g,a)` be the observed or otherwise bounded realized intervention. In
the registered action metric `M_u`, the receipt computes

\[
\widehat\varepsilon_a
=
\max_{g\in G}
\sqrt{
(u^{\rm real}_{ga}-u^{\rm cmd}_{ga})^\top
M_u
(u^{\rm real}_{ga}-u^{\rm cmd}_{ga})
}.
\]

The receipt is issued only when

\[
\widehat\varepsilon_a\leq\varepsilon_a
\]

for every action, where `epsilon_a` is the declared realization radius.

Given an action-Lipschitz loss constant `K_a`, the one-action loss margin is

\[
m_a=K_a\varepsilon_a,
\]

and the pairwise regret margin is

\[
m_{ab}=m_a+m_b.
\]

This matrix is exported directly for the Prob4D realized equivariant-decision
certificate. It is symmetric and has zero diagonal because comparing an action
with itself creates no regret uncertainty.

## Radius scope

The receipt distinguishes three scopes:

- `deterministic-complete`: the supplied radius covers every group element and
  target case claimed by the caller;
- `registered-group-nodes-only`: the radius has only been checked on the finite
  registered orbit nodes;
- `externally-calibrated`: a separate statistical procedure supplies the target
  statement, identified by `radius_provenance_id`.

The receipt does not promote a node-only maximum to a continuous-group or target-
population guarantee. Such promotion requires an additional cover/Lipschitz
argument or a separately valid calibration result.

## Failure semantics

The constructor fails closed on:

- malformed or nonassociative multiplication tables;
- invalid identity or nonbijective rows/columns;
- matrices that are not group representations;
- transformations that are not isometries in the declared metric;
- mismatched state/action group order;
- a commanded orbit that does not equal the registered action transform;
- a realized intervention outside the declared radius;
- negative radii or Lipschitz constants;
- unregistered radius scope;
- missing provenance identifiers.

It does not repair a near-group, infer a better metric, enlarge a radius, reorder
an orbit, or synthesize a fallback.

## Cross-repository composition

A complete action path is:

1. **Prob4D** emits a quotient belief and complete unresolved group support.
2. **Causal4D** validates the state/action group contract and emits this shared-
   gauge intervention receipt.
3. **Prob4D** combines the ideal complete-orbit action gap with the receipt's
   pairwise realization margin.
4. **BayesianPhysTwin** executes the minimax action only when the resulting bound
   satisfies the registered tolerance; otherwise it restores the exact caller-
   owned fallback.

The important object is not a point state. It is a proof chain linking quotient
information, group support, the state/action transform, physical intervention
realization, registered loss, and fallback identity.

## Scientific boundary

This receipt proves an algebraic and supplied-envelope statement. It does not
establish that the registered group is the true physical symmetry, that a learned
provider exposes the correct orbit, that an actuator remains within the radius
on unseen targets, that a loss is globally Lipschitz with the supplied constant,
or that executing the admitted action is safe. Those are separate empirical and
statistical obligations.
