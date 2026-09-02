# PokeFlex active-probe protocol v2

## Scientific question

Can one source-trained, task-conditioned policy select a single real diagnostic
poke whose measured response improves prediction of a disjoint held interaction
on a previously unopened physical object, beyond no probing, a source-fixed probe,
and generic information maximisation?

The intended contribution is not merely that another observation helps. The
experiment tests whether **which** physical interaction should be observed depends
on the registered downstream query.

## Three safeguards added in v2

### Reset-compatible logged transport

Every probe/challenge pair must pass the separately source-frozen initial-state
matching certificate. This constrains the logged surrogate to interactions from the
same object whose earliest dynamic meshes lie inside a source-derived reset envelope.
It does not assert identical microscopic state.

### Primary and replication panels

A salted object split creates nine source objects, six primary target objects and
three replication target objects. No target object is used for hyperparameter or
threshold selection. All source choices are made by nested leave-one-source-object-
out analysis.

Predictions for all nine target objects are generated under one frozen method and
bound by one joint prediction seal before the primary outcomes are scored. The three
replication objects are scored only after the six-object primary gate passes. No
prediction may change between panels.

### Matched relation-breaking controls

Every non-oracle policy receives the same candidate pokes, source data, initial-state
information, physical cost and risk estimates, and a budget of one revealed response.
The controls are:

- no probe;
- random safe probing;
- one source-fixed safe probe;
- generic mutual information;
- an action-only query heuristic;
- within-object dependence destruction;
- a reset- and action-matched response from the wrong object;
- a double placebo breaking both object and query dependence.

The oracle is diagnostic only and cannot contribute to a gate.

## Task-specificity requirement

The method must choose different diagnostic pokes for different registered queries.
A source gate requires at least 25% query-conditioned switching, and the primary
panel requires at least one third. Without this test, a favorable result could be
explained by discovering one universally good poke rather than active query-directed
experiment design.

## Registered queries

1. local response under a disjoint held poke;
2. impact geometry under a held drop;
3. settled geometry under that held drop.

The poke-to-drop queries are cross-intervention logged surrogates. They are not
individual-level counterfactual observations.

## Custody order

1. Scan all retained repository and workflow evidence for historical exposure.
2. Freeze the nine-source/six-primary/three-replication split.
3. Verify reset compatibility using earliest meshes only.
4. Qualify the task policy and every threshold on source objects only.
5. Select one response per policy and query without reading response payloads.
6. Reveal only selected responses and produce all nine target prediction panels.
7. Bind those panels in one joint seal.
8. Score the primary six objects once.
9. Score the replication three only if the primary gate passes.
10. Retain a positive or negative terminal result; target retry is forbidden.

## Inferential unit

The physical object is the primary unit. Frames, vertices, queries and policy rows
are repeated measurements and cannot be treated as independent samples. Results are
reported per query and in aggregate, with paired object bootstrap intervals and an
exact object sign test.

## Claim enabled by a complete positive result

> On previously unopened physical objects, a source-frozen task-conditioned policy
> selects one real diagnostic poke and uses only its revealed response to improve
> prediction of disjoint held interactions relative to matched fixed and generic-
> information probes. The advantage survives a separately scored replication panel
> and collapses under object- and query-dependence placebos.

## Boundary

This is an offline logged active-probing experiment. It cannot establish online
closed-loop execution, identical microscopic resets, an individual counterfactual,
deployment safety, unique material identification, or general state of the art.
