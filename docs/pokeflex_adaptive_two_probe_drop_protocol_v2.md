# PokeFlex adaptive two-probe drop-query protocol v2

**Status:** registered before source response-model fitting and before any
primary target drop outcome is opened.

## Scientific question

Can a physical twin acquire **less than two logged diagnostic interactions on
average**, adapt the identity of a second interaction to the response of the
first, and improve a sealed cross-intervention drop query over both:

1. the best registered one-probe policy; and
2. the best source-fixed two-probe set?

This is the public-real-data precursor to sequential decision-identifiable
physical twins. It tests adaptive acquisition of recorded physical responses.
It does not claim that the two pokes were executed online as one physical
sequence.

## Why this is stronger than protocol v1

A one-probe result can still be explained as ordinary feature selection: find a
generally useful diagnostic interaction and reuse it. Protocol v2 requires a
strictly harder separation:

```text
best single probe
    !=
best fixed two-probe set
    !=
response-conditioned two-stage acquisition
```

The proposed policy must choose a first probe, inspect only that selected
response, and then either stop or choose a different second probe. Its advantage
must survive matched probe budgets, source-frozen risk limits, and a generic
information baseline.

## Evidence class

The diagnostic-poke panel is retrospective because historical PokeFlex poke
responses have already been broadly inspected in the larger project. The
primary drop challenge is called prospective only if a separate exact exposure
scan verifies that every registered target drop outcome remained unopened.

The acquisition is **offline adaptive logged acquisition**. First- and
second-probe responses come from separately recorded reset interactions. The
study therefore does not establish online sequential execution or identical
microscopic state between the two probes.

## Frozen object split

The metadata audit established 18 eligible objects. The family-stratified split
is inherited unchanged from the target-blind v1 registration:

| Role | Printed | Foam | Soft |
|---|---|---|---|
| Target | `3dPrintedBunny`, `3dPrintedCylinder` | `Sponge`, `MemoryFoam` | `Beanbag`, `Pillow` |
| Calibration | `3dPrintedPizza` | `FoamCylinder` | `PlushOctopus` |
| Source | `3dPrintedHeart`, `3dPrintedPyramid` | `FoamDice`, `FoamHalfSphere`, `ToiletPaperRoll` | `PlushDice`, `PlushMoon`, `PlushTurtle`, `PlushVolleyball` |

No target may be replaced after any target response or drop outcome is opened.

## Candidate library and horizon

Exactly four complete poking takes form each target object's candidate library.
The ordering is frozen by SHA-256 under
`PokeFlex-adaptive-two-probe-library-v2-2026-09-02`.

The acquisition horizon is two:

1. choose and seal the first probe identity without reading any target probe
   response;
2. reveal only the selected first response;
3. conditionally stop or seal one different second probe identity;
4. reveal only that second response when selected;
5. seal all drop-query predictions;
6. open target drop outcomes once for scoring.

At no stage may an unselected target probe response be searched, decoded, or
used for model selection.

## Adaptive objective

Let `q` denote one registered drop query and `e_1` the first diagnostic poke.
After observing response summary `y_1`, the policy may stop or choose a second
probe `e_2 != e_1`. Source and calibration objects define the estimated
cost-adjusted query risk

```text
estimated query loss
+ lambda * cumulative normalized probe cost
```

subject to a cumulative source-predicted physical-risk cap.

A branch stops after zero or one probe whenever every safe remaining probe has
nonpositive estimated net query value. Thus the policy is not rewarded merely
for consuming the full two-probe budget.

## Registered queries

Two geometry-invariant queries avoid requiring persistent material vertex
identity.

### Drop impact geometry

- maximum template-diagonal-normalized bounding-box compression;
- time to maximum compression;
- rebound ratio.

### Drop settled geometry

- final normalized symmetric Chamfer distance to the initial mesh;
- final bounding-box compression;
- final centroid drift.

The primary per-query loss is source-standardized squared error.

## Baselines

The primary comparison set is:

1. `no-probe`;
2. `task-conditioned-best-single-probe`;
3. `source-fixed-two-probe-set`;
4. `generic-information-adaptive-two-probe`;
5. `task-conditioned-adaptive-two-probe`;
6. `dependence-destroyed-adaptive-two-probe`.

Additional random and source-fixed single-probe controls are retained, and an
oracle is diagnostic only.

The best-single policy uses the same candidate library, risk cap, source model,
and query as the adaptive method but may reveal at most one response.

The fixed-pair baseline chooses one safe unordered pair from source and
calibration objects and always reveals both responses for every target object
and both queries. It therefore controls for receiving two physical
measurements without adaptive routing.

The generic-information baseline has the same horizon, cost, and risk budget,
but optimizes full latent information rather than the registered query.

## Dependence falsification

The dependence-destroyed policy preserves probe-response marginals while
permuting the object-matched relation between probe responses and later drop
queries under a frozen source-only derangement.

A positive mechanism claim requires at least half of the adaptive advantage over
the best-single policy to disappear under this control.

## Source gate

No target probe is selected unless calibration objects establish all of the
following:

- lower mean loss than no probing for both queries;
- lower mean loss than the best registered single probe for both queries;
- lower cost-adjusted loss than the source-fixed two-probe set;
- lower cost-adjusted loss than generic-information adaptive acquisition;
- fewer than two revealed probes on average;
- at least two branch-dependent second-probe identities;
- different acquisition trees for impact and settled queries on at least one
  calibration object;
- at least half of the adaptive-over-single advantage removed by dependence
  destruction;
- exact stopping or no-probe fallback on every unsafe or nonpositive-value
  branch.

A failed source gate terminates v2 before target response access.

## Confirmatory target gate

The primary target panel contains six physical objects. A positive v2 claim
requires:

- positive object-mean raw query-loss gain over no probing and the best single
  probe;
- positive object-mean cost-adjusted gain over the fixed two-probe set and the
  generic-information policy;
- six of six object-level wins over the best single probe;
- six of six cost-adjusted object-level wins over the fixed pair;
- positive lower 95% paired-object-bootstrap bounds for both decisive
  comparisons;
- exact one-sided six-object sign probability no larger than `0.015625`;
- fewer than two probes on average;
- use of the second stage on 25--75% of target object-query decisions;
- branch-dependent second-probe selection on at least three target objects;
- query-dependent acquisition trees on at least two target objects;
- at least half of the adaptive advantage removed by dependence destruction.

A failed target gate is retained as a negative result. No target-side retuning,
replacement, or second confirmatory attempt is authorized.

## Statistical unit and efficiency

The physical object is the inferential unit. Drop repetitions, queries, frames,
vertices, and query coordinates are nested observations and are never counted
as independent samples.

The analysis reports raw query loss, cost-adjusted loss, mean probe count,
cumulative normalized probe cost, and the fractions of decisions stopping after
zero, one, and two probes.

## Claim boundary

A positive result supports:

> Offline response-conditioned acquisition of at most two separately logged
> physical pokes improves sealed cross-intervention drop queries over the best
> registered single probe and a source-fixed two-probe set.

It does not establish online sequential execution, identical reset states,
same-state counterfactual outcomes, unique material identification, calibrated
full-state uncertainty, deployment safety, or closed-loop manipulation success.

The machine-readable owner is
`configs/causal4d_public/pokeflex_adaptive_two_probe_drop_protocol_v2.json`.
Its canonical SHA-256 is `698c2b5d8f41527f14868ca6f35268637990b8c23942fb224f998ba70f76f1ea`.
