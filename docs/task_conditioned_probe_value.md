# Task-conditioned probe value for causal physical twins

**Status:** experimental finite-hypothesis kernel and controlled mechanism study.
This does not change the retained negative result of
`active_causal_belief_adaptivity.md`, authorize a real probe, or establish
provider, calibration, physical, or control benefit.

## Motivation

The existing active-belief study chooses a safe probe by expected entropy
reduction over the complete latent hypothesis label. Its retained result is
negative: resolving more total latent uncertainty did not improve the
post-probe challenge prediction. That result exposes a sharper question.

A probe should be valuable only when its outcome changes the posterior over
latent distinctions that matter to a **registered downstream query or
decision**. Generic mutual information can instead be dominated by nuisance
variables that are easy to observe but irrelevant to the challenge.

For finite physical hypotheses \(h\), prior masses \(w_h\), a candidate probe
\(a\) with outcomes \(y\), and a vector query \(q_h\), the module computes

\[
R_Q(w)
  = \min_{\widehat q}
      \sum_h w_h
      (q_h-\widehat q)^\top M(q_h-\widehat q)
\]

and

\[
V_Q(a)
  = R_Q(w)
    - \sum_y p(y\mid a)\,
      R_Q\!\left(w(\cdot\mid y,a)\right).
\]

For a finite registered decision set with loss \(L(d,h)\), it analogously uses

\[
V_D(a)
  = \min_d \sum_h w_hL(d,h)
    - \sum_y\min_d\sum_h
      w_hp(y\mid h,a)L(d,h).
\]

These are classical finite Bayesian value-of-information calculations. The
candidate contribution is not their invention. The purpose here is to make
task relevance, prospective physical risk, generic information, cost, and exact
no-probe fallback distinct objects in the Causal4D evidence chain.

## Software boundary

`causal4d.task_conditioned_design` provides:

- strict finite probe likelihood contracts;
- exact expected squared-query Bayes risk;
- exact finite-decision Bayes risk;
- mutual information in nats as a separate baseline;
- a prospective scalar physical-risk cap;
- deterministic task, decision, or information selection;
- exact no-probe fallback when no safe positive-value probe exists; and
- weight-preserving hypothesis-payload permutations for attribution controls.

The module consumes already registered likelihoods, query values, losses,
risks, and costs. It does not estimate any of them. It does not execute a
physical intervention, construct a BayesianPhysTwin belief, or decide whether a
Prob4D provider is valid.

A weight-preserving permutation is a diagnostic intervention on the
hypothesis-to-query alignment. It preserves the weighted marginal query values
and probe likelihoods but is not itself a physical model. It is useful only
when the declared permutation removes the coupling whose role is being tested.

## Controlled mechanism

The frozen study has eight equiprobable hypotheses formed by one binary
task-relevant latent variable and one four-class nuisance variable. The
registered downstream query is a two-coordinate vector
\((-50,-20)\) or \((50,20)\) mm, and the registered finite decision chooses the
negative or positive challenge action.

Four candidate probes are fixed before source or target simulation:

| Probe | Information carried | Prospective risk |
|---|---|---:|
| `nuisance-rich` | Four-class nuisance, 97% correct | 0.010 |
| `target-moderate` | Task sign, 80% correct | 0.015 |
| `target-risky` | Task sign, 98% correct | 0.080 |
| `uninformative-safe` | None | 0.000 |

The risk cap is 0.020. Consequently, the most task-informative probe is
inadmissible. Among safe probes, `nuisance-rich` has much greater mutual
information about the full hypothesis than `target-moderate`, but exactly zero
query and decision value. The registered task criterion selects
`target-moderate`.

The analytic controls are:

- prior query Bayes risk: \(2900\ {\rm mm}^2\);
- expected risk after `target-moderate`: \(1856\ {\rm mm}^2\);
- query value: \(1044\ {\rm mm}^2\);
- finite decision risk: \(0.5\rightarrow0.2\);
- decision value: \(0.3\);
- `nuisance-rich` query and decision value: exactly zero;
- `target-risky` query value: \(2672.64\ {\rm mm}^2\), but rejected by risk;
- generic information: `nuisance-rich` \(1.21859\) nats versus
  `target-moderate` \(0.192745\) nats.

A fixed weight-preserving permutation changes the task sign to the product of
the original task sign and nuisance parity. The task marginal remains four
positive and four negative hypotheses, while it becomes conditionally balanced
given either individual probe. All safe one-probe task values therefore
collapse to zero, and the selector returns exact no-probe fallback. This
attributes the positive value to the probe--query dependence rather than to the
probe marginal or query marginal alone.

## Source gate and target panel

The experiment first runs a source simulation. It verifies the analytic
ordering, risk rejection, dependence-control collapse, and predefined empirical
margins against the generic-information policy. Only then does the code generate
the separately seeded target panel.

Each episode samples one latent physical hypothesis and one probe outcome.
That episode—not latent coordinates, outcomes, hypotheses, or posterior
entries—is the statistical unit. All policies share the same latent episode
stream within a seed. Identical selected probes share the same outcome stream,
so the task-query and task-decision policies must produce identical summaries.

The target evaluation measures actual posterior query prediction error and
finite decision loss after the selected observation. It does not evaluate a
predicted covariance reduction against itself. The controlled simulator is
correctly specified, however, so a positive result establishes the mechanism
under known likelihoods rather than learned real-world performance.

## Reproduction

```bash
python -m pytest -q tests/test_task_conditioned_design.py

python scripts/experiments/task_conditioned_probe_value.py \
  --source-revision "$(git rev-parse HEAD)" \
  --output build/task-conditioned-probe-value/result.json

python scripts/ci/verify_task_conditioned_probe_value.py \
  build/task-conditioned-probe-value/result.json \
  --output-json build/task-conditioned-probe-value/verification.json
```

The workflow retains the result and verification as one artifact. Its
verification checks the source-before-target gate, exact analytic values,
policy ordering, risk rejection, dependence collapse, and predefined target
effect margins.

## Relation to the broader paper program

This controlled study addresses the conceptual failure exposed by the existing
active-belief negative: information about the complete latent label need not be
information about the challenge. It is a methodological bridge between:

1. Prob4D's dependence-bearing observations;
2. BayesianPhysTwin's complete uncertain physical belief and exact fallback;
3. Causal4D's candidate interventions and registered held-out queries.

The next empirical milestone is not another synthetic variant. After a provider
passes a separately frozen source qualification, construct the joint predictive
law of one candidate probe outcome and one held-out physical query from matched
posterior twin draws. Compare task-conditioned value, generic information,
fixed-safe probing, and no-probe fallback on independent objects or episodes.
Destroy only the matched probe--query dependence for the attribution control
while retaining their marginals. Evaluate held-out query proper scores,
registered decision regret, accepted-probe harm, and worst-group regret.

That future result would support a bounded claim that preserving physical
dependencies changes intervention choice and downstream value. The present
study supports only the finite-hypothesis mechanism and implementation.
