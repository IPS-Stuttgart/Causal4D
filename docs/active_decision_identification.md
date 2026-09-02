# Active decision identification

`causal4d.active_decision_identification` implements the missing one-step
`act / probe / fallback` control flow for finite, decision-identifiable physical
twins.

The module consumes the exact query-decision certificate already validated by
`causal4d.decision_identifiable_intervention`. It does not construct a physical
quotient or infer a probe model.

## Control rule

Given a current certificate, the router first asks whether one terminal action
is already uniquely robustly optimal or uniquely admissible under the registered
regret tolerance. If so, it acts immediately.

Otherwise, every candidate probe supplies:

- a finite outcome distribution;
- one posterior decision certificate for each outcome;
- a prospective physical-risk value; and
- an optional acquisition cost.

For probe $e$, the implementation computes

$$
V_{\mathrm{DI}}(e)
=
\overline R_{\mathrm{current}}
-
\sum_y p(y\mid e)\,\overline R_y
-
\lambda c(e),
$$

where $\overline R_y$ is the posterior certificate's minimax worst-case
regret. It also reports the probability that the realized outcome yields a
unique certified terminal action. A probe is eligible only when it is below the
registered risk cap, exceeds the minimum net value, and meets the registered
certification-probability threshold. The default threshold is one, so every
positive-probability outcome must identify a terminal action unless the caller
explicitly registers a weaker requirement.

The deterministic policy is therefore:

1. **act** when the current decision is already certified;
2. **probe** when one safe candidate has positive decision-identification value;
3. **fallback** when neither condition holds.

A realized probe outcome is routed through the same existing fail-closed
certificate consumer. No outcome branch can silently select an unsupported
latent representative.

## Mechanism covered by tests

The focused tests establish:

- immediate action when the current certificate identifies one action;
- selection of a task-relevant probe whose outcomes identify opposing actions;
- rejection of an unsafe high-value probe;
- exact fallback for an uninformative or dependence-destroyed probe;
- optional minimum certification probability;
- deterministic tie breaking;
- validation of outcome probabilities and probe identities; and
- immutable, claim-bounded decision records.

## Scientific boundary

The calculation is exact only relative to caller-supplied finite certificates,
outcome probabilities, hypothesis support, losses, risk values, and costs. It
does not validate those inputs, provide target-domain coverage, authorize a
physical intervention, or establish deployment safety.

The next empirical layer is the source-gated Tracking Cloth V2 observation
study. It tests whether a query-conditioned equal-budget observation improves a
held future cloth query over query-agnostic global-state selection, exact
constant velocity, and a dependence-destroyed control. A later physical study
must replace passive marker reveal with matched low-risk robot interventions and
evaluate a distinct terminal action.
