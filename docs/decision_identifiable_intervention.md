# Decision-identifiable intervention consumption

## Purpose

A rank-deficient physical observation can leave the complete latent state—and
sometimes a continuous physical query—ambiguous while still supporting one
finite downstream action. BayesianPhysTwin owns the exact quotient decision
certificate. Causal4D consumes that certificate and either authorizes one
uniquely certified intervention or executes the caller-owned fallback exactly.

This produces the information hierarchy

```text
complete state identification
    => registered query identification
        => registered finite-decision identification
```

Neither converse is generally valid.

## Ownership boundary

The three repositories have separate responsibilities:

- **Prob4D** represents observation uncertainty, unresolved gauge directions,
  lineage, and joint dependence.
- **BayesianPhysTwin** constructs the registered physical quotient and computes
  the exact finite-action worst-case-regret certificate.
- **Causal4D** consumes the certificate for intervention choice. It does not
  select an unsupported within-class physical explanation and does not re-claim
  the general decision-theory result.

The certificate implementation is merged in
`IPS-Stuttgart/BayesianPhysTwin#830` at
`754a2a97a4a10369ad15236d84f4d2a81c454055`. The theorem and executable
finite mechanism are merged in
`FlorianPfaff/BayesianPhysTwin-Paper#139` at
`527ed1d8f7c9161af5fd4878d18f7401cc4a65f5`.

## Consumer rule

For each action, BayesianPhysTwin reports exact worst-case regret over every
prior-supported complete belief compatible with the registered quotient
posterior. Causal4D applies the following fail-closed rule:

1. If exactly one action is robustly optimal, authorize it.
2. Otherwise, if exactly one action is within the separately registered regret
   tolerance, authorize it.
3. Otherwise, execute the exact caller-owned fallback.

Causal4D deliberately does not choose among multiple admissible actions. Such a
set can be passed to a higher-level planner, but it does not identify one action
under this contract.

Before acting, the consumer independently checks:

- certificate version and semantics;
- action count and action-name uniqueness;
- pairwise worst-case loss-gap matrix and zero diagonal;
- action-wise regret reconstructed from that matrix;
- robust and tolerance-admissible masks;
- deterministic minimax index and regret;
- agreement of all certificate summary fields.

A malformed, semantically incompatible, or numerically inconsistent certificate
raises an error rather than authorizing an action.

## Direct use with a certificate

```python
from causal4d.decision_identifiable_intervention import (
    consume_query_decision_certificate,
)

result = consume_query_decision_certificate(
    certificate,
    action_names=("retain", "update"),
    fallback_action_name="complete-physics-fallback",
)

execute_registered_action(result.action_name)
assert result.used_exact_fallback == (result.certified_action_name is None)
```

The input may be the live BayesianPhysTwin named tuple or a serialized mapping
containing the same fields plus its `summary` mapping.

## Optional end-to-end constructor

With a compatible optional BayesianPhysTwin installation:

```python
from causal4d.decision_identifiable_intervention import (
    decision_identifiable_intervention_from_quotient,
)

result = decision_identifiable_intervention_from_quotient(
    prior_weights=prior,
    quotient_weights=quotient_posterior,
    class_index=hypothesis_to_quotient_class,
    loss_by_hypothesis_action=registered_loss,
    action_names=("retain", "update"),
    fallback_action_name="complete-physics-fallback",
    regret_tolerance=0.0,
)
```

The BayesianPhysTwin import is local. Core Causal4D remains usable without the
optional provider package.

## Relation to task-conditioned probe value

`task_conditioned_design.py` evaluates whether a prospective observation is
worth acquiring. The present module answers the later question of whether the
updated partial physical belief supports one downstream action. The two gates
are distinct:

```text
prospective observation value and risk gate
    -> observation and physical-belief update
        -> decision-identifiability certificate
            -> certified action or exact fallback
```

A high-value probe does not automatically certify an action, and an action can
be decision-identifiable even when the complete state is not.

## Evidence boundary

The strongest existing real-data mechanism evidence is the source-frozen
controlled-gauge DEFORM DLO4/DLO5 evaluation in Prob4D: a gauge-insensitive
segment-centroid query was admitted and improved in all 28 trajectory groups,
while the gauge-sensitive off-axis query was rejected everywhere with exact
fallback. That experiment establishes query-selective physical value, not this
new decision layer on a learned held-out provider.

A future decision-level real-data experiment must register before outcome access:

- complete physical hypotheses and prior support;
- the observation-induced quotient;
- finite action roster and complete loss matrix;
- regret tolerance and fallback action;
- independent physical units and target-access order;
- realized decision-loss and harmful-certification metrics.

## Claim boundary

This consumer validates and applies a supplied certificate. It does not establish
that the quotient is physically correct, validate Prob4D or BayesianPhysTwin,
identify a unique physical cause, calibrate uncertainty, justify the loss or
regret tolerance, establish held-out transport, authorize deployment, or certify
safety.
