# Active and sequential decision identification

Causal4D now separates two acquisition levels.

- `causal4d.active_decision_identification` composes already constructed
  posterior decision certificates with a one-step `act / probe / fallback`
  router.
- `causal4d.sequential_decision_identification` works directly on a registered
  finite physical-hypothesis support, terminal-action losses, and finite probe
  channels. It computes an exact finite-horizon adaptive policy.

Neither layer reconstructs a preferred latent physical state. A terminal action
is emitted only when it is uniquely admissible under the registered support-wise
regret tolerance. Otherwise the policy acquires a registered probe or returns the
caller-owned fallback.

## Support-wise terminal certificate

For supported hypotheses `h` and terminal actions `a`, define

```text
regret(a) = max_h [loss(h, a) - min_b loss(h, b)].
```

An action is admissible when its regret is at most the registered tolerance. The
implementation acts only when exactly one action is admissible. Predictive prior
weights do not weaken this certificate: changing positive prior magnitudes while
preserving support cannot turn an ambiguous decision into a certified one.

Weights are still used for legitimate predictive quantities: probe-outcome
probabilities, mutual information, and expected acquisition cost.

## Exact non-myopic Bellman recursion

For a finite remaining probe roster `E`, horizon `k`, posterior weight vector
`w`, and additive registered risk budget `rho`, the sequential solver applies:

1. act with zero further acquisition cost when the current terminal action is
   uniquely certified;
2. otherwise consider each probe whose risk charge fits within `rho`;
3. update the posterior for every positive-probability outcome;
4. retain a probe only when every possible branch reaches a certified action
   within `k - 1` further acquisitions; and
5. choose the feasible policy with minimum expected or worst-case acquisition
   cost, using deterministic tie breaking.

If no branch-complete policy exists, the result is exact fallback rather than an
arbitrary state completion. Each probe may be acquired at most once. A hard node
budget makes exponential finite-support search fail closed.

This is strictly stronger than ranking probes by immediate regret reduction. A
routing observation may have zero one-step decision value but reveal which cheap
branch-specific diagnostic should be acquired next.

## Probe--action quotient theorem

For the fixed finite interface, associate each hypothesis with the signature

```text
(normalized terminal action-loss differences,
 likelihood row of probe 1,
 ...,
 likelihood row of probe m).
```

Two hypotheses are equivalent exactly when these signatures agree. Aggregating
predictive mass over the resulting classes preserves:

- every terminal action comparison and support-wise regret certificate;
- every registered probe-outcome probability;
- every Bayesian class-mass update under any finite observation history; and
- by induction on the horizon, the complete adaptive policy, expected cost,
  worst-case cost, and worst-case additive risk charge.

This is the coarsest quotient preserving all registered terminal loss differences
and all registered probe conditional laws. A decision-only quotient is generally
insufficient for sequential acquisition because hypotheses with identical
terminal decisions may require different next probes.

The theorem is deliberately interface-relative. It does not preserve arbitrary
future losses, unregistered probes, or unrestricted physical-state covariance.

## Controlled strict separation

The deterministic mechanism contains 32 complete hypotheses:

- one task bit determining the correct terminal action;
- one route bit determining which local diagnostic is informative;
- two nuisance bits visible to a high-information probe; and
- one duplicated microstate bit irrelevant to every registered action and probe.

Five probes are registered. A direct global task probe costs `0.50`. A routing
probe costs `0.05`; after its outcome, the relevant local task probe costs
`0.30`. A four-way nuisance probe has the largest generic mutual information but
cannot certify the decision.

The exact results are:

| Policy | First probe | Guaranteed expected cost |
|---|---|---:|
| Generic mutual information | nuisance four-way | not decision sufficient |
| One-step decision value | global task | 0.50 |
| One-probe exact policy | global task | 0.50 |
| Cheapest non-adaptive sufficient set | global task | 0.50 |
| **Two-step exact adaptive policy** | **route** | **0.35** |

The routing probe has exactly zero immediate regret reduction. Nevertheless, the
non-myopic policy chooses it and then acquires `local-r0` or `local-r1` according
to the observed route. Every terminal leaf certifies the correct action while
eight complete hypotheses remain compatible. Thus decision identification is
strictly weaker than complete-state identification.

The probe--action quotient reduces the 32 complete hypotheses to 16 classes and
reproduces the complete adaptive policy and both cost criteria exactly.

## One-step certificate composition

The original one-step API remains available for callers that already construct
posterior certificates externally. A probe supplies a finite outcome
probability, one posterior certificate per outcome, a prospective physical-risk
value, and an optional cost. The router acts when the current certificate is
unique, probes when one safe candidate has registered positive decision value,
and falls back otherwise.

A realized probe outcome is always routed through the existing fail-closed
certificate consumer; no branch silently selects an unsupported latent
representative.

## Scientific boundary

All exactness is conditional on the supplied finite support, losses, probe
likelihoods, conditional-independence model, costs, additive risk charges,
regret tolerance, and horizon. The implementation does not validate physical
hypotheses or sensors, estimate target-domain support, establish calibrated
realized risk, authorize intervention, or provide a deployment-safety guarantee.

The decisive empirical successor is a prospective physical task in which an
initial observation leaves contact topology ambiguous, a low-risk routing probe
chooses a branch-specific second probe, and the resulting terminal action is
scored against generic information acquisition, a direct global probe, and exact
fallback at matched physical risk and sensing cost.
