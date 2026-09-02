# PokeFlex anytime decision-directed probing

## Addition beyond one-shot probe selection

The one-probe experiment asks which diagnostic interaction is most useful for a
registered downstream query. This secondary extension asks a stronger question:

> Can the physical twin stop acquiring observations as soon as the downstream
> action is decision-identifiable?

The controller has a maximum budget of two reset-matched logged probe responses.
At every stage it either:

1. acts because the current worst-case decision regret is below the source-frozen
   tolerance;
2. reveals the safe candidate with the largest expected reduction in robust
   downstream regret net of probe cost; or
3. returns the exact no-probe physical fallback because no positive-value probe
   remains.

## Decision-directed value

For current compatible physical beliefs \(\mathcal C_t\), terminal action \(a\),
and candidate probe \(e\), the registered score is

\[
V_t(e)=
\min_a\max_{p\in\mathcal C_t}R_p(a)
-\mathbb E_y\!\left[
  \min_a\max_{p\in\mathcal C_{t+1}^{e,y}}R_p(a)
\right]
-\lambda c(e).
\]

A probe is admissible only when it passes the physical-risk gate and \(V_t(e)>0\).
The acquisition terminates immediately when one action satisfies the bounded-regret
certificate.

## Matched baselines

- no probe;
- one source-fixed probe order;
- random safe probes with the same stopping rule;
- greedy generic mutual information with the same maximum budget;
- greedy task-conditioned regret reduction;
- a dependence-destroyed task policy;
- an oracle diagnostic upper bound.

All non-oracle methods receive the same candidate set, source data, initial-state
certificate, physical costs, risk estimates and maximum response budget.

## Primary endpoints

The study reports object-balanced downstream decision regret, area under the
regret-versus-probe-cost curve, mean probe count, decision-certification rate,
fallback rate and harmful-nonfallback rate. Probe count is not a nuisance metric:
using fewer physical interactions for the same or lower regret is part of the
scientific claim.

## Confirmatory order

This extension is secondary and is evaluated only after the one-probe primary gate
passes. Its complete zero-, one- and two-probe prediction panel is included in the
same pre-scoring joint seal. It cannot be used to rescue a negative one-probe result.

## Strong claim enabled by a positive result

> A decision-identifiable physical twin acquires only the physical observations
> needed for the registered action: it matches or improves generic information
> acquisition in downstream regret while using no more probes, and it returns exact
> fallback when additional observation has no positive decision value.

## Boundary

Because PokeFlex contains separately reset logged interactions, this is offline
reset-matched multi-observation acquisition. It is not evidence that two probes can
be executed sequentially on one continuously evolving physical state, and it does
not establish deployment safety.
