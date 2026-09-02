# Distributionally robust active decision identification

## Motivation

The certificate-level active policy in Causal4D is exact for one supplied finite
model: one current query-decision certificate, one probability for every probe
outcome, one posterior certificate per outcome, and scalar probe costs and risk
scores. That is the correct finite-support result, but it is brittle when the
probe channel or the certificate regret is estimated from limited physical data.

This extension replaces point assumptions by registered uncertainty sets and
fails closed whenever a useful probe is not certified over the entire set.

## Robust probe interface

For probe `e` with outcomes `y = 1, ..., K`, register the box-simplex

\[
\mathcal P_e = \left\{p:\; \ell_y \le p_y \le u_y,\quad
\sum_y p_y=1\right\}.
\]

For every outcome, the existing structural certificate supplies posterior
minimax regret `r_y`. A nonnegative branch margin `q_y` produces the upper bound

\[
r_y^+ = r_y + q_y.
\]

A structurally identified action remains executable only if its selected regret
plus `q_y` is still within the registered tolerance. Otherwise that outcome is
routed to the exact caller-owned fallback.

The planner computes

\[
\overline R_e = \max_{p\in\mathcal P_e}\sum_y p_y r_y^+,
\]

and the worst-case probability of obtaining an executable certificate,

\[
\underline C_e = \min_{p\in\mathcal P_e}
\sum_y p_y\,\mathbf 1\{\text{outcome }y\text{ is certified}\}.
\]

With current minimax regret `R_0`, probe cost `c_e`, and cost multiplier
`lambda`, its guaranteed net value is

\[
V_e^- = R_0 - \overline R_e - \lambda c_e.
\]

A probe is eligible only when:

1. its prospective physical-risk upper bound is below the registered cap;
2. `C_e^-` reaches the registered certification-probability threshold; and
3. `V_e^-` is strictly above the registered minimum.

The policy acts immediately when the current certificate already identifies an
action. It otherwise chooses the eligible probe with greatest guaranteed net
value, using deterministic tie breaking. If no probe qualifies, it returns the
exact fallback.

## Exact box-simplex optimization

The objective is linear in `p`. The extremizer is therefore obtained without a
generic optimizer:

1. assign every outcome its lower probability bound;
2. sort outcomes by branch value;
3. allocate remaining probability mass up to each upper bound in descending
   order for maximization or ascending order for minimization.

This greedy construction is an exact solution of the box-constrained simplex
linear program. The implementation retains the extremizing distributions as
checkable witnesses.

## Complete-group regret calibration

Let group `j` be one complete physical object, trajectory, or condition. It must
contain every registered probe branch, action, query, and horizon used by the
future planner. For structural bounds `b[j,t]` and realized regrets `r[j,t]`, use

\[
s_j = \max_t \bigl(r[j,t]-b[j,t]\bigr).
\]

For `n` calibration groups and miscoverage level `alpha_r`, the one-indexed
split-conformal rank is

\[
k = \left\lceil(n+1)(1-\alpha_r)\right\rceil.
\]

The additive margin is the `k`-th sorted score, floored at zero. If `k > n`, a
finite margin is unavailable and the implementation returns positive infinity;
conversion to a probe then fails closed.

Under exchangeability of complete calibration and future groups, and provided
the structural procedure was fixed without using calibration outcomes, this is
a finite-sample group-marginal statement. It is not conditional coverage.

## Selection-safe outcome-probability boxes

For `n` independent, identically distributed complete trials of a categorical
probe and `M` pre-registered probability coordinates across all candidate
probes, the implementation uses the two-sided Hoeffding radius

\[
\epsilon = \min\left\{1,
\sqrt{\frac{\log(2M/\alpha_p)}{2n}}\right\}.
\]

Each empirical probability `p_hat_y` receives bounds

\[
\ell_y=\max(0,\hat p_y-\epsilon),\qquad
u_y=\min(1,\hat p_y+\epsilon).
\]

The union bound covers all `M` registered coordinates with probability at least
`1-alpha_p`. Supplying only the number of outcomes of the current probe is valid
for that probe alone; supplying the total number of outcome coordinates is
required before selecting among multiple probes.

If the complete-group regret event and the simultaneous probability event are
both valid, a union bound gives joint miscoverage no greater than
`alpha_r + alpha_p`. This statement is still conditional on the registered
physical contexts matching the calibration contexts.

## Reduction to the nominal planner

Setting every lower and upper probability bound equal to its nominal outcome
probability and setting all regret inflations to zero embeds the existing
certificate probe as a singleton ambiguity set. The robust calculation then
recovers the nominal expected regret, certification probability, and probe
selection.

## Controlled mechanism

The registered experiment contains two probes:

- **fragile high nominal value:** 95 of 100 calibration trials produce a useful
  branch, but another admissible outcome leaves the decision unresolved;
- **stable lower nominal value:** both branches identify opposite terminal
  actions, but each receives a larger branch-regret upper bound.

The point model chooses the fragile probe. After simultaneous probability
calibration and complete-group regret inflation, the robust policy chooses the
stable probe because it is certified for every distribution in its ambiguity
set. A larger group-calibration margin revokes both actions and restores exact
fallback. Too few complete calibration groups also fail closed.

## Physical evaluation needed for the paper claim

The decisive empirical panel should use repeated interventions on complete
objects, with candidate diagnostic probes and held terminal challenges separated
before payload inspection. PokeFlex is currently the best public route because
it contains repeated poking interactions and dropping trials that can support
both same-family and cross-intervention queries.

For every object-level target split, report:

- nominal versus robust probe identity;
- terminal regret and harmful-action rate;
- immediate-act, probe, and fallback fractions;
- observation cost and physical-risk upper bound;
- empirical coverage of the group-calibrated regret bound;
- sensitivity to `alpha_r`, `alpha_p`, probability family size, and risk cap;
- a dependence-destroyed control; and
- an oracle using the held challenge outcome only as an upper bound.

## Claim boundary

The robust planner does not make an incorrect support complete. It does not turn
unvalidated risk scores into safety guarantees, establish exchangeability,
protect against unregistered outcomes, or authorize deployment. Its guarantee
is set-relative: it is valid only when the registered probability boxes,
regret upper bounds, action support, and risk upper bounds contain the deployed
physical process.
