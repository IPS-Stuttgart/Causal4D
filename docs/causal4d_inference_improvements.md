# Session-Aware Abduction and Stable Discrepancy Dynamics

This development extension addresses limitations in multi-execution inference
without changing the frozen `v0.3.0-causal4d-aip` path.

## Session-aware composite evidence

`abduct_hierarchical_interventions` accepts `session_ids`. Executions in the
same grasp or reset session share nuisance errors, so they should not each count
as an independent unit of evidence for persistent variables.

For a session with `n_s` executions, each marginalized execution log evidence is
weighted by `1/n_s`. The local execution likelihood remains unpowered when
recovering its `kappa_e` posterior. Omitting `session_ids` reproduces the original
independent-execution product exactly. Explicit `execution_evidence_powers` can
be supplied for a source-frozen alternative composite likelihood.

The result metadata records the session IDs, evidence powers, session count, and
evidence mode.

## Optional session-level latent intervention

The original hierarchy forces one persistent intervention realization `phi` to
be shared by every session. For a future protocol,
`session_phi_transition[g, f]` can instead define

```text
p(phi_session = f | phi_bar = g)
```

on the existing finite `phi` support. The resulting hierarchy is

```text
theta                         shared physical particle
phi_bar                       global hardware/intervention hyperstate
phi_session                   persistent realization for one session
kappa_execution               local contact and slip realization
```

For each session, powered execution evidence is first aggregated in its local
`phi_session` state. The session state is marginalized when updating
`(phi_bar, theta)`, and its posterior is then recovered conditional on all
sessions. Each execution keeps its full local `kappa_execution` posterior.
The returned `SessionHierarchyPosterior` contains:

- global `(phi_bar, theta)` weights;
- one `(phi_session, theta)` posterior per ordered session;
- the exact execution-to-session binding and evidence powers;
- the finite transition matrix; and
- predictive `(phi_session, theta)` weights for a new session.

A zero-session-variance model is represented by the exact identity transition:

```python
result = abduct_hierarchical_interventions(
    banks,
    observations,
    prefix_frame_counts=prefix_frame_counts,
    session_ids=session_ids,
    session_phi_transition=np.eye(phi_count),
)
```

That path is an exact fallback: global weights and every execution posterior are
required to equal the legacy shared-`phi` result bit for bit. Exact zeros in a
nonidentity transition preserve excluded support. Invalid, nonfinite,
non-stochastic, or dimensionally inconsistent transitions fail closed.

A nonidentity transition must be selected from source sessions or preregistered.
It must not be tuned on confirmatory targets. The finite transition can encode
session-to-session variation in actuator gain, delay, controller-frame bias, or
another persistent intervention coordinate already represented by `phi`. It does
not create support absent from the registered rollout banks.

The global `phi_marginal` is a `phi_bar` marginal when this option is active.
Use `session_phi_marginals` for observed sessions and
`predictive_session_joint_weights` for a new session. Treating the global
hyperstate directly as a realized session intervention would be a semantic
error.

## Scale-invariant and partial identifiability

`assess_intervention_identifiability` accepts characteristic
`parameter_scales`. Intervention sensitivity columns are multiplied by those
scales before nuisance projection and information analysis. This makes the
result invariant to unit conversions such as degrees versus radians or frames
versus seconds, provided the corresponding scales are converted consistently.

The result retains identifiable and nullspace bases. The helper
`preserve_prior_within_unidentified_subspace` removes unsupported posterior
distinctions while retaining evidence between distinguishable projection
groups:

- full rank returns the supplied update;
- rank zero returns the normalized prior exactly;
- partial rank preserves prior-relative weights within each indistinguishable
  group.

The frozen guarded-abduction behavior remains conservative: this helper is
opt-in and does not silently replace exact-prior abstention.

## Stable discrepancy mean dynamics

`StableDiscrepancyTransitionModel` augments the existing action-conditioned
innovation covariance with a mean transition

```text
d_(t+1) = A(f_t) d_t + b(f_t) + epsilon_t.
```

The transition is built from

```text
G(f) = S(f) - C(f)
A(f) = expm(G(f)),
```

where `S` is skew-symmetric and `C` is positive semidefinite. This permits
source-fitted graph-mode rotation and contraction while keeping the transition
non-expansive. A feature-conditioned drift is optional and norm capped.

`StableDiscrepancyTransitionModel.identity(...)` gives exact graph persistence.
`forecast_action_conditioned_dynamics` propagates

```text
m_(t+1) = A m_t + b
P_(t+1) = A P_t A^T + Q(f_t),
```

using the existing `ActionConditionedDiscrepancyModel` as the innovation model.
Transition generators, feature weights, and drift caps must be selected on
source executions or preregistered before confirmatory evaluation.

## Claim boundary

These additions are inference machinery, not new real-data evidence. The
session-level latent intervention is intended for a separately versioned future
protocol; it is not part of the locked 36-execution estimator. Promotion still
requires independent-session calibration, source-only hierarchy selection,
leave-one-session and held-out-action evaluation, finite-support simulation-
based calibration, and exact identity/prior fallback controls.
