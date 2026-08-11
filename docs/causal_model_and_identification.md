# Causal model and identification boundary

## Status

This document formalizes the structural assumptions already used by Causal4D.
It does not change an estimator, intervention support, physical protocol,
calibration unit, target-access rule, or recorded result. The registered
36-execution physical method remains frozen.

Causal4D distinguishes a commanded action from the intervention that is
physically realized by a deformable object:

```text
command U_t
    |
    v
realized intervention Z_t = (Phi_s, K_t)
    |
    v
physical state Xi_t and observable readout R_t
```

Upper-case symbols below denote random variables. Lower-case symbols denote
realizations. An execution belongs to a grasp/session `s` and is indexed by
physical time `t`.

## Variables

The physical-twin state is

```text
Xi_t = (X_t, V_t, Theta),
```

where `X_t` and `V_t` stack graph-node positions and velocities and `Theta`
contains persistent physical parameters. The remaining variables are:

- `U_t`: commanded controller trajectory;
- `Phi_s`: persistent or slowly varying realization variables for session `s`,
  including gain, delay, and controller-frame effects;
- `K_t`: event-specific contact state, attachment patch, and slip variables;
- `Z_t = (Phi_s, K_t)`: the realized intervention;
- `Delta_t`: unresolved process or readout discrepancy, kept separate from
  simulator position and velocity unless a state-assimilation mechanism passes
  its own prospective controls;
- `R_t`: physical readout used for prediction and downstream queries;
- `O_t`: an admitted observation factor, with its covariance, provenance,
  correlation group, and causal interval; and
- `E_*`: exogenous variables for initial conditions, actuator realization,
  contact, unmodeled dynamics, and observation noise.

BayesianPhysTwin supplies an uncertain belief over endpoint state, physical
parameters, and discrepancy. Causal4D owns realized-intervention abduction,
explicit intervention, and held-out prediction.

## Structural equations

A compatible structural causal model can be written as

```text
Theta                 = f_theta(E_theta)
(X_0, V_0)            = f_initial(Theta, E_initial)
Phi_s                  = f_phi(H_s, E_phi_s)
K_t                    = f_kappa(X_t, V_t, U_t, Phi_s, K_{t-1}, E_kappa_t)
A_t                    = f_realize(U_t, Phi_s)
(X_{t+1}, V_{t+1})    = F_Theta(X_t, V_t, A_t, K_t, E_dynamics_t)
Delta_{t+1}            = G(Delta_t, X_t, V_t, A_t, K_t, E_delta_t)
R_t                    = h(X_t, V_t, Delta_t)
O_t                    = g(R_t, E_observation_t)
```

`H_s` denotes source-only session or hardware history available before a target
execution. The important graph separation is that `U_t` is not inserted into
`F_Theta` as though it were the realized force/contact. It first passes through
`f_realize`, and contact/slip remain explicit event variables.

The equations are a semantic contract, not a claim that every exogenous
variable is point identifiable. Finite rollout banks approximate the posterior
support of the variables needed for a registered query.

## Causal timing and factual abduction

Let `t_0` be the intervention boundary and `tau` the exclusive end of the
registered early-response prefix. The information sets are half-open:

```text
O-       = O_[0, t_0)
O+prefix = O_[t_0, tau)
future   = O_[tau, T)
```

The BayesianPhysTwin endpoint belief uses `O-`. Causal4D may use only the
registered `O+prefix` and separately owned actuator/contact factors to abduct
`(Theta, Phi_s, K_obs)`. No observation whose source interval crosses `tau` may
enter factual abduction, method selection, calibration fitting, or a predictive
artifact for that execution.

The factual posterior is therefore

```text
p(Theta, Xi_tau, Delta_tau, Phi_s, K_obs | O-, O+prefix, U_obs, S),
```

where `S` denotes independently admitted sensor factors. Evidence ownership
requires one raw factor to enter one independent likelihood path. Correlated
state/intervention use must be represented as one joint factor rather than two
renamed independent factors.

## Abduction, intervention, and prediction

For a counterfactual command `u_cf`, Causal4D applies three operations:

1. **Abduction:** condition the finite support on the allowed factual evidence.
2. **Action:** replace the command-generating mechanism with `do(U = u_cf)`.
3. **Prediction:** propagate the posterior physical state and the variables that
   are assumed transportable under the registered contact policy.

The intervention does not rewrite `Theta`, the abducted endpoint state, or the
physical evidence ledger. Persistent `Phi_s` is transported only under the
session/hardware invariance assumption below. Event state is handled explicitly:

- `same_grasp` with fixed event state carries `K_obs`;
- `same_grasp` with evolving slip carries the registered contact patch and
  resamples the branch-local slip variable; and
- `new_contact` transports `(Theta, Phi_s)` and samples a fresh `K_cf` from the
  registered conditional support.

These policies are part of the estimand. They are not interchangeable numerical
options.

## Interventional trajectory contrasts

A physical posterior gives a branch marginal. A causal comparison additionally
needs a declared cross-world coupling. For a registered linear trajectory query
`Q`, the analysis-only contrast is

```text
Delta_Q = Q(R^a) - Q(R^b),
```

where branch `a` is always the positive direction. The corresponding posterior
is

```text
p(Delta_Q | D, do(U = u_a), do(U = u_b), C),
```

and `C` is the coupling policy. Branch marginals alone do not identify `C`.
Causal4D therefore records it explicitly and verifies that the paired support
preserves both source marginals.

The supported couplings are:

### `shared_component`

Pair equal complete support components. The branches must have identical
component identities, weights, `Theta`, `Phi`, `K`, hypothesis indices, and
physical-particle indices. This is appropriate for controlled shared-exogenous
comparisons and fixed-event same-grasp branches whose banks use the same finite
support.

### `shared_twin_phi`

Share the physical particle and persistent `Phi`. Remaining event variables are
drawn conditionally and independently inside each shared stratum. An optional,
explicit list of `shared_kappa_names` may additionally hold a contact patch or
another registered event coordinate fixed while branch-local slip is resampled.
Every shared stratum must have the same marginal mass in both branches.

### `independent_product`

Form the product of the two branch marginals. This is a population-level
uncoupled diagnostic. It is not an individual-level cross-world effect and must
not be described as paired real counterfactual ground truth.

`PhysicalPosterior` does not encode cross-branch conditional discrepancy
covariance. The contrast API consequently supports only:

- `component_means_only`, which reports the finite mixture of component-mean
  contrasts; or
- `independent_readout`, which adds the two declared branch query covariances and
  explicitly assumes zero cross-branch conditional covariance.

The API never infers unrecorded discrepancy cancellation.

## Identification assumptions

A contrast has the stated interpretation only under all applicable assumptions.

### 1. Causal cutoff and no future leakage

Every source factor is bound to an exclusive causal interval. Outcomes at or
after the held-out boundary cannot affect abduction, coupling selection, query
selection, thresholds, exclusions, or calibration fitting.

### 2. Structural modularity

`do(U = u_cf)` replaces only the command mechanism. It does not silently alter
`Theta`, the endpoint belief, the discrepancy definition, observation noise, or
other structural equations.

### 3. Persistent-variable transport

The comparison assumes the transported `Theta` and `Phi_s` would remain the
same across the two branches. For a new independent session, the appropriate
object is the session-predictive hierarchy rather than the observed session's
realization.

### 4. Explicit event-state policy

Contact and slip are not automatically transportable. Fixed grasp, fixed patch
with evolving slip, and fresh contact are different causal queries and must use
the corresponding support coupling.

### 5. Positivity and support adequacy

Every required `(Theta, Phi, K)` stratum must have positive registered support.
A contrast is unsupported if a branch cannot represent the transported factual
mass. Renormalization may not hide missing original posterior mass.

### 6. State/discrepancy separation

A persistent residual is not automatically a physical displacement. `Delta`
may change a readout or process moment without changing simulator position or
velocity. A state interpretation requires a separately validated assimilation
mechanism.

### 7. Observation and nuisance accounting

Association, prior reliability, posterior responsibility, and composite weight
remain distinct. Conditional covariance may be paired with explicit nuisance
variables, or nuisance uncertainty may be marginalized into covariance, but the
same uncertainty cannot be counted in both forms.

### 8. Cross-world coupling is an assumption

A coupling is not learned from two branch marginals. `shared_component` and
`shared_twin_phi` are justified by the structural design and support identities;
`independent_product` deliberately declines such a paired assumption.

### 9. Real-data estimand boundary

Repeated real executions from matched initial conditions identify held-out
interventional prediction at the registered execution/session level. They do not
supply both potential outcomes for one physical execution. Real contact recovery
also requires independent contact instrumentation; a posterior mode alone is not
ground truth.

### 10. Independent calibration units

Coverage or calibration claims require held-out independent executions/sessions.
Frames, nodes, views, coordinates, and posterior components are not independent
calibration units.

## What the contrast posterior may report

Under the declared coupling and conditional-variance policy, the artifact may
report:

- posterior mean and covariance of `Delta_Q`;
- marginal posterior standard deviations and equal-tail credible intervals;
- `P(Delta_Q > 0)` for each registered query output;
- support size, effective support, and source-marginal reconstruction error; and
- exact source posterior, source query, coupling, query, and archive identities.

These are posterior summaries. They do not by themselves establish empirical
calibration, causal sufficiency, physical benefit, deployment safety, or state
of the art.

## Relationship to the registered physical experiment

The formalization and contrast API are additive analysis infrastructure. They do
not enter, replace, or rescue the frozen primary 36-execution comparison. Any
paper-facing contrast query and coupling must be registered before target
outcomes are accessed. A failed or weak contrast remains a complete negative or
bounded result.
