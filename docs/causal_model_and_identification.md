# Causal model and identification boundary

## Purpose

Causal4D distinguishes a **commanded action** from the **physical intervention
realized by the object**. This document makes that distinction explicit as a
structural causal model (SCM), states the cross-world assumptions used by the
counterfactual operator, and separates quantities that are identified in the
controlled benchmark from quantities that can only be estimated through
repeated real executions.

This is a semantic specification of the existing method. It does not change the
frozen six-frame estimator, the registered 18-session/36-execution physical
protocol, the intervention support, Prob4D admission, BayesianPhysTwin provider
contracts, or the physical evidence count.

## Variables

The model separates persistent, session-local, execution-local, state,
discrepancy, and observation variables.

| Symbol | Meaning | Typical Causal4D representation |
| --- | --- | --- |
| `Theta` | physical parameters of the deformable twin | BayesianPhysTwin particle |
| `X_0` | pre-intervention physical state | particle endpoint position/velocity |
| `Phi_bar` | persistent hardware or controller hyperstate | optional hierarchy |
| `Phi_s` | session realization: gain, delay, controller-frame bias | `phi` |
| `Kappa_e,t` | execution/event contact state: patch, attachment, slip | `kappa` |
| `U_e,t` | commanded controller trajectory | `u_obs` or `u_cf` |
| `A_e,t` | realized force/contact actuation | simulator input |
| `X_e,t` | physical deformable-object state | state trajectory |
| `Delta_e,t` | unresolved process/readout discrepancy | separate discrepancy belief |
| `Y_e,t` | physical observable before sensor corruption | discrepancy-aware readout |
| `O_e,t` | recorded observation | camera, tracker, tactile, wrench, actuator evidence |
| `E_e` | exogenous execution conditions | reset, support, gravity, disturbances |

The realized intervention is

```text
Z_e = (Phi_s, Kappa_e)
```

and is not identified by the command `U_e` alone.

## Structural equations

A compact SCM is

```text
Theta       := f_theta(E_theta)
X_0         := f_initial(Theta, E_initial)
Phi_bar     := f_hardware(E_hardware)
Phi_s       := f_session(Phi_bar, E_session)
Kappa_e,t   := f_contact(X_e,t, U_e, Phi_s, E_contact,e,t)
A_e,t       := f_realize(U_e,t, Phi_s, Kappa_e,t)
X_e,t+1     := f_dynamics(X_e,t, A_e,t, Theta, E_process,e,t)
Delta_e,t+1 := f_discrepancy(Delta_e,t, X_e,t, A_e,t, E_delta,e,t)
Y_e,t       := h(X_e,t) + Delta_e,t
O_e,t       := f_sensor(Y_e,t, E_sensor,e,t)
```

The implementation may use finite support rather than continuous variables, but
the semantic roles remain distinct. In particular:

- `Phi` changes how a controller command is realized;
- `Kappa` identifies event-specific contact and slip;
- `Theta` changes the object dynamics;
- `Delta` absorbs unresolved predictive mismatch and is not silently injected
  into simulator state; and
- observation noise or gauge uncertainty belongs to the sensor factor, not to
  the realized intervention.

## Factual abduction

Let `tau` be the exclusive causal cutoff inside the factual post-intervention
window. Causal4D forms

```text
p(Theta, X_tau, Delta_tau, Phi_s, Kappa_obs |
  O_minus, O_plus[:tau], U_obs)
```

using only the allowed prefix. The untouched future cannot affect this
posterior. Independent actuator or force/torque evidence may contribute a
separate factor only when evidence ownership proves that the same raw factor is
not already present in the physical-state update.

The posterior is an **abduction of the realized intervention**, not proof that a
particular real contact node was recovered. Real contact recovery requires
independent contact instrumentation or another identification argument.

## Intervention operator

For a counterfactual command `u_cf`, the action step replaces the command
mechanism:

```text
do(U = u_cf).
```

It does not condition on a future response to `u_cf`. The prediction stage then
propagates the factual posterior under one of three contact semantics.

### Fixed same grasp

The branch shares

```text
(Theta, Phi_s, Kappa_obs, X_tau, Delta_tau)
```

and changes only the command. This is appropriate when the same attachment and
slip state are part of the intended intervention definition.

### Same patch with evolving slip

The branch shares

```text
(Theta, Phi_s, contact patch, X_tau, Delta_tau)
```

but resamples or propagates slip. This avoids treating a future slip event as a
persistent property of the factual grasp.

### New contact

The branch shares

```text
(Theta, Phi_s, X_tau, Delta_tau)
```

and samples a fresh `Kappa_cf` from the declared new-contact mechanism. Factual
contact cannot be silently reused.

## Interventional estimands

For a registered linear or nonlinear query `Q`, the basic action contrast is

```text
Delta_Q(a, b) = Q(Y^do(U=a)) - Q(Y^do(U=b)).
```

Examples include endpoint displacement, average contact-patch motion,
early-to-late deformation, or a task-specific linear readout. The sign and
ordering must be explicit; the analysis artifact uses `left - right`.

The marginal posteriors for the two actions do not uniquely determine the
posterior of `Delta_Q`. A **cross-world coupling** is required. Causal4D's
analysis-only contrast contract therefore records which latent variables are
shared:

| Contact semantics | Default shared variables |
| --- | --- |
| fixed same grasp | `Theta`, `Phi`, complete `Kappa` |
| same patch, evolving slip | `Theta`, `Phi`, contact patch |
| new contact | `Theta`, `Phi`; contact is conditionally resampled |
| independent sensitivity | none |

Within each shared-latent stratum, the default coupling is the product of the
two conditional component distributions. It preserves both source posterior
marginals exactly. The independent product is a sensitivity analysis, not the
default causal coupling, because it destroys cancellation of shared physical
and intervention uncertainty.

## Conditional readout uncertainty

`PhysicalPosterior.readout_variance_m2` is a marginal conditional variance for
one predicted readout. It does not, by itself, identify the covariance between
two potential readouts. For a readout contrast,

```text
Var(epsilon_a - epsilon_b)
  = Var(epsilon_a) + Var(epsilon_b)
    - 2 rho sqrt(Var(epsilon_a) Var(epsilon_b)).
```

The cross-world correlation `rho` is therefore an explicit sensitivity
parameter. Omitting it produces a finite-support contrast over component means
only. Setting `rho=0` assumes conditionally independent readout errors; setting
`rho=1` assumes a shared standardized discrepancy mode. Neither setting is
silently inferred from the marginal variances.

State-trajectory contrasts do not use readout variance.

## Identification assumptions

The following assumptions are required for the corresponding interpretation.

### 1. Causal timing

All factual abduction inputs precede the exclusive cutoff. No source window,
learned feature, calibration statistic, or semantic forecast may contain future
frames beyond that cutoff.

### 2. Consistency

When the commanded and realized intervention equal a supported intervention,
the observed outcome follows the corresponding structural branch. Ambiguous or
mixed command/contact definitions must not be relabelled after observing the
outcome.

### 3. Positivity on finite support

The counterfactual action and required realization/contact states must have
support in the declared rollout bank. Exact-zero posterior or prior support is
never resurrected.

### 4. Transportability of physical parameters

`Theta` is assumed stable across the compared actions for the same object and
registered reset regime. Cross-object transport requires a separate hierarchy
or experiment.

### 5. Transportability of persistent realization variables

`Phi` may be transferred between actions within the declared session. Transfer
to a new session requires either the shared-`Phi` model or the registered
session hierarchy. A session realization must not be treated as a globally
fixed hardware value without that assumption.

### 6. Contact-policy validity

Fixed-grasp, evolving-slip, and new-contact branches encode different
interventions. Automatic coupling is valid only when both compared posteriors
use compatible contact semantics. Mixed semantics require an explicit
sensitivity policy and cannot be described as a pure command effect.

### 7. Stable exogenous conditions

The controlled benchmark can share simulator exogenous variables across
factual and alternative interventions. Real experiments cannot observe the
same execution under two commands. They require matched resets and repeated
independent executions, with residual reset and replay variation reported.

### 8. Discrepancy/state separation

A persistent camera or trajectory residual is not automatically a physical
state displacement. `Delta` remains a readout/process discrepancy unless a
state-assimilation mechanism passes its own dynamical and held-out controls.

### 9. One raw factor, one inference path

Observation, actuator, wrench, tactile, and calibration factors are owned across
stages. Renaming a factor does not make it independent. A factor used jointly
for state and intervention inference must appear in one explicit joint
likelihood.

### 10. Independent statistical units

Frames, nodes, coordinates, and views are not independent real executions. The
registered physical study uses grasp sessions/executions as its analysis and
calibration units.

## What is identified where

### Controlled benchmark

When simulator exogenous variables and initial conditions are shared, the
benchmark supports controlled counterfactual prediction and direct recovery of
known latent interventions. This is the strongest setting for validating the
causal operator itself.

### Repeated physical experiment

A real object cannot reveal both potential outcomes for one execution. The
registered experiment estimates held-out interventional prediction from matched
initial conditions and session-level action effects. It does not establish
individual-level real counterfactual ground truth.

### Camera-only real interaction

Without independent actuation/contact information, persistent bias, physical
state, contact realization, and discrepancy may remain partially identified.
Posterior concentration is not proof of identification. The structured
identifiability and oracle-gap diagnostics bound that limitation.

## Software mapping

| SCM operation | Causal4D surface |
| --- | --- |
| uncertain `Theta`, `X_0`, `Delta` | `TwinBelief` |
| prefix-only abduction of `Phi`, `Kappa_obs` | `FactualIntervention` |
| explicit `do(u_cf)` and contact semantics | `CounterfactualQuery` |
| action-specific physical/readout support | `PhysicalPosterior` |
| coupled `Delta_Q(a,b)` analysis | `interventional_contrast` |
| dependence-aware held-out scoring | `posterior_scoring` |
| evidence reuse prevention | `ConsumedEvidenceLedgerV1` |

## Claim boundary

A valid interventional-contrast artifact proves that two source posteriors share
factual lineage, that an explicit coupling preserves both marginals, and that a
registered query was evaluated consistently. It does not prove causal
sufficiency, correct contact recovery, calibrated physical uncertainty,
transport to another object, deployment safety, or state of the art.
