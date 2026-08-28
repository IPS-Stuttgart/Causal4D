# Causal4D Paper Scope

## Submission decision

The first Causal4D paper is a **public-data-only, evidence-bounded paper**. It
does not require a new hardware experiment.

Its central claim is:

> **Bayesian abduction of realized interventions improves held-out
> interventional prediction in controlled deformable-object systems and enables
> auditable prediction, abstention, and failure analysis on public
> deformable-object data.**

The paper studies the distinction

```text
command u != realized intervention z = (phi, kappa)
```

where `phi` contains persistent actuation variables such as gain, delay, and
controller-frame bias, while `kappa` contains event-specific contact and slip.
An uncertain physical twin supplies state, parameter, and discrepancy beliefs.

## Contribution chain

The paper must establish the following chain without relying on newly collected
data:

1. define commanded and realized interventions as distinct variables;
2. infer a posterior over `z` jointly with uncertainty over the physical twin;
3. implement explicit abduction, `do(u_cf)`, and posterior prediction;
4. demonstrate held-out contact/action gains under controlled shared-exogenous
   conditions;
5. demonstrate target-closed held-out action prediction on public data;
6. retain negative public-data source gates and uncertainty failures rather than
   hiding them; and
7. state exact identification, calibration, and generalization boundaries.

No paper claim requires the previously registered 18-session/36-execution
hardware protocol.

## Claim hierarchy

| Tier | Component | Paper role |
| --- | --- | --- |
| Core | `u` versus realized `z=(phi,kappa)` | central problem and contribution |
| Core | joint twin/intervention posterior | central method |
| Core | abduction--intervention--prediction | central causal operator |
| Core | controlled held-out contacts/actions | causal validation |
| Public evidence | Deform360 held-out action | target-closed public-data validation |
| Public failure control | PokeFlex rejected source backend | model-class and support boundary |
| Diagnostic | released PhysTwin interaction | undercoverage and discrepancy localization |
| Formal boundary | query-specific intervention equivalence | predictive versus physical identity |
| Supporting backend | BayesianPhysTwin / physical simulators | uncertain physical model |
| Optional feeder | Prob4D | not required for the paper |
| Optional future work | 18-session/36-execution hardware protocol | nonblocking future validation |

## Evidence status

| Link | Status | Evidence |
| --- | --- | --- |
| command/realization decomposition | implemented | typed `u_obs`, `phi`, `kappa_obs`, and `u_cf` artifacts |
| joint abduction | implemented | prefix-only posterior over twin and realized intervention |
| explicit causal operator | implemented | separate abduction, action, and prediction stages |
| controlled held-out gains | confirmed | shifted-contact RMSE `4.132 -> 0.805 mm`; coverage `77.9% -> 90.8%` |
| topology-aware identification boundary | confirmed diagnostic | exact node `75%`, one hop `100%`; all misses improve trajectory RMSE |
| public held-out action | confirmed for one Deform360 action | visual-only CD `47.58 mm` vs persistence `71.84 mm` |
| public short-prefix tactile state | rejected for that target | six-frame tactile CD `59.74 mm`, worse than visual-only |
| public source-backend transfer | rejected for first PokeFlex backend | `0/5` leave-one-take-out wins; `23.771` vs `10.093 mm` persistence |
| released physical uncertainty | diagnostic only | undercoverage and model-discrepancy-dominated headroom |

**Current decision:** these results are sufficient for a bounded first paper.
Additional public-data studies may improve breadth, but no uncollected physical
execution is a submission prerequisite.

## Empirical spine

### 1. Controlled latent-contact benchmark

Use the frozen shifted-contact result to establish the mechanism under shared
simulator exogenous conditions. Report RMSE, coverage, oracle-gap closure, and
held-out topology results. Do not describe this as real external validation.

### 2. Intervention identity versus predictive equivalence

Use the independent topology diagnostic to show that exact latent-label recovery
and future-query usefulness are different endpoints. The unchanged exact-node
gate fails; one-hop recovery and trajectory improvement are secondary. The
query-equivalence certificate formalizes this distinction prospectively but does
not imply physical equivalence.

### 3. Deform360 public held-out action

Use the frozen `001-rope` source/calibration/target split. Predictions are sealed
before target future masks and full target tactile are opened. The main public
comparison is:

| Method | Future CD [mm] | Future track [mm] |
| --- | ---: | ---: |
| Constant persistence | 71.84 | 78.87 |
| Visual-only contact | 47.58 | 60.16 |
| Six-frame tactile-conditioned `z` | 59.74 | 69.67 |
| Full-tactile oracle | 46.70 | 59.80 |

This supports one held-out public action for the exact reduced centerline model.
It does not establish pooling benefit, arbitrary-object transfer, material-point
tracking, commanded-versus-realized actuation recovery, or general tactile
benefit.

### 4. PokeFlex public negative control

Report the first sparse official-Warp backend as a retained negative source
gate. The method loses to persistence even under per-take source oracles, so the
sealed target is not opened. This is evidence that Causal4D's causal layer does
not rescue an incompetent physical backend.

### 5. Released single-interaction diagnostic

Use the existing released interaction only to localize limitations: modest
marginalization gain, severe undercoverage, failed transferred calibration, and
model-discrepancy-dominated oracle headroom. It is not an independent
confirmation cohort.

## Main comparison structure

The paper should keep comparison arms conceptually aligned where each dataset
supports them:

| Arm | Purpose |
| --- | --- |
| persistence or nominal physical baseline | unchanged predictive reference |
| nominal intervention | command treated as realization |
| Causal4D posterior prediction | realized-intervention marginalization |
| contact/action oracle | diagnostic upper bound only |
| rejected or unsupported backend | fail-closed negative control |

Primary endpoints are held-out trajectory error or a declared proper score at
the independent episode/take/seed level. Coverage, width, contact recovery,
fallback rate, and oracle attribution are supporting endpoints.

## Language discipline

Use the following terms precisely:

- **Controlled counterfactual prediction:** simulator exogenous conditions are
  shared across factual and alternative interventions.
- **Held-out interventional prediction on public records:** a future or action
  episode is withheld under a frozen public-data split.
- **Realized-intervention posterior:** a predictive latent intervention belief;
  it is not automatically verified physical contact or actuation.
- **Calibrated:** reserved for a declared held-out independent-unit calibration
  result. The released physical diagnostic is not calibrated.
- **Negative source gate:** a model class failed before target evaluation; this
  is evidence, not a missing result.

Do not claim individual-level real counterfactual ground truth, real contact
recovery without independent instrumentation, arbitrary-object generalization,
validated robot control, real Prob4D provider competence, or overall state of
the art.

## Paper structure

1. Problem: commanded actions do not identify realized physical interventions.
2. Model: uncertain deformable twin plus persistent and event-specific
   intervention variables.
3. Inference: factual abduction followed by explicit intervention and
   prediction.
4. Controlled evaluation: shifted contact, topology, and query-equivalence
   boundaries.
5. Public-data evaluation: Deform360 held-out action and PokeFlex negative source
   gate.
6. Released diagnostic and limitations: undercoverage, discrepancy, support,
   and generalization boundaries.

## Optional future hardware protocol

The registered same-object 18-session/36-execution protocol remains an immutable
future-validation design. Its current `0/36` state is neither a failed result nor
a submission blocker. It may be executed later by a collaborator with suitable
hardware, but it cannot be used to delay, rescue, or retroactively redefine the
public-data-only first paper.

## Submission criterion

The public-data-only manuscript is claim-complete when:

1. every numerical statement is bound to the checked controlled, Deform360,
   PokeFlex, or released-diagnostic artifact;
2. positive and negative public-data results are reported together;
3. no hardware-acquisition result is implied;
4. the exact-node, calibration, physical-equivalence, and generalization
   limitations remain explicit; and
5. the generated manuscript and evidence record reproduce in CI.
