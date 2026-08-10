# Post-freeze calibration diagnostics

## Scope

These diagnostics are deliberately additive. They do **not** alter the registered
36-execution real estimator, the frozen graph-persistence baseline, the target
information boundary, or the method-freeze rules.

They address two questions that remain useful after the primary method is frozen:

1. does the finite Bayesian contact/physics update recover draws from its own
   declared controlled generative model; and
2. can graph discrepancy uncertainty grow with horizon without fitting one dense
   cross-mode transition matrix that may become unstable or transfer poorly?

Neither diagnostic substitutes for the independent-execution calibration gate.

## Simulation-based calibration

`causal4d.simulation_calibration.run_contact_rollout_sbc` performs randomized-rank
simulation-based calibration (SBC) directly on a `ContactRolloutBank`.

For each trial it:

1. samples one joint contact/physical-parameter component from the bank prior;
2. treats that component trajectory as controlled ground truth;
3. adds Gaussian observation noise;
4. reruns the ordinary prefix-only `ContactRolloutBank.update_weights` update;
5. computes randomized posterior PIT/rank values for the complete joint state,
   contact state, and each physical-parameter coordinate; and
6. records posterior concentration, entropy reduction, effective sample size,
   true-component posterior mass, and parameter error.

Example:

```python
from causal4d.simulation_calibration import run_contact_rollout_sbc

result = run_contact_rollout_sbc(
    bank,
    trials=5000,
    prefix_frame_count=prefix,
    likelihood_scale_m=0.0015,
    likelihood_power=1.0,
    dynamic_likelihood_weight=0.0,
    observation_noise_std_m=0.0015,
    seed=7,
    bin_count=10,
)
print(result.as_dict())
```

When `likelihood_power=1`, `dynamic_likelihood_weight=0`, and the simulated
observation standard deviation equals `likelihood_scale_m`, the experiment is an
exact self-consistency check for the finite independent-Gaussian prefix likelihood.
Approximately uniform randomized ranks are then expected up to Monte Carlo error.
Changing likelihood power, adding derivative terms, or deliberately mismatching
noise creates a sensitivity diagnostic rather than an exact SBC null.

### Controlled benchmark command

The leave-one-topology benchmark can run the exact self-consistency diagnostic in
one invocation without changing benchmark defaults:

```bash
causal4d benchmark latent-contact \
  --seeds 0:5 \
  --sbc-trials-per-fold 5000 \
  --sbc-bins 10 \
  --output-dir runs/causal4d-latent-contact-v1
```

The normal benchmark artifacts are unchanged. The opt-in SBC result is published
atomically to `OUTPUT_DIR/sbc.json` unless `--sbc-output-json` is supplied. The
command refuses to replace an existing SBC artifact by default and performs that
check before running the expensive benchmark. Replacement requires the explicit
flag:

```bash
causal4d benchmark latent-contact \
  --seeds 0:5 \
  --sbc-trials-per-fold 5000 \
  --sbc-output-json runs/controlled-sbc.json \
  --overwrite-sbc-output
```

The no-overwrite publication is race-safe: a destination created after the initial
check still causes a clean failure rather than replacement. JSON serialization
forbids non-finite values and the atomic writer does not leave a partial destination
on process or machine failure.

Every saved result contains a `producer` object with the Causal4D distribution and
version, the runtime diagnostic module name, and the SHA-256 of the exact loaded
module bytes. This supplements the benchmark configuration and seed inventory with
a direct implementation identity.

Each seed contains three leave-one-topology-out folds. The target object's
physical-parameter posterior is fitted from its training/validation interactions,
the contact prior is learned from the other two topologies exactly as in the
controlled latent-contact study, and SBC truths are sampled from that finite bank's
own declared joint prior.

`causal4d.controlled_latent_contact_sbc.run_controlled_latent_contact_sbc` exposes
the same operation as a Python API. The aggregate report sums rank histograms across
folds and uses trial-count weighting for posterior summaries; it never relabels
folds, frames, points, or coordinates as independent physical executions.

### Interpretation boundary

Passing SBC means that the finite inference implementation is self-consistent under
its own controlled model. It does **not** establish:

- real observation-model validity;
- real simulator adequacy;
- transfer across actions, contacts, objects, or acquisition sessions;
- calibrated physical deployment uncertainty; or
- Prob4D/BayesianPhysTwin provider competence.

This distinction is useful for current Causal4D failure attribution: if SBC is well
behaved while real coverage remains poor, the remaining evidence points toward
model/discrepancy shift rather than a basic finite-posterior implementation error.

## Mode-wise graph discrepancy dynamics

`causal4d.graph_modewise_discrepancy` supplies a conservative stochastic alternative
to the dense learned graph transition used by the existing diagnostic.

For graph coefficient `a_(j,t)` of mode `j`, it fits

```text
a_(j,t+1) = rho_j a_(j,t) + epsilon_(j,t)
0 <= rho_j <= 1
```

with one retention coefficient per graph mode and a separate innovation variance for
each spatial coordinate. The fit pools the three coordinates when estimating
`rho_j` but does not couple different graph modes.

A persistence prior shrinks `rho_j` toward one. The prior is represented as a
fraction of observed source energy, so its effect is invariant to a uniform change
of coefficient scale. `persistence_prior_weight=0` gives ordinary clipped AR(1)
fitting; larger values move the model continuously toward graph persistence.

Example:

```python
from causal4d.graph_modewise_discrepancy import (
    fit_modewise_graph_discrepancy,
    forecast_modewise_graph_discrepancy,
)

dynamics = fit_modewise_graph_discrepancy(
    graph_model,
    source_residual_m,
    source_valid,
    persistence_prior_weight=0.25,
)

mean, variance = forecast_modewise_graph_discrepancy(
    graph_model,
    dynamics,
    prefix_residual_m,
    prefix_valid,
    total_frame_count=target_frame_count,
)
```

The forecast starts with zero latent-coefficient uncertainty at the last observed
prefix coefficient and recursively accumulates per-mode innovation variance. The
existing projection variance remains as an irreducible node-coordinate floor. This
produces explicit horizon-dependent uncertainty without requiring a dense cross-mode
covariance transition.

### Real diagnostic comparison

The existing graph-discrepancy command evaluates all five arms in one evidence
bundle:

```text
current_random_walk_readout
state_only
graph_persistence
graph_modewise
graph_temporal
```

Use the unchanged diagnostic route:

```bash
causal4d diagnostic discrepancy graph-temporal \
  physical.npz final_data.pkl optimal_params.pkl parameter_profile.npz \
  graph_discrepancy.json \
  --modewise-persistence-prior-weight 0.25 \
  --modewise-minimum-retention 0.0 \
  --modewise-maximum-retention 1.0
```

The mode-wise fit reads O-minus only. The target contributes only the same declared
O-plus prefix used by the persistence and dense-AR arms. The saved model NPZ binds
per-mode retention, per-coordinate innovation variance, and shrinkage settings; the
moments NPZ adds `graph_modewise_mean_m` and `graph_modewise_variance_m2` without
storing held-out future labels.

## Prospective use

For any future post-freeze study, compare at least:

- graph persistence;
- the existing dense learned graph transition; and
- mode-wise stochastic dynamics with a source-frozen persistence prior.

Select the prior weight and retention bounds using controlled/source data only.
Report held-out trajectory accuracy, proper scores, horizon coverage, interval width,
worst-group coverage, and the full negative result when persistence still wins.

Do not promote the mode-wise model into the registered 36-execution primary method.
A future primary-method change requires a new protocol/version under the existing
method-freeze rules.
