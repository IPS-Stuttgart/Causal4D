# Target-free real-experiment design sensitivity

## Purpose

`causal4d.real_design_sensitivity` evaluates the operating characteristics of
the already registered real-effect interval rule before target outcomes are
opened. The independent unit is one complete physical session. Frames, nodes,
coordinates, camera views, and posterior components are never counted as
independent replicates.

The audit calls the exact production functions from
`causal4d.real_analysis_intervals`:

1. the session bootstrap-t interval is primary;
2. the Student-t mean interval is the required robustness interval;
3. both lower bounds must be strictly positive for a positive claim;
4. the Student-t interval may veto but cannot rescue the primary interval; and
5. a degenerate or non-estimable panel fails closed.

It does not read physical data, choose a new threshold, change the frozen
36-execution method, or authorize target access.

## Default sensitivity panel

The default run evaluates sample counts 18, 15, and 12 to expose precision loss
from smaller endpoints or preregistered exclusions. Mean effects are expressed
in assumed between-session standard-deviation units. Five zero-mean session
models are included:

- Gaussian effects;
- variance-standardized Student-t effects with five degrees of freedom;
- a variance-standardized 10% contaminated Gaussian mixture;
- a centered, variance-standardized skewed lognormal distribution; and
- one adverse session per panel, with a mean-preserving offset.

For every scenario and sample count, the report records:

- the null positive-gate rate;
- the rate of estimable, nondegenerate panels;
- median and 90th-percentile primary interval width;
- positive-gate probability across the fixed effect grid;
- Monte Carlo standard errors;
- the probability that at least one session is nonpositive; and
- the first tested effect reaching the declared target power, when any does.

The final quantity is a grid result, not an interpolated minimum detectable
effect.

## Running the audit

Run the deterministic default audit from an installed package or checkout:

```bash
python -m causal4d.real_design_sensitivity \
  --output-json build/causal4d-real-design-sensitivity.json
```

A smaller smoke run can reduce only the Monte Carlo budgets:

```bash
python -m causal4d.real_design_sensitivity \
  --simulation-replicates 100 \
  --bootstrap-replicates 500 \
  --output-json build/causal4d-real-design-sensitivity-smoke.json
```

Custom sample counts, scenarios, effect grids, and target power are available
through `RealDesignSensitivityConfig` in Python. The JSON output is finite,
key-sorted, content-addressed, and published without overwrite unless
`--overwrite` is supplied.

## Interpretation boundary

The report supports statements about expected precision under declared
session-level distributions. It cannot establish the real effect, empirical
coverage, physical benefit, or an admissible new threshold. A low predicted
probability of passing the positive gate does not authorize inspecting target
outcomes early or changing the registered method.

For the frozen experiment, archive the chosen target-free report with the final
method freeze and software environment. Do not rerun it with target-informed
assumptions after acquisition begins.
