# DEFORM DLO4/DLO5 decision-identifiability evaluation

This evaluation tests the finite-action consequence of the decision-identifiability
certificate on released real deformable-linear-object trajectories. It is a
retrospective public-data analysis. It does not collect data, issue robot commands,
or infer counterfactual outcomes for actions that were not executed in the source
dataset.

## Scientific question

An official evaluation trajectory may have several source-supported future
continuations after its observed prefix. The experiment asks whether that unresolved
continuation ambiguity nevertheless permits one finite forecast decision:

1. `apply_bayesian_update`: continue with the source-panel Bayesian posterior mean;
2. `retain_observed_state`: retain the final observed prefix state;
3. `fallback_retain_observed_state`: the caller-owned exact fallback, numerically
   identical to retaining the observed state but outside the candidate action
   roster.

The exact fallback is returned whenever Causal4D cannot consume one unique robust
or tolerance-admissible BayesianPhysTwin certificate action.

## Publisher-defined split and independent units

The workflow reads the checksum-verified public DLO4 and DLO5 holdings at
`/mnt/seagate10tb/florianpfaff/datasets/deform/data_set/` on `gpuserver4090`.
For each DLO, the official repository contains 56 files under `train/` and 14 files
under `eval/`. The training files supply the source-supported hypothesis panel; the
14 evaluation files are held-out targets. No synthetic 14-by-5 action grouping is
assumed.

The primary reporting unit is one complete official evaluation trajectory. The 28
target trajectories are nested within only two physical DLOs, so trajectory-level
intervals are descriptive and do not establish population-level object
generalization. Results are also reported separately for DLO4 and DLO5.

The official files are pickle payloads. Loading them is permitted only when the
reviewed request asserts that the mounted files are the checksum-verified official
copy. The evaluator records every load failure and never silently substitutes for a
missing or malformed trajectory.

## Target-closed construction

For each DLO and each of its 14 official evaluation trajectories:

1. Only the first 30% of the evaluation trajectory is supplied to the pre-outcome
   builder.
2. Each of the 56 official training trajectories is aligned to that prefix over a
   frozen grid of temporal delays and scalar gains. Offset, delay, gain, likelihood,
   and posterior weight use the prefix only.
3. The 56 aligned training trajectories form the finite prior-supported hypothesis
   set. They are registered as one prefix-compatible quotient class.
4. The Bayesian continuation and retain-state continuation are scored against the
   suffix of every source-supported hypothesis. These losses, not the held-out
   evaluation suffix, define the exact BayesianPhysTwin certificate.
5. Causal4D authorizes a unique robust action, otherwise a unique 1 mm
   tolerance-admissible action, otherwise the exact retain-state fallback.
6. Decision records, action losses, prediction hashes, and an NPZ of all candidate
   and selected predictions are written and hashed.
7. The public target file has necessarily been decoded in this retrospective
   analysis, and its registered sequence length is available as horizon metadata.
   No held-out suffix value is supplied to alignment, prediction, source-loss
   construction, certification, or action selection. Suffix scoring starts only
   after the pre-outcome seal exists.

The one-class quotient is intentionally conservative: posterior weights may form
the Bayesian update prediction, but the certificate maximizes regret over every
positive-prior training trajectory rather than trusting a within-class posterior
allocation.

## Frozen primary endpoints

The artifact reports:

- finite-action certification rate;
- update, retain, and exact-fallback counts;
- certification despite at least 1 mm source-supported future ambiguity;
- held-out RMSE of the selected rule, always-update, always-retain, expected-source-
  loss selection, single-hypothesis completion, and the two-action oracle;
- realized decision regret;
- harmful certified-update rate, where a certified update is worse than exact
  retention on the held-out suffix;
- a Wilson interval for harmful certified updates;
- a 20,000-replicate DLO-stratified trajectory bootstrap interval for selected
  improvement over retention;
- separate DLO4 and DLO5 means and win/tie/loss counts.

A result is claim-eligible only when both DLOs contribute all 70 usable files, the
publisher-defined 56/14 path split is recovered exactly, and harmonization discards
no file. A positive retrospective result additionally requires at least one
certified update in an ambiguous case, a nonnegative lower endpoint for the
trajectory bootstrap, nonnegative mean improvement in both DLO strata, and a
harmful-update Wilson upper endpoint no greater than 10%.

## Interpretation boundary

A positive result would support the narrow empirical statement that a finite
source-supported forecast decision can be certified despite nonzero real-trajectory
future ambiguity. It would not establish unique physical-state identification,
causal effects of unexecuted controls, closed-loop safety, prospective robot
performance, or population-level transfer to unseen physical objects. DOT remains
the appropriate untouched cohort for a later frozen confirmation after the action
roster and analysis are accepted.

## Reproduction

The reviewed request file is
`ops/deform-dlo45-public-gpuserver4090-request.json`. Changing that file on `main`
triggers `.github/workflows/deform-dlo45-public-gpuserver4090.yml`. The workflow
checks out the exact Causal4D revision and exact BayesianPhysTwin revision pinned by
`requirements/ci/bayesian-phystwin-guarded-provider.sha`, verifies the official
56-train/14-eval split for both DLOs, uses no GitHub secrets, hashes every DLO4/DLO5
source file, and uploads the complete evidence directory.
