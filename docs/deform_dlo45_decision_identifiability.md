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
3. The Bayesian continuation and retain-state continuation are scored against the
   suffix of every source-supported hypothesis. These losses, not the held-out
   evaluation suffix, define the exact BayesianPhysTwin certificate.
4. Causal4D authorizes a unique robust action, otherwise a unique tolerance-
   admissible action, otherwise the exact retain-state fallback.
5. Decision records, action losses, prediction hashes, and an NPZ of all candidate
   and selected predictions are written and hashed.
6. The public target file has necessarily been decoded in this retrospective
   analysis, and its registered sequence length is available as horizon metadata.
   No held-out suffix value is supplied to alignment, prediction, source-loss
   construction, certification, or action selection. Suffix scoring starts only
   after the pre-outcome seal exists.

## Frozen primary arm

The strict primary was fixed before the first official-evaluation scoring. It places
all 56 positive-support source hypotheses in one quotient class and uses a 1 mm
worst-case-regret tolerance. This deliberately permits arbitrary within-class mass
allocation and therefore provides a severe test of decision identifiability.

The first complete primary execution was claim-eligible but inconclusive: one of 28
ambiguous targets admitted an update, 27 returned exact fallback, no admitted update
was harmful, and selected RMSE was slightly below exact retention. Those values are
retained rather than redefined after outcome access.

## Explicitly exploratory timing quotient

After reviewing the conservative primary result, a secondary arm was added. It is
marked `post-primary-retrospective-exploratory` in the request, every case record,
and the evidence summary. It cannot support a confirmatory claim on DLO4/DLO5.

The fixed rule partitions source hypotheses using only the sign of their prefix-
fitted delay:

- negative delay: early timing regime;
- zero delay: nominal timing regime;
- positive delay: late timing regime.

The registered quotient mass of each timing regime is the sum of the prefix-only
posterior weights of its member hypotheses. Exact worst-case regret is still taken
over every positive-support hypothesis within each timing regime. The exploratory
regret tolerance is the round physical value 10 mm. No target suffix value enters
the partition, quotient masses, losses, certificate, or action selection. Because
the class rule and tolerance were selected after primary-outcome review, only an
untouched cohort can confirm them.

## Frozen endpoints

Both arms report:

- finite-action certification rate;
- update, retain, and exact-fallback counts;
- certification despite at least 1 mm source-supported future ambiguity;
- held-out RMSE of selected, always-update, always-retain, expected-source-loss,
  single-hypothesis completion, and the two-action oracle;
- realized decision regret;
- harmful certified-update rate and Wilson interval;
- a 20,000-replicate DLO-stratified trajectory bootstrap interval for selected
  improvement over retention;
- separate DLO4 and DLO5 means and win/tie/loss counts.

Claim eligibility applies to the strict primary only and requires all 70 files per
DLO, exact recovery of the publisher's 56/14 split, no load failure or harmonization
discard, and all 28 official evaluation targets. The exploratory arm is always
recorded as non-confirmatory, regardless of its numerical outcome.

## Interpretation boundary

A positive strict result would support the narrow empirical statement that a finite
source-supported forecast decision can be certified despite nonzero real-trajectory
future ambiguity. An exploratory timing signal can only motivate a frozen test on an
untouched cohort. Neither establishes unique physical-state identification, causal
effects of unexecuted controls, closed-loop safety, prospective robot performance,
or population-level transfer to unseen physical objects. DOT remains the intended
untouched confirmation cohort.

## Reproduction

The reviewed request file is
`ops/deform-dlo45-public-gpuserver4090-request.json`. Changing that file on `main`
triggers `.github/workflows/deform-dlo45-public-gpuserver4090.yml`. The workflow
checks out the exact Causal4D revision and exact BayesianPhysTwin revision pinned by
`requirements/ci/bayesian-phystwin-guarded-provider.sha`, verifies the official
56-train/14-eval split for both DLOs, uses no GitHub secrets, hashes every DLO4/DLO5
source file, and uploads the complete evidence directory.

A hosted parity control checks out the exact official DEFORM Git revision and runs
the same evaluator. The `gpuserver4090` run remains the authoritative local-byte
verification because it uses the separately checksum-verified mounted copy.
