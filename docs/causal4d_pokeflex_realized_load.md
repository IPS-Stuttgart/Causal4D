# PokeFlex realized-load prefix forecasting

## Purpose

This study tests one public-data-only consequence of the Causal4D causal model:
a recorded tool trajectory does not by itself identify the load and contact state
realized by a deformable object.  The estimator observes a short factual wrench
response, abducts a posterior over a finite source-take/gain/delay intervention
bank, and forecasts the remaining measured load and contact state.

No robot, new acquisition, or simulated target outcome is used.  The study reads
only PokeFlex records already released by the dataset authors.

## Frozen source stage

The first stage is restricted to the five metadata-selected development takes of
`3dPrintedBunny`:

```text
3dPrintedBunny_T1
3dPrintedBunny_T3
3dPrintedBunny_T4
3dPrintedBunny_T6
3dPrintedBunny_T7
```

The prior source QA identity is
`e09d36db4e1ba8a38c70e112c3af9ab95516ee245302f71a853f36cd2dd0e0e7`.
It established aligned robot/wrench records and pose--surface contact support.
The calibration take `T5` and target take `T2` remain unopened.  The source
workflow never names or parses their payloads; any fallback layout discovery
examines path names only.

Each development take is evaluated once as a complete held-out unit.  The other
four takes form the source bank.  Frames, force coordinates, posterior
candidates, and time samples are nested observations and are not treated as
independent replicates.

## Factual prefix and future query

Contact onset is the first three consecutive frames whose released force-axis
value exceeds 3 N.  The factual response consists of the first six frames from
that onset.  The registered target is the following 48-frame force-axis and
binary contact trajectory.

The complete released tool trajectory over the registered window is supplied as
conditioning evidence.  It is a measured kinematic record, not an independently
logged command and not counterfactual ground truth.

The forecast constructor receives only:

- the six observed force values;
- the target tool-trajectory phase and speed; and
- the four complete source takes.

Its target-conditioning type contains no future-force field.  Predictions and
content hashes are written for all five folds before suffix scoring begins.

## Estimator

Source load profiles are expressed on a phase coordinate combining normalized
time and cumulative tool-path length.  For each source take, gain in
`[0.60, 1.40]` and response delay in `[-3, 3]` frames define one realized-load
hypothesis.  A Student-t prefix likelihood with source-derived scale produces a
posterior over the finite bank.  The posterior mean and total mixture variance
are the principal forecast.

The comparison set is:

1. last-prefix-value persistence;
2. linear prefix extrapolation;
3. source-mean profile with prefix offset;
4. a ridge predictor conditioned on the released tool kinematics;
5. the posterior MAP intervention;
6. the posterior mixture; and
7. a dependence-destroyed control.

The dependence control cyclically reassigns source prefixes to source suffixes.
It preserves the empirical prefix and suffix profile marginals while removing
their matched relation.  Its posterior weights are still computed from the
original source prefixes.  A gain over this arm therefore cannot be attributed
only to the marginal set of possible prefixes or futures.

## Metrics and source gate

For each complete held-out take, the study reports force RMSE and MAE, Gaussian
NLL, nominal 90% coverage and width, peak/mean force error, and contact Brier and
log scores.

The backend passes the development source gate only when every frozen criterion
holds:

- all five complete development takes contribute;
- every fold has a prediction seal before scoring;
- mean force RMSE is at least 5% below persistence;
- the posterior wins on at least 60% of takes;
- mean force RMSE is at least 2% below the dependence-destroyed control;
- no take is worse than 1.10 times persistence RMSE; and
- mean RMSE is no more than 1.05 times the kinematics-conditioned ridge arm.

A negative or bounded result is terminal for this method version.  A positive
source gate still does not open `T5` or `T2`: calibration and target access would
require a separate reviewed and content-addressed protocol.

## Reproduction

The source gate requires a manual dispatch from `main` and consumes the exact
reviewed request file:

```text
ops/pokeflex-realized-load-source-gpuserver6000-request.json
```

The self-hosted workflow runs read-only on `gpuserver6000` against
`/mnt/lexar4tb/pokeflex` and uploads only compact predictions, seals, metrics,
and provenance:

```text
.github/workflows/pokeflex-realized-load-source-gpuserver6000.yml
```

The implementation and synthetic boundary tests are:

```text
src/causal4d_public/pokeflex_realized_load.py
src/causal4d_public/cli/pokeflex_realized_load_source.py
tests/test_pokeflex_realized_load.py
```

## Claim boundary

A positive development result would justify testing this exact backend under a
separate public calibration/target protocol.  It would not itself establish a
held-out target effect, persistent material identity, mesh forecasting,
population-level calibration, commanded-versus-measured actuation recovery,
individual counterfactual ground truth, robot control, or safety.
