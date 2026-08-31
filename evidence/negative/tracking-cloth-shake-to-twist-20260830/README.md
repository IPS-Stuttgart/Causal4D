# Negative evidence: Tracking Cloth shake-to-twist transfer

**Record:** `tracking-cloth-shake-to-twist-20260830`  
**Status:** completed negative result  
**Paper claim authorized:** no  
**New experiment required by this record:** no

## Bottom line

On the public **Tracking Cloth Deformation** dataset, the registered reduced
spring-mesh transfer from shaking recordings to held-out twisting forecasts
did **not** beat state persistence.

The best physics arm was `map_physics`:

| Arm | Specimen-balanced RMSE [mm] | Coordinate NLL | 90% coverage |
| --- | ---: | ---: | ---: |
| persistence | **75.1451** | **-1.7340** | **92.72%** |
| map physics | 119.1681 | -0.7879 | 76.13% |
| Bayesian physics | 132.2905 | -0.5454 | 70.41% |
| last residual | 141.0674 | -0.5661 | 80.03% |
| guarded Bayesian physics | 156.4626 | -0.4849 | 70.82% |
| nominal physics | 208.7978 | -0.6192 | 75.75% |

The best physics RMSE was **44.0230 mm higher than persistence**, or
**58.58% worse relative to persistence**.

This is a completed result. It is not a TODO, does not require new data
collection, and must not be omitted merely because the outcome is negative.

## Registered information boundary

Study: `tracking-cloth-shake-to-twist-pilot-v1`

- 120 verified public recordings.
- 32 shaking recordings used as source data.
- 32 twisting recordings used as target data.
- 56 collision recordings reserved and not numerically read.
- One-second all-marker initialization prefix.
- Five-second forecast interval.
- Known future measured corner positions supplied as boundary inputs.
- Predictions for all 32 targets sealed before target scoring.
- Primary inference unit: eight material-size specimens, with recordings
  aggregated within specimens.
- Raw recordings were not uploaded to GitHub.

The experiment is a reduced spring-mesh pilot. It is not a PhysTwin/FEM
reproduction, command-conditioned rollout, or fully online robot forecast.

## Guard result

The source-only guard accepted six of eight specimen candidates, corresponding
to 24 of 32 target recordings; eight recordings used exact fallback.

There were no harmful accepted records relative to **nominal physics** and no
exact-fallback violations. This does not imply safety relative to the stronger
persistence comparator: guarded Bayesian physics remained substantially worse
than persistence.

## Retrospective active-probe replay

Study: `tracking-cloth-task-directed-active-probe-v1`

A later logged-probe replay also remained negative:

- primary one-probe task-directed RMSE: **119.8653 mm**;
- persistence RMSE: **75.1451 mm**;
- regret: **44.7202 mm**;
- specimen-bootstrap 95% interval for regret:
  **[39.6057, 49.7935] mm**;
- losses versus persistence: **8/8 specimens**;
- best exploratory budget among 0, 1, 2, and 4 probes:
  **119.6666 mm**, still **59.25% worse** than persistence;
- task-directed and parameter-information policies selected the same one-probe
  condition for all eight specimens, so the mechanism-discrimination gate
  failed.

This replay had prior target-outcome exposure and is therefore a retrospective
diagnostic, not fresh confirmation.

## Scientific interpretation

This record supports the following statement:

> Under the registered public-data shake-to-twist protocol, the reduced
> spring-mesh physics backend did not transfer well enough to beat persistence,
> and the logged active-probe replay did not rescue it.

It does **not** support any of the following broader claims:

- Causal4D fails generally.
- Physics-informed deformable-object prediction is universally inferior.
- The covariance model is jointly calibrated.
- The result establishes online-control, safety, material-identification,
  unseen-object, or state-of-the-art performance.
- Later, differently specified protocols may overwrite this result.

The appropriate paper use is as a negative real-world transfer result,
limitation, or failure-mode study.

## Provenance

Primary evaluation:

- GitHub Actions run:
  [`33302686759`](https://github.com/IPS-Stuttgart/Causal4D/actions/runs/33302686759)
- evaluated commit: `0cd567b7b640f5e0b73eba8c1eeb1acb1f09f4c1`
- protocol ID:
  `4dfcabe6cca23b07244676441e61cc13d113ea7631a2af32707f69af55e00515`
- prediction-seal SHA-256:
  `db6156a5bbf6fdf536e8b37cef00e80b5b68b20833fa19001ea97ea6dba97e77`
- metrics SHA-256:
  `496cd70ab4221985897b8a768510ff5ccf3691aae2099895cadc034867af38c1`

Active-probe replay:

- GitHub Actions run:
  [`33319276977`](https://github.com/IPS-Stuttgart/Causal4D/actions/runs/33319276977)
- evaluated commit: `f8d5f16aa1b104108e6876db40bf2ed9c2e526fa`
- protocol ID:
  `ad01b8abc081999b7e042fb2563aaad84197ecbb48f1d9f5e09d322206c4af1c`
- prediction-seal SHA-256:
  `8dc08bd35dc434b98d103db94454bfad19957273a1a34131948cb4bf6310e8f2`
- metrics SHA-256:
  `2692d907558b520ed38f65807aece60f4827656057d449326cd65d7e83ec5898`

Downloaded evidence-package hashes:

```text
44156a6adcef956c88508cb53b013add462bab248ad79905bf73a19ec7151ec8  tracking-cloth-evaluate.zip
51ea6d6b39cd2a41f75c0ba4d734535c2e2c043f0bc1d83a38033f074f22d1ce  tracking-cloth-active-evaluate.zip
```

See [`result.json`](result.json) for the machine-readable record.
