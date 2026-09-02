# Tracking Cloth V2 query-conditioned observation: terminal source-gate negative

Status: **source-gate-failed-target-closed-v2**.

The exact-main workflow completed successfully on the `gpuserver4090`-labelled
self-hosted runner. The negative classification is scientific, not technical:
cotton alone selected the two-marker policies and fitted all prediction models;
those frozen models then failed the independent denim gate. Wool and polyester
trajectory contents remained unopened.

## Immutable execution

- Causal4D merge revision: `1802cd544a04fb1eaaf4342cb68d516e4e4ef100`
- workflow run: `33588742638`
- self-hosted job: `100118290916`
- result ID: `c7220ab39d869d95fe691e0ac58a6f595203f5999c78730f068e8cb79c02420f`
- artifact ID: `9831004928`
- artifact archive SHA-256: `8f75c19531e5c1df44e747a828143d957912d2c2fcf6a86074e8e15b7d1dec75`
- registered request ID: `328737f73b038c81f0a9b7d4d1ed9a40e40d5cbad214e85812b69c6405f0277e`
- dataset: official Tracking Cloth Deformation v1, 120 CSV recordings
- source fit: cotton
- independent source gate: denim
- unopened target materials: wool and polyester

## Denim gate

The gate evaluated 29 complete denim recordings and 87 recording--horizon
rows. Complete recordings, after averaging the three horizons, were the gate
units.

| Policy | Equal-recording RMSE [mm] | Gaussian NLL | marginal 90% coverage | normalized joint NEES |
|---|---:|---:|---:|---:|
| exact constant velocity | 50.8147 | -8.4138 | 88.86% | 2.980 |
| source mean residual | 50.5720 | -8.2798 | 89.14% | 3.090 |
| dependence destroyed | 50.8112 | -8.3413 | 89.13% | 3.061 |
| global-state conditioned | 84.1911 | 0.0233 | 87.62% | 7.752 |
| task conditioned | 113.6809 | 14.1452 | 88.58% | 14.954 |

Task-conditioned MSE was:

- `-82.324%` relative improvement versus the global-state policy;
- `-400.491%` versus exact constant velocity;
- `-400.559%` versus the dependence-destroyed control;
- better than the global-state policy on `44.828%` of complete recordings.

Task and global-state selection differed in eight scenario--horizon cells, so
the failure is not a trivial identical-policy result. Scenario-level task/global
MSE ratios were:

- shake: `1.9058`;
- twist: `1.8964`;
- table: `0.6753`;
- self-collision: `1.0000`.

## Interpretation

The cotton-fitted two-marker residual map does not transfer to denim. The
task-conditioned coupling amplifies error most strongly for shake and twist.
The dependence-destroyed arm is almost identical to exact constant velocity,
which localizes the failure to the transported observation--query coupling
rather than to merely revealing two marker values. The tablecloth subgroup is
the only favorable aggregate scenario, but it cannot rescue the frozen overall
gate.

This result rules out presenting passive cotton-to-denim marker selection as a
positive active-perception result. Threshold relaxation, target opening, or
subgroup-only promotion is not authorized by this evidence.

## Claim boundary

This is a terminal negative result for the exact V2 passive observation model.
It does not invalidate the finite-certificate `act / probe / fallback`
implementation, query-identifiability results on DEFORM, or finite-orbit
failure-prevention result. It does show that an active physical-twin paper needs
a source-qualified intervention-response model and cannot infer probe value
from a weak cross-material residual regression.
