# Deform360 official point-cloud source pilot V5 result

## Status

**Retained limited positive result; not selected as a headline paper result.**

The frozen internal gate passed, but the gain over the strongest simple baseline
was negligible and the evaluation covered only one object. This record preserves
both the numerical result and the decision not to overstate it.

## Question evaluated

The pilot asked whether a prefix-conditioned generalized-Bayesian damped local
transport model could improve short-horizon point-trajectory prediction on
released real Deform360 point clouds.

The experiment was deliberately bounded to a same-object, cross-action,
source-only observed-reset setting. It did not test held-out objects, target-domain
transfer, probe selection, a Prob4D uncertainty provider, or deployment safety.

## Dataset and protocol

- Dataset root:
  `/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository/processed`
- Object: `002-rope-silk`
- Source episodes: `0, 2, 5, 6, 7, 9`
- Source actions: lift side, drag side, lift sides, lift middle and side,
  drag sides, and fold
- Forbidden episodes, whose payloads were not opened: `1, 3, 4, 8`
- Input field used: `pts`
- Released velocity arrays used: no
- Selected persistent points per frame: 256
- Statistical unit: source episode/action
- Horizons: 1, 3, and 6 frames
- Training: equal-episode prior fitted on the other registered source actions
- Adaptation: causal-prefix-only generalized likelihood with capped temporal
  information
- Compared predictors: persistence, constant velocity, posterior MAP,
  posterior mean, and guarded posterior-or-persistence

The primary six-frame gate was fixed before execution:

1. at least 5% identity-RMSE improvement over persistence;
2. wins in at least four of six source actions; and
3. worst action-level RMSE ratio no larger than 1.10.

## Aggregate results

| Horizon | Guarded RMSE | Persistence RMSE | Improvement | Wins vs persistence | Worst ratio | Marginal 90% coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1 frame | 1.546324 mm | 1.610876 mm | 4.007% | 6/6 | 0.995853 | 97.448% |
| 3 frames | 3.878750 mm | 4.074407 mm | 4.802% | 6/6 | 0.994142 | 94.744% |
| 6 frames | 4.763945 mm | 5.045290 mm | 5.576% | 6/6 | 0.992638 | 89.275% |

The registered primary gate therefore passed. The result payload classified the
run as `source-only-real-point-cloud-positive-pilot` and retained
`paper_claim_authorized: false`.

At six frames, the guarded predictor also obtained a symmetric Chamfer error of
2.754864 mm and a mean full 90% marginal interval width of 4.464691 mm.
The guard accepted every update and the exact persistence fallback was never
used.

## Stronger-baseline comparison

The conclusion changes when the posterior is compared with constant-velocity
extrapolation rather than persistence.

| Six-frame metric | Guarded posterior | Constant velocity | Difference |
|---|---:|---:|---:|
| Identity RMSE | 4.763945 mm | 4.772428 mm | 0.008482 mm better; 0.178% |
| Symmetric Chamfer | 2.754864 mm | 2.702236 mm | 0.052628 mm worse; 1.948% |
| Action-level RMSE wins | 3/6 | 3/6 | tie |

Thus, the Bayesian method was not meaningfully better than a competent kinematic
baseline. The persistence comparison is positive, but it is not sufficient to
support a substantial forecasting contribution.

## Per-action six-frame result

| Episode | Action | Guarded RMSE | Persistence RMSE | Constant-velocity RMSE | Improvement vs persistence | Improvement vs constant velocity |
|---:|---|---:|---:|---:|---:|---:|
| 0 | lift side | 1.159885 mm | 1.708942 mm | 0.915739 mm | 32.128% | -26.661% |
| 2 | drag side | 0.747752 mm | 0.995631 mm | 0.903836 mm | 24.897% | 17.269% |
| 5 | lift sides | 0.678277 mm | 0.828893 mm | 0.749071 mm | 18.171% | 9.451% |
| 6 | lift middle and side | 1.848448 mm | 2.169298 mm | 2.017664 mm | 14.790% | 8.387% |
| 7 | drag sides | 1.056615 mm | 1.305015 mm | 0.966220 mm | 19.034% | -9.356% |
| 9 | fold | 23.092695 mm | 23.263964 mm | 23.082038 mm | 0.736% | -0.046% |

The fold action dominates the aggregate error and is essentially unchanged by
the posterior. The largest gains over persistence occur on the easier,
low-error actions. This further limits the scientific strength of the aggregate
5.576% improvement.

## Interpretation and paper decision

This experiment is useful as a real-data mechanics and interface sanity check:

- the complete execution succeeded on the registered self-hosted runner;
- all six source actions improved over persistence;
- the primary gate passed without an adverse worst-action result; and
- marginal 90% coverage at six frames was close to nominal at 89.275%.

It is not sufficient as a central paper result because:

- only one physical object and material were evaluated;
- source actions from one object do not establish unseen-object generalization;
- the identity-RMSE advantage over constant velocity was only 0.178%;
- constant velocity obtained the better symmetric Chamfer score;
- the method beat constant velocity on only three of six actions;
- the difficult fold action showed only a 0.736% gain over persistence;
- the exact fallback mechanism was not exercised;
- released reconstructed trajectories were used directly, so the experiment
  does not validate Prob4D as an uncertainty provider; and
- the protocol did not test task-conditioned physical probe selection or
  probe-target dependence.

**Decision:** retain the result for provenance and, at most, as an appendix-level
feasibility result. Do not use it as evidence for a major real-world forecasting,
uncertainty-calibration, held-out-transfer, or active-causal-design claim.
Further work on this exact same-object prediction setup is not prioritized unless
a revised experiment introduces a materially stronger claim and beats constant
velocity by a practically meaningful margin.

## Reproducibility and provenance

- Repository revision: `a95027963d757637ee6134e5624d192678c4bc0c`
- Workflow: `Deform360 official point-cloud source pilot V5`
- Workflow file:
  [`.github/workflows/deform360-official-pcd-source-pilot-v5.yml`](../.github/workflows/deform360-official-pcd-source-pilot-v5.yml)
- Workflow run: [33331692891](https://github.com/IPS-Stuttgart/Causal4D/actions/runs/33331692891)
- Artifact ID: `9737849569`
- Artifact name:
  `deform360-official-pcd-source-pilot-v5-a95027963d757637ee6134e5624d192678c4bc0c`
- Artifact archive SHA-256:
  `d690b43770289a532e1f60abf3dfb6682d3f31f2117f63de19cae27f492cf02f`
- Raw `result.json` SHA-256:
  `ce36bf4a6e0f9136b8f712bddfd62f45de09c1ceb684da2841da892e09579be8`
- Result payload digest:
  `9de10389671a6fb2660ea2739f696658330d34b19621a9b2b5431e9ed4b7f34f`
- Frozen module-file SHA-256:
  `b4cc4014adf44b12f120219e6a4508bb1e7a201e728ba8ad53f8e7592b9eb68c`
- Runtime: Python 3.10.12, NumPy 2.2.6
- Runner name: `workstation1`, selected through the `gpuserver4090` labels
- Hardware reported by the run: two NVIDIA GeForce RTX 4090 GPUs

### Evidence-manifest caveat

The uploaded `SHA256SUMS` correctly records the final `result.json`,
`runtime.json`, and `status.json`. It does not completely seal the final bundle:
its `run.log` entry contains the empty-file digest, while the uploaded log is
nonempty, and `runner.txt` was written after the manifest and is absent from it.
The workflow-run identity, artifact archive digest, raw result-file digest, and
validated internal result digest remain available above. The manifest itself
must not be described as a complete checksum of every final artifact member.
