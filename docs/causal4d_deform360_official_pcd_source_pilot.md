# Deform360 official point-cloud source pilot

This experiment is a bounded source-only real-data diagnostic on released
Deform360 `pcd_clean.tar` archives. It is separate from the terminal
`causal4d-deform360-shared-physics-replication-v1` route and does not revise that
registered result.

## Question

Can a generalized-Bayesian damped local transport belief, updated only from the
causal prefix of a held source episode, improve short-horizon prediction of the
released persistent point trajectories relative to exact persistence?

The pilot is an interface and mechanics sanity check. It is not yet the full
cross-object task-conditioned probe-selection experiment.

## Frozen data boundary

- Dataset revision: `7fea8e20231a47641d1d2bc8791920ec4e62ec5e`.
- Object: `002-rope-silk`.
- Source episodes: `0, 2, 5, 6, 7, 9`.
- Forbidden episodes: `1, 3, 4, 8`.
- Input archive: `processed-repository/processed/002-rope-silk/episode_<n>/pcd_clean.tar`.
- Only the `pts` array is read from each NPZ member; released velocity arrays are
  deliberately ignored.
- Forbidden episode payloads are never opened.
- The mounted dataset is read-only and no new physical data are collected.

## Evaluation

Each source action is held out in turn. The remaining source actions define an
equal-episode Gaussian prior over a scalar velocity-persistence coefficient. The
held episode contributes only its prefix to a capped generalized likelihood. A
posterior predictive mean and marginal variance are then evaluated at horizons
of 1, 3, and 6 frames.

Comparators are exact persistence, constant velocity, posterior MAP, posterior
mean, and a source-calibrated guarded posterior that returns exact persistence
when the guard rejects the update. Frames, point coordinates, and resets remain
nested within the episode; the source episode/action is the statistical unit.

The primary six-frame decision requires all of the following:

- at least 5% mean RMSE improvement over persistence;
- wins on at least four of six source episodes;
- no episode worse than 1.10 times persistence.

A positive result supports only a same-object, cross-action, source-only pilot on
released reconstructed point trajectories. It does not establish held-out target
transfer, unseen-object generalization, calibrated real physical uncertainty,
Prob4D provider competence, online robot probing, or safety.

## Validation and execution

The source implementation and focused synthetic tests are checked with the pinned
repository Ruff formatter, Ruff linting, mypy, and pytest before the request may
reach the self-hosted lane.

The reviewed workflow runs on
`[self-hosted, Linux, X64, nvidia-smi, gpuserver4090]` and reads
`/mnt/seagate10tb/florianpfaff/datasets/deform360/processed-repository/processed`.
It is triggered by the canonical request file
`ops/deform360-official-pcd-source-pilot-request.json` after merge to `main`.
