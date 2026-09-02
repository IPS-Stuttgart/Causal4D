# Deform360 source-only sensor-reveal audit

## Purpose

This protocol is the first public-data step after the proof-carrying sensor-reveal mechanism study. It asks whether the mounted Deform360 `002-rope-silk` source cohort contains enough synchronized physical carriers to construct finite optional-sensor messages and to test task-conditioned routing without reading any protected episode.

It is deliberately an **audit and source-side diagnostic**, not a held-target result. A passing run authorizes the next source-only step: fit complete finite sequential policies on training episodes, seal every possible sensor-outcome branch, and replay those policies on a disjoint source episode before any target object is considered.

## Frozen cohort

The cohort reuses the previously registered official point-cloud source split:

- source episodes: `0, 2, 5, 6, 7, 9`;
- forbidden episodes: `1, 3, 4, 8`;
- five deterministic reset points per source episode;
- six causal prefix frames per sensor message; and
- a six-frame future robot-translation carrier.

Complete episodes, not frames or sensor pixels, are the cross-validation unit.

## Carrier boundary

The audit opens only source-episode robot and tactile values. It inventories point-cloud carriers from an uncompressed `pcd_clean.tar` by reading TAR headers only; it does not extract or decode an NPZ member. It does not decode raw camera video.

For each source episode, it requires:

1. a point-cloud frame carrier;
2. synchronized `openings` and `T_worlds` robot arrays;
3. at least two tactile groups present in every source episode; and
4. enough common frames for all registered prefix and future windows.

The source sensor groups are:

- robot opening range over the causal prefix;
- accumulated end-effector translation over the causal prefix; and
- accumulated mean-absolute tactile energy for every tactile group shared by all source episodes.

## Finite messages and grouped diagnostic

Each scalar prefix feature is converted to a three-outcome message using source-tercile thresholds. The future carrier is converted to a three-action routing target using independently refitted training-fold thresholds.

The output reports:

- outcome occupancy and mutual information on the complete source support;
- same-support lookup accuracy as a descriptive upper bound;
- leave-one-complete-episode-out conditional-mode accuracy;
- training-fold majority baselines;
- unseen sensor-outcome-key counts; and
- the ten best fixed sensor pairs under grouped evaluation.

The codebooks are refitted inside every held-episode fold. Frames from one episode never appear in both training and evaluation.

## Fail-closed behavior

The readiness gate is false when any registered source point-cloud or robot carrier is absent, fewer than two tactile groups are common to all source episodes, or the expected `6 × 5 = 30` reset cases cannot be formed. An unavailable modality is not silently imputed.

The output is content-addressed, generated twice, byte-compared, and independently revalidated before artifact upload.

## Self-hosted execution

The workflow has a hosted contract job for pull requests. The data-reading job can allocate `gpuserver4090` only from a reviewed `main` revision, triggered by the exact request-file change or manual dispatch. It checks out `github.sha`, verifies the clean tree, uses no secrets, and treats all dataset paths as read-only.

## Claim boundary

A passing audit establishes source-side carrier availability, finite-message occupancy, and grouped source routing evidence. It does not establish held-target performance, object-disjoint transport, causal value of a real sensor acquisition, optimal sequential policy, online execution, deployment authorization, or safety.
