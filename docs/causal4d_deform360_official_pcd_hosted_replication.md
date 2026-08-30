# Deform360 official point-cloud hosted replication

This one-shot workflow independently reproduces the frozen source-only
Deform360 point-cloud pilot on a GitHub-hosted runner. It exists because the
primary `gpuserver4090` evaluation can remain queued when the self-hosted runner
is unavailable.

The replication does not replace the primary self-hosted run. It downloads the
same six registered `002-rope-silk` `pcd_clean.tar` archives from the public
`brownu/deform360_processed` repository, verifies each byte size against the
metadata-only `gpuserver4090` inventory from GitHub Actions run `33322193007`,
and executes the unchanged method protocol
`causal4d-deform360-official-pcd-source-pilot-v1`.

## Frozen source boundary

- Source episodes: `0, 2, 5, 6, 7, 9`.
- Forbidden episodes: `1, 3, 4, 8`.
- Expected source bytes: `369807360`.
- The public repository `main` revision is resolved exactly once before any
  source archive is downloaded; the returned 40-hex revision is then pinned for
  all six downloads and recorded in the evidence bundle.
- Forbidden archives are neither downloaded nor opened.
- Only `pts` arrays are consumed by the unchanged pilot; released velocity
  arrays remain ignored.

## Interpretation

Agreement between this hosted lane and the primary self-hosted lane would show
that the metric-bearing pilot does not depend on a private runner environment.
A positive pilot still supports only a same-object, cross-action, source-only
short-horizon result on released reconstructed point trajectories. It does not
authorize the broader task-conditioned probe-selection paper claim, unseen-object
transfer, Prob4D provider validation, online robot probing, or deployment safety.
