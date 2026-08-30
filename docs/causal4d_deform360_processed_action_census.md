# Deform360 processed action-schema census

## Purpose

This one-shot, source-only census decides whether the mounted processed
Deform360 release has enough episode-level action and reset semantics to define
a prospective object-disjoint experiment for Causal4D issue #442.

The census is deliberately narrower than a prediction experiment. It opens only
per-episode `metadata.json` files and reports action labels, bimanual indicators,
and explicit reset/initial-state identifiers. It does not open point clouds,
robot arrays, tactile arrays, videos, or target scores.

## Why this gate is needed

A larger Prob4D/BayesianPhysTwin/Causal4D contribution requires more than lower
average trajectory error. The claim-bearing experiment must test whether a
matched dependence-bearing belief changes a safe intervention decision and
improves a registered held-out physical query. Logged episodes can support that
claim only when candidate probes and challenge actions have frozen semantics and
comparable starting conditions.

The mounted processed bundle was previously admitted by audit
`7ad71382a8234676aa02f203317bc8788aba96da5a2eb2a9ee12f1693ae89d80`.
That audit found 62 objects and 584 processed episodes but intentionally did not
open file contents. This census is the next bounded source step.

## Dispositions

The workflow emits exactly one classification:

- `physical-probe-metadata-identifiable`: action diversity and explicit shared
  reset groups satisfy the frozen metadata gates. Geometry and robot-state
  equivalence still need a separate source-only gate before target access.
- `observation-selection-only`: action semantics are populated, but reset-group
  evidence is insufficient. The admissible experiment is then task-conditioned
  observation, view, or temporal-window acquisition—not counterfactual physical
  probing.
- `metadata-insufficient`: no claim-bearing roster can be frozen from this
  source. The route stops before geometry futures or target scores are opened.

A favorable classification is not a paper result and does not authorize target
access.

## Frozen exclusions

Objects already used in score-bearing or method-development work are marked as
opened and removed from prospective split capacity:

- `001-rope`
- `002-rope-silk`
- `003-cable`
- `004-rubber-band`

The census still reports their metadata for release-level schema diagnosis. A
later experiment manifest must add every other object whose outcomes were opened
before freezing the final source/calibration/target split.

## Run

The request file is
`ops/deform360-processed-action-census-request.json`. A change to that file on
branch `science/deform360-processed-action-census-v1` runs the workflow on the
self-hosted `gpuserver4090` label. Outputs are written under `RUNNER_TEMP` and
uploaded as a GitHub Actions artifact; the dataset root remains read-only.
