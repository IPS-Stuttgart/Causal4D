# Per-view residual localization

`causal4d.per_view_residual_localization` provides a prefix-only diagnostic for
the per-camera evidence retained by the physical acquisition contract. Its
purpose is to avoid interpreting camera-specific reconstruction error as a
physical or state discrepancy.

For view `v`, prefix frame `t`, material node `n`, and coordinate `c`, the
diagnostic fits

```text
observed[v,t,n,c] - predicted[t,n,c]
  = view_contrast[v,c]
  + shared_frame_offset[t,c]
  + graph_field[t,n,c]
  + residual[v,t,n,c].
```

The graph field is represented in a supplied node basis. Every basis mode is
centered across nodes and RMS-normalized before fitting, which prevents a global
translation from being assigned simultaneously to the graph and shared-frame
terms. One highest-support view is selected as the reference, so view terms are
contrasts rather than absolute camera biases.

## Inputs and causal boundary

- observed material points: `(view, frame, node, 3)`;
- predicted material points: `(frame, node, 3)`;
- Boolean validity: `(view, frame, node)`;
- optional continuous confidence in `[0, 1]`;
- optional graph basis: `(node, mode)`;
- an exact evidence artifact ID; and
- an optional exclusive causal-prefix stop.

Only the selected prefix is hashed and fitted. Future observations can remain
retained for blind evaluation without affecting the result. Every view and
every selected prefix frame must contribute positive weighted support.

## Reported diagnostics

The content-addressed result contains immutable arrays for the fitted
reference-view contrasts, shared frame offsets, graph coefficients and graph
field. It also reports per-view RMS residuals before and after fitting, support
counts, weighted sums of squared errors for nested ablations, and the unique
explained fraction attributable to each diagnostic family.

The dominant-source label is conservative:

- `view_specific`: relative camera/view effects dominate;
- `shared_frame`: a common frame/gauge translation dominates;
- `object_coherent`: a centered graph field dominates;
- `mixed`: the leading diagnostic families are too close to separate; or
- `unresolved`: the fitted decomposition explains too little energy.

A pre-allocation design-memory guard fails before constructing an oversized
dense regression matrix.

## Interpretation

This decomposition is a localization aid, not a causal conclusion. A strong
object-coherent fraction supports investigation of a readout/rest-geometry
correction, but it does not by itself prove simulator-state discrepancy.
Likewise, a view-specific fraction may reflect calibration, occlusion,
association, confidence calibration, or reconstruction bias.

The result explicitly records `target_outcomes_used=false` and
`physical_evidence_increment=0`. It does not change the frozen estimator, the
registered 36-execution protocol, exclusion rules, covariance calibration, or
the paper's physical evidence count.
