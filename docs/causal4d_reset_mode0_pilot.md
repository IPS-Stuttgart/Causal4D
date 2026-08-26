# Fresh-reset mode-0 cross-check

The v5 pre-acquisition amendment preregisters a scale check for the released
`single_lift` state result. This check is descriptive and target-blind. It asks
whether fresh-reset variation in graph mode zero is commensurate with the
released 13.736 mm per-node vector RMS correction. It does not select a model,
change the slip gate, or establish that reset error caused the released result.

Run the check only on fresh-reset pilot sessions, before any confirmatory
execution. Positions must remain in the locked world frame; do not align each
reset independently before supplying the input.

## Input contract

The input NPZ contains:

```text
session_ids                       (reset,) string
reference_positions_world_m       (node, 3) float
reset_positions_world_m           (reset, node, 3) float
graph_mode0                        (node,) float
registration_uncertainty_95_m      scalar float
world_frame_id                     scalar string
units                              scalar string, exactly "m"
positions_are_pre_alignment        scalar bool, true
fresh_reset_mask                   (reset,) bool, all true
data_role                          scalar string,
                                   "preacquisition_fresh_reset_pilot"
target_outcomes_used               scalar bool, false
object_registration_sha256         scalar SHA-256
contact_registration_sha256        scalar SHA-256
```

At least five unique fresh-reset sessions are required, matching the minimum
slip-pilot execution count. The node count must equal the registered 6,895-node
released reference. The supplied mode must match the released basis's constant
mode-zero direction up to scale and sign; an arbitrary or nonconstant vector is
rejected. The tool also verifies that the frozen initial mode energy divided by
6,895 yields the frozen per-node RMS exactly. Projection uses the unweighted
Euclidean node inner product.

## Frozen estimator

For each reset, the tool projects the locked-frame displacement onto mode zero
and computes per-node vector RMS. It takes the empirical 95th percentile using
NumPy's conservative `higher` order-statistic rule, then adds the preregistered
95% registration uncertainty:

```text
pilot statistic
= higher-quantile-0.95(fresh-reset mode-0 RMS)
+ registration uncertainty 95% bound
```

The reset-scale explanation is weakened when the released 13.736 mm reference
is greater than twice this statistic. Otherwise the result is only
`scale_compatible`; compatibility never confirms reset error as the cause.

The report also records, per session, locked-frame translation, the best-fit
SE(3) component, the post-SE(3) residual, and its remaining mode-0 component.
These are secondary decompositions and cannot revise the registered decision.

## Command

```bash
causal4d protocol reset-mode0-crosscheck \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/preacquisition/reset-pilot.npz \
  /data/causal4d-sloth-multi-action-v1/preacquisition/reset-mode0-crosscheck.json
```

The command validates the complete canonical protocol-v2-v3-v4-v5 chain,
then validates approved schema-4 `object_registration.json` and
`contact_registration.json`, including their source-file hashes and registered
two-pass review. The NPZ must name those exact artifact digests. The command
hashes the NPZ and its principal arrays and publishes the JSON exactly once. It
rejects target use, post-alignment positions, non-fresh resets, duplicate
session IDs, nonfinite arrays, a wrong node count, registration mismatch, or
replacement of an existing result.

For v5 readiness this report is a required component of pilot completion. The
operator flow cannot advance to the 12-run source panel until it has replayed
the registered NPZ and reproduced the report exactly. A missing report remains
an incomplete prerequisite; a report, registration, or NPZ mismatch is invalid
evidence and stops the workflow.
