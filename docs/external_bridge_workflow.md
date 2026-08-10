# External forecast–physics bridge workflow

This workflow is the collaborator-facing path for combining a sparse learned
future such as MolmoMotion with a finite bank of complete physical rollouts.
The motion model and simulator remain in their existing environments. Causal4D
consumes only non-pickled NumPy artifacts and never requires simulator gradients.

Run the complete command family through one module:

```bash
python -m causal4d.cli.external_bridge_workflow --help
```

## Three inputs

### 1. Sparse future forecast

Use the existing external-forecast contract:

```bash
python -m causal4d.cli.external_bridge_workflow import-forecast \
  producer_forecast.npz \
  external_forecast_manifest.json \
  canonical_forecast.npz
```

The forecast contains persistent material/node IDs, metric anchors, explicit
future times, one or more caption/sample trajectories, and validity masks.

### 2. Complete physical candidates

Export the external simulator support as:

```python
np.savez_compressed(
    "producer_rollouts.npz",
    node_ids=node_ids,                         # [N]
    trajectories_world_m=trajectories_world_m, # [R,T,N,3]
    rollout_weights=rollout_weights,           # [R]
    frame_times_s=frame_times_s,                # [T], includes t0
    rollout_ids=rollout_ids,                    # [R], optional
    parameter_values=parameter_values,          # [R,D], optional
)
```

Then import it:

```bash
python -m causal4d.cli.external_bridge_workflow import-rollouts \
  producer_rollouts.npz \
  external_rollout_manifest.json \
  canonical_rollout_bank.npz
```

### 3. Optional evaluation reference

For automatic ADE/FDE, coverage, constant-velocity, and component-oracle
reporting, provide a closed NPZ:

```python
np.savez_compressed(
    "reference.npz",
    case_id=np.asarray("single_lift_cloth"),
    node_ids=node_ids,                         # [N]
    positions_world_m=positions_world_m,       # [T,N,3]
    frame_times_s=frame_times_s,                # [T]
    validity_mask=validity_mask,                # [T,N] or [T,N,3], optional
)
```

Reference data are evaluation-only. They are not consumed as semantic evidence
and are never used to alter the physical rollout support.

## Audit material-point correspondence

When the visual query does not already contain exact simulator node IDs, run a
one-to-one geometric assignment:

```bash
python -m causal4d.cli.external_bridge_workflow map-nodes \
  query_anchors.npz \
  simulator_nodes.npz \
  query_node_mapping.json \
  --output-npz query_node_mapping.npz \
  --output-svg query_node_mapping.svg \
  --maximum-distance-m 0.005
```

Default keys are:

```text
query_anchors.npz:    anchor_positions_world_m
simulator_nodes.npz:  node_positions_world_m, node_ids
```

Use key options when producer names differ. The command solves a global
minimum-total-distance one-to-one assignment and returns exit status `3` when
any assignment exceeds the frozen tolerance. The report explicitly states that
geometric proximity is an audited convenience, not proof of material identity;
folded or self-contacting cloth should use exact material labels whenever
possible.

## Preflight doctor

Before any positive semantic weighting:

```bash
python -m causal4d.cli.external_bridge_workflow doctor \
  canonical_forecast.npz \
  canonical_rollout_bank.npz \
  instruction \
  bridge_doctor.json \
  --strict-warnings
```

The doctor checks case identity, exact node membership, timeline overlap,
interpolation, anchor mismatch, forecast-to-rollout motion scale, validity
coverage, and byte-identical beta-zero fallback.

## One-command beta sweep and report

```bash
python -m causal4d.cli.external_bridge_workflow run \
  canonical_forecast.npz \
  canonical_rollout_bank.npz \
  instruction \
  bridge_results \
  --reference reference.npz \
  --beta 0 --beta 1 --beta 3 --beta 6 --beta 12 \
  --scale-m 0.05
```

The result directory contains:

```text
bridge_results/
├── doctor.json
├── summary.json
├── summary.md
├── metrics.csv
├── weights.csv
├── error_vs_horizon.csv
├── error_vs_horizon.svg
├── predictions.npz
└── manifest.json
```

The manifest is written last and binds the exact hashes and sizes of every
published companion artifact.

Without a reference trajectory, the command still publishes the preflight,
physical-versus-semantic weight shift, effective support size, KL divergence,
posterior means, intervals, and exact beta-zero fallback. With a reference, it
also reports:

- zero-motion and, when two prefix frames are available, constant-velocity
  baselines;
- the external forecast itself;
- physical-prior and semantic-reweighted ADE, FDE, coordinate RMSE, and
  empirical coordinate coverage;
- error versus horizon;
- a component oracle labelled diagnostic-only; and
- an `evaluation_only_best_beta` that is explicitly prohibited from serving as
  deployment calibration.

Use `--require-clean-doctor` to return exit status `3` after publishing the
complete report when preflight warnings remain.

## Trust and claim boundary

The beta sweep is exploratory. Import compatibility, a clean doctor report, or
an evaluation-only best beta does not establish semantic competence. Positive
semantic trust must be selected on disjoint source executions and confirmed on
an independent panel. If the trust gate rejects, beta must be zero and the
physical weights remain byte-identical.

Prob4D belongs on the observed prefix when multi-view, overlapping-window,
gauge, or correlated reconstruction uncertainty is material. A predicted
MolmoMotion future remains a separate task factor and must not be relabelled as
an independent physical observation.
