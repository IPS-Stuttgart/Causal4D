# MolmoMotion → Bayesian physics bridge

This example is the lowest-effort integration path for a motion model such as
MolmoMotion:

1. keep the motion model frozen in its existing environment;
2. export sparse 3-D query trajectories and exact physical node identities;
3. normalize them through Causal4D's portable external-forecast contract;
4. score only the matching readout `H_Q(X)` of complete physical rollouts; and
5. retain the physical posterior exactly when semantic weight `beta` is zero.

The learned trajectory never overwrites simulator state, physical parameters,
contact variables, or model discrepancy.

## Install the lightweight consumer

From the Causal4D checkout:

```bash
python -m pip install -e .
```

The producer-side helper in this directory only needs NumPy, so it can also be
copied into an existing MolmoMotion environment.

## Run the self-contained export demo

From the repository root:

```bash
workdir=/tmp/molmomotion-physics-bridge
mkdir -p "${workdir}"

python examples/molmomotion_physics_bridge/make_demo_input.py \
  "${workdir}/molmo_raw.npz"

python examples/molmomotion_physics_bridge/export_molmo_forecast.py \
  "${workdir}/molmo_raw.npz" \
  "${workdir}/producer_forecast.npz" \
  "${workdir}/external_forecast_manifest.json" \
  --case-id single_lift_cloth \
  --source-revision demo-checkpoint \
  --anchor-physical-frame 70 \
  --physical-fps 30 \
  --forecast-fps 15 \
  --forecast 'instruction=Lift the cloth upward.' \
  --forecast 'paraphrase=Raise the cloth vertically with one hand.' \
  --forecast 'shuffled=Push the cloth sideways across the table.'

python -m causal4d.cli.external_forecast_import \
  "${workdir}/producer_forecast.npz" \
  "${workdir}/external_forecast_manifest.json" \
  "${workdir}/canonical_forecast.npz"
```

At 30 physical frames per second and 15 forecast frames per second, an anchor at
frame 70 maps the six forecast steps to physical frames 72, 74, ..., 82. The
importer verifies this mapping rather than inferring it from filenames.

## Export a real MolmoMotion result

After the existing MolmoMotion inference call, save one small NPZ:

```python
import numpy as np

# P exact PhysTwin object-node identities.
node_indices = np.asarray(node_indices, dtype=np.int64)              # (P,)

# Query positions at t0, in metres in the physical world frame.
anchor_positions_world_m = np.asarray(anchor_world, dtype=np.float64)  # (P, 3)

# Forecast order must match the repeated --forecast arguments below.
# K forecasts, P material points, F future timestamps, xyz coordinates.
future_positions_world_m = np.asarray(future_world, dtype=np.float64)  # (K,P,F,3)

validity_mask = np.all(np.isfinite(future_positions_world_m), axis=-1) # (K,P,F)

np.savez_compressed(
    "molmo_raw.npz",
    node_indices=node_indices,
    anchor_positions_world_m=anchor_positions_world_m,
    future_positions_world_m=future_positions_world_m,
    validity_mask=validity_mask,
)
```

Then run the export helper with the real case, checkpoint revision, anchor
frame, source rate, forecast rate, and captions. Eight, 16, and 32 points use the
same format; point count is not hard-coded. For the first result, use one fixed
eight-point query set and keep denser queries as a secondary ablation.

The helper deliberately accepts only metric world-frame trajectories. The
underlying generic importer also supports camera coordinates, `m`/`cm`/`mm`,
`PFC`/`FPC`/`KPFC`/`KFPC` layouts, explicit frame indices, and coordinate-level
validity; see [`docs/external_forecast_import.md`](../../docs/external_forecast_import.md).

## Import rollouts from an existing simulator

The simulator can remain in its own environment and only needs to export one
non-pickled NPZ:

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

Copy and edit `external_rollout_manifest_v1.json`, then run:

```bash
python -m causal4d.cli.external_rollout_import \
  producer_rollouts.npz \
  external_rollout_manifest.json \
  canonical_rollout_bank.npz

python -m causal4d.cli.external_bridge_doctor \
  canonical_forecast.npz \
  canonical_rollout_bank.npz \
  instruction \
  bridge_doctor.json
```

The doctor checks node identities, time overlap, anchor alignment, motion scale,
and the byte-identical zero-semantic-weight fallback before any positive
reweighting is attempted. See
[`docs/external_rollout_import.md`](../../docs/external_rollout_import.md).

## Generate the physical posterior

BayesianPhysTwin supplies uncertain state/parameter particles and PhysTwin/Warp
replay. Causal4D adds realized-intervention uncertainty such as contact,
actuation gain, delay, slip, and frame error. A typical sequence is:

```bash
causal4d evidence bpt-belief export \
  PHYSTWIN_REPO \
  CASE_DIRECTORY \
  parameter_profile.npz \
  refit_checkpoint.pt \
  belief.npz

causal4d experiment phystwin rollout-bank \
  PHYSTWIN_REPO \
  CASE_DIRECTORY \
  parameter_profile.npz \
  refit_checkpoint.pt \
  known_action_bank.npz \
  --action-setting known \
  --twin-belief belief.npz

causal4d experiment phystwin abduct-intervention \
  known_action_bank.npz \
  belief.npz \
  CASE_DIRECTORY/final_data.pkl \
  factual_intervention.npz \
  factual_evaluation.json

causal4d experiment phystwin counterfactual \
  PHYSTWIN_REPO \
  CASE_DIRECTORY \
  parameter_profile.npz \
  refit_checkpoint.pt \
  belief.npz \
  factual_intervention.npz \
  physical_posterior.npz \
  --counterfactual-action-id ACTION_ID \
  --contact-policy CONTACT_POLICY
```

`ACTION_ID` and `CONTACT_POLICY` must come from the selected registered case
configuration. For an initial cloth study, compare:

- `known`: true future controller trajectory is available to physics;
- `ambiguous`: the true action is one member of a feasible candidate set; and
- `hidden`: future controls are withheld and proposed from observed history.

The known-action setting tests whether MolmoMotion contributes deformation
information beyond recognizing the word “upward.”

## Apply the sparse forecast without changing physics

First prove the exact fallback:

```bash
causal4d experiment semantic build-task-posterior \
  physical_posterior.npz \
  "${workdir}/canonical_forecast.npz" \
  instruction \
  task_beta0.npz \
  --beta 0
```

The command reports `weights_bit_identical: true`. For an exploratory source-only
sweep:

```bash
for beta in 1 3 6 12; do
  causal4d experiment semantic build-task-posterior \
    physical_posterior.npz \
    "${workdir}/canonical_forecast.npz" \
    instruction \
    "task_beta${beta}.npz" \
    --beta "${beta}"
done
```

A positive beta is not admitted merely because the import succeeds. It requires
independent source-only competence and trust evidence. Rejection must return the
byte-identical physical weights.

## Add Prob4D only after the first bridge works

The released PhysTwin case already provides metric tracks and calibration, so
Prob4D is optional for the first experiment. In an RGB/video-only extension,
Prob4D can produce a causally sealed observation belief with covariance,
reliability, dependence, gauge uncertainty, and source lineage:

```bash
prob4d observation export-calibrated \
  outputs/single_lift_cloth/predictions.json \
  outputs/single_lift_cloth/observation_belief.npz \
  --case-id single_lift_cloth \
  --causal-frame-stop 71 \
  --metric-gauge-anchor calibration/metric_gauge_anchor.json \
  --gauge-covariance-calibration calibration/gauge.json \
  --point-uncertainty-calibration calibration/point.json \
  --source-revision "$(git rev-parse HEAD)" \
  --summary-json outputs/single_lift_cloth/observation_belief_summary.json

bpt observation validate \
  outputs/single_lift_cloth/observation_belief.npz
```

Prob4D evidence belongs on the observed prefix. A predicted MolmoMotion future
must remain a separate task/readout factor, not be relabeled as an independent
physical observation.

## Minimum comparison

Report zero motion, constant velocity, rigid hand transform, MolmoMotion,
nominal PhysTwin, BayesianPhysTwin, BayesianPhysTwin+Causal4D, and the gated
hybrid. Use 3-D ADE/FDE, full-object track or surface error, grasp slip,
strain/contact violations, predictive coverage, and runtime. The component
oracle is diagnostic only.
