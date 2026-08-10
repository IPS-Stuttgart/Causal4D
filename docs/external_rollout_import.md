# External physical rollout-bank import

Causal4D can ingest a finite bank of complete trajectories from an external MPM,
IPC, FEM, mass-spring, or other forward simulator without importing that
simulator or differentiating through it.

The first portable contract intentionally supports a flat rollout axis:

```text
R complete candidates × T frames × N persistent nodes × C coordinates
```

Each candidate becomes one discrete `JointRolloutBank` hypothesis. Physical
parameters can be retained as metadata for every candidate. More structured
hypothesis-by-parameter support can already be expressed directly with
`JointRolloutBank`; the flat importer is the low-burden collaborator boundary.

## Producer NPZ

A minimal producer file contains:

```python
import numpy as np

np.savez_compressed(
    "producer_rollouts.npz",
    node_ids=node_ids,                         # integer [N]
    trajectories_world_m=trajectories_world_m, # float [R,T,N,3]
    rollout_weights=rollout_weights,           # float [R]
    frame_times_s=frame_times_s,                # float [T]
    rollout_ids=rollout_ids,                    # text [R], optional
    parameter_values=parameter_values,          # float [R,D], optional
)
```

Requirements:

- node identities are persistent and unique;
- frame times are in seconds and strictly increasing;
- `anchor_time_s` in the manifest matches exactly one frame;
- candidate weights are finite, nonnegative, and have positive total mass;
- trajectories are complete and finite; and
- object arrays and pickle are forbidden.

## Import manifest v1

```json
{
  "schema": "causal4d.external_rollout_import",
  "schema_version": 1,
  "case_id": "single_lift_cloth",
  "source": {
    "simulator": "external-mpm",
    "revision": "git-or-configuration-revision",
    "artifact_id": "optional-upstream-run-id"
  },
  "arrays": {
    "node_ids": "node_ids",
    "trajectories": "trajectories_world_m",
    "frame_times_s": "frame_times_s",
    "rollout_weights": "rollout_weights",
    "rollout_ids": "rollout_ids",
    "parameter_values": "parameter_values"
  },
  "layout": "RTNC",
  "coordinate_frame": "world",
  "position_unit": "m",
  "anchor_time_s": 0.0,
  "parameter_names": [
    "young_modulus_pa",
    "damping",
    "table_friction",
    "grasp_stiffness"
  ],
  "variance_floor_m2": 1e-6,
  "confidence_level": 0.9,
  "metadata": {
    "action_setting": "known"
  }
}
```

The importer accepts `m`, `cm`, and `mm`. Camera-frame 3-D trajectories are also
accepted when `arrays.camera_to_world` names a finite rigid `4 x 4` transform;
its translation is expressed in metres.

## Import and preflight validation

```bash
python -m causal4d.cli.external_rollout_import \
  producer_rollouts.npz \
  external_rollout_manifest.json \
  canonical_rollout_bank.npz
```

Then compare the canonical bank with a canonical external forecast:

```bash
python -m causal4d.cli.external_bridge_doctor \
  canonical_forecast.npz \
  canonical_rollout_bank.npz \
  instruction \
  bridge_doctor.json
```

The doctor validates:

- case identity;
- exact forecast-node membership in the rollout bank;
- forecast/rollout time overlap;
- interpolation frame indices;
- anchor mismatch;
- forecast-to-rollout motion-scale ratio;
- validity-mask coverage; and
- byte-identical physical weights at zero semantic weight.

Use `--strict-warnings` when anchor or motion-scale warnings should return exit
status `3` while retaining a complete JSON report.

## Scientific boundary

Import success establishes only contract compatibility. It does not establish
that the rollout support is physically calibrated, that a positive semantic
weight is beneficial, or that an external forecast is an independent physical
observation. Positive semantic trust still requires source-only competence and
held-out confirmation; rejection must preserve the physical weights exactly.
