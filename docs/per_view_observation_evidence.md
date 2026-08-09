# Per-view observation evidence

Causal4D can optionally retain a content-addressed per-view evidence manifest
beside the fused RGB-D observation used by the registered estimator. The
contract preserves enough raw evidence to distinguish view-specific
observation bias from coherent physical-model discrepancy after collection.

This is an acquisition and diagnosis improvement. It does not change the
frozen estimator, the six-frame information boundary, the registered
18-session/36-execution design, any quality threshold, or any target identity.
The existing required artifact inventory remains unchanged.

## Why retain per-view evidence

A fused 3-D track is sufficient to replay the registered predictor, but it can
hide whether a residual is:

- coherent across cameras and therefore compatible with state or physics error;
- concentrated in one camera and therefore compatible with calibration,
  occlusion, or reconstruction error;
- correlated with continuous observation confidence;
- stable in the object frame but not the world frame; or
- aligned with a surface normal rather than tangential motion.

The per-view contract retains the inputs needed for those later diagnostics
without allowing them to influence the primary registered analysis.

## Contract

`causal4d.per-view-observation-evidence` version 1 binds:

- the protocol design, execution, session, clock domain, frame count, and causal
  prefix stop;
- the observation producer and exact software-environment capsule;
- the camera-calibration revision and ordered camera-key inventory;
- a common coordinate frame and material-point identity contract;
- the object-frame definition and time-indexed object-from-world transform;
- continuous confidence semantics;
- at least two ordered camera views;
- synchronized RGB, depth, timestamps, material points, validity masks, and
  confidence for every view;
- optional surface normals with an explicit unavailable reason;
- commanded control, measured actuation, gripper state, support registration,
  and reset/drift/slip evidence;
- optional contact-wrench evidence with an explicit unavailable reason; and
- the fused observation as a derived product of the complete ordered camera
  inventory, never as the sole retained observation.

Every file descriptor contains a safe relative path, SHA-256 digest, byte
count, and media type. Time-indexed descriptors additionally contain the clock
domain and sample count. Artifact paths must be unique inside the manifest.

## Information boundary

The contract requires all of the following:

```text
causal_prefix_frame_start = 0
causal_prefix_frame_stop < frame_count
raw_full_execution_retained = true
future_frames_retained_for_blind_evaluation = true
future_frames_used_for_inference = false
target_outcomes_used_for_inference = false
target_outcomes_used_for_model_selection = false
target_outcomes_used_for_exclusion = false
target_outcomes_used_for_calibration = false
fused_observation_is_sole_retained_evidence = false
```

Retaining the complete execution is therefore allowed for later blind
measurement, while use of the held-out future for inference, selection,
exclusion, or calibration is forbidden.

## Execution-manifest integration

New execution templates contain the optional descriptor:

```json
{
  "per_view_observation_evidence": {
    "path": null,
    "sha256": null,
    "bytes": null
  }
}
```

Leaving all three values null preserves the historical execution contract. When
`path` is populated, execution validation loads the nested manifest and binds
it to:

- the exact protocol and design digest;
- the execution and session IDs;
- the synchronized RGB-D clock domain;
- the registered frame count; and
- `intervention_frame + o_plus_prefix_frames` as the causal-prefix stop.

With file verification enabled, validation also checks every nested artifact
beneath the execution root through the ordinary-file, no-symlink boundary.
Partial optional descriptors are rejected.

## Python API

```python
from causal4d.per_view_observation_evidence import (
    build_per_view_observation_evidence,
    load_per_view_observation_evidence,
    validate_per_view_observation_evidence,
    write_per_view_observation_evidence,
)
```

Build the manifest only after the complete source descriptors are available.
Publication is atomic and exactly once by default:

```python
manifest = build_per_view_observation_evidence(
    protocol_id=protocol_id,
    protocol_design_sha256=design_sha256,
    execution_id=execution_id,
    session_id=session_id,
    clock_domain_id=clock_id,
    frame_count=frame_count,
    common_coordinate_frame="world",
    material_identity_contract="frame-0 material indices retained across views",
    observation_producer=producer,
    camera_calibration=calibration,
    object_frame=object_frame,
    confidence_semantics=confidence_semantics,
    views=views,
    shared_sensors=shared_sensors,
    fused_observation=fused_observation,
    information_boundary=information_boundary,
)
write_per_view_observation_evidence(
    execution_root / "per-view-observation.json",
    manifest,
)
```

Load and verify it relative to the execution root:

```python
load_per_view_observation_evidence(
    "per-view-observation.json",
    artifact_root=execution_root,
    verify_files=True,
    expected_protocol_id=protocol_id,
    expected_protocol_design_sha256=design_sha256,
    expected_execution_id=execution_id,
    expected_session_id=session_id,
    expected_clock_domain_id=clock_id,
    expected_frame_count=frame_count,
    expected_causal_prefix_frame_stop=prefix_stop,
)
```

## Scientific boundary

The manifest is evidence-retention infrastructure, not a result. It cannot
increment the confirmatory execution count, alter inclusion, select a model,
fit calibration, admit Prob4D, or rescue a failed registered endpoint. It makes
future observation-versus-physics attribution possible only after genuine
per-view artifacts have been collected and independently validated.
