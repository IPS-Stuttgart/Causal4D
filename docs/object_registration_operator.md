# Fixed-object registration

The first physical prerequisite after the v5 single-operator scaffold is
`object_registration.json`. It binds the physical sloth instance, the exact
PhysTwin model artifact, and the three preregistered canonical contact-node sets.
It does not inspect outcomes, operate hardware, or count as an execution.

Place each canonical node-set file below the fresh dataset root, then seal the
registration once:

```bash
causal4d protocol real object-registration-seal \
  /opt/causal4d-frozen/configs/causal4d/sloth_multi_action_v1.json \
  /mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1-v5 \
  --object-instance-serial '<physical serial>' \
  --phystwin-model-id '<model identifier>' \
  --phystwin-model-file /absolute/path/to/exact/model/artifact \
  --left-forepaw-node-set contact_node_sets/left_forepaw.json \
  --left-forepaw-node-count '<count>' \
  --right-forepaw-node-set contact_node_sets/right_forepaw.json \
  --right-forepaw-node-count '<count>' \
  --upper-torso-node-set contact_node_sets/upper_torso.json \
  --upper-torso-node-count '<count>'
```

The command:

- requires the exact scaffolded `protocol.json` and unchanged registration
  template;
- computes the model and node-set SHA-256 values from ordinary files;
- rejects symlinks and node-set files outside the dataset root;
- validates all registered fields before publication;
- writes `object_registration.json` atomically; and
- refuses replacement of any existing registration.

Use `--phystwin-model-sha256` instead of `--phystwin-model-file` only when the
precomputed digest has already been obtained from the exact immutable model
artifact. The node counts remain operator-supplied because the protocol does not
prescribe a serialization format for canonical node sets.

After sealing, recompute the registered next action with file-hash verification.
Do not select or revise node sets using source-panel or confirmatory outcome
quality.

## Contact approval remains separate

`object_registration.json` is the compact fixed-object and canonical-node-set
index. It is not the authoritative physical contact approval. Before the slip
pilot, complete `contact_registration.json` as a schema-4
`PhysicalContactRegistration` with the registered weighted patches, multiview
overlays, rejected alternatives, frame and support geometry, and two strictly
chronological review passes by the registered v5 self-attesting operator.

The registered next-action decision must therefore advance in this order:

```text
object_registration.json
-> contact_registration.json
-> slip_pilot.json
```

Under v5, this is disclosed single-operator self-review; it is not an
independent-attestation claim. The slip pilot must not start while the
authoritative contact registration is absent, invalid, or unapproved.
