# Approved anatomical node sets v8

This source-only bundle preserves Florian Pfaff's two-pass anatomical review of
the fixed PhysTwin graph used by the locked sloth protocol. The final selections
are:

| Protocol region | Candidate | Canonical nodes |
| --- | --- | ---: |
| anatomical left forepaw | `L1` | 37 |
| anatomical right forepaw | `P1` | 108 |
| upper torso between the shoulders | `F2` | 26 |

The approval artifact binds the exact node-set files and four review overlays.
Its canonical artifact ID is
`fa1b5c9425fd2d3d1a590664aac5ef146fbb3829bddacc562b4544b06b9e3885`.
The selected PhysTwin checkpoint has SHA-256
`e7b853f8369ccb5b0d56dee0991fd6e95482a2baa37a913fc7f4b22db93044ad`.

This bundle approves anatomy only. It does not claim that a physical contact
patch, frame transform, support surface, or physical object serial has been
measured. It does not create `object_registration.json`, authorize the slip
pilot, inspect target outcomes, or send a physical command.

Prepare a source-only seal packet with:

```bash
python scripts/ci/prepare_object_registration_seal_packet.py \
  --protocol configs/causal4d/sloth_multi_action_v1.json \
  --evidence-root evidence/object-registration-anatomy-v8 \
  --phystwin-model-id phystwin-single_lift_sloth-best_199 \
  --phystwin-model-sha256 \
    e7b853f8369ccb5b0d56dee0991fd6e95482a2baa37a913fc7f4b22db93044ad \
  --output /tmp/object_registration.seal_packet.v8.json
```

Without `--object-instance-serial`, the packet deliberately records
`ready_to_seal_object_registration=false`. Supply only the stable inventory
identifier attached to the exact physical sloth; do not substitute a test
placeholder or the logical protocol object ID.

The committed pending packet has canonical packet ID
`7685d393f0ab0d31342213dd44a5d453b610096c61570397fc2c71124a2cdeac`
and file SHA-256
`088887536ebd77c77c995ac2e29eadbabe224e350cc151ba814cdad8e0e203ac`.
