# Pre-acquisition next-action packet

The readiness workflow can publish one exactly-once ZIP packet that binds the
current machine-readable next-action decision to the human-readable operator
instructions derived from that same decision.

Create a packet only from the current hash-verified evidence tree:

```bash
causal4d protocol readiness next-action-packet \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/operator-handoff/next-action.zip
```

The command returns exit code `3` while the evidence is valid but the first
confirmatory execution is not yet authorized. The packet is still complete and
useful: it contains exactly the next permissible scaffold, repair, acquisition,
review, freeze, or verification action. Invalid evidence returns exit code `2`.

## Packet contents

The deterministic archive contains exactly three uncompressed members in a
fixed order:

```text
decision.json
instructions.md
manifest.json
```

`manifest.json` binds:

- the protocol, design, pre-acquisition plan, and amendment identities;
- the decision evidence and host-local status digests;
- the action, category, and registered execution identity;
- the SHA-256 and byte count of `decision.json` and `instructions.md`;
- whether physical acquisition is required; and
- the explicit boundaries that target outcomes and registered-method changes are
  forbidden.

The packet ID is the canonical SHA-256 of the logical manifest. ZIP timestamps,
permissions, member order, and compression are fixed so the same decision
produces identical packet bytes.

Packet publication is exactly once. An existing destination is never replaced.
Use a new path after the readiness evidence changes.

## Validate immediately before use

```bash
causal4d protocol readiness next-action-packet-validate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/operator-handoff/next-action.zip \
  --output-json \
  /data/operator-handoff/next-action-validation.json
```

Validation fails closed unless:

1. the archive has the exact registered member set and order;
2. every retained member matches its manifest digest and byte count;
3. `instructions.md` is exactly regenerated from `decision.json`;
4. the decision hashes are internally valid;
5. the decision names the current checkout and evidence tree; and
6. rebuilding the hash-verified next action produces the same logical decision
   and action identity.

This prevents a stale Markdown handoff from being paired with a newer JSON
decision, or a current instruction sheet from being paired with stale evidence.
The validation report records the packet file digest, packet ID, current
evidence identity, action identity, and host-local status digest.

## Scientific boundary

The packet is operational and provenance infrastructure. It neither authorizes
collection by itself nor changes the registered estimator, protocol, execution
order, statistical unit, target split, calibration policy, or evidence count.
Physical acquisition remains forbidden until the separately hash-verified
readiness decision permits the corresponding registered action.
