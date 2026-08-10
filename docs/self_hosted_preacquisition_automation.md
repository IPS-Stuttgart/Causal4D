# Self-hosted automatable pre-acquisition actions

Causal4D may execute a registered pre-acquisition action on `workstation2` only
when the current hash-verified next-action decision classifies it as both
nonphysical and mechanically automatable. This surface is deliberately narrower
than the general Causal4D command line.

The workflow can be dispatched manually from reviewed `main`, or triggered by the
registered maintainer opening an issue with the exact title:

```text
[self-hosted] execute Causal4D automatable next action
```

The issue body, labels, and comments are not used as executable input. The
self-hosted job is allocated only on reviewed `main`, checks out the exact issue
event SHA, verifies a clean tree, builds and installs that exact Causal4D wheel,
receives a read-only repository token, and receives no GitHub secrets.

## Current allowlist

The only executable action is:

```text
scaffold_operator_registry
```

It corresponds exactly to:

```bash
causal4d protocol readiness scaffold-operator-registry \
  /mnt/lexar4tb/causal4d-physical/causal4d-frozen \
  /mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1
```

Before execution, the workflow requires all of the following:

- exactly one registered repository/dataset root pair is complete;
- that pair is `workstation2-persistent`;
- the current registered action is exactly `scaffold_operator_registry`;
- the action is marked `automatable=true`;
- `physical_acquisition_required=false`;
- `target_outcomes_permitted=false`;
- `changes_registered_method=false`; and
- the registered `command_argv` equals the allowlisted command byte-for-byte.

The executor snapshots every ordinary file in the registered dataset before and
after the action. Success requires exactly one added file and no modified or
removed files:

```text
preacquisition/operator_registry.template.json
```

The created artifact must remain an empty, unsealed template with
`target_outcomes_used=false`. It contains no operator identities, person
digests, approval, seal, or scientific evidence.

After execution, the workflow recomputes the hash-verified next action. Success
requires the next boundary to be:

```text
seal_operator_registry
```

with `automatable=false` and `physical_acquisition_required=false`. The workflow
then stops. It cannot populate identities, derive institutional HMACs, seal the
registry, approve a gate, attest a freeze, publish source evidence, open a device
node, or dispatch a robot or sensor command.

## Evidence boundary

The retained artifact includes the exact reviewed revision and wheel digest, the
pre- and post-action readiness reports, the one-file dataset delta, the created
template digest and byte count, runner identity, and GPU inventory.

Creating the empty template is an operational dataset modification, but it does
not count as a physical execution:

```text
target_outcomes_used=false
physical_command_sent=false
physical_evidence_increment=0
```

Any future automatable action requires a separate code change, tests, hosted CI,
review, and explicit addition to the executor allowlist.
