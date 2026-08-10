# Self-hosted acquisition readiness inspection

The physical Causal4D campaign may use a self-hosted GitHub Actions runner as
the orchestration host, but the runner must expose the registered frozen checkout,
dataset mount, and local robot/sensor interfaces. GPU qualification alone does
not establish those capabilities.

The `Self-hosted acquisition readiness inspection` workflow performs a read-only
qualification on the exact reviewed `main` revision. It can be dispatched
manually from `main`, or directly by the registered maintainer opening an issue
with the exact title:

```text
[self-hosted] inspect Causal4D acquisition readiness
```

For the issue route, the self-hosted job is allocated only after the event is
verified as an issue-open event from the exact registered maintainer login and
numeric account ID, on reviewed `main`, with the exact trigger title. Neither the
issue body nor its labels reach the runner. The self-hosted job receives a
read-only repository token and no GitHub secrets.

The inspection records only sanitized capability evidence:

- presence, readability, writability, and free space for
  `/opt/causal4d-frozen` and
  `/data/causal4d-sloth-multi-action-v1`;
- counts and SHA-256 digests of candidate serial, video, HID, and input-device
  identities without publishing raw device names;
- availability plus hashed output summaries for `ros2 topic list` and `lsusb`;
- the hash-verified registered next-action category, operator role, physical
  requirement, and automation flag when the registered roots are available; and
- exact runner, wheel, revision, and GPU identities.

The inspection never opens a device node, sends a robot or sensor command,
modifies the registered dataset, reads target outcomes, publishes a source-panel
manifest, or increments physical evidence. A physical action remains blocked
unless the registered next action permits it and a separately reviewed local
acquisition driver plus operator safety interlock exists.
