# Self-hosted acquisition readiness inspection

The physical Causal4D campaign may use a self-hosted GitHub Actions runner as
the orchestration host, but the runner must expose one complete registered pair
consisting of a frozen checkout and its matching dataset root, together with the
local robot/sensor interfaces. GPU qualification alone does not establish those
capabilities.

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

## Registered root-pair selection

The workflow currently inspects these complete pairs rather than testing or
combining individual paths independently:

| Candidate | Frozen checkout | Dataset root |
| --- | --- | --- |
| `canonical` | `/opt/causal4d-frozen` | `/data/causal4d-sloth-multi-action-v1` |
| `workstation2-persistent` | `/mnt/lexar4tb/causal4d-physical/causal4d-frozen` | `/mnt/lexar4tb/causal4d-physical/causal4d-sloth-multi-action-v1` |

Selection is fail-closed:

- exactly one complete pair of ordinary directories is selected;
- repository and dataset roots from different candidates are never mixed;
- a partial pair is reported but cannot be selected;
- any symlink component or existing non-directory makes that candidate invalid;
- no complete pair yields `runner_not_provisioned_with_registered_roots`; and
- more than one complete pair yields `registered_root_selection_ambiguous` and
  requires an explicit operator decision before readiness is derived.

The probe also retains the explicit `--repository-root` and `--dataset-root`
interface for controlled one-pair inspection. Both must be supplied together and
cannot be combined with repeated `--root-candidate` arguments.

The inspection records only sanitized capability evidence:

- candidate-pair state, selected candidate ID, presence, readability,
  writability, symlink status, and free space for the selected roots;
- counts and SHA-256 digests of candidate serial, video, HID, and input-device
  identities without publishing raw device names;
- availability plus hashed output summaries for `ros2 topic list` and `lsusb`;
- the hash-verified registered next-action category, operator role, physical
  requirement, and automation flag when exactly one registered root pair is
  available;
- the exact hash-verification policy supplied to the next-action builder, rather
  than an inferred field from the returned decision artifact; and
- exact runner, wheel, revision, and GPU identities.

The inspection never opens a device node, sends a robot or sensor command,
modifies the registered dataset, reads target outcomes, publishes a source-panel
manifest, or increments physical evidence. A physical action remains blocked
unless the registered next action permits it and a separately reviewed local
acquisition driver plus operator safety interlock exists.
