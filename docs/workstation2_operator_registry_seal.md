# Workstation2 operator-registry seal

The registered Causal4D acquisition campaign requires a person-level operator
roster before any method freeze, independent attestation, or operational gate
approval. The empty protocol-bound template already exists on the persistent
workstation2 dataset, so the next registered action is
`seal_operator_registry`.

## Locked role assignment

The one-shot seal uses the previously selected acquisition roles:

| Project-local operator ID | Person | Registered roles |
| --- | --- | --- |
| `freezer.primary` | Florian Pfaff | `freezer` |
| `verifier.independent` | Anna Seel | `independent_verifier` |
| `gate.operational` | Markus Rummel | `gate_approver` |
| `environment.approver` | Michael Feurer | `gate_approver`, `software_environment_approver` |

Registering a role does not perform the later approval. Anna Seel must still
perform the independent freeze attestation, and each gate must still be approved
through its registered command by an eligible operator.

## Private identity material

The issue-triggered workstation2 workflow creates or reuses a 32-byte HMAC key
and a fixed principal roster below:

```text
/mnt/lexar4tb/causal4d-physical/private/operator-registry-v1
```

The directory is owner-only. The key and private principal roster remain outside
the repository and acquisition dataset and are never uploaded or printed. The
dataset receives only the project-local IDs, roles, active flags, and
`hmac-sha256-domain-separated-v1` person digests required by the existing
operator-registry contract.

A rerun is idempotent only when the workstation-private key, private roster, and
sealed dataset registry agree exactly. Missing private lineage, a changed roster,
a changed digest, an unknown file, or a symlink fails closed.

## Authorized execution

Open an issue with the exact title:

```text
[self-hosted] seal Causal4D operator registry
```

Only the registered maintainer account can allocate the self-hosted job. The
workflow checks out exact reviewed `main`, builds and installs that wheel, derives
the hash-verified current action, and requires all of the following before any
dataset write:

- current action `seal_operator_registry`;
- operator role `principal_investigator`;
- `automatable=false`;
- `physical_acquisition_required=false`;
- `target_outcomes_permitted=false`; and
- `changes_registered_method=false`.

The permitted dataset delta is exactly:

```text
modified preacquisition/operator_registry.template.json
added    preacquisition/operator_registry.json
```

The populated template changes only its `operators` array. The canonical registry
is then validated and published exactly once through the existing
`seal_operator_registry` implementation.

## Stop boundary

After sealing, the workflow recomputes and reports the next registered action,
then stops. It does not:

- approve an operational gate;
- seal or attest the method freeze;
- collect a source-panel or confirmatory execution;
- open a device node or command a robot or sensor;
- read target outcomes; or
- increment physical evidence.

The uploaded artifact contains only a sanitized receipt, registry identities,
dataset delta, exact reviewed revision and wheel digest, and the derived next
action. It excludes the HMAC key, private principals, person digests, populated
template, and sealed registry bytes.
