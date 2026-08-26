# Source-independent v5 owner-identity bootstrap

Status: **locked before the replacement bootstrap execution**

This is an operational provenance amendment to the v5 single-operator policy. It
does not change the registered scientific design, source/target split, mechanism
gates, acquisition roster, or analysis.

## Why a replacement is required

The first v5 bootstrap workflow copied the public operator record from an
ephemeral v4 dataset tree. That tree and its private HMAC material are no longer
available on `workstation2`, so the workflow cannot start. The canceled legacy
run reached no job steps and created no scaffold, physical evidence, or target
access.

The surviving public correction receipt proves what the old registry claimed,
but it does not contain the private key needed to reproduce its person digest.
The old digest must therefore not be guessed, reconstructed from public
metadata, or claimed as continuous.

## Replacement identity semantics

The v2 bootstrap initializes a new owner-only identity exactly once:

- canonical principal: `github-login-v1:FlorianPfaff`;
- digest method: `hmac-sha256-domain-separated-v1`;
- HMAC domain: `causal4d-operator-v1` followed by a NUL byte;
- one 32-byte random private key;
- active roles: `freezer`, `gate_approver`, and
  `software_environment_approver`;
- no `independent_verifier` role.

The private key and principal roster live only below the fixed owner-only root:

```text
/mnt/lexar4tb/causal4d-physical/private/operator-registry-v5
```

The directory must have mode `0700`; its two files must have mode `0600`.
Neither file, the canonical principal, nor the person digest is uploaded as a
workflow artifact. The public registry necessarily contains the HMAC-derived
person digest required by the registered Causal4D identity contract.

Every v2 receipt and sanitized report states:

```text
identity_initialization_mode=fresh_owner_hmac_v1
historical_registry_available=false
historical_registry_reused=false
identity_digest_continuity_claimed=false
independent_preacquisition_attestation_claimed=false
```

## Transaction and rerun behavior

Private material is published from an owner-only staging directory by atomic
rename. The acquisition scaffold is separately built in a sibling staging tree
and atomically renamed to the fixed v5 dataset root. If scaffold creation fails
after the private identity is initialized, a later run must reuse and revalidate
that same private identity; it may not generate another one.

An idempotent rerun verifies the private material, operator registry, bootstrap
receipt, protocol, v5 amendment, and registered next action byte-for-value. It
also binds the receipt to the exact bootstrap implementation SHA-256 and byte
count. It does not reseal or modify the dataset. Partial private roots,
unexpected files, wrong permissions, symlinks, changed private material,
changed registry bytes, or any already-created governed evidence fail closed.

## Trigger and evidence boundary

Only an issue opened by the registered repository owner on reviewed `main` with
this exact title may allocate the self-hosted job:

```text
[self-hosted] bootstrap Causal4D v5 owner identity scaffold v2
```

The older title is revoked on current `main`. The replacement job receives no
GitHub secrets, builds and imports the exact reviewed wheel, and uploads only a
sanitized report, runner identity, GPU inventory, and wheel digest.

Successful bootstrap advances only to:

```text
complete_object_registration
```

It sends no physical command, opens no device node, uses no target outcome,
changes no registered method, and adds zero physical executions. The standing
paper disclosure remains: one registered operator self-attests the checks and
no independent pre-acquisition attestation is claimed.
