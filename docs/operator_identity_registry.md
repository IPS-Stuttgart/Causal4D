# Operator identity registry

The acquisition workflow uses a sealed operator roster so approvals are bound to
a real person rather than free-form display strings. Under the active
pre-acquisition v5 policy, one registered person may perform every
pre-acquisition role. The resulting evidence is self-attested; it is not
independently attested.

This governance change does not alter the estimator, 18-session/36-execution
schedule, source/target split, registered analysis, exclusion policy, or any
scientific threshold.

## Active v5 policy

The active policy is
`causal4d-sloth-preacquisition-v5-single-operator`. It permits the same
registered person to:

- seal and self-attest the method freeze;
- approve operational and software-environment gates;
- review and publish source-panel evidence; and
- perform the two chronological contact-registration review passes.

Every report using this study must disclose:

> One registered operator performed the pre-acquisition checks and
> self-attested the freeze; no independent pre-acquisition attestation is
> claimed.

A genuinely distinct verifier can still be added later, but is not required for
v5 readiness. Do not create aliases or duplicate identities to simulate
independence.

## Identity model

The canonical artifact is:

```text
preacquisition/operator_registry.json
```

Each entry contains a stable project-local `operator_id`, an `active` flag,
registered roles, and `person_identity_sha256`. Raw email addresses, account
names, personnel numbers, and the HMAC key must stay outside the repository and
dataset.

The registered digest method is:

```text
HMAC-SHA256(
  institution-held secret,
  b"causal4d-operator-v1\0" + stable_institutional_principal
)
```

The same person must always produce the same digest. Two registry entries may
not share a person digest.

## Roles

Supported roles are:

- `freezer`;
- `gate_approver`;
- `software_environment_approver`; and
- `independent_verifier`, retained for genuinely independent future review.

For a one-person v5 registry, one active operator needs at least
`freezer`, `gate_approver`, and `software_environment_approver`.
The `independent_verifier` role must not be assigned unless a distinct person
actually performs that role.

## Lifecycle

Create the v5-bound template in a fresh, non-overwriting dataset scaffold:

```bash
causal4d protocol readiness scaffold-operator-registry \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1
```

Edit only the `operators` array in
`preacquisition/operator_registry.template.json`. A valid one-person entry has
this shape, with roles sorted canonically:

```json
{
  "operator_id": "florianpfaff",
  "person_identity_sha256": "<64 lowercase hex>",
  "active": true,
  "roles": [
    "freezer",
    "gate_approver",
    "software_environment_approver"
  ]
}
```

Seal the roster exactly once:

```bash
causal4d protocol readiness seal-operator-registry \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  /data/causal4d-sloth-multi-action-v1/preacquisition/operator_registry.template.json \
  --sealed-by florianpfaff
```

The seal fails closed if evidence already exists, identities or roles are
invalid, the template is not bound to v5, or target outcomes entered the
artifact. An old v4-bound registry must not be edited in place; start a fresh
v5-bound scaffold and retain the old tree as historical provenance.

## Governed approvals

Use the same registered operator ID honestly:

```bash
causal4d protocol freeze seal \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  --frozen-by florianpfaff

causal4d protocol freeze attest \
  /data/causal4d-sloth-multi-action-v1/method_freeze.json \
  configs/causal4d/sloth_multi_action_v1.json \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1/method_freeze_validation.json \
  --verified-by florianpfaff

causal4d protocol readiness seal-gate \
  /opt/causal4d-frozen \
  /data/causal4d-sloth-multi-action-v1 \
  support_registration_passed \
  --approved-by florianpfaff
```

The registry must predate every governed approval. Readiness revalidates the
complete identity chain, all roles, chronology, source hashes, and the explicit
self-attestation policy at status time.

## Legacy v4 boundary

Artifacts created under v4 remain immutable and retain their original
two-person requirement. V5 does not reinterpret them. Only evidence bound to the
v5 plan and amendment digest may use the single-operator policy.
