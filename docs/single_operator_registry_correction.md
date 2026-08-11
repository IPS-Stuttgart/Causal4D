# Single-operator registry correction

The workstation2 operator registry created in August 2026 contained unsupported
person assignments. Those assignments were not provided or authorized by the
project owner and must not be used as evidence of participation, consent,
institutional authority, or independent review.

The corrected project roster contains one real participant only:

| Operator ID | Person | Registered non-independent roles |
| --- | --- | --- |
| `florianpfaff` | Florian Pfaff | `freezer`, `gate_approver`, `software_environment_approver` |

This is the complete participant list for the current project. Historical hashes,
operator IDs, workflow receipts, or Git metadata must not be interpreted as
identifying any additional participant.

No `independent_verifier` is registered. Causal4D must therefore report
`independent_verifier_available=false` and stop before object registration,
source-panel acquisition, gate approval, method-freeze attestation, or any
confirmatory physical execution under the independently attested protocol.

## Why the registry remains blocked

The registered study requires the method freezer and independent verifier to be
different people. Assigning two aliases or two roles to Florian Pfaff would not
provide independence. The person-level digest check remains strict for every
actual attestation. The correction does not weaken or bypass that check.

A single-person registry is valid as a truthful roster, but it is not sufficient
to authorize the current protocol. The derived next action is:

```text
stop_independent_verifier_unavailable
```

with:

```text
category=governance_blocker
physical_acquisition_required=false
automatable=false
target_outcomes_permitted=false
```

Proceeding would require either a separately reviewed protocol amendment that
removes every independent-review claim or the later registration of a real,
distinct person. Neither is performed by this correction.

## Workstation2 correction

The one-shot correction is restricted to the exact known unsupported registry
hashes and refuses to run after any governed evidence exists. It verifies that
there is no object registration, slip pilot, calibration approval, source-panel
manifest, method freeze, freeze attestation, approved gate, confirmatory session,
or confirmatory execution.

The permitted dataset delta is exactly:

```text
modified preacquisition/operator_registry.template.json
modified preacquisition/operator_registry.json
added    preacquisition/operator_registry_correction_v1.json
```

The private HMAC key and private principal roster are replaced with owner-only
material containing only `github-login-v1:FlorianPfaff`. Neither private file is
uploaded. The public correction receipt records only old and new artifact hashes,
operator counts, the single project-local operator ID, the unavailable
independent-verifier status, and the zero-evidence boundary.

The correction reads no target outcome, opens no device node, sends no robot or
sensor command, changes no estimator or registered analysis, and increments
physical evidence by zero.

## Historical record

The earlier merge and workflow receipts are retracted as authorization evidence.
Their hashes may remain in Git history and issue chronology for auditability, but
they must not be interpreted as proving that any unsupported person participated
in Causal4D. Current `main`, the correction receipt, and the post-correction
hash-verified readiness decision are authoritative.
