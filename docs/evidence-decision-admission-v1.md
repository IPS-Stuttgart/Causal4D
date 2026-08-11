# Evidence-decision admission v1

Causal4D admits claim-bearing BayesianPhysTwin results through an independent
validator for `bayesian_phystwin.evidence_decision` wire version 1. The admission
layer does not import BayesianPhysTwin or Prob4D at runtime. It verifies the
closed decision envelope and then requires the exact repository states that
participated in the result.

```python
from causal4d.evidence_decision_v1 import (
    admit_causal_claim_v1,
    load_evidence_decision_v1,
)

decision = load_evidence_decision_v1("decision.json")
admission = admit_causal_claim_v1(
    decision,
    claim_id="registered.causal.claim",
    protocol_id="registered-protocol-v1",
    expected_bayesian_phystwin_revision=(
        "0123456789abcdef0123456789abcdef01234567"
    ),
    expected_causal4d_revision=(
        "89abcdef0123456789abcdef0123456789abcdef"
    ),
    expected_prob4d_revision=(
        "fedcba9876543210fedcba9876543210fedcba98"
    ),
    minimum_evidence_level=3,
    require_prob4d_binding=True,
)
```

## Admission conditions

Admission fails closed unless all of the following hold:

- the decision has the exact version-1 closed shape and content-derived ID;
- the decision explicitly authorizes a passing confirmatory claim;
- the claim ID, protocol ID, and minimum evidence level match;
- exactly one clean BayesianPhysTwin repository is present as `primary`;
- exactly one clean Causal4D repository is present as `downstream`;
- the BayesianPhysTwin and Causal4D revisions match the expected revisions; and
- when required, exactly one clean Prob4D repository is present as
  `observation`, `upstream`, or `dependency` at the expected revision.

The canonical `IPS-Stuttgart` repositories and historical `FlorianPfaff`
repository aliases are recognized. Multiple aliases for the same project in one
decision are rejected rather than guessed between.

A Prob4D binding is optional for decisions whose evidence path genuinely does
not use Prob4D. When one is present, it is still validated. Set
`require_prob4d_binding=True` to make it mandatory for a registered lane. A mandatory Prob4D lane must also provide
`expected_prob4d_revision`; presence without an exact revision lock is
rejected.

## Source and integration locks

The generated Causal4D validator is locked to the BayesianPhysTwin contract
merged at revision `4ee702f5130cfedbea7bce6be5e72483c92f63da` and JSON
Schema SHA-256
`d5615258c6cf666d0ed9684a87930989adf91817fe99b0387e83a31479dcd465`.

The installed-boundary workflow also pins the independent Prob4D consumer at
merge revision `c9273e8a55f812c532105a86c885a5a7627d3df3`. It checks out the
exact Causal4D head under review, installs all three packages, emits a real
version-1 decision through BayesianPhysTwin, and requires both independent
consumers to agree on the decision ID, authorization, schema digest, and exact
Prob4D repository binding. The admission receipt is retained as a workflow
artifact.

## Command line

```bash
python -m causal4d.evidence_decision_v1 \
  decision.json \
  --claim-id registered.causal.claim \
  --protocol-id registered-protocol-v1 \
  --expected-bayesian-phystwin-revision 0123456789abcdef0123456789abcdef01234567 \
  --expected-causal4d-revision 89abcdef0123456789abcdef0123456789abcdef \
  --expected-prob4d-revision fedcba9876543210fedcba9876543210fedcba98 \
  --minimum-evidence-level 3 \
  --require-prob4d-binding
```

The command prints a compact admission receipt containing the decision ID and
admitted repository bindings.

## Scientific boundary

Admission establishes contract conformance, authorization semantics, and exact
repository provenance. It does not independently establish physical-state
identification, uncertainty calibration, causal sufficiency, intervention
benefit, transfer to a new object or session, deployment safety, or state of the
art. Those conclusions remain governed by their registered physical and
statistical evidence.
