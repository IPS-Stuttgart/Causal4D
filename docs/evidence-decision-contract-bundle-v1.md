# Evidence-decision contract bundle v1

Causal4D retains an offline-auditable copy of the exact
`bayesian_phystwin.evidence_decision` version-1 wire contract consumed by
`causal4d.evidence_decision_v1`.

The bundle is installed with the Causal4D wheel under:

```text
causal4d/contract_data/evidence_decision_v1/
├── evidence-decision-v1.schema.json
├── manifest.json
└── vectors/
    └── authorized.json
```

## Source lock

The schema bytes come from `IPS-Stuttgart/BayesianPhysTwin` revision
`4ee702f5130cfedbea7bce6be5e72483c92f63da`. Their SHA-256 digest is:

```text
d5615258c6cf666d0ed9684a87930989adf91817fe99b0387e83a31479dcd465
```

`manifest.json` binds that source revision, digest, wire name, wire version,
consumer module, and every retained vector digest. A change to any retained
byte therefore requires an explicit new source lock or a new contract version.

## Offline verification

The installed package can be checked without a BayesianPhysTwin checkout:

```python
import hashlib
import json
from importlib.resources import files

root = files("causal4d").joinpath(
    "contract_data", "evidence_decision_v1"
)
manifest = json.loads(root.joinpath("manifest.json").read_text())
schema = root.joinpath(manifest["json_schema"]["path"]).read_bytes()
assert hashlib.sha256(schema).hexdigest() == manifest["json_schema"]["sha256"]
```

The retained authorized vector is synthetic. It verifies closed-shape parsing,
content identity, repository-role binding, exact revisions, and downstream
admission; it is not scientific evidence.

## Scientific boundary

This bundle establishes contract identity and independent consumer conformance.
It does not establish physical-state identification, calibrated uncertainty,
causal sufficiency, intervention benefit, transfer, deployment safety, or state
of the art. Those claims require separately registered evidence decisions and
exact repository bindings.
