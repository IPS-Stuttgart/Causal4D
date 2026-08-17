# End-to-end AIP demonstration

The deterministic CPU-only demonstration exercises the supported Causal4D
information flow without BayesianPhysTwin, Warp, a GPU, or external data:

```text
TwinBelief
    -> factual prefix likelihood
    -> FactualIntervention
    -> do(counterfactual action)
    -> PhysicalPosterior
    -> registered horizon/node projection
```

Run it from an installed package or checkout:

```bash
python -m causal4d.demo.aip --output-dir build/aip-demo
```

The output directory contains:

- `twin_belief.npz`;
- `factual_intervention.npz`;
- `counterfactual_query.npz`;
- `physical_posterior.npz`;
- `projected_posterior.npz`; and
- `summary.json` with the verified artifact identities and compact diagnostics.

Every NPZ is written by the non-pickled contract codec and reloaded before the
summary is emitted. The demonstration also reruns factual abduction after
replacing every held-out suffix coordinate by a large offset. The factual
artifact identity must remain unchanged, making the causal-cutoff invariant
visible in an executable workflow.

## Versioned integration namespaces

New artifact-only consumers should import from:

```python
from causal4d.artifacts.v1 import TwinBelief, load_contract, save_contract
```

Estimator consumers should import the AIP operations separately:

```python
from causal4d.inference.v1 import (
    abduct_factual_intervention,
    apply_counterfactual_operator,
    project_physical_posterior,
)
```

`causal4d.api.v1` and the historical lazy package-root exports remain unchanged.
The split namespaces are additive and do not alter frozen artifacts, registered
methods, provider compatibility, or scientific claims.

## Evidence boundary

This is a controlled software and contract demonstration only. Its synthetic
arrays, posterior concentration, and output trajectories must not be counted as
physical evidence or used to promote a Causal4D claim.
