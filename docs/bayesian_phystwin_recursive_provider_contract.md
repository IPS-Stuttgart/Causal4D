# Complete BayesianPhysTwin provider-v2 recursive contract

## Purpose

`causal4d.belief_provider_v2_contract` intentionally validates the historical
horizon-discrepancy subset of BayesianPhysTwin's provider-v2 surface. That
contract remains unchanged because existing consumers require only endpoint
model averaging and source-calibrated horizon moments.

BayesianPhysTwin's provider-v2 module now also exposes an append-only recursive
Prob4D stream surface. Causal4D must not infer support for that larger interface
merely because the horizon subset is compatible.

`causal4d.belief_provider_v2_recursive_contract` adds a separate, fail-closed
validator for the complete recursive surface.

## Required recursive capabilities

In addition to every historical horizon capability, the complete validator
requires:

- claim-bearing recursive Prob4D stream consumption;
- append-only complete-belief routing;
- exact complete-belief fallback after rejection;
- explicit posterior-covariance semantics;
- provider, calibration, and runtime policy locking;
- an explicit recursive nuisance policy; and
- stream-member and row-identity revalidation.

It also requires the version-1 artifacts:

```text
Prob4DObservationFactorStream
Prob4DStreamObservationBinding
ClaimBearingProb4DStreamStep
ClaimBearingProb4DStreamRun
PosteriorCovarianceSemantics
RecursiveNuisancePolicy
```

The exact recursive-stream claim boundary is validated separately from ordinary
capability and schema matching. A provider may therefore remain compatible with
the horizon-only consumer while being rejected by the recursive consumer.

## Usage

```python
from causal4d.belief_provider_v2_recursive_contract import (
    require_bayesian_phystwin_belief_provider_v2_recursive,
)

manifest = require_bayesian_phystwin_belief_provider_v2_recursive(
    provider_revision=expected_bayesian_phystwin_revision,
)
```

Call the complete validator before opening a recursive stream, observation
binding, posterior-covariance-semantics artifact, or nuisance-policy artifact.
The existing horizon-only validator remains appropriate for consumers that open
none of those values.

## Validation boundary

Tests establish that:

- the complete current manifest is accepted;
- removing one recursive capability leaves the historical horizon subset
  compatible but makes the recursive contract incompatible;
- recursive artifact-schema drift fails closed; and
- a missing, non-string, or promoted recursive claim is rejected.

## Scientific boundary

Provider compatibility establishes interface and semantic agreement only. It
does not establish real Prob4D competence, calibrated covariance, physical-state
identifiability, fresh-object BayesianPhysTwin benefit, Causal4D intervention
benefit, deployment safety, or state of the art.

The recursive contract does not permit Causal4D to consume raw Prob4D factors
after BayesianPhysTwin has already used them. A future recursive belief handoff
must still bind the selected complete belief, exact fallback route, nuisance
policy, covariance semantics, causal intervals, and evidence-ownership ledger.
