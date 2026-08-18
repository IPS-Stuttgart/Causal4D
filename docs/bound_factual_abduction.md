# Action-bound factual abduction

## Motivation

A factual rollout bank is meaningful only when every finite-support hypothesis
represents the exact observed action `u_obs` recorded by the source
`TwinBelief`. The historical factual-abduction implementation checks that each
hypothesis is marked as factual, but its frozen compatibility surface does not
add a new action-identity field to existing artifacts.

`causal4d.bound_factual_abduction` provides an additive path for future
claim-bearing studies. It validates the bank before scoring and records the
binding in the resulting `FactualIntervention` metadata.

## Validation

`validate_factual_rollout_action_binding` requires every hypothesis to contain:

- an action metadata mapping;
- a nonempty `proposal_id` equal to `belief.context.u_obs.action_id`; and
- the boolean `future_action_observed=true`.

The returned binding records:

- the complete rollout-bank content identity;
- the source TwinBelief identity;
- protocol and case identities;
- the observed action ID, interval, and trajectory digest; and
- the ordered action identity for every hypothesis.

A relabelled bank, a counterfactual bank, malformed action metadata, or mixed
action support fails before any likelihood is evaluated.

## Usage

```python
from causal4d.bound_factual_abduction import (
    abduct_factual_intervention_bound,
)

factual = abduct_factual_intervention_bound(
    factual_bank,
    twin_belief,
    observations,
    prefix_frame_count=6,
)
```

For valid input, the wrapper calls the existing estimator with the supplied
settings. Posterior weights and intervention support therefore match ordinary
factual abduction; the artifact identity changes because the exact action and
rollout-bank binding is now part of the metadata.

## Compatibility and scientific boundary

The historical path remains unchanged for frozen evidence and compatibility.
This module does not modify the registered 36-execution estimator, open target
outcomes, establish action correctness, or create physical evidence. New studies
that use this path must still bind the actual action trajectory and rollout
producer through their protocol and environment artifacts.
