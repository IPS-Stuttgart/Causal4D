# Trajectory-calibrated support-robust interventions

`causal4d.support_robust_intervention` is the Causal4D consumer for the
trajectory-level split-conformal regret envelope produced by BayesianPhysTwin.
It is deliberately independent of BayesianPhysTwin at runtime: live records and
serialized mappings are both accepted, but every numerical field that controls
the action is reconstructed before use.

## Operational rule

For registered finite-support regret bounds `B[a]`, conformal radius `rho`,
regret budget `epsilon`, and a registered nonfallback action roster, Causal4D
reconstructs

```text
inflated[a] = B[a] + rho
admissible[a] = candidate[a] and inflated[a] <= epsilon
```

It executes a nonfallback action only when exactly one admissible action has the
smallest inflated regret. Otherwise it returns the caller-owned fallback. In
particular:

- an infinite conformal radius returns fallback;
- two equally good admissible actions return fallback;
- the fallback cannot appear in the candidate mask;
- supplied selected indices, masks, and inflated bounds are never trusted;
- malformed or tampered records are rejected before an action name is exposed.

## Example

```python
from causal4d.support_robust_intervention import (
    consume_support_robust_decision,
)

result = consume_support_robust_decision(
    bayesian_phystwin_record,
    ("physical_fallback", "half_update", "full_update"),
    expected_fallback_action_index=0,
)

if result.used_fallback:
    execute_registered_physical_fallback()
else:
    execute_registered_action(result.selected_action_name)
```

The action roster is caller-owned and must be registered independently of the
record. Optional BayesianPhysTwin `version` and `semantics` metadata are checked
when available.

## Statistical boundary

The underlying split-conformal statement uses one score per complete calibration
trajectory. Under exchangeability, the inflated bounds cover every registered
decision and candidate action within one future complete trajectory with the
declared trajectory-marginal probability. The guarantee is not pointwise
conditional, does not validate exchangeability, and does not establish
unseen-object transport.

With too few calibration trajectories for a requested finite-sample rank, the
BayesianPhysTwin radius is positive infinity. Causal4D preserves that result as
an exact fallback rather than treating correlated windows as extra samples.

This consumer verifies arithmetic and action-selection semantics only. It does
not validate the physical support, provider, loss, regret budget, intervention
cost, deployment context, or safety.
