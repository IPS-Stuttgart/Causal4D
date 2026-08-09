# Independent prospective V2 confirmation

A positive prospective V2 promotion result is a candidate-selection result. The
selection panel has already been inspected to choose the candidate and therefore
cannot also provide an unbiased estimate of that candidate's predictive
performance. `causal4d.prospective_v2_confirmation` implements the required
independent confirmation boundary.

This path is separate from the frozen 18-session/36-execution physical protocol.
It does not alter that estimator, its six-frame information boundary, or its
evidence count.

## Confirmation freeze

`build_prospective_v2_confirmation_freeze_v1` first revalidates the complete
selection result against its original freeze, target opening, and unit
evaluations. A confirmation freeze can be created only when the selection panel
chose a non-baseline candidate.

The confirmation freeze binds:

- the exact selection result, freeze, opening, unit evaluations, decision traces,
  raw metric values, and scoring runs;
- the exact registered baseline and the one selected candidate configuration;
- the original stack lock, metric contract, and promotion policy;
- a new target-access seal and the complete confirmation-unit inventory; and
- every target-free factual context, counterfactual query, and implementation
  artifact required for the confirmation panel.

`validate_prospective_v2_confirmation_freeze_v1` reconstructs this freeze from
the original selection evidence and requires exact equality, so an archived
confirmation registration remains independently auditable.

The candidate ladder is not present as an active decision surface. The panel
records:

```text
panel_role = independent_confirmation
candidate_selection_performed = false
selection_panel_outcomes_used = true
confirmation_target_outcomes_used = false
```

## Independence checks

The confirmation freeze rejects overlap with the selection panel in any of the
following identities:

- unit ID or unit binding;
- target artifact;
- endpoint-specific independent group;
- protocol/session pair; or
- target-access seal.

Confirmation unit IDs, bindings, target artifacts, and endpoint independence
groups must also be unique within the confirmation inventory. Each endpoint must
contain at least the number of independent units required by the unchanged
promotion policy.

These checks prevent a relabelled selection row, target artifact, or physical
session from being presented as independent evidence.

## Opening and unit evaluation

`build_prospective_v2_confirmation_opening_v1` opens exactly the ordered target
inventory registered by the confirmation freeze.

Each confirmation trace must retain the normal prospective V2 trace bindings and
add:

```text
confirmation_freeze_id
confirmation_selection_result_id
confirmation_panel_role = independent_confirmation
```

`build_prospective_v2_confirmation_unit_evaluation_v1` evaluates only the fixed
selected candidate against the exact registered baseline. It rejects reuse of a
selection-panel decision trace, raw metric-value artifact, or scoring run. The
ordinary V2 trace still derives candidate admission, exact per-unit fallback,
and harmful accepted-update status; callers cannot supply those labels.

## Confirmation result

`evaluate_prospective_v2_confirmation_v1` requires exactly one evaluation for
every frozen confirmation unit. It aggregates factual continuation, same-grasp
transfer, and new-contact transfer separately using the same metric contract and
thresholds used for selection.

There is no second candidate search. The result has only two possible deployment
outcomes:

1. the previously selected configuration when every endpoint passes; or
2. the exact registered baseline configuration when any endpoint fails.

The result records that candidate selection was not performed on the confirmation
panel and that selection-panel performance was not reused. A passing result means
the fixed-candidate confirmation gate passed on the registered independent panel;
it is not evidence that the candidate was optimal among untested alternatives.

`validate_prospective_v2_confirmation_result_v1` recomputes the complete result
from its frozen evidence and requires exact equality.

## Scientific boundary

The confirmation contracts create no physical observation or result by
themselves. A confirmation freeze must be produced before its confirmation target
outcomes are accessed, and a result may be reported only from the complete bound
panel. The mechanism cannot admit Prob4D into the frozen acquisition, retune the
selected candidate, replace the primary 36-execution analysis, or convert a
selection-panel estimate into independent evidence.
