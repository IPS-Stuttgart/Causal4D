# Evidence-bound prospective V2 promotion

The prospective V2 promotion path is a separate candidate-selection experiment.
It does not modify the frozen physical-acquisition estimator or the registered
18-session/36-execution protocol.

The implementation is split into three boundaries:

1. source-only certification and the fixed V2 decision inventory;
2. one sealed opening of a preregistered target inventory; and
3. endpoint-wise candidate selection from artifact-bound evaluations.

## Target-free freeze

`ProspectiveV2PromotionFreezeV1` is created before target access. It binds:

- the exact stack lock;
- the fixed candidate ladder, from the registered baseline through the sparse
  contact-patch candidate;
- every independent evaluation unit and its endpoint, protocol, case, session,
  factual context, counterfactual query, target artifact, and access seal;
- the scoring implementation and exact metric semantics;
- the harmful-regret threshold and all endpoint promotion thresholds; and
- the source artifact inventory.

The freeze is explicitly a `candidate_selection_only` panel. It always records
that an unbiased post-selection performance claim is unsupported and that an
independent confirmation panel is required.

## One target opening

`build_prospective_v2_target_opening_v1` derives the opening inventory from the
freeze. The opening ID binds the complete ordered target-artifact inventory and
the preregistered target-access seal. The promotion evaluator rejects a subset,
extra artifact, reordered inventory, different seal, or different freeze.

No caller-supplied `one_target_opening_verified` Boolean exists. The property is
established by exact identity and inventory validation.

## Bound unit evaluations

Each non-baseline candidate and registered unit produces:

1. a prospective V2 decision trace;
2. baseline and candidate prediction artifacts; and
3. one raw metric-value artifact.

The decision trace must pass the complete V2 profile and content-bind the
candidate ID, candidate configuration, evaluation unit, and target-access seal.
The raw metric artifact must bind:

- the target opening;
- the frozen unit and candidate bindings;
- the exact target artifact;
- the baseline and candidate prediction artifacts; and
- the frozen metric contract.

`build_prospective_v2_unit_evaluation_v1` then derives, rather than accepts from
the caller:

- candidate acceptance from the validated trace;
- exact fallback from the deployed prediction identity;
- baseline-relative log-score, Brier, and trajectory effects;
- coverage error and interval-width ratio; and
- harmful accepted-update status from the frozen regret threshold.

Consequently, metric rows cannot self-declare acceptance, fallback, or
harmfulness.

## Promotion result

`evaluate_prospective_v2_promotion_v1` requires the complete Cartesian product of
registered units and non-baseline candidates. It aggregates at the independent
unit level for factual continuation, same-grasp transfer, and new-contact
transfer separately. A candidate passes only when every endpoint passes every
frozen threshold.

The selected configuration is the highest passing member of the frozen ladder.
When none passes, the selected configuration is the exact registered baseline
configuration. `validate_prospective_v2_promotion_result_v1` recomputes the
complete result and requires exact equality with its bound source evaluations.

Even a positive result remains a selection result. The result artifact always
contains:

```text
selection_panel_role = candidate_selection_only
unbiased_post_selection_performance_claimed = false
independent_confirmation_required = true
```

A fresh independent panel is required before reporting post-selection predictive
performance for the selected candidate.

## Scientific boundary

This infrastructure creates no physical execution, observation, accuracy result,
calibration result, or evidence count by itself. It cannot admit Prob4D into the
frozen acquisition, change the six-frame information boundary, revise the
registered method, or rescue a failed 36-execution primary result.
