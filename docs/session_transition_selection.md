# Source-only session-transition selection

## Purpose

Causal4D's session hierarchy can represent

```text
p(phi_session | phi_bar)
```

on the registered finite persistent-intervention support. Supplying that
transition matrix manually is useful for controlled studies, but a future
multi-session protocol needs an auditable source-only selection procedure.

`causal4d.session_transition_selection` selects one matrix from a frozen finite
candidate set by leave-one-independent-session-out log predictive density. It
never estimates or selects the transition from confirmatory target outcomes.

## Model and score

For candidate transition `T[g, f]`, training source sessions update the global
posterior over `(phi_bar=g, theta=p)`. The held-out source session is scored by

```text
sum_g,p,f
    p(phi_bar=g, theta=p | training sessions)
    T[g, f]
    p(held-out session evidence | phi_session=f, theta=p).
```

Every source session contributes one predictive-score unit. Frames, executions,
nodes, views, coordinates, and posterior components are not treated as
independent selection units.

The exact identity transition must be included and named explicitly. When its
mean source score lies within the preregistered tolerance of the best candidate,
the identity transition is selected. Otherwise the first highest-scoring
candidate in the frozen candidate order is selected.

## Usage

```python
from causal4d.session_transition_selection import (
    select_session_phi_transition_source_only,
)

selection = select_session_phi_transition_source_only(
    source_session_log_evidence,
    source_session_ids=source_session_ids,
    phi_prior=phi_prior,
    parameter_prior=parameter_prior,
    candidate_ids=("identity", "weak-variation", "moderate-variation"),
    candidate_transitions=candidate_transition_matrices,
    identity_candidate_id="identity",
    selection_tolerance=1.0e-3,
    metadata={"registered_before_target_access": True},
)

hierarchy = infer_session_phi_hierarchy(
    target_closed_source_execution_log_evidence,
    phi_prior=selection.phi_prior,
    parameter_prior=selection.parameter_prior,
    session_ids=source_execution_session_ids,
    execution_evidence_powers=source_execution_powers,
    session_phi_transition=selection.selected_transition,
)
```

Candidate matrices must be finite, nonnegative, row-stochastic, and defined on
the same finite `phi` support. The identity candidate must be the exact identity
matrix rather than a numerically close approximation.

## Artifact and validation

`SessionTransitionSelectionV1` binds:

- unique independent source-session IDs;
- normalized `phi` and physical-parameter priors;
- the complete session-level log-evidence tensor;
- ordered candidate IDs and transition matrices;
- the exact identity candidate;
- every leave-one-session-out log score;
- equal-session mean scores;
- the identity-favoring selection tolerance and selected candidate; and
- finite diagnostic metadata and the scientific claim boundary.

Construction recomputes the complete source-fold score table and selected
candidate. Supplied score or selection drift is rejected.

Focused tests cover stable sessions selecting identity, variable sessions
selecting a diffuse candidate, exact identity preference under a score tie,
invalid source/candidate designs, and evidence-sensitive content identity.

## Scientific boundary

This selector is intended for a separately versioned future protocol. It does
not enter or modify the frozen 36-execution estimator. A source-selected
transition does not establish new-session calibration, correct hierarchy
support, physical parameter identifiability, counterfactual accuracy, or object
class generalization.

Candidate matrices, candidate order, priors, score, tolerance, source-session
roster, and all exclusions must be frozen before target access. A failed or
identity-selected result is complete evidence and must not be retuned on the
same target sessions.
