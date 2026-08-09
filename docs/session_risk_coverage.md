# Session-clustered risk--coverage diagnostics

`causal4d.selective_prediction` provides a prospective, diagnostic-only
risk--coverage contract for Causal4D predictions. It answers a narrow question:

> When a source-frozen uncertainty or reliability score is used to abstain, how
> does held-out risk change as complete independent sessions are retained?

The diagnostic does **not** change the registered 36-execution analysis, select a
new threshold from target outcomes, rescue a failed primary endpoint, or promote a
candidate.

## Why the independent unit is the session

The physical protocol contains multiple executions within a grasp session. Those
executions share initialization and nuisance conditions, so treating them as
independent would overstate the effective sample size. The diagnostic therefore:

- aggregates unit risk by an equal-weight mean within each complete session;
- aggregates abstention score by the maximum within each session;
- removes an entire session when any registered unit in that session is excluded;
- admits all sessions with an equal score together; and
- reports coverage both relative to eligible sessions and relative to all
  registered sessions.

Using the maximum score means a session is retained only when every registered unit
meets the same threshold.

## Target-access boundary

The ranking contract must identify a content-addressed score artifact and assert all
of the following:

```text
frozen_before_target_access = true
target_outcomes_used = false
lower_score_more_confident = true
```

The score may be predictive scale, posterior entropy, support distance, or another
source-only reliability quantity. Held-out outcomes may be used only for the risk
values evaluated after this contract is frozen. Every threshold in the curve is an
observed value of the frozen score; target risk cannot choose or move a threshold.

## Python example

```python
from causal4d.selective_prediction import (
    SessionRiskCoverageRankingContract,
    SessionRiskCoverageRecord,
    build_session_risk_coverage_diagnostic,
    write_session_risk_coverage_diagnostic,
)

ranking = SessionRiskCoverageRankingContract(
    ranking_artifact_id="<64-character SHA-256>",
    score_name="source_predictive_scale",
    score_semantics="maximum source-only predictive scale in metres",
    frozen_before_target_access=True,
    target_outcomes_used=False,
)

records = [
    SessionRiskCoverageRecord(
        unit_id="session-01-execution-01",
        session_id="session-01",
        included=True,
        risk=0.021,
        abstention_score=0.014,
    ),
    SessionRiskCoverageRecord(
        unit_id="session-01-execution-02",
        session_id="session-01",
        included=True,
        risk=0.026,
        abstention_score=0.018,
    ),
]

diagnostic = build_session_risk_coverage_diagnostic(
    records,
    ranking,
    risk_name="track_error",
    risk_unit="m",
)
write_session_risk_coverage_diagnostic(
    "session-risk-coverage.json",
    diagnostic,
)
```

Excluded units must omit both target risk and ranking score and retain their
preregistered reason:

```python
SessionRiskCoverageRecord(
    unit_id="session-02-execution-01",
    session_id="session-02",
    included=False,
    risk=None,
    abstention_score=None,
    exclusion_reason="registered timing failure",
)
```

## Output contract

The content-addressed output contains:

- the exact ranking contract and registered records;
- complete unit and session accounting;
- one canonical summary per eligible session;
- tie-preserving risk--coverage points;
- full-eligible-coverage risk; and
- explicit scientific-boundary fields.

`validate_session_risk_coverage_diagnostic` recomputes the complete output from its
source records and ranking contract. `write_session_risk_coverage_diagnostic`
rejects a stale or tampered content identity and publishes through the repository's
atomic JSON writer.

This result is descriptive and diagnostic. Any operating threshold intended for a
future method must be selected on a separate source/calibration panel and then
tested on a fresh independent confirmation panel.
