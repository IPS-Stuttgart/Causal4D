# Stale `agent/*` branch hygiene

Causal4D uses short-lived `agent/*` branches for reviewed pull requests. Merged and
superseded branches can otherwise accumulate indefinitely, making repository
navigation and provenance audits harder.

The `Stale agent branch report` workflow is deliberately **read-only**. It creates
an auditable JSON and Markdown inventory but never updates or deletes a branch.

## Candidate policy

A branch is reported as a manual cleanup candidate only when all of the following
are true:

- its tip is at least the configured number of days old;
- it is not protected or allowlisted;
- it has no open pull request; and
- either its exact tip is reachable from the default branch or that exact tip was
  merged through a pull request.

Closed-but-unmerged pull requests, branches with unmerged follow-up commits,
future-dated commits, malformed API responses, and ambiguous histories are
excluded. The report fails closed instead of guessing.

## Manual cleanup

Run the workflow from reviewed `main`, download the report, inspect every candidate,
and remove approved branches manually through GitHub. The workflow has only
`contents: read` and `pull-requests: read`; it has no write-capable job or deletion
mode.

## Allowlist

The allowlist has a deliberately small strict schema:

```json
{
  "schema_version": 1,
  "branches": ["agent/exact-long-lived-branch"],
  "prefixes": ["agent/evidence/"]
}
```

Unknown fields, duplicate keys or entries, empty strings, symlinked files, and
unsupported schema versions are rejected.

## Boundary

This report does not modify the frozen estimator, acquisition protocol,
target-access boundary, retained evidence, scientific result, branch references,
or physical execution count.
