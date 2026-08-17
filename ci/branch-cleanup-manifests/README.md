# Reviewed agent-branch cleanup manifests

This directory stores the immutable inputs for a provenance-safe cleanup tranche
tracked in [issue #336](https://github.com/IPS-Stuttgart/Causal4D/issues/336).
The scheduled stale-branch workflow remains read-only. Branch deletion is a
separate, manual operation and is impossible without a reviewed manifest merged
to `main`, its exact SHA-256 at dispatch time, and the explicit approval phrase.

## Review sequence

1. Run **Stale agent branch report** from reviewed `main` with the frozen
   30-day minimum age and download its JSON artifact.
2. Copy the exact report JSON into this directory without reformatting it.
3. Independently inspect every proposed entry. Verify the exact branch name and
   tip SHA, absence of an open pull request, age, reachability or exact merged-PR
   lineage, and that deletion cannot remove the only named reference to a
   scientific artifact.
4. Add a manifest beside the copied report and submit both in a normal pull
   request. A manifest may contain at most 20 branches and expires no more than
   seven days after review.
5. After the manifest PR is merged, dispatch **Execute reviewed agent branch
   cleanup** from `main` with the manifest path, the SHA-256 of its exact bytes,
   and the approval phrase `delete-reviewed-agent-branches`.
6. Archive the validation and execution receipts and rerun the read-only report
   to record the before/after branch counts.

The validation job is read-only. Repository write permission exists only in the
separate execution job after validation succeeds. The executor rechecks the
complete tranche before deleting any ref. Immediately before each deletion it
rechecks the repository default branch, allowlist, branch protection, exact tip
SHA, and open-PR state. It then deletes only the exact `agent/*` ref and confirms
that the branch no longer resolves. A changed, protected, allowlisted, expired,
sensitive, unmerged, or active branch fails closed.

Git ref deletion is not transactional. Every execution therefore writes a receipt,
including the already-confirmed deletions and the exact failing branch and phase
when a later deletion or confirmation fails. Archive that partial-failure receipt
before deciding whether a new reviewed tranche is appropriate.

## Manifest schema

```json
{
  "schema_version": 1,
  "artifact_kind": "Causal4DReviewedAgentBranchCleanupManifest",
  "repository": "IPS-Stuttgart/Causal4D",
  "default_branch": "main",
  "source_report_path": "ci/branch-cleanup-manifests/REPORT.json",
  "source_report_sha256": "<64 lowercase hex characters>",
  "source_report_generated_at_utc": "2026-08-17T00:00:00Z",
  "minimum_age_days": 30,
  "reviewed_at_utc": "2026-08-17T01:00:00Z",
  "expires_at_utc": "2026-08-20T01:00:00Z",
  "reviewed_by": "<GitHub login or named reviewer>",
  "issue_url": "https://github.com/IPS-Stuttgart/Causal4D/issues/336",
  "entries": [
    {
      "name": "agent/example-merged-change",
      "expected_sha": "<40 lowercase hex characters>",
      "eligibility_reason": "tip_reachable_from_default",
      "merged_pull_requests": [],
      "artifact_reference_reviewed": true,
      "review_note": "Tip is reachable from main and is not a unique artifact ref."
    }
  ]
}
```

For `eligibility_reason: "exact_tip_merged_in_pull_request"`,
`merged_pull_requests` must contain the exact sorted PR numbers recorded by the
source report. For a tip reachable from `main`, that list must be empty.

Branches whose names contain acquisition, evidence, freeze, frozen, provider,
registered, release, or replication tokens are deliberately outside this
executor even when an old report happened to list them. Tags and non-`agent/*`
refs are always outside scope.
