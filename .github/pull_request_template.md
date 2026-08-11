<!--
Complete every field or write "not applicable". Do not remove the scientific and
information-boundary section. See CONTRIBUTING.md for the repository rules.
-->

## Summary

<!-- Explain what changed, why it changed, and the user or scientific impact. -->

## Change classification

Select one primary classification:

- [ ] Frozen or registered method/analysis change requiring a new version
- [ ] Future or experimental method
- [ ] BayesianPhysTwin, Prob4D, or other provider/artifact contract
- [ ] Acquisition, calibration, readiness, or evidence tooling
- [ ] Diagnostic, source-only, retrospective, or controlled study
- [ ] Maintenance, documentation, packaging, CI, or security

Evidence status, when applicable:

- [ ] Software or contract evidence only
- [ ] Controlled synthetic evidence
- [ ] Source/calibration-only evidence
- [ ] Retrospective or already-open diagnostic evidence
- [ ] Confirmatory physical evidence
- [ ] No scientific evidence produced

## Scientific and information boundary

Replace each placeholder and explain every `yes` answer.

- Frozen estimator or registered analysis changed: `<yes/no>`
- Target or held-out outcomes accessed: `<yes/no>`
- Registered physical-acquisition dataset modified: `<yes/no>`
- Physical evidence increment: `<integer, normally 0>`
- Existing scientific claim changed or promoted: `<yes/no>`
- Independent statistical unit: `<object/session/execution/not applicable>`
- Source, calibration, and target split: `<identities or not applicable>`
- Exact fallback preserved for rejected optional evidence: `<yes/no/not applicable>`

Do not combine a method change with target evaluation or claim promotion. A negative
or bounded result must remain reportable without retuning on the same target cohort.

## Provenance and compatibility

- Base revision or frozen source identity:
- Contract, schema, provider, or protocol versions:
- Frozen artifacts or milestones affected:
- Compatibility, migration, and fallback behavior:

## Validation

- [ ] A regression fails on the unpatched implementation, when applicable
- [ ] Valid inputs and expected behavior are covered
- [ ] Malformed, mutable, or provenance-inconsistent inputs are covered
- [ ] Serialization, distribution, or installed-artifact behavior is covered
- [ ] Cross-repository compatibility is covered when a provider boundary changes

Commands and results:

```text
<commands, passed checks, and meaningful skips>
```

## Merge disposition

Select one:

- [ ] Product change intended for review and merge
- [ ] Execution/bootstrap helper that must be closed without merge
- [ ] Diagnostic result that cannot promote a method or scientific claim

Describe any required post-merge execution, human-governed action, or cleanup:
