# Stable-core typing ratchet

## Purpose

Causal4D's stable counterfactual and observation core is now checked by a dedicated
MyPy workflow in addition to the narrower provider/artifact checks in the main CI
job. The expanded scope covers the public contracts, counterfactual operator,
grouped likelihood, factual intervention abduction, grouped observation evidence,
and versioned API surface.

The first complete check exposed four pre-existing diagnostics. They are retained as
an **exact debt set**, not hidden through broad `ignore_errors`, module exclusions,
or unchecked `# type: ignore` comments. The gate executes MyPy over the complete
scope and accepts only these exact file, line, error-code, and message identities:

| File | Line | Code | Current diagnostic |
| --- | ---: | --- | --- |
| `src/causal4d/contracts.py` | 1081 | `arg-type` | NumPy's `savez_compressed` stub interprets a dynamic archive member as its reserved Boolean keyword. |
| `src/causal4d/intervention_abduction.py` | 253 | `var-annotated` | `flat_indices` needs an explicit NumPy array annotation under the current inference settings. |
| `src/causal4d/intervention_abduction.py` | 608 | `no-redef` | the posterior metadata mapping reuses the earlier loop variable name `metadata`. |
| `src/causal4d/intervention_abduction.py` | 664 | `var-annotated` | the reconstructed joint-weight matrix needs an explicit NumPy array annotation. |

Any additional diagnostic fails. A moved line, changed error code, changed message,
duplicate diagnostic, or missing registered diagnostic also fails. Missing debt is a
failure because it means source and debt manifest changed independently; the repair
and manifest removal must land together in one reviewed pull request.

## Why exact debt instead of a broad allowlist

A broad MyPy exclusion would establish no protection for these modules. An error-code
allowlist would permit unrelated errors of the same class. A path-only allowlist
would permit arbitrary typing regressions in a large scientific module. The exact
manifest therefore acts as a monotonic ratchet:

```text
current diagnostics == registered diagnostics
```

The workflow runs from `scripts/ci/check_stable_core_mypy.py`. Its parser consumes
non-pretty, error-code-bearing MyPy output, and adversarial tests cover missing,
unexpected, duplicate, and message-drift cases. Source-line tests bind every debt
entry to the current statement that produces it.

## Retiring debt

Each item should be removed by a focused repair:

1. change the source without altering numerical, archive, or artifact semantics;
2. remove the matching `EXPECTED_DEBT` entry and its source-line assertion in the
   same commit or pull request;
3. run the dedicated workflow and the affected behavioral tests; and
4. retain no replacement ignore unless the external type contract is impossible to
   express and the ignore is narrower and more auditable than the former debt.

The NumPy archive item should be resolved with a typed wrapper or cast around the
runtime's arbitrary named-member interface, while preserving the archive member
inventory and bytes. The three abduction items require annotations or a local
variable rename only. None requires an estimator change.

## Scope

The permanent target inventory includes:

- immutable and atomic artifact utilities;
- `contracts.py`;
- `counterfactual.py`;
- `grouped_likelihood.py`;
- `intervention_abduction.py`;
- `observation_evidence.py`;
- `api/v1.py`;
- provider and replay-provider contracts; and
- CI utilities themselves.

Expanding this inventory is additive. A proposed module must first be checked under
the same command. New debt should normally be repaired before inclusion rather than
registered.

## Scientific boundary

This ratchet changes static-analysis enforcement only. It does not change a
likelihood, posterior, counterfactual trajectory, covariance, artifact identity,
provider schema, registered protocol, target-access boundary, acquisition dataset,
physical evidence count, or scientific claim. The four entries are implementation
debt, not empirical findings.
