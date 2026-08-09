# Changelog

## Unreleased

### Added

- Add an optional, content-addressed per-view observation-evidence contract
  that retains ordered camera streams, material identities, confidence,
  calibration, object-frame transforms, shared sensor context, and the derived
  fused observation. Execution manifests bind it to the registered clock,
  frame count, and six-frame causal prefix without changing the required frozen
  artifact inventory.
- Add a fixed-candidate independent-confirmation contract for prospective V2
  selections. It revalidates the selection evidence, rejects unit, target,
  independence-group, session, seal, trace, metric, and scoring-run reuse, keeps
  the metric contract and thresholds fixed, prohibits candidate re-selection,
  and preserves the exact registered baseline when confirmation fails.
- Add evidence-bound prospective V2 promotion contracts that freeze the
  target inventory and metric semantics, derive acceptance, exact fallback,
  and harmful-update status from validated decision traces and raw scores,
  and require a fresh independent confirmation panel after candidate selection.
- Add source-only task-projected functional-support certification over frozen
  linear readouts, including posterior-mixture covariance, optional
  component-specific low-rank uncertainty modes, exact Gaussian-mixture
  intervals, and content-addressed source provenance.
- Add the versioned prospective V2 deployment-decision profile, fixing the
  required gate inventory, producer, stage, and exact baseline fallback without
  admitting the separate target-opening promotion experiment.
- Add a strict non-pickled physical evaluation target with canonical float32
  observation identities, target/context validation, and atomic exactly-once
  publication.
- Add an explicit legacy importer that requires unsafe-pickle consent and an
  independently obtained SHA-256 before opening PhysTwin `final_data.pkl`.
- Add a declarative Bayesian-PhysTwin provider registry that owns the complete
  versioned import inventory, API suffixes, roles, lifecycle labels, and local
  contract modules. The AST import boundary and documentation validation now
  consume that registry instead of maintaining a separate hard-coded allowlist.
- Document the additive belief-provider-v2 horizon-discrepancy boundary, released
  artifact-v2 surface, verified scheduled-contact replay integration, and portable
  external sparse-trajectory forecast import without changing frozen v1 results.
- Add a locked, source-only Deform360 filament-support boundary that preserves
  exact registered graphs on connected resets and applies a deterministic
  component-level minimum-spanning bridge only to the frozen disconnected
  filament resets.
- Bind the structure candidate to the completed observed-reset result, require
  exact common-case parity and predeclared bridge/locality and same-object
  geometry gates, and keep mechanics rescoring, calibration, and target data
  closed.
- Add a locked source-only Deform360 contact/support mechanism diagnostic that
  reproduces the archived backend before comparing support-height, visual
  contact-patch, source-fitted opening-schedule, and no-contact controls across
  the unchanged 30 opened source episodes.
- Bind the mechanism panel to the completed source-failure and negative
  prefix-kinematics results, apply independent preregistered gates to each
  physical candidate, and keep calibration and target outcomes closed.
- Add a locked, source-only Deform360 prefix-kinematics diagnostic that
  reproduces the archived zero-velocity baseline, compares rigid and
  graph-harmonic causal velocity fields on the unchanged source candidates,
  preserves the p99 strain constraint, and keeps calibration and target data
  closed.
- Bind the diagnostic to the frozen source milestone and terminal backend
  decision, record exact runtime provenance, and provide focused tests plus a
  manual self-hosted evidence workflow.
- Add a fail-closed source-panel status that validates the exact registered
  12-execution prefix, reports the next physical execution with its complete
  command profile, distinguishes invalid evidence from valid incompleteness, and
  requires hash verification before completion.
- Add exactly-once source-manifest publication. The publisher admits only the
  next registered execution, recursively rejects target-outcome fields, verifies
  every referenced artifact digest and byte count, validates the temporary
  manifest, and never overwrites a final evidence path.
- Add the physical source-panel operator runbook and adversarial coverage for
  out-of-order completion, modified templates, stale artifact hashes, incomplete
  status, and repeated publication.
- Add content-addressed rollout-bank archives with exact member inventories,
  strict finite-JSON manifests, atomic validation-before-replace publication,
  explicit no-overwrite support, and legacy archive loading.
- Add a stable rollout-bank identity over hypothesis metadata, priors, physical
  parameter support, trajectories, variance floor, and confidence level.
- Add explicit dense factual-abduction likelihood semantics. `legacy_v1` remains
  the registered identity-preserving default, while opt-in `normalized_v2`
  retains particle-specific scale normalization, includes the
  endpoint-to-first-response increment, and models adjacent-frame correlation.

This control plane advances pre-acquisition operations without creating physical
evidence. Source-panel executions remain source-only and cannot increment the
`0/36` confirmatory evidence count.

### Fixed

- Read trusted pickle and Bayesian-PhysTwin NumPy archives from one descriptor-bound, symlink-free snapshot; reject duplicate, unsafe, oversized, object-dtype, or digest-mismatched NPZ inputs before use.
- Reject lossy Boolean, string, and floating-point coercion at sparse trajectory, observed-node, and Bayesian-PhysTwin grid-index boundaries.
- Name the registered dense factual update explicitly as `update_from_observations_legacy_v1` while retaining the historical method as an exact compatibility alias.
- Advance the acquisition-environment capsule to schema version 2 and bind the
  active Causal4D and BayesianPhysTwin installation provenance to the exact
  supplied wheel bytes through strict PEP 610 metadata, a second archive
  SHA-256/size check, and byte verification of every installed wheel member
  except the installer-rewritten `RECORD`. Same-version wheel substitution or
  post-installation member drift is rejected before staging or sealing.
- Bind every strict independent-sensor update to any consumed-evidence ledger
  already embedded in its factual posterior, rejecting stale-ledger rollback and
  duplicate factor multiplication while preserving valid sequential updates.
- Make independent actuator and contact-wrench evidence publication atomic,
  validate the exact temporary archive before publication, refuse replacement by
  default, and reject symlinked inputs, duplicate members, extra arrays, or
  non-strict descriptor JSON during loading.
- Include the runnable MolmoMotion bridge helpers in source distributions and
  execute their end-to-end regression from the extracted archive, preventing a
  published sdist from retaining tests and documentation for files it omits.
- Remove the completed issue-233 reviewed-publisher workflow and extend the
  mergeable-head policy to reject both temporary and `publish-reviewed-*`
  one-shot workflows after their bounded purpose is complete.
- Remove direct pickle loading from the stable claim-bearing physical evaluator,
  bind results to posterior, query, physical-target, and held-out-suffix IDs, and
  publish result JSON atomically with no-overwrite behavior by default.
- Replace mutable `dict`/`list` subclasses used for frozen JSON with read-only
  mapping and sequence value objects, closing `dict.__setitem__`, `dict.update`,
  `list.append`, and `list.__setitem__` base-class mutation bypasses while
  preserving valid JSON values and content identities. Frozen containers now
  expose the explicit read-only `Mapping`/`Sequence` protocols; use
  `plain_json()` at mutable built-in or serialization boundaries.
- Back frozen NumPy arrays with immutable byte storage so callers cannot restore
  write access through `setflags(write=True)` or `flags.writeable = True`.
  Serialization boundaries now export explicit plain JSON, preserving existing
  content identities and artifact schemas for valid inputs.
- Add adversarial regression and source-policy coverage for built-in container
  mutation bypasses and attempts to re-enable NumPy write access.
- Reject coercible or schema-drifted Causal4D contract descriptors, archive
  inventories, support indices, and grouped-observation indices before they can
  change content identity or evidence selection. Preserve exact JSON numeric
  types during loading so integer-valued semantic temperatures round-trip.
- Reject non-finite grouped Student-t mixture controls and observation prefixes
  that exceed the scored rollout instead of propagating invalid likelihoods.
- Preserve the initial prior mass removed by contact-path pruning, including
  one-frame paths, and fail explicitly when the threshold removes every initial
  contact regime.
- Reject non-finite, zero, or negative fixed-contact posterior temperatures
  before simulation instead of allowing invalid posterior weights to propagate.
- Enforce strict prefix-only validation for fixed-contact and rollout-bank online
  updates, including zero-power calls, and reject invalid parameter-support limits.
- Normalize and recursively freeze rollout-hypothesis metadata so external or
  nested mutation cannot change intervention semantics after bank construction.
- Route rollout-producing and rollout-consuming commands through one strict
  archive implementation, while retaining exact support for legacy banks.
- Reject non-finite factual-abduction controls, correlation outside `(-1, 1)`,
  legacy requests that specify a correlation model, and contradictory requests
  for normalized dense and grouped full-covariance likelihoods.

These changes harden diagnostic accounting, artifact integrity, and validation.
The registered `legacy_v1` likelihood and posterior remain unchanged, and
`normalized_v2` is not admitted into the frozen estimator, protocol, thresholds,
evidence, or target identities.

## 0.5.0

### Breaking CLI consolidation

- Install exactly one executable: `causal4d`.
- Remove all 67 historical `causal4d-*` console scripts from current packages.
- Preserve every historical name as machine-readable migration metadata.
- Expose all retained functionality through typed grouped routes.
- Add lifecycle, optional-extra, provider, owner, and claim-boundary metadata.
- Require installed wheel and source-distribution tests to exercise `--help` for
  every grouped route and reject any residual historical wrapper.
- Update the registered real-analysis command strings to grouped invocations.
- Preserve the same fail-closed provenance, evidence, and posterior validation
  inside each handler; the migration changes invocation paths, not scientific
  semantics or admissibility checks.

Frozen tags, milestone files, and recorded environments are unchanged.
