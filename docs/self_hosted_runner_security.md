# Self-hosted runner security boundary

Causal4D self-hosted jobs execute on long-lived machines that may expose GPU
state, local caches, services, and approved source-only datasets. A read-only
`GITHUB_TOKEN` does not protect those resources from unreviewed repository code.
This document defines the authorization and recovery boundary for every job
whose runner selection can include `self-hosted`.

## Machine-readable inventory

`.github/self-hosted-jobs.json` is the complete allowlist. Each entry binds the
workflow file, job identifier, authorization model, required runner labels,
purpose, dataset-access class, and whether GitHub secrets are permitted. The
repository-wide policy test discovers self-hosted jobs directly from
`.github/workflows/` and fails when a job is missing from the registry, a stale
entry remains, or a registered job violates its authorization contract.

The repository has two reviewed-main authorization models:

- `main-only` for manual `workflow_dispatch` from `refs/heads/main`; and
- `maintainer-issue-main` for one exact registered-maintainer issue trigger
  whose workflow is itself already present on reviewed `main`.

Neither model permits pull-request source on a self-hosted runner. Introducing a
protected exact-PR-head model still requires a new registry schema, independent
approval through a protected environment, hosted authorization of the exact
PR/head pair, and separate positive and stale-head negative controls.

## Main-only authorization

A main-only job must be rejected before runner allocation unless all applicable
conditions hold:

1. the event is a manual `workflow_dispatch`, or the workflow has no event other
   than `workflow_dispatch`;
2. the job-level condition requires `github.ref == 'refs/heads/main'`;
3. the first repository checkout is explicitly bound to `${{ github.sha }}`;
4. every checkout uses `persist-credentials: false` and a full commit-pinned
   action;
5. the job verifies `git rev-parse HEAD == GITHUB_SHA` and a clean work tree
   before installing or executing repository code;
6. workflow permissions default to `contents: read`; and
7. the self-hosted job references no GitHub secret or write permission.

Hybrid hosted/self-hosted jobs may continue to validate pull requests on hosted
runners, but their job-level expression must make self-hosted selection imply a
manual dispatch from `main`. The runtime preflight repeats that implication so a
future expression regression fails before substantive work.

## Exact maintainer-issue authorization

`maintainer-issue-main` exists for connector-driven executions when the
client cannot call GitHub's workflow-dispatch endpoint. It is deliberately
narrower than a general issue-triggered job. The self-hosted job must require all
of the following before runner allocation:

1. the workflow listens only for newly opened issues;
2. `github.ref == 'refs/heads/main'`;
3. the issue login is exactly `FlorianPfaff`;
4. the issue account ID is exactly `6773539`;
5. the issue title equals the one registered literal trigger;
6. no issue title, body, label, comment, or attachment enters a shell command,
   path, package argument, test selector, or executable configuration;
7. checkout, action pinning, exact-SHA verification, clean-tree verification,
   permissions, and secret rules are identical to `main-only`; and
8. the executed revision is the default-branch `GITHUB_SHA` selected by the
   issue event, never a value supplied by the issue.

A separate GitHub-hosted reporting job may receive `issues: write` solely to
post the fixed workflow result, exact main SHA, run URL, and artifact name to the
trigger issue. It does not check out or execute repository code. The
self-hosted job remains read-only and secret-free.

## Runner-account isolation

The runner account must be a dedicated, non-administrative account. It must not
share a home directory, SSH agent, browser profile, cloud CLI configuration,
Docker credential store, or interactive login session with researchers or
operators. Repository work directories and virtual environments are disposable.
The account must not have passwordless privilege escalation.

Long-lived service credentials must not be present in environment variables,
shell startup files, Git configuration, credential helpers, or process-global
agent sockets. Public repositories are fetched without deploy keys. GitHub
Actions caches are disabled in self-hosted jobs unless a separately reviewed,
content-addressed cache design is registered.

## Dataset mounts

Registry value `dataset_access: none` means the runner job must not depend on an
unopened or private dataset mount. `approved-source-only-deform360` permits only
the already-approved source/calibration-side Deform360 replication root used by
the owning source-only protocol. Confirmation objects, confirmatory Causal4D
executions, and unrelated laboratory data must be absent or inaccessible.

A workflow authorization is not scientific data-access authorization. The
owning preregistered protocol must independently permit every mounted object,
episode, or execution. Changing a mount from source/calibration to confirmation
requires a new protocol gate and cannot be achieved by editing this registry.

## Network and credential policy

Self-hosted jobs may use outbound network access only for public, commit-pinned
source retrieval and declared Python packages. Inbound network services should
be disabled. The runner must not expose laboratory file shares, credential
brokers, personal SSH keys, cloud metadata credentials, or writable package
mirrors to job processes.

The repository and environment secret sets for registered jobs must remain
empty. If a future job genuinely requires a secret, it needs a distinct
protected authorization model, least-privilege credential, independent review,
and an explicit rotation and incident-response procedure before registration.

## Evidence and cleanup

Operational jobs retain the exact source SHA, workflow run and attempt IDs,
runner identity, package or wheel identities where applicable, resolved runtime,
and output hashes. Failed jobs retain their partial status artifacts when the
workflow supports them. Temporary virtual environments and provider checkouts
are removed in `always()` cleanup steps where practical.

The policy test includes accepted fixtures for both authorization models and
negative fixtures for unauthorized dispatch, broad issue execution, stale
checkout, missing registered maintainer identity, missing exact title, unpinned
actions, credential persistence, and absent clean-tree verification.

## Incident response

Treat any execution of unreviewed source, unexpected secret visibility, unknown
process, altered mount, or unexplained cache content as a runner compromise.
Immediately:

1. remove the runner from GitHub and stop the runner service;
2. revoke and rotate every credential that could have been visible to the
   account or host;
3. preserve relevant workflow, system, process, and network logs;
4. quarantine generated evidence until provenance is re-established;
5. rebuild the runner from a trusted image rather than cleaning it in place;
6. re-register only the required labels and approved mounts; and
7. run the positive and unauthorized negative policy controls before restoring
   operational use.

## Rebuild and rotation

Rebuild the runner after a security incident, a privilege or mount change, an
unreviewed administrative login, or a material base-image/runtime upgrade.
Rotate the runner registration token during every rebuild. Periodically verify
that the account remains non-privileged, approved mounts match the registry,
there are no credential helpers or unexpected services, and no abandoned work
or environment directory is reused across claim-bearing runs.
