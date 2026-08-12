# Automation integrity

This document separates three automation concerns that have different trust and
reproducibility requirements.

## Authenticated issue commands

Several workflows intentionally use a newly opened issue as a narrowly scoped
operator command. Each command authenticates the exact account login and numeric
account ID, requires one exact title, and requires the reviewed `main` revision.

Concurrency belongs to the authenticated execution job, not to the workflow as a
whole. GitHub Actions permits at most one pending run in a concurrency group. If
all opened issues enter a workflow-level group before the job guard is evaluated,
an unrelated issue can replace a pending authorized command. Job-scoped
concurrency means an unrelated issue produces only skipped jobs and never enters
the command's serialization group. The execution-job groups also use `queue: max`, so
multiple authenticated commands wait instead of replacing an earlier pending
command.

The policy test in `tests/test_issue_command_concurrency.py` inventories every
opened-issue workflow. Adding another issue command therefore requires an
explicit job name and concurrency group in the registry.

## Frozen and rolling compatibility lanes

The existing three-repository installed-wheel workflow is the frozen lane. It
uses reviewed immutable BayesianPhysTwin and Prob4D revisions, creates a
content-addressed stack lock, and remains the lane used for reproducibility and
claim-bearing release checks.

`three-repository-rolling-canary.yml` is intentionally different. It follows the
current `main` branches of Prob4D, BayesianPhysTwin, and Causal4D, records their
exact revisions, builds all three wheels, verifies installed-only imports, creates
an ephemeral stack lock, and runs the shared compatibility and provider-v2
contracts. Its artifacts always declare:

- `claim_bearing = false`;
- `frozen_pins_used = false`; and
- the exact three repository revisions used by the run.

A successful rolling canary detects compatibility at those moving revisions. It
does not update immutable pins and does not establish physical accuracy,
calibration, counterfactual validity, deployment safety, or any scientific claim.
A failure is an integration signal; frozen revisions remain unchanged until a
reviewed pin update passes the frozen lane.

## Release metadata integrity

`scripts/ci/check_release_metadata.py` validates the package version against
`CITATION.cff`, setuptools' dynamic version source, the current project-status
contract, and the changelog. The release workflow then builds one wheel and one
source distribution and requires agreement among:

- `causal4d.__version__`;
- the wheel filename and wheel `METADATA`;
- the source-distribution filename and `PKG-INFO`;
- the installed distribution metadata; and
- the citation and project-status metadata required for a tag.

For a tag, the tag must be exactly `v<package-version>`, the package must not be a
development or prerelease version, the `Unreleased` section must contain no
pending changes, and the project-status required version must match. The workflow
is read-only: it validates release inputs and uploads evidence but does not create
a tag or publish a release.

## Change procedure

For an issue-command workflow, keep the authorization guard on the execution job
and place its concurrency block below that guard. For compatibility changes, run
both the moving-head canary and the immutable installed-wheel lane before updating
any frozen pin. For a software release, first move all pending changelog entries
into the release section, synchronize package, citation, and project-status
versions, and only then create the exact matching tag.
