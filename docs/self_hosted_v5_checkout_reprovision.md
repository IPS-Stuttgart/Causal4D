# Self-hosted v5 acquisition-checkout reprovision

The registered v5 acquisition dataset is persistent evidence. The software
checkout used to interpret it is a separate operational object. When the
checkout is incomplete before the method freeze, Causal4D may replace that
checkout from an exact reviewed `main` revision without editing the dataset.

The issue-triggered workflow uses the exact title:

```text
[self-hosted] reprovision Causal4D v5 acquisition checkout
```

It runs only for the registered maintainer on reviewed `main` and only on the
`workstation2` runner carrying the acquisition-data label.

## Admission boundary

Reprovision is admitted only when all of the following are true:

- the issue-event checkout is clean and exactly equals `GITHUB_SHA`;
- the deployed checkout is an ordinary, clean Git checkout;
- the v5 dataset is an ordinary directory with no symlinked members;
- `method_freeze.json`, its validation, and the registered analysis manifest do
  not exist;
- no confirmatory execution or session manifest exists;
- the hash-verified next action derived from the issue-event checkout is exactly
  `complete_object_registration`;
- that action is nonphysical, nonautomatable, target-free, and does not change
  the registered method; and
- a newly staged clone derives the same portable readiness-evidence identity.

A dirty deployed checkout, a later next action, any freeze artifact, any
physical-execution evidence, or any target-access permission fails closed before
the deployed path changes.

## Atomic replacement

The workflow clones the exact reviewed source revision into a sibling staging
path, checks the required v5 files, verifies a clean detached commit and Git tree,
and replays the registered next-action decision against the existing dataset.
Only then does it perform two same-filesystem renames:

1. move the prior deployed checkout to a deterministic retained rollback path;
2. move the validated staging checkout into the registered deployment path.

The retained rollback checkout is never selected by the readiness workflow. A
post-installation verification failure quarantines the new checkout and restores
the original path.

## Dataset noninterference

Every ordinary dataset file is hashed before and after replacement, so the
registered dataset is byte-preserved. Success requires the complete descriptor
list and aggregate tree digest to be identical. The workflow then reruns the
ordinary read-only acquisition-readiness probe from the newly deployed checkout.

The retained report states:

```text
dataset_modified=false
target_outcomes_used=false
device_nodes_opened=false
physical_command_sent=false
registered_method_changed=false
physical_evidence_increment=0
```

Physical evidence increment: `0`.

## Result interpretation

A successful run repairs software provisioning and should recover the registered
`complete_object_registration` action. It does not complete object registration,
select contact nodes, approve a gate, freeze the method, send a robot command, or
authorize confirmatory execution. The physical object identity and registered
contact geometry still require the separately governed operator procedure.
