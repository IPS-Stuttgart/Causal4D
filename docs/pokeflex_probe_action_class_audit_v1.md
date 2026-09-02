# PokeFlex source-only probe action-class audit

## Why this audit exists

The active PokeFlex protocol must not assume that archive labels such as `T1`
through `T6` denote comparable diagnostic actions across physical objects.
Before any held target object or drop outcome is opened, this audit tests that
assumption using only source and calibration poking robot records.

The audit is a prerequisite for the larger sequential experiment:

```text
initial state
  -> first diagnostic action
  -> observed response
  -> adaptive second diagnostic action or stop
  -> held drop-query prediction
```

## Frozen information boundary

The only readable payload is `robot_data.json` from the twelve registered
source and calibration objects and from poking takes `T1` through `T6`.

The audit may use:

- `frame`;
- `T_WT`, to describe the commanded/realized tool path; and
- `forces`, as a source-only response-diversity diagnostic.

It may not open:

- any archive belonging to the six registered target objects;
- any dropping archive;
- any mesh, image, point-cloud, or camera member; or
- any held challenge outcome.

## Action-class test

For every source/calibration object and take, a translation-invariant action
descriptor is formed from tool-path direction, path length, straightness,
duration, and mean step length. All descriptors are standardized on the
72-record source panel.

Take identity is then predicted under leave-one-object-out nearest-centroid
classification. The audit also compares the median distance between take-class
centroids with the median within-class distance.

The common `T1`--`T6` roster is admitted as a source-supported finite action
interface only when:

1. all `12 x 6 = 72` robot records are present;
2. leave-one-object-out take accuracy is at least 0.70;
3. the between/within separation ratio is at least 1.20;
4. at least five of six action classes have nonzero response variance; and
5. no target object is accessed.

A negative result does not terminate active probing. It means that the next
study must define probes directly from continuous tool-path descriptors or
source-only clustering rather than from take indices.

## Claim boundary

A pass establishes only that the six take indices are repeatable enough to act
as a finite source-supported diagnostic-action interface. It does not establish
probe value, query prediction, target transfer, online execution, or safety.
