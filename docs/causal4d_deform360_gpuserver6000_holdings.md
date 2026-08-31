# Deform360 public holdings on `gpuserver6000`

## Purpose

This workflow qualifies and optionally preprocesses the public Deform360 data
already mounted on `gpuserver6000`. It does not collect new physical data and it
does not make completion of another experiment a paper requirement.

The local holdings are sufficient for two bounded tasks:

1. verify the exact raw-data prerequisites of the completed `001-rope` public
   case; and
2. identify additional ten-episode objects that can be converted to the official
   aligned Deform360 representation for later retrospective analysis.

They are not sufficient for a uniform 26- or 27-object benchmark because most
objects contain only one episode or incomplete camera coverage.

## Fixed mounted roots

The reviewed workflow reads these public-data roots:

```text
/mnt/lexar4tb/datasets/deform360/data-7fea8e2/raw
/mnt/lexar4tb/datasets/deform360/fresh-download-20260811-v1/raw
/mnt/lexar4tb/datasets/deform360-official-hub-visuotactile-v1
```

Derived files are written only below:

```text
/mnt/lexar4tb/datasets/deform360/causal4d-public-expansion-v1
```

The raw sources are fingerprinted before and after preprocessing. Any metadata
change in a source object is a hard failure.

## Qualification rules

A ten-episode raw candidate must have:

- shipped calibration files;
- at least 36 camera streams with ten paired video/timestamp records; and
- four tactile streams with ten paired array/timestamp records.

The exact `001-rope` reproduction prerequisite additionally requires all 41 raw
camera streams. A single-episode object can be recorded as a multiview tactile
calibration unit, but it is not promoted to a held-out action benchmark.

The six objects in the previously locked shared-physics cohort are inventoried
metadata-only and excluded from preprocessing by this workflow. Their protected
target episodes are not opened.

## Execution model

The operational workflow is:

```text
.github/workflows/deform360-public-holdings-gpuserver6000.yml
```

Its self-hosted job is main-only and uses:

```text
[self-hosted, Linux, X64, nvidia-smi, gpuserver6000]
```

A change to:

```text
ops/deform360-gpuserver6000-request.json
```

runs a GitHub-hosted dispatcher that invokes the reviewed main workflow. The
current request permits at most four objects: the exact `001-rope` reproduction
plus the fixed exploratory candidates `003-cable`, `086-cotton-scarf-cloth`, and
`171-penguin`. Each object is processed only when the live inventory independently
passes the registered ten-episode qualification. The six protected locked-cohort
objects remain excluded before any preprocessing payload is opened.

## Interpretation

The resulting artifact answers whether the mounted files are sufficient and
whether official preprocessing succeeds. Qualification or preprocessing alone
is not a new predictive result and authorizes no new paper claim. Existing
Causal4D metrics remain bound to their existing evidence records.
