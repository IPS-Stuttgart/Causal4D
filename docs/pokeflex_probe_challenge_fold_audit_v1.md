# PokeFlex probe-to-challenge fold audit v1

## Purpose

This audit asks whether the complete public PokeFlex mirror can support a
target-blind offline study of active, task-conditioned physical probing. It
freezes candidate diagnostic pokes and two held cross-intervention queries per
eligible object:

1. predict a different held robot-poke response; and
2. predict a held dropping response.

The audit is deliberately limited to filesystem metadata and ZIP central
directories. It does not open, decompress, hash, or extract any archive member.
In particular, it reads no robot trajectory, force response, mesh, image,
pointcloud, probe response, or challenge outcome.

## Frozen roster rule

For each parsed object identity, complete poking takes are ordered by a
content-independent SHA-256 ordering using the registered salt. One poke is
reserved as the held poke challenge, one as calibration, and the remaining
complete pokes form the candidate diagnostic-probe library. Dropping takes are
ordered independently; one is held as the drop challenge and a second is kept
as a reserve.

An object is eligible for the dual-query panel only when it has:

- at least three candidate diagnostic pokes after calibration and held-poke
  reservation;
- a held poke challenge; and
- at least two complete dropping takes.

The public mirror is admitted to the next source-only protocol stage only when
all 170 expected archives are classified as 116 poking and 54 dropping
interactions, every ZIP central directory is readable, no unsafe member path is
present, and at least 12 objects satisfy the dual-query roster.

## Next stage

A passing audit authorizes only:

> freeze a source-only action/initial-state carrier contract and a staged reveal
> protocol.

That next protocol may inspect candidate commands and permitted initial-state
carriers, but it must still select a probe before revealing the selected
probe response and must seal the post-probe prediction before opening the held
challenge outcome.

## Claim boundary

A passing audit proves roster feasibility, not probe value. Logged PokeFlex
interactions can support an offline real-data approximation to probe selection,
but they cannot by themselves establish online closed-loop robot execution,
safety, counterfactual outcomes from an identical microscopic state, or
deployment competence.
