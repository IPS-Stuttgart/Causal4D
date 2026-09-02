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

Poking and dropping folders reuse raw stems such as `Object_T1`. The audit
therefore assigns the canonical action-qualified identities `poking:Object_T1`
and `dropping:Object_T1`; raw stems remain metadata only. For each parsed
object identity, complete poking interactions are ordered by a
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

## Historical-exposure boundary

The active-probing claim cannot be called globally prospective merely because
this workflow has not opened a payload. The earlier BayesianPhysTwin freshness
audit records 108 previously exposed public poke takes. Its subsequent fresh6
panel scored six of the eight then-unexposed candidates. Consequently, almost
all historically available poke outcomes have already appeared somewhere in the
broader project evidence chain.

The present audit therefore separates two questions:

1. whether the complete mirror has enough action and outcome carriers to define
   the experiment; and
2. which challenge outcomes remain eligible for a fresh target claim after an
   independent exact-take exposure scan.

The strongest credible next route is expected to be **retrospective diagnostic
pokes followed by previously unopened dropping challenges**. A target protocol
must verify that exact drop-take outcomes are absent from prior repositories,
workflow artifacts, and retained runtime paths before it calls those outcomes
fresh. If that scan fails, the study remains a retrospective real-data mechanism
analysis rather than a prospective confirmation.

A held-poke fold produced by this audit is useful for source development and
mechanism controls. It is not automatically a fresh target merely because the
salt selected it.

## Next stage

A passing audit authorizes only:

> freeze a source-only action/initial-state carrier contract and a staged reveal
> protocol.

That next protocol may inspect candidate commands and permitted initial-state
carriers, but it must still select a probe before revealing the selected
probe response and must seal the post-probe prediction before opening the held
challenge outcome.

For target objects, `robot_data.json` co-locates measured tool poses and force
responses. The next stage must therefore use an isolated semantic slicer: it may
export a registered action descriptor from the permitted pose fields, but it may
not expose force values to probe selection. This is a semantic information
boundary, not a byte-level claim that the carrier was never opened.

## Intended real-data progression

If the metadata and exposure gates pass, the source-only study should compare:

- no probe;
- deterministic random-safe probing;
- one source-selected fixed-safe rule;
- generic latent-information probing;
- task-conditioned query or decision value; and
- an outcome-aware oracle reported only as a diagnostic ceiling.

The selected poke response is then revealed, the physical belief is updated,
and predictions for all registered drop challenges are sealed before any drop
mesh outcome is opened. Objects, not frames or mesh vertices, are the statistical
units. A dependence-destroying control must preserve the marginal probe and drop
summaries while breaking their object-matched relation.

## Claim boundary

A passing audit proves roster feasibility, not probe value. Logged PokeFlex
interactions can support an offline real-data approximation to probe selection,
but they cannot by themselves establish online closed-loop robot execution,
safety, counterfactual outcomes from an identical microscopic state, or
deployment competence.

Even a positive fresh-drop result would support only target-blind
cross-intervention query or decision improvement after selecting one logged
physical poke. It would not prove that a robot can execute the selected probe
online from the same state or that the action is deployment-safe.
