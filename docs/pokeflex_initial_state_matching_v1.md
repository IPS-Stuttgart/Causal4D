# PokeFlex initial-state matching certificate

## Reviewer question

The public interactions are separate physical executions. An offline active-probing
study must therefore show that a diagnostic poke is transported only to challenges
whose **pre-intervention geometry is sufficiently comparable**. Otherwise a policy
could appear useful merely because object reset quality varies between takes.

## Source-frozen answer

The certificate reads exactly the earliest framed dynamic OBJ mesh from every
registered interaction and no later mesh, robot record, force trace, probe response,
or challenge outcome. It uses nine source objects to freeze the 95th percentile of
a rigid-invariant, scale-normalised symmetric nearest-neighbour distance. Three
calibration objects test that caliper without changing it. Six target objects are
only assessed for pre-intervention eligibility.

Each object has four candidate diagnostic pokes selected by an outcome-free hash.
A challenge remains eligible only when at least three candidates fall below the
source-frozen reset caliper. The source-only policy experiment may proceed only if
all calibration objects and all target objects satisfy the registered query-coverage
gates.

## What the certificate establishes

A positive audit establishes a bounded logged-transport condition:

> the selected diagnostic observation and the held challenge originate from the
> same physical object and from initial geometries lying inside a source-frozen
> within-object reset envelope.

It does **not** establish identical microscopic state, an individual counterfactual,
online probe execution, deployment safety, or a causal effect of the probe itself.

## Information boundary

The result records the exact archive, mesh member, frame, byte count, vertex count,
and SHA-256 for every allowed initial mesh read. Any read of a terminal mesh,
unselected archive member, robot record, force trace, probe response, or challenge
outcome invalidates the audit.
