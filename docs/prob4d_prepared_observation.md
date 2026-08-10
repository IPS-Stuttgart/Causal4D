# Prepared Prob4D joint observations

`causal4d.prob4d_prepared_observation` is the reusable execution layer for
validated Prob4D full-joint observation evidence.

The existing `joint_observation_from_prob4d` adapter remains the semantic and
provenance boundary. Preparation performs that validation once, constructs the
same `LinearJointObservationEvidence`, and compiles its sparse rollout operator
and structured Gaussian base solver:

```python
from causal4d.prob4d_prepared_observation import (
    prepare_prob4d_joint_observation,
)

prepared = prepare_prob4d_joint_observation(
    descriptor,
    arrays,
    rollout_frame_ids=rollout_frame_ids,
    entity_to_node=entity_to_node,
)

log_likelihood, diagnostics = prepared.log_likelihoods(
    rollout_components_m,
    prefix_frame_count=prefix_frame_count,
    component_chunk_size=32,
)
```

## Guarantees

Preparation does not alter the Prob4D covariance model. Local `3 x 3`
covariance blocks, the shared low-rank gauge factor, row ordering, frame/entity
mapping, reliability policy, and source provenance are inherited from the
validated adapter without reinterpretation.

Repeated calls reuse the exact base covariance factorization, use the compiled
sparse selector operator, and respect the prepared path's explicit working-memory
budget. Component-specific covariance and low-rank factors retain their existing
semantics. Posterior updates preserve exact zero prior support.

Portable strict descriptors and historical adapter fixtures place source
identity in slightly different locations. The prepared adapter accepts
`source_revision` and `source_artifact_sha256` either at descriptor level or in
descriptor metadata, while rejecting any disagreement between the two.

## Relationship to the tree-block handoff

This path accelerates finite-rollout likelihood evaluation for validated
Prob4D observation beliefs. It is complementary to the strict
BayesianPhysTwin tree-block query provider and belief handoff:

- the tree-block path transfers an admitted BayesianPhysTwin posterior and
  registered query covariance without dense posterior materialization;
- the prepared path evaluates Causal4D rollout support repeatedly against one
  validated full-joint Prob4D observation artifact.

Neither path permits Causal4D to consume the same observation factors twice.
The existing evidence-ownership ledger remains authoritative at the
BayesianPhysTwin handoff.

## Scientific boundary

This is numerical execution infrastructure. It does not establish observation
competence, empirical covariance calibration, physical benefit, intervention
benefit, deployment safety, or a new physical evidence count.
