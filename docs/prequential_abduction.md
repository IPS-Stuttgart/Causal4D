# Prequential factual-abduction diagnostics

## Scope

`causal4d.prequential_abduction` evaluates the unchanged factual-intervention
estimator at a strictly increasing sequence of causal `O+` prefix stops. It is a
diagnostic view of posterior evolution, not a recursive filter and not a new
estimator. No posterior from one prefix is used as the prior for another prefix.
Consequently, the final path step is exactly the ordinary one-shot
`abduct_factual_intervention` result for the same final prefix and settings.

The diagnostic does not change the frozen 36-execution method, registered
analysis, physical dataset, evidence count, fallback, or scientific claim.

## Outputs

`build_prequential_abduction_path` returns:

- every immutable `FactualIntervention` step;
- a content-addressed `PrequentialAbductionPathV1` binding the rollout bank,
  `TwinBelief`, component support, prefix stops, and factual artifact identities;
- the complete posterior weight vector at every prefix;
- entropy, effective sample size, maximum posterior mass, and MAP component;
- stabilized KL divergence and total-variation distance from the preceding
  prefix; and
- explicit metadata stating that zero future frames were read.

The KL diagnostic uses `numpy.finfo(float).tiny` only when an earlier posterior
has numerically underflowed to exact zero. The estimator and stored posterior
weights are not floored or otherwise modified.

## Dense observations

```python
from causal4d.prequential_abduction import build_prequential_abduction_path

result = build_prequential_abduction_path(
    bank,
    twin_belief,
    observations_from_endpoint_m,
    prefix_frame_counts=(2, 3, 4, 5, 6),
    observation_mask=observation_mask,
    config=abduction_config,
)

print(result.path.posterior_effective_sample_size)
print(result.path.previous_step_total_variation)
```

Changing observations after the largest declared prefix leaves the complete
path artifact byte-identical. Changing a later admissible prefix cannot alter
an earlier path step.

## Grouped and correlated observations

`grouped_observation_prefix` derives one covariance-consistent evidence object
for a prefix. Coordinates at or after the prefix stop are omitted. When an
observation group crosses the stop, the retained values and the matching
principal covariance submatrix are used; contributor identities, reliability,
outlier settings, source identity, and view identity are preserved.

```python
from causal4d.prequential_abduction import (
    build_prequential_abduction_path,
    grouped_observation_prefix,
)

prefix_evidence = grouped_observation_prefix(
    grouped_evidence,
    prefix_frame_count=4,
)

result = build_prequential_abduction_path(
    bank,
    twin_belief,
    observations_from_endpoint_m,
    prefix_frame_counts=(2, 3, 4, 5, 6),
    grouped_evidence=grouped_evidence,
    grouped_component_batch_size=64,
)
```

The bounded-memory grouped path must produce the same posterior steps and path
identity as dense component evaluation. Structured factual-abduction uncertainty
and identifiability results are supplied as mappings keyed by prefix count,
because those artifacts may be evidence- and query-specific.

## Intended diagnostics

The path helps identify:

- the first prefix at which the registered query becomes identifiable;
- posterior collapse caused by excessive or correlated evidence;
- abrupt changes attributable to one source, view, or contributor;
- a prefix that is longer than needed for stable inference;
- concentration without commensurate predictive stability; and
- sensitivity concentrated in the final permitted frame.

These summaries do not establish real-data calibration, provider competence,
physical contact ground truth, causal sufficiency, or counterfactual accuracy.
They should be interpreted alongside held-out execution-level prediction,
proper scores, coverage, interval width, and exact fallback accounting.
