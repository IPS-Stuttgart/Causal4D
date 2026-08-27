# Query-specific intervention-equivalence certificate v1

## Purpose

A realized-intervention posterior can fail exact latent-label recovery while still
supporting an accurate downstream prediction. The reverse can also occur: a
posterior can select the correct label but remain too diffuse, unsupported, or
poorly calibrated for the registered future query. These are different
scientific questions.

`causal4d.intervention_equivalence_v1` records them separately for one finite
Causal4D intervention support:

1. **exact identity** — whether the posterior MAP intervention is the registered
   truth in a controlled study;
2. **prefix indistinguishability** — whether hypotheses have sufficiently similar
   predictions under the permitted causal response prefix;
3. **query equivalence** — whether hypotheses have sufficiently similar
   predictions under one frozen downstream query operator; and
4. **physical equivalence** — whether hypotheses describe the same physical
   contact, force, delay, slip, or actuation mechanism.

The certificate implements the first three. It never promotes query equivalence
or prefix indistinguishability to physical equivalence. Physical equivalence
requires an independent measurement channel, such as force/torque, tactile,
actuator, or contact-onset evidence, under a separately registered study.

## Registered finite-support problem

Let the intervention support be

```text
Z = {z_1, ..., z_n},       posterior weights w_i >= 0, sum_i w_i = 1.
```

For each hypothesis, the caller supplies two *predicted* signatures:

```text
s_i = causal-prefix response signature
q_i = registered future-query signature
```

and component-wise positive scales `sigma_s` and `sigma_q`. Distances are
standardized root-mean-square distances,

```text
d_s(i,j) = rms((s_i - s_j) / sigma_s)
d_q(i,j) = rms((q_i - q_j) / sigma_q).
```

The query signature may be a complete trajectory vector, a registered linear
projection, a collection of moments, or another frozen finite-dimensional query.
It must not contain an observed target future. The scale, signature construction,
and both diameter tolerances must be frozen before the evaluation cohort is
opened.

## Why a threshold graph is insufficient

The pairwise relation `d(i,j) <= epsilon` is generally not transitive. Connected
components can therefore create a chain in which adjacent hypotheses are close
but the two endpoints are far apart. That would make an “equivalence class”
depend on chaining rather than a certified within-class diameter.

The v1 certificate instead uses deterministic complete-link agglomeration:

1. start from singleton blocks sorted by intervention identity;
2. admit a merge only when the full union diameter is within the registered
   tolerance;
3. choose the admissible merge with the smallest resulting diameter, using
   lexicographic member identity as the exact tie-break; and
4. stop when no admissible merge remains.

This produces an input-order-invariant partition in which every prefix block and
every query block satisfies its own registered diameter tolerance. The joint
partition is the common refinement of the prefix and query partitions.

Approximate blocks are protocol-specific tolerance certificates. They are not a
claim that the underlying physical interventions are mathematically identical.

## Quotient posterior

The posterior mass of a block is the sum of its member masses. The artifact
reports exact-support, prefix-block, query-block, and joint-block entropy,
effective support, MAP-block mass, and credible block identities. In a controlled
study with known truth, the MAP result is classified without changing the exact
endpoint:

```text
exact_identity
jointly_equivalent
query_equivalent_only
prefix_indistinguishable_only
distinct_or_unresolved
```

The registered exact-identity success indicator remains unchanged. A favorable
query-equivalence result is an additional endpoint, not a relabeling of an exact
miss.

## Finite-support query concentration bound

Let `z_hat` be the MAP intervention, `B_Q(z_hat)` its query block, and

```text
m = posterior mass of B_Q(z_hat)
r = max_{z in B_Q(z_hat)} d_q(z, z_hat)
R = max_{z in Z} d_q(z, z_hat).
```

For the posterior mean query `q_bar = sum_i w_i q_i`, norm convexity gives

```text
d_q(q_bar, q(z_hat))
    <= sum_i w_i d_q(q_i, q(z_hat))
    <= m r + (1 - m) R.
```

The certificate recomputes all three quantities and verifies both inequalities.
The bound answers a narrow question: how far the posterior-mean registered query
can be from the MAP-intervention query in the declared standardized RMS metric.
It is not a proper-score guarantee, an empirical calibration result, a physical
identification theorem, or a safety certificate.

## API

```python
from causal4d.intervention_equivalence_v1 import (
    build_intervention_equivalence_certificate_v1,
    write_intervention_equivalence_certificate_v1,
)

certificate = build_intervention_equivalence_certificate_v1(
    protocol_id="controlled-contact-v2",
    query_id="held-out-trajectory-v1",
    intervention_ids=component_ids,
    posterior_weights=posterior_weights,
    prefix_signatures=predicted_prefix_vectors,
    prefix_scale=registered_prefix_scale,
    query_signatures=predicted_query_vectors,
    query_scale=registered_query_scale,
    prefix_diameter_tolerance=registered_prefix_tolerance,
    query_diameter_tolerance=registered_query_tolerance,
    confidence_level=0.90,
    truth_intervention_id=truth_id,  # controlled studies only
)

write_intervention_equivalence_certificate_v1(
    "intervention-equivalence.json",
    certificate,
)
```

Inputs are sorted by intervention identity before computation. Posterior weights
are normalized deterministically, MAP ties are broken lexicographically, every
block has a content identity, and the complete artifact has a canonical SHA-256
identity. Loading independently recomputes every derived block, mass, entropy,
recovery label, and query bound. Publication is no-clobber; a byte-identical
replay is idempotent.

## Required prospective evaluation

A claim-bearing evaluation should use disjoint development and evaluation seed or
physical-session sets. Before evaluation access, freeze:

- the intervention support and component identities;
- the causal-prefix cutoff and prefix-signature operator;
- the downstream query and query-signature operator;
- component scales and both complete-link diameter tolerances;
- the confidence level and exact MAP tie-break;
- the baseline, posterior construction, and fallback policy; and
- all exact-identity, query-equivalence, calibration, and harmful-group gates.

Report at complete independent-unit level:

- exact MAP recovery and credible-set coverage;
- prefix-, query-, and joint-block recovery;
- exact and quotient posterior entropy;
- MAP query-block mass and the concentration bound;
- held-out trajectory proper scores and calibration;
- results by topology, action, contact support, and intervention family; and
- every unsupported or exact-fallback case.

A query-equivalence endpoint cannot rescue a failed trajectory score,
calibration gate, support gate, or exact physical-intervention endpoint. Likewise,
exact recovery cannot rescue a poor held-out query.

## Relationship to the existing Causal4D evidence

The retained controlled topology diagnostic reports 75% exact-node recovery and
100% one-hop recovery, with all exact-node misses improving trajectory RMSE. That
result motivates a query-specific quotient analysis, but it is not retroactively
converted into a v1 certificate: its one-hop, graph-diffusion, and symmetry
proxies were diagnostic quantities, not a prospectively frozen query partition.
The failed 80% exact-node gate remains failed.

The registered 18-session/36-execution physical experiment is still the decisive
next evidence for the first Causal4D paper. This additive module does not change
that estimator, protocol, endpoint, acquisition order, or evidence count and
must not be used as an optional-branch rescue. A future controlled replication or
the separately registered blinded known-intervention challenge may freeze this
certificate as a secondary endpoint before opening its new evaluation cohort.

## Claim boundary

A valid certificate establishes deterministic finite-support partitioning,
quotient posterior accounting, and the stated MAP-query concentration bound for
the exact supplied signatures and tolerances. It does not establish:

- real contact, force, delay, slip, or actuation recovery;
- physical equivalence of two interventions;
- individual-level counterfactual ground truth;
- calibrated predictive uncertainty;
- unseen-object or object-class generalization;
- real Prob4D provider competence;
- closed-loop robot safety; or
- state of the art.
