# Independent-sensor ablation and causal attribution

## Status

This is a diagnostic-only extension to the existing sensor-factorized abduction
path. It does not modify the frozen factual estimator, the six-frame object
prefix, the registered 18-session/36-execution protocol, any calibration rule,
or the exact physical fallback.

The purpose is to distinguish information supplied by independently measured
actuator motion from information supplied by force/contact evidence after the
object-response prefix has already been consumed exactly once.

## Four posterior arms

`build_independent_sensor_ablation` constructs four arms from one immutable
`FactualIntervention`:

1. `object_prefix`: the input posterior unchanged;
2. `actuator_only`: the object-prefix posterior multiplied by the actuator
   factor only;
3. `wrench_only`: the object-prefix posterior multiplied by the contact-wrench
   factor only; and
4. `actuator_and_wrench`: both independent factors applied jointly to the same
   object-prefix posterior.

Every arm begins from the same source posterior. The function rejects a source
that already records an independent-sensor update, preventing accidental double
counting. Missing, zero-powered, invalid-only, or component-invariant factors
return the exact input `FactualIntervention` object.

```python
from causal4d.independent_sensor_ablation import (
    build_independent_sensor_ablation,
    save_independent_sensor_ablation_report,
)

result = build_independent_sensor_ablation(
    factual,
    actuator_evidence=actuator,
    predicted_actuator_positions_m=predicted_actuator,
    wrench_evidence=wrench,
    predicted_contact_wrench=predicted_wrench,
    component_metrics={
        "heldout_track_error_mm": per_component_track_error_mm,
        "contact_graph_distance": per_component_contact_distance,
    },
    metric_units={
        "heldout_track_error_mm": "mm",
        "contact_graph_distance": "graph_edges",
    },
    metadata={"diagnostic_only": True},
)

save_independent_sensor_ablation_report(
    "independent-sensor-ablation.json",
    result.report,
)
```

The optional component metrics must be fixed before the posterior arm is
interpreted. They may represent controlled truth losses, source-panel placebo
losses, contact graph distance, intervention-parameter error, or held-out
prediction loss under the applicable information boundary. The report stores
only posterior expectations, MAP-component values, units, and hashes of the
complete component-value arrays.

## Attribution quantities

For every arm the report records:

- posterior identity and exact-source-fallback status;
- effective sample size and active support count;
- complete-component entropy;
- marginal entropy of persistent `phi` and event-specific `kappa`;
- KL divergence from the object-prefix posterior;
- MAP component and maximum probability;
- the exact factor diagnostics retained by the sensor update; and
- optional expected and MAP component metrics.

The attribution section reports entropy reductions for actuator-only,
wrench-only, and combined evidence. It also records the nonadditive factor
interaction

```text
H(object) - H(actuator) - H(wrench) + H(actuator+wrench)
```

for complete support, `phi`, and `kappa`. This value is descriptive: it may be
positive or negative and is not a causal-effect estimator by itself.

For each supplied component loss, the report gives expected improvement relative
to the object-prefix posterior and the incremental gain of the combined arm over
the better single-factor arm.

## Content and evidence binding

The report identity binds:

- the source `FactualIntervention` identity;
- every posterior-arm identity;
- actuator and wrench evidence identities and clock names;
- canonical hashes and shapes of component prediction arrays;
- broadcasted prediction-variance arrays;
- the complete independent-sensor configuration;
- optional component-loss hashes and units; and
- finite diagnostic metadata.

Reports are strict finite JSON, loaded from exact ordinary-file bytes, and
published atomically without overwrite by default. A changed field with a stale
artifact identity is rejected.

## Relationship to posterior scoring

The optional component metrics are deterministic losses attached to finite
posterior components. For distribution-level logarithmic, energy, variogram, or
registered linear-query scores, evaluate each returned arm separately through
`causal4d.posterior_scoring` and bind the resulting score-artifact identities in
the ablation report metadata. This keeps proper-score computation separate from
the evidence-factor attribution and prevents the ablation builder from reading
held-out values implicitly. Because metadata participates in the report content
address, adding or replacing a bound score artifact changes the ablation report
identity.

## Interpretation boundary

An actuator-only contraction of `phi` shows that measured realization is
informative relative to the already-consumed object prefix. A wrench-only
contraction of `kappa` shows that independent contact evidence is informative.
Neither result proves that the corresponding latent variable is physically
identified unless the registered identifiability and held-out prediction gates
also pass.

The diagnostic must not be used to select a method, threshold, exclusion, or
calibration transform from confirmatory target outcomes. For the current
physical protocol it is an attribution artifact only; the registered primary
comparison and exact fallback remain unchanged.
