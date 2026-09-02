# Proof-carrying sensor-reveal replay

This module is the benchmark boundary between synchronized public sensor data
and Causal4D's exact sequential `act / probe / fallback` planner.

The central question is not whether a model can reconstruct every hidden state.
It is whether a model can request only the observations needed for a declared
physical decision, seal the complete acquisition path before scoring, and carry
an independently checkable explanation of why no other sensor value entered the
trace.

## Why synchronized public data are sufficient

A logged manipulation dataset generally cannot reveal the outcome of a robot
action that was not executed. It can, however, contain many camera and tactile
streams that observed the same executed physical trajectory. Such streams can be
hidden and disclosed one at a time without fabricating an unexecuted physical
action.

This is an **offline selective-observation replay**. It evaluates information
acquisition, not the physical energy, latency, or perturbation caused by moving a
camera or touching the object with a new sensor. Sensor costs and risks are
registered benchmark quantities and must not be presented as measured robot
costs unless a dataset supplies that evidence.

## Information separation

The implementation uses six distinct records.

| Record | Owner | Contains |
| --- | --- | --- |
| `SensorRevealManifest` | challenge | public context identity, action and sensor rosters, outcome vocabularies, costs, risks, and a commitment to hidden truth |
| `SensorRevealTruth` | challenge only | one realized outcome and payload digest per sensor, adapter identities, realized terminal-action losses, and a nonce |
| `SensorRevealSubmission` | provider | finite physical hypotheses, weights, action losses, sensor likelihood models, and planner settings |
| `SensorRevealPlan` | provider/evaluator | the complete content-addressed sequential policy tree, frozen before optional sensor disclosure |
| `SensorRevealTrace` | evaluator | only the sensor outcomes actually requested along the realized policy path and the terminal action or exact fallback |
| `SensorRevealScore` | challenge | realized terminal loss and regret, opened only after the complete trace is sealed and verified |

The public manifest commits to every hidden sensor outcome and terminal loss. A
later change to an outcome, payload digest, adapter identity, loss, or nonce
invalidates that commitment.

## Requested-channel-only execution

`execute_sensor_reveal_plan` starts from the frozen policy root. At each probe
node it:

1. reads the index of the requested sensor;
2. discloses only that sensor's finite outcome, raw-payload SHA-256, and adapter
   identity;
3. follows exactly the matching policy branch;
4. prevents repeated disclosure of the same sensor; and
5. stops at a certified action or caller-owned fallback.

No unrequested payload digest or realized action loss is serialized in the
trace. The trace is content-addressed and marked `sealed_before_scoring=true`
before `score_sensor_reveal_trace` may read the terminal losses.

## Independent verification

`causal4d.sensor_reveal_verifier` uses only the Python standard library. It does
not import the producer, the challenge implementation, or the sequential
planner. It independently checks:

- exact manifest, plan, and trace schemas and content identities;
- internal consistency of every carried terminal certificate;
- policy branch probabilities and horizon reduction;
- recursive expected cost, worst-case cost, and risk accounting;
- absence of repeated sensors along every policy path;
- requested-sensor-only trace disclosure;
- outcome-to-branch and policy-node identity consistency;
- exact terminal action identity; and
- exact use of the caller-owned fallback whenever the policy terminates in
  fallback.

The verifier does not recompute the policy from the provider's physical model.
It therefore verifies the carried acquisition proof and fail-closed execution,
not provider competence or policy optimality.

## Controlled strict-separation result

The deterministic mechanism study contains 32 complete hypotheses, three
terminal actions, and five optional sensors. One cheap camera reveals a routing
variable. Depending on its outcome, one of two tactile sensors reveals the task.
A global task camera resolves the decision in one step, while a four-way nuisance
sensor has the largest generic mutual information.

The checked result is:

| Policy | First sensor | Mean sensor cost | Mean fraction revealed | Terminal accuracy | Complete state identified |
| --- | --- | ---: | ---: | ---: | ---: |
| generic mutual information | nuisance four-way | -- | -- | -- | -- |
| one-step decision value | global task camera | 0.50 | 20% | 100% | 0% |
| minimum fixed decision-sufficient set | global task camera | 0.50 | 20% | 100% | 0% |
| **exact sequential decision acquisition** | route camera, then branch-local tactile | **0.35** | **40%** | **100%** | **0%** |

Every terminal leaf retains eight compatible complete hypotheses. The policy
therefore identifies the decision without identifying the complete state. All
32 traces verify independently, and no trace contains a digest from an
unrequested sensor.

A noisy full-support sensor is included as a fail-closed control. Because every
outcome leaves opposing terminal actions possible, the planner discloses no
sensor and returns the exact fallback.

Run the deterministic capsule with:

```bash
PYTHONPATH=src python scripts/experiments/proof_carrying_sensor_reveal.py \
  --output build/proof-carrying-sensor-reveal/result.json
```

## Public-data adapter contract

A real-data adapter may map a high-dimensional synchronized stream to one finite
outcome only when all of the following are frozen before held scoring:

1. the raw stream identity, timestamps, and payload digest;
2. the causal prefix available to the adapter;
3. preprocessing and finite outcome codebook;
4. source-fitted likelihood rows for every physical hypothesis;
5. sensor cost and any prospective risk charge;
6. the terminal query, action set, and caller-owned fallback; and
7. the independent object or complete-trajectory grouping used for inference.

The adapter must never receive future target geometry, future target tactile
values, terminal loss, or an unrequested sensor payload. A challenge may retain
all synchronized streams internally, but the submission and trace must expose
only the registered initial context and requested reveal path.

## Deform360 continuation

The existing public Deform360 audit already defines a concrete starting point:
`001-rope` has ten episodes, 41 raw camera streams, four tactile streams, a
metadata-only episode split, synchronized processed streams, robot state, and
metric-geometry products. That cohort is suitable for source-side adapter and
interface development.

The next reviewed stage should:

1. inventory source episodes without decoding a protected target;
2. define fixed initial views and optional camera/tactile sensor groups;
3. fit finite sensor-outcome adapters and likelihoods on source episodes only;
4. calibrate sensor-message and trajectory-regret errors on disjoint complete
   episodes;
5. compare sequential decision value with all sensors, fixed subsets, random
   cost-matched acquisition, generic information, query variance, and one-step
   decision value; and
6. freeze a complete plan/submission before any held sensor outcome or terminal
   physical query is opened.

The final claim-bearing study should be object-disjoint. The already-developed
`001-rope` episode is not sufficient by itself for arbitrary-object or fresh
confirmation claims.

## Claim boundary

The current result is deterministic finite-interface mechanism evidence. The
trace verifier establishes content integrity, requested-only disclosure,
policy-path consistency, cost/risk arithmetic, and exact fallback. It does not
validate a camera or tactile adapter, learned provider, physical hypothesis
support, exchangeability assumption, target transport, online sensing,
deployment authorization, or safety.
