# Active-causal belief-adaptivity falsification

Status: **controlled negative boundary; physical protocol unchanged**.

The all-action spring-graph experiment established a positive controlled chain:

```text
safe diagnostic intervention
        -> lower causal-hypothesis entropy
        -> better subsequent held-out-action forecast
```

Its selected action was nevertheless almost constant within each topology. This
follow-up asks a narrower and more demanding question:

> Does the episode-specific causal belief materially improve action choice over a
> topology-conditioned safe action selected once from source information?

The answer under the frozen controlled protocol is **no**. The correct supported
wording is therefore *topology-conditioned risk-aware experiment design*, not a
general episode-specific belief-adaptive planner.

## Frozen controls

The base simulator, four contact hypotheses, candidate actions, safety event,
source/target splits, tuning grid, downstream `diagonal_hook` challenge, and 288
held-out episodes are unchanged. Three controls were declared before execution.

### Source-prior fixed-safe action

For each excluded topology, one action is selected using the source-frozen prior,
the target topology's nominal rollout bank, and the source-tuned risk threshold and
action-cost weight. That action is reused for all 96 held-out episodes:

| Held-out topology | Fixed-safe action |
| --- | --- |
| cloth | `reverse_sweep` |
| rope | `centre_pulse` |
| soft block | `right_drag` |

This comparator has the same topology and source information as the proposed
method but does not use the episode's screening posterior for action selection.
Posterior updating after the action still starts from each episode's own belief.

### Within-topology belief shuffle

Each held-out episode receives another episode's screening posterior through a
deterministic cyclic derangement. The shuffled belief is used **only** to select
the action. The actual episode belief and actual selected-action outcome are then
used for posterior updating and forecasting. This isolates sensitivity of the
policy decision from sensitivity of inference.

### Belief-switch panel

For an identical topology and predictive model, all six ambiguous pairs of the
four contact hypotheses are evaluated. The paired hypotheses receive probability
0.49 each and the two off-pair hypotheses receive 0.01 each. This checks whether
the policy is capable of switching actions when supplied deliberately different
belief states, even if the naturally generated held-out beliefs do not trigger
such switching.

## Locked result

| Policy | Entropy reduction | Challenge RMSE | Safety violations |
| --- | ---: | ---: | ---: |
| Source-prior fixed-safe | **0.373806 nats** | **2.594191 mm** | 0/288 |
| Episode-specific risk-constrained information gain | 0.369364 nats | 2.606706 mm | 0/288 |
| Shuffled-belief risk-constrained | 0.369354 nats | 2.602016 mm | 0/288 |

The episode-specific policy minus the fixed-safe comparator changed entropy
reduction by `-0.004442 nats`, with paired 95% interval
`[-0.009616, -0.000659]`. Thus the fixed-safe comparator was slightly better on
this endpoint.

The corresponding RMSE difference was `+0.012514 mm`, with paired 95% interval
`[+0.000093, +0.028839] mm`; positive values favor the fixed-safe comparator.
The NLL interval included zero.

Belief shuffling changed the selected action in only `10/288 = 3.47%` of episodes.
The proposed-minus-shuffled entropy difference was `0.000010 nats`, with paired
95% interval `[-0.006555, 0.005908]`.

Natural held-out action diversity was:

| Topology | Distinct actions selected |
| --- | ---: |
| cloth | 1 |
| rope | 2 |
| soft block | 1 |

The deliberately constructed belief-switch panel selected two actions on each
topology. The implementation can therefore react to belief, but the natural
held-out belief variation does not provide evidence that this adaptivity improves
the result.

## Frozen gate outcome

Passed:

- the belief shuffle changed at least one action;
- the belief-switch panel selected multiple actions on every topology;
- the proposed policy retained zero simulated safety violations;
- the proposed policy had no more violations than the fixed-safe comparator.

Failed:

- more than one naturally selected action on every topology;
- positive entropy advantage over fixed-safe;
- downstream RMSE or NLL advantage over fixed-safe;
- entropy degradation under belief shuffling with a 95% interval excluding zero.

No failed gate was repaired, reweighted, or retuned after outcome access.

## Scientific interpretation

The positive controlled result remains useful: source calibration and a safety
constraint identify diagnostic actions that reduce causal uncertainty and improve
a later forecast relative to passivity. The new controls show that this gain is
explained adequately by a topology-conditioned action rule under the present
hypothesis set and action library.

The controlled paper claim should therefore be:

> Source-calibrated, topology-conditioned safe experiment design resolves contact
> ambiguity and improves a separate subsequent forecast in the controlled
> spring-graph setting.

It should not be:

> An episode-specific belief-adaptive planner has been demonstrated.

A future claim of episode-specific adaptation needs either a richer action library,
a richer range of naturally occurring ambiguous beliefs, or a physical setting in
which different pre-action beliefs prospectively select different safe actions and
outperform a source-frozen fixed-action comparator. That is a new protocol, not a
repair to this result.

## Run

Install the repository, then execute:

```bash
python -m pip install -e .

python scripts/experiments/active_causal_belief_adaptivity.py \
  --output-dir build/active-causal-belief-adaptivity

python scripts/ci/verify_active_causal_belief_adaptivity.py \
  build/active-causal-belief-adaptivity \
  --output-json build/active-causal-belief-adaptivity/verification.json
```

The full experiment was independently replayed with byte-identical JSON, CSV,
and manifest outputs.

## Physical boundary

This result is simulated controlled evidence. It does not increment the registered
physical evidence count and does not alter the frozen 18-session, 36-execution
Causal4D protocol. The next claim-changing milestone remains the registered physical
experiment and its independent-execution calibration.
