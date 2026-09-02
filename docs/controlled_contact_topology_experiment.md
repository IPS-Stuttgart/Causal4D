# Prospective controlled contact-topology experiment

## Scientific question

Can a physical twin choose a state-changing diagnostic sequence that certifies a
consequential terminal manipulation at lower interaction cost than generic
information acquisition, one-step decision value, and fixed diagnostics?

The experiment is designed to test controlled decision identification rather
than trajectory prediction alone. It must demonstrate all three policy modes:

```text
act when the terminal decision is already certified
intervene when a branch-complete certifying sequence exists
fallback when no registered safe sequence can certify the decision
```

## Physical counterfactual fork

A deformable linear object is routed around a partially occluded post. Complete
physical states vary in:

- left-side versus right-side routing;
- over versus under topology;
- sticking, slipping, or detached contact regime;
- low versus high friction;
- low versus high preload; and
- material stiffness.

The initial observation is deliberately chosen so that several states remain
compatible. Terminal actions are `pull-left`, `pull-right`, `release`, and the
caller-owned `hold/reset` fallback. The realized terminal loss combines task
failure, excess peak force, entanglement growth, and normalized completion time.
All weights and normalizations must be frozen from source trials.

## Registered diagnostic interventions

The initial roster should contain at least:

1. a small lateral tug;
2. a small vertical lift;
3. a low-amplitude oscillation; and
4. a direct but more expensive global inspection or regrasp.

Each intervention changes the rope configuration and emits synchronized visual,
kinematic, and force observations. It must therefore be represented by a joint
transition-observation kernel

```text
K_e[h, h_next, y] = P(H_next=h_next, Y=y | H=h, e),
```

not by a static likelihood. Probe cost should include elapsed time and work;
registered risk should include peak-force and workspace-limit charges. The
terminal action itself is not counted as a diagnostic.

A successful routing mechanism has the following structure: a cheap first
intervention reveals which branch-specific second diagnostic is informative,
while having little or no immediate terminal-regret reduction by itself.

## Three-stage evidence custody

### Stage A: source development

Use source trials only to define:

- the complete finite physical-state roster or certified continuous cells;
- transition-observation kernel estimation;
- terminal action losses;
- intervention costs and risk charges;
- the regret tolerance;
- sensor preprocessing and discretization;
- maximum policy horizon and search budget; and
- every baseline and metric.

All source failures remain visible. Source trials may be repeated for engineering
qualification but may not be promoted to confirmation evidence.

### Stage B: calibration

Use complete calibration episodes as the statistical units. Calibrate only
predeclared uncertainty or support-mismatch envelopes. Multiple frames, nodes,
contacts, actions, or horizons within one episode remain nested measurements and
must not be counted as independent calibration units.

The complete policy, thresholds, transition model, action losses, and stop rules
are sealed before confirmation begins.

### Stage C: one-shot confirmation

Randomize hidden topology, material, friction, preload, and initial appearance.
For every confirmation episode:

1. reveal only the registered initial observation;
2. compute and seal the first policy node;
3. execute exactly the selected diagnostic or terminal action;
4. reveal only the resulting registered observation;
5. continue the sealed policy tree without refitting; and
6. open terminal outcomes only after all episode actions and predictions are
   immutable.

No confirmation episode may be replaced because of an unfavorable physical or
numerical result. Technical failures must be classified prospectively and
reported separately.

## Baselines

Every baseline receives the same initial information, diagnostic roster, action
roster, and physical budgets.

- caller-owned exact fallback only;
- deterministic point-estimate physical twin;
- Bayesian twin without support-wise decision certification;
- generic mutual-information acquisition;
- one-step decision-value acquisition;
- cheapest source-frozen fixed diagnostic sequence;
- static observation-only planner that ignores diagnostic state transitions;
- full controlled decision-identifying policy;
- outcome-aware oracle, diagnostic only.

A strong controlled result requires the static planner to fail or overpay on
states where diagnostics alter contact topology. Merely matching its choices is
not evidence for the controlled extension.

## Primary endpoints

The primary endpoint is complete-episode terminal regret relative to the best
registered terminal action under the realized physical state. Report jointly:

- mean and median terminal regret;
- task-success rate;
- harmful terminal-action count;
- exact-fallback frequency and identity violations;
- cumulative diagnostic work and elapsed time;
- peak diagnostic force;
- number of interventions;
- decision-certification coverage; and
- worst-topology and worst-material results.

The primary comparison is the paired episode-level difference between the full
controlled policy and one-step decision value at matched registered risk. A
secondary comparison evaluates controlled versus static observation-only
planning at matched diagnostic cost.

Trajectory RMSE and state-classification accuracy are secondary mechanism
metrics. They cannot substitute for a terminal decision result.

## Predeclared success criteria

The controlled claim is promoted only if all of the following hold on the
confirmation cohort:

1. zero exact-fallback identity violations;
2. lower paired terminal regret than one-step decision value;
3. lower mean diagnostic cost than the cheapest fixed sufficient sequence;
4. no increase in the registered harmful-action rate relative to exact fallback
   beyond the frozen tolerance;
5. a nonzero subset where the static planner and controlled planner differ due
   to state transitions; and
6. favorable controlled-versus-static terminal regret on that subset.

If criterion 5 fails, the dataset does not exercise the controlled contribution.
If criterion 6 fails, the controlled model is falsified for the registered
interface. Neither case authorizes retuning on confirmation outcomes.

## Sample-size discipline

Complete physical episodes are the independent units. With zero observed harmful
terminal actions, at least 29 independent confirmation episodes are required for
a one-sided 95% Clopper--Pearson upper bound below 10%; 40--60 episodes are
preferable. Repeated sensor frames or policy branches do not increase this count.

The study should be stratified across topology and material, with every stratum
represented in confirmation. Report both pooled paired results and the complete
per-stratum table.

## Public-data pilot and robot confirmation

Logged PokeFlex or Deform360 interactions may be used for source-side kernel
qualification, state-transition diagnostics, and offline policy-value studies.
They generally do not contain all alternative branch actions from the same
microscopic state, so they cannot alone establish the online controlled claim.

The definitive experiment requires prospective robot execution. A modest
single-arm setup with one rope, one post, force sensing, and two calibrated
cameras is sufficient if the randomized hidden-contact fork and confirmation
custody are respected.

## Claim boundary

A positive result would establish lower-cost decision certification for the
registered rope/post task, state roster, diagnostic interventions, terminal
actions, and confirmation distribution. It would not establish universal
physical-state identification, arbitrary-object transfer, unrestricted
continuous-action optimality, deployment safety, or a guarantee outside the
registered transition-observation support.
