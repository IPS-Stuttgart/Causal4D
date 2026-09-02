# PokeFlex task-conditioned drop-query protocol v1

**Status:** registered before any new drop-outcome access; activation is
conditional on the metadata and exact historical-exposure gates.

## Scientific question

Can one logged physical poke be selected for the information it carries about a
specific, different physical query—rather than for generic latent information—and
then improve a sealed dropping-response prediction?

This is the public-real-data precursor to active decision-identifiable physical
twins. It evaluates an offline `act / probe / fallback` composition; it does not
claim that the selected poke was executed online.

## Evidence class

The broader project has already exposed almost all historical poking outcomes.
The diagnostic-poke panel is therefore explicitly retrospective. The primary
drop challenge is called prospective only if a separate exact-take exposure scan
finds no prior drop-outcome access in the relevant repositories, workflow
artifacts, or retained runtime paths. Otherwise the complete study is labeled
retrospective mechanism evidence.

This distinction is immutable in the registered protocol. A secondary
all-object cross-fit cannot upgrade the freshness of the six-object primary
panel.

## Object split

Eligible objects are grouped without outcome access:

- `printed`: object ID starts with `3dPrinted`;
- `foam`: object ID contains `Foam`, or equals `Sponge` or
  `ToiletPaperRoll`;
- `soft`: every remaining registered object.

Within each family, object IDs are ordered by SHA-256 under the frozen salt
`PokeFlex-active-drop-target-v1-2026-09-02`. The first two are primary targets,
the next independently hashed object is calibration, and all remaining objects
are source objects. If all 18 expected objects pass metadata eligibility, the
split is:

| Role | Printed | Foam | Soft |
|---|---|---|---|
| Target | `3dPrintedBunny`, `3dPrintedCylinder` | `Sponge`, `MemoryFoam` | `Beanbag`, `Pillow` |
| Calibration | `3dPrintedPizza` | `FoamCylinder` | `PlushOctopus` |
| Source | `3dPrintedHeart`, `3dPrintedPyramid` | `FoamDice`, `FoamHalfSphere`, `ToiletPaperRoll` | `PlushDice`, `PlushMoon`, `PlushTurtle`, `PlushVolleyball` |

No target may be replaced after outcome access. Insufficient metadata support
stops the primary protocol instead of selecting a more favorable object.

## Target roles and reveal order

Each target object receives exactly four candidate diagnostic pokes, selected by
a second salted hash from complete poking takes. All complete drop takes of that
object are held challenges, with at least two required.

The custody order is:

1. metadata-only fold audit;
2. exact historical exposure scan for every target drop take;
3. source model fitting;
4. calibration-only model and threshold selection;
5. target action-descriptor slicing and probe selection;
6. reveal of only the selected probe response;
7. one joint seal over every target drop-query prediction; and
8. a single target drop-outcome opening and scoring stage.

Unselected target probe responses remain unavailable to the experiment.

## Co-located action and response carrier

PokeFlex stores measured tool pose and force in the same `robot_data.json`
carrier. The selector may use only `frame` and `T_WT`. An isolated semantic
slicer exports the registered path displacement, direction, speed profile, and
tool location relative to the initial mesh. Force, torque, contact response, and
future geometry are not exported.

This is intentionally described as semantic slicing, not byte-level
non-access: the isolation process must parse a co-located carrier to remove
forbidden fields. The selected probe response becomes available only after the
probe identity has been sealed.

## Registered queries

Two geometry-invariant drop queries avoid relying on persistent material vertex
identity:

1. **Drop impact geometry:** maximum template-diagonal-normalized bounding-box
   compression, time to maximum compression, and rebound ratio.
2. **Drop settled geometry:** final normalized symmetric Chamfer distance to the
   initial mesh, final bounding-box compression, and final centroid drift.

The primary loss is source-standardized squared error. Frames, vertices, drop
repetitions, and query coordinates are nested observations; the physical object
is the statistical unit.

## Policies

The frozen comparison set is:

1. no probe;
2. deterministic random-safe probe;
3. one source-fixed safe rule;
4. generic latent-information probe;
5. task-conditioned query-value probe;
6. a dependence-destroyed task-value control; and
7. a target-outcome oracle used only as a diagnostic ceiling.

Risk uses source-predicted peak force and force impulse. Cost uses path length,
duration, and predicted impulse. If no safe probe has positive task value, the
system returns the exact no-probe fallback.

## Source gate

Target probe selection and drop access are forbidden unless calibration shows,
for both queries, positive task-conditioned value and lower mean loss than no
probe. The task-conditioned policy must be no worse than the source-fixed and
generic-information policies. Destroying the object-matched probe--query
dependence must remove at least half of the task-conditioned advantage, and the
selector must exhibit nontrivial probe diversity. Unsafe probes and zero-value
cases must fail closed.

A failed gate terminates this method version before target access.

## Primary analysis

The target comparison uses paired physical-object losses. The frozen bootstrap
uses 100,000 object resamples at seed 20260902. Primary contrasts are
 task-conditioned probing versus no probe, source-fixed probing, and generic
information. The analysis also reports whether the two registered queries choose
different probes and whether the benefit collapses under dependence destruction.

## Claim boundary

A positive result supports:

> Offline task-conditioned selection of one logged real poke improves a sealed
> cross-intervention drop query relative to passive, fixed, and generic-
> information baselines.

It does not establish online execution, same-state counterfactual outcomes,
unique material identification, calibrated full-state uncertainty, deployment
safety, or closed-loop manipulation success.

The machine-readable owner is
`configs/causal4d_public/pokeflex_task_conditioned_drop_protocol_v1.json`, whose
canonical SHA-256 is
`4788fcc012c7809266d041adda50ac5333f5f21f0b828a9d088cca6253df49b9`.
