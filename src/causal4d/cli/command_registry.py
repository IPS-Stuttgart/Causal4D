"""Authoritative command catalog for the single ``causal4d`` executable."""

from __future__ import annotations

import importlib.metadata
from dataclasses import asdict, dataclass
from typing import Any, Literal

Lifecycle = Literal["stable", "diagnostic", "experimental", "public-study", "archive"]
REMOVED_EXECUTABLE_VERSION = "0.5.0"
PRIMARY_EXECUTABLE = "causal4d"
PRIMARY_TARGET = "causal4d.cli.root:main"


@dataclass(frozen=True)
class CommandSpec:
    """One current grouped route and its historical compatibility metadata."""

    route: tuple[str, ...]
    target: str
    summary: str
    lifecycle: Lifecycle
    historical_name: str | None = None
    extras: tuple[str, ...] = ()
    owner: str = "Causal4D"
    claim_bearing: bool = False
    requires: tuple[str, ...] = ()
    removed_in: str | None = None

    def __post_init__(self) -> None:
        if not self.route or any(
            not token or token.startswith("-") for token in self.route
        ):
            raise ValueError("command routes must contain non-option tokens")
        if ":" not in self.target:
            raise ValueError("command targets must use module:function syntax")
        if not self.summary or not self.owner:
            raise ValueError("command summary and owner must be nonempty")
        if self.historical_name is not None:
            if not self.historical_name.startswith("causal4d-"):
                raise ValueError("historical names must start with causal4d-")
            if self.removed_in is None:
                raise ValueError("removed historical names require removed_in")
        elif self.removed_in is not None:
            raise ValueError("removed_in requires a historical name")
        if self.lifecycle == "archive" and self.claim_bearing:
            raise ValueError("archived commands cannot be claim-bearing")

    @property
    def route_name(self) -> str:
        return " ".join(self.route)

    @property
    def invocation(self) -> tuple[str, ...]:
        return (PRIMARY_EXECUTABLE, *self.route)

    @property
    def invocation_text(self) -> str:
        return " ".join(self.invocation)

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["route"] = list(self.route)
        values["route_name"] = self.route_name
        values["invocation"] = list(self.invocation)
        values["invocation_text"] = self.invocation_text
        values["extras"] = list(self.extras)
        values["requires"] = list(self.requires)
        values["historical_executable_installed"] = False
        return values


COMMANDS = (
    CommandSpec(
        route=("benchmark", "counterfactual"),
        target="causal4d.cli.counterfactual_benchmark:main",
        summary="Run the controlled counterfactual benchmark.",
        lifecycle="stable",
        historical_name="causal4d-counterfactual-benchmark",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("benchmark", "latent-contact"),
        target="causal4d.cli.latent_contact_benchmark:main",
        summary="Run the controlled latent-contact benchmark.",
        lifecycle="stable",
        historical_name="causal4d-latent-contact-benchmark",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("benchmark", "dynamic-contact"),
        target="causal4d.cli.dynamic_contact_benchmark:main",
        summary="Run the dynamic contact-path benchmark.",
        lifecycle="experimental",
        historical_name="causal4d-dynamic-contact-benchmark",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "phystwin", "rollout-bank"),
        target="causal4d.cli.phystwin_rollout_bank:main",
        summary="Build or resume a PhysTwin rollout bank.",
        lifecycle="experimental",
        historical_name="causal4d-phystwin-rollout-bank",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("evidence", "bpt-belief", "export"),
        target="causal4d.cli.export_bpt_belief:main",
        summary="Export a BayesianPhysTwin belief into the Causal4D contract.",
        lifecycle="stable",
        historical_name="causal4d-export-bpt-belief",
        extras=("phystwin",),
        claim_bearing=True,
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("evidence", "observation-lineage"),
        target="causal4d.cli.observation_lineage:main",
        summary="Validate or bind observation provenance.",
        lifecycle="stable",
        historical_name="causal4d-observation-lineage",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "phystwin", "abduct-intervention"),
        target="causal4d.cli.abduct_phystwin_intervention:main",
        summary="Abduce a realized intervention from a PhysTwin response prefix.",
        lifecycle="experimental",
        historical_name="causal4d-abduct-phystwin-intervention",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "phystwin", "counterfactual"),
        target="causal4d.cli.counterfactual_phystwin:main",
        summary="Run a PhysTwin counterfactual rollout from an abducted intervention.",
        lifecycle="experimental",
        historical_name="causal4d-counterfactual-phystwin",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("evidence", "physical-target", "import-legacy"),
        target="causal4d.cli.import_physical_target:main",
        summary="Convert a trusted legacy target into the safe target contract.",
        lifecycle="stable",
        extras=("phystwin",),
        claim_bearing=True,
        requires=("BayesianPhysTwin provider", "trusted legacy pickle"),
    ),
    CommandSpec(
        route=("evidence", "physical-counterfactual", "evaluate"),
        target="causal4d.cli.evaluate_physical_counterfactual:main",
        summary="Evaluate registered physical counterfactual predictions.",
        lifecycle="stable",
        historical_name="causal4d-evaluate-physical-counterfactual",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "semantic", "build-task-posterior"),
        target="causal4d.cli.molmo_task_posterior:main",
        summary="Build an exploratory Molmo task posterior.",
        lifecycle="experimental",
        historical_name="causal4d-build-molmo-task-posterior",
        extras=("vision",),
        requires=("MolmoMotion artifacts",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "semantic", "fit-trust"),
        target="causal4d.cli.fit_semantic_trust:main",
        summary="Fit the source-only semantic trust gate.",
        lifecycle="experimental",
        historical_name="causal4d-fit-semantic-trust",
        extras=("vision",),
        requires=("MolmoMotion artifacts",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "semantic", "adaptive-task-posterior"),
        target="causal4d.cli.adaptive_molmo_task_posterior:main",
        summary="Build an adaptive exploratory Molmo task posterior.",
        lifecycle="experimental",
        historical_name="causal4d-adaptive-molmo-task-posterior",
        extras=("vision",),
        requires=("MolmoMotion artifacts",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("archive", "semantic", "forecast-v1"),
        target="causal4d.cli.molmo_phystwin_forecast:main",
        summary="Run the frozen version-1 Molmo/PhysTwin forecast path.",
        lifecycle="archive",
        historical_name="causal4d-molmo-phystwin-forecast",
        extras=("vision", "phystwin"),
        requires=("MolmoMotion artifacts", "BayesianPhysTwin provider"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "semantic", "forecast"),
        target="causal4d.cli.molmo_phystwin_forecast_v2:main",
        summary="Run the current exploratory Molmo/PhysTwin forecast.",
        lifecycle="experimental",
        historical_name="causal4d-molmo-phystwin-forecast-v2",
        extras=("vision", "phystwin"),
        requires=("MolmoMotion artifacts", "BayesianPhysTwin provider"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "semantic", "phystwin-evaluation"),
        target="causal4d.cli.evaluate_phystwin_molmo:main",
        summary="Evaluate exploratory PhysTwin/Molmo forecasts.",
        lifecycle="diagnostic",
        historical_name="causal4d-evaluate-phystwin-molmo",
        extras=("vision", "phystwin"),
        requires=("MolmoMotion artifacts", "BayesianPhysTwin provider"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "semantic", "acceptance"),
        target="causal4d.cli.evaluate_molmo_acceptance:main",
        summary="Evaluate the locked semantic-prior acceptance gate.",
        lifecycle="diagnostic",
        historical_name="causal4d-evaluate-molmo-acceptance",
        extras=("vision",),
        requires=("MolmoMotion artifacts",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "real", "oracle-gap"),
        target="causal4d.cli.audit_real_oracle_gap:main",
        summary="Audit inference, proposal, and model-discrepancy headroom.",
        lifecycle="diagnostic",
        historical_name="causal4d-audit-real-oracle-gap",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "real", "failure-attribution"),
        target="causal4d.cli.real_failure_attribution:main",
        summary="Aggregate execution-accounted real failure diagnostics.",
        lifecycle="diagnostic",
        historical_name="causal4d-aggregate-real-failure-attribution",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "real"),
        target="causal4d.cli.real_protocol:main",
        summary="Scaffold, validate, and inspect the locked real protocol.",
        lifecycle="stable",
        historical_name="causal4d-real-protocol",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "freeze"),
        target="causal4d.cli.real_experiment_freeze:main",
        summary="Seal or validate a confirmatory method freeze.",
        lifecycle="stable",
        historical_name="causal4d-real-experiment-freeze",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "parameter-support"),
        target="causal4d.cli.audit_parameter_support:main",
        summary="Audit physical-parameter support and truncation.",
        lifecycle="diagnostic",
        historical_name="causal4d-audit-parameter-support",
        extras=("phystwin",),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("calibration", "real"),
        target="causal4d.cli.real_calibration:main",
        summary="Fit or evaluate diagnostic real predictive calibration.",
        lifecycle="diagnostic",
        historical_name="causal4d-real-calibration",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("calibration", "execution-block"),
        target="causal4d.cli.execution_block_calibration:main",
        summary="Fit or evaluate registered execution-block calibration.",
        lifecycle="stable",
        historical_name="causal4d-execution-block-calibration",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "discrepancy", "graph-temporal"),
        target="causal4d.cli.evaluate_graph_temporal_discrepancy:main",
        summary="Evaluate graph-temporal discrepancy persistence.",
        lifecycle="diagnostic",
        historical_name="causal4d-evaluate-graph-temporal-discrepancy",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "rest-geometry", "evaluate"),
        target="causal4d.cli.evaluate_rest_geometry:main",
        summary="Evaluate rest-geometry candidates.",
        lifecycle="diagnostic",
        historical_name="causal4d-evaluate-rest-geometry",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "rest-geometry", "protocol"),
        target="causal4d.cli.rest_geometry_protocol:main",
        summary="Run the prospective rest-geometry protocol.",
        lifecycle="experimental",
        historical_name="causal4d-rest-geometry-protocol",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "rest-geometry", "cross-action"),
        target="causal4d.cli.rest_geometry_cross_action:main",
        summary="Evaluate cross-action rest-geometry transfer.",
        lifecycle="diagnostic",
        historical_name="causal4d-rest-geometry-cross-action",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "rest-geometry", "candidate-evidence"),
        target="causal4d.cli.rest_geometry_candidate_evidence:main",
        summary="Build diagnostic rest-geometry candidate evidence.",
        lifecycle="diagnostic",
        historical_name="causal4d-rest-geometry-candidate-evidence",
        extras=("phystwin",),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "rest-geometry", "source-correction"),
        target="causal4d.cli.rest_geometry_source_correction:main",
        summary="Fit a source-only rest-geometry correction.",
        lifecycle="experimental",
        historical_name="causal4d-rest-geometry-source-correction",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "rest-geometry", "transfer"),
        target="causal4d.cli.phystwin_rest_geometry_transfer:main",
        summary="Run a PhysTwin rest-geometry transfer experiment.",
        lifecycle="experimental",
        historical_name="causal4d-phystwin-rest-geometry-transfer",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "rest-geometry", "protocol-result"),
        target="causal4d.cli.rest_geometry_protocol_result:main",
        summary="Summarize a rest-geometry protocol result.",
        lifecycle="diagnostic",
        historical_name="causal4d-rest-geometry-protocol-result",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("experiment", "rest-geometry", "build-canonical-graph"),
        target="causal4d.cli.phystwin_canonical_graph:main",
        summary="Build the canonical PhysTwin graph used by structural studies.",
        lifecycle="experimental",
        historical_name="causal4d-build-phystwin-canonical-graph",
        extras=("phystwin",),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "rest-geometry", "register-graph"),
        target="causal4d.cli.rest_geometry_register_graph:main",
        summary="Register a canonical graph for the physical protocol.",
        lifecycle="stable",
        historical_name="causal4d-rest-geometry-register-graph",
        extras=("phystwin",),
        claim_bearing=True,
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("archive", "preacquisition", "v2"),
        target="causal4d.cli.preacquisition_protocol:main",
        summary="Validate the frozen version-2 pre-acquisition protocol.",
        lifecycle="archive",
        historical_name="causal4d-preacquisition-protocol",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("archive", "preacquisition", "v3"),
        target="causal4d.cli.preacquisition_protocol_v3:main",
        summary="Validate the frozen version-3 pre-acquisition protocol.",
        lifecycle="archive",
        historical_name="causal4d-preacquisition-protocol-v3",
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "preacquisition-v4"),
        target="causal4d.cli.preacquisition_protocol_v4:main",
        summary="Validate the version-4 pre-acquisition amendment.",
        lifecycle="stable",
        historical_name="causal4d-preacquisition-protocol-v4",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "actuator-realization"),
        target="causal4d.cli.actuator_realization:main",
        summary="Calibrate and validate realized actuator motion.",
        lifecycle="stable",
        historical_name="causal4d-calibrate-actuator-realization",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "contact-registration"),
        target="causal4d.cli.contact_registration:main",
        summary="Create or validate registered physical contact geometry.",
        lifecycle="stable",
        historical_name="causal4d-contact-registration",
        claim_bearing=True,
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "mechanism-gate-controls"),
        target="causal4d.cli.mechanism_gate_controls:main",
        summary="Audit prospective physical-mechanism gate controls.",
        lifecycle="diagnostic",
        historical_name="causal4d-audit-mechanism-gate-controls",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "structural"),
        target="causal4d.cli.structural_protocol:main",
        summary="Validate or run the structural physical protocol.",
        lifecycle="stable",
        historical_name="causal4d-structural-protocol",
        extras=("phystwin", "warp"),
        claim_bearing=True,
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "discrepancy", "localize"),
        target="causal4d.cli.phystwin_discrepancy_localization:main",
        summary="Localize PhysTwin model discrepancy.",
        lifecycle="diagnostic",
        historical_name="causal4d-diagnose-phystwin-discrepancy-location",
        extras=("phystwin",),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "discrepancy", "aggregate-localization"),
        target="causal4d.cli.discrepancy_localization_aggregate:main",
        summary="Aggregate PhysTwin discrepancy-localization evidence.",
        lifecycle="diagnostic",
        historical_name="causal4d-aggregate-phystwin-discrepancy-location",
        extras=("phystwin",),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "state", "propagated"),
        target="causal4d.cli.phystwin_propagated_state:main",
        summary="Diagnose propagated PhysTwin state hypotheses.",
        lifecycle="diagnostic",
        historical_name="causal4d-diagnose-phystwin-propagated-state",
        extras=("phystwin", "warp"),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("diagnostic", "state", "aggregate-propagated"),
        target="causal4d.cli.phystwin_propagated_state_aggregate:main",
        summary="Aggregate propagated-state diagnostics.",
        lifecycle="diagnostic",
        historical_name="causal4d-aggregate-phystwin-propagated-state",
        extras=("phystwin",),
        requires=("BayesianPhysTwin provider",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "pokeflex", "preflight"),
        target="causal4d_public.cli.pokeflex_preflight:main",
        summary="Run the source-locked PokeFlex preflight.",
        lifecycle="public-study",
        historical_name="causal4d-pokeflex-preflight",
        owner="Causal4D public studies",
        requires=("PokeFlex dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "pokeflex", "fixture"),
        target="causal4d_public.cli.pokeflex_fixture:main",
        summary="Build a PokeFlex protocol fixture.",
        lifecycle="public-study",
        historical_name="causal4d-pokeflex-fixture",
        owner="Causal4D public studies",
        requires=("PokeFlex dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "pokeflex", "source-qa"),
        target="causal4d_public.cli.pokeflex_source_qa:main",
        summary="Run PokeFlex source-only quality assurance.",
        lifecycle="public-study",
        historical_name="causal4d-pokeflex-source-qa",
        owner="Causal4D public studies",
        requires=("PokeFlex dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "pokeflex", "warp-source"),
        target="causal4d_public.cli.pokeflex_warp_source:main",
        summary="Run the source-only PokeFlex Warp study.",
        lifecycle="public-study",
        historical_name="causal4d-pokeflex-warp-source",
        extras=("warp",),
        owner="Causal4D public studies",
        requires=("PokeFlex dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "preflight"),
        target="causal4d_public.cli.deform360_preflight:main",
        summary="Run the source-locked Deform360 preflight.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-preflight",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "contact"),
        target="causal4d_public.cli.deform360_contact:main",
        summary="Fit, seal, or evaluate Deform360 contact evidence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-contact",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "sam2-masks"),
        target="causal4d_public.cli.deform360_sam2_masks:main",
        summary="Build source-only Deform360 SAM2 masks.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-sam2-masks",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset", "SAM2 checkpoint"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "sam2-views"),
        target="causal4d_public.cli.deform360_sam2_views:main",
        summary="Build source-only Deform360 SAM2 view evidence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-sam2-views",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset", "SAM2 checkpoint"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "sam2-prefix"),
        target="causal4d_public.cli.deform360_sam2_prefix:main",
        summary="Build causal-prefix Deform360 SAM2 evidence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-sam2-prefix",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset", "SAM2 checkpoint"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "sam2-suffix"),
        target="causal4d_public.cli.deform360_sam2_suffix:main",
        summary="Evaluate held-out Deform360 SAM2 suffix evidence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-sam2-suffix",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset", "SAM2 checkpoint"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "splat-probe"),
        target="causal4d_public.cli.deform360_splat_probe:main",
        summary="Run the Deform360 splat feasibility probe.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-splat-probe",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-sequence"),
        target="causal4d_public.cli.deform360_rope_sequence:main",
        summary="Prepare the Deform360 rope sequence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-sequence",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-observation"),
        target="causal4d_public.cli.deform360_rope_observation:main",
        summary="Build Deform360 rope observation evidence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-observation",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-fit"),
        target="causal4d_public.cli.deform360_rope_fit:main",
        summary="Fit the source-only Deform360 rope model.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-fit",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-prefix"),
        target="causal4d_public.cli.deform360_rope_prefix:main",
        summary="Build the causal Deform360 rope prefix.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-prefix",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-predict"),
        target="causal4d_public.cli.deform360_rope_predict:main",
        summary="Predict the Deform360 rope future.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-predict",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-future"),
        target="causal4d_public.cli.deform360_rope_future:main",
        summary="Prepare held-out Deform360 rope future evidence.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-future",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-oracle"),
        target="causal4d_public.cli.deform360_rope_oracle:main",
        summary="Run the diagnostic Deform360 rope oracle.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-oracle",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "rope-evaluate"),
        target="causal4d_public.cli.deform360_rope_evaluate:main",
        summary="Evaluate Deform360 rope predictions.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-rope-evaluate",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "replication"),
        target="causal4d_public.cli.deform360_replication:main",
        summary="Run the source-locked Deform360 replication.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-replication",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "phystwin-feasibility"),
        target="causal4d_public.cli.deform360_phystwin_feasibility:main",
        summary="Audit Deform360-to-PhysTwin feasibility.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-phystwin-feasibility",
        extras=("vision", "phystwin"),
        owner="Causal4D public studies",
        requires=("Deform360 dataset", "BayesianPhysTwin provider"),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "pooling-control"),
        target="causal4d_public.cli.deform360_pooling_control:main",
        summary="Run the Deform360 pooling control.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-pooling-control",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("public", "deform360", "source-qa"),
        target="causal4d_public.cli.deform360_replication_source_qa:main",
        summary="Run Deform360 source-only quality assurance.",
        lifecycle="public-study",
        historical_name="causal4d-deform360-source-qa",
        extras=("vision",),
        owner="Causal4D public studies",
        requires=("Deform360 dataset",),
        removed_in=REMOVED_EXECUTABLE_VERSION,
    ),
    CommandSpec(
        route=("protocol", "readiness"),
        target="causal4d.cli.preacquisition_readiness:main",
        summary="Verify evidence-bound readiness before confirmatory collection.",
        lifecycle="stable",
        claim_bearing=True,
    ),
    CommandSpec(
        route=("protocol", "acquisition"),
        target="causal4d.cli.acquisition_operations:main",
        summary="Run the acquisition doctor, watchdog, and append-only journal.",
        lifecycle="stable",
    ),
    CommandSpec(
        route=("evidence", "real-report-shell"),
        target="causal4d.registered_real_report_shell:main",
        summary=(
            "Render or validate the target-free registered real-analysis report shell."
        ),
        lifecycle="stable",
    ),
    CommandSpec(
        route=("paper", "reproduce"),
        target="causal4d.paper_reproduction:main",
        summary="Build or verify an immutable reviewer-facing paper bundle.",
        lifecycle="stable",
    ),
    CommandSpec(
        route=("diagnostic", "uncertainty", "decompose-query"),
        target="causal4d.cli.query_variance_decomposition:main",
        summary="Attribute a fixed query covariance to declared sources.",
        lifecycle="diagnostic",
    ),
    CommandSpec(
        route=("evidence", "interpret-real-result"),
        target="causal4d.cli.real_result_interpretation:main",
        summary="Apply the preregistered real-result interpretation tree.",
        lifecycle="stable",
        claim_bearing=True,
    ),
)


def grouped_commands() -> tuple[CommandSpec, ...]:
    """Return every supported route after validating catalog invariants."""

    routes = [command.route for command in COMMANDS]
    historical = [
        command.historical_name
        for command in COMMANDS
        if command.historical_name is not None
    ]
    if len(routes) != len(set(routes)):
        raise RuntimeError("grouped command routes are not unique")
    if len(historical) != len(set(historical)):
        raise RuntimeError("historical executable names are not unique")
    if len(historical) != 67:
        raise RuntimeError("historical executable inventory is incomplete")
    return tuple(sorted(COMMANDS, key=lambda command: command.route))


def historical_commands() -> tuple[CommandSpec, ...]:
    """Return current routes that replace removed historical executables."""

    return tuple(
        command for command in grouped_commands() if command.historical_name is not None
    )


def command_inventory(*, removed_only: bool = False) -> tuple[CommandSpec, ...]:
    commands = historical_commands() if removed_only else grouped_commands()
    return tuple(commands)


def find_command(name: str) -> CommandSpec:
    """Resolve a grouped route, slash route, or removed executable name."""

    normalized = " ".join(name.replace("/", " ").split())
    for command in grouped_commands():
        if normalized in {command.route_name, command.historical_name}:
            return command
    raise KeyError(name)


def _installed_console_scripts() -> dict[str, str] | None:
    try:
        distribution = importlib.metadata.distribution("causal4d")
    except importlib.metadata.PackageNotFoundError:
        return None
    scripts: dict[str, str] = {}
    duplicates: list[str] = []
    for entry_point in distribution.entry_points:
        if entry_point.group != "console_scripts":
            continue
        if entry_point.name in scripts:
            duplicates.append(entry_point.name)
        scripts[entry_point.name] = entry_point.value
    if duplicates:
        raise RuntimeError(
            "installed console script names are duplicated: "
            + ", ".join(sorted(set(duplicates)))
        )
    return scripts


def validate_runtime_command_inventory(
    *,
    require_installed: bool = False,
) -> dict[str, Any]:
    """Require one installed executable and no historical wrappers."""

    commands = grouped_commands()
    scripts = _installed_console_scripts()
    installed = scripts is not None
    relevant = (
        {}
        if scripts is None
        else {
            name: target
            for name, target in scripts.items()
            if name == PRIMARY_EXECUTABLE or name.startswith("causal4d-")
        }
    )
    expected = {PRIMARY_EXECUTABLE: PRIMARY_TARGET}
    missing = sorted(set(expected) - set(relevant))
    unexpected = sorted(set(relevant) - set(expected))
    mismatched = sorted(
        name
        for name in set(expected) & set(relevant)
        if expected[name] != relevant[name]
    )
    removed_installed = sorted(
        name for name in relevant if name.startswith("causal4d-")
    )
    valid = (
        not missing
        and not unexpected
        and not mismatched
        and not removed_installed
        and (installed or not require_installed)
    )
    return {
        "schema_version": 2,
        "artifact_kind": "Causal4DCommandInventoryValidation",
        "valid": valid,
        "installed_distribution_present": installed,
        "require_installed": require_installed,
        "primary_executable": PRIMARY_EXECUTABLE,
        "primary_target": PRIMARY_TARGET,
        "grouped_route_count": len(commands),
        "historical_executable_count": len(historical_commands()),
        "missing_console_scripts": missing,
        "unexpected_console_scripts": unexpected,
        "target_mismatches": mismatched,
        "removed_historical_executables_installed": removed_installed,
    }


__all__ = [
    "COMMANDS",
    "CommandSpec",
    "Lifecycle",
    "PRIMARY_EXECUTABLE",
    "PRIMARY_TARGET",
    "REMOVED_EXECUTABLE_VERSION",
    "command_inventory",
    "find_command",
    "grouped_commands",
    "historical_commands",
    "validate_runtime_command_inventory",
]
