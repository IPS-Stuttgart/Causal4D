"""Deterministic CPU-only abduction-intervention-prediction demonstration."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.artifacts.v1 import (
    CounterfactualQuery,
    TwinBelief,
    build_causal_context,
    load_contract,
    save_contract,
)
from causal4d.inference.v1 import (
    FactualAbductionConfig,
    abduct_factual_intervention,
    apply_counterfactual_operator,
    project_physical_posterior,
)
from causal4d.rollout_bank import JointRolloutBank


SUMMARY_SCHEMA_NAME = "causal4d.aip-demo-summary"
SUMMARY_SCHEMA_VERSION = 1


def _contact_metadata(
    *,
    action_id: str,
    future_action_observed: bool,
    gain: float,
    delay_steps: int,
    rotation_degrees: float,
    attachment_shift: int,
    slip_fraction: float,
) -> dict[str, Any]:
    return {
        "action": {
            "proposal_id": action_id,
            "future_action_observed": future_action_observed,
        },
        "contact": {
            "gain_multiplier": gain,
            "delay_steps": delay_steps,
            "rotation_degrees": rotation_degrees,
            "attachment_shifts": [attachment_shift],
            "slip_fraction": slip_fraction,
        },
    }


def _rollout_bank(
    *,
    action_id: str,
    future_action_observed: bool,
    axis: int,
) -> JointRolloutBank:
    parameter_particles = np.asarray([[0.9], [1.1]], dtype=float)
    parameter_weights = np.asarray([0.55, 0.45], dtype=float)
    endpoint = np.asarray(
        [
            [[0.00, 0.000, 0.00], [0.10, 0.000, 0.00]],
            [[0.00, 0.002, 0.00], [0.10, 0.002, 0.00]],
        ],
        dtype=float,
    )
    trajectories = np.empty((2, 2, 4, 2, 3), dtype=np.float32)
    for hypothesis in range(2):
        for particle in range(2):
            trajectories[hypothesis, particle, 0] = endpoint[particle]
            speed = 0.008 + 0.004 * hypothesis + 0.001 * particle
            for frame in range(1, 4):
                displacement = np.zeros(3, dtype=float)
                displacement[axis] = frame * speed
                trajectories[hypothesis, particle, frame] = (
                    endpoint[particle] + displacement
                )
    metadata = (
        _contact_metadata(
            action_id=action_id,
            future_action_observed=future_action_observed,
            gain=1.0,
            delay_steps=0,
            rotation_degrees=0.0,
            attachment_shift=0,
            slip_fraction=0.0,
        ),
        _contact_metadata(
            action_id=action_id,
            future_action_observed=future_action_observed,
            gain=1.2,
            delay_steps=1,
            rotation_degrees=5.0,
            attachment_shift=1,
            slip_fraction=0.1,
        ),
    )
    return JointRolloutBank(
        hypothesis_ids=("nominal-contact", "shifted-contact"),
        hypothesis_metadata=metadata,
        hypothesis_prior_weights=np.asarray([0.5, 0.5], dtype=float),
        parameter_particles=parameter_particles,
        parameter_weights=parameter_weights,
        trajectories=trajectories,
        variance_floor_m2=1.0e-8,
        confidence_level=0.90,
    )


def _build_inputs() -> tuple[
    TwinBelief,
    JointRolloutBank,
    JointRolloutBank,
    np.ndarray,
    CounterfactualQuery,
    dict[str, Any],
]:
    observed_action_id = "observed-pull"
    counterfactual_action_id = "counterfactual-lift"
    factual_bank = _rollout_bank(
        action_id=observed_action_id,
        future_action_observed=True,
        axis=0,
    )
    counterfactual_bank = _rollout_bank(
        action_id=counterfactual_action_id,
        future_action_observed=False,
        axis=2,
    )
    selected_hypothesis = 1
    selected_particle = 1
    factual_observations = np.asarray(
        factual_bank.trajectories[selected_hypothesis, selected_particle],
        dtype=float,
    )
    complete_observations = np.empty((5, 2, 3), dtype=float)
    complete_observations[0] = factual_observations[0]
    complete_observations[0, :, 0] -= 0.004
    complete_observations[1] = factual_observations[0]
    complete_observations[2:] = factual_observations[1:]
    observed_actions = np.zeros((5, 1, 3), dtype=float)
    counterfactual_actions = np.zeros((5, 1, 3), dtype=float)
    observed_actions[:, 0, 0] = np.linspace(0.0, 0.04, 5)
    counterfactual_actions[2:, 0, 2] = np.asarray([0.01, 0.02, 0.03])
    context = build_causal_context(
        protocol_id="aip-demo-v1",
        case_id="controlled-demo",
        observations=complete_observations,
        observed_actions=observed_actions,
        counterfactual_actions=counterfactual_actions,
        intervention_frame=2,
        observed_action_id=observed_action_id,
        counterfactual_action_id=counterfactual_action_id,
        observed_action_provenance="deterministic demo command",
        counterfactual_action_provenance="deterministic demo intervention",
    )
    belief = TwinBelief(
        context=context,
        endpoint_frame=context.o_minus.frame_stop - 1,
        particle_ids=("theta-low", "theta-high"),
        theta_names=("stiffness_scale",),
        endpoint_position_m=np.asarray(factual_bank.trajectories[0, :, 0], dtype=float),
        endpoint_velocity_mps=np.zeros((2, 2, 3), dtype=float),
        theta=factual_bank.parameter_particles,
        discrepancy_mean_m=np.zeros((2, 2, 3), dtype=float),
        discrepancy_variance_m2=np.full((2, 2, 3), 1.0e-8, dtype=float),
        weights=factual_bank.parameter_weights,
        metadata={
            "source": "deterministic CPU-only demo",
            "scientific_evidence": False,
        },
    )
    query = CounterfactualQuery(
        context=context,
        controller_points_m=counterfactual_actions[2:],
        horizon_frames=3,
        contact_policy="same_grasp",
        source_factual_intervention_id="0" * 64,
        query_node_indices=np.asarray([0, 1], dtype=np.int64),
        metadata={"same_grasp_semantics": "fixed_kappa"},
    )
    manifest = {"causal_context": context.as_dict()}
    return (
        belief,
        factual_bank,
        counterfactual_bank,
        factual_observations,
        query,
        manifest,
    )


def _verified_artifact_id(path: Path, expected: str) -> str:
    restored = load_contract(path)
    if restored.artifact_id != expected:
        raise RuntimeError(f"reloaded artifact identity mismatch for {path.name}")
    return restored.artifact_id


def run_demo(output_dir: str | Path) -> dict[str, Any]:
    """Run the complete deterministic AIP path and write a verified bundle."""

    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (
        belief,
        factual_bank,
        counterfactual_bank,
        observations,
        query_template,
        manifest,
    ) = _build_inputs()
    prefix_frame_count = 2
    config = FactualAbductionConfig(
        observation_scale_m=0.002,
        likelihood_power=8.0,
        dynamic_likelihood_weight=0.25,
        degrees_of_freedom=4.0,
    )
    factual = abduct_factual_intervention(
        factual_bank,
        belief,
        observations,
        prefix_frame_count=prefix_frame_count,
        config=config,
    )

    changed_future = observations.copy()
    changed_future[prefix_frame_count:] += 10.0
    future_variant = abduct_factual_intervention(
        factual_bank,
        belief,
        changed_future,
        prefix_frame_count=prefix_frame_count,
        config=config,
    )
    if future_variant.artifact_id != factual.artifact_id:
        raise RuntimeError("factual abduction changed after modifying held-out frames")

    query = CounterfactualQuery(
        context=query_template.context,
        controller_points_m=query_template.controller_points_m,
        horizon_frames=query_template.horizon_frames,
        contact_policy=query_template.contact_policy,
        source_factual_intervention_id=factual.artifact_id,
        language=query_template.language,
        query_node_indices=query_template.query_node_indices,
        metadata=query_template.metadata,
    )
    posterior = apply_counterfactual_operator(
        counterfactual_bank,
        manifest,
        belief,
        factual,
        query,
    )
    projected = project_physical_posterior(posterior, query)

    artifacts = {
        "twin_belief": ("twin_belief.npz", belief),
        "factual_intervention": ("factual_intervention.npz", factual),
        "counterfactual_query": ("counterfactual_query.npz", query),
        "physical_posterior": ("physical_posterior.npz", posterior),
        "projected_posterior": ("projected_posterior.npz", projected),
    }
    artifact_ids: dict[str, str] = {}
    artifact_files: dict[str, str] = {}
    for name, (filename, artifact) in artifacts.items():
        path = target / filename
        save_contract(path, artifact)
        artifact_ids[name] = _verified_artifact_id(path, artifact.artifact_id)
        artifact_files[name] = filename

    mean_readout = np.tensordot(
        projected.weights,
        projected.readout_trajectories_m,
        axes=(0, 0),
    )
    top_index = int(np.argmax(factual.weights))
    summary: dict[str, Any] = {
        "schema_name": SUMMARY_SCHEMA_NAME,
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "scientific_evidence": False,
        "future_suffix_invariant": True,
        "future_frames_read_by_abduction": int(
            factual.metadata["future_frames_read_by_abduction"]
        ),
        "artifact_ids": artifact_ids,
        "artifact_files": artifact_files,
        "factual_top_component": factual.component_ids[top_index],
        "factual_top_weight": float(factual.weights[top_index]),
        "projected_frame_count": int(projected.state_trajectories_m.shape[1]),
        "projected_node_count": int(projected.state_trajectories_m.shape[2]),
        "counterfactual_final_mean_m": mean_readout[-1].tolist(),
    }
    summary_path = target / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a deterministic CPU-only Causal4D "
            "abduction-intervention-prediction demo."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/aip-demo"),
        help="Directory for verified NPZ contracts and summary.json.",
    )
    arguments = parser.parse_args(argv)
    summary = run_demo(arguments.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
