from __future__ import annotations

from pathlib import Path

import causal4d.contracts as contracts
import causal4d.counterfactual as counterfactual
import causal4d.intervention_abduction as intervention_abduction
from causal4d.artifacts import v1 as artifacts_v1
from causal4d.artifacts.v1 import load_contract
from causal4d.demo.aip import run_demo
from causal4d.inference import v1 as inference_v1


ARTIFACTS_V1_EXPORTS = (
    "CONTRACT_VERSION",
    "PUBLIC_API_NAME",
    "PUBLIC_API_VERSION",
    "ActionWindow",
    "CausalContext",
    "CounterfactualQuery",
    "FactualIntervention",
    "ObservationWindow",
    "PhysicalPosterior",
    "TaskPosterior",
    "TwinBelief",
    "array_sha256",
    "build_causal_context",
    "load_contract",
    "save_contract",
)
INFERENCE_V1_EXPORTS = (
    "PUBLIC_API_NAME",
    "PUBLIC_API_VERSION",
    "FactualAbductionConfig",
    "HierarchicalAbductionResult",
    "abduct_factual_intervention",
    "abduct_hierarchical_interventions",
    "apply_counterfactual_operator",
    "factual_joint_weights",
    "project_physical_posterior",
)


def test_artifact_api_v1_has_reviewed_exact_surface() -> None:
    assert tuple(artifacts_v1.__all__) == ARTIFACTS_V1_EXPORTS
    assert artifacts_v1.PUBLIC_API_NAME == "causal4d.artifacts.v1"
    assert artifacts_v1.PUBLIC_API_VERSION == 1
    assert artifacts_v1.TwinBelief is contracts.TwinBelief
    assert artifacts_v1.FactualIntervention is contracts.FactualIntervention
    assert artifacts_v1.save_contract is contracts.save_contract
    assert artifacts_v1.load_contract is contracts.load_contract


def test_inference_api_v1_has_reviewed_exact_surface() -> None:
    assert tuple(inference_v1.__all__) == INFERENCE_V1_EXPORTS
    assert inference_v1.PUBLIC_API_NAME == "causal4d.inference.v1"
    assert inference_v1.PUBLIC_API_VERSION == 1
    assert (
        inference_v1.abduct_factual_intervention
        is intervention_abduction.abduct_factual_intervention
    )
    assert (
        inference_v1.apply_counterfactual_operator
        is counterfactual.apply_counterfactual_operator
    )
    assert (
        inference_v1.project_physical_posterior
        is counterfactual.project_physical_posterior
    )


def _validate_bundle(root: Path, summary: dict[str, object]) -> None:
    artifact_ids = summary["artifact_ids"]
    artifact_files = summary["artifact_files"]
    assert isinstance(artifact_ids, dict)
    assert isinstance(artifact_files, dict)
    for name, expected_id in artifact_ids.items():
        filename = artifact_files[name]
        restored = load_contract(root / str(filename))
        assert restored.artifact_id == expected_id


def test_aip_demo_is_deterministic_and_causally_bounded(tmp_path: Path) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first = run_demo(first_root)
    second = run_demo(second_root)

    assert first == second
    assert first["scientific_evidence"] is False
    assert first["future_suffix_invariant"] is True
    assert first["future_frames_read_by_abduction"] == 0
    assert first["projected_frame_count"] == 3
    assert first["projected_node_count"] == 2
    assert float(first["factual_top_weight"]) > 0.5
    assert (first_root / "summary.json").is_file()
    _validate_bundle(first_root, first)
    _validate_bundle(second_root, second)
