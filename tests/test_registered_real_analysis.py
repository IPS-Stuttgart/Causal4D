from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from causal4d.cli.real_protocol import build_parser
from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    bind_repository_interval_amendment,
    expected_real_analysis_interval_amendment,
)
from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_protocol import load_protocol
from causal4d.real_result_source_verification import verify_real_result_sources
from causal4d.registered_real_analysis import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
    analysis_id_for_payload,
    build_registered_real_analysis_manifest,
    validate_registered_real_analysis_manifest,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d/sloth_multi_action_v1.json"


def _method_freeze() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": MILESTONE_ID,
        "status": "sealed",
        "locked_before_confirmatory_collection": True,
        "target_outcomes_observed_at_freeze": False,
        "causal4d": {"commit_sha": "a" * 40},
        "bayesian_phystwin": {"commit_sha": "b" * 40},
        "protocol": {"design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256},
        "preacquisition": {
            "amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
        },
        "interval_amendment": bind_repository_interval_amendment(ROOT),
        "analysis_contract": {
            "entrypoints": [
                "causal4d protocol real",
                "causal4d calibration execution-block",
                "causal4d evidence physical-counterfactual evaluate",
            ],
            "diagnostic_only_entrypoints": ["causal4d calibration real"],
            "allowed_observation_prefix_frames": 6,
            "effect_interval": {
                "amendment_path": (REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH),
                "amendment_id": expected_real_analysis_interval_amendment()[
                    "amendment_id"
                ],
                "primary_method": "target_session_bootstrap_t",
                "required_robustness_method": "student_t_mean",
                "historical_sensitivity_method": (
                    "target_session_percentile_bootstrap"
                ),
                "positive_claim_requires_both_lower_bounds_positive": True,
                "robustness_may_rescue_primary_failure": False,
            },
            "confirmatory_calibration": {
                "entrypoint": "causal4d calibration execution-block",
                "confidence_level": 0.90,
                "outer_fold_count": 12,
                "expected_calibration_units_per_outer_fold": 9,
                "order_statistic_rank_one_based": 9,
                "calibration_unit": (
                    "one preregistered execution per independent session"
                ),
                "score_kind": "max_abs_standardized_coordinate_v1",
                "target_threshold_reselection_allowed": False,
                "pooled_coordinate_conformal_claimed": False,
                "worst_group_coverage_guarantee_claimed": False,
            },
            "target_outcomes_may_select_method_or_hyperparameters": False,
            "optional_branches_may_change_primary_analysis": False,
        },
        "reporting_contract": {
            "report_success_or_well_powered_negative_result": True,
            "report_all_36_executions_or_preregistered_exclusions": True,
            "report_independent_execution_calibration": True,
            "report_effect_intervals_and_replay_reset_variance": True,
            "positive_claim_requires_primary_and_robustness_intervals": True,
            "historical_percentile_interval_is_sensitivity_only": True,
            (
                "optional_semantic_or_public_data_results_cannot_rescue_primary_failure"
            ): True,
        },
    }


def _build(method_freeze_sha256: str = "c" * 64) -> dict[str, object]:
    return build_registered_real_analysis_manifest(
        load_protocol(PROTOCOL),
        _method_freeze(),
        method_freeze_sha256=method_freeze_sha256,
        interval_amendment_binding=bind_repository_interval_amendment(ROOT),
        registered_by="independent-registrar",
        registered_at_utc="2026-08-07T00:00:00+00:00",
    )


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_manifest_is_complete_content_addressed_and_protocol_derived() -> None:
    manifest = _build()

    assert manifest["analysis_id"] == analysis_id_for_payload(manifest)
    assert manifest["effect_reporting"]["bootstrap_replicates"] == (
        BOOTSTRAP_REPLICATES
    )
    assert manifest["effect_reporting"]["bootstrap_seed"] == BOOTSTRAP_SEED
    inventory = manifest["effect_reporting"]["endpoint_inventory"]
    assert [entry["registered_units"] for entry in inventory] == [36, 18, 12]
    assert [entry["target_sessions"] for entry in inventory] == [18, 18, 12]
    assert manifest["confirmatory_calibration"]["order_statistic_rank_one_based"] == 9
    assert manifest["effect_reporting"]["primary_interval_method"] == (
        "target_session_bootstrap_t"
    )
    assert manifest["effect_reporting"]["required_robustness_interval_method"] == (
        "student_t_mean"
    )
    assert manifest["interval_amendment"] == bind_repository_interval_amendment(ROOT)
    diagnostic_ids = {
        arm["arm_id"]
        for arm in manifest["comparison_arms"]
        if arm["role"] == "diagnostic_only"
    }
    assert "causal4d_map_joint_component" in diagnostic_ids
    assert "causal4d_z_with_prior_twin" in diagnostic_ids
    assert "intervention_oracle" in diagnostic_ids

    restored = validate_registered_real_analysis_manifest(
        manifest,
        expected_method_freeze_sha256="c" * 64,
    )
    assert restored == manifest


def test_manifest_rejects_consistently_readdressed_policy_tampering() -> None:
    manifest = _build()
    tampered = copy.deepcopy(manifest)
    tampered["effect_reporting"]["bootstrap_seed"] += 1
    tampered["analysis_id"] = analysis_id_for_payload(tampered)

    with pytest.raises(ValueError, match="effect_reporting changed"):
        validate_registered_real_analysis_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["comparison_arms"][2]["role"] = "diagnostic_only"
    tampered["analysis_id"] = analysis_id_for_payload(tampered)
    with pytest.raises(ValueError, match="comparison_arms changed"):
        validate_registered_real_analysis_manifest(tampered)

    tampered = copy.deepcopy(manifest)
    tampered["interval_amendment"]["contract"]["primary_interval"]["method"] = (
        "target_session_percentile_bootstrap"
    )
    tampered["analysis_id"] = analysis_id_for_payload(tampered)
    with pytest.raises(ValueError, match="interval amendment"):
        validate_registered_real_analysis_manifest(tampered)


def test_source_verification_accepts_schema_v3_and_rejects_target_access(
    tmp_path: Path,
) -> None:
    freeze_path = tmp_path / "method-freeze.json"
    freeze_sha = _write_json(freeze_path, _method_freeze())
    manifest = _build(freeze_sha)
    analysis_path = tmp_path / "registered-analysis.json"
    analysis_sha = _write_json(analysis_path, manifest)

    @dataclass(frozen=True)
    class Binding:
        protocol_id: str = EXPECTED_PROTOCOL_ID
        protocol_design_sha256: str = EXPECTED_PROTOCOL_DESIGN_SHA256
        preacquisition_amendment_sha256: str = EXPECTED_PREACQUISITION_SHA256
        method_freeze_sha256: str = freeze_sha
        analysis_manifest_sha256: str = analysis_sha

    verification = verify_real_result_sources(
        Binding(),
        method_freeze_path=freeze_path,
        analysis_manifest_path=analysis_path,
    )
    assert verification["registered_analysis_manifest"]["sha256"] == analysis_sha

    tampered = copy.deepcopy(manifest)
    tampered["target_outcomes_observed_at_registration"] = True
    tampered["analysis_id"] = analysis_id_for_payload(tampered)
    _write_json(analysis_path, tampered)
    tampered_sha = hashlib.sha256(analysis_path.read_bytes()).hexdigest()

    @dataclass(frozen=True)
    class TamperedBinding:
        protocol_id: str = EXPECTED_PROTOCOL_ID
        protocol_design_sha256: str = EXPECTED_PROTOCOL_DESIGN_SHA256
        preacquisition_amendment_sha256: str = EXPECTED_PREACQUISITION_SHA256
        method_freeze_sha256: str = freeze_sha
        analysis_manifest_sha256: str = tampered_sha

    with pytest.raises(ValueError, match="target_outcomes_observed.*changed"):
        verify_real_result_sources(
            TamperedBinding(),
            method_freeze_path=freeze_path,
            analysis_manifest_path=analysis_path,
        )


def test_real_protocol_cli_exposes_analysis_manifest_routes() -> None:
    parser = build_parser()
    seal = parser.parse_args(
        [
            "analysis-manifest-seal",
            "/repo",
            "protocol.json",
            "freeze.json",
            "analysis.json",
            "--registered-by",
            "reviewer",
        ]
    )
    assert seal.command == "analysis-manifest-seal"
    validate = parser.parse_args(
        [
            "analysis-manifest-validate",
            "/repo",
            "protocol.json",
            "freeze.json",
            "analysis.json",
        ]
    )
    assert validate.command == "analysis-manifest-validate"
