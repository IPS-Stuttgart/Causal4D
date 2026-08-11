from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import causal4d.preacquisition_readiness as readiness_module
from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    bind_repository_interval_amendment,
    expected_real_analysis_interval_amendment,
)
from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_protocol import load_protocol
from causal4d.registered_real_analysis import (
    EXPECTED_PREACQUISITION_SHA256,
    analysis_id_for_payload,
    build_registered_real_analysis_manifest,
)
from causal4d.registered_real_analysis_prerequisite import (
    validate_registered_real_analysis_prerequisite,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = ROOT / "configs/causal4d/sloth_multi_action_v1.json"


def _freeze() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "milestone_id": MILESTONE_ID,
        "status": "sealed",
        "frozen_at_utc": "2026-08-07T00:00:00+00:00",
        "locked_before_confirmatory_collection": True,
        "target_outcomes_observed_at_freeze": False,
        "causal4d": {"commit_sha": "a" * 40},
        "bayesian_phystwin": {"commit_sha": "b" * 40},
        "protocol": {
            "design_sha256": (
                "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
            )
        },
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


def _write_json(path: Path, payload: dict[str, object]) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ready_fixture(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    protocol = load_protocol(PROTOCOL_PATH)
    freeze = _freeze()
    freeze_sha = _write_json(tmp_path / "method_freeze.json", freeze)
    analysis = build_registered_real_analysis_manifest(
        protocol,
        freeze,
        method_freeze_sha256=freeze_sha,
        interval_amendment_binding=bind_repository_interval_amendment(ROOT),
        registered_by="independent-registrar",
        registered_at_utc="2026-08-07T00:30:00+00:00",
    )
    _write_json(tmp_path / "registered-analysis.json", analysis)
    return protocol, {
        "valid": True,
        "sha256": freeze_sha,
    }


def test_registered_analysis_is_a_valid_readiness_prerequisite(tmp_path: Path) -> None:
    protocol, freeze_result = _ready_fixture(tmp_path)

    result = validate_registered_real_analysis_prerequisite(
        protocol,
        tmp_path,
        freeze_result,
    )

    assert result["valid"]
    assert result["analysis_id"]
    assert result["target_outcomes_used"] is False
    assert (
        result["sha256"]
        == hashlib.sha256(
            (tmp_path / "registered-analysis.json").read_bytes()
        ).hexdigest()
    )


def test_registered_analysis_rejects_policy_tampering(tmp_path: Path) -> None:
    protocol, freeze_result = _ready_fixture(tmp_path)
    path = tmp_path / "registered-analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["confirmatory_calibration"]["target_threshold_reselection_allowed"] = True
    payload["analysis_id"] = analysis_id_for_payload(payload)
    _write_json(path, payload)

    result = validate_registered_real_analysis_prerequisite(
        protocol,
        tmp_path,
        freeze_result,
    )

    assert not result["valid"]
    assert "confirmatory_calibration changed" in result["error"]


def test_registered_analysis_must_follow_the_method_freeze(tmp_path: Path) -> None:
    protocol, freeze_result = _ready_fixture(tmp_path)
    path = tmp_path / "registered-analysis.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["registered_at_utc"] = "2026-08-06T23:59:59+00:00"
    payload["analysis_id"] = analysis_id_for_payload(payload)
    _write_json(path, payload)

    result = validate_registered_real_analysis_prerequisite(
        protocol,
        tmp_path,
        freeze_result,
    )

    assert not result["valid"]
    assert "predates the method freeze" in result["error"]


def test_canonical_readiness_builder_requires_registered_analysis() -> None:
    parameter = inspect.signature(
        readiness_module.evaluate_preacquisition_readiness
    ).parameters["require_registered_analysis"]
    assert parameter.default is False
    source = inspect.getsource(readiness_module.build_preacquisition_readiness)
    assert "require_registered_analysis=True" in source
