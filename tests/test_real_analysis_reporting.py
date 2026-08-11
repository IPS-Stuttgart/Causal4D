from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.real_analysis_reporting import main as reporting_main
from causal4d.execution_block_calibration import (
    ExecutionBlockCalibrationCase,
    evaluate_execution_block_cases,
    fit_execution_block_conformal_calibration,
)
from causal4d.real_analysis_reporting import (
    EXPECTED_OBJECT_ID,
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
    build_real_analysis_effect_report,
    effect_table_id_for_payload,
    summarize_execution_block_utility,
)
from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_result_source_verification import (
    REGISTERED_ANALYSIS_ARTIFACT_KIND,
)


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d/sloth_multi_action_v1.json"


def _write_json(path: Path, payload: object) -> str:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_pair(tmp_path: Path) -> tuple[Path, Path, str, str]:
    freeze = tmp_path / "method-freeze.json"
    freeze_sha = _write_json(
        freeze,
        {
            "schema_version": SCHEMA_VERSION,
            "milestone_id": MILESTONE_ID,
            "status": "sealed",
            "locked_before_confirmatory_collection": True,
            "target_outcomes_observed_at_freeze": False,
            "protocol": {"design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256},
            "preacquisition": {"amendment_sha256": EXPECTED_PREACQUISITION_SHA256},
            "analysis_contract": {
                "target_outcomes_may_select_method_or_hyperparameters": False,
                "optional_branches_may_change_primary_analysis": False,
            },
        },
    )
    analysis = tmp_path / "registered-analysis.json"
    analysis_sha = _write_json(
        analysis,
        {
            "schema_version": 1,
            "artifact_kind": REGISTERED_ANALYSIS_ARTIFACT_KIND,
            "analysis_id": "registered-real-analysis-v1",
            "protocol_id": EXPECTED_PROTOCOL_ID,
            "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
            "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
            "method_freeze_sha256": freeze_sha,
            "primary_analysis_locked": True,
            "target_outcomes_may_select_method_or_hyperparameters": False,
            "optional_branches_may_change_primary_analysis": False,
        },
    )
    return freeze, analysis, freeze_sha, analysis_sha


def _factual_payload(
    *,
    freeze_sha: str,
    analysis_sha: str,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    executions = {value["execution_id"]: value for value in protocol["executions"]}
    records = []
    for split in protocol["splits"]["factual_continuation"]:
        execution = executions[split["execution_id"]]
        index = execution["acquisition_execution_index"]
        baseline = 2.0 + 0.01 * index
        records.append(
            {
                "unit_id": execution["execution_id"],
                "source_execution_id": None,
                "target_execution_id": execution["execution_id"],
                "session_id": execution["session_id"],
                "acquisition_execution_index": index,
                "action_id": execution["command_profile_id"],
                "contact_region_id": execution["contact_region_id"],
                "realization_condition_id": execution["realization_condition_id"],
                "included": True,
                "exclusion_reason": None,
                "baseline_value": baseline,
                "candidate_value": baseline - 1.0,
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Causal4DRealAnalysisEffectTable",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
        "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
        "method_freeze_sha256": freeze_sha,
        "analysis_manifest_sha256": analysis_sha,
        "endpoint": "factual_continuation",
        "metric_id": "track_error_m",
        "metric_unit": "m",
        "lower_is_better": True,
        "target_outcomes_used": True,
        "target_informed_selection": False,
        "object_id": EXPECTED_OBJECT_ID,
        "records": records,
    }
    payload["effect_table_id"] = effect_table_id_for_payload(payload)
    return payload


def test_report_uses_sessions_as_the_resampling_unit(tmp_path: Path) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)

    report = build_real_analysis_effect_report(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )

    primary = report["primary_session_clustered_effect"]
    assert report["accounting"]["expected_unit_count"] == 36
    assert report["accounting"]["expected_session_count"] == 18
    assert primary["session_is_resampling_unit"] is True
    assert primary["executions_are_not_treated_as_independent"] is True
    assert primary["equal_session_weighted_improvement"]["mean"] == pytest.approx(1.0)

    registered = primary["confidence_interval"]
    robustness = primary["required_robustness_interval"]
    historical = primary["historical_percentile_sensitivity_interval"]
    decision = primary["interval_decision"]

    assert registered["method"] == "target_session_bootstrap_t"
    assert registered["estimable"] is False
    assert registered["point_estimate"] == pytest.approx(1.0)
    assert registered["lower"] is None
    assert registered["upper"] is None
    assert registered["degenerate_sample"] is True

    assert robustness["method"] == "student_t_mean"
    assert robustness["estimable"] is False
    assert robustness["point_estimate"] == pytest.approx(1.0)
    assert robustness["lower"] is None
    assert robustness["upper"] is None
    assert robustness["degenerate_sample"] is True

    assert historical["method"] == "target_session_percentile_bootstrap"
    assert historical["estimable"] is True
    assert historical["lower"] == pytest.approx(1.0)
    assert historical["upper"] == pytest.approx(1.0)
    assert historical["degenerate_sample"] is True

    assert decision["registered_interval_inputs_match"] is True
    assert decision["degenerate_session_panel"] is True
    assert decision["degenerate_session_panel_blocks_positive_claim"] is True
    assert decision["positive_claim_interval_gate_passed"] is False
    assert report["claim_boundary"]["object_class_generalization_claimed"] is False
    assert (
        report["design_diagnostics"]["condition_comparisons_are_descriptive_only"]
        is True
    )
    assert report["design_diagnostics"]["fully_crossed"] is True
    assert report["design_diagnostics"]["balanced_across_actions"] is False


def test_session_weighting_survives_one_preregistered_exclusion(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    records = payload["records"]
    assert isinstance(records, list)
    for record in records:
        record["candidate_value"] = record["baseline_value"]
    first_session = records[0]["session_id"]
    first_records = [
        record for record in records if record["session_id"] == first_session
    ]
    first_records[0]["candidate_value"] = first_records[0]["baseline_value"] - 10.0
    first_records[1]["included"] = False
    first_records[1]["exclusion_reason"] = "preregistered technical exclusion"
    first_records[1]["baseline_value"] = None
    first_records[1]["candidate_value"] = None
    payload["effect_table_id"] = effect_table_id_for_payload(payload)
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)

    report = build_real_analysis_effect_report(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )

    session_mean = report["primary_session_clustered_effect"][
        "equal_session_weighted_improvement"
    ]["mean"]
    execution_mean = report["unweighted_execution_diagnostic"]["mean"]
    assert session_mean == pytest.approx(10.0 / 18.0)
    assert execution_mean == pytest.approx(10.0 / 35.0)
    assert report["accounting"]["excluded_unit_count"] == 1


def test_report_rejects_missing_or_relabelled_registered_units(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    records = payload["records"]
    assert isinstance(records, list)
    removed = records.pop()
    payload["effect_table_id"] = effect_table_id_for_payload(payload)
    effect_table = tmp_path / "missing.json"
    _write_json(effect_table, payload)
    with pytest.raises(ValueError, match="accounting differs"):
        build_real_analysis_effect_report(
            effect_table,
            PROTOCOL,
            method_freeze_path=freeze,
            analysis_manifest_path=analysis,
        )

    records.append(removed)
    records[0]["action_id"] = "target-informed-relabel"
    payload["effect_table_id"] = effect_table_id_for_payload(payload)
    effect_table = tmp_path / "relabelled.json"
    _write_json(effect_table, payload)
    with pytest.raises(ValueError, match="differs from the locked protocol"):
        build_real_analysis_effect_report(
            effect_table,
            PROTOCOL,
            method_freeze_path=freeze,
            analysis_manifest_path=analysis,
        )


def test_reporting_recomputes_the_complete_protocol_digest(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    protocol["executions"][0]["command_profile_id"] = "tampered-action"
    tampered_protocol = tmp_path / "tampered-protocol.json"
    _write_json(tampered_protocol, protocol)

    with pytest.raises(ValueError, match="design SHA-256 does not match"):
        build_real_analysis_effect_report(
            effect_table,
            tampered_protocol,
            method_freeze_path=freeze,
            analysis_manifest_path=analysis,
        )


def test_reporting_rejects_duplicate_keys_in_bound_analysis_source(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, _ = _source_pair(tmp_path)
    duplicate_payload = (
        "{"
        '"schema_version":1,'
        f'"artifact_kind":"{REGISTERED_ANALYSIS_ARTIFACT_KIND}",'
        '"analysis_id":"first","analysis_id":"second",'
        f'"protocol_id":"{EXPECTED_PROTOCOL_ID}",'
        f'"protocol_design_sha256":"{EXPECTED_PROTOCOL_DESIGN_SHA256}",'
        f'"preacquisition_amendment_sha256":"{EXPECTED_PREACQUISITION_SHA256}",'
        f'"method_freeze_sha256":"{freeze_sha}",'
        '"primary_analysis_locked":true,'
        '"target_outcomes_may_select_method_or_hyperparameters":false,'
        '"optional_branches_may_change_primary_analysis":false}'
    ).encode("utf-8")
    analysis.write_bytes(duplicate_payload)
    analysis_sha = hashlib.sha256(duplicate_payload).hexdigest()
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)

    with pytest.raises(ValueError, match="duplicate JSON object key"):
        build_real_analysis_effect_report(
            effect_table,
            PROTOCOL,
            method_freeze_path=freeze,
            analysis_manifest_path=analysis,
        )


def test_module_cli_writes_the_registered_report(tmp_path: Path) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    output = tmp_path / "report.json"
    _write_json(effect_table, payload)

    result = reporting_main(
        [
            str(effect_table),
            str(PROTOCOL),
            str(output),
            "--method-freeze",
            str(freeze),
            "--analysis-manifest",
            str(analysis),
            "--require-estimable",
        ]
    )

    assert result == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["accounting"]["complete"] is True
    assert report["report_id"]


def _calibration_case(
    execution_id: str,
    session_id: str,
    role: str,
    residual: float,
) -> ExecutionBlockCalibrationCase:
    mean = np.zeros((4, 2, 3), dtype=float)
    variance = np.full_like(mean, 0.01**2)
    truth = np.full_like(mean, residual * 0.01)
    return ExecutionBlockCalibrationCase(
        execution_id=execution_id,
        session_id=session_id,
        outer_fold_id="fold-1",
        split_role=role,
        prediction_case_id="sloth-object",
        action_id="lift_low",
        contact_region_id="left_forepaw",
        mean_m=mean,
        variance_m2=variance,
        truth_m=truth,
        valid=np.ones(mean.shape[:2], dtype=bool),
        start_frame=1,
    )


def test_calibration_utility_reports_width_and_fragility() -> None:
    fit = tuple(
        _calibration_case(f"fit-{index}", f"fit-session-{index}", "fit", 1.0)
        for index in range(3)
    )
    calibration_cases = tuple(
        _calibration_case(
            f"cal-{index}",
            f"cal-session-{index}",
            "calibration",
            float(index + 1),
        )
        for index in range(9)
    )
    calibration = fit_execution_block_conformal_calibration(
        fit,
        calibration_cases,
        expected_fit_units=3,
    )
    targets = (
        _calibration_case("target-a", "target-session-a", "target", 2.0),
        _calibration_case("target-b", "target-session-b", "target", 20.0),
    )
    evaluation = evaluate_execution_block_cases(targets, calibration)

    summary = summarize_execution_block_utility(calibration, evaluation)

    assert summary["target_execution_count"] == 2
    assert summary["target_coordinate_count"] > 0
    assert summary["interval_width_m"]["mean_of_execution_means"] > 0.0
    assert (
        summary["calibration_fragility"]["maximum_score"]
        >= summary["calibration_fragility"]["second_largest_score"]
    )
    assert summary["coverage_without_interval_width_is_sufficient"] is False
    assert summary["fragility_may_select_threshold"] is False
