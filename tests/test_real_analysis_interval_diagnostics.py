from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from causal4d.cli.real_analysis_interval_diagnostics import main as diagnostics_main
from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    bind_repository_interval_amendment,
    expected_real_analysis_interval_amendment,
)
from causal4d.real_analysis_interval_diagnostics import (
    bootstrap_t_sensitivity_interval,
    build_real_analysis_interval_diagnostics,
    student_t_sensitivity_interval,
)
from causal4d.real_analysis_reporting import (
    EXPECTED_OBJECT_ID,
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
    build_real_analysis_effect_report,
    effect_table_id_for_payload,
)
from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_protocol import load_protocol
from causal4d.registered_real_analysis import (
    build_registered_real_analysis_manifest,
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
    interval_amendment = bind_repository_interval_amendment(ROOT)
    freeze_payload: dict[str, object] = {
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
        "interval_amendment": interval_amendment,
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
    freeze = tmp_path / "method-freeze.json"
    freeze_sha = _write_json(freeze, freeze_payload)
    analysis_payload = build_registered_real_analysis_manifest(
        load_protocol(PROTOCOL),
        freeze_payload,
        method_freeze_sha256=freeze_sha,
        interval_amendment_binding=interval_amendment,
        registered_by="independent-registrar",
        registered_at_utc="2026-08-07T00:30:00+00:00",
    )
    analysis = tmp_path / "registered-analysis.json"
    analysis_sha = _write_json(analysis, analysis_payload)
    return freeze, analysis, freeze_sha, analysis_sha


def _factual_payload(
    *,
    freeze_sha: str,
    analysis_sha: str,
) -> dict[str, object]:
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    executions = {value["execution_id"]: value for value in protocol["executions"]}
    session_order: dict[str, int] = {}
    records = []
    for split in protocol["splits"]["factual_continuation"]:
        execution = executions[split["execution_id"]]
        session_id = execution["session_id"]
        if session_id not in session_order:
            session_order[session_id] = len(session_order)
        index = execution["acquisition_execution_index"]
        baseline = 2.0 + 0.01 * index
        improvement = 0.25 + 0.03 * session_order[session_id]
        records.append(
            {
                "unit_id": execution["execution_id"],
                "source_execution_id": None,
                "target_execution_id": execution["execution_id"],
                "session_id": session_id,
                "acquisition_execution_index": index,
                "action_id": execution["command_profile_id"],
                "contact_region_id": execution["contact_region_id"],
                "realization_condition_id": execution["realization_condition_id"],
                "included": True,
                "exclusion_reason": None,
                "baseline_value": baseline,
                "candidate_value": baseline - improvement,
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


def test_sensitivity_intervals_are_translation_equivariant() -> None:
    values = [-0.2, -0.05, 0.1, 0.25, 0.6]
    offset = 3.25
    shifted = [value + offset for value in values]

    for builder in (
        student_t_sensitivity_interval,
        bootstrap_t_sensitivity_interval,
    ):
        original = builder(values)
        translated = builder(shifted)
        assert translated["point_estimate"] == pytest.approx(
            original["point_estimate"] + offset
        )
        assert translated["lower"] == pytest.approx(original["lower"] + offset)
        assert translated["upper"] == pytest.approx(original["upper"] + offset)
        assert translated["may_change_primary_decision"] is False
        assert translated["finite_sample_coverage_guaranteed"] is False


def test_bootstrap_t_is_deterministic_and_finite() -> None:
    values = np.linspace(-0.3, 0.7, 18).tolist()
    first = bootstrap_t_sensitivity_interval(values)
    second = bootstrap_t_sensitivity_interval(values)

    assert first == second
    assert first["estimable"] is True
    assert first["replicates"] == 20_000
    assert first["seed"] == 20_260_726
    assert first["finite_studentized_replicate_fraction"] > 0.99
    assert np.isfinite(first["lower"])
    assert np.isfinite(first["upper"])
    assert first["lower"] <= first["point_estimate"] <= first["upper"]


def test_degenerate_samples_produce_explicit_point_intervals() -> None:
    values = [1.5] * 12

    for result in (
        student_t_sensitivity_interval(values),
        bootstrap_t_sensitivity_interval(values),
    ):
        assert result["estimable"] is True
        assert result["degenerate_sample"] is True
        assert result["lower"] == pytest.approx(1.5)
        assert result["upper"] == pytest.approx(1.5)


def test_companion_artifact_verifies_registered_intervals_and_sources(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    _write_json(effect_table, payload)

    primary = build_real_analysis_effect_report(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )
    first = build_real_analysis_interval_diagnostics(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )
    second = build_real_analysis_interval_diagnostics(
        effect_table,
        PROTOCOL,
        method_freeze_path=freeze,
        analysis_manifest_path=analysis,
    )

    assert first == second
    assert first["diagnostic_id"] == second["diagnostic_id"]
    assert first["source_primary_report_id"] == primary["report_id"]
    assert first["source_verification"] == primary["source_verification"]
    assert first["included_session_count"] == 18
    registered = first["registered_intervals"]
    reported = primary["primary_session_clustered_effect"]
    assert registered["primary_bootstrap_t"] == reported["confidence_interval"]
    assert (
        registered["required_student_t_robustness"]
        == reported["required_robustness_interval"]
    )
    assert (
        registered["historical_percentile_sensitivity"]
        == reported["historical_percentile_sensitivity_interval"]
    )
    assert registered["decision"] == reported["interval_decision"]
    assert first["interpretation"]["bootstrap_t_is_registered_primary"] is True
    assert first["interpretation"]["student_t_may_veto_positive_claim"] is True
    assert first["interpretation"]["student_t_may_rescue_primary_failure"] is False
    assert first["interpretation"]["target_informed_selection"] is False
    assert (
        first["claim_boundary"]["physical_target_outcomes_used_to_choose_interval"]
        is False
    )


def test_cli_publishes_atomically_and_requires_explicit_overwrite(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _source_pair(tmp_path)
    payload = _factual_payload(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    effect_table = tmp_path / "effects.json"
    output = tmp_path / "interval-diagnostics.json"
    _write_json(effect_table, payload)
    arguments = [
        str(effect_table),
        str(PROTOCOL),
        str(output),
        "--method-freeze",
        str(freeze),
        "--analysis-manifest",
        str(analysis),
    ]

    assert diagnostics_main(arguments) == 0
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["artifact_kind"] == "Causal4DRealAnalysisIntervalDiagnostics"
    assert published["registered_intervals"]["primary_bootstrap_t"]["role"] == (
        "primary"
    )

    with pytest.raises(FileExistsError):
        diagnostics_main(arguments)
    assert diagnostics_main([*arguments, "--overwrite"]) == 0
