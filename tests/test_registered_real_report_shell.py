from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from causal4d.cli.command_registry import find_command
from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    bind_repository_interval_amendment,
    expected_real_analysis_interval_amendment,
)
from causal4d.real_experiment_freeze import MILESTONE_ID, SCHEMA_VERSION
from causal4d.real_protocol import load_protocol
from causal4d.registered_real_analysis import (
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    build_registered_real_analysis_manifest,
)
from causal4d.registered_real_report_shell import (
    build_registered_real_report_shell,
    main,
    render_registered_real_report_shell_markdown,
    report_shell_id_for_payload,
    validate_registered_real_report_shell,
    validate_registered_real_report_shell_against_analysis,
    validate_registered_real_report_shell_markdown,
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


def _analysis() -> dict[str, object]:
    return build_registered_real_analysis_manifest(
        load_protocol(PROTOCOL),
        _method_freeze(),
        method_freeze_sha256="c" * 64,
        interval_amendment_binding=bind_repository_interval_amendment(ROOT),
        registered_by="independent-registrar",
        registered_at_utc="2026-08-08T00:00:00+00:00",
    )


def _analysis_bytes(analysis: dict[str, object]) -> bytes:
    return (
        json.dumps(analysis, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _shell(analysis: dict[str, object] | None = None) -> dict[str, object]:
    values = _analysis() if analysis is None else analysis
    payload = _analysis_bytes(values)
    return build_registered_real_report_shell(
        values,
        analysis_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        analysis_manifest_byte_count=len(payload),
    )


def test_shell_is_deterministic_content_addressed_and_result_free() -> None:
    first = _shell()
    second = _shell()

    assert first == second
    assert first["shell_id"] == report_shell_id_for_payload(first)
    assert first["status"] == "target-free-template"
    assert first["safety_boundary"] == {
        "target_outcomes_loaded": False,
        "target_metric_values_loaded": False,
        "confirmatory_execution_evidence_count": 0,
        "may_select_method_or_hyperparameters": False,
        "manual_table_or_figure_selection_allowed": False,
        "derived_artifact_is_claim_bearing": False,
        "report_shell_is_a_scientific_result": False,
    }
    assert all(table["rows"] == [] for table in first["table_plan"])
    assert all(figure["values"] == [] for figure in first["figure_plan"])
    assert all(not item["selected"] for item in first["narrative_plan"])
    endpoint_tables = [
        table["endpoint_id"]
        for table in first["table_plan"]
        if table["table_id"].startswith("endpoint-")
    ]
    assert endpoint_tables == [
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
    ]
    contract = first["analysis_contract"]
    assert (
        contract["interval_amendment"]["amendment_id"]
        == (expected_real_analysis_interval_amendment()["amendment_id"])
    )
    endpoint_table = next(
        table
        for table in first["table_plan"]
        if table["table_id"] == "endpoint-factual_continuation-effects"
    )
    assert "primary_interval_lower" in endpoint_table["required_columns"]
    assert "required_robustness_interval_lower" in endpoint_table["required_columns"]
    assert "positive_claim_interval_gate_passed" in endpoint_table["required_columns"]
    assert (
        "student_t_robustness_may_veto_but_never_rescue_a_positive_claim"
        in first["completion_checks"]
    )
    assert (
        "map_and_prior_twin_abduction_arms_remain_diagnostic_only"
        in first["completion_checks"]
    )


def test_markdown_renders_every_registered_path_without_results() -> None:
    shell = _shell()
    markdown = render_registered_real_report_shell_markdown(shell)

    assert "TARGET-FREE TEMPLATE — NOT A SCIENTIFIC RESULT" in markdown
    assert "No confirmatory outcomes or target metric values are loaded" in markdown
    assert "Factual Continuation" in markdown
    assert "Same Grasp Transfer" in markdown
    assert "New Contact Transfer" in markdown
    assert "Registered Success" in markdown
    assert "Registered Bounded Negative" in markdown
    assert "Independent-execution calibration" in markdown
    assert "Technical failures and preregistered exclusions" in markdown
    assert "Diagnostic intervention-oracle gap attribution" in markdown
    assert shell["shell_id"] in markdown
    assert "not populated" in markdown
    assert validate_registered_real_report_shell_markdown(shell, markdown) == markdown


def test_markdown_validation_rejects_stale_edits_and_non_utf8_bytes() -> None:
    shell = _shell()
    markdown = render_registered_real_report_shell_markdown(shell)

    with pytest.raises(ValueError, match="does not match"):
        validate_registered_real_report_shell_markdown(
            shell,
            markdown.replace("not populated", "selected result", 1),
        )
    with pytest.raises(ValueError, match="must be UTF-8"):
        validate_registered_real_report_shell_markdown(shell, b"\xff")


def test_shell_rejects_target_values_and_selected_result_narratives() -> None:
    shell = _shell()

    tampered = copy.deepcopy(shell)
    tampered["safety_boundary"]["target_outcomes_loaded"] = True
    tampered["shell_id"] = report_shell_id_for_payload(tampered)
    with pytest.raises(ValueError, match="safety boundary changed"):
        validate_registered_real_report_shell(tampered)

    tampered = copy.deepcopy(shell)
    tampered["table_plan"][1]["rows"] = [{"candidate_minus_baseline": -1.0}]
    tampered["shell_id"] = report_shell_id_for_payload(tampered)
    with pytest.raises(ValueError, match="plan changed|rows must be empty"):
        validate_registered_real_report_shell(tampered)

    tampered = copy.deepcopy(shell)
    tampered["narrative_plan"][0]["selected"] = True
    tampered["shell_id"] = report_shell_id_for_payload(tampered)
    with pytest.raises(ValueError, match="plan changed|remain unselected"):
        validate_registered_real_report_shell(tampered)


def test_shell_is_bound_to_the_exact_registered_analysis() -> None:
    analysis = _analysis()
    payload = _analysis_bytes(analysis)
    shell = build_registered_real_report_shell(
        analysis,
        analysis_manifest_sha256=hashlib.sha256(payload).hexdigest(),
        analysis_manifest_byte_count=len(payload),
    )

    changed = copy.deepcopy(analysis)
    changed["software"]["causal4d_commit_sha"] = "d" * 40
    with pytest.raises(ValueError, match="does not match"):
        validate_registered_real_report_shell_against_analysis(
            shell,
            changed,
            analysis_manifest_sha256=hashlib.sha256(payload).hexdigest(),
            analysis_manifest_byte_count=len(payload),
        )


def test_single_executable_route_is_registered_and_non_claim_bearing() -> None:
    command = find_command("evidence real-report-shell")

    assert command.target == "causal4d.registered_real_report_shell:main"
    assert command.lifecycle == "stable"
    assert command.claim_bearing is False
    assert command.historical_name is None


def test_cli_renders_validates_and_refuses_partial_overwrite(
    tmp_path: Path,
) -> None:
    analysis = _analysis()
    analysis_path = tmp_path / "registered-analysis.json"
    analysis_path.write_bytes(_analysis_bytes(analysis))
    shell_path = tmp_path / "report-shell.json"
    markdown_path = tmp_path / "report-shell.md"

    assert (
        main(
            [
                "render",
                str(analysis_path),
                "--output-json",
                str(shell_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )
        == 0
    )
    assert shell_path.is_file()
    assert markdown_path.is_file()
    assert (
        main(
            [
                "validate",
                str(shell_path),
                "--analysis-manifest",
                str(analysis_path),
                "--markdown",
                str(markdown_path),
            ]
        )
        == 0
    )

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        main(
            [
                "render",
                str(analysis_path),
                "--output-json",
                str(shell_path),
                "--output-markdown",
                str(tmp_path / "other.md"),
            ]
        )
    assert not (tmp_path / "other.md").exists()


def test_cli_rejects_a_stale_markdown_pair(tmp_path: Path) -> None:
    analysis = _analysis()
    analysis_path = tmp_path / "registered-analysis.json"
    analysis_path.write_bytes(_analysis_bytes(analysis))
    shell_path = tmp_path / "report-shell.json"
    markdown_path = tmp_path / "report-shell.md"
    assert (
        main(
            [
                "render",
                str(analysis_path),
                "--output-json",
                str(shell_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )
        == 0
    )
    markdown_path.write_text(
        markdown_path.read_text(encoding="utf-8").replace(
            "not populated",
            "manually selected",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match"):
        main(
            [
                "validate",
                str(shell_path),
                "--markdown",
                str(markdown_path),
            ]
        )


def test_render_removes_orphan_markdown_when_json_publication_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    analysis = _analysis()
    analysis_path = tmp_path / "registered-analysis.json"
    analysis_path.write_bytes(_analysis_bytes(analysis))
    shell_path = tmp_path / "report-shell.json"
    markdown_path = tmp_path / "report-shell.md"

    def fail_json_publication(*args: object, **kwargs: object) -> None:
        raise OSError("simulated JSON publication failure")

    monkeypatch.setattr(
        "causal4d.atomic_io.atomic_write_json",
        fail_json_publication,
    )
    with pytest.raises(OSError, match="simulated JSON publication failure"):
        main(
            [
                "render",
                str(analysis_path),
                "--output-json",
                str(shell_path),
                "--output-markdown",
                str(markdown_path),
            ]
        )
    assert not shell_path.exists()
    assert not markdown_path.exists()
