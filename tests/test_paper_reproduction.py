from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from causal4d.cli.command_registry import find_command
from causal4d.paper_reproduction import (
    main,
    publish_paper_reproduction_bundle,
    verify_paper_reproduction_bundle,
)
from causal4d.real_analysis_interval_amendment import (
    bind_repository_interval_amendment,
)
from causal4d.real_analysis_reporting import (
    EXPECTED_PREACQUISITION_SHA256,
    EXPECTED_PROTOCOL_DESIGN_SHA256,
    EXPECTED_PROTOCOL_ID,
)
from causal4d.real_protocol import load_protocol
from causal4d.registered_real_analysis import (
    build_registered_real_analysis_manifest,
)
from tests.test_real_analysis_reporting import _factual_payload
from tests.test_registered_real_report_shell import _method_freeze

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "configs/causal4d/sloth_multi_action_v1.json"


def _json_bytes(payload: object) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sources(tmp_path: Path) -> tuple[Path, Path, str, str]:
    freeze_path = tmp_path / "method-freeze.json"
    freeze_payload = _method_freeze()
    freeze_bytes = _json_bytes(freeze_payload)
    freeze_path.write_bytes(freeze_bytes)
    freeze_sha = hashlib.sha256(freeze_bytes).hexdigest()

    analysis = build_registered_real_analysis_manifest(
        load_protocol(PROTOCOL),
        freeze_payload,
        method_freeze_sha256=freeze_sha,
        interval_amendment_binding=bind_repository_interval_amendment(ROOT),
        registered_by="independent-registrar",
        registered_at_utc="2026-08-12T00:00:00+00:00",
    )
    analysis_path = tmp_path / "registered-analysis.json"
    analysis_bytes = _json_bytes(analysis)
    analysis_path.write_bytes(analysis_bytes)
    analysis_sha = hashlib.sha256(analysis_bytes).hexdigest()
    return freeze_path, analysis_path, freeze_sha, analysis_sha


def _incomplete_gate_summary(
    *,
    freeze_sha: str,
    analysis_sha: str,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DRealResultGateSummary",
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
        "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
        "method_freeze_sha256": freeze_sha,
        "analysis_manifest_sha256": analysis_sha,
        "evidence_status": "incomplete",
        "factual_continuation": "not_estimable",
        "same_grasp_transfer": "not_estimable",
        "new_contact_transfer": "not_estimable",
        "execution_block_calibration": "not_estimable",
        "oracle_diagnosis": "not_estimable",
        "technical_failure_count": 0,
        "preregistered_exclusion_count": 0,
        "target_informed_selection": False,
    }


def _complete_gate_summary(
    *,
    freeze_sha: str,
    analysis_sha: str,
) -> dict[str, object]:
    payload = _incomplete_gate_summary(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    payload.update(
        {
            "evidence_status": "complete",
            "factual_continuation": "passed",
            "same_grasp_transfer": "passed",
            "new_contact_transfer": "passed",
            "execution_block_calibration": "passed",
        }
    )
    return payload


def _endpoint_payload(
    endpoint: str,
    *,
    freeze_sha: str,
    analysis_sha: str,
) -> dict[str, object]:
    if endpoint == "factual_continuation":
        return _factual_payload(
            freeze_sha=freeze_sha,
            analysis_sha=analysis_sha,
        )
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    split_name = {
        "same_grasp_transfer": "same_grasp_intervention_prediction",
        "new_contact_transfer": "new_contact_intervention_prediction",
    }[endpoint]
    executions = {value["execution_id"]: value for value in protocol["executions"]}
    records = []
    for split in protocol["splits"][split_name]:
        source_id = split["source_execution_id"]
        target_id = split["target_execution_id"]
        target = executions[target_id]
        baseline = 2.0 + 0.01 * target["acquisition_execution_index"]
        records.append(
            {
                "unit_id": f"{source_id}->{target_id}",
                "source_execution_id": source_id,
                "target_execution_id": target_id,
                "session_id": target["session_id"],
                "acquisition_execution_index": (target["acquisition_execution_index"]),
                "action_id": target["command_profile_id"],
                "contact_region_id": target["contact_region_id"],
                "realization_condition_id": target["realization_condition_id"],
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
        "endpoint": endpoint,
        "metric_id": "track_error_m",
        "metric_unit": "m",
        "lower_is_better": True,
        "target_outcomes_used": True,
        "target_informed_selection": False,
        "object_id": "sloth_plush_instance_1",
        "records": records,
    }
    from causal4d.real_analysis_reporting import effect_table_id_for_payload

    payload["effect_table_id"] = effect_table_id_for_payload(payload)
    return payload


def test_target_free_bundle_is_portable_and_exactly_reproducible(
    tmp_path: Path,
) -> None:
    freeze, analysis, _, _ = _sources(tmp_path)
    bundle = tmp_path / "paper-bundle"

    result = publish_paper_reproduction_bundle(
        bundle,
        PROTOCOL,
        analysis,
        method_freeze_path=freeze,
    )

    assert result["passed"] is True
    assert result["status"] == "target-free-plan"
    assert result["effect_report_count"] == 0
    assert result["gate_summary_present"] is False
    assert (bundle / "manifest.json").is_file()
    assert (bundle / "paper-reproduction.json").is_file()
    assert (bundle / "semantic-conformance.json").is_file()
    assert (bundle / "report-shell.md").is_file()
    assert verify_paper_reproduction_bundle(bundle) == result

    second_bundle = tmp_path / "paper-bundle-second"
    second_result = publish_paper_reproduction_bundle(
        second_bundle,
        PROTOCOL,
        analysis,
        method_freeze_path=freeze,
    )
    assert second_result == result
    first_files = {path.name: path.read_bytes() for path in bundle.iterdir()}
    second_files = {path.name: path.read_bytes() for path in second_bundle.iterdir()}
    assert second_files == first_files

    index = json.loads((bundle / "paper-reproduction.json").read_text())
    assert index["claim_boundary"] == {
        "contains_complete_registered_result": False,
        "bundle_is_a_scientific_result": False,
        "bundle_is_claim_bearing": False,
        "frozen_estimator_changed": False,
        "optional_branch_rescue_permitted": False,
        "physical_evidence_increment": 0,
        "raw_sensor_data_included": False,
        "registered_protocol_changed": False,
        "target_informed_selection_permitted": False,
    }


def test_bundle_regenerates_effect_report_and_incomplete_interpretation(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _sources(tmp_path)
    effect_table = tmp_path / "factual-effects.json"
    effect_table.write_bytes(
        _json_bytes(
            _factual_payload(
                freeze_sha=freeze_sha,
                analysis_sha=analysis_sha,
            )
        )
    )
    gate_path = tmp_path / "real-result-gates.json"
    gate_path.write_bytes(
        _json_bytes(
            _incomplete_gate_summary(
                freeze_sha=freeze_sha,
                analysis_sha=analysis_sha,
            )
        )
    )
    bundle = tmp_path / "paper-bundle"

    result = publish_paper_reproduction_bundle(
        bundle,
        PROTOCOL,
        analysis,
        method_freeze_path=freeze,
        effect_table_paths=(effect_table,),
        gate_summary_path=gate_path,
    )

    assert result["status"] == "incomplete-result"
    assert result["effect_report_count"] == 1
    assert result["gate_summary_present"] is True
    assert result["complete_evidence_registry"] is False
    assert (bundle / "effect-report-factual-continuation-track-error-m.json").is_file()
    interpretation = json.loads(
        (bundle / "real-result-interpretation.json").read_text()
    )
    assert interpretation["paper_status"] == "incomplete"
    assert interpretation["rule_id"] == "incomplete_evidence"


def test_require_complete_fails_closed_before_publication(tmp_path: Path) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _sources(tmp_path)
    gate_path = tmp_path / "real-result-gates.json"
    gate_path.write_bytes(
        _json_bytes(
            _incomplete_gate_summary(
                freeze_sha=freeze_sha,
                analysis_sha=analysis_sha,
            )
        )
    )
    bundle = tmp_path / "paper-bundle"

    with pytest.raises(ValueError, match="complete evidence registry"):
        publish_paper_reproduction_bundle(
            bundle,
            PROTOCOL,
            analysis,
            method_freeze_path=freeze,
            gate_summary_path=gate_path,
            require_complete=True,
        )

    assert not bundle.exists()


def test_verifier_rejects_a_modified_product(tmp_path: Path) -> None:
    freeze, analysis, _, _ = _sources(tmp_path)
    bundle = tmp_path / "paper-bundle"
    publish_paper_reproduction_bundle(
        bundle,
        PROTOCOL,
        analysis,
        method_freeze_path=freeze,
    )
    markdown = bundle / "report-shell.md"
    markdown.write_text(markdown.read_text() + "manual edit\n", encoding="utf-8")

    with pytest.raises(ValueError, match="(checksum|byte count) changed"):
        verify_paper_reproduction_bundle(bundle)


def test_cli_builds_and_verifies_the_same_bundle(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    freeze, analysis, _, _ = _sources(tmp_path)
    bundle = tmp_path / "paper-bundle"

    assert (
        main(
            [
                "--protocol",
                str(PROTOCOL),
                "--analysis-manifest",
                str(analysis),
                "--method-freeze",
                str(freeze),
                "--output-dir",
                str(bundle),
            ]
        )
        == 0
    )
    built = json.loads(capsys.readouterr().out)
    assert main(["--verify", str(bundle)]) == 0
    verified = json.loads(capsys.readouterr().out)
    assert verified == built


def test_single_executable_route_is_stable_and_non_claim_bearing() -> None:
    command = find_command("paper reproduce")

    assert command.target == "causal4d.paper_reproduction:main"
    assert command.lifecycle == "stable"
    assert command.claim_bearing is False
    assert command.historical_name is None


def test_complete_gate_requires_all_registered_effect_endpoints(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _sources(tmp_path)
    gate_path = tmp_path / "real-result-gates.json"
    gate_path.write_bytes(
        _json_bytes(
            _complete_gate_summary(
                freeze_sha=freeze_sha,
                analysis_sha=analysis_sha,
            )
        )
    )

    with pytest.raises(ValueError, match="every registered endpoint"):
        publish_paper_reproduction_bundle(
            tmp_path / "paper-bundle",
            PROTOCOL,
            analysis,
            method_freeze_path=freeze,
            gate_summary_path=gate_path,
        )


def test_complete_bundle_covers_all_registered_endpoints(tmp_path: Path) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _sources(tmp_path)
    effect_tables = []
    for endpoint in (
        "factual_continuation",
        "same_grasp_transfer",
        "new_contact_transfer",
    ):
        path = tmp_path / f"{endpoint}-effects.json"
        path.write_bytes(
            _json_bytes(
                _endpoint_payload(
                    endpoint,
                    freeze_sha=freeze_sha,
                    analysis_sha=analysis_sha,
                )
            )
        )
        effect_tables.append(path)
    gate_path = tmp_path / "real-result-gates.json"
    gate_path.write_bytes(
        _json_bytes(
            _complete_gate_summary(
                freeze_sha=freeze_sha,
                analysis_sha=analysis_sha,
            )
        )
    )
    bundle = tmp_path / "paper-bundle"

    result = publish_paper_reproduction_bundle(
        bundle,
        PROTOCOL,
        analysis,
        method_freeze_path=freeze,
        effect_table_paths=effect_tables,
        gate_summary_path=gate_path,
        require_complete=True,
    )

    assert result["status"] == "complete-result"
    assert result["effect_report_count"] == 3
    assert result["complete_evidence_registry"] is True
    assert (
        verify_paper_reproduction_bundle(
            bundle,
            require_complete=True,
        )
        == result
    )


def test_target_informed_gate_is_retained_as_failed_conformance(
    tmp_path: Path,
) -> None:
    freeze, analysis, freeze_sha, analysis_sha = _sources(tmp_path)
    gate = _incomplete_gate_summary(
        freeze_sha=freeze_sha,
        analysis_sha=analysis_sha,
    )
    gate["target_informed_selection"] = True
    gate_path = tmp_path / "real-result-gates.json"
    gate_path.write_bytes(_json_bytes(gate))
    bundle = tmp_path / "paper-bundle"

    result = publish_paper_reproduction_bundle(
        bundle,
        PROTOCOL,
        analysis,
        method_freeze_path=freeze,
        gate_summary_path=gate_path,
    )

    assert result["semantic_conformance_status"] == "failed"
    interpretation = json.loads(
        (bundle / "real-result-interpretation.json").read_text()
    )
    assert interpretation["rule_id"] == "confirmatory_boundary_violated"
    conformance = json.loads((bundle / "semantic-conformance.json").read_text())
    assert conformance["status"] == "failed"
    assert conformance["checks"]["gate_summary_target_informed_selection"] is True
