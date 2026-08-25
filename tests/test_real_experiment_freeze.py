import hashlib
import json
import subprocess
from copy import deepcopy
from pathlib import Path

import pytest

from causal4d.preacquisition_protocol_v5 import single_operator_governance_policy
from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
    expected_real_analysis_interval_amendment,
    expected_real_analysis_interval_evidence,
)
from causal4d.real_experiment_freeze import (
    ACQUISITION_CANDIDATE_PATH,
    BPT_PIN_PATH,
    DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS,
    MECHANISM_GATE_EVIDENCE_PATH,
    PREACQUISITION_PATH,
    PREACQUISITION_PLAN_ID,
    REQUIRED_ANALYSIS_ENTRYPOINTS,
    REQUIRED_LOCKED_PATHS,
    build_method_freeze_manifest,
    validate_method_freeze_manifest,
    write_method_freeze_manifest,
    validate_repository_checkout,
)


BPT_SHA = "c7ad36aad7e592ce8a391c9ca2d4db7389dee3ac"
CAUSAL4D_SHA = "a" * 40
PROTOCOL_SHA = "b" * 64


def _canonical_payload_sha256(
    values: dict[str, object],
    *,
    omitted_field: str,
) -> str:
    payload = dict(values)
    payload.pop(omitted_field, None)
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "repository"
    for relative in REQUIRED_LOCKED_PATHS:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"locked:{relative}\n", encoding="utf-8")
    (root / "configs/causal4d/sloth_multi_action_v1.json").write_text(
        json.dumps({"design_sha256": PROTOCOL_SHA}), encoding="utf-8"
    )
    (root / REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH).write_text(
        json.dumps(expected_real_analysis_interval_evidence()),
        encoding="utf-8",
    )
    (root / REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH).write_text(
        json.dumps(expected_real_analysis_interval_amendment()),
        encoding="utf-8",
    )

    evidence: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "MechanismGateControlEvidence",
        "acceptance_checks": {
            "placebo_null_full_gate_upper_below_5_percent": True,
            "positive_control_full_gate_lower_above_80_percent": True,
            "wrong_family_on_positive_upper_below_5_percent": True,
        },
        "frozen_v3_gate_supported_in_controlled_benchmark": True,
    }
    evidence["result_sha256"] = _canonical_payload_sha256(
        evidence,
        omitted_field="result_sha256",
    )
    (root / MECHANISM_GATE_EVIDENCE_PATH).write_text(
        json.dumps(evidence), encoding="utf-8"
    )

    amendment: dict[str, object] = {
        "schema_version": 1,
        "plan_id": PREACQUISITION_PLAN_ID,
        "status": "supersedes_v4_before_any_physical_execution",
        "supersedes": {
            "physical_executions_completed_before_supersession": 0,
        },
        "governance": single_operator_governance_policy(),
        "base_protocol": {
            "design_sha256": PROTOCOL_SHA,
            "confirmatory_execution_count": 36,
        },
        "mechanism_gate_control_lock": {
            "evidence_artifact": MECHANISM_GATE_EVIDENCE_PATH,
            "evidence_sha256": evidence["result_sha256"],
        },
    }
    amendment["amendment_sha256"] = _canonical_payload_sha256(
        amendment,
        omitted_field="amendment_sha256",
    )
    (root / PREACQUISITION_PATH).write_text(json.dumps(amendment), encoding="utf-8")

    (root / BPT_PIN_PATH).write_text(BPT_SHA + "\n", encoding="utf-8")
    candidate: dict[str, object] = {
        "schema_version": 1,
        "candidate_id": "causal4d-sloth-primary-acquisition-v1",
        "status": "selected_before_source_panel",
        "protocol_design_sha256": PROTOCOL_SHA,
        "physical_model": {"bayesian_phystwin_commit_sha": BPT_SHA},
        "information_boundary": {
            "allowed_post_intervention_prefix_frames": 6,
            "source_or_target_outcomes_used_for_selection": False,
            "confirmation_outcomes_used": False,
            "target_outcomes_may_select_method_or_hyperparameters": False,
        },
        "observation_path": {
            "prob4d": {
                "used": False,
                "package_compatibility_is_not_method_admission": True,
            }
        },
        "semantic_path": {"molmomotion_beta": 0},
    }
    candidate["candidate_sha256"] = _canonical_payload_sha256(
        candidate,
        omitted_field="candidate_sha256",
    )
    (root / ACQUISITION_CANDIDATE_PATH).write_text(
        json.dumps(candidate),
        encoding="utf-8",
    )
    (root / "pyproject.toml").write_text(
        "[project.optional-dependencies]\n"
        "phystwin = [\n"
        '  "bayesian-phystwin>=0.4,<0.5",\n'
        "]\n",
        encoding="utf-8",
    )
    return root


def test_freeze_binds_method_files_dependency_and_registered_calibration(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
        frozen_at_utc="2026-07-26T18:00:00+00:00",
    )
    result = validate_method_freeze_manifest(
        manifest,
        root,
        expected_causal4d_commit_sha=CAUSAL4D_SHA,
    )
    assert result["locked_files_checked"] == len(REQUIRED_LOCKED_PATHS)
    assert result["bayesian_phystwin_commit_sha"] == BPT_SHA
    assert result["prob4d_used"] is False
    assert (
        result["real_analysis_interval_amendment_id"]
        == expected_real_analysis_interval_amendment()["amendment_id"]
    )
    assert result["real_analysis_interval_amendment_sha256"]
    assert (
        result["acquisition_candidate_sha256"]
        == manifest["acquisition_candidate"]["candidate_sha256"]
    )
    assert (
        result["preacquisition_amendment_sha256"]
        == manifest["preacquisition"]["amendment_sha256"]
    )
    assert result["passed"]

    analysis = manifest["analysis_contract"]
    assert analysis["entrypoints"] == list(REQUIRED_ANALYSIS_ENTRYPOINTS)
    assert "causal4d calibration execution-block" in analysis["entrypoints"]
    assert "causal4d calibration real" not in analysis["entrypoints"]
    assert analysis["diagnostic_only_entrypoints"] == list(
        DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS
    )
    assert analysis["effect_interval"] == {
        "amendment_path": REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
        "amendment_id": expected_real_analysis_interval_amendment()["amendment_id"],
        "primary_method": "target_session_bootstrap_t",
        "required_robustness_method": "student_t_mean",
        "historical_sensitivity_method": "target_session_percentile_bootstrap",
        "positive_claim_requires_both_lower_bounds_positive": True,
        "robustness_may_rescue_primary_failure": False,
    }
    assert analysis["confirmatory_calibration"] == {
        "entrypoint": "causal4d calibration execution-block",
        "confidence_level": 0.90,
        "outer_fold_count": 12,
        "expected_calibration_units_per_outer_fold": 9,
        "order_statistic_rank_one_based": 9,
        "calibration_unit": "one preregistered execution per independent session",
        "score_kind": "max_abs_standardized_coordinate_v1",
        "target_threshold_reselection_allowed": False,
        "pooled_coordinate_conformal_claimed": False,
        "worst_group_coverage_guarantee_claimed": False,
    }


def test_freeze_rejects_file_drift_and_target_informed_selection(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    (root / "docs/causal4d_paper_scope.md").write_text("changed after freeze\n")
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_method_freeze_manifest(manifest, root)

    root = _repository(tmp_path / "second")
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    changed = deepcopy(manifest)
    changed["target_outcomes_observed_at_freeze"] = True
    with pytest.raises(ValueError, match="target outcomes"):
        validate_method_freeze_manifest(changed, root)

    changed = deepcopy(manifest)
    changed["analysis_contract"]["optional_branches_may_change_primary_analysis"] = True
    with pytest.raises(ValueError, match="analysis contract"):
        validate_method_freeze_manifest(changed, root)

    changed = deepcopy(manifest)
    changed["analysis_contract"]["confirmatory_calibration"]["entrypoint"] = (
        "causal4d calibration real"
    )
    with pytest.raises(ValueError, match="analysis contract"):
        validate_method_freeze_manifest(changed, root)


def test_freeze_rejects_preacquisition_or_gate_control_drift(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )

    amendment_path = root / PREACQUISITION_PATH
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["base_protocol"]["design_sha256"] = "c" * 64
    amendment["amendment_sha256"] = _canonical_payload_sha256(
        amendment,
        omitted_field="amendment_sha256",
    )
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(ValueError, match="different base protocol"):
        validate_method_freeze_manifest(manifest, root, verify_files=False)

    root = _repository(tmp_path / "gate")
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    evidence_path = root / MECHANISM_GATE_EVIDENCE_PATH
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["acceptance_checks"]["placebo_null_full_gate_upper_below_5_percent"] = (
        False
    )
    evidence["result_sha256"] = _canonical_payload_sha256(
        evidence,
        omitted_field="result_sha256",
    )
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="different gate-control evidence"):
        validate_method_freeze_manifest(manifest, root, verify_files=False)


def test_freeze_rejects_interval_amendment_or_evidence_drift(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )

    evidence_path = root / REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    evidence["physical_target_outcomes_used"] = True
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    with pytest.raises(ValueError, match="interval evidence"):
        validate_method_freeze_manifest(manifest, root, verify_files=False)

    root = _repository(tmp_path / "amendment")
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    amendment_path = root / REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH
    amendment = json.loads(amendment_path.read_text(encoding="utf-8"))
    amendment["required_robustness_interval"]["may_rescue_primary_failure"] = True
    amendment_path.write_text(json.dumps(amendment), encoding="utf-8")
    with pytest.raises(ValueError, match="interval amendment"):
        validate_method_freeze_manifest(manifest, root, verify_files=False)


def test_freeze_rejects_checkout_or_bpt_pin_mismatch(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    manifest = build_method_freeze_manifest(
        root,
        causal4d_commit_sha=CAUSAL4D_SHA,
        frozen_by="operator-1",
    )
    with pytest.raises(ValueError, match="checkout does not match"):
        validate_method_freeze_manifest(
            manifest,
            root,
            expected_causal4d_commit_sha="c" * 40,
        )

    bpt_pin = root / BPT_PIN_PATH
    bpt_pin.write_text("d" * 40 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Bayesian-PhysTwin pin"):
        validate_method_freeze_manifest(manifest, root, verify_files=False)


def test_checkout_validation_rejects_tracked_or_untracked_drift(
    tmp_path: Path,
) -> None:
    root = _repository(tmp_path)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=root,
        check=True,
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=root, check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze"], cwd=root, check=True)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = build_method_freeze_manifest(
        root, causal4d_commit_sha=commit, frozen_by="operator-1"
    )
    assert validate_repository_checkout(manifest, root)["commit_sha"] == commit

    (root / "untracked-analysis.py").write_text("print('drift')\n")
    with pytest.raises(ValueError, match="checkout is dirty"):
        validate_repository_checkout(manifest, root)


def test_method_freeze_publication_is_once_only(tmp_path: Path) -> None:
    target = tmp_path / "method_freeze.json"
    first = {"schema_version": 1, "value": "first"}
    second = {"schema_version": 1, "value": "second"}

    write_method_freeze_manifest(target, first)
    original = target.read_bytes()
    with pytest.raises(FileExistsError):
        write_method_freeze_manifest(target, second)

    assert target.read_bytes() == original
    assert json.loads(target.read_text(encoding="utf-8")) == first
