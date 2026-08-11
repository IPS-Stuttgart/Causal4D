"""Content-addressed registered analysis for the physical Causal4D protocol."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, cast

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json
from causal4d.execution_block_calibration import (
    EXECUTION_BLOCK_CALIBRATION_UNIT,
    EXECUTION_BLOCK_SCORE_KIND,
)
from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
    bind_repository_interval_amendment,
    expected_real_analysis_interval_amendment,
    validate_real_analysis_interval_amendment,
)
from causal4d.real_analysis_intervals import (
    REAL_EFFECT_BOOTSTRAP_REPLICATES,
    REAL_EFFECT_BOOTSTRAP_SEED,
    REAL_EFFECT_CONFIDENCE_LEVEL,
)
from causal4d.real_experiment_freeze import (
    DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS,
    MILESTONE_ID,
    REQUIRED_ANALYSIS_ENTRYPOINTS,
    SCHEMA_VERSION as METHOD_FREEZE_SCHEMA_VERSION,
    validate_method_freeze_manifest,
    validate_repository_checkout,
)
from causal4d.real_protocol import load_protocol, validate_protocol

REGISTERED_ANALYSIS_SCHEMA_VERSION: Final = 3
REGISTERED_ANALYSIS_ARTIFACT_KIND: Final = "Causal4DRegisteredRealAnalysisManifest"
REGISTERED_ANALYSIS_STATUS: Final = "sealed"
EXPECTED_PROTOCOL_ID: Final = "causal4d-sloth-multi-action-v1"
EXPECTED_PROTOCOL_DESIGN_SHA256: Final = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
EXPECTED_PREACQUISITION_SHA256: Final = (
    "0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f"
)
EXPECTED_OBJECT_ID: Final = "sloth_plush_instance_1"
BOOTSTRAP_REPLICATES: Final = REAL_EFFECT_BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED: Final = REAL_EFFECT_BOOTSTRAP_SEED
BOOTSTRAP_CONFIDENCE_LEVEL: Final = REAL_EFFECT_CONFIDENCE_LEVEL

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "analysis_id",
        "status",
        "registered_at_utc",
        "registered_by",
        "milestone_id",
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_amendment_sha256",
        "method_freeze_sha256",
        "interval_amendment",
        "software",
        "primary_analysis_locked",
        "locked_before_target_access",
        "target_outcomes_observed_at_registration",
        "target_outcomes_may_select_method_or_hyperparameters",
        "optional_branches_may_change_primary_analysis",
        "information_boundary",
        "command_contract",
        "comparison_arms",
        "effect_reporting",
        "confirmatory_calibration",
        "failure_and_exclusion_policy",
        "reporting_contract",
        "claim_boundary",
    }
)

_INFORMATION_BOUNDARY = {
    "allowed_observation_prefix_frames": 6,
    "future_frames_read_before_prediction": 0,
    "target_outcomes_available_for_selection": False,
    "individual_real_counterfactual_ground_truth_claimed": False,
    "real_evaluation_semantics": (
        "held_out_interventional_prediction_from_matched_initial_conditions"
    ),
}
_COMMAND_CONTRACT = {
    "primary_entrypoints": list(REQUIRED_ANALYSIS_ENTRYPOINTS),
    "diagnostic_only_entrypoints": list(DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS),
}
_COMPARISON_ARMS = [
    {
        "arm_id": "nominal_phystwin",
        "role": "baseline",
        "twin_uncertainty": False,
        "realized_intervention_inference": False,
    },
    {
        "arm_id": "bayesian_phystwin_nominal_z",
        "role": "baseline",
        "twin_uncertainty": True,
        "realized_intervention_inference": False,
    },
    {
        "arm_id": "frozen_causal4d",
        "role": "primary_candidate",
        "twin_uncertainty": True,
        "realized_intervention_inference": True,
    },
    {
        "arm_id": "causal4d_map_joint_component",
        "role": "diagnostic_only",
        "twin_uncertainty": "single_joint_component",
        "realized_intervention_inference": "joint_map",
    },
    {
        "arm_id": "causal4d_z_with_prior_twin",
        "role": "diagnostic_only",
        "twin_uncertainty": "prior_parameter_weights",
        "realized_intervention_inference": True,
    },
    {
        "arm_id": "intervention_oracle",
        "role": "diagnostic_only",
        "twin_uncertainty": True,
        "realized_intervention_inference": "oracle",
    },
]
_EXPECTED_ENDPOINT_INVENTORY = [
    {
        "endpoint_id": "factual_continuation",
        "split_key": "factual_continuation",
        "registered_units": 36,
        "target_sessions": 18,
        "primary_estimate": "equal_target_session_mean_candidate_minus_baseline",
        "interval_method": "target_session_bootstrap_t",
    },
    {
        "endpoint_id": "same_grasp_transfer",
        "split_key": "same_grasp_intervention_prediction",
        "registered_units": 18,
        "target_sessions": 18,
        "primary_estimate": "equal_target_session_mean_candidate_minus_baseline",
        "interval_method": "target_session_bootstrap_t",
    },
    {
        "endpoint_id": "new_contact_transfer",
        "split_key": "new_contact_intervention_prediction",
        "registered_units": 12,
        "target_sessions": 12,
        "primary_estimate": "equal_target_session_mean_candidate_minus_baseline",
        "interval_method": "target_session_bootstrap_t",
    },
]
_EFFECT_REPORTING = {
    "resampling_unit": "target_grasp_session",
    "equal_session_weighting": True,
    "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    "bootstrap_seed": BOOTSTRAP_SEED,
    "confidence_level": BOOTSTRAP_CONFIDENCE_LEVEL,
    "primary_interval_method": "target_session_bootstrap_t",
    "required_robustness_interval_method": "student_t_mean",
    "historical_sensitivity_interval_method": ("target_session_percentile_bootstrap"),
    "positive_claim_interval_rule": (
        "primary_and_required_robustness_lower_bounds_strictly_positive"
    ),
    "robustness_interval_may_rescue_primary_failure": False,
    "execution_mean_role": "diagnostic_only",
    "endpoint_inventory": _EXPECTED_ENDPOINT_INVENTORY,
}
_CONFIRMATORY_CALIBRATION = {
    "entrypoint": "causal4d calibration execution-block",
    "confidence_level": 0.90,
    "outer_fold_count": 12,
    "expected_calibration_units_per_outer_fold": 9,
    "order_statistic_rank_one_based": 9,
    "calibration_unit": EXECUTION_BLOCK_CALIBRATION_UNIT,
    "score_kind": EXECUTION_BLOCK_SCORE_KIND,
    "target_threshold_reselection_allowed": False,
    "pooled_coordinate_conformal_claimed": False,
    "worst_group_coverage_guarantee_claimed": False,
}
_FAILURE_AND_EXCLUSION_POLICY = {
    "all_registered_units_accounted_for": True,
    "technical_failures_retained": True,
    "excluded_units_retain_reason_without_target_metric_values": True,
    "silent_replacement_forbidden": True,
    "target_outcomes_may_not_select_exclusions": True,
}
_REPORTING_CONTRACT = {
    "report_success_or_well_powered_negative_result": True,
    "report_all_36_executions_or_preregistered_exclusions": True,
    "report_independent_execution_calibration": True,
    "report_effect_intervals_and_replay_reset_variance": True,
    "positive_claim_requires_primary_and_robustness_intervals": True,
    "historical_percentile_interval_is_sensitivity_only": True,
    "optional_semantic_or_public_data_results_cannot_rescue_primary_failure": True,
    "factual_same_grasp_new_contact_and_calibration_reported_separately": True,
    "calibration_cannot_rescue_failed_prediction_gates": True,
}
_CLAIM_BOUNDARY = {
    "object_id": EXPECTED_OBJECT_ID,
    "physical_object_count": 1,
    "registered_contact_region_count": 3,
    "registered_action_profile_count": 4,
    "object_class_generalization_claimed": False,
    "raw_covariance_calibration_claimed": False,
    "general_robot_safety_claimed": False,
}
_ENDPOINT_SPECS = (
    ("factual_continuation", "factual_continuation", "execution_id"),
    (
        "same_grasp_transfer",
        "same_grasp_intervention_prediction",
        "target_execution_id",
    ),
    (
        "new_contact_transfer",
        "new_contact_intervention_prediction",
        "target_execution_id",
    ),
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    _require(all(type(key) is str for key in value), f"{name} keys must be strings")
    return cast(Mapping[str, Any], value)


def _nonempty_string(value: Any, *, name: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{name} is missing")
    return value


def _sha(value: Any, *, name: str, length: int) -> str:
    _require(
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase {length}-hex digest",
    )
    return value


def _interval_amendment_binding(value: Any) -> dict[str, Any]:
    binding = _mapping(value, name="interval amendment binding")
    expected_fields = {
        "repository_path",
        "amendment_id",
        "sha256",
        "bytes",
        "contract",
        "operating_characteristic_evidence",
    }
    _require_equal(
        set(binding),
        expected_fields,
        name="interval amendment binding fields",
    )
    _require_equal(
        binding.get("repository_path"),
        REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
        name="interval amendment path",
    )
    contract = validate_real_analysis_interval_amendment(
        _mapping(binding.get("contract"), name="interval amendment contract")
    )
    amendment_id = _sha(
        binding.get("amendment_id"),
        name="interval amendment ID",
        length=64,
    )
    _require_equal(
        amendment_id,
        contract["amendment_id"],
        name="interval amendment ID",
    )
    digest = _sha(
        binding.get("sha256"),
        name="interval amendment SHA-256",
        length=64,
    )
    byte_count = binding.get("bytes")
    _require(
        type(byte_count) is int and byte_count > 0,
        "interval amendment byte count must be a positive integer",
    )
    evidence = _mapping(
        binding.get("operating_characteristic_evidence"),
        name="interval operating-characteristic evidence binding",
    )
    _require_equal(
        set(evidence),
        {"repository_path", "result_sha256", "sha256", "bytes"},
        name="interval evidence binding fields",
    )
    _require_equal(
        evidence.get("repository_path"),
        REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
        name="interval evidence path",
    )
    evidence_result = _sha(
        evidence.get("result_sha256"),
        name="interval evidence result SHA-256",
        length=64,
    )
    _require_equal(
        evidence_result,
        contract["operating_characteristic_evidence"]["result_sha256"],
        name="interval evidence result SHA-256",
    )
    evidence_digest = _sha(
        evidence.get("sha256"),
        name="interval evidence file SHA-256",
        length=64,
    )
    evidence_bytes = evidence.get("bytes")
    _require(
        type(evidence_bytes) is int and evidence_bytes > 0,
        "interval evidence byte count must be a positive integer",
    )
    return {
        "repository_path": REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
        "amendment_id": amendment_id,
        "sha256": digest,
        "bytes": byte_count,
        "contract": contract,
        "operating_characteristic_evidence": {
            "repository_path": REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
            "result_sha256": evidence_result,
            "sha256": evidence_digest,
            "bytes": evidence_bytes,
        },
    }


def _utc_timestamp(value: Any, *, name: str) -> str:
    text = _nonempty_string(value, name=name)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError(f"{name} is not ISO 8601") from error
    _require(parsed.tzinfo is not None, f"{name} must include a timezone")
    _require(
        parsed.utcoffset() == timezone.utc.utcoffset(parsed),
        f"{name} must use UTC",
    )
    return text


def _canonical_sha256(payload: Mapping[str, Any], *, omitted: str) -> str:
    values = dict(payload)
    values.pop(omitted, None)
    encoded = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def analysis_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the content identity of a registered-analysis manifest."""

    return _canonical_sha256(payload, omitted="analysis_id")


def _require_equal(actual: Any, expected: Any, *, name: str) -> None:
    _require(actual == expected, f"registered analysis {name} changed")


def _endpoint_inventory(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    executions = protocol.get("executions")
    _require(isinstance(executions, list), "protocol executions are missing")
    session_by_execution: dict[str, str] = {}
    for position, raw in enumerate(executions):
        execution = _mapping(raw, name=f"execution {position}")
        execution_id = _nonempty_string(
            execution.get("execution_id"),
            name=f"execution {position} id",
        )
        session_by_execution[execution_id] = _nonempty_string(
            execution.get("session_id"),
            name=f"execution {position} session",
        )
    splits = _mapping(protocol.get("splits"), name="protocol splits")
    inventory: list[dict[str, Any]] = []
    for endpoint_id, split_key, target_field in _ENDPOINT_SPECS:
        units = splits.get(split_key)
        _require(isinstance(units, list), f"{split_key} split is missing")
        targets = [
            _nonempty_string(
                _mapping(unit, name=f"{split_key} unit").get(target_field),
                name=f"{split_key} target",
            )
            for unit in units
        ]
        _require(len(targets) == len(set(targets)), f"{split_key} targets duplicate")
        _require(
            all(target in session_by_execution for target in targets),
            f"{split_key} contains an unknown target",
        )
        inventory.append(
            {
                "endpoint_id": endpoint_id,
                "split_key": split_key,
                "registered_units": len(targets),
                "target_sessions": len(
                    {session_by_execution[target] for target in targets}
                ),
                "primary_estimate": (
                    "equal_target_session_mean_candidate_minus_baseline"
                ),
                "interval_method": "target_session_bootstrap_t",
            }
        )
    _require_equal(inventory, _EXPECTED_ENDPOINT_INVENTORY, name="endpoint inventory")
    return inventory


def _validate_freeze_contract(
    freeze: Mapping[str, Any],
    *,
    interval_amendment_binding: Mapping[str, Any],
) -> Mapping[str, Any]:
    _require_equal(
        freeze.get("schema_version"),
        METHOD_FREEZE_SCHEMA_VERSION,
        name="method-freeze schema",
    )
    _require_equal(freeze.get("milestone_id"), MILESTONE_ID, name="milestone")
    _require_equal(freeze.get("status"), "sealed", name="method-freeze status")
    _require_equal(
        _interval_amendment_binding(freeze.get("interval_amendment")),
        _interval_amendment_binding(interval_amendment_binding),
        name="method freeze interval amendment",
    )
    analysis = _mapping(freeze.get("analysis_contract"), name="analysis contract")
    _require_equal(
        analysis.get("entrypoints"),
        list(REQUIRED_ANALYSIS_ENTRYPOINTS),
        name="primary entrypoints",
    )
    _require_equal(
        analysis.get("diagnostic_only_entrypoints"),
        list(DIAGNOSTIC_ONLY_ANALYSIS_ENTRYPOINTS),
        name="diagnostic entrypoints",
    )
    _require_equal(
        analysis.get("allowed_observation_prefix_frames"),
        6,
        name="observation prefix",
    )
    interval_amendment = validate_real_analysis_interval_amendment(
        _mapping(
            expected_real_analysis_interval_amendment(),
            name="expected interval amendment",
        )
    )
    _require_equal(
        analysis.get("effect_interval"),
        {
            "amendment_path": (REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH),
            "amendment_id": interval_amendment["amendment_id"],
            "primary_method": "target_session_bootstrap_t",
            "required_robustness_method": "student_t_mean",
            "historical_sensitivity_method": ("target_session_percentile_bootstrap"),
            "positive_claim_requires_both_lower_bounds_positive": True,
            "robustness_may_rescue_primary_failure": False,
        },
        name="effect interval",
    )
    _require_equal(
        analysis.get("confirmatory_calibration"),
        _CONFIRMATORY_CALIBRATION,
        name="confirmatory calibration",
    )
    _require_equal(
        analysis.get("target_outcomes_may_select_method_or_hyperparameters"),
        False,
        name="target-selection boundary",
    )
    _require_equal(
        analysis.get("optional_branches_may_change_primary_analysis"),
        False,
        name="optional-branch boundary",
    )
    return analysis


def build_registered_real_analysis_manifest(
    protocol: Mapping[str, Any],
    method_freeze: Mapping[str, Any],
    *,
    method_freeze_sha256: str,
    interval_amendment_binding: Mapping[str, Any],
    registered_by: str,
    registered_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build the complete primary-analysis contract from locked sources."""

    validate_protocol(protocol)
    _require_equal(protocol.get("protocol_id"), EXPECTED_PROTOCOL_ID, name="protocol")
    _require_equal(
        protocol.get("design_sha256"),
        EXPECTED_PROTOCOL_DESIGN_SHA256,
        name="protocol digest",
    )
    object_record = _mapping(protocol.get("object"), name="protocol object")
    _require_equal(object_record.get("object_id"), EXPECTED_OBJECT_ID, name="object")
    _sha(method_freeze_sha256, name="method freeze SHA-256", length=64)
    interval_amendment = _interval_amendment_binding(interval_amendment_binding)
    _validate_freeze_contract(
        method_freeze,
        interval_amendment_binding=interval_amendment,
    )
    protocol_record = _mapping(method_freeze.get("protocol"), name="freeze protocol")
    preacquisition = _mapping(
        method_freeze.get("preacquisition"),
        name="freeze preacquisition",
    )
    _require_equal(
        protocol_record.get("design_sha256"),
        EXPECTED_PROTOCOL_DESIGN_SHA256,
        name="freeze protocol digest",
    )
    _require_equal(
        preacquisition.get("amendment_sha256"),
        EXPECTED_PREACQUISITION_SHA256,
        name="preacquisition digest",
    )
    causal4d = _mapping(method_freeze.get("causal4d"), name="freeze Causal4D")
    bpt = _mapping(method_freeze.get("bayesian_phystwin"), name="freeze BPT")
    reporting = _mapping(
        method_freeze.get("reporting_contract"),
        name="reporting contract",
    )
    expected_freeze_reporting = dict(_REPORTING_CONTRACT)
    expected_freeze_reporting.pop(
        "factual_same_grasp_new_contact_and_calibration_reported_separately"
    )
    expected_freeze_reporting.pop("calibration_cannot_rescue_failed_prediction_gates")
    _require_equal(
        dict(reporting),
        expected_freeze_reporting,
        name="freeze reporting contract",
    )

    manifest: dict[str, Any] = {
        "schema_version": REGISTERED_ANALYSIS_SCHEMA_VERSION,
        "artifact_kind": REGISTERED_ANALYSIS_ARTIFACT_KIND,
        "analysis_id": "",
        "status": REGISTERED_ANALYSIS_STATUS,
        "registered_at_utc": _utc_timestamp(
            registered_at_utc or datetime.now(timezone.utc).isoformat(),
            name="registration timestamp",
        ),
        "registered_by": _nonempty_string(
            registered_by,
            name="registered_by",
        ).strip(),
        "milestone_id": MILESTONE_ID,
        "protocol_id": EXPECTED_PROTOCOL_ID,
        "protocol_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
        "preacquisition_amendment_sha256": EXPECTED_PREACQUISITION_SHA256,
        "method_freeze_sha256": method_freeze_sha256,
        "interval_amendment": interval_amendment,
        "software": {
            "causal4d_commit_sha": _sha(
                causal4d.get("commit_sha"),
                name="Causal4D commit",
                length=40,
            ),
            "bayesian_phystwin_commit_sha": _sha(
                bpt.get("commit_sha"),
                name="BayesianPhysTwin commit",
                length=40,
            ),
            "observation_provider_bound_by_software_environment_gate": True,
            "prob4d_may_change_primary_analysis": False,
        },
        "primary_analysis_locked": True,
        "locked_before_target_access": True,
        "target_outcomes_observed_at_registration": False,
        "target_outcomes_may_select_method_or_hyperparameters": False,
        "optional_branches_may_change_primary_analysis": False,
        "information_boundary": dict(_INFORMATION_BOUNDARY),
        "command_contract": dict(_COMMAND_CONTRACT),
        "comparison_arms": [dict(arm) for arm in _COMPARISON_ARMS],
        "effect_reporting": {
            **dict(_EFFECT_REPORTING),
            "endpoint_inventory": _endpoint_inventory(protocol),
        },
        "confirmatory_calibration": dict(_CONFIRMATORY_CALIBRATION),
        "failure_and_exclusion_policy": dict(_FAILURE_AND_EXCLUSION_POLICY),
        "reporting_contract": dict(_REPORTING_CONTRACT),
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }
    manifest["analysis_id"] = analysis_id_for_payload(manifest)
    return validate_registered_real_analysis_manifest(manifest)


def validate_registered_real_analysis_manifest(
    payload: Mapping[str, Any],
    *,
    expected_protocol_id: str = EXPECTED_PROTOCOL_ID,
    expected_protocol_design_sha256: str = EXPECTED_PROTOCOL_DESIGN_SHA256,
    expected_preacquisition_amendment_sha256: str = EXPECTED_PREACQUISITION_SHA256,
    expected_method_freeze_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate a closed, self-contained registered-analysis manifest."""

    values = _mapping(payload, name="registered analysis manifest")
    _require_equal(set(values), set(_TOP_LEVEL_FIELDS), name="top-level fields")
    for name, expected in (
        ("schema_version", REGISTERED_ANALYSIS_SCHEMA_VERSION),
        ("artifact_kind", REGISTERED_ANALYSIS_ARTIFACT_KIND),
        ("status", REGISTERED_ANALYSIS_STATUS),
        ("milestone_id", MILESTONE_ID),
        ("protocol_id", expected_protocol_id),
        ("protocol_design_sha256", expected_protocol_design_sha256),
        (
            "preacquisition_amendment_sha256",
            expected_preacquisition_amendment_sha256,
        ),
        ("primary_analysis_locked", True),
        ("locked_before_target_access", True),
        ("target_outcomes_observed_at_registration", False),
        ("target_outcomes_may_select_method_or_hyperparameters", False),
        ("optional_branches_may_change_primary_analysis", False),
    ):
        _require_equal(values[name], expected, name=name)
    _utc_timestamp(values["registered_at_utc"], name="registered_at_utc")
    _nonempty_string(values["registered_by"], name="registered_by")
    method_sha = _sha(
        values["method_freeze_sha256"],
        name="method_freeze_sha256",
        length=64,
    )
    if expected_method_freeze_sha256 is not None:
        _require_equal(method_sha, expected_method_freeze_sha256, name="freeze digest")
    _interval_amendment_binding(values["interval_amendment"])

    software = _mapping(values["software"], name="software")
    _require_equal(
        set(software),
        {
            "causal4d_commit_sha",
            "bayesian_phystwin_commit_sha",
            "observation_provider_bound_by_software_environment_gate",
            "prob4d_may_change_primary_analysis",
        },
        name="software fields",
    )
    _sha(software["causal4d_commit_sha"], name="Causal4D commit", length=40)
    _sha(
        software["bayesian_phystwin_commit_sha"],
        name="BayesianPhysTwin commit",
        length=40,
    )
    _require_equal(
        software["observation_provider_bound_by_software_environment_gate"],
        True,
        name="observation provider gate",
    )
    _require_equal(
        software["prob4d_may_change_primary_analysis"],
        False,
        name="Prob4D boundary",
    )
    for name, expected in (
        ("information_boundary", _INFORMATION_BOUNDARY),
        ("command_contract", _COMMAND_CONTRACT),
        ("comparison_arms", _COMPARISON_ARMS),
        ("effect_reporting", _EFFECT_REPORTING),
        ("confirmatory_calibration", _CONFIRMATORY_CALIBRATION),
        ("failure_and_exclusion_policy", _FAILURE_AND_EXCLUSION_POLICY),
        ("reporting_contract", _REPORTING_CONTRACT),
        ("claim_boundary", _CLAIM_BOUNDARY),
    ):
        _require_equal(values[name], expected, name=name)
    analysis_id = _sha(values["analysis_id"], name="analysis_id", length=64)
    _require_equal(
        analysis_id,
        analysis_id_for_payload(values),
        name="analysis_id",
    )
    return dict(values)


def load_registered_real_analysis_manifest(
    path: str | Path,
) -> tuple[dict[str, Any], str, int]:
    """Load and validate one exact-byte registered-analysis file."""

    snapshot = read_regular_file(path, name="registered analysis manifest")
    payload = load_strict_json_object(
        snapshot.payload,
        name="registered analysis manifest",
    )
    return (
        validate_registered_real_analysis_manifest(payload),
        snapshot.sha256,
        snapshot.byte_count,
    )


def write_registered_real_analysis_manifest(
    path: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Publish a registered-analysis manifest atomically and exactly once."""

    output = Path(path)
    atomic_write_json(
        output,
        validate_registered_real_analysis_manifest(manifest),
        overwrite=False,
    )
    return output


def seal_registered_real_analysis_manifest(
    repository_root: str | Path,
    protocol_path: str | Path,
    method_freeze_path: str | Path,
    output_path: str | Path,
    *,
    registered_by: str,
    registered_at_utc: str | None = None,
) -> dict[str, Any]:
    """Validate frozen sources and publish their complete analysis contract."""

    root = Path(repository_root)
    protocol = load_protocol(protocol_path)
    freeze_snapshot = read_regular_file(method_freeze_path, name="method freeze")
    freeze = load_strict_json_object(freeze_snapshot.payload, name="method freeze")
    validate_method_freeze_manifest(freeze, root, verify_files=True)
    validate_repository_checkout(freeze, root)
    interval_amendment = bind_repository_interval_amendment(root)
    manifest = build_registered_real_analysis_manifest(
        protocol,
        freeze,
        method_freeze_sha256=freeze_snapshot.sha256,
        interval_amendment_binding=interval_amendment,
        registered_by=registered_by,
        registered_at_utc=registered_at_utc,
    )
    output = write_registered_real_analysis_manifest(output_path, manifest)
    snapshot = read_regular_file(output, name="registered analysis manifest")
    return {
        "passed": True,
        "analysis_id": manifest["analysis_id"],
        "path": str(output.resolve()),
        "sha256": snapshot.sha256,
        "bytes": snapshot.byte_count,
        "method_freeze_sha256": freeze_snapshot.sha256,
        "interval_amendment_id": interval_amendment["amendment_id"],
        "interval_amendment_sha256": interval_amendment["sha256"],
        "target_outcomes_used": False,
    }


def validate_registered_real_analysis_sources(
    repository_root: str | Path,
    protocol_path: str | Path,
    method_freeze_path: str | Path,
    analysis_manifest_path: str | Path,
) -> dict[str, Any]:
    """Reopen and validate the exact protocol, freeze, and analysis bytes."""

    root = Path(repository_root)
    protocol = load_protocol(protocol_path)
    validate_protocol(protocol)
    freeze_snapshot = read_regular_file(method_freeze_path, name="method freeze")
    freeze = load_strict_json_object(freeze_snapshot.payload, name="method freeze")
    validate_method_freeze_manifest(freeze, root, verify_files=True)
    validate_repository_checkout(freeze, root)
    manifest, manifest_sha, manifest_bytes = load_registered_real_analysis_manifest(
        analysis_manifest_path
    )
    validate_registered_real_analysis_manifest(
        manifest,
        expected_protocol_id=str(protocol["protocol_id"]),
        expected_protocol_design_sha256=str(protocol["design_sha256"]),
        expected_preacquisition_amendment_sha256=str(
            freeze["preacquisition"]["amendment_sha256"]
        ),
        expected_method_freeze_sha256=freeze_snapshot.sha256,
    )
    interval_amendment = bind_repository_interval_amendment(root)
    _require_equal(
        manifest["interval_amendment"],
        interval_amendment,
        name="interval amendment binding",
    )
    _require_equal(
        manifest["software"]["causal4d_commit_sha"],
        freeze["causal4d"]["commit_sha"],
        name="Causal4D commit",
    )
    _require_equal(
        manifest["software"]["bayesian_phystwin_commit_sha"],
        freeze["bayesian_phystwin"]["commit_sha"],
        name="BayesianPhysTwin commit",
    )
    return {
        "passed": True,
        "analysis_id": manifest["analysis_id"],
        "analysis_manifest_sha256": manifest_sha,
        "analysis_manifest_bytes": manifest_bytes,
        "method_freeze_sha256": freeze_snapshot.sha256,
        "interval_amendment_id": interval_amendment["amendment_id"],
        "interval_amendment_sha256": interval_amendment["sha256"],
        "target_outcomes_used": False,
    }


__all__ = [
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EXPECTED_OBJECT_ID",
    "EXPECTED_PREACQUISITION_SHA256",
    "EXPECTED_PROTOCOL_DESIGN_SHA256",
    "EXPECTED_PROTOCOL_ID",
    "REGISTERED_ANALYSIS_ARTIFACT_KIND",
    "REGISTERED_ANALYSIS_SCHEMA_VERSION",
    "analysis_id_for_payload",
    "build_registered_real_analysis_manifest",
    "load_registered_real_analysis_manifest",
    "seal_registered_real_analysis_manifest",
    "validate_registered_real_analysis_manifest",
    "validate_registered_real_analysis_sources",
    "write_registered_real_analysis_manifest",
]
