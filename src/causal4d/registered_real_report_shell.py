"""Deterministic, target-free shell for the registered real-analysis report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, cast

from causal4d.real_analysis_interval_amendment import (
    REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
    REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
    validate_real_analysis_interval_amendment,
)

REPORT_SHELL_SCHEMA_VERSION: Final = 2
REPORT_SHELL_ARTIFACT_KIND: Final = "Causal4DRegisteredRealReportShell"
REPORT_SHELL_STATUS: Final = "target-free-template"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "shell_id",
        "status",
        "source",
        "safety_boundary",
        "analysis_contract",
        "table_plan",
        "figure_plan",
        "narrative_plan",
        "completion_checks",
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "analysis_id",
        "analysis_manifest_sha256",
        "analysis_manifest_bytes",
        "method_freeze_sha256",
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_amendment_sha256",
        "software",
    }
)
_CONTRACT_FIELDS = frozenset(
    {
        "interval_amendment",
        "comparison_arms",
        "effect_reporting",
        "confirmatory_calibration",
        "failure_and_exclusion_policy",
        "reporting_contract",
        "claim_boundary",
    }
)
_SAFETY_BOUNDARY = {
    "target_outcomes_loaded": False,
    "target_metric_values_loaded": False,
    "confirmatory_execution_evidence_count": 0,
    "may_select_method_or_hyperparameters": False,
    "manual_table_or_figure_selection_allowed": False,
    "derived_artifact_is_claim_bearing": False,
    "report_shell_is_a_scientific_result": False,
}
_ENDPOINT_COLUMNS = (
    "arm_id",
    "registered_units",
    "valid_units",
    "excluded_units",
    "equal_target_session_mean",
    "candidate_minus_baseline",
    "primary_interval_lower",
    "primary_interval_upper",
    "primary_interval_method",
    "required_robustness_interval_lower",
    "required_robustness_interval_upper",
    "required_robustness_interval_method",
    "positive_claim_interval_gate_passed",
    "historical_percentile_interval_lower",
    "historical_percentile_interval_upper",
    "replay_reset_variance",
)
_COMPLETION_CHECKS = (
    "all_registered_units_or_preregistered_exclusions_are_accounted_for",
    "factual_same_grasp_new_contact_and_calibration_are_reported_separately",
    "candidate_effects_use_equal_target_session_weighting",
    "effect_intervals_use_the_registered_resampling_unit_and_seed",
    "bootstrap_t_is_the_registered_primary_effect_interval",
    "student_t_robustness_may_veto_but_never_rescue_a_positive_claim",
    "historical_percentile_interval_remains_sensitivity_only",
    "technical_failures_and_exclusion_reasons_are_retained",
    "coverage_interval_width_and_finite_calibration_sensitivity_are_reported",
    "intervention_oracle_remains_diagnostic_only",
    "map_and_prior_twin_abduction_arms_remain_diagnostic_only",
    "positive_and_bounded_negative_narratives_are_both_renderable",
    "optional_branches_cannot_rescue_a_primary_failure",
    "claim_language_remains_inside_the_registered_boundary",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    _require(all(type(key) is str for key in value), f"{name} keys must be strings")
    return cast(Mapping[str, Any], value)


def _sequence(value: Any, *, name: str) -> Sequence[Any]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{name} must be a JSON array",
    )
    return cast(Sequence[Any], value)


def _nonempty_string(value: Any, *, name: str) -> str:
    _require(type(value) is str and bool(value.strip()), f"{name} is missing")
    return value.strip()


def _hex_digest(value: Any, *, name: str, length: int) -> str:
    text = _nonempty_string(value, name=name)
    _require(
        len(text) == length
        and all(character in "0123456789abcdef" for character in text),
        f"{name} must be a lowercase {length}-hex digest",
    )
    return text


def _positive_int(value: Any, *, name: str) -> int:
    _require(type(value) is int and value > 0, f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, *, name: str) -> int:
    _require(
        type(value) is int and value >= 0,
        f"{name} must be a nonnegative integer",
    )
    return value


def _json_copy(value: Any, *, name: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be finite JSON") from error


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


def report_shell_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Return the canonical content identity of a report shell."""

    return _canonical_sha256(payload, omitted="shell_id")


def _normalize_software(value: Any) -> dict[str, Any]:
    software = _mapping(value, name="analysis software")
    required = {
        "causal4d_commit_sha",
        "bayesian_phystwin_commit_sha",
        "observation_provider_bound_by_software_environment_gate",
        "prob4d_may_change_primary_analysis",
    }
    _require(set(software) == required, "analysis software fields changed")
    _require(
        software["observation_provider_bound_by_software_environment_gate"] is True,
        "observation provider must be bound by the software-environment gate",
    )
    _require(
        software["prob4d_may_change_primary_analysis"] is False,
        "Prob4D may not change the registered primary analysis",
    )
    return {
        "causal4d_commit_sha": _hex_digest(
            software["causal4d_commit_sha"],
            name="Causal4D commit",
            length=40,
        ),
        "bayesian_phystwin_commit_sha": _hex_digest(
            software["bayesian_phystwin_commit_sha"],
            name="BayesianPhysTwin commit",
            length=40,
        ),
        "observation_provider_bound_by_software_environment_gate": True,
        "prob4d_may_change_primary_analysis": False,
    }


def _normalize_interval_amendment(value: Any) -> dict[str, Any]:
    binding = _mapping(value, name="interval amendment binding")
    required = {
        "repository_path",
        "amendment_id",
        "sha256",
        "bytes",
        "contract",
        "operating_characteristic_evidence",
    }
    _require(set(binding) == required, "interval amendment binding fields changed")
    _require(
        binding.get("repository_path")
        == REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
        "interval amendment path changed",
    )
    contract = validate_real_analysis_interval_amendment(
        _mapping(binding.get("contract"), name="interval amendment contract")
    )
    amendment_id = _hex_digest(
        binding.get("amendment_id"),
        name="interval amendment ID",
        length=64,
    )
    _require(
        amendment_id == contract["amendment_id"],
        "interval amendment ID differs from its contract",
    )
    evidence = _mapping(
        binding.get("operating_characteristic_evidence"),
        name="interval operating-characteristic evidence",
    )
    _require(
        set(evidence) == {"repository_path", "result_sha256", "sha256", "bytes"},
        "interval evidence binding fields changed",
    )
    _require(
        evidence.get("repository_path")
        == REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
        "interval evidence path changed",
    )
    result_sha256 = _hex_digest(
        evidence.get("result_sha256"),
        name="interval evidence result SHA-256",
        length=64,
    )
    _require(
        result_sha256 == contract["operating_characteristic_evidence"]["result_sha256"],
        "interval amendment binds different operating-characteristic evidence",
    )
    return {
        "repository_path": REAL_ANALYSIS_INTERVAL_AMENDMENT_REPOSITORY_PATH,
        "amendment_id": amendment_id,
        "sha256": _hex_digest(
            binding.get("sha256"),
            name="interval amendment file SHA-256",
            length=64,
        ),
        "bytes": _positive_int(
            binding.get("bytes"),
            name="interval amendment byte count",
        ),
        "contract": contract,
        "operating_characteristic_evidence": {
            "repository_path": REAL_ANALYSIS_INTERVAL_EVIDENCE_REPOSITORY_PATH,
            "result_sha256": result_sha256,
            "sha256": _hex_digest(
                evidence.get("sha256"),
                name="interval evidence file SHA-256",
                length=64,
            ),
            "bytes": _positive_int(
                evidence.get("bytes"),
                name="interval evidence byte count",
            ),
        },
    }


def _normalize_comparison_arms(value: Any) -> list[dict[str, Any]]:
    raw_arms = _sequence(value, name="comparison arms")
    _require(len(raw_arms) >= 3, "comparison arms are incomplete")
    arms: list[dict[str, Any]] = []
    seen: set[str] = set()
    primary_candidates = 0
    diagnostic_arms = 0
    for index, raw in enumerate(raw_arms):
        arm = _mapping(raw, name=f"comparison arm {index}")
        arm_id = _nonempty_string(arm.get("arm_id"), name=f"arm {index} id")
        _require(arm_id not in seen, "comparison arm IDs must be unique")
        seen.add(arm_id)
        role = _nonempty_string(arm.get("role"), name=f"arm {arm_id} role")
        _require(
            role in {"baseline", "primary_candidate", "diagnostic_only"},
            f"arm {arm_id} has an unsupported role",
        )
        primary_candidates += int(role == "primary_candidate")
        diagnostic_arms += int(role == "diagnostic_only")
        normalized = _json_copy(dict(arm), name=f"comparison arm {arm_id}")
        normalized["arm_id"] = arm_id
        normalized["role"] = role
        arms.append(normalized)
    _require(primary_candidates == 1, "exactly one primary candidate is required")
    _require(diagnostic_arms >= 1, "a diagnostic-only arm is required")
    return arms


def _normalize_effect_reporting(value: Any) -> dict[str, Any]:
    reporting = _mapping(value, name="effect reporting")
    inventory = _sequence(
        reporting.get("endpoint_inventory"),
        name="endpoint inventory",
    )
    endpoints: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(inventory):
        endpoint = _mapping(raw, name=f"endpoint {index}")
        endpoint_id = _nonempty_string(
            endpoint.get("endpoint_id"),
            name=f"endpoint {index} id",
        )
        _require(endpoint_id not in seen, "endpoint IDs must be unique")
        seen.add(endpoint_id)
        registered_units = _positive_int(
            endpoint.get("registered_units"),
            name=f"endpoint {endpoint_id} registered units",
        )
        target_sessions = _positive_int(
            endpoint.get("target_sessions"),
            name=f"endpoint {endpoint_id} target sessions",
        )
        _require(
            target_sessions <= registered_units,
            f"endpoint {endpoint_id} has more sessions than units",
        )
        endpoints.append(
            {
                "endpoint_id": endpoint_id,
                "split_key": _nonempty_string(
                    endpoint.get("split_key"),
                    name=f"endpoint {endpoint_id} split key",
                ),
                "registered_units": registered_units,
                "target_sessions": target_sessions,
                "primary_estimate": _nonempty_string(
                    endpoint.get("primary_estimate"),
                    name=f"endpoint {endpoint_id} primary estimate",
                ),
                "interval_method": _nonempty_string(
                    endpoint.get("interval_method"),
                    name=f"endpoint {endpoint_id} interval method",
                ),
            }
        )
    _require(endpoints, "endpoint inventory must not be empty")
    normalized = _json_copy(dict(reporting), name="effect reporting")
    normalized["endpoint_inventory"] = endpoints
    _require(
        normalized.get("equal_session_weighting") is True,
        "equal target-session weighting is required",
    )
    _positive_int(
        normalized.get("bootstrap_replicates"),
        name="bootstrap replicate count",
    )
    _nonnegative_int(normalized.get("bootstrap_seed"), name="bootstrap seed")
    confidence = normalized.get("confidence_level")
    _require(
        type(confidence) in {int, float} and 0.0 < float(confidence) < 1.0,
        "bootstrap confidence level must be in (0, 1)",
    )
    _require(
        normalized.get("primary_interval_method") == "target_session_bootstrap_t",
        "registered primary interval method changed",
    )
    _require(
        normalized.get("required_robustness_interval_method") == "student_t_mean",
        "registered robustness interval method changed",
    )
    _require(
        normalized.get("historical_sensitivity_interval_method")
        == "target_session_percentile_bootstrap",
        "historical interval method changed",
    )
    _require(
        normalized.get("positive_claim_interval_rule")
        == "primary_and_required_robustness_lower_bounds_strictly_positive",
        "positive-claim interval rule changed",
    )
    _require(
        normalized.get("robustness_interval_may_rescue_primary_failure") is False,
        "robustness interval may not rescue a primary failure",
    )
    return normalized


def _normalize_contract(analysis: Mapping[str, Any]) -> dict[str, Any]:
    for name, expected in (
        ("primary_analysis_locked", True),
        ("locked_before_target_access", True),
        ("target_outcomes_observed_at_registration", False),
        ("target_outcomes_may_select_method_or_hyperparameters", False),
        ("optional_branches_may_change_primary_analysis", False),
    ):
        _require(analysis.get(name) is expected, f"registered analysis {name} changed")

    contract = {
        "interval_amendment": _normalize_interval_amendment(
            analysis.get("interval_amendment")
        ),
        "comparison_arms": _normalize_comparison_arms(analysis.get("comparison_arms")),
        "effect_reporting": _normalize_effect_reporting(
            analysis.get("effect_reporting")
        ),
        "confirmatory_calibration": _json_copy(
            dict(
                _mapping(
                    analysis.get("confirmatory_calibration"),
                    name="confirmatory calibration",
                )
            ),
            name="confirmatory calibration",
        ),
        "failure_and_exclusion_policy": _json_copy(
            dict(
                _mapping(
                    analysis.get("failure_and_exclusion_policy"),
                    name="failure and exclusion policy",
                )
            ),
            name="failure and exclusion policy",
        ),
        "reporting_contract": _json_copy(
            dict(
                _mapping(
                    analysis.get("reporting_contract"),
                    name="reporting contract",
                )
            ),
            name="reporting contract",
        ),
        "claim_boundary": _json_copy(
            dict(_mapping(analysis.get("claim_boundary"), name="claim boundary")),
            name="claim boundary",
        ),
    }
    exclusion = cast(dict[str, Any], contract["failure_and_exclusion_policy"])
    reporting = cast(dict[str, Any], contract["reporting_contract"])
    _require(
        exclusion.get("all_registered_units_accounted_for") is True,
        "all registered units must be accounted for",
    )
    _require(
        exclusion.get("technical_failures_retained") is True,
        "technical failures must be retained",
    )
    _require(
        reporting.get("report_success_or_well_powered_negative_result") is True,
        "both positive and bounded-negative reporting paths are required",
    )
    _require(
        reporting.get(
            "optional_semantic_or_public_data_results_cannot_rescue_primary_failure"
        )
        is True,
        "optional results may not rescue a primary failure",
    )
    _require(
        reporting.get("positive_claim_requires_primary_and_robustness_intervals")
        is True,
        "positive claims require both registered intervals",
    )
    _require(
        reporting.get("historical_percentile_interval_is_sensitivity_only") is True,
        "historical percentile interval must remain sensitivity-only",
    )
    return contract


def _source_from_analysis(
    analysis: Mapping[str, Any],
    *,
    analysis_manifest_sha256: str,
    analysis_manifest_byte_count: int,
) -> dict[str, Any]:
    return {
        "analysis_id": _hex_digest(
            analysis.get("analysis_id"),
            name="analysis ID",
            length=64,
        ),
        "analysis_manifest_sha256": _hex_digest(
            analysis_manifest_sha256,
            name="analysis manifest SHA-256",
            length=64,
        ),
        "analysis_manifest_bytes": _positive_int(
            analysis_manifest_byte_count,
            name="analysis manifest byte count",
        ),
        "method_freeze_sha256": _hex_digest(
            analysis.get("method_freeze_sha256"),
            name="method freeze SHA-256",
            length=64,
        ),
        "protocol_id": _nonempty_string(
            analysis.get("protocol_id"),
            name="protocol ID",
        ),
        "protocol_design_sha256": _hex_digest(
            analysis.get("protocol_design_sha256"),
            name="protocol design SHA-256",
            length=64,
        ),
        "preacquisition_amendment_sha256": _hex_digest(
            analysis.get("preacquisition_amendment_sha256"),
            name="pre-acquisition amendment SHA-256",
            length=64,
        ),
        "software": _normalize_software(analysis.get("software")),
    }


def _table_plan(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    effect_reporting = _mapping(contract["effect_reporting"], name="effect reporting")
    endpoints = _sequence(
        effect_reporting["endpoint_inventory"],
        name="endpoint inventory",
    )
    tables: list[dict[str, Any]] = [
        {
            "table_id": "execution-accounting",
            "title": "Registered execution and exclusion accounting",
            "required_columns": [
                "execution_id",
                "session_id",
                "registered_condition",
                "completion_state",
                "included",
                "preregistered_exclusion_reason",
                "technical_failure_reason",
            ],
            "rows": [],
            "row_source": "validated confirmatory evidence registry",
        }
    ]
    for raw in endpoints:
        endpoint = _mapping(raw, name="endpoint")
        endpoint_id = cast(str, endpoint["endpoint_id"])
        tables.append(
            {
                "table_id": f"endpoint-{endpoint_id}-effects",
                "title": f"Registered {endpoint_id.replace('_', ' ')} effects",
                "endpoint_id": endpoint_id,
                "registered_units": endpoint["registered_units"],
                "target_sessions": endpoint["target_sessions"],
                "primary_estimate": endpoint["primary_estimate"],
                "required_columns": list(_ENDPOINT_COLUMNS),
                "rows": [],
                "row_source": "registered blind analysis output",
            }
        )
    tables.extend(
        (
            {
                "table_id": "execution-block-calibration",
                "title": "Independent-execution calibration",
                "required_columns": [
                    "outer_fold",
                    "held_out_session",
                    "calibration_unit_count",
                    "order_statistic_rank",
                    "nominal_coverage",
                    "observed_coverage",
                    "interval_width",
                    "normalized_nees",
                    "finite_calibration_sensitivity",
                ],
                "rows": [],
                "row_source": "registered execution-block calibration output",
            },
            {
                "table_id": "failure-and-exclusion-accounting",
                "title": "Technical failures and preregistered exclusions",
                "required_columns": [
                    "execution_id",
                    "failure_or_exclusion_class",
                    "registered_reason",
                    "target_metrics_redacted_when_excluded",
                    "replacement_performed",
                ],
                "rows": [],
                "row_source": "validated evidence registry",
            },
            {
                "table_id": "oracle-gap-attribution",
                "title": "Diagnostic intervention-oracle gap attribution",
                "required_columns": [
                    "endpoint_id",
                    "candidate_error",
                    "current_bank_oracle_error",
                    "full_state_oracle_error",
                    "inference_share",
                    "proposal_coverage_share",
                    "model_discrepancy_share",
                ],
                "rows": [],
                "row_source": "diagnostic-only oracle analysis",
            },
        )
    )
    return tables


def _figure_plan(contract: Mapping[str, Any]) -> list[dict[str, Any]]:
    endpoints = _sequence(
        _mapping(contract["effect_reporting"], name="effect reporting")[
            "endpoint_inventory"
        ],
        name="endpoint inventory",
    )
    figures: list[dict[str, Any]] = []
    for raw in endpoints:
        endpoint = _mapping(raw, name="endpoint")
        endpoint_id = cast(str, endpoint["endpoint_id"])
        figures.append(
            {
                "figure_id": f"endpoint-{endpoint_id}-paired-effects",
                "title": f"{endpoint_id.replace('_', ' ').title()} paired effects",
                "endpoint_id": endpoint_id,
                "required_series": [
                    "target_session",
                    "candidate_minus_baseline",
                    "primary_bootstrap_t_interval",
                    "required_student_t_robustness_interval",
                    "historical_percentile_sensitivity_interval",
                ],
                "values": [],
                "data_source": "registered blind analysis output",
            }
        )
    figures.extend(
        (
            {
                "figure_id": "coverage-and-width",
                "title": "Coverage and interval width by endpoint and horizon",
                "required_series": [
                    "endpoint_id",
                    "horizon",
                    "nominal_coverage",
                    "observed_coverage",
                    "interval_width",
                ],
                "values": [],
                "data_source": "registered calibration output",
            },
            {
                "figure_id": "execution-accounting-flow",
                "title": "Registered, completed, excluded, and failed executions",
                "required_series": [
                    "registered",
                    "completed",
                    "included",
                    "excluded",
                    "technical_failures",
                ],
                "values": [],
                "data_source": "validated evidence registry",
            },
            {
                "figure_id": "oracle-gap-decomposition",
                "title": "Diagnostic oracle-gap decomposition",
                "required_series": [
                    "inference",
                    "proposal_coverage",
                    "model_discrepancy",
                ],
                "values": [],
                "data_source": "diagnostic-only oracle analysis",
            },
        )
    )
    return figures


def _narrative_plan() -> list[dict[str, Any]]:
    return [
        {
            "outcome_id": "registered_success",
            "selected": False,
            "selection_source": "source-verified real-result interpretation",
            "required_statements": [
                "factual prediction gate passed",
                "same-grasp transfer gate passed",
                "new-contact transfer gate passed",
                "independent-execution calibration gate passed",
                "bootstrap-t and required Student-t effect gates passed",
                "all failures and exclusions remain reported",
            ],
            "forbidden_claims": [
                "object-class generalization",
                "raw covariance calibration",
                "general robot safety",
                "individual real counterfactual ground truth",
            ],
        },
        {
            "outcome_id": "registered_bounded_negative",
            "selected": False,
            "selection_source": "source-verified real-result interpretation",
            "required_statements": [
                "failed prediction or transfer gates are identified separately",
                "calibration does not rescue a failed prediction gate",
                "Student-t robustness cannot rescue a failed bootstrap-t gate",
                "all failures and exclusions remain reported",
                "the measured transfer or calibration boundary is quantified",
                "no target-informed retuning is performed",
            ],
            "forbidden_claims": [
                "optional branch rescue of the primary result",
                "silent replacement of failed executions",
                "generalization beyond the registered claim boundary",
            ],
        },
    ]


def _build_shell_payload(
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": REPORT_SHELL_SCHEMA_VERSION,
        "artifact_kind": REPORT_SHELL_ARTIFACT_KIND,
        "shell_id": "",
        "status": REPORT_SHELL_STATUS,
        "source": dict(source),
        "safety_boundary": dict(_SAFETY_BOUNDARY),
        "analysis_contract": dict(contract),
        "table_plan": _table_plan(contract),
        "figure_plan": _figure_plan(contract),
        "narrative_plan": _narrative_plan(),
        "completion_checks": list(_COMPLETION_CHECKS),
    }
    payload["shell_id"] = report_shell_id_for_payload(payload)
    return payload


def build_registered_real_report_shell(
    analysis_manifest: Mapping[str, Any],
    *,
    analysis_manifest_sha256: str,
    analysis_manifest_byte_count: int,
) -> dict[str, Any]:
    """Build a complete report plan without reading target outcomes."""

    analysis = _mapping(analysis_manifest, name="registered analysis manifest")
    source = _source_from_analysis(
        analysis,
        analysis_manifest_sha256=analysis_manifest_sha256,
        analysis_manifest_byte_count=analysis_manifest_byte_count,
    )
    contract = _normalize_contract(analysis)
    return validate_registered_real_report_shell(_build_shell_payload(source, contract))


def _validate_empty_result_slots(payload: Mapping[str, Any]) -> None:
    for raw in _sequence(payload["table_plan"], name="table plan"):
        table = _mapping(raw, name="table specification")
        _require(table.get("rows") == [], "report-shell table rows must be empty")
    for raw in _sequence(payload["figure_plan"], name="figure plan"):
        figure = _mapping(raw, name="figure specification")
        _require(
            figure.get("values") == [],
            "report-shell figure values must be empty",
        )
    for raw in _sequence(payload["narrative_plan"], name="narrative plan"):
        narrative = _mapping(raw, name="narrative specification")
        _require(
            narrative.get("selected") is False,
            "result narratives must remain unselected before target access",
        )


def validate_registered_real_report_shell(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a deterministic report shell with empty result slots."""

    values = _mapping(payload, name="registered real report shell")
    _require(set(values) == _TOP_LEVEL_FIELDS, "report-shell top-level fields changed")
    _require(
        values.get("schema_version") == REPORT_SHELL_SCHEMA_VERSION,
        "report-shell schema version changed",
    )
    _require(
        values.get("artifact_kind") == REPORT_SHELL_ARTIFACT_KIND,
        "report-shell artifact kind changed",
    )
    _require(values.get("status") == REPORT_SHELL_STATUS, "report-shell status changed")

    source = _mapping(values.get("source"), name="report-shell source")
    _require(set(source) == _SOURCE_FIELDS, "report-shell source fields changed")
    normalized_source = {
        "analysis_id": _hex_digest(
            source["analysis_id"],
            name="analysis ID",
            length=64,
        ),
        "analysis_manifest_sha256": _hex_digest(
            source["analysis_manifest_sha256"],
            name="analysis manifest SHA-256",
            length=64,
        ),
        "analysis_manifest_bytes": _positive_int(
            source["analysis_manifest_bytes"],
            name="analysis manifest byte count",
        ),
        "method_freeze_sha256": _hex_digest(
            source["method_freeze_sha256"],
            name="method freeze SHA-256",
            length=64,
        ),
        "protocol_id": _nonempty_string(source["protocol_id"], name="protocol ID"),
        "protocol_design_sha256": _hex_digest(
            source["protocol_design_sha256"],
            name="protocol design SHA-256",
            length=64,
        ),
        "preacquisition_amendment_sha256": _hex_digest(
            source["preacquisition_amendment_sha256"],
            name="pre-acquisition amendment SHA-256",
            length=64,
        ),
        "software": _normalize_software(source["software"]),
    }

    safety = _mapping(values.get("safety_boundary"), name="safety boundary")
    _require(dict(safety) == _SAFETY_BOUNDARY, "report-shell safety boundary changed")
    raw_contract = _mapping(values.get("analysis_contract"), name="analysis contract")
    _require(set(raw_contract) == _CONTRACT_FIELDS, "analysis-contract fields changed")
    synthetic_analysis = {
        "primary_analysis_locked": True,
        "locked_before_target_access": True,
        "target_outcomes_observed_at_registration": False,
        "target_outcomes_may_select_method_or_hyperparameters": False,
        "optional_branches_may_change_primary_analysis": False,
        **dict(raw_contract),
    }
    normalized_contract = _normalize_contract(synthetic_analysis)
    expected = _build_shell_payload(normalized_source, normalized_contract)
    _require(
        dict(values) == expected,
        "report-shell plan changed from the embedded analysis contract",
    )
    _validate_empty_result_slots(values)
    _require(
        values["shell_id"] == report_shell_id_for_payload(values),
        "report-shell content identity changed",
    )
    return dict(values)


def validate_registered_real_report_shell_against_analysis(
    payload: Mapping[str, Any],
    analysis_manifest: Mapping[str, Any],
    *,
    analysis_manifest_sha256: str,
    analysis_manifest_byte_count: int,
) -> dict[str, Any]:
    """Require an exact shell match to one validated registered analysis."""

    shell = validate_registered_real_report_shell(payload)
    expected = build_registered_real_report_shell(
        analysis_manifest,
        analysis_manifest_sha256=analysis_manifest_sha256,
        analysis_manifest_byte_count=analysis_manifest_byte_count,
    )
    _require(shell == expected, "report shell does not match the analysis manifest")
    return shell


def _markdown_claim_boundary(contract: Mapping[str, Any]) -> list[str]:
    boundary = _mapping(contract["claim_boundary"], name="claim boundary")
    lines = [
        "## Registered claim boundary",
        "",
        f"- Physical object count: {boundary.get('physical_object_count')}",
        (
            "- Registered contact-region count: "
            f"{boundary.get('registered_contact_region_count')}"
        ),
        (
            "- Registered action-profile count: "
            f"{boundary.get('registered_action_profile_count')}"
        ),
    ]
    for key, value in sorted(boundary.items()):
        if key.endswith("_claimed") and value is False:
            label = key.removesuffix("_claimed").replace("_", " ")
            lines.append(f"- Not claimed: {label}")
    lines.append("")
    return lines


def render_registered_real_report_shell_markdown(
    payload: Mapping[str, Any],
) -> str:
    """Render the validated shell as a deterministic, result-free report."""

    shell = validate_registered_real_report_shell(payload)
    source = _mapping(shell["source"], name="source")
    contract = _mapping(shell["analysis_contract"], name="analysis contract")
    software = _mapping(source["software"], name="software")
    arms = _sequence(contract["comparison_arms"], name="comparison arms")
    endpoints = _sequence(
        _mapping(contract["effect_reporting"], name="effect reporting")[
            "endpoint_inventory"
        ],
        name="endpoint inventory",
    )

    lines = [
        "# Registered Causal4D real-analysis report shell",
        "",
        "> **TARGET-FREE TEMPLATE — NOT A SCIENTIFIC RESULT.**",
        "> No confirmatory outcomes or target metric values are loaded.",
        "",
        "## Bound sources",
        "",
        f"- Analysis ID: `{source['analysis_id']}`",
        f"- Analysis manifest SHA-256: `{source['analysis_manifest_sha256']}`",
        f"- Method-freeze SHA-256: `{source['method_freeze_sha256']}`",
        f"- Protocol: `{source['protocol_id']}`",
        f"- Causal4D commit: `{software['causal4d_commit_sha']}`",
        f"- BayesianPhysTwin commit: `{software['bayesian_phystwin_commit_sha']}`",
        "",
        "## Comparison arms",
        "",
        "| Arm | Role | Twin uncertainty | Realized intervention inference |",
        "|---|---|---:|---:|",
    ]
    for raw in arms:
        arm = _mapping(raw, name="comparison arm")
        lines.append(
            "| "
            f"`{arm['arm_id']}` | {arm['role']} | "
            f"{arm.get('twin_uncertainty')} | "
            f"{arm.get('realized_intervention_inference')} |"
        )
    lines.extend(
        (
            "",
            "## Registered endpoints",
            "",
            "| Endpoint | Registered units | Target sessions | Primary estimate |",
            "|---|---:|---:|---|",
        )
    )
    for raw in endpoints:
        endpoint = _mapping(raw, name="endpoint")
        lines.append(
            "| "
            f"`{endpoint['endpoint_id']}` | {endpoint['registered_units']} | "
            f"{endpoint['target_sessions']} | `{endpoint['primary_estimate']}` |"
        )

    for raw in _sequence(shell["table_plan"], name="table plan"):
        table = _mapping(raw, name="table")
        columns = [str(column) for column in table["required_columns"]]
        lines.extend(
            (
                "",
                f"## Table: {table['title']}",
                "",
                "| " + " | ".join(columns) + " |",
                "|" + "|".join("---" for _ in columns) + "|",
                "| " + " | ".join("not populated" for _ in columns) + " |",
                "",
                f"Source after blind analysis: {table['row_source']}.",
            )
        )

    lines.extend(("", "## Figure plan", ""))
    for raw in _sequence(shell["figure_plan"], name="figure plan"):
        figure = _mapping(raw, name="figure")
        series = ", ".join(f"`{item}`" for item in figure["required_series"])
        lines.append(f"- **{figure['title']}**: {series}")

    lines.extend(("", "## Predeclared interpretation paths", ""))
    for raw in _sequence(shell["narrative_plan"], name="narrative plan"):
        narrative = _mapping(raw, name="narrative")
        title = str(narrative["outcome_id"]).replace("_", " ").title()
        lines.extend((f"### {title}", ""))
        lines.append(f"Selection source: {narrative['selection_source']}.")
        lines.extend(("", "Required statements:"))
        for statement in narrative["required_statements"]:
            lines.append(f"- {statement}")
        lines.extend(("", "Forbidden claims:"))
        for statement in narrative["forbidden_claims"]:
            lines.append(f"- {statement}")
        lines.append("")

    lines.extend(_markdown_claim_boundary(contract))
    lines.extend(
        (
            "## Completion checklist",
            "",
            *[f"- [ ] {item}" for item in shell["completion_checks"]],
            "",
            f"Shell ID: `{shell['shell_id']}`",
            "",
        )
    )
    return "\n".join(lines)


def validate_registered_real_report_shell_markdown(
    payload: Mapping[str, Any],
    markdown: str | bytes,
) -> str:
    """Require Markdown to be the exact deterministic rendering of ``payload``.

    The JSON shell is the authoritative completion marker.  This check binds a
    separately stored Markdown rendering back to those exact validated bytes so
    a stale, manually edited, or partially replaced document cannot be paired
    with a valid shell.
    """

    shell = validate_registered_real_report_shell(payload)
    if isinstance(markdown, bytes):
        try:
            text = markdown.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError(
                "registered real report shell Markdown must be UTF-8"
            ) from error
    elif type(markdown) is str:
        text = markdown
    else:
        raise ValueError("registered real report shell Markdown must be text")
    expected = render_registered_real_report_shell_markdown(shell)
    _require(
        text == expected,
        "registered real report shell Markdown does not match the shell",
    )
    return text


def _load_shell(path: str | Path) -> tuple[dict[str, Any], str, int]:
    from causal4d.artifact_io import load_strict_json_object, read_regular_file

    snapshot = read_regular_file(path, name="registered real report shell")
    payload = load_strict_json_object(
        snapshot.payload,
        name="registered real report shell",
    )
    shell = validate_registered_real_report_shell(payload)
    return shell, snapshot.sha256, snapshot.byte_count


def _preflight_output_paths(
    json_path: Path,
    markdown_path: Path,
    *,
    overwrite: bool,
) -> None:
    _require(
        json_path.resolve() != markdown_path.resolve(),
        "JSON and Markdown outputs must differ",
    )
    if overwrite:
        return
    for path in (json_path, markdown_path):
        if os.path.lexists(path):
            raise FileExistsError(f"refusing to overwrite existing output: {path}")


def _render_command(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(prog="causal4d evidence real-report-shell render")
    parser.add_argument("analysis_manifest")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-markdown", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parsed = parser.parse_args(arguments)
    json_path = Path(parsed.output_json)
    markdown_path = Path(parsed.output_markdown)
    _preflight_output_paths(
        json_path,
        markdown_path,
        overwrite=parsed.overwrite,
    )

    from causal4d.atomic_io import atomic_write_json, atomic_write_text
    from causal4d.registered_real_analysis import (
        load_registered_real_analysis_manifest,
    )

    analysis, sha256, byte_count = load_registered_real_analysis_manifest(
        parsed.analysis_manifest
    )
    shell = build_registered_real_report_shell(
        analysis,
        analysis_manifest_sha256=sha256,
        analysis_manifest_byte_count=byte_count,
    )
    markdown = render_registered_real_report_shell_markdown(shell)
    validate_registered_real_report_shell_markdown(shell, markdown)

    # Publish the human-readable derivative first and the validated JSON shell
    # last.  The JSON file is the completion marker: after an interruption there
    # can be an orphan Markdown draft, but never a new authoritative shell that
    # points at a missing Markdown rendering.  For the default no-overwrite path,
    # remove that orphan when the second publication raises in-process.
    atomic_write_text(markdown_path, markdown, overwrite=parsed.overwrite)
    try:
        atomic_write_json(json_path, shell, overwrite=parsed.overwrite)
    except BaseException:
        if not parsed.overwrite:
            markdown_path.unlink(missing_ok=True)
        raise
    markdown_payload = markdown.encode("utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "shell_id": shell["shell_id"],
                "output_json": str(json_path.resolve()),
                "output_markdown": str(markdown_path.resolve()),
                "markdown_sha256": hashlib.sha256(markdown_payload).hexdigest(),
                "markdown_bytes": len(markdown_payload),
                "target_outcomes_loaded": False,
                "confirmatory_execution_evidence_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_command(arguments: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="causal4d evidence real-report-shell validate"
    )
    parser.add_argument("report_shell")
    parser.add_argument("--analysis-manifest")
    parser.add_argument("--markdown")
    parsed = parser.parse_args(arguments)
    shell, sha256, byte_count = _load_shell(parsed.report_shell)
    markdown_sha256 = None
    markdown_byte_count = None
    if parsed.markdown:
        from causal4d.artifact_io import read_regular_file

        markdown_snapshot = read_regular_file(
            parsed.markdown,
            name="registered real report shell Markdown",
        )
        validate_registered_real_report_shell_markdown(
            shell,
            markdown_snapshot.payload,
        )
        markdown_sha256 = markdown_snapshot.sha256
        markdown_byte_count = markdown_snapshot.byte_count
    if parsed.analysis_manifest:
        from causal4d.registered_real_analysis import (
            load_registered_real_analysis_manifest,
        )

        analysis, analysis_sha256, analysis_bytes = (
            load_registered_real_analysis_manifest(parsed.analysis_manifest)
        )
        validate_registered_real_report_shell_against_analysis(
            shell,
            analysis,
            analysis_manifest_sha256=analysis_sha256,
            analysis_manifest_byte_count=analysis_bytes,
        )
    print(
        json.dumps(
            {
                "passed": True,
                "shell_id": shell["shell_id"],
                "sha256": sha256,
                "bytes": byte_count,
                "markdown_sha256": markdown_sha256,
                "markdown_bytes": markdown_byte_count,
                "target_outcomes_loaded": False,
                "confirmatory_execution_evidence_count": 0,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Render or validate the target-free report shell."""

    arguments = list(argv) if argv is not None else None
    parser = argparse.ArgumentParser(prog="causal4d evidence real-report-shell")
    parser.add_argument("operation", choices=("render", "validate"))
    parsed, remaining = parser.parse_known_args(arguments)
    if parsed.operation == "render":
        return _render_command(remaining)
    return _validate_command(remaining)


__all__ = [
    "REPORT_SHELL_ARTIFACT_KIND",
    "REPORT_SHELL_SCHEMA_VERSION",
    "REPORT_SHELL_STATUS",
    "build_registered_real_report_shell",
    "main",
    "render_registered_real_report_shell_markdown",
    "report_shell_id_for_payload",
    "validate_registered_real_report_shell",
    "validate_registered_real_report_shell_against_analysis",
    "validate_registered_real_report_shell_markdown",
]


if __name__ == "__main__":
    raise SystemExit(main())
