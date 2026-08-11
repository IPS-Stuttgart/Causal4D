"""Source-verified diagnostics for the registered real-effect intervals.

The target-free preacquisition amendment promotes session-clustered bootstrap-t
as the primary interval and requires Student-t as a veto-only robustness check.
This companion artifact verifies that the report uses the shared implementation,
retains the historical percentile result, and binds the operating-characteristic
evidence that motivated the amendment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json
from causal4d.real_analysis_interval_amendment import (
    BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID,
    BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID,
    INTERVAL_COMPARISON_AUDIT_ID,
    INTERVAL_COMPARISON_RUN_ID,
    OPERATING_CHARACTERISTIC_TARGET_SHA,
    expected_real_analysis_interval_amendment,
)
from causal4d.real_analysis_intervals import (
    REAL_EFFECT_BOOTSTRAP_REPLICATES,
    REAL_EFFECT_BOOTSTRAP_SEED,
    REAL_EFFECT_CONFIDENCE_LEVEL,
    bootstrap_t_mean_interval,
    percentile_bootstrap_mean_interval,
    registered_positive_effect_interval_decision,
    student_t_mean_interval,
)
from causal4d.real_analysis_reporting import (
    build_real_analysis_effect_report,
    load_real_analysis_effect_table,
)
from causal4d.registered_real_analysis import (
    REGISTERED_ANALYSIS_SCHEMA_VERSION,
    validate_registered_real_analysis_manifest,
)


REAL_ANALYSIS_INTERVAL_DIAGNOSTICS_SCHEMA_VERSION = 2


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def student_t_sensitivity_interval(
    values: Sequence[float],
    *,
    confidence_level: float = REAL_EFFECT_CONFIDENCE_LEVEL,
) -> dict[str, Any]:
    """Return the historical non-decision-making Student-t view."""

    result = student_t_mean_interval(
        values,
        confidence_level=confidence_level,
    )
    return {
        **result,
        "method": "student_t_mean_sensitivity",
        "may_change_primary_decision": False,
    }


def bootstrap_t_sensitivity_interval(
    values: Sequence[float],
    *,
    confidence_level: float = REAL_EFFECT_CONFIDENCE_LEVEL,
    replicates: int = REAL_EFFECT_BOOTSTRAP_REPLICATES,
    seed: int = REAL_EFFECT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return the historical non-decision-making bootstrap-t view."""

    result = bootstrap_t_mean_interval(
        values,
        confidence_level=confidence_level,
        replicates=replicates,
        seed=seed,
    )
    return {
        **result,
        "method": "bootstrap_t_mean_sensitivity",
        "may_change_primary_decision": False,
    }


def _included_session_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    lower_is_better: bool,
) -> tuple[float, ...]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not bool(record["included"]):
            continue
        baseline = float(record["baseline_value"])
        candidate = float(record["candidate_value"])
        improvement = baseline - candidate if lower_is_better else candidate - baseline
        grouped[str(record["session_id"])].append(improvement)
    return tuple(float(np.mean(values)) for values in grouped.values())


def _registered_interval_payload(values: Sequence[float]) -> dict[str, Any]:
    primary = bootstrap_t_mean_interval(values)
    robustness = student_t_mean_interval(values)
    historical = percentile_bootstrap_mean_interval(values)
    return {
        "primary_bootstrap_t": {
            **primary,
            "role": "primary",
            "may_change_primary_decision": True,
        },
        "required_student_t_robustness": {
            **robustness,
            "role": "required_positive_claim_robustness",
            "may_veto_positive_claim": True,
            "may_rescue_primary_failure": False,
        },
        "historical_percentile_sensitivity": {
            **historical,
            "role": "historical_sensitivity",
            "may_change_primary_decision": False,
        },
        "decision": registered_positive_effect_interval_decision(
            primary,
            robustness,
        ),
    }


def build_real_analysis_interval_diagnostics(
    effect_table_path: str | Path,
    protocol_path: str | Path,
    *,
    method_freeze_path: str | Path,
    analysis_manifest_path: str | Path,
) -> dict[str, Any]:
    """Verify the registered intervals and publish their target-free rationale."""

    primary_report = build_real_analysis_effect_report(
        effect_table_path,
        protocol_path,
        method_freeze_path=method_freeze_path,
        analysis_manifest_path=analysis_manifest_path,
    )
    analysis_snapshot = read_regular_file(
        analysis_manifest_path,
        name="registered analysis manifest",
    )
    analysis_payload = load_strict_json_object(
        analysis_snapshot.payload,
        name="registered analysis manifest",
    )
    if analysis_payload.get("schema_version") != REGISTERED_ANALYSIS_SCHEMA_VERSION:
        raise ValueError(
            "registered interval diagnostics require a schema-3 analysis manifest"
        )
    analysis = validate_registered_real_analysis_manifest(
        analysis_payload,
        expected_protocol_id=str(primary_report["protocol_id"]),
        expected_protocol_design_sha256=str(primary_report["protocol_design_sha256"]),
        expected_preacquisition_amendment_sha256=str(
            primary_report["preacquisition_amendment_sha256"]
        ),
        expected_method_freeze_sha256=str(primary_report["method_freeze_sha256"]),
    )
    table, _ = load_real_analysis_effect_table(effect_table_path)
    values = _included_session_effects(
        table.records,
        lower_is_better=table.lower_is_better,
    )
    expected = _registered_interval_payload(values)
    reported = primary_report["primary_session_clustered_effect"]
    if reported["confidence_interval"] != expected["primary_bootstrap_t"]:
        raise ValueError("primary report does not use registered bootstrap-t interval")
    if (
        reported["required_robustness_interval"]
        != expected["required_student_t_robustness"]
    ):
        raise ValueError("primary report does not use registered Student-t robustness")
    if (
        reported["historical_percentile_sensitivity_interval"]
        != expected["historical_percentile_sensitivity"]
    ):
        raise ValueError("primary report changed the historical percentile interval")
    if reported["interval_decision"] != expected["decision"]:
        raise ValueError("primary report changed the registered interval decision")

    primary_summary = reported["equal_session_weighted_improvement"]
    expected_point = None if primary_summary is None else float(primary_summary["mean"])
    actual_point = None if not values else float(np.mean(values))
    if expected_point is None:
        if actual_point is not None:
            raise ValueError("companion point estimate differs from primary report")
    elif actual_point is None or not np.isclose(expected_point, actual_point):
        raise ValueError("companion point estimate differs from primary report")

    amendment = expected_real_analysis_interval_amendment()
    if analysis["interval_amendment"]["amendment_id"] != amendment["amendment_id"]:
        raise ValueError("registered analysis binds another interval amendment")
    payload: dict[str, Any] = {
        "schema_version": REAL_ANALYSIS_INTERVAL_DIAGNOSTICS_SCHEMA_VERSION,
        "artifact_kind": "Causal4DRealAnalysisIntervalDiagnostics",
        "protocol_id": primary_report["protocol_id"],
        "protocol_design_sha256": primary_report["protocol_design_sha256"],
        "preacquisition_amendment_sha256": primary_report[
            "preacquisition_amendment_sha256"
        ],
        "method_freeze_sha256": primary_report["method_freeze_sha256"],
        "analysis_manifest_sha256": primary_report["analysis_manifest_sha256"],
        "interval_amendment_id": amendment["amendment_id"],
        "endpoint": primary_report["endpoint"],
        "metric_id": primary_report["metric_id"],
        "metric_unit": primary_report["metric_unit"],
        "source_primary_report_id": primary_report["report_id"],
        "source_effect_table": primary_report["source_effect_table"],
        "source_protocol": primary_report["source_protocol"],
        "source_verification": primary_report["source_verification"],
        "included_session_count": len(values),
        "point_estimate": actual_point,
        "registered_intervals": expected,
        "operating_characteristic_evidence": {
            "implementation_target_sha": OPERATING_CHARACTERISTIC_TARGET_SHA,
            "percentile_bootstrap": {
                "workflow_run_id": BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID,
                "audit_id": BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID,
                "tested_session_counts": [12, 18],
                "finite_sample_coverage_guaranteed": False,
            },
            "interval_comparison": {
                "workflow_run_id": INTERVAL_COMPARISON_RUN_ID,
                "audit_id": INTERVAL_COMPARISON_AUDIT_ID,
                "bootstrap_t_mean_absolute_coverage_error": 0.019,
                "bootstrap_t_worst_absolute_coverage_error": 0.042,
                "student_t_maximum_favorable_type_i_error": (0.02666666666666667),
            },
            "physical_target_outcomes_used": False,
        },
        "interpretation": {
            "bootstrap_t_is_registered_primary": True,
            "student_t_may_veto_positive_claim": True,
            "student_t_may_rescue_primary_failure": False,
            "percentile_interval_is_historical_sensitivity": True,
            "negative_or_bounded_result_remains_reportable": True,
            "target_informed_selection": False,
        },
        "claim_boundary": {
            **primary_report["claim_boundary"],
            "physical_target_outcomes_used_to_choose_interval": False,
            "interval_diagnostics_may_change_registered_policy": False,
        },
    }
    payload["diagnostic_id"] = _canonical_sha256(payload)
    return payload


def write_real_analysis_interval_diagnostics(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Atomically publish a finite companion interval artifact."""

    atomic_write_json(path, payload, overwrite=overwrite)


__all__ = [
    "BOOTSTRAP_OPERATING_CHARACTERISTIC_AUDIT_ID",
    "BOOTSTRAP_OPERATING_CHARACTERISTIC_RUN_ID",
    "INTERVAL_COMPARISON_AUDIT_ID",
    "INTERVAL_COMPARISON_RUN_ID",
    "REAL_ANALYSIS_INTERVAL_DIAGNOSTICS_SCHEMA_VERSION",
    "bootstrap_t_sensitivity_interval",
    "build_real_analysis_interval_diagnostics",
    "student_t_sensitivity_interval",
    "write_real_analysis_interval_diagnostics",
]
