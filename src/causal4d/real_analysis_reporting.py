"""Registered session-clustered reporting for the Causal4D real experiment."""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from numpy.typing import NDArray

from causal4d.artifact_io import (
    ArtifactFileSnapshot,
    load_strict_json_object,
    read_regular_file,
)
from causal4d.atomic_io import atomic_write_json
from causal4d.execution_block_calibration import (
    ExecutionBlockConformalCalibration,
)
from causal4d.immutable_json import plain_json
from causal4d.real_analysis_intervals import (
    REAL_EFFECT_BOOTSTRAP_REPLICATES,
    REAL_EFFECT_BOOTSTRAP_SEED,
    REAL_EFFECT_CONFIDENCE_LEVEL,
    bootstrap_t_mean_interval,
    percentile_bootstrap_mean_interval,
    registered_positive_effect_interval_decision,
    student_t_mean_interval,
)
from causal4d.real_protocol import validate_protocol
from causal4d.real_result_source_verification import verify_real_result_sources

REAL_ANALYSIS_EFFECT_TABLE_SCHEMA_VERSION = 1
REAL_ANALYSIS_REPORT_SCHEMA_VERSION = 2
EXPECTED_PROTOCOL_ID = "causal4d-sloth-multi-action-v1"
EXPECTED_PROTOCOL_DESIGN_SHA256 = (
    "6d61f2bea96af0ba04faaf3476990b58cd87e0a9c826420c254a012dec647968"
)
EXPECTED_PREACQUISITION_SHA256 = (
    "0e167538a7824e5ec053031d8359d4e9b4ff89ad61a85666400a86c2a88ac42f"
)
EXPECTED_OBJECT_ID = "sloth_plush_instance_1"
BOOTSTRAP_REPLICATES = REAL_EFFECT_BOOTSTRAP_REPLICATES
BOOTSTRAP_SEED = REAL_EFFECT_BOOTSTRAP_SEED
BOOTSTRAP_CONFIDENCE_LEVEL = REAL_EFFECT_CONFIDENCE_LEVEL

Endpoint = Literal[
    "factual_continuation",
    "same_grasp_transfer",
    "new_contact_transfer",
]

_ENDPOINT_SPLIT_KEYS: dict[Endpoint, str] = {
    "factual_continuation": "factual_continuation",
    "same_grasp_transfer": "same_grasp_intervention_prediction",
    "new_contact_transfer": "new_contact_intervention_prediction",
}
_ENDPOINT_COUNTS: dict[Endpoint, tuple[int, int]] = {
    "factual_continuation": (36, 18),
    "same_grasp_transfer": (18, 18),
    "new_contact_transfer": (12, 12),
}
_TABLE_FIELDS = frozenset(
    {
        "schema_version",
        "artifact_kind",
        "effect_table_id",
        "protocol_id",
        "protocol_design_sha256",
        "preacquisition_amendment_sha256",
        "method_freeze_sha256",
        "analysis_manifest_sha256",
        "endpoint",
        "metric_id",
        "metric_unit",
        "lower_is_better",
        "target_outcomes_used",
        "target_informed_selection",
        "object_id",
        "records",
    }
)
_RECORD_FIELDS = frozenset(
    {
        "unit_id",
        "source_execution_id",
        "target_execution_id",
        "session_id",
        "acquisition_execution_index",
        "action_id",
        "contact_region_id",
        "realization_condition_id",
        "included",
        "exclusion_reason",
        "baseline_value",
        "candidate_value",
    }
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a JSON object")
    _require(all(type(key) is str for key in value), f"{name} keys must be strings")
    return cast(Mapping[str, Any], value)


def _json_array(value: Any, *, name: str) -> list[Any]:
    _require(isinstance(value, list), f"{name} must be a JSON array")
    return cast(list[Any], value)


def _exact_fields(
    value: Mapping[str, Any],
    expected: frozenset[str],
    *,
    name: str,
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    _require(
        not missing and not extra,
        f"{name} fields changed; missing={missing}, extra={extra}",
    )


def _string(value: Any, *, name: str, optional: bool = False) -> str | None:
    if optional and value is None:
        return None
    _require(type(value) is str and bool(value), f"{name} must be a nonempty string")
    return value


def _sha256(value: Any, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _integer(value: Any, *, name: str) -> int:
    _require(type(value) is int and value >= 0, f"{name} must be nonnegative")
    return value


def _number(value: Any, *, name: str) -> float:
    _require(
        type(value) in {int, float} and math.isfinite(float(value)),
        f"{name} must be a finite JSON number",
    )
    return float(value)


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def effect_table_id_for_payload(payload: Mapping[str, Any]) -> str:
    """Content-address one effect table without hashing its self identity."""

    canonical = dict(payload)
    canonical.pop("effect_table_id", None)
    return _canonical_sha256(canonical)


@dataclass(frozen=True)
class RealAnalysisEffectTable:
    """Validated complete effect inventory for one endpoint and metric."""

    protocol_id: str
    protocol_design_sha256: str
    preacquisition_amendment_sha256: str
    method_freeze_sha256: str
    analysis_manifest_sha256: str
    endpoint: Endpoint
    metric_id: str
    metric_unit: str
    lower_is_better: bool
    object_id: str
    records: tuple[Mapping[str, Any], ...]
    effect_table_id: str


def _validate_record(value: Mapping[str, Any], *, position: int) -> dict[str, Any]:
    record = _mapping(value, name=f"effect record {position}")
    _exact_fields(record, _RECORD_FIELDS, name=f"effect record {position}")
    included = record["included"]
    _require(type(included) is bool, "included must be Boolean")
    normalized: dict[str, Any] = {
        "unit_id": _string(record["unit_id"], name="unit_id"),
        "source_execution_id": _string(
            record["source_execution_id"],
            name="source_execution_id",
            optional=True,
        ),
        "target_execution_id": _string(
            record["target_execution_id"],
            name="target_execution_id",
        ),
        "session_id": _string(record["session_id"], name="session_id"),
        "acquisition_execution_index": _integer(
            record["acquisition_execution_index"],
            name="acquisition_execution_index",
        ),
        "action_id": _string(record["action_id"], name="action_id"),
        "contact_region_id": _string(
            record["contact_region_id"],
            name="contact_region_id",
        ),
        "realization_condition_id": _string(
            record["realization_condition_id"],
            name="realization_condition_id",
        ),
        "included": included,
        "exclusion_reason": _string(
            record["exclusion_reason"],
            name="exclusion_reason",
            optional=True,
        ),
    }
    baseline = record["baseline_value"]
    candidate = record["candidate_value"]
    if included:
        _require(
            normalized["exclusion_reason"] is None,
            "included records cannot have an exclusion reason",
        )
        normalized["baseline_value"] = _number(baseline, name="baseline_value")
        normalized["candidate_value"] = _number(candidate, name="candidate_value")
    else:
        _require(
            normalized["exclusion_reason"] is not None,
            "excluded records require an exclusion reason",
        )
        _require(
            baseline is None and candidate is None,
            "excluded records must not contain target metric values",
        )
        normalized["baseline_value"] = None
        normalized["candidate_value"] = None
    return normalized


def _load_json_snapshot(
    path: str | Path,
    *,
    name: str,
) -> tuple[ArtifactFileSnapshot, dict[str, Any]]:
    snapshot = read_regular_file(path, name=name)
    return snapshot, load_strict_json_object(snapshot.payload, name=name)


def load_real_analysis_effect_table(
    path: str | Path,
) -> tuple[RealAnalysisEffectTable, ArtifactFileSnapshot]:
    """Load one exact-byte, content-addressed effect table."""

    snapshot, payload = _load_json_snapshot(path, name="real-analysis effect table")
    _exact_fields(payload, _TABLE_FIELDS, name="real-analysis effect table")
    _require(
        payload["schema_version"] == REAL_ANALYSIS_EFFECT_TABLE_SCHEMA_VERSION,
        "unsupported real-analysis effect-table schema",
    )
    _require(
        payload["artifact_kind"] == "Causal4DRealAnalysisEffectTable",
        "unexpected real-analysis effect-table artifact kind",
    )
    endpoint = _string(payload["endpoint"], name="endpoint")
    _require(endpoint in _ENDPOINT_SPLIT_KEYS, "unknown real endpoint")
    for name in (
        "lower_is_better",
        "target_outcomes_used",
        "target_informed_selection",
    ):
        _require(type(payload[name]) is bool, f"{name} must be Boolean")
    _require(payload["target_outcomes_used"] is True, "target outcomes are required")
    _require(
        payload["target_informed_selection"] is False,
        "target-informed selection invalidates registered reporting",
    )
    records_value = payload["records"]
    _require(isinstance(records_value, list), "records must be a JSON array")
    records = tuple(
        sorted(
            (
                _validate_record(_mapping(item, name="effect record"), position=index)
                for index, item in enumerate(records_value)
            ),
            key=lambda record: (
                record["acquisition_execution_index"],
                record["unit_id"],
            ),
        )
    )
    _require(bool(records), "effect table records must be nonempty")
    unit_ids = [record["unit_id"] for record in records]
    targets = [record["target_execution_id"] for record in records]
    _require(len(unit_ids) == len(set(unit_ids)), "effect unit IDs must be unique")
    _require(len(targets) == len(set(targets)), "target executions must be unique")
    table = RealAnalysisEffectTable(
        protocol_id=str(_string(payload["protocol_id"], name="protocol_id")),
        protocol_design_sha256=_sha256(
            payload["protocol_design_sha256"],
            name="protocol_design_sha256",
        ),
        preacquisition_amendment_sha256=_sha256(
            payload["preacquisition_amendment_sha256"],
            name="preacquisition_amendment_sha256",
        ),
        method_freeze_sha256=_sha256(
            payload["method_freeze_sha256"],
            name="method_freeze_sha256",
        ),
        analysis_manifest_sha256=_sha256(
            payload["analysis_manifest_sha256"],
            name="analysis_manifest_sha256",
        ),
        endpoint=endpoint,  # type: ignore[arg-type]
        metric_id=str(_string(payload["metric_id"], name="metric_id")),
        metric_unit=str(_string(payload["metric_unit"], name="metric_unit")),
        lower_is_better=payload["lower_is_better"],
        object_id=str(_string(payload["object_id"], name="object_id")),
        records=records,
        effect_table_id=_sha256(payload["effect_table_id"], name="effect_table_id"),
    )
    _require(table.protocol_id == EXPECTED_PROTOCOL_ID, "unexpected protocol")
    _require(
        table.protocol_design_sha256 == EXPECTED_PROTOCOL_DESIGN_SHA256,
        "effect table protocol digest differs from the locked design",
    )
    _require(
        table.preacquisition_amendment_sha256 == EXPECTED_PREACQUISITION_SHA256,
        "effect table amendment digest differs from locked v4",
    )
    _require(table.object_id == EXPECTED_OBJECT_ID, "unexpected physical object")
    _require(
        table.effect_table_id == effect_table_id_for_payload(payload),
        "real-analysis effect-table checksum mismatch",
    )
    return table, snapshot


def _registered_units(
    protocol: Mapping[str, Any],
    endpoint: Endpoint,
) -> tuple[dict[str, Any], ...]:
    validate_protocol(protocol)
    _require(protocol.get("protocol_id") == EXPECTED_PROTOCOL_ID, "wrong protocol")
    _require(
        protocol.get("design_sha256") == EXPECTED_PROTOCOL_DESIGN_SHA256,
        "protocol design digest differs from the locked design",
    )
    object_record = _mapping(protocol.get("object"), name="protocol object")
    _require(object_record.get("object_id") == EXPECTED_OBJECT_ID, "wrong object")
    raw_executions = _json_array(
        protocol.get("executions"),
        name="protocol executions",
    )
    executions: dict[str, Mapping[str, Any]] = {}
    for raw in raw_executions:
        execution = _mapping(raw, name="protocol execution")
        execution_id = str(_string(execution.get("execution_id"), name="execution_id"))
        _require(
            execution_id not in executions,
            "protocol execution IDs are duplicated",
        )
        executions[execution_id] = execution
    splits = _mapping(protocol.get("splits"), name="protocol splits")
    entries = _json_array(
        splits.get(_ENDPOINT_SPLIT_KEYS[endpoint]),
        name="endpoint split",
    )
    units: list[dict[str, Any]] = []
    for raw in entries:
        entry = _mapping(raw, name="endpoint unit")
        if endpoint == "factual_continuation":
            source_id = None
            target_id = str(_string(entry.get("execution_id"), name="execution_id"))
            unit_id = target_id
        else:
            source_id = str(
                _string(entry.get("source_execution_id"), name="source_execution_id")
            )
            target_id = str(
                _string(entry.get("target_execution_id"), name="target_execution_id")
            )
            unit_id = f"{source_id}->{target_id}"
            _require(source_id in executions, f"unknown source execution {source_id}")
        _require(target_id in executions, f"unknown target execution {target_id}")
        target = executions[target_id]
        units.append(
            {
                "unit_id": unit_id,
                "source_execution_id": source_id,
                "target_execution_id": target_id,
                "session_id": target["session_id"],
                "acquisition_execution_index": target["acquisition_execution_index"],
                "action_id": target["command_profile_id"],
                "contact_region_id": target["contact_region_id"],
                "realization_condition_id": target["realization_condition_id"],
            }
        )
    expected_units, expected_sessions = _ENDPOINT_COUNTS[endpoint]
    _require(len(units) == expected_units, "registered endpoint unit count changed")
    _require(
        len({unit["session_id"] for unit in units}) == expected_sessions,
        "registered endpoint session count changed",
    )
    return tuple(units)


def _validate_accounting(
    table: RealAnalysisEffectTable,
    registered: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(unit["unit_id"]): unit for unit in registered}
    actual = {str(record["unit_id"]): record for record in table.records}
    missing = sorted(set(expected) - set(actual))
    extra = sorted(set(actual) - set(expected))
    _require(
        not missing and not extra,
        f"effect-table accounting differs from the registered endpoint; "
        f"missing={missing}, extra={extra}",
    )
    fields = (
        "source_execution_id",
        "target_execution_id",
        "session_id",
        "acquisition_execution_index",
        "action_id",
        "contact_region_id",
        "realization_condition_id",
    )
    for unit_id, unit in expected.items():
        record = actual[unit_id]
        for field in fields:
            _require(
                record[field] == unit[field],
                f"effect record {unit_id} {field} differs from the locked protocol",
            )


def _improvement(record: Mapping[str, Any], *, lower_is_better: bool) -> float:
    baseline = float(record["baseline_value"])
    candidate = float(record["candidate_value"])
    return baseline - candidate if lower_is_better else candidate - baseline


def _summary(values: Sequence[float]) -> dict[str, Any] | None:
    if not values:
        return None
    array: NDArray[np.float64] = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _session_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    lower_is_better: bool,
) -> tuple[dict[str, float], dict[str, float]]:
    effects: dict[str, list[float]] = defaultdict(list)
    indices: dict[str, list[float]] = defaultdict(list)
    for record in records:
        if not record["included"]:
            continue
        session = str(record["session_id"])
        effects[session].append(_improvement(record, lower_is_better=lower_is_better))
        indices[session].append(float(record["acquisition_execution_index"]))
    return (
        {session: float(np.mean(values)) for session, values in effects.items()},
        {session: float(np.mean(indices[session])) for session in effects},
    )


def _registered_intervals(values: Sequence[float]) -> dict[str, Any]:
    """Build the registered primary, robustness, and historical intervals."""

    primary = bootstrap_t_mean_interval(values)
    robustness = student_t_mean_interval(values)
    historical = percentile_bootstrap_mean_interval(values)
    return {
        "primary": {
            **primary,
            "role": "primary",
            "may_change_primary_decision": True,
        },
        "required_robustness": {
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


def _drift(
    effects: Mapping[str, float],
    indices: Mapping[str, float],
) -> dict[str, Any]:
    sessions = sorted(effects, key=lambda session: indices[session])
    result: dict[str, Any] = {
        "estimable": len(sessions) >= 3,
        "session_count": len(sessions),
        "slope_per_execution_index": None,
        "fitted_early_to_late_change": None,
        "pearson_correlation": None,
        "late_minus_early_mean": None,
        "may_select_exclusions": False,
        "may_change_primary_decision": False,
    }
    if len(sessions) < 3:
        return result
    x: NDArray[np.float64] = np.asarray(
        [indices[session] for session in sessions],
        dtype=np.float64,
    )
    y: NDArray[np.float64] = np.asarray(
        [effects[session] for session in sessions],
        dtype=np.float64,
    )
    centered = x - np.mean(x)
    slope = float(np.dot(centered, y - np.mean(y)) / np.dot(centered, centered))
    correlation = 0.0
    if np.std(y) > 0.0:
        correlation = float(np.corrcoef(x, y)[0, 1])
    split = len(sessions) // 2
    result.update(
        slope_per_execution_index=slope,
        fitted_early_to_late_change=float(slope * (np.max(x) - np.min(x))),
        pearson_correlation=correlation,
        late_minus_early_mean=float(np.mean(y[-split:]) - np.mean(y[:split])),
    )
    return result


def _subgroups(
    records: Sequence[Mapping[str, Any]],
    *,
    field: str,
    lower_is_better: bool,
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[field])].append(record)
    result = {}
    for group_id, group_records in sorted(groups.items()):
        effects, _ = _session_effects(
            group_records,
            lower_is_better=lower_is_better,
        )
        result[group_id] = {
            "registered_unit_count": len(group_records),
            "included_unit_count": sum(
                bool(record["included"]) for record in group_records
            ),
            "included_session_count": len(effects),
            "equal_session_weighted_improvement": _summary(list(effects.values())),
            "primary_decision_eligible": False,
        }
    return result


def _design_diagnostics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    actions = sorted({str(record["action_id"]) for record in records})
    conditions = sorted({str(record["realization_condition_id"]) for record in records})
    counts = {
        condition: {
            action: sum(
                record["action_id"] == action
                and record["realization_condition_id"] == condition
                for record in records
            )
            for action in actions
        }
        for condition in conditions
    }
    timing = {}
    for condition in conditions:
        values = [
            int(record["acquisition_execution_index"])
            for record in records
            if record["realization_condition_id"] == condition
        ]
        timing[condition] = {
            "minimum_acquisition_index": min(values),
            "median_acquisition_index": float(np.median(values)),
            "maximum_acquisition_index": max(values),
        }
    return {
        "action_ids": actions,
        "realization_condition_ids": conditions,
        "registered_action_condition_counts": counts,
        "fully_crossed": all(
            count > 0
            for action_counts in counts.values()
            for count in action_counts.values()
        ),
        "balanced_across_actions": all(
            len(set(action_counts.values())) == 1 for action_counts in counts.values()
        ),
        "condition_acquisition_timing": timing,
        "condition_comparisons_are_descriptive_only": True,
        "condition_comparisons_may_change_primary_decision": False,
    }


def build_real_analysis_effect_report(
    effect_table_path: str | Path,
    protocol_path: str | Path,
    *,
    method_freeze_path: str | Path,
    analysis_manifest_path: str | Path,
) -> dict[str, Any]:
    """Build a source-verified, session-clustered paired-effect report."""

    table, table_snapshot = load_real_analysis_effect_table(effect_table_path)
    protocol_snapshot, protocol = _load_json_snapshot(
        protocol_path,
        name="registered real protocol",
    )
    _validate_accounting(table, _registered_units(protocol, table.endpoint))
    verification = verify_real_result_sources(
        table,
        method_freeze_path=method_freeze_path,
        analysis_manifest_path=analysis_manifest_path,
    )
    effects, indices = _session_effects(
        table.records,
        lower_is_better=table.lower_is_better,
    )
    session_values = list(effects.values())
    intervals = _registered_intervals(session_values)
    expected_units, expected_sessions = _ENDPOINT_COUNTS[table.endpoint]
    tolerance = 1e-12
    report: dict[str, Any] = {
        "schema_version": REAL_ANALYSIS_REPORT_SCHEMA_VERSION,
        "artifact_kind": "Causal4DSessionClusteredEffectReport",
        "protocol_id": table.protocol_id,
        "protocol_design_sha256": table.protocol_design_sha256,
        "preacquisition_amendment_sha256": table.preacquisition_amendment_sha256,
        "method_freeze_sha256": table.method_freeze_sha256,
        "analysis_manifest_sha256": table.analysis_manifest_sha256,
        "endpoint": table.endpoint,
        "metric_id": table.metric_id,
        "metric_unit": table.metric_unit,
        "improvement_sign": (
            "baseline_minus_candidate"
            if table.lower_is_better
            else "candidate_minus_baseline"
        ),
        "positive_values_favor_candidate": True,
        "source_effect_table": {
            "effect_table_id": table.effect_table_id,
            "sha256": table_snapshot.sha256,
            "bytes": table_snapshot.byte_count,
        },
        "source_protocol": {
            "sha256": protocol_snapshot.sha256,
            "bytes": protocol_snapshot.byte_count,
            "semantic_design_sha256": EXPECTED_PROTOCOL_DESIGN_SHA256,
        },
        "source_verification": verification,
        "accounting": {
            "complete": True,
            "expected_unit_count": expected_units,
            "registered_unit_count": len(table.records),
            "included_unit_count": sum(
                bool(record["included"]) for record in table.records
            ),
            "excluded_unit_count": sum(
                not record["included"] for record in table.records
            ),
            "expected_session_count": expected_sessions,
            "included_session_count": len(effects),
            "exclusions": [
                {
                    "unit_id": record["unit_id"],
                    "target_execution_id": record["target_execution_id"],
                    "reason": record["exclusion_reason"],
                }
                for record in table.records
                if not record["included"]
            ],
        },
        "primary_session_clustered_effect": {
            "estimable": len(session_values) >= 2,
            "equal_session_weighted_improvement": _summary(session_values),
            "confidence_interval": intervals["primary"],
            "required_robustness_interval": intervals["required_robustness"],
            "historical_percentile_sensitivity_interval": intervals[
                "historical_percentile_sensitivity"
            ],
            "interval_decision": intervals["decision"],
            "candidate_better_session_count": sum(
                value > tolerance for value in session_values
            ),
            "candidate_tied_session_count": sum(
                abs(value) <= tolerance for value in session_values
            ),
            "candidate_worse_session_count": sum(
                value < -tolerance for value in session_values
            ),
            "session_is_resampling_unit": True,
            "executions_are_not_treated_as_independent": True,
        },
        "unweighted_execution_diagnostic": _summary(
            [
                _improvement(record, lower_is_better=table.lower_is_better)
                for record in table.records
                if record["included"]
            ]
        ),
        "acquisition_order_diagnostic": _drift(effects, indices),
        "secondary_subgroups": {
            "action": _subgroups(
                table.records,
                field="action_id",
                lower_is_better=table.lower_is_better,
            ),
            "contact_region": _subgroups(
                table.records,
                field="contact_region_id",
                lower_is_better=table.lower_is_better,
            ),
            "realization_condition": _subgroups(
                table.records,
                field="realization_condition_id",
                lower_is_better=table.lower_is_better,
            ),
        },
        "design_diagnostics": _design_diagnostics(table.records),
        "claim_boundary": {
            "object_id": table.object_id,
            "same_object_protocol_only": True,
            "object_class_generalization_claimed": False,
            "individual_real_counterfactual_ground_truth_claimed": False,
            "hardware_safety_claimed": False,
            "subgroup_summaries_may_change_primary_decision": False,
            "drift_diagnostics_may_select_exclusions": False,
            "target_informed_selection": False,
            "bootstrap_t_is_primary_interval": True,
            "student_t_robustness_may_veto_positive_claim": True,
            "student_t_robustness_may_rescue_primary_failure": False,
            "historical_percentile_interval_may_change_primary_decision": False,
        },
    }
    report["report_id"] = _canonical_sha256(report)
    return report


def summarize_execution_block_utility(
    calibration: ExecutionBlockConformalCalibration,
    evaluation: Mapping[str, Any],
) -> dict[str, Any]:
    """Report interval utility and fragility without changing calibration."""

    _require(
        evaluation.get("calibration_id") == calibration.calibration_id,
        "execution-block evaluation identifies another calibration",
    )
    _require(
        evaluation.get("outer_fold_id") == calibration.outer_fold_id,
        "execution-block evaluation identifies another outer fold",
    )
    raw_cases = _json_array(evaluation.get("cases"), name="evaluation cases")
    _require(bool(raw_cases), "evaluation cases missing")
    cases = [_mapping(value, name="evaluation case") for value in raw_cases]
    mean_widths = [
        _number(case.get("mean_interval_width_m"), name="mean_interval_width_m")
        for case in cases
    ]
    maximum_widths = [
        _number(
            case.get("maximum_interval_width_m"),
            name="maximum_interval_width_m",
        )
        for case in cases
    ]
    coordinate_counts = [
        _integer(case.get("coordinate_count"), name="coordinate_count")
        for case in cases
    ]
    _require(all(value > 0 for value in coordinate_counts), "coordinate count is zero")
    coverage = _number(
        evaluation.get("execution_block_coverage"),
        name="execution_block_coverage",
    )
    _require(0.0 <= coverage <= 1.0, "execution-block coverage must lie in [0, 1]")
    return {
        "schema_version": 1,
        "artifact_kind": "Causal4DExecutionBlockUtilitySummary",
        "calibration_id": calibration.calibration_id,
        "outer_fold_id": calibration.outer_fold_id,
        "confidence_level": float(calibration.confidence_level),
        "target_execution_count": len(cases),
        "execution_block_coverage": coverage,
        "target_coordinate_count": int(sum(coordinate_counts)),
        "interval_width_m": {
            "mean_of_execution_means": float(np.mean(mean_widths)),
            "median_of_execution_means": float(np.median(mean_widths)),
            "maximum_execution_mean": float(np.max(mean_widths)),
            "maximum_coordinate_width": float(np.max(maximum_widths)),
        },
        "calibration_threshold": float(calibration.threshold),
        "calibration_fragility": plain_json(calibration.fragility_diagnostics),
        "coverage_without_interval_width_is_sufficient": False,
        "fragility_may_select_threshold": False,
        "worst_group_coverage_guarantee_claimed": False,
        "pooled_coordinate_conformal_claimed": False,
    }


def write_real_analysis_effect_report(
    path: str | Path,
    report: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Publish one finite report atomically."""

    atomic_write_json(path, dict(report), overwrite=overwrite)


__all__ = [
    "BOOTSTRAP_CONFIDENCE_LEVEL",
    "BOOTSTRAP_REPLICATES",
    "BOOTSTRAP_SEED",
    "EXPECTED_OBJECT_ID",
    "EXPECTED_PREACQUISITION_SHA256",
    "EXPECTED_PROTOCOL_DESIGN_SHA256",
    "EXPECTED_PROTOCOL_ID",
    "RealAnalysisEffectTable",
    "build_real_analysis_effect_report",
    "effect_table_id_for_payload",
    "load_real_analysis_effect_table",
    "summarize_execution_block_utility",
    "write_real_analysis_effect_report",
]
