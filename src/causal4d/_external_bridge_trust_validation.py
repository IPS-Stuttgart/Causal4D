"""Validation helpers and schema constants for external bridge trust."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from causal4d.immutable_json import plain_json, validated_json_mapping

EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA = (
    "causal4d.external_forecast_rollout_trust_calibration"
)

EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA_VERSION = 1

EXTERNAL_BRIDGE_TRUST_STUDY_SCHEMA = "causal4d.external_bridge_trust_study"

EXTERNAL_BRIDGE_TRUST_STUDY_SCHEMA_VERSION = 1

EXTERNAL_BRIDGE_TRUST_DECISION_SCHEMA = (
    "causal4d.external_forecast_rollout_trust_decision"
)

EXTERNAL_BRIDGE_TRUST_DECISION_SCHEMA_VERSION = 1

_CALIBRATION_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "calibration_id",
        "study_manifest_sha256",
        "beta_candidates",
        "selected_beta",
        "admitted_beta",
        "confirmed",
        "selection",
        "confirmation",
        "thresholds",
        "gates",
        "settings",
        "source_cases",
        "reasons",
        "metadata",
    }
)

_STUDY_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "selection_cases",
        "confirmation_cases",
    }
)

_STUDY_OPTIONAL_FIELDS = frozenset({"metadata"})

_CASE_FIELDS = frozenset(
    {"case_id", "forecast", "rollouts", "reference", "forecast_id"}
)

_CASE_OPTIONAL_FIELDS = frozenset({"control_forecast_ids"})

_THRESHOLD_FIELDS = frozenset(
    {
        "maximum_support_distance_m",
        "maximum_anchor_error_m",
        "minimum_semantic_motion_m",
        "minimum_motion_ratio",
        "maximum_motion_ratio",
        "minimum_valid_coordinate_fraction",
        "require_clean_doctor",
    }
)

_GATE_FIELDS = frozenset(
    {
        "minimum_selection_relative_improvement",
        "minimum_confirmation_relative_improvement",
        "maximum_case_relative_harm",
        "support_margin",
        "controls_required",
        "minimum_control_advantage_m",
    }
)

_SETTING_FIELDS = frozenset(
    {
        "scale_m",
        "degrees_of_freedom",
        "anchor_tolerance_m",
        "doctor_motion_ratio_min",
        "doctor_motion_ratio_max",
    }
)

_SELECTION_FIELDS = frozenset(
    {
        "case_ids",
        "mean_ade_m_by_beta",
        "mean_fde_m_by_beta",
        "physical_prior_mean_ade_m",
        "selected_beta_mean_ade_m",
        "relative_improvement",
        "physical_prior_mean_fde_m",
        "selected_beta_mean_fde_m",
        "fde_relative_improvement",
        "maximum_case_relative_harm",
        "maximum_case_fde_relative_harm",
        "case_results",
        "controls_evaluated",
        "minimum_instruction_control_advantage_m",
    }
)

_CONFIRMATION_FIELDS = _SELECTION_FIELDS | frozenset({"evaluated", "passed"})

_SOURCE_CASE_FIELDS = frozenset({"selection", "confirmation"})

_SOURCE_CASE_RECORD_FIELDS = frozenset(
    {
        "case_id",
        "forecast_artifact_id",
        "rollout_artifact_id",
        "rollout_bank_artifact_id",
        "reference_artifact_id",
        "forecast_id",
        "control_forecast_ids",
    }
)

_LOWER_HEX = frozenset("0123456789abcdef")


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    if any(type(key) is not str for key in value):
        raise ValueError(f"{name} keys must be strings")
    return value


def _require_fields(
    value: Any,
    *,
    name: str,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(required - actual)
    unexpected = sorted(actual - required - optional)
    if missing or unexpected:
        raise ValueError(
            f"{name} fields do not match schema; "
            f"missing={missing}, unexpected={unexpected}"
        )
    return mapping


def _require_string(value: Any, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(
            f"{name} must be an integer greater than or equal to {minimum}"
        )
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in _LOWER_HEX for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _require_finite(value: Any, *, name: str) -> float:
    if type(value) not in {int, float} or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _require_nonnegative(value: Any, *, name: str) -> float:
    result = _require_finite(value, name=name)
    if result < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return result


def _require_positive(value: Any, *, name: str) -> float:
    result = _require_finite(value, name=name)
    if result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _string_tuple(
    values: Any,
    *,
    name: str,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(
        _require_string(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if not result and not allow_empty:
        raise ValueError(f"{name} must be nonempty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    return result


def _float_tuple(values: Any, *, name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence of numbers")
    result = tuple(
        _require_nonnegative(value, name=f"{name}[{index}]")
        for index, value in enumerate(values)
    )
    if (
        not result
        or result[0] != 0.0
        or any(right <= left for left, right in zip(result, result[1:], strict=False))
    ):
        raise ValueError(f"{name} must be strictly increasing and start at zero")
    return result


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        plain_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _safe_relative_path(value: Any, *, name: str) -> str:
    text = _require_string(value, name=name)
    if "\\" in text or text.startswith("/") or text.endswith("/") or "//" in text:
        raise ValueError(f"{name} must be a safe POSIX relative path")
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{name} must be a safe POSIX relative path")
    return text


def _nullable_finite(value: Any, *, name: str) -> float | None:
    if value is None:
        return None
    return _require_finite(value, name=name)


def _validate_panel(
    value: Any,
    *,
    name: str,
    confirmation: bool,
) -> Mapping[str, Any]:
    required = _CONFIRMATION_FIELDS if confirmation else _SELECTION_FIELDS
    panel = _require_fields(value, name=name, required=required)
    evaluated = True
    if confirmation:
        if type(panel["evaluated"]) is not bool or type(panel["passed"]) is not bool:
            raise ValueError(f"{name}.evaluated and passed must be Boolean")
        evaluated = bool(panel["evaluated"])
    case_ids = _string_tuple(
        panel["case_ids"],
        name=f"{name}.case_ids",
        allow_empty=confirmation,
    )
    case_results = panel["case_results"]
    if not isinstance(case_results, list):
        raise ValueError(f"{name}.case_results must be a JSON array")
    if evaluated and len(case_results) != len(case_ids):
        raise ValueError(f"{name}.case_results must match case_ids")
    if not evaluated and len(case_results) not in {0, len(case_ids)}:
        raise ValueError(f"{name}.case_results must be empty or match case_ids")
    for metric_name in ("mean_ade_m_by_beta", "mean_fde_m_by_beta"):
        metric = _require_mapping(panel[metric_name], name=f"{name}.{metric_name}")
        if not metric and evaluated:
            raise ValueError(f"{name}.{metric_name} must be nonempty when evaluated")
        for key, item in metric.items():
            _require_string(key, name=f"{name}.{metric_name} key")
            _require_nonnegative(item, name=f"{name}.{metric_name}[{key!r}]")
    nullable_fields = {
        "minimum_instruction_control_advantage_m",
    }
    if confirmation:
        nullable_fields.update(
            {
                "physical_prior_mean_ade_m",
                "selected_beta_mean_ade_m",
                "relative_improvement",
                "physical_prior_mean_fde_m",
                "selected_beta_mean_fde_m",
                "fde_relative_improvement",
                "maximum_case_relative_harm",
                "maximum_case_fde_relative_harm",
            }
        )
    numeric_fields = {
        "physical_prior_mean_ade_m",
        "selected_beta_mean_ade_m",
        "relative_improvement",
        "physical_prior_mean_fde_m",
        "selected_beta_mean_fde_m",
        "fde_relative_improvement",
        "maximum_case_relative_harm",
        "maximum_case_fde_relative_harm",
        "minimum_instruction_control_advantage_m",
    }
    for field_name in numeric_fields:
        if field_name in nullable_fields:
            _nullable_finite(panel[field_name], name=f"{name}.{field_name}")
        else:
            _require_finite(panel[field_name], name=f"{name}.{field_name}")
    if type(panel["controls_evaluated"]) is not bool:
        raise ValueError(f"{name}.controls_evaluated must be Boolean")
    if confirmation:
        if not evaluated and any(
            panel[field_name] is not None
            for field_name in (
                numeric_fields - {"minimum_instruction_control_advantage_m"}
            )
        ):
            raise ValueError(f"{name} unevaluated numeric results must be null")
        if not evaluated and panel["passed"]:
            raise ValueError(f"{name} cannot pass without evaluation")
    return validated_json_mapping(
        panel,
        error_message=f"{name} must be finite JSON data",
    )
