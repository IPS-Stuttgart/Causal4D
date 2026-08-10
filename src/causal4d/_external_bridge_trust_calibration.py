"""Immutable calibration and target-decision models for bridge trust."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from causal4d._external_bridge_trust_validation import (
    EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA,
    EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA_VERSION,
    EXTERNAL_BRIDGE_TRUST_DECISION_SCHEMA,
    EXTERNAL_BRIDGE_TRUST_DECISION_SCHEMA_VERSION,
    _GATE_FIELDS,
    _SETTING_FIELDS,
    _SOURCE_CASE_FIELDS,
    _SOURCE_CASE_RECORD_FIELDS,
    _THRESHOLD_FIELDS,
    _canonical_json,
    _float_tuple,
    _require_fields,
    _require_finite,
    _require_mapping,
    _require_nonnegative,
    _require_positive,
    _require_sha256,
    _require_string,
    _string_tuple,
    _validate_panel,
)
from causal4d.immutable_json import plain_json, validated_json_mapping


@dataclass(frozen=True)
class ExternalBridgeTrustCalibration:
    """Source-selected and independently confirmed semantic trust settings."""

    study_manifest_sha256: str
    beta_candidates: tuple[float, ...]
    selected_beta: float
    admitted_beta: float
    confirmed: bool
    selection: Mapping[str, Any]
    confirmation: Mapping[str, Any]
    thresholds: Mapping[str, Any]
    gates: Mapping[str, Any]
    settings: Mapping[str, Any]
    source_cases: Mapping[str, Any]
    reasons: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        digest = _require_sha256(
            self.study_manifest_sha256,
            name="study_manifest_sha256",
        )
        candidates = _float_tuple(self.beta_candidates, name="beta_candidates")
        selected = _require_nonnegative(self.selected_beta, name="selected_beta")
        admitted = _require_nonnegative(self.admitted_beta, name="admitted_beta")
        if selected not in candidates or admitted not in {0.0, selected}:
            raise ValueError("selected/admitted beta must respect beta_candidates")
        if type(self.confirmed) is not bool:
            raise ValueError("confirmed must be Boolean")
        if self.confirmed != (admitted > 0.0):
            raise ValueError("confirmed must be equivalent to positive admitted_beta")
        selection = _validate_panel(
            self.selection,
            name="selection",
            confirmation=False,
        )
        confirmation = _validate_panel(
            self.confirmation,
            name="confirmation",
            confirmation=True,
        )
        thresholds = _require_fields(
            self.thresholds,
            name="thresholds",
            required=_THRESHOLD_FIELDS,
        )
        gates = _require_fields(self.gates, name="gates", required=_GATE_FIELDS)
        settings = _require_fields(
            self.settings,
            name="settings",
            required=_SETTING_FIELDS,
        )
        normalized_thresholds = {
            "maximum_support_distance_m": _require_positive(
                thresholds["maximum_support_distance_m"],
                name="maximum_support_distance_m",
            ),
            "maximum_anchor_error_m": _require_positive(
                thresholds["maximum_anchor_error_m"],
                name="maximum_anchor_error_m",
            ),
            "minimum_semantic_motion_m": _require_positive(
                thresholds["minimum_semantic_motion_m"],
                name="minimum_semantic_motion_m",
            ),
            "minimum_motion_ratio": _require_nonnegative(
                thresholds["minimum_motion_ratio"],
                name="minimum_motion_ratio",
            ),
            "maximum_motion_ratio": _require_positive(
                thresholds["maximum_motion_ratio"],
                name="maximum_motion_ratio",
            ),
            "minimum_valid_coordinate_fraction": _require_finite(
                thresholds["minimum_valid_coordinate_fraction"],
                name="minimum_valid_coordinate_fraction",
            ),
            "require_clean_doctor": thresholds["require_clean_doctor"],
        }
        if (
            normalized_thresholds["minimum_motion_ratio"]
            >= normalized_thresholds["maximum_motion_ratio"]
        ):
            raise ValueError("motion-ratio thresholds must be ordered")
        if not 0.0 <= normalized_thresholds[
            "minimum_valid_coordinate_fraction"
        ] <= 1.0:
            raise ValueError("minimum_valid_coordinate_fraction must lie in [0, 1]")
        if type(normalized_thresholds["require_clean_doctor"]) is not bool:
            raise ValueError("require_clean_doctor must be Boolean")
        normalized_gates = {
            "minimum_selection_relative_improvement": _require_nonnegative(
                gates["minimum_selection_relative_improvement"],
                name="minimum_selection_relative_improvement",
            ),
            "minimum_confirmation_relative_improvement": _require_nonnegative(
                gates["minimum_confirmation_relative_improvement"],
                name="minimum_confirmation_relative_improvement",
            ),
            "maximum_case_relative_harm": _require_nonnegative(
                gates["maximum_case_relative_harm"],
                name="maximum_case_relative_harm",
            ),
            "support_margin": _require_finite(
                gates["support_margin"],
                name="support_margin",
            ),
            "controls_required": gates["controls_required"],
            "minimum_control_advantage_m": _require_nonnegative(
                gates["minimum_control_advantage_m"],
                name="minimum_control_advantage_m",
            ),
        }
        if normalized_gates["support_margin"] < 1.0:
            raise ValueError("support_margin must be at least one")
        if type(normalized_gates["controls_required"]) is not bool:
            raise ValueError("controls_required must be Boolean")
        normalized_settings = {
            "scale_m": _require_positive(settings["scale_m"], name="scale_m"),
            "degrees_of_freedom": _require_positive(
                settings["degrees_of_freedom"],
                name="degrees_of_freedom",
            ),
            "anchor_tolerance_m": _require_positive(
                settings["anchor_tolerance_m"],
                name="anchor_tolerance_m",
            ),
            "doctor_motion_ratio_min": _require_positive(
                settings["doctor_motion_ratio_min"],
                name="doctor_motion_ratio_min",
            ),
            "doctor_motion_ratio_max": _require_positive(
                settings["doctor_motion_ratio_max"],
                name="doctor_motion_ratio_max",
            ),
        }
        if (
            normalized_settings["doctor_motion_ratio_min"]
            >= normalized_settings["doctor_motion_ratio_max"]
        ):
            raise ValueError("doctor motion-ratio bounds must be ordered")
        source_case_mapping = _require_fields(
            self.source_cases,
            name="source_cases",
            required=_SOURCE_CASE_FIELDS,
        )
        for panel_name in ("selection", "confirmation"):
            records = source_case_mapping[panel_name]
            if not isinstance(records, list):
                raise ValueError(f"source_cases.{panel_name} must be a JSON array")
            for index, record in enumerate(records):
                record_mapping = _require_fields(
                    record,
                    name=f"source_cases.{panel_name}[{index}]",
                    required=_SOURCE_CASE_RECORD_FIELDS,
                )
                for digest_name in (
                    "forecast_artifact_id",
                    "rollout_artifact_id",
                    "rollout_bank_artifact_id",
                    "reference_artifact_id",
                ):
                    _require_sha256(
                        record_mapping.get(digest_name),
                        name=f"source_cases.{panel_name}[{index}].{digest_name}",
                    )
                _require_string(
                    record_mapping.get("case_id"),
                    name=f"source_cases.{panel_name}[{index}].case_id",
                )
                _require_string(
                    record_mapping.get("forecast_id"),
                    name=f"source_cases.{panel_name}[{index}].forecast_id",
                )
                _string_tuple(
                    record_mapping.get("control_forecast_ids", ()),
                    name=(
                        f"source_cases.{panel_name}[{index}].control_forecast_ids"
                    ),
                    allow_empty=True,
                )
        source_cases = validated_json_mapping(
            source_case_mapping,
            error_message="source_cases must be finite JSON data",
        )
        selection_source_ids = [
            str(record["case_id"]) for record in source_cases["selection"]
        ]
        confirmation_source_ids = [
            str(record["case_id"]) for record in source_cases["confirmation"]
        ]
        if selection_source_ids != list(selection["case_ids"]):
            raise ValueError("selection case IDs differ from source_cases")
        if confirmation_source_ids != list(confirmation["case_ids"]):
            raise ValueError("confirmation case IDs differ from source_cases")
        expected_selection_keys = {format(value, ".17g") for value in candidates}
        actual_selection_keys = set(selection["mean_ade_m_by_beta"])
        if actual_selection_keys != expected_selection_keys:
            raise ValueError("selection beta metrics differ from beta_candidates")
        if set(selection["mean_fde_m_by_beta"]) != expected_selection_keys:
            raise ValueError("selection FDE metrics differ from beta_candidates")
        if confirmation["evaluated"]:
            expected_confirmation_keys = {
                format(0.0, ".17g"),
                format(selected, ".17g"),
            }
            if selected == 0.0:
                expected_confirmation_keys = {format(0.0, ".17g")}
            if set(confirmation["mean_ade_m_by_beta"]) != expected_confirmation_keys:
                raise ValueError("confirmation beta metrics differ from selected beta")
            if set(confirmation["mean_fde_m_by_beta"]) != expected_confirmation_keys:
                raise ValueError("confirmation FDE metrics differ from selected beta")
        reasons = _string_tuple(self.reasons, name="reasons", allow_empty=True)
        if self.confirmed and reasons:
            raise ValueError("confirmed calibration must not contain rejection reasons")
        if not self.confirmed and not reasons:
            raise ValueError("rejected calibration must contain at least one reason")
        if self.confirmed and not confirmation["passed"]:
            raise ValueError("confirmed calibration requires a passed confirmation panel")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="trust calibration metadata must be finite JSON data",
        )
        object.__setattr__(self, "study_manifest_sha256", digest)
        object.__setattr__(self, "beta_candidates", candidates)
        object.__setattr__(self, "selected_beta", selected)
        object.__setattr__(self, "admitted_beta", admitted)
        object.__setattr__(self, "selection", selection)
        object.__setattr__(self, "confirmation", confirmation)
        object.__setattr__(
            self,
            "thresholds",
            validated_json_mapping(normalized_thresholds),
        )
        object.__setattr__(self, "gates", validated_json_mapping(normalized_gates))
        object.__setattr__(
            self,
            "settings",
            validated_json_mapping(normalized_settings),
        )
        object.__setattr__(self, "source_cases", source_cases)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "metadata", metadata)

    def _descriptor_without_id(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA,
            "schema_version": EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA_VERSION,
            "study_manifest_sha256": self.study_manifest_sha256,
            "beta_candidates": list(self.beta_candidates),
            "selected_beta": self.selected_beta,
            "admitted_beta": self.admitted_beta,
            "confirmed": self.confirmed,
            "selection": plain_json(self.selection),
            "confirmation": plain_json(self.confirmation),
            "thresholds": plain_json(self.thresholds),
            "gates": plain_json(self.gates),
            "settings": plain_json(self.settings),
            "source_cases": plain_json(self.source_cases),
            "reasons": list(self.reasons),
            "metadata": plain_json(self.metadata),
        }

    @property
    def calibration_id(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._descriptor_without_id())
        ).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        payload = self._descriptor_without_id()
        payload["calibration_id"] = self.calibration_id
        return payload


@dataclass(frozen=True)
class ExternalBridgeTrustDecision:
    """Label-free target admission decision for one frozen calibration."""

    calibration_id: str
    forecast_artifact_id: str
    rollout_artifact_id: str
    forecast_id: str
    admitted_beta: float
    applied_beta: float
    accepted: bool
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        calibration_id = _require_sha256(self.calibration_id, name="calibration_id")
        forecast_artifact_id = _require_sha256(
            self.forecast_artifact_id,
            name="forecast_artifact_id",
        )
        rollout_artifact_id = _require_sha256(
            self.rollout_artifact_id,
            name="rollout_artifact_id",
        )
        forecast_id = _require_string(self.forecast_id, name="forecast_id")
        admitted = _require_nonnegative(self.admitted_beta, name="admitted_beta")
        applied = _require_nonnegative(self.applied_beta, name="applied_beta")
        if applied not in {0.0, admitted}:
            raise ValueError("applied_beta must be zero or the admitted beta")
        if type(self.accepted) is not bool or self.accepted != (applied > 0.0):
            raise ValueError("accepted must be equivalent to positive applied_beta")
        reasons = _string_tuple(self.reasons, name="reasons", allow_empty=True)
        if self.accepted and reasons:
            raise ValueError("accepted trust decision must not contain reasons")
        if not self.accepted and not reasons:
            raise ValueError("rejected trust decision must contain reasons")
        diagnostics = validated_json_mapping(
            _require_mapping(self.diagnostics, name="diagnostics"),
            error_message="trust decision diagnostics must be finite JSON data",
        )
        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "forecast_artifact_id", forecast_artifact_id)
        object.__setattr__(self, "rollout_artifact_id", rollout_artifact_id)
        object.__setattr__(self, "forecast_id", forecast_id)
        object.__setattr__(self, "admitted_beta", admitted)
        object.__setattr__(self, "applied_beta", applied)
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "diagnostics", diagnostics)

    def _descriptor_without_id(self) -> dict[str, Any]:
        return {
            "schema": EXTERNAL_BRIDGE_TRUST_DECISION_SCHEMA,
            "schema_version": EXTERNAL_BRIDGE_TRUST_DECISION_SCHEMA_VERSION,
            "calibration_id": self.calibration_id,
            "forecast_artifact_id": self.forecast_artifact_id,
            "rollout_artifact_id": self.rollout_artifact_id,
            "forecast_id": self.forecast_id,
            "admitted_beta": self.admitted_beta,
            "applied_beta": self.applied_beta,
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "diagnostics": plain_json(self.diagnostics),
        }

    @property
    def decision_id(self) -> str:
        return hashlib.sha256(
            _canonical_json(self._descriptor_without_id())
        ).hexdigest()

    def descriptor(self) -> dict[str, Any]:
        payload = self._descriptor_without_id()
        payload["decision_id"] = self.decision_id
        return payload
