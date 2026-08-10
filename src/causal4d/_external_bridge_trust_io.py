"""Strict persistence for external bridge trust studies and calibrations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from causal4d._external_bridge_trust_calibration import ExternalBridgeTrustCalibration
from causal4d._external_bridge_trust_study import (
    ExternalBridgeTrustCaseSpec,
    ExternalBridgeTrustStudy,
)
from causal4d._external_bridge_trust_validation import (
    EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA,
    EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA_VERSION,
    EXTERNAL_BRIDGE_TRUST_STUDY_SCHEMA,
    EXTERNAL_BRIDGE_TRUST_STUDY_SCHEMA_VERSION,
    _CALIBRATION_FIELDS,
    _CASE_FIELDS,
    _CASE_OPTIONAL_FIELDS,
    _STUDY_FIELDS,
    _STUDY_OPTIONAL_FIELDS,
    _require_fields,
    _require_integer,
    _require_sha256,
)
from causal4d.artifact_io import load_strict_json_object, read_regular_file
from causal4d.atomic_io import atomic_write_json


def _case_spec(value: Any, *, name: str) -> ExternalBridgeTrustCaseSpec:
    mapping = _require_fields(
        value,
        name=name,
        required=_CASE_FIELDS,
        optional=_CASE_OPTIONAL_FIELDS,
    )
    return ExternalBridgeTrustCaseSpec(
        case_id=mapping["case_id"],
        forecast=mapping["forecast"],
        rollouts=mapping["rollouts"],
        reference=mapping["reference"],
        forecast_id=mapping["forecast_id"],
        control_forecast_ids=tuple(mapping.get("control_forecast_ids", ())),
    )


def load_external_bridge_trust_study(
    path: str | Path,
) -> ExternalBridgeTrustStudy:
    """Load a strict source-selection and independent-confirmation manifest."""

    snapshot = read_regular_file(path, name="external bridge trust study")
    payload = _require_fields(
        load_strict_json_object(
            snapshot.payload,
            name="external bridge trust study",
        ),
        name="external bridge trust study",
        required=_STUDY_FIELDS,
        optional=_STUDY_OPTIONAL_FIELDS,
    )
    if payload["schema"] != EXTERNAL_BRIDGE_TRUST_STUDY_SCHEMA:
        raise ValueError("unexpected external bridge trust study schema")
    if (
        _require_integer(
            payload["schema_version"],
            name="external bridge trust study schema_version",
            minimum=1,
        )
        != EXTERNAL_BRIDGE_TRUST_STUDY_SCHEMA_VERSION
    ):
        raise ValueError("unsupported external bridge trust study schema version")
    selection_raw = payload["selection_cases"]
    confirmation_raw = payload["confirmation_cases"]
    if not isinstance(selection_raw, list) or not isinstance(confirmation_raw, list):
        raise ValueError("selection_cases and confirmation_cases must be JSON arrays")
    selection = tuple(
        _case_spec(value, name=f"selection_cases[{index}]")
        for index, value in enumerate(selection_raw)
    )
    confirmation = tuple(
        _case_spec(value, name=f"confirmation_cases[{index}]")
        for index, value in enumerate(confirmation_raw)
    )
    return ExternalBridgeTrustStudy(
        manifest_path=snapshot.path,
        manifest_sha256=snapshot.sha256,
        selection_cases=selection,
        confirmation_cases=confirmation,
        metadata=payload.get("metadata", {}),
    )


def save_external_bridge_trust_calibration(
    path: str | Path,
    calibration: ExternalBridgeTrustCalibration,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically save and revalidate one trust calibration artifact."""

    atomic_write_json(path, calibration.descriptor(), overwrite=overwrite)
    restored = load_external_bridge_trust_calibration(path)
    if restored.calibration_id != calibration.calibration_id:
        raise ValueError("trust calibration changed during serialization")


def load_external_bridge_trust_calibration(
    path: str | Path,
) -> ExternalBridgeTrustCalibration:
    """Load and independently revalidate one trust calibration artifact."""

    snapshot = read_regular_file(path, name="external bridge trust calibration")
    payload = _require_fields(
        load_strict_json_object(
            snapshot.payload,
            name="external bridge trust calibration",
        ),
        name="external bridge trust calibration",
        required=_CALIBRATION_FIELDS,
    )
    if payload["schema"] != EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA:
        raise ValueError("unexpected external bridge trust calibration schema")
    if (
        _require_integer(
            payload["schema_version"],
            name="external bridge trust calibration schema_version",
            minimum=1,
        )
        != EXTERNAL_BRIDGE_TRUST_CALIBRATION_SCHEMA_VERSION
    ):
        raise ValueError("unsupported external bridge trust calibration schema version")
    expected_id = _require_sha256(payload["calibration_id"], name="calibration_id")
    calibration = ExternalBridgeTrustCalibration(
        study_manifest_sha256=payload["study_manifest_sha256"],
        beta_candidates=tuple(payload["beta_candidates"]),
        selected_beta=payload["selected_beta"],
        admitted_beta=payload["admitted_beta"],
        confirmed=payload["confirmed"],
        selection=payload["selection"],
        confirmation=payload["confirmation"],
        thresholds=payload["thresholds"],
        gates=payload["gates"],
        settings=payload["settings"],
        source_cases=payload["source_cases"],
        reasons=tuple(payload["reasons"]),
        metadata=payload["metadata"],
    )
    if calibration.calibration_id != expected_id:
        raise ValueError("external bridge trust calibration ID does not match payload")
    return calibration


__all__ = [
    "load_external_bridge_trust_calibration",
    "load_external_bridge_trust_study",
    "save_external_bridge_trust_calibration",
]
