"""Study-manifest models for external bridge trust calibration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from causal4d._external_bridge_trust_validation import (
    _require_sha256,
    _require_string,
    _safe_relative_path,
    _string_tuple,
)
from causal4d.immutable_json import validated_json_mapping


@dataclass(frozen=True)
class ExternalBridgeTrustCaseSpec:
    """One source or independent-confirmation case in a trust study."""

    case_id: str
    forecast: str
    rollouts: str
    reference: str
    forecast_id: str
    control_forecast_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        case_id = _require_string(self.case_id, name="case_id")
        forecast = _safe_relative_path(self.forecast, name="forecast")
        rollouts = _safe_relative_path(self.rollouts, name="rollouts")
        reference = _safe_relative_path(self.reference, name="reference")
        forecast_id = _require_string(self.forecast_id, name="forecast_id")
        controls = _string_tuple(
            self.control_forecast_ids,
            name="control_forecast_ids",
            allow_empty=True,
        )
        if forecast_id in controls:
            raise ValueError("control_forecast_ids must exclude forecast_id")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "forecast", forecast)
        object.__setattr__(self, "rollouts", rollouts)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "forecast_id", forecast_id)
        object.__setattr__(self, "control_forecast_ids", controls)

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "forecast": self.forecast,
            "rollouts": self.rollouts,
            "reference": self.reference,
            "forecast_id": self.forecast_id,
            "control_forecast_ids": list(self.control_forecast_ids),
        }


@dataclass(frozen=True)
class ExternalBridgeTrustStudy:
    """Strict portable source-selection and independent-confirmation manifest."""

    manifest_path: Path
    manifest_sha256: str
    selection_cases: tuple[ExternalBridgeTrustCaseSpec, ...]
    confirmation_cases: tuple[ExternalBridgeTrustCaseSpec, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        manifest_path = Path(self.manifest_path)
        digest = _require_sha256(self.manifest_sha256, name="manifest_sha256")
        selection = tuple(self.selection_cases)
        confirmation = tuple(self.confirmation_cases)
        if not selection:
            raise ValueError("trust study requires at least one selection case")
        identifiers = [case.case_id for case in (*selection, *confirmation)]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("trust study case IDs must be globally unique")
        metadata = validated_json_mapping(
            self.metadata,
            error_message="trust study metadata must be finite JSON data",
        )
        object.__setattr__(self, "manifest_path", manifest_path)
        object.__setattr__(self, "manifest_sha256", digest)
        object.__setattr__(self, "selection_cases", selection)
        object.__setattr__(self, "confirmation_cases", confirmation)
        object.__setattr__(self, "metadata", metadata)

    @property
    def root(self) -> Path:
        return self.manifest_path.parent
