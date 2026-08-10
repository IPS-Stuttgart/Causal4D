"""Diagnostic attribution across independent actuator and wrench factors."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from causal4d.artifact_io import (
    ArtifactValidationError,
    load_strict_json_object,
    read_regular_file,
)
from causal4d.atomic_io import atomic_write_binary
from causal4d.contracts import FactualIntervention, array_sha256
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.sensor_evidence import ActuatorEvidence, ContactWrenchEvidence
from causal4d.sensor_factorized_abduction import (
    IndependentSensorAbductionConfig,
    reweight_factual_intervention_with_independent_sensors,
)

INDEPENDENT_SENSOR_ABLATION_SCHEMA_VERSION = 1
INDEPENDENT_SENSOR_ABLATION_ARTIFACT_KIND = (
    "IndependentSensorAblationReportV1"
)
INDEPENDENT_SENSOR_ABLATION_ARMS = (
    "object_prefix",
    "actuator_only",
    "wrench_only",
    "actuator_and_wrench",
)


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        plain_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _array_binding(values: np.ndarray | None) -> dict[str, Any] | None:
    if values is None:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "shape": list(array.shape),
        "sha256": array_sha256(array),
    }


def _broadcast_array_binding(
    values: np.ndarray | None,
    *,
    expected_shape: tuple[int, ...] | None,
) -> dict[str, Any] | None:
    if values is None:
        return None
    if expected_shape is None:
        raise ValueError("variance binding requires its prediction array")
    try:
        array = np.broadcast_to(
            np.asarray(values, dtype=float),
            expected_shape,
        ).copy()
    except ValueError as error:
        raise ValueError(
            f"predicted variance must broadcast to {expected_shape}"
        ) from error
    return {
        "shape": list(array.shape),
        "sha256": array_sha256(array),
    }


def _entropy_nats(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    selected = values > 0.0
    return float(-np.sum(values[selected] * np.log(values[selected])))


def _normalized_entropy(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    active = int(np.sum(values > 0.0))
    if active <= 1:
        return 0.0
    return float(_entropy_nats(values) / np.log(active))


def _effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    return float(1.0 / np.sum(np.square(values)))


def _kl_divergence_nats(posterior: np.ndarray, prior: np.ndarray) -> float:
    post = np.asarray(posterior, dtype=float)
    base = np.asarray(prior, dtype=float)
    selected = post > 0.0
    if np.any(base[selected] <= 0.0):
        raise RuntimeError("sensor ablation resurrected excluded prior support")
    return float(
        np.sum(post[selected] * (np.log(post[selected]) - np.log(base[selected])))
    )


def _marginal_entropy_nats(values: np.ndarray, weights: np.ndarray) -> float:
    rows = np.asarray(values, dtype=float)
    if rows.ndim != 2 or rows.shape[0] != len(weights):
        raise ValueError("marginal support must align with posterior weights")
    masses: dict[tuple[float, ...], float] = {}
    for row, weight in zip(rows, weights, strict=True):
        key = tuple(float(value) for value in row)
        masses[key] = masses.get(key, 0.0) + float(weight)
    return _entropy_nats(np.asarray(tuple(masses.values()), dtype=float))


def _validated_component_metrics(
    component_metrics: Mapping[str, np.ndarray] | None,
    *,
    component_count: int,
) -> dict[str, np.ndarray]:
    if component_metrics is None:
        return {}
    result: dict[str, np.ndarray] = {}
    for name, values in component_metrics.items():
        if type(name) is not str or not name:
            raise ValueError("component metric names must be nonempty strings")
        array = np.asarray(values, dtype=float)
        if array.shape != (component_count,):
            raise ValueError(
                f"component metric {name!r} must have shape ({component_count},)"
            )
        if not np.all(np.isfinite(array)) or np.any(array < 0.0):
            raise ValueError(
                f"component metric {name!r} must be finite and nonnegative"
            )
        result[name] = array.copy()
    return result


def _validated_metric_units(
    metric_units: Mapping[str, str] | None,
    *,
    metric_names: set[str],
) -> dict[str, str]:
    if metric_units is None:
        return {name: "" for name in metric_names}
    if set(metric_units) != metric_names:
        raise ValueError("metric_units must identify every component metric exactly")
    result: dict[str, str] = {}
    for name, unit in metric_units.items():
        if type(unit) is not str:
            raise ValueError("component metric units must be strings")
        result[name] = unit
    return result


def _factor_diagnostics(posterior: FactualIntervention) -> list[dict[str, Any]]:
    diagnostics = posterior.metadata.get("independent_sensor_abduction")
    if not isinstance(diagnostics, Mapping):
        return []
    factors = diagnostics.get("factors")
    if not isinstance(factors, Sequence) or isinstance(factors, str):
        return []
    return [plain_json(value) for value in factors]


def _arm_summary(
    arm_name: str,
    posterior: FactualIntervention,
    source: FactualIntervention,
    *,
    component_metrics: Mapping[str, np.ndarray],
    metric_units: Mapping[str, str],
) -> dict[str, Any]:
    weights = np.asarray(posterior.weights, dtype=float)
    source_weights = np.asarray(source.weights, dtype=float)
    if weights.shape != source_weights.shape:
        raise ValueError("sensor-ablation posterior support changed shape")
    if np.any((source_weights == 0.0) & (weights != 0.0)):
        raise RuntimeError("sensor ablation resurrected excluded prior support")
    maximum = float(np.max(weights))
    map_index = int(np.flatnonzero(weights == maximum)[0])
    metrics: dict[str, Any] = {}
    for name, values in sorted(component_metrics.items()):
        metrics[name] = {
            "unit": metric_units[name],
            "component_values_sha256": array_sha256(values),
            "posterior_expected_value": float(np.dot(weights, values)),
            "map_component_value": float(values[map_index]),
            "minimum_supported_value": float(np.min(values[source_weights > 0.0])),
        }
    return {
        "arm_name": arm_name,
        "posterior_id": posterior.artifact_id,
        "exact_source_fallback": posterior is source,
        "component_count": len(weights),
        "active_component_count": int(np.sum(weights > 0.0)),
        "effective_sample_size": _effective_sample_size(weights),
        "component_entropy_nats": _entropy_nats(weights),
        "normalized_component_entropy": _normalized_entropy(weights),
        "phi_entropy_nats": _marginal_entropy_nats(posterior.phi, weights),
        "kappa_entropy_nats": _marginal_entropy_nats(posterior.kappa_obs, weights),
        "kl_from_object_prefix_nats": _kl_divergence_nats(
            weights,
            source_weights,
        ),
        "maximum_component_probability": maximum,
        "map_component_id": posterior.component_ids[map_index],
        "factor_diagnostics": _factor_diagnostics(posterior),
        "component_metrics": metrics,
    }


def _entropy_attribution(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source = arms["object_prefix"]
    actuator = arms["actuator_only"]
    wrench = arms["wrench_only"]
    combined = arms["actuator_and_wrench"]
    result: dict[str, Any] = {}
    for field_name, label in (
        ("component_entropy_nats", "component"),
        ("phi_entropy_nats", "phi"),
        ("kappa_entropy_nats", "kappa"),
    ):
        source_value = float(source[field_name])
        actuator_value = float(actuator[field_name])
        wrench_value = float(wrench[field_name])
        combined_value = float(combined[field_name])
        result[f"{label}_entropy_reduction_nats"] = {
            "actuator_only": source_value - actuator_value,
            "wrench_only": source_value - wrench_value,
            "actuator_and_wrench": source_value - combined_value,
        }
        result[f"{label}_factor_interaction_nats"] = (
            source_value - actuator_value - wrench_value + combined_value
        )
    return result


def _metric_attribution(arms: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    source_metrics = arms["object_prefix"]["component_metrics"]
    result: dict[str, Any] = {}
    for metric_name, source_summary in source_metrics.items():
        source_value = float(source_summary["posterior_expected_value"])
        actuator_value = float(
            arms["actuator_only"]["component_metrics"][metric_name][
                "posterior_expected_value"
            ]
        )
        wrench_value = float(
            arms["wrench_only"]["component_metrics"][metric_name][
                "posterior_expected_value"
            ]
        )
        combined_value = float(
            arms["actuator_and_wrench"]["component_metrics"][metric_name][
                "posterior_expected_value"
            ]
        )
        result[metric_name] = {
            "unit": source_summary["unit"],
            "expected_improvement_over_object_prefix": {
                "actuator_only": source_value - actuator_value,
                "wrench_only": source_value - wrench_value,
                "actuator_and_wrench": source_value - combined_value,
            },
            "combined_increment_over_best_single": (
                min(actuator_value, wrench_value) - combined_value
            ),
        }
    return result


@dataclass(frozen=True)
class IndependentSensorAblationReport:
    """Content-addressed diagnostic summary of four posterior evidence arms."""

    source_factual_intervention_id: str
    arm_summaries: Mapping[str, Any]
    attribution: Mapping[str, Any]
    evidence: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (
            type(self.source_factual_intervention_id) is not str
            or not self.source_factual_intervention_id
        ):
            raise ValueError("source_factual_intervention_id must be nonempty")
        arms = validated_json_mapping(
            self.arm_summaries,
            error_message="arm_summaries must contain finite JSON data",
        )
        if set(arms) != set(INDEPENDENT_SENSOR_ABLATION_ARMS):
            raise ValueError("arm_summaries must contain every registered arm")
        for arm_name, summary in arms.items():
            if not isinstance(summary, Mapping):
                raise ValueError("each arm summary must be a JSON object")
            if summary.get("arm_name") != arm_name:
                raise ValueError("arm summary name disagrees with its key")
            posterior_id = summary.get("posterior_id")
            if type(posterior_id) is not str or not posterior_id:
                raise ValueError("each arm summary needs a posterior_id")
            if type(summary.get("exact_source_fallback")) is not bool:
                raise ValueError("exact_source_fallback must be a boolean")
        object.__setattr__(self, "arm_summaries", arms)
        object.__setattr__(
            self,
            "attribution",
            validated_json_mapping(
                self.attribution,
                error_message="attribution must contain finite JSON data",
            ),
        )
        object.__setattr__(
            self,
            "evidence",
            validated_json_mapping(
                self.evidence,
                error_message="evidence must contain finite JSON data",
            ),
        )
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="metadata must contain finite JSON data",
            ),
        )

    def _identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": INDEPENDENT_SENSOR_ABLATION_SCHEMA_VERSION,
            "artifact_kind": INDEPENDENT_SENSOR_ABLATION_ARTIFACT_KIND,
            "source_factual_intervention_id": self.source_factual_intervention_id,
            "arm_order": list(INDEPENDENT_SENSOR_ABLATION_ARMS),
            "arms": plain_json(self.arm_summaries),
            "attribution": plain_json(self.attribution),
            "evidence": plain_json(self.evidence),
            "metadata": plain_json(self.metadata),
        }

    @property
    def artifact_id(self) -> str:
        return _canonical_sha256(self._identity_payload())

    def as_dict(self) -> dict[str, Any]:
        payload = self._identity_payload()
        payload["artifact_id"] = self.artifact_id
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IndependentSensorAblationReport:
        expected_fields = {
            "schema_version",
            "artifact_kind",
            "source_factual_intervention_id",
            "arm_order",
            "arms",
            "attribution",
            "evidence",
            "metadata",
            "artifact_id",
        }
        if set(payload) != expected_fields:
            raise ArtifactValidationError(
                "independent-sensor ablation report has an unexpected field inventory"
            )
        if (
            type(payload["schema_version"]) is not int
            or payload["schema_version"]
            != INDEPENDENT_SENSOR_ABLATION_SCHEMA_VERSION
        ):
            raise ArtifactValidationError(
                "unsupported independent-sensor ablation schema version"
            )
        if payload["artifact_kind"] != INDEPENDENT_SENSOR_ABLATION_ARTIFACT_KIND:
            raise ArtifactValidationError(
                "unexpected independent-sensor ablation artifact kind"
            )
        if payload["arm_order"] != list(INDEPENDENT_SENSOR_ABLATION_ARMS):
            raise ArtifactValidationError(
                "independent-sensor ablation arm order changed"
            )
        report = cls(
            source_factual_intervention_id=payload[
                "source_factual_intervention_id"
            ],
            arm_summaries=payload["arms"],
            attribution=payload["attribution"],
            evidence=payload["evidence"],
            metadata=payload["metadata"],
        )
        artifact_id = payload["artifact_id"]
        if type(artifact_id) is not str or artifact_id != report.artifact_id:
            raise ArtifactValidationError(
                "independent-sensor ablation artifact identity mismatch"
            )
        return report


@dataclass(frozen=True)
class IndependentSensorAblationResult:
    """Four posterior arms and their immutable attribution report."""

    object_prefix: FactualIntervention
    actuator_only: FactualIntervention
    wrench_only: FactualIntervention
    actuator_and_wrench: FactualIntervention
    report: IndependentSensorAblationReport

    def __post_init__(self) -> None:
        source = self.object_prefix
        for arm_name in INDEPENDENT_SENSOR_ABLATION_ARMS:
            posterior = getattr(self, arm_name)
            if posterior.component_ids != source.component_ids:
                raise ValueError("sensor-ablation component identities changed")
            if not np.array_equal(posterior.phi, source.phi):
                raise ValueError("sensor-ablation phi support changed")
            if not np.array_equal(posterior.kappa_obs, source.kappa_obs):
                raise ValueError("sensor-ablation kappa support changed")
            expected_id = self.report.arm_summaries[arm_name]["posterior_id"]
            if posterior.artifact_id != expected_id:
                raise ValueError("sensor-ablation report/posterior identity mismatch")
        if self.report.source_factual_intervention_id != source.artifact_id:
            raise ValueError("sensor-ablation report identifies another source")

    def posterior(self, arm_name: str) -> FactualIntervention:
        """Return one registered posterior arm by name."""

        if arm_name not in INDEPENDENT_SENSOR_ABLATION_ARMS:
            raise KeyError(f"unknown independent-sensor ablation arm: {arm_name}")
        return getattr(self, arm_name)


def build_independent_sensor_ablation(
    factual: FactualIntervention,
    *,
    actuator_evidence: ActuatorEvidence | None = None,
    predicted_actuator_positions_m: np.ndarray | None = None,
    predicted_actuator_variance_m2: np.ndarray | None = None,
    wrench_evidence: ContactWrenchEvidence | None = None,
    predicted_contact_wrench: np.ndarray | None = None,
    predicted_wrench_variance: np.ndarray | None = None,
    config: IndependentSensorAbductionConfig | None = None,
    component_metrics: Mapping[str, np.ndarray] | None = None,
    metric_units: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> IndependentSensorAblationResult:
    """Build object-prefix, single-factor, and combined diagnostic posteriors.

    Every arm starts from the same factual posterior, which has already consumed
    the allowed object-response prefix. The object likelihood is never replayed.
    Missing, zero-powered, invalid-only, or component-invariant factors preserve
    the exact input ``FactualIntervention`` object.
    """

    if "independent_sensor_abduction" in factual.metadata:
        raise ValueError(
            "source factual posterior must precede independent-sensor reweighting"
        )
    settings = config or IndependentSensorAbductionConfig()
    actuator_only = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=actuator_evidence,
        predicted_actuator_positions_m=predicted_actuator_positions_m,
        predicted_actuator_variance_m2=predicted_actuator_variance_m2,
        config=settings,
    )
    wrench_only = reweight_factual_intervention_with_independent_sensors(
        factual,
        wrench_evidence=wrench_evidence,
        predicted_contact_wrench=predicted_contact_wrench,
        predicted_wrench_variance=predicted_wrench_variance,
        config=settings,
    )
    combined = reweight_factual_intervention_with_independent_sensors(
        factual,
        actuator_evidence=actuator_evidence,
        predicted_actuator_positions_m=predicted_actuator_positions_m,
        predicted_actuator_variance_m2=predicted_actuator_variance_m2,
        wrench_evidence=wrench_evidence,
        predicted_contact_wrench=predicted_contact_wrench,
        predicted_wrench_variance=predicted_wrench_variance,
        config=settings,
    )
    metrics = _validated_component_metrics(
        component_metrics,
        component_count=len(factual.weights),
    )
    units = _validated_metric_units(
        metric_units,
        metric_names=set(metrics),
    )
    posteriors = {
        "object_prefix": factual,
        "actuator_only": actuator_only,
        "wrench_only": wrench_only,
        "actuator_and_wrench": combined,
    }
    arm_summaries = {
        arm_name: _arm_summary(
            arm_name,
            posterior,
            factual,
            component_metrics=metrics,
            metric_units=units,
        )
        for arm_name, posterior in posteriors.items()
    }
    actuator_prediction_shape = (
        None
        if predicted_actuator_positions_m is None
        else np.asarray(predicted_actuator_positions_m, dtype=float).shape
    )
    wrench_prediction_shape = (
        None
        if predicted_contact_wrench is None
        else np.asarray(predicted_contact_wrench, dtype=float).shape
    )
    evidence = {
        "actuator_evidence_id": (
            None if actuator_evidence is None else actuator_evidence.artifact_id
        ),
        "wrench_evidence_id": (
            None if wrench_evidence is None else wrench_evidence.artifact_id
        ),
        "actuator_clock_id": (
            None if actuator_evidence is None else actuator_evidence.clock_id
        ),
        "wrench_clock_id": (
            None if wrench_evidence is None else wrench_evidence.clock_id
        ),
        "predicted_actuator_positions_m": _array_binding(
            predicted_actuator_positions_m
        ),
        "predicted_actuator_variance_m2": _broadcast_array_binding(
            predicted_actuator_variance_m2,
            expected_shape=actuator_prediction_shape,
        ),
        "predicted_contact_wrench": _array_binding(predicted_contact_wrench),
        "predicted_wrench_variance": _broadcast_array_binding(
            predicted_wrench_variance,
            expected_shape=wrench_prediction_shape,
        ),
        "common_clock_verified": bool(
            actuator_evidence is None
            or wrench_evidence is None
            or actuator_evidence.clock_id == wrench_evidence.clock_id
        ),
        "config": asdict(settings),
        "object_observation_likelihood_reused": False,
        "future_object_frames_read": 0,
    }
    report = IndependentSensorAblationReport(
        source_factual_intervention_id=factual.artifact_id,
        arm_summaries=arm_summaries,
        attribution={
            **_entropy_attribution(arm_summaries),
            "component_metrics": _metric_attribution(arm_summaries),
        },
        evidence=evidence,
        metadata={} if metadata is None else metadata,
    )
    return IndependentSensorAblationResult(
        object_prefix=factual,
        actuator_only=actuator_only,
        wrench_only=wrench_only,
        actuator_and_wrench=combined,
        report=report,
    )


def save_independent_sensor_ablation_report(
    path: str | Path,
    report: IndependentSensorAblationReport,
    *,
    overwrite: bool = False,
) -> None:
    """Publish a validated report atomically and exactly once by default."""

    if type(overwrite) is not bool:
        raise TypeError("overwrite must be an exact boolean")
    encoded = (
        json.dumps(
            report.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")

    def write_report(handle: Any) -> None:
        handle.write(encoded)

    def validate_report(candidate: Path) -> None:
        restored = load_independent_sensor_ablation_report(candidate)
        if restored.artifact_id != report.artifact_id:
            raise ArtifactValidationError(
                "published independent-sensor ablation identity changed"
            )

    atomic_write_binary(
        path,
        write_report,
        overwrite=overwrite,
        validate=validate_report,
    )


def load_independent_sensor_ablation_report(
    path: str | Path,
) -> IndependentSensorAblationReport:
    """Load exact ordinary-file bytes and verify the closed report contract."""

    snapshot = read_regular_file(
        path,
        name="independent-sensor ablation report",
    )
    payload = load_strict_json_object(
        snapshot.payload,
        name="independent-sensor ablation report",
    )
    return IndependentSensorAblationReport.from_dict(payload)


__all__ = [
    "INDEPENDENT_SENSOR_ABLATION_ARMS",
    "INDEPENDENT_SENSOR_ABLATION_SCHEMA_VERSION",
    "IndependentSensorAblationReport",
    "IndependentSensorAblationResult",
    "build_independent_sensor_ablation",
    "load_independent_sensor_ablation_report",
    "save_independent_sensor_ablation_report",
]
