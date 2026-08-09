"""Content-addressed per-view evidence retained beside fused observations.

The physical acquisition manifest historically retained a synchronized RGB-D
manifest and a fused observation product.  Those artifacts are sufficient to
replay the registered estimator, but they are not sufficient to distinguish
view-specific observation bias from coherent physical-model discrepancy.  This
module defines an additive, non-claim-bearing retention contract for the raw
per-view evidence and its sensor context.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from causal4d.artifact_io import (
    ArtifactFileSnapshot,
    load_strict_json_object,
    read_regular_file,
    read_regular_file_beneath,
)
from causal4d.atomic_io import atomic_write_binary


PER_VIEW_OBSERVATION_SCHEMA = "causal4d.per-view-observation-evidence"
PER_VIEW_OBSERVATION_SCHEMA_VERSION = 1
PER_VIEW_OBSERVATION_ARTIFACT_KIND = "Causal4DPerViewObservationEvidence"

_TOP_LEVEL_FIELDS = frozenset(
    {
        "schema",
        "schema_version",
        "artifact_kind",
        "artifact_id",
        "protocol_id",
        "protocol_design_sha256",
        "execution_id",
        "session_id",
        "clock_domain_id",
        "frame_count",
        "common_coordinate_frame",
        "material_identity_contract",
        "observation_producer",
        "camera_calibration",
        "object_frame",
        "confidence_semantics",
        "views",
        "shared_sensors",
        "fused_observation",
        "information_boundary",
    }
)
_PRODUCER_FIELDS = frozenset(
    {
        "name",
        "version",
        "artifact_contract",
        "software_environment_capsule_id",
    }
)
_CAMERA_CALIBRATION_FIELDS = frozenset({"revision_id", "descriptor", "camera_keys"})
_OBJECT_FRAME_FIELDS = frozenset({"frame_id", "definition", "object_from_world"})
_CONFIDENCE_FIELDS = frozenset(
    {
        "continuous",
        "higher_is_better",
        "minimum",
        "maximum",
        "missing_value_policy",
    }
)
_VIEW_FIELDS = frozenset(
    {
        "camera_id",
        "calibration_camera_key",
        "material_point_count",
        "rgb_stream",
        "depth_stream",
        "timestamps",
        "material_points",
        "validity_mask",
        "confidence",
        "surface_normals",
        "surface_normals_unavailable_reason",
    }
)
_SHARED_SENSOR_FIELDS = frozenset(
    {
        "commanded_control",
        "measured_actuation",
        "gripper_state",
        "contact_wrench",
        "contact_wrench_unavailable_reason",
        "support_registration",
        "reset_drift_slip",
    }
)
_FUSED_FIELDS = frozenset(
    {
        "descriptor",
        "source_camera_ids",
        "aggregation_method",
        "material_point_count",
        "derived_from_per_view_evidence",
        "sole_retained_observation",
    }
)
_INFORMATION_BOUNDARY_FIELDS = frozenset(
    {
        "causal_prefix_frame_start",
        "causal_prefix_frame_stop",
        "raw_full_execution_retained",
        "future_frames_retained_for_blind_evaluation",
        "future_frames_used_for_inference",
        "target_outcomes_used_for_inference",
        "target_outcomes_used_for_model_selection",
        "target_outcomes_used_for_exclusion",
        "target_outcomes_used_for_calibration",
        "fused_observation_is_sole_retained_evidence",
    }
)
_STATIC_DESCRIPTOR_FIELDS = frozenset({"path", "sha256", "bytes", "media_type"})
_TIMED_DESCRIPTOR_FIELDS = frozenset(
    {"path", "sha256", "bytes", "media_type", "clock_id", "sample_count"}
)
_LOWER_HEX = frozenset("0123456789abcdef")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{name} must be a mapping")
    _require(
        all(type(key) is str for key in value),
        f"{name} keys must be strings",
    )
    return value


def _require_exact_fields(
    value: Any,
    *,
    name: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, name=name)
    actual = set(mapping)
    missing = sorted(fields - actual)
    unexpected = sorted(actual - fields)
    _require(
        not missing and not unexpected,
        f"{name} fields do not match schema; "
        f"missing={missing}, unexpected={unexpected}",
    )
    return mapping


def _require_nonempty_string(value: Any, *, name: str) -> str:
    _require(type(value) is str and bool(value), f"{name} must be a nonempty string")
    return value


def _require_sha256(value: Any, *, name: str) -> str:
    _require(
        type(value) is str
        and len(value) == 64
        and all(character in _LOWER_HEX for character in value),
        f"{name} must be a lowercase SHA-256 digest",
    )
    return value


def _require_integer(value: Any, *, name: str, minimum: int = 0) -> int:
    _require(
        type(value) is int and value >= minimum,
        f"{name} must be an integer greater than or equal to {minimum}",
    )
    return value


def _require_bool(value: Any, *, name: str) -> bool:
    _require(type(value) is bool, f"{name} must be a boolean")
    return value


def _require_finite(value: Any, *, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _safe_posix_relative_path(value: Any, *, name: str) -> str:
    path = _require_nonempty_string(value, name=f"{name} path")
    _require(
        "\x00" not in path
        and "\\" not in path
        and not path.startswith("/")
        and not path.endswith("/")
        and "//" not in path,
        f"{name} path must be a safe POSIX relative path",
    )
    parts = tuple(path.split("/"))
    _require(
        all(part not in {"", ".", ".."} for part in parts),
        f"{name} path must be a safe POSIX relative path",
    )
    return path


def _canonical_sha256(value: Mapping[str, Any]) -> str:
    payload = deepcopy(dict(value))
    payload.pop("artifact_id", None)
    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(
            "per-view observation evidence must contain finite JSON data"
        ) from error
    return hashlib.sha256(encoded).hexdigest()


def _plain_json_mapping(value: Mapping[str, Any], *, name: str) -> dict[str, Any]:
    try:
        encoded = json.dumps(
            dict(value),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite JSON data") from error
    return dict(_require_mapping(decoded, name=name))


def _read_artifact_snapshot(
    path: str | Path,
    *,
    artifact_root: str | Path | None,
    name: str,
) -> ArtifactFileSnapshot:
    if artifact_root is None:
        return read_regular_file(path, name=name)
    return read_regular_file_beneath(
        artifact_root,
        str(path),
        name=name,
    )


def _validate_descriptor(
    value: Any,
    *,
    name: str,
    timed: bool,
    clock_domain_id: str,
    required_sample_count: int | None,
    artifact_root: str | Path | None,
    verify_files: bool,
    seen_paths: dict[str, str],
) -> Mapping[str, Any]:
    fields = _TIMED_DESCRIPTOR_FIELDS if timed else _STATIC_DESCRIPTOR_FIELDS
    descriptor = _require_exact_fields(value, name=name, fields=fields)
    path = _safe_posix_relative_path(descriptor["path"], name=name)
    previous = seen_paths.get(path)
    _require(
        previous is None,
        f"{name} reuses artifact path already bound to {previous}: {path}",
    )
    seen_paths[path] = name
    expected_sha256 = _require_sha256(descriptor["sha256"], name=f"{name} SHA-256")
    expected_bytes = _require_integer(descriptor["bytes"], name=f"{name} bytes")
    _require_nonempty_string(descriptor["media_type"], name=f"{name} media_type")
    if timed:
        _require(
            descriptor["clock_id"] == clock_domain_id,
            f"{name} clock domain differs from the execution clock",
        )
        sample_count = _require_integer(
            descriptor["sample_count"],
            name=f"{name} sample_count",
            minimum=1,
        )
        if required_sample_count is not None:
            _require(
                sample_count == required_sample_count,
                f"{name} sample_count differs from the registered frame count",
            )
    if verify_files:
        _require(
            artifact_root is not None,
            "artifact_root is required when verify_files is true",
        )
        snapshot = _read_artifact_snapshot(
            path,
            artifact_root=artifact_root,
            name=name,
        )
        _require(snapshot.sha256 == expected_sha256, f"{name} checksum mismatch")
        _require(snapshot.byte_count == expected_bytes, f"{name} byte count mismatch")
    return descriptor


def _optional_timed_descriptor(
    value: Any,
    reason: Any,
    *,
    name: str,
    clock_domain_id: str,
    required_sample_count: int | None,
    artifact_root: str | Path | None,
    verify_files: bool,
    seen_paths: dict[str, str],
) -> None:
    if value is None:
        _require_nonempty_string(reason, name=f"{name}_unavailable_reason")
        return
    _require(reason is None, f"{name}_unavailable_reason must be null when retained")
    _validate_descriptor(
        value,
        name=name,
        timed=True,
        clock_domain_id=clock_domain_id,
        required_sample_count=required_sample_count,
        artifact_root=artifact_root,
        verify_files=verify_files,
        seen_paths=seen_paths,
    )


def validate_per_view_observation_evidence(
    value: Mapping[str, Any],
    *,
    artifact_root: str | Path | None = None,
    verify_files: bool = False,
    expected_protocol_id: str | None = None,
    expected_protocol_design_sha256: str | None = None,
    expected_execution_id: str | None = None,
    expected_session_id: str | None = None,
    expected_clock_domain_id: str | None = None,
    expected_frame_count: int | None = None,
    expected_causal_prefix_frame_stop: int | None = None,
) -> dict[str, Any]:
    """Validate one per-view evidence manifest and optionally all bound files."""

    payload = _require_exact_fields(
        value,
        name="per-view observation evidence",
        fields=_TOP_LEVEL_FIELDS,
    )
    _require(
        payload["schema"] == PER_VIEW_OBSERVATION_SCHEMA,
        "per-view observation schema changed",
    )
    _require(
        payload["schema_version"] == PER_VIEW_OBSERVATION_SCHEMA_VERSION,
        "unsupported per-view observation schema version",
    )
    _require(
        payload["artifact_kind"] == PER_VIEW_OBSERVATION_ARTIFACT_KIND,
        "unexpected per-view observation artifact kind",
    )
    artifact_id = _require_sha256(payload["artifact_id"], name="artifact_id")
    _require(
        artifact_id == _canonical_sha256(payload),
        "per-view observation artifact ID does not match its contents",
    )

    protocol_id = _require_nonempty_string(payload["protocol_id"], name="protocol_id")
    design_sha256 = _require_sha256(
        payload["protocol_design_sha256"],
        name="protocol_design_sha256",
    )
    execution_id = _require_nonempty_string(
        payload["execution_id"],
        name="execution_id",
    )
    session_id = _require_nonempty_string(payload["session_id"], name="session_id")
    clock_domain_id = _require_nonempty_string(
        payload["clock_domain_id"],
        name="clock_domain_id",
    )
    frame_count = _require_integer(
        payload["frame_count"],
        name="frame_count",
        minimum=2,
    )
    _require_nonempty_string(
        payload["common_coordinate_frame"],
        name="common_coordinate_frame",
    )
    _require_nonempty_string(
        payload["material_identity_contract"],
        name="material_identity_contract",
    )

    expected_values = (
        (expected_protocol_id, protocol_id, "protocol_id"),
        (expected_protocol_design_sha256, design_sha256, "protocol_design_sha256"),
        (expected_execution_id, execution_id, "execution_id"),
        (expected_session_id, session_id, "session_id"),
        (expected_clock_domain_id, clock_domain_id, "clock_domain_id"),
        (expected_frame_count, frame_count, "frame_count"),
    )
    for expected, actual, name in expected_values:
        if expected is not None:
            _require(actual == expected, f"per-view observation {name} binding changed")

    producer = _require_exact_fields(
        payload["observation_producer"],
        name="observation_producer",
        fields=_PRODUCER_FIELDS,
    )
    for field in ("name", "version", "artifact_contract"):
        _require_nonempty_string(producer[field], name=f"observation_producer.{field}")
    _require_sha256(
        producer["software_environment_capsule_id"],
        name="observation_producer.software_environment_capsule_id",
    )

    seen_paths: dict[str, str] = {}
    calibration = _require_exact_fields(
        payload["camera_calibration"],
        name="camera_calibration",
        fields=_CAMERA_CALIBRATION_FIELDS,
    )
    _require_nonempty_string(
        calibration["revision_id"],
        name="camera_calibration.revision_id",
    )
    _validate_descriptor(
        calibration["descriptor"],
        name="camera_calibration.descriptor",
        timed=False,
        clock_domain_id=clock_domain_id,
        required_sample_count=None,
        artifact_root=artifact_root,
        verify_files=verify_files,
        seen_paths=seen_paths,
    )

    object_frame = _require_exact_fields(
        payload["object_frame"],
        name="object_frame",
        fields=_OBJECT_FRAME_FIELDS,
    )
    _require_nonempty_string(object_frame["frame_id"], name="object_frame.frame_id")
    _validate_descriptor(
        object_frame["definition"],
        name="object_frame.definition",
        timed=False,
        clock_domain_id=clock_domain_id,
        required_sample_count=None,
        artifact_root=artifact_root,
        verify_files=verify_files,
        seen_paths=seen_paths,
    )
    _validate_descriptor(
        object_frame["object_from_world"],
        name="object_frame.object_from_world",
        timed=True,
        clock_domain_id=clock_domain_id,
        required_sample_count=frame_count,
        artifact_root=artifact_root,
        verify_files=verify_files,
        seen_paths=seen_paths,
    )

    confidence = _require_exact_fields(
        payload["confidence_semantics"],
        name="confidence_semantics",
        fields=_CONFIDENCE_FIELDS,
    )
    _require(confidence["continuous"] is True, "confidence values must be continuous")
    _require_bool(
        confidence["higher_is_better"],
        name="confidence_semantics.higher_is_better",
    )
    minimum = _require_finite(
        confidence["minimum"],
        name="confidence_semantics.minimum",
    )
    maximum = _require_finite(
        confidence["maximum"],
        name="confidence_semantics.maximum",
    )
    _require(maximum > minimum, "confidence maximum must exceed its minimum")
    _require_nonempty_string(
        confidence["missing_value_policy"],
        name="confidence_semantics.missing_value_policy",
    )

    views = payload["views"]
    _require(
        isinstance(views, Sequence) and not isinstance(views, (str, bytes)),
        "views must be a sequence",
    )
    _require(len(views) >= 2, "per-view evidence requires at least two camera views")
    camera_ids: list[str] = []
    camera_keys: list[str] = []
    point_counts: list[int] = []
    surface_normal_count = 0
    for index, raw_view in enumerate(views):
        name = f"views[{index}]"
        view = _require_exact_fields(raw_view, name=name, fields=_VIEW_FIELDS)
        camera_id = _require_nonempty_string(
            view["camera_id"],
            name=f"{name}.camera_id",
        )
        camera_key = _require_nonempty_string(
            view["calibration_camera_key"],
            name=f"{name}.calibration_camera_key",
        )
        point_count = _require_integer(
            view["material_point_count"],
            name=f"{name}.material_point_count",
            minimum=1,
        )
        camera_ids.append(camera_id)
        camera_keys.append(camera_key)
        point_counts.append(point_count)
        for field in (
            "rgb_stream",
            "depth_stream",
            "timestamps",
            "material_points",
            "validity_mask",
            "confidence",
        ):
            _validate_descriptor(
                view[field],
                name=f"{name}.{field}",
                timed=True,
                clock_domain_id=clock_domain_id,
                required_sample_count=frame_count,
                artifact_root=artifact_root,
                verify_files=verify_files,
                seen_paths=seen_paths,
            )
        normals = view["surface_normals"]
        reason = view["surface_normals_unavailable_reason"]
        _optional_timed_descriptor(
            normals,
            reason,
            name=f"{name}.surface_normals",
            clock_domain_id=clock_domain_id,
            required_sample_count=frame_count,
            artifact_root=artifact_root,
            verify_files=verify_files,
            seen_paths=seen_paths,
        )
        if normals is not None:
            surface_normal_count += 1
    _require(len(set(camera_ids)) == len(camera_ids), "camera IDs must be unique")
    _require(
        len(set(camera_keys)) == len(camera_keys),
        "calibration camera keys must be unique",
    )
    _require(len(set(point_counts)) == 1, "material point counts differ across views")

    declared_camera_keys = calibration["camera_keys"]
    _require(
        isinstance(declared_camera_keys, Sequence)
        and not isinstance(declared_camera_keys, (str, bytes))
        and list(declared_camera_keys) == camera_keys,
        "camera calibration keys must exactly match the ordered view inventory",
    )

    sensors = _require_exact_fields(
        payload["shared_sensors"],
        name="shared_sensors",
        fields=_SHARED_SENSOR_FIELDS,
    )
    for field in ("commanded_control", "measured_actuation", "gripper_state"):
        _validate_descriptor(
            sensors[field],
            name=f"shared_sensors.{field}",
            timed=True,
            clock_domain_id=clock_domain_id,
            required_sample_count=None,
            artifact_root=artifact_root,
            verify_files=verify_files,
            seen_paths=seen_paths,
        )
    _optional_timed_descriptor(
        sensors["contact_wrench"],
        sensors["contact_wrench_unavailable_reason"],
        name="shared_sensors.contact_wrench",
        clock_domain_id=clock_domain_id,
        required_sample_count=None,
        artifact_root=artifact_root,
        verify_files=verify_files,
        seen_paths=seen_paths,
    )
    for field in ("support_registration", "reset_drift_slip"):
        _validate_descriptor(
            sensors[field],
            name=f"shared_sensors.{field}",
            timed=False,
            clock_domain_id=clock_domain_id,
            required_sample_count=None,
            artifact_root=artifact_root,
            verify_files=verify_files,
            seen_paths=seen_paths,
        )

    fused = _require_exact_fields(
        payload["fused_observation"],
        name="fused_observation",
        fields=_FUSED_FIELDS,
    )
    _validate_descriptor(
        fused["descriptor"],
        name="fused_observation.descriptor",
        timed=True,
        clock_domain_id=clock_domain_id,
        required_sample_count=frame_count,
        artifact_root=artifact_root,
        verify_files=verify_files,
        seen_paths=seen_paths,
    )
    _require(
        isinstance(fused["source_camera_ids"], Sequence)
        and not isinstance(fused["source_camera_ids"], (str, bytes))
        and list(fused["source_camera_ids"]) == camera_ids,
        "fused observation source cameras must match the ordered view inventory",
    )
    _require_nonempty_string(
        fused["aggregation_method"],
        name="fused_observation.aggregation_method",
    )
    fused_point_count = _require_integer(
        fused["material_point_count"],
        name="fused_observation.material_point_count",
        minimum=1,
    )
    _require(
        fused_point_count == point_counts[0],
        "fused observation material point count differs from the views",
    )
    _require(
        fused["derived_from_per_view_evidence"] is True,
        "fused observation must be declared as a derived product",
    )
    _require(
        fused["sole_retained_observation"] is False,
        "fused observation cannot be the sole retained evidence",
    )

    boundary = _require_exact_fields(
        payload["information_boundary"],
        name="information_boundary",
        fields=_INFORMATION_BOUNDARY_FIELDS,
    )
    _require(
        boundary["causal_prefix_frame_start"] == 0,
        "causal prefix must start at frame zero",
    )
    prefix_stop = _require_integer(
        boundary["causal_prefix_frame_stop"],
        name="information_boundary.causal_prefix_frame_stop",
        minimum=1,
    )
    _require(
        prefix_stop < frame_count,
        "causal prefix must stop before the retained future",
    )
    if expected_causal_prefix_frame_stop is not None:
        _require(
            prefix_stop == expected_causal_prefix_frame_stop,
            "per-view observation causal-prefix binding changed",
        )
    required_true = (
        "raw_full_execution_retained",
        "future_frames_retained_for_blind_evaluation",
    )
    required_false = (
        "future_frames_used_for_inference",
        "target_outcomes_used_for_inference",
        "target_outcomes_used_for_model_selection",
        "target_outcomes_used_for_exclusion",
        "target_outcomes_used_for_calibration",
        "fused_observation_is_sole_retained_evidence",
    )
    for field in required_true:
        _require(boundary[field] is True, f"information_boundary.{field} must be true")
    for field in required_false:
        _require(
            boundary[field] is False,
            f"information_boundary.{field} must be false",
        )

    return {
        "valid": True,
        "artifact_id": artifact_id,
        "protocol_id": protocol_id,
        "execution_id": execution_id,
        "session_id": session_id,
        "clock_domain_id": clock_domain_id,
        "frame_count": frame_count,
        "causal_prefix_frame_stop": prefix_stop,
        "camera_count": len(camera_ids),
        "camera_ids": camera_ids,
        "material_point_count": point_counts[0],
        "surface_normal_view_count": surface_normal_count,
        "contact_wrench_retained": sensors["contact_wrench"] is not None,
        "bound_file_count": len(seen_paths),
        "file_hashes_verified": True if verify_files else None,
        "target_outcomes_used": False,
    }


def build_per_view_observation_evidence(
    *,
    protocol_id: str,
    protocol_design_sha256: str,
    execution_id: str,
    session_id: str,
    clock_domain_id: str,
    frame_count: int,
    common_coordinate_frame: str,
    material_identity_contract: str,
    observation_producer: Mapping[str, Any],
    camera_calibration: Mapping[str, Any],
    object_frame: Mapping[str, Any],
    confidence_semantics: Mapping[str, Any],
    views: Sequence[Mapping[str, Any]],
    shared_sensors: Mapping[str, Any],
    fused_observation: Mapping[str, Any],
    information_boundary: Mapping[str, Any],
) -> dict[str, Any]:
    """Build and validate one finite, content-addressed retention manifest."""

    payload = _plain_json_mapping(
        {
            "schema": PER_VIEW_OBSERVATION_SCHEMA,
            "schema_version": PER_VIEW_OBSERVATION_SCHEMA_VERSION,
            "artifact_kind": PER_VIEW_OBSERVATION_ARTIFACT_KIND,
            "protocol_id": protocol_id,
            "protocol_design_sha256": protocol_design_sha256,
            "execution_id": execution_id,
            "session_id": session_id,
            "clock_domain_id": clock_domain_id,
            "frame_count": frame_count,
            "common_coordinate_frame": common_coordinate_frame,
            "material_identity_contract": material_identity_contract,
            "observation_producer": observation_producer,
            "camera_calibration": camera_calibration,
            "object_frame": object_frame,
            "confidence_semantics": confidence_semantics,
            "views": list(views),
            "shared_sensors": shared_sensors,
            "fused_observation": fused_observation,
            "information_boundary": information_boundary,
        },
        name="per-view observation evidence",
    )
    payload["artifact_id"] = _canonical_sha256(payload)
    validate_per_view_observation_evidence(payload)
    return payload


def load_per_view_observation_evidence(
    path: str | Path,
    *,
    artifact_root: str | Path | None = None,
    verify_files: bool = False,
    expected_file_sha256: str | None = None,
    expected_file_bytes: int | None = None,
    **expected_bindings: Any,
) -> dict[str, Any]:
    """Load one strict JSON manifest from an ordinary file and validate it."""

    snapshot = _read_artifact_snapshot(
        path,
        artifact_root=artifact_root,
        name="per-view observation evidence",
    )
    if expected_file_sha256 is not None:
        _require_sha256(expected_file_sha256, name="expected manifest SHA-256")
        _require(
            snapshot.sha256 == expected_file_sha256,
            "per-view observation manifest checksum mismatch",
        )
    if expected_file_bytes is not None:
        _require_integer(
            expected_file_bytes,
            name="expected manifest bytes",
            minimum=1,
        )
        _require(
            snapshot.byte_count == expected_file_bytes,
            "per-view observation manifest byte count mismatch",
        )
    payload = load_strict_json_object(
        snapshot.payload,
        name="per-view observation evidence",
    )
    validate_per_view_observation_evidence(
        payload,
        artifact_root=artifact_root,
        verify_files=verify_files,
        **expected_bindings,
    )
    return payload


def write_per_view_observation_evidence(
    path: str | Path,
    value: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    """Publish one validated manifest atomically and exactly once by default."""

    payload = _plain_json_mapping(value, name="per-view observation evidence")
    validate_per_view_observation_evidence(payload)
    serialized = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")

    def validate_temporary(temporary: Path) -> None:
        snapshot = read_regular_file(
            temporary,
            name="temporary per-view observation evidence",
        )
        restored = load_strict_json_object(
            snapshot.payload,
            name="temporary per-view observation evidence",
        )
        validate_per_view_observation_evidence(restored)
        _require(restored == payload, "published per-view observation bytes changed")

    target = Path(path)
    atomic_write_binary(
        target,
        lambda handle: handle.write(serialized),
        overwrite=overwrite,
        validate=validate_temporary,
    )
    restored = load_per_view_observation_evidence(target)
    _require(restored == payload, "published per-view observation manifest changed")
    return target


__all__ = [
    "PER_VIEW_OBSERVATION_ARTIFACT_KIND",
    "PER_VIEW_OBSERVATION_SCHEMA",
    "PER_VIEW_OBSERVATION_SCHEMA_VERSION",
    "build_per_view_observation_evidence",
    "load_per_view_observation_evidence",
    "validate_per_view_observation_evidence",
    "write_per_view_observation_evidence",
]
