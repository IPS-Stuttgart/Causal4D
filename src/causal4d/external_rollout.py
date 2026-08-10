"""Portable import boundary for external finite physical rollout banks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from causal4d._external_rollout_schema import (
    EXTERNAL_ROLLOUT_BANK_SCHEMA,
    EXTERNAL_ROLLOUT_BANK_SCHEMA_VERSION,
    EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
    EXTERNAL_ROLLOUT_IMPORT_SCHEMA_VERSION,
    ExternalRolloutBundle,
    _BANK_DESCRIPTOR_FIELDS,
    _BANK_SOURCE_FIELDS,
    _COORDINATE_FRAMES,
    _IMPORT_ARRAY_OPTIONAL_FIELDS,
    _IMPORT_ARRAY_REQUIRED_FIELDS,
    _IMPORT_OPTIONAL_FIELDS,
    _IMPORT_REQUIRED_FIELDS,
    _SOURCE_OPTIONAL_FIELDS,
    _SOURCE_REQUIRED_FIELDS,
    _SUPPORTED_LAYOUTS,
    _UNIT_SCALE_TO_M,
    _camera_to_world,
    _file_sha256,
    _load_json_mapping,
    _require_fields,
    _require_finite_number,
    _require_integer,
    _require_mapping,
    _require_nonempty_string,
    _require_optional_string,
    _require_positive_number,
    _source_array,
    _text_vector,
    _validate_sha256,
    _validated_string_tuple,
)
from causal4d.contracts import array_sha256
from causal4d.immutable_array import readonly_integer_array
from causal4d.immutable_json import plain_json, validated_json_mapping
from causal4d.rollout_bank import JointRolloutBank
from causal4d.rollout_bank_io import load_rollout_bank, save_rollout_bank


def import_external_rollouts(
    source_npz: str | Path,
    import_manifest_json: str | Path,
) -> ExternalRolloutBundle:
    """Normalize one flat external rollout support through a strict manifest."""

    manifest_hash = _file_sha256(import_manifest_json)
    manifest = _require_fields(
        _load_json_mapping(import_manifest_json, name="external rollout manifest"),
        name="external rollout manifest",
        required=_IMPORT_REQUIRED_FIELDS,
        optional=_IMPORT_OPTIONAL_FIELDS,
    )
    if _file_sha256(import_manifest_json) != manifest_hash:
        raise ValueError("external rollout manifest changed during import")
    if (
        _require_nonempty_string(manifest["schema"], name="schema")
        != EXTERNAL_ROLLOUT_IMPORT_SCHEMA
    ):
        raise ValueError("unexpected external rollout import schema")
    if (
        _require_integer(manifest["schema_version"], name="schema_version", minimum=1)
        != EXTERNAL_ROLLOUT_IMPORT_SCHEMA_VERSION
    ):
        raise ValueError("unsupported external rollout import schema version")

    case_id = _require_nonempty_string(manifest["case_id"], name="case_id")
    source = _require_fields(
        manifest["source"],
        name="source",
        required=_SOURCE_REQUIRED_FIELDS,
        optional=_SOURCE_OPTIONAL_FIELDS,
    )
    source_simulator = _require_nonempty_string(
        source["simulator"],
        name="source.simulator",
    )
    source_revision = _require_optional_string(
        source.get("revision"),
        name="source.revision",
    )
    source_artifact_id = _require_optional_string(
        source.get("artifact_id"),
        name="source.artifact_id",
    )
    arrays = _require_fields(
        manifest["arrays"],
        name="arrays",
        required=_IMPORT_ARRAY_REQUIRED_FIELDS,
        optional=_IMPORT_ARRAY_OPTIONAL_FIELDS,
    )
    layout = _require_nonempty_string(manifest["layout"], name="layout")
    if layout not in _SUPPORTED_LAYOUTS:
        raise ValueError(f"layout must be one of {sorted(_SUPPORTED_LAYOUTS)}")
    coordinate_frame = _require_nonempty_string(
        manifest["coordinate_frame"],
        name="coordinate_frame",
    )
    if coordinate_frame not in _COORDINATE_FRAMES:
        raise ValueError(
            f"coordinate_frame must be one of {sorted(_COORDINATE_FRAMES)}"
        )
    position_unit = _require_nonempty_string(
        manifest["position_unit"],
        name="position_unit",
    )
    if position_unit not in _UNIT_SCALE_TO_M:
        raise ValueError(f"position_unit must be one of {sorted(_UNIT_SCALE_TO_M)}")
    anchor_time_s = _require_finite_number(
        manifest["anchor_time_s"],
        name="anchor_time_s",
    )
    parameter_names = _validated_string_tuple(
        manifest.get("parameter_names", ()),
        name="parameter_names",
        allow_empty=True,
    )
    variance_floor_m2 = _require_positive_number(
        manifest.get("variance_floor_m2", 1e-6),
        name="variance_floor_m2",
    )
    confidence_level = _require_finite_number(
        manifest.get("confidence_level", 0.90),
        name="confidence_level",
    )
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie in (0, 1)")
    producer_metadata = validated_json_mapping(
        _require_mapping(manifest.get("metadata", {}), name="metadata"),
        error_message="manifest metadata must be finite JSON data",
    )

    source_hash = _file_sha256(source_npz)
    with np.load(source_npz, allow_pickle=False) as archive:
        nodes = readonly_integer_array(
            _source_array(archive, arrays["node_ids"], name="node_ids"),
            name="node_ids",
        )
        trajectories = np.asarray(
            _source_array(archive, arrays["trajectories"], name="trajectories"),
            dtype=np.float64,
        )
        frame_times = np.asarray(
            _source_array(archive, arrays["frame_times_s"], name="frame_times_s"),
            dtype=np.float64,
        )
        rollout_weights = np.asarray(
            _source_array(
                archive,
                arrays["rollout_weights"],
                name="rollout_weights",
            ),
            dtype=np.float64,
        )
        if "rollout_ids" in arrays:
            rollout_ids = _text_vector(
                _source_array(
                    archive,
                    arrays["rollout_ids"],
                    name="rollout_ids",
                ),
                name="rollout_ids",
            )
        else:
            rollout_ids = tuple(
                f"rollout_{index:04d}" for index in range(trajectories.shape[0])
            )
        parameter_values = None
        if "parameter_values" in arrays:
            parameter_values = np.asarray(
                _source_array(
                    archive,
                    arrays["parameter_values"],
                    name="parameter_values",
                ),
                dtype=np.float64,
            )
        transform = None
        if "camera_to_world" in arrays:
            transform = np.asarray(
                _source_array(
                    archive,
                    arrays["camera_to_world"],
                    name="camera_to_world",
                ),
                dtype=np.float64,
            )

    if _file_sha256(source_npz) != source_hash:
        raise ValueError("source NPZ changed during import")
    if nodes.ndim != 1 or not len(nodes):
        raise ValueError("node_ids must be a nonempty vector")
    if np.any(nodes < 0) or len(np.unique(nodes)) != len(nodes):
        raise ValueError("node_ids must be unique and nonnegative")
    if trajectories.ndim != 4 or trajectories.shape[-1] not in {2, 3}:
        raise ValueError("RTNC trajectories must have shape (R, T, N, 2|3)")
    rollout_count, frame_count, node_count, coordinate_count = trajectories.shape
    if rollout_count < 1 or frame_count < 2 or node_count != len(nodes):
        raise ValueError("rollout trajectories must have nonempty matching dimensions")
    if not np.all(np.isfinite(trajectories)):
        raise ValueError("rollout trajectories must be finite")
    if frame_times.shape != (frame_count,) or not np.all(np.isfinite(frame_times)):
        raise ValueError("frame_times_s must be a finite vector matching T")
    if np.any(np.diff(frame_times) <= 0.0):
        raise ValueError("frame_times_s must be strictly increasing")
    if len(rollout_ids) != rollout_count:
        raise ValueError("rollout_ids must identify every rollout")
    if rollout_weights.shape != (rollout_count,):
        raise ValueError("rollout_weights must identify every rollout")
    if not np.all(np.isfinite(rollout_weights)) or np.any(rollout_weights < 0.0):
        raise ValueError("rollout_weights must be finite and nonnegative")
    if float(np.sum(rollout_weights)) <= 0.0:
        raise ValueError("rollout_weights must have positive mass")
    if parameter_values is None:
        if parameter_names:
            raise ValueError("parameter_names require arrays.parameter_values")
    else:
        if parameter_values.shape != (rollout_count, len(parameter_names)):
            raise ValueError("parameter_values must have shape (R, D)")
        if not np.all(np.isfinite(parameter_values)):
            raise ValueError("parameter_values must be finite")

    scale = _UNIT_SCALE_TO_M[position_unit]
    trajectories = trajectories * scale
    camera_to_world_sha256 = None
    if coordinate_frame == "camera":
        if transform is None:
            raise ValueError("camera-frame rollouts require arrays.camera_to_world")
        if coordinate_count != 3:
            raise ValueError("camera-frame transformation requires 3-D trajectories")
        camera_to_world_sha256 = array_sha256(transform)
        trajectories = _camera_to_world(trajectories, transform)
    elif transform is not None:
        raise ValueError(
            "arrays.camera_to_world is only valid for coordinate_frame='camera'"
        )

    hypothesis_metadata: list[dict[str, Any]] = []
    for index, rollout_id in enumerate(rollout_ids):
        entry: dict[str, Any] = {
            "hypothesis_id": rollout_id,
            "source_rollout_index": index,
        }
        if parameter_values is not None:
            entry["parameters"] = {
                name: float(parameter_values[index, parameter_index])
                for parameter_index, name in enumerate(parameter_names)
            }
        hypothesis_metadata.append(entry)

    bank = JointRolloutBank(
        hypothesis_ids=rollout_ids,
        hypothesis_metadata=tuple(hypothesis_metadata),
        hypothesis_prior_weights=rollout_weights,
        parameter_particles=np.zeros((1, 0), dtype=np.float64),
        parameter_weights=np.ones(1, dtype=np.float64),
        trajectories=trajectories[:, None].astype(np.float32),
        variance_floor_m2=variance_floor_m2,
        confidence_level=confidence_level,
    )
    metadata = {
        "importer": {
            "schema": EXTERNAL_ROLLOUT_IMPORT_SCHEMA,
            "schema_version": EXTERNAL_ROLLOUT_IMPORT_SCHEMA_VERSION,
            "source_layout": layout,
            "source_coordinate_frame": coordinate_frame,
            "source_position_unit": position_unit,
            "camera_to_world_sha256": camera_to_world_sha256,
        },
        "producer": plain_json(producer_metadata),
    }
    return ExternalRolloutBundle(
        bank=bank,
        case_id=case_id,
        source_simulator=source_simulator,
        source_revision=source_revision,
        source_artifact_id=source_artifact_id,
        node_ids=nodes,
        frame_times_s=frame_times,
        anchor_time_s=anchor_time_s,
        parameter_names=parameter_names,
        source_npz_sha256=source_hash,
        import_manifest_sha256=manifest_hash,
        metadata=metadata,
    )


def save_external_rollout_bank(
    path: str | Path,
    bundle: ExternalRolloutBundle,
    *,
    overwrite: bool = True,
) -> None:
    """Atomically publish a canonical external rollout bank."""

    save_rollout_bank(path, bundle.bank, bundle.descriptor(), overwrite=overwrite)
    restored = load_external_rollout_bank(path)
    if restored.artifact_id != bundle.artifact_id:
        raise ValueError("external rollout bundle changed during serialization")


def load_external_rollout_bank(path: str | Path) -> ExternalRolloutBundle:
    """Load and fully revalidate a canonical external rollout bank."""

    bank, descriptor = load_rollout_bank(path)
    manifest = _require_fields(
        descriptor,
        name="external rollout bank descriptor",
        required=_BANK_DESCRIPTOR_FIELDS,
    )
    if (
        _require_nonempty_string(manifest["schema"], name="schema")
        != EXTERNAL_ROLLOUT_BANK_SCHEMA
    ):
        raise ValueError("rollout bank is not a canonical external rollout bank")
    if (
        _require_integer(manifest["schema_version"], name="schema_version", minimum=1)
        != EXTERNAL_ROLLOUT_BANK_SCHEMA_VERSION
    ):
        raise ValueError("unsupported external rollout bank schema version")
    source = _require_fields(
        manifest["source"],
        name="source",
        required=_BANK_SOURCE_FIELDS,
    )
    bundle = ExternalRolloutBundle(
        bank=bank,
        case_id=manifest["case_id"],
        source_simulator=source["simulator"],
        source_revision=source["revision"],
        source_artifact_id=source["artifact_id"],
        node_ids=np.asarray(manifest["node_ids"]),
        frame_times_s=np.asarray(manifest["frame_times_s"]),
        anchor_time_s=manifest["anchor_time_s"],
        parameter_names=tuple(manifest["parameter_names"]),
        source_npz_sha256=manifest["source_npz_sha256"],
        import_manifest_sha256=manifest["import_manifest_sha256"],
        metadata=manifest["metadata"],
    )
    if bundle.anchor_frame_index != _require_integer(
        manifest["anchor_frame_index"],
        name="anchor_frame_index",
    ):
        raise ValueError("external rollout anchor frame index does not match timebase")
    if bundle.bank.artifact_id != _validate_sha256(
        manifest["bank_artifact_id"],
        name="bank_artifact_id",
    ):
        raise ValueError("external rollout bank artifact ID does not match payload")
    expected_id = _validate_sha256(manifest["artifact_id"], name="artifact_id")
    if bundle.artifact_id != expected_id:
        raise ValueError("external rollout artifact ID does not match descriptor")
    return bundle


__all__ = [
    "EXTERNAL_ROLLOUT_BANK_SCHEMA",
    "EXTERNAL_ROLLOUT_BANK_SCHEMA_VERSION",
    "EXTERNAL_ROLLOUT_IMPORT_SCHEMA",
    "EXTERNAL_ROLLOUT_IMPORT_SCHEMA_VERSION",
    "ExternalRolloutBundle",
    "import_external_rollouts",
    "load_external_rollout_bank",
    "save_external_rollout_bank",
]
