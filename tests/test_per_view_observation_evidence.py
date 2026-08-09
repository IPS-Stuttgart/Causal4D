from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from causal4d.per_view_observation_evidence import (
    build_per_view_observation_evidence,
    load_per_view_observation_evidence,
    validate_per_view_observation_evidence,
    write_per_view_observation_evidence,
)
from causal4d.real_protocol import (
    build_same_object_real_protocol,
    execution_manifest_template,
    validate_execution_manifest,
)


def _write(
    root: Path,
    relative: str,
    payload: bytes | None = None,
) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    content = payload if payload is not None else f"artifact:{relative}\n".encode()
    path.write_bytes(content)
    return {
        "path": relative,
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "media_type": "application/octet-stream",
    }


def _timed(
    root: Path,
    relative: str,
    *,
    sample_count: int = 20,
) -> dict[str, object]:
    return {
        **_write(root, relative),
        "clock_id": "ptp-clock-0",
        "sample_count": sample_count,
    }


def _readdress(value: dict) -> dict:
    payload = deepcopy(value)
    payload.pop("artifact_id", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    value["artifact_id"] = hashlib.sha256(encoded).hexdigest()
    return value


def _fixture(
    root: Path,
    *,
    protocol_id: str = "causal4d-sloth-multi-action-v1",
    protocol_design_sha256: str = "a" * 64,
    execution_id: str = "execution-001",
    session_id: str = "session-001",
    clock_domain_id: str = "ptp-clock-0",
    frame_count: int = 20,
    causal_prefix_frame_stop: int = 8,
) -> dict:
    views = []
    for index in range(2):
        prefix = f"views/camera-{index}"
        views.append(
            {
                "camera_id": f"camera-{index}",
                "calibration_camera_key": f"calibration-camera-{index}",
                "material_point_count": 32,
                "rgb_stream": _timed(
                    root,
                    f"{prefix}/rgb.bin",
                    sample_count=frame_count,
                ),
                "depth_stream": _timed(
                    root,
                    f"{prefix}/depth.bin",
                    sample_count=frame_count,
                ),
                "timestamps": _timed(
                    root,
                    f"{prefix}/timestamps.bin",
                    sample_count=frame_count,
                ),
                "material_points": _timed(
                    root,
                    f"{prefix}/material-points.bin",
                    sample_count=frame_count,
                ),
                "validity_mask": _timed(
                    root,
                    f"{prefix}/validity.bin",
                    sample_count=frame_count,
                ),
                "confidence": _timed(
                    root,
                    f"{prefix}/confidence.bin",
                    sample_count=frame_count,
                ),
                "surface_normals": (
                    _timed(
                        root,
                        f"{prefix}/surface-normals.bin",
                        sample_count=frame_count,
                    )
                    if index == 0
                    else None
                ),
                "surface_normals_unavailable_reason": (
                    None if index == 0 else "not emitted by camera-1 producer"
                ),
            }
        )
    return build_per_view_observation_evidence(
        protocol_id=protocol_id,
        protocol_design_sha256=protocol_design_sha256,
        execution_id=execution_id,
        session_id=session_id,
        clock_domain_id=clock_domain_id,
        frame_count=frame_count,
        common_coordinate_frame="world",
        material_identity_contract="frame-0 material index retained across views",
        observation_producer={
            "name": "multi-view-rgbd-producer",
            "version": "1.0.0",
            "artifact_contract": "causal4d.multi-view-rgbd/v1",
            "software_environment_capsule_id": "b" * 64,
        },
        camera_calibration={
            "revision_id": "camera-calibration-2026-08-09",
            "descriptor": _write(root, "calibration/cameras.json"),
            "camera_keys": [
                "calibration-camera-0",
                "calibration-camera-1",
            ],
        },
        object_frame={
            "frame_id": "sloth-object-frame",
            "definition": _write(root, "registration/object-frame.json"),
            "object_from_world": _timed(
                root,
                "registration/object-from-world.bin",
                sample_count=frame_count,
            ),
        },
        confidence_semantics={
            "continuous": True,
            "higher_is_better": True,
            "minimum": 0.0,
            "maximum": 1.0,
            "missing_value_policy": "validity mask false and confidence zero",
        },
        views=views,
        shared_sensors={
            "commanded_control": _timed(
                root,
                "sensors/commanded-control.bin",
                sample_count=100,
            ),
            "measured_actuation": _timed(
                root,
                "sensors/measured-actuation.bin",
                sample_count=100,
            ),
            "gripper_state": _timed(
                root,
                "sensors/gripper-state.bin",
                sample_count=100,
            ),
            "contact_wrench": None,
            "contact_wrench_unavailable_reason": "sensor not fitted for this run",
            "support_registration": _write(
                root,
                "registration/support.json",
            ),
            "reset_drift_slip": _write(
                root,
                "registration/reset-drift-slip.json",
            ),
        },
        fused_observation={
            "descriptor": _timed(
                root,
                "derived/fused-observation.bin",
                sample_count=frame_count,
            ),
            "source_camera_ids": ["camera-0", "camera-1"],
            "aggregation_method": "calibrated robust multi-view triangulation",
            "material_point_count": 32,
            "derived_from_per_view_evidence": True,
            "sole_retained_observation": False,
        },
        information_boundary={
            "causal_prefix_frame_start": 0,
            "causal_prefix_frame_stop": causal_prefix_frame_stop,
            "raw_full_execution_retained": True,
            "future_frames_retained_for_blind_evaluation": True,
            "future_frames_used_for_inference": False,
            "target_outcomes_used_for_inference": False,
            "target_outcomes_used_for_model_selection": False,
            "target_outcomes_used_for_exclusion": False,
            "target_outcomes_used_for_calibration": False,
            "fused_observation_is_sole_retained_evidence": False,
        },
    )


def test_round_trip_verifies_every_bound_file(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    evidence = _fixture(execution_root)
    path = execution_root / "per-view-observation.json"

    write_per_view_observation_evidence(path, evidence)
    restored = load_per_view_observation_evidence(
        "per-view-observation.json",
        artifact_root=execution_root,
        verify_files=True,
        expected_protocol_id=evidence["protocol_id"],
        expected_protocol_design_sha256=evidence["protocol_design_sha256"],
        expected_execution_id="execution-001",
        expected_session_id="session-001",
        expected_clock_domain_id="ptp-clock-0",
        expected_frame_count=20,
        expected_causal_prefix_frame_stop=8,
    )
    summary = validate_per_view_observation_evidence(
        restored,
        artifact_root=execution_root,
        verify_files=True,
    )

    assert restored == evidence
    assert summary["camera_count"] == 2
    assert summary["material_point_count"] == 32
    assert summary["surface_normal_view_count"] == 1
    assert summary["contact_wrench_retained"] is False
    assert summary["file_hashes_verified"] is True
    assert summary["bound_file_count"] == 22


def test_publication_is_exactly_once_by_default(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    path = tmp_path / "per-view-observation.json"
    write_per_view_observation_evidence(path, evidence)

    with pytest.raises(FileExistsError):
        write_per_view_observation_evidence(path, evidence)
    write_per_view_observation_evidence(path, evidence, overwrite=True)


def test_camera_inventory_and_material_identity_are_closed(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    duplicate = deepcopy(evidence)
    duplicate["views"][1]["camera_id"] = "camera-0"
    _readdress(duplicate)
    with pytest.raises(ValueError, match="camera IDs must be unique"):
        validate_per_view_observation_evidence(duplicate)

    mismatched = deepcopy(evidence)
    mismatched["views"][1]["material_point_count"] = 31
    _readdress(mismatched)
    with pytest.raises(ValueError, match="material point counts"):
        validate_per_view_observation_evidence(mismatched)

    calibration = deepcopy(evidence)
    calibration["camera_calibration"]["camera_keys"].reverse()
    _readdress(calibration)
    with pytest.raises(ValueError, match="camera calibration keys"):
        validate_per_view_observation_evidence(calibration)


def test_fused_observation_cannot_replace_per_view_evidence(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    one_view = deepcopy(evidence)
    one_view["views"] = one_view["views"][:1]
    one_view["camera_calibration"]["camera_keys"] = one_view["camera_calibration"][
        "camera_keys"
    ][:1]
    one_view["fused_observation"]["source_camera_ids"] = ["camera-0"]
    _readdress(one_view)
    with pytest.raises(ValueError, match="at least two camera views"):
        validate_per_view_observation_evidence(one_view)

    sole = deepcopy(evidence)
    sole["fused_observation"]["sole_retained_observation"] = True
    _readdress(sole)
    with pytest.raises(ValueError, match="sole retained"):
        validate_per_view_observation_evidence(sole)


def test_missing_optional_sensor_evidence_requires_an_explicit_reason(
    tmp_path: Path,
) -> None:
    evidence = _fixture(tmp_path)
    missing_reason = deepcopy(evidence)
    missing_reason["shared_sensors"]["contact_wrench_unavailable_reason"] = None
    _readdress(missing_reason)
    with pytest.raises(ValueError, match="unavailable_reason"):
        validate_per_view_observation_evidence(missing_reason)

    contradictory = deepcopy(evidence)
    contradictory["views"][0]["surface_normals_unavailable_reason"] = "missing"
    _readdress(contradictory)
    with pytest.raises(ValueError, match="must be null when retained"):
        validate_per_view_observation_evidence(contradictory)


def test_information_boundary_retains_future_without_using_it(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    for field in (
        "future_frames_used_for_inference",
        "target_outcomes_used_for_inference",
        "target_outcomes_used_for_model_selection",
        "target_outcomes_used_for_exclusion",
        "target_outcomes_used_for_calibration",
    ):
        changed = deepcopy(evidence)
        changed["information_boundary"][field] = True
        _readdress(changed)
        with pytest.raises(ValueError, match=field + ".*false"):
            validate_per_view_observation_evidence(changed)

    no_future = deepcopy(evidence)
    no_future["information_boundary"]["causal_prefix_frame_stop"] = 20
    _readdress(no_future)
    with pytest.raises(ValueError, match="stop before the retained future"):
        validate_per_view_observation_evidence(no_future)


def test_timed_artifacts_share_the_execution_clock_and_view_frame_count(
    tmp_path: Path,
) -> None:
    evidence = _fixture(tmp_path)
    wrong_clock = deepcopy(evidence)
    wrong_clock["views"][0]["confidence"]["clock_id"] = "camera-clock"
    _readdress(wrong_clock)
    with pytest.raises(ValueError, match="clock domain"):
        validate_per_view_observation_evidence(wrong_clock)

    wrong_count = deepcopy(evidence)
    wrong_count["views"][0]["depth_stream"]["sample_count"] = 19
    _readdress(wrong_count)
    with pytest.raises(ValueError, match="sample_count"):
        validate_per_view_observation_evidence(wrong_count)


def test_file_verification_rejects_tampering_and_symlinks(tmp_path: Path) -> None:
    execution_root = tmp_path / "execution"
    evidence = _fixture(execution_root)
    confidence = execution_root / evidence["views"][0]["confidence"]["path"]
    confidence.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch|byte count mismatch"):
        validate_per_view_observation_evidence(
            evidence,
            artifact_root=execution_root,
            verify_files=True,
        )

    other_root = tmp_path / "symlink-execution"
    linked = _fixture(other_root)
    source = other_root / linked["views"][0]["rgb_stream"]["path"]
    target = other_root / "outside.bin"
    target.write_bytes(source.read_bytes())
    source.unlink()
    source.symlink_to(target)
    with pytest.raises(ValueError, match="ordinary readable file|symbolic"):
        validate_per_view_observation_evidence(
            linked,
            artifact_root=other_root,
            verify_files=True,
        )


def test_descriptor_paths_are_unique_and_cannot_escape(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    duplicate = deepcopy(evidence)
    duplicate["views"][1]["rgb_stream"] = deepcopy(duplicate["views"][0]["rgb_stream"])
    _readdress(duplicate)
    with pytest.raises(ValueError, match="reuses artifact path"):
        validate_per_view_observation_evidence(duplicate)

    escaping = deepcopy(evidence)
    escaping["views"][0]["rgb_stream"]["path"] = "../rgb.bin"
    _readdress(escaping)
    with pytest.raises(ValueError, match="safe POSIX relative path"):
        validate_per_view_observation_evidence(escaping)


def test_strict_loading_rejects_duplicate_keys_and_content_id_drift(
    tmp_path: Path,
) -> None:
    evidence = _fixture(tmp_path)
    changed = deepcopy(evidence)
    changed["execution_id"] = "changed"
    with pytest.raises(ValueError, match="artifact ID"):
        validate_per_view_observation_evidence(changed)

    path = tmp_path / "duplicate.json"
    path.write_text('{"schema": 1, "schema": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        load_per_view_observation_evidence(path)

    extra = deepcopy(evidence)
    extra["target_outcome"] = 1.0
    _readdress(extra)
    with pytest.raises(ValueError, match="unexpected=.*target_outcome"):
        validate_per_view_observation_evidence(extra)


def test_expected_execution_bindings_fail_closed(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    with pytest.raises(ValueError, match="execution_id binding changed"):
        validate_per_view_observation_evidence(
            evidence,
            expected_execution_id="execution-002",
        )
    with pytest.raises(ValueError, match="causal-prefix binding changed"):
        validate_per_view_observation_evidence(
            evidence,
            expected_causal_prefix_frame_stop=9,
        )


def test_nonfinite_confidence_semantics_are_rejected(tmp_path: Path) -> None:
    evidence = _fixture(tmp_path)
    changed = deepcopy(evidence)
    changed["confidence_semantics"]["maximum"] = float("nan")
    with pytest.raises(ValueError, match="finite JSON"):
        validate_per_view_observation_evidence(changed)

    serialized = json.dumps(evidence, allow_nan=False)
    assert "NaN" not in serialized


def test_execution_manifest_optionally_binds_per_view_evidence(
    tmp_path: Path,
) -> None:
    protocol = build_same_object_real_protocol()
    execution = next(
        value
        for value in protocol["executions"]
        if value["realization_condition_id"] != "slip_low_force"
    )
    execution_root = tmp_path / execution["execution_id"]
    execution_root.mkdir()
    frame_count = 20
    intervention_frame = 5
    prefix_stop = intervention_frame + 6
    evidence = _fixture(
        execution_root,
        protocol_id=protocol["protocol_id"],
        protocol_design_sha256=protocol["design_sha256"],
        execution_id=execution["execution_id"],
        session_id=execution["session_id"],
        frame_count=frame_count,
        causal_prefix_frame_stop=prefix_stop,
    )
    evidence_path = execution_root / "per-view-observation.json"
    write_per_view_observation_evidence(evidence_path, evidence)

    manifest = execution_manifest_template(protocol, execution["execution_id"])
    assert (
        "per_view_observation_evidence"
        not in protocol["recording_contract"]["required_artifacts"]
    )
    manifest["acquisition_status"] = "complete"
    manifest["acquisition"] = {
        "operator_id": "operator-1",
        "hardware_run_id": "run-1",
        "started_at_utc": "2026-08-09T07:00:00Z",
    }
    manifest["timing"] = {
        "frame_count": frame_count,
        "intervention_frame": intervention_frame,
        "o_plus_prefix_frames": 6,
    }
    for name in protocol["recording_contract"]["required_artifacts"]:
        path = execution_root / "base" / f"{name}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = f"base:{name}\n".encode()
        path.write_bytes(payload)
        descriptor = {
            "path": path.relative_to(execution_root).as_posix(),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        if name in protocol["recording_contract"]["timestamped_artifacts"]:
            descriptor["clock_id"] = "ptp-clock-0"
        manifest["artifacts"][name] = descriptor
    evidence_bytes = evidence_path.read_bytes()
    manifest["artifacts"]["per_view_observation_evidence"] = {
        "path": evidence_path.relative_to(execution_root).as_posix(),
        "sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "bytes": len(evidence_bytes),
    }
    manifest["quality"] = {
        "reset_passed": True,
        "rgbd_actuator_sync_error_ms": 1.0,
        "initial_state_chamfer_m": 0.001,
        "end_effector_reset_error_m": 0.001,
        "contact_centroid_error_m": 0.002,
        "dropped_rgbd_frames": 0,
        "slip_displacement_m": None,
        "complete_release_observed": None,
    }
    manifest["drift_indicators"] = {
        "wear_cycle_count": 1,
        "minutes_since_first_execution": 1.0,
        "object_temperature_c": 22.0,
        "room_temperature_c": 21.0,
        "notes": None,
    }
    manifest["exclusion"] = {
        "status": "included",
        "reason": None,
        "decided_before_target_evaluation": True,
    }

    result = validate_execution_manifest(
        protocol,
        manifest,
        execution_root=execution_root,
        verify_files=True,
    )

    assert result["per_view_observation_evidence_retained"] is True

    malformed = deepcopy(manifest)
    malformed["artifacts"]["per_view_observation_evidence"] = "not-a-descriptor"
    with pytest.raises(ValueError, match="descriptor must be a mapping"):
        validate_execution_manifest(
            protocol,
            malformed,
            execution_root=execution_root,
            verify_files=False,
        )

    partial = deepcopy(manifest)
    partial["artifacts"]["per_view_observation_evidence"] = {
        "path": None,
        "sha256": "0" * 64,
        "bytes": None,
    }
    with pytest.raises(ValueError, match="partially populated"):
        validate_execution_manifest(
            protocol,
            partial,
            execution_root=execution_root,
            verify_files=False,
        )

    extra = deepcopy(manifest)
    extra["artifacts"]["per_view_observation_evidence"]["media_type"] = (
        "application/json"
    )
    with pytest.raises(ValueError, match="descriptor fields changed"):
        validate_execution_manifest(
            protocol,
            extra,
            execution_root=execution_root,
            verify_files=False,
        )

    wrong_manifest_hash = deepcopy(manifest)
    wrong_manifest_hash["artifacts"]["per_view_observation_evidence"]["sha256"] = (
        "0" * 64
    )
    with pytest.raises(ValueError, match="manifest checksum mismatch"):
        validate_execution_manifest(
            protocol,
            wrong_manifest_hash,
            execution_root=execution_root,
            verify_files=False,
        )

    changed = deepcopy(evidence)
    changed["execution_id"] = "wrong-execution"
    _readdress(changed)
    write_per_view_observation_evidence(
        evidence_path,
        changed,
        overwrite=True,
    )
    changed_bytes = evidence_path.read_bytes()
    manifest["artifacts"]["per_view_observation_evidence"].update(
        sha256=hashlib.sha256(changed_bytes).hexdigest(),
        bytes=len(changed_bytes),
    )
    with pytest.raises(ValueError, match="execution_id binding changed"):
        validate_execution_manifest(
            protocol,
            manifest,
            execution_root=execution_root,
            verify_files=True,
        )
