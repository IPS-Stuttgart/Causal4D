from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

from causal4d_public.deform360_sensor_reveal_source import (
    _payload_sha256,
    run_source_audit,
    validate_source_audit,
)


def _write_config(path: Path, source_ids: list[int]) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "artifact_kind": "Deform360SensorRevealSourceAuditConfig",
        "protocol_id": "test-deform360-sensor-reveal-source-audit-v1",
        "dataset_repository": "brownu/deform360",
        "dataset_revision": "7fea8e20231a47641d1d2bc8791920ec4e62ec5e",
        "object_id": "002-rope-silk",
        "source_episode_ids": source_ids,
        "forbidden_episode_ids": [1, 3],
        "minimum_frames": 24,
        "reset_count": 5,
        "prefix_window_frames": 6,
        "future_horizon_frames": 6,
        "minimum_common_tactile_groups": 2,
        "sensor_costs": {
            "robot_opening": 0.05,
            "robot_translation": 0.10,
            "tactile": 0.20,
        },
        "information_boundary": {
            "source_only": True,
            "source_episode_payloads_allowed": True,
            "forbidden_episode_payloads_read": False,
            "held_target_payloads_read": False,
            "raw_camera_video_decoded": False,
            "dataset_modified": False,
            "new_physical_data_collected": False,
            "paper_claim_authorized": False,
        },
    }
    payload["config_sha256"] = _payload_sha256(payload, digest_field="config_sha256")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _write_pcd_tar(path: Path, frame_count: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w:") as archive:
        for frame in range(frame_count):
            raw = f"payload-{frame}".encode()
            info = tarfile.TarInfo(name=f"frame_{frame:04d}.npz")
            info.size = len(raw)
            archive.addfile(info, io.BytesIO(raw))


def _write_episode(root: Path, episode_id: int, *, omit_right: bool = False) -> None:
    frame_count = 40
    episode = root / "002-rope-silk" / f"episode_{episode_id:04d}"
    _write_pcd_tar(episode / "pcd_clean.tar", frame_count)

    phase = 0.2 * episode_id
    time = np.arange(frame_count, dtype=np.float64)
    speed = 0.001 + 0.0005 * (1.0 + np.sin(time / 5.0 + phase))
    x = np.cumsum(speed)
    transforms = np.broadcast_to(np.eye(4), (frame_count, 2, 4, 4)).copy()
    transforms[:, 0, 0, 3] = x
    transforms[:, 1, 0, 3] = 0.8 * x
    openings = np.stack(
        [0.03 + 0.01 * np.sin(time / 7.0 + phase), 0.04 + 0.008 * np.cos(time / 8.0)],
        axis=1,
    )
    robot_dir = episode / "robot"
    robot_dir.mkdir(parents=True)
    np.savez(robot_dir / "robot.npz", openings=openings, T_worlds=transforms)

    left_energy = speed + 0.0001 * episode_id
    right_energy = 0.5 + 0.05 * np.cos(time / 3.0 + phase)
    for name, values in (
        ("left_tactile", left_energy),
        ("right_tactile", right_energy),
    ):
        if omit_right and name == "right_tactile":
            continue
        sensor_dir = episode / name
        sensor_dir.mkdir()
        array = np.broadcast_to(values[:, None, None], (frame_count, 3, 4)).copy()
        np.save(sensor_dir / "synced_tactile.npy", array)


def test_source_audit_builds_grouped_routing_evidence(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    source_ids = [0, 2, 4]
    for episode_id in source_ids:
        _write_episode(processed, episode_id)
    config = tmp_path / "config.json"
    _write_config(config, source_ids)
    output = tmp_path / "result.json"

    first = run_source_audit(processed, processed, config, output)
    validate_source_audit(first)
    first_bytes = output.read_bytes()
    second = run_source_audit(processed, processed, config, output)
    validate_source_audit(second)

    assert output.read_bytes() == first_bytes
    assert first["capability_gates"] == second["capability_gates"]
    assert first["capability_gates"]["ready_for_source_only_sensor_reveal_pilot"]
    assert first["feature_case_count"] == 15
    assert first["capability_gates"]["common_tactile_group_count"] == 2
    assert all(
        record["point_cloud"]["member_payloads_read"] is False
        for record in first["episode_records"]
    )
    assert first["information_boundary"]["forbidden_episode_paths_opened"] is False
    routing = first["routing_diagnostic"]
    assert routing["single_sensor_diagnostics"]
    assert routing["best_fixed_pairs"]
    assert all(
        row["leave_one_episode_out"]["episode_count"] == 3
        for row in routing["single_sensor_diagnostics"]
    )


def test_missing_common_tactile_group_fails_closed(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    source_ids = [0, 2, 4]
    for episode_id in source_ids:
        _write_episode(processed, episode_id, omit_right=episode_id == 4)
    config = tmp_path / "config.json"
    _write_config(config, source_ids)
    result = run_source_audit(processed, processed, config, tmp_path / "result.json")
    validate_source_audit(result)

    assert result["capability_gates"]["common_tactile_group_count"] == 1
    assert result["capability_gates"]["tactile_prefix_carriers_complete"] is False
    assert (
        result["capability_gates"]["ready_for_source_only_sensor_reveal_pilot"] is False
    )


def test_config_checksum_and_source_forbidden_overlap_are_rejected(
    tmp_path: Path,
) -> None:
    processed = tmp_path / "processed"
    for episode_id in [0, 2, 4]:
        _write_episode(processed, episode_id)
    config = tmp_path / "config.json"
    payload = _write_config(config, [0, 2, 4])
    payload["reset_count"] = 4
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        run_source_audit(processed, processed, config, tmp_path / "result.json")

    payload = _write_config(config, [0, 2, 4])
    payload["forbidden_episode_ids"] = [1, 2]
    payload["config_sha256"] = _payload_sha256(payload, digest_field="config_sha256")
    config.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlap"):
        run_source_audit(processed, processed, config, tmp_path / "result.json")
