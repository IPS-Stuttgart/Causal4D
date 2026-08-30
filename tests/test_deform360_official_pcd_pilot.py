from __future__ import annotations

from io import BytesIO
import hashlib
import json
from pathlib import Path
import tarfile
from typing import Any

import numpy as np
import pytest

from causal4d_public.deform360_official_pcd_pilot import (
    PILOT_CONFIG_KIND,
    load_episode_archive,
    load_pilot_config,
    run_official_point_cloud_source_pilot,
    select_reset_positions,
    validate_official_point_cloud_source_pilot,
)


SOURCE_EPISODES = (0, 2, 5, 6, 7, 9)
FORBIDDEN_EPISODES = (1, 3, 4, 8)


def _canonical_config_sha256(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("config_sha256", None)
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_config(path: Path) -> Path:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "artifact_kind": PILOT_CONFIG_KIND,
        "protocol_id": "synthetic-official-pcd-source-pilot",
        "status": "source-only-real-data-pilot",
        "dataset_repository": "brownu/deform360",
        "dataset_revision": "0" * 40,
        "processed_root_suffix": "processed-repository/processed",
        "object_id": "002-rope-silk",
        "source_episode_ids": list(SOURCE_EPISODES),
        "forbidden_episode_ids": list(FORBIDDEN_EPISODES),
        "episode_actions": {
            str(episode): f"action-{episode}" for episode in SOURCE_EPISODES
        },
        "horizon_frames": [1, 3, 6],
        "reset_count": 3,
        "maximum_points": 16,
        "minimum_frames": 20,
        "rho_grid_min": -0.5,
        "rho_grid_max": 1.25,
        "rho_grid_count": 71,
        "prior_variance_floor": 0.0025,
        "likelihood_information_cap": 20,
        "predictive_variance_floor_m2": 1e-10,
        "guard": {
            "minimum_win_fraction": 0.6,
            "minimum_relative_improvement": 0.0,
            "maximum_worst_episode_ratio": 1.25,
        },
        "decision": {
            "primary_horizon_frames": 6,
            "minimum_relative_improvement": 0.05,
            "minimum_episode_win_fraction": 2.0 / 3.0,
            "maximum_worst_episode_ratio": 1.1,
        },
        "information_boundary": {
            "source_only": True,
            "source_future_positions_allowed_for_scoring": True,
            "official_velocity_arrays_used": False,
            "forbidden_episode_payloads_read": False,
            "dataset_modified": False,
            "new_physical_data_collected": False,
            "paper_claim_authorized": False,
        },
    }
    payload["config_sha256"] = _canonical_config_sha256(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _write_episode_archive(
    path: Path,
    *,
    rho: float,
    seed: int,
    frame_count: int = 36,
    point_count: int = 24,
) -> None:
    rng = np.random.default_rng(seed)
    base = rng.normal(scale=0.02, size=(point_count, 3))
    velocity = rng.normal(scale=0.002, size=(point_count, 3))
    positions = [base]
    for _ in range(frame_count - 1):
        velocity = rho * velocity + rng.normal(
            scale=0.00015,
            size=velocity.shape,
        )
        positions.append(positions[-1] + velocity)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, mode="w") as archive:
        for frame_id, points in enumerate(positions):
            stream = BytesIO()
            np.savez_compressed(
                stream,
                pts=points.astype(np.float32),
                vels=np.zeros_like(points, dtype=np.float32),
                visibility_matrix=np.ones((point_count, 2), dtype=bool),
            )
            data = stream.getvalue()
            member = tarfile.TarInfo(f"pcd_clean/{frame_id}.npz")
            member.size = len(data)
            archive.addfile(member, BytesIO(data))


def _write_dataset(root: Path) -> Path:
    object_root = root / "002-rope-silk"
    for index, episode_id in enumerate(SOURCE_EPISODES):
        _write_episode_archive(
            object_root / f"episode_{episode_id}" / "pcd_clean.tar",
            rho=0.65 + 0.05 * index,
            seed=episode_id + 10,
        )
    return root


def test_config_and_rosters_are_content_addressed(tmp_path: Path) -> None:
    config, payload = load_pilot_config(_write_config(tmp_path / "config.json"))
    assert config.source_episode_ids == SOURCE_EPISODES
    assert config.forbidden_episode_ids == FORBIDDEN_EPISODES
    assert payload["config_sha256"] == _canonical_config_sha256(payload)

    payload["source_episode_ids"] = [*SOURCE_EPISODES, 1]
    payload["config_sha256"] = _canonical_config_sha256(payload)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="overlap"):
        load_pilot_config(bad)


def test_archive_loader_uses_positions_only_and_preserves_point_identity(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "pcd_clean.tar"
    _write_episode_archive(archive, rho=0.8, seed=3)
    sequence = load_episode_archive(
        archive,
        episode_id=0,
        action="lift side",
        maximum_points=12,
        minimum_frames=20,
    )
    assert sequence.positions_m.shape == (36, 12, 3)
    assert sequence.frame_ids.tolist() == list(range(36))
    assert sequence.archive_record["used_npz_keys"] == ["pts"]
    assert sequence.archive_record["official_velocity_arrays_used"] is False
    assert "vels" in sequence.archive_record["available_npz_keys"]


def test_reset_selection_depends_only_on_frame_availability() -> None:
    assert select_reset_positions(
        36,
        reset_count=3,
        maximum_horizon=6,
    ) == (3, 16, 29)


def test_synthetic_damped_motion_runs_end_to_end(tmp_path: Path) -> None:
    processed = _write_dataset(tmp_path / "processed")
    config = _write_config(tmp_path / "config.json")
    output = tmp_path / "result.json"
    result = run_official_point_cloud_source_pilot(processed, config, output)
    validate_official_point_cloud_source_pilot(result)

    assert output.is_file()
    assert [row["episode_id"] for row in result["episode_records"]] == list(
        SOURCE_EPISODES
    )
    assert result["dataset"]["forbidden_episode_ids"] == list(FORBIDDEN_EPISODES)
    assert result["information_boundary"] == {
        "source_only": True,
        "source_episode_payloads_read": True,
        "forbidden_episode_payloads_read": False,
        "official_velocity_arrays_used": False,
        "only_causal_prefix_used_for_held_episode_adaptation": True,
        "dataset_modified": False,
        "new_physical_data_collected": False,
        "paper_claim_authorized": False,
    }
    primary = result["horizon_summaries"]["6"]
    assert primary["comparisons"]["guarded"][
        "relative_improvement_vs_persistence"
    ] > 0.0
    assert result["decision"]["paper_claim_authorized"] is False


def test_missing_source_archive_fails_before_any_result(tmp_path: Path) -> None:
    processed = _write_dataset(tmp_path / "processed")
    missing = (
        processed
        / "002-rope-silk"
        / f"episode_{SOURCE_EPISODES[-1]}"
        / "pcd_clean.tar"
    )
    missing.unlink()
    output = tmp_path / "result.json"
    with pytest.raises(ValueError, match="archive is missing"):
        run_official_point_cloud_source_pilot(
            processed,
            _write_config(tmp_path / "config.json"),
            output,
        )
    assert not output.exists()
