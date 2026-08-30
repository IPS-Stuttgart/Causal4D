from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
REMOTE = ROOT / "scripts" / "remote"
if str(REMOTE) not in sys.path:
    sys.path.insert(0, str(REMOTE))

from audit_deform360_gpuserver6000_holdings import build_report  # noqa: E402


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _raw_object(
    root: Path,
    object_id: str,
    *,
    camera_count: int,
    episode_count: int,
    tactile_count: int = 4,
) -> Path:
    object_dir = root / object_id
    for camera_index in range(camera_count):
        stream = object_dir / f"left_cam_{camera_index:02d}"
        for episode_index in range(episode_count):
            _touch(stream / f"{episode_index}.mp4")
            _touch(stream / f"{episode_index}.txt")
    for tactile_index in range(tactile_count):
        stream = object_dir / f"left_tactile_{tactile_index:02d}"
        for episode_index in range(episode_count):
            _touch(stream / f"{episode_index}.npy")
            _touch(stream / f"{episode_index}.txt")
    calibration = object_dir / "calibration_refined"
    for name in ("intrinsics.npy", "extrinsics.npy", "dist.npy"):
        _touch(calibration / name)
    return object_dir


def _processed_episode(
    root: Path,
    object_id: str,
    episode_index: int,
    *,
    camera_count: int = 32,
    tactile_count: int = 0,
) -> None:
    episode = root / "aligned" / object_id / f"episode_{episode_index:04d}"
    for camera_index in range(camera_count):
        camera = episode / f"camera_{camera_index:02d}"
        _touch(camera / "undistorted.mp4")
        _touch(camera / "aligned_timestamps.txt")
    for tactile_index in range(tactile_count):
        _touch(episode / f"tactile_{tactile_index:02d}" / "synced_tactile.npy")


def _config(*roots: Path, expected: list[str]) -> dict[str, object]:
    return {
        "schema_version": 1,
        "protocol_id": "causal4d-deform360-gpuserver6000-holdings-v1",
        "runner_label": "gpuserver6000",
        "dataset_repository": "brownu/deform360",
        "dataset_revision": "7fea8e20231a47641d1d2bc8791920ec4e62ec5e",
        "roots": [
            {"role": f"root_{index}", "path": str(root)}
            for index, root in enumerate(roots)
        ],
        "expected_gpuserver6000_object_ids": expected,
        "known_external_processed_only_object_ids": ["004-rubber-band"],
        "exact_reproduction_object_ids": ["001-rope"],
        "exploratory_preprocessing_object_ids": [
            "003-cable",
            "086-cotton-scarf-cloth",
            "171-penguin",
        ],
        "qualification": {
            "expected_episode_count": 10,
            "exact_raw_camera_count": 41,
            "minimum_ten_episode_camera_count": 36,
            "minimum_single_episode_camera_count": 32,
            "expected_tactile_sensor_count": 4,
            "minimum_processed_camera_count": 32,
        },
    }


def test_exact_and_37_camera_candidates_are_distinguished(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_object(raw, "001-rope", camera_count=41, episode_count=10)
    _raw_object(raw, "003-cable", camera_count=37, episode_count=10)
    report = build_report(_config(raw, expected=["001-rope", "003-cable"]))

    records = {item["object_id"]: item for item in report["raw_records"]}
    assert records["001-rope"]["classification"] == "exact_ten_episode_raw_candidate"
    assert records["003-cable"]["classification"] == (
        "ten_episode_multiview_tactile_candidate"
    )
    assert report["summary"]["exact_001_rope_raw_candidate"] is True
    assert report["interpretation"]["enough_for_multi_object_preprocessing"] is True


def test_single_episode_raw_is_calibration_only(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_object(raw, "097-pillow", camera_count=41, episode_count=1)
    report = build_report(_config(raw, expected=["097-pillow"]))

    record = report["raw_records"][0]
    assert record["classification"] == "single_episode_multiview_tactile_calibration"
    assert record["ten_episode_candidate"] is False
    assert report["interpretation"]["enough_for_uniform_26_object_benchmark"] is False


def test_processed_rgb_and_visuotactile_layouts_are_reported(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    for episode_index in range(10):
        _processed_episode(processed, "004-rubber-band", episode_index)
    _processed_episode(
        processed,
        "026-sock-cloth",
        0,
        tactile_count=4,
    )
    report = build_report(
        _config(processed, expected=["004-rubber-band", "026-sock-cloth"])
    )

    records = {item["object_id"]: item for item in report["processed_records"]}
    assert records["004-rubber-band"]["classification"] == (
        "ten_episode_processed_rgb_candidate"
    )
    assert records["026-sock-cloth"]["classification"] == (
        "processed_visuotactile_calibration"
    )


def test_report_preserves_metadata_only_information_boundary(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    _raw_object(raw, "001-rope", camera_count=41, episode_count=10)
    report = build_report(_config(raw, expected=["001-rope"]))

    assert report["information_boundary"] == {
        "public_data_only": True,
        "new_physical_data_collected": False,
        "metadata_only": True,
        "file_payloads_read": False,
        "media_decoded": False,
        "raw_sources_modified": False,
        "symlinks_followed": False,
        "protected_locked_targets_opened": False,
        "new_paper_claim_authorized": False,
    }
    json.dumps(report, allow_nan=False)
