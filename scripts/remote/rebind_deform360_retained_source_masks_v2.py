#!/usr/bin/env python3
"""Rebind retained Deform360 source masks without rerunning SAM2.

This source-only helper byte-hashes the existing ``mask_refined.h5`` files,
recomputes first-frame multiview consistency, and records propagation emptiness.
It does not open target episodes or invoke SAM2 inference.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from causal4d_public.deform360 import (
    load_deform360_protocol_config,
    validate_deform360_preflight,
)
from causal4d_public.deform360_sam2 import (
    DEFORM360_SAM2_MASK_SCHEMA_VERSION,
    PINNED_SAM2_CHECKPOINT_SHA256,
    PINNED_SAM2_CHECKPOINT_URL,
    PINNED_SAM2_COMMIT,
    PINNED_SAM2_MODEL_CONFIG,
    PINNED_SAM2_REPOSITORY,
    RopeSam2MaskConfig,
    sam2_mask_artifact_sha256,
    validate_sam2_episode_access,
    write_sam2_mask_audit,
)
from causal4d_public.deform360_sam2_views import (
    CrossViewMaskReliabilityConfig,
    build_sam2_view_audit,
    multiview_mask_consistency,
    sam2_view_audit_sha256,
    write_sam2_view_audit,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("processed_root", type=Path)
    parser.add_argument("episode_index", type=int)
    parser.add_argument("preflight_json", type=Path)
    parser.add_argument("view_output_json", type=Path)
    parser.add_argument("mask_output_json", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        config = load_deform360_protocol_config(args.config)
        preflight = json.loads(args.preflight_json.read_text(encoding="utf-8"))
        validate_deform360_preflight(preflight)
        access = validate_sam2_episode_access(
            args.episode_index,
            config,
            held_out_prediction_seal_sha256=None,
        )
        _require(access["split"] == "source", "retained-mask rebind is source-only")
        _require(
            preflight["protocol_id"] == config.protocol_id,
            "preflight and protocol differ",
        )

        try:
            from deform360.annotations import H5Array
            from deform360.layout import resolve_episode_dir
            from deform360.processing.episode import (
                episode_cameras,
                load_episode_calibration,
            )
        except ImportError as error:
            raise RuntimeError("the pinned Deform360 source checkout is required") from error

        episode_dir = resolve_episode_dir(args.processed_root, args.episode_index)
        candidate_cameras = sorted(
            camera
            for camera in episode_cameras(episode_dir)
            if (episode_dir / camera / "mask_refined.h5").is_file()
            and (episode_dir / camera / "undistorted.mp4").is_file()
        )
        _require(len(candidate_cameras) >= 8, "too few retained source mask cameras")

        intrinsics, extrinsics = load_episode_calibration(episode_dir)
        first_masks: dict[str, np.ndarray] = {}
        mask_records: dict[str, dict[str, Any]] = {}
        camera_diagnostics = []
        for camera in candidate_cameras:
            path = episode_dir / camera / "mask_refined.h5"
            with H5Array(path) as stored:
                frame_count = len(stored)
                _require(frame_count >= 1, f"retained mask is empty: {camera}")
                first = np.asarray(stored[0], dtype=bool)
                areas = []
                for frame_index in range(frame_count):
                    mask = np.asarray(stored[frame_index], dtype=bool)
                    _require(mask.shape == first.shape, f"mask shape changed: {camera}")
                    areas.append(int(np.count_nonzero(mask)))
            first_masks[camera] = first
            mask_records[camera] = {
                "path": path.name,
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }
            camera_diagnostics.append(
                {
                    "camera": camera,
                    "initialization": "retained-published-source-mask",
                    "retained_mask_sha256": mask_records[camera]["sha256"],
                    "propagation": {
                        "frame_count": frame_count,
                        "empty_frame_count": int(sum(area == 0 for area in areas)),
                        "area_pixels": {
                            "minimum": int(min(areas)),
                            "median": float(np.median(areas)),
                            "maximum": int(max(areas)),
                        },
                    },
                }
            )

        reliability_config = CrossViewMaskReliabilityConfig()
        consistency = multiview_mask_consistency(
            first_masks,
            intrinsics,
            extrinsics,
            reliability_config,
        )
        view_payload = build_sam2_view_audit(
            protocol_id=config.protocol_id,
            episode_access=access,
            automatic_view_diagnostics=[
                {
                    "camera": record["camera"],
                    "automatic_selected": True,
                    "initialization": "retained-published-source-mask",
                    "retained_mask_sha256": record["retained_mask_sha256"],
                }
                for record in camera_diagnostics
            ],
            consistency=consistency,
            reliability_config=reliability_config,
        )
        view_payload["claim_boundary"] = (
            "No new SAM2 inference was executed. First-frame multiview consistency "
            "was recomputed from byte-hashed retained source mask_refined.h5 files."
        )
        view_payload["result_sha256"] = sam2_view_audit_sha256(view_payload)
        write_sam2_view_audit(args.view_output_json, view_payload)

        mask_payload: dict[str, Any] = {
            "schema_version": DEFORM360_SAM2_MASK_SCHEMA_VERSION,
            "artifact_kind": "Deform360RopeSam2MaskAudit",
            "protocol_id": config.protocol_id,
            "episode_access": access,
            "upstream": {
                "repository": PINNED_SAM2_REPOSITORY,
                "commit": PINNED_SAM2_COMMIT,
                "checkpoint_url": PINNED_SAM2_CHECKPOINT_URL,
                "checkpoint_sha256": PINNED_SAM2_CHECKPOINT_SHA256,
                "model_config": PINNED_SAM2_MODEL_CONFIG,
                "execution": "not-rerun-retained-source-mask-rebind",
            },
            "parameters": asdict(RopeSam2MaskConfig()),
            "model_id": "retained-published-001-rope-source-mask",
            "view_selection": {
                "view_audit_result_sha256": view_payload["result_sha256"],
                "cross_view_gate_applied": True,
            },
            "camera_diagnostics": camera_diagnostics,
            "outputs": mask_records,
            "claim_boundary": (
                "This artifact binds existing source-only mask_refined.h5 bytes and "
                "freshly recomputed QA; SAM2 was not rerun in this workflow."
            ),
        }
        mask_payload["result_sha256"] = sam2_mask_artifact_sha256(mask_payload)
        write_sam2_mask_audit(args.mask_output_json, mask_payload)
    except (OSError, KeyError, RuntimeError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2

    print(
        json.dumps(
            {
                "passed": True,
                "episode_index": args.episode_index,
                "candidate_camera_count": len(candidate_cameras),
                "accepted_camera_count": consistency["accepted_camera_count"],
                "view_result_sha256": view_payload["result_sha256"],
                "mask_result_sha256": mask_payload["result_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
