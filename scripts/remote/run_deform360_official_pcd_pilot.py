#!/usr/bin/env python3
"""Run the locked source-only pilot on official Deform360 point-cloud archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_official_pcd_pilot import (
    run_official_point_cloud_source_pilot,
    validate_official_point_cloud_source_pilot,
)


def _parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            root
            / "configs"
            / "causal4d_public"
            / "deform360_official_pcd_source_pilot_v1.json"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = run_official_point_cloud_source_pilot(
        args.processed_root,
        args.config,
        args.output,
    )
    validate_official_point_cloud_source_pilot(result)
    primary = result["horizon_summaries"][
        str(result["decision"]["primary_horizon_frames"])
    ]
    print(
        json.dumps(
            {
                "classification": result["decision"]["classification"],
                "decision_passed": result["decision"]["passed"],
                "episode_count": len(result["episode_records"]),
                "guarded_relative_improvement": primary["comparisons"]["guarded"][
                    "relative_improvement_vs_persistence"
                ],
                "guarded_episode_win_fraction": primary["comparisons"]["guarded"][
                    "episode_win_fraction_vs_persistence"
                ],
                "guarded_worst_episode_ratio": primary["comparisons"]["guarded"][
                    "worst_episode_ratio_vs_persistence"
                ],
                "result_path": str(args.output.resolve()),
                "result_sha256": result["result_sha256"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
