"""Run and publish one external forecast/rollout bridge analysis."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global analyze_external_bridge
    global load_external_forecast
    global load_external_reference
    global load_external_rollout_bank
    global publish_external_bridge_run

    from causal4d.external_bridge_run import (
        analyze_external_bridge,
        publish_external_bridge_run,
    )
    from causal4d.external_forecast import load_external_forecast
    from causal4d.external_reference import load_external_reference
    from causal4d.external_rollout import load_external_rollout_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a beta sweep over a canonical external forecast and finite "
            "physical rollout bank, then publish doctor, weight, prediction, "
            "and optional reference-evaluation artifacts."
        )
    )
    parser.add_argument("external_forecast_npz")
    parser.add_argument("external_rollout_bank_npz")
    parser.add_argument("forecast_id")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--reference",
        help=(
            "Optional closed NPZ with case_id, node_ids, positions_world_m, "
            "frame_times_s, and optional validity_mask."
        ),
    )
    parser.add_argument(
        "--beta",
        action="append",
        type=float,
        help="Repeat to define the beta grid; defaults to 0,1,3,6,12.",
    )
    parser.add_argument("--scale-m", type=float, default=0.05)
    parser.add_argument("--degrees-of-freedom", type=float, default=3.0)
    parser.add_argument("--anchor-tolerance-m", type=float, default=0.01)
    parser.add_argument("--motion-ratio-min", type=float, default=0.10)
    parser.add_argument("--motion-ratio-max", type=float, default=10.0)
    parser.add_argument(
        "--require-clean-doctor",
        action="store_true",
        help="Publish the complete report but return exit status 3 on warnings.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    forecast = load_external_forecast(args.external_forecast_npz)
    rollouts = load_external_rollout_bank(args.external_rollout_bank_npz)
    reference = load_external_reference(args.reference) if args.reference else None
    beta_values = args.beta if args.beta is not None else [0.0, 1.0, 3.0, 6.0, 12.0]
    report, arrays = analyze_external_bridge(
        forecast,
        args.forecast_id,
        rollouts,
        beta_values=beta_values,
        scale_m=args.scale_m,
        degrees_of_freedom=args.degrees_of_freedom,
        anchor_tolerance_m=args.anchor_tolerance_m,
        motion_ratio_min=args.motion_ratio_min,
        motion_ratio_max=args.motion_ratio_max,
        reference=reference,
    )
    manifest = publish_external_bridge_run(
        args.output_dir,
        report,
        arrays,
        overwrite=args.overwrite,
    )
    summary = {
        "case_id": report["case_id"],
        "doctor_warning_count": len(report["doctor"]["warnings"]),
        "evaluation_only_best_beta": report["evaluation_only_best_beta"],
        "exact_beta_zero_fallback": report["doctor"][
            "beta_zero_weights_bit_identical"
        ],
        "manifest_id": manifest["manifest_id"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "reference_evaluated": reference is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_clean_doctor and report["doctor"]["warnings"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
