"""Validate a canonical external forecast against an external rollout bank."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global build_external_bridge_report
    global load_external_forecast
    global load_external_rollout_bank
    global plain_json

    from causal4d.external_bridge import build_external_bridge_report
    from causal4d.external_forecast import load_external_forecast
    from causal4d.external_rollout import load_external_rollout_bank
    from causal4d.immutable_json import plain_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Check case identity, node correspondence, time overlap, anchor "
            "alignment, motion scale, and exact zero-weight fallback."
        )
    )
    parser.add_argument("external_forecast_npz")
    parser.add_argument("external_rollout_bank_npz")
    parser.add_argument("forecast_id")
    parser.add_argument("output_json")
    parser.add_argument("--anchor-tolerance-m", type=float, default=0.01)
    parser.add_argument("--motion-ratio-min", type=float, default=0.10)
    parser.add_argument("--motion-ratio-max", type=float, default=10.0)
    parser.add_argument("--scale-m", type=float, default=0.05)
    parser.add_argument("--degrees-of-freedom", type=float, default=3.0)
    parser.add_argument(
        "--strict-warnings",
        action="store_true",
        help="return exit status 3 when the contract is valid but warnings remain",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    forecast = load_external_forecast(args.external_forecast_npz)
    rollouts = load_external_rollout_bank(args.external_rollout_bank_npz)
    report = build_external_bridge_report(
        forecast,
        args.forecast_id,
        rollouts,
        anchor_tolerance_m=args.anchor_tolerance_m,
        motion_ratio_min=args.motion_ratio_min,
        motion_ratio_max=args.motion_ratio_max,
        scale_m=args.scale_m,
        degrees_of_freedom=args.degrees_of_freedom,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = plain_json(report)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 3 if args.strict_warnings and payload["warnings"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
