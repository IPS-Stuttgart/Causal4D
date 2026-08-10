"""Fit and independently confirm trust for an external bridge forecast."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global fit_external_bridge_trust
    global load_external_bridge_trust_study
    global save_external_bridge_trust_calibration

    from causal4d.external_bridge_trust import (
        fit_external_bridge_trust,
        load_external_bridge_trust_study,
        save_external_bridge_trust_calibration,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select semantic beta on source cases, freeze label-free OOD thresholds, "
            "and admit a positive beta only after an independent confirmation panel."
        )
    )
    parser.add_argument("study_manifest_json")
    parser.add_argument("output_calibration_json")
    parser.add_argument(
        "--beta",
        action="append",
        type=float,
        help="Repeat to define beta candidates; defaults to 0,1,3,6,12.",
    )
    parser.add_argument("--scale-m", type=float, default=0.05)
    parser.add_argument("--degrees-of-freedom", type=float, default=3.0)
    parser.add_argument("--anchor-tolerance-m", type=float, default=0.01)
    parser.add_argument("--doctor-motion-ratio-min", type=float, default=0.10)
    parser.add_argument("--doctor-motion-ratio-max", type=float, default=10.0)
    parser.add_argument(
        "--minimum-selection-relative-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--minimum-confirmation-relative-improvement",
        type=float,
        default=0.0,
    )
    parser.add_argument("--maximum-case-relative-harm", type=float, default=0.05)
    parser.add_argument("--support-margin", type=float, default=1.5)
    parser.add_argument("--require-controls", action="store_true")
    parser.add_argument("--minimum-control-advantage-m", type=float, default=0.0)
    parser.add_argument(
        "--allow-doctor-warnings",
        action="store_true",
        help="do not require clean source and confirmation doctor reports",
    )
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument(
        "--require-admission",
        action="store_true",
        help="return exit status 3 when calibration safely falls back to beta=0",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    study = load_external_bridge_trust_study(args.study_manifest_json)
    beta_values = args.beta if args.beta is not None else [0.0, 1.0, 3.0, 6.0, 12.0]
    calibration = fit_external_bridge_trust(
        study,
        beta_candidates=beta_values,
        scale_m=args.scale_m,
        degrees_of_freedom=args.degrees_of_freedom,
        anchor_tolerance_m=args.anchor_tolerance_m,
        doctor_motion_ratio_min=args.doctor_motion_ratio_min,
        doctor_motion_ratio_max=args.doctor_motion_ratio_max,
        minimum_selection_relative_improvement=(
            args.minimum_selection_relative_improvement
        ),
        minimum_confirmation_relative_improvement=(
            args.minimum_confirmation_relative_improvement
        ),
        maximum_case_relative_harm=args.maximum_case_relative_harm,
        support_margin=args.support_margin,
        controls_required=args.require_controls,
        minimum_control_advantage_m=args.minimum_control_advantage_m,
        require_clean_doctor=not args.allow_doctor_warnings,
    )
    save_external_bridge_trust_calibration(
        args.output_calibration_json,
        calibration,
        overwrite=not args.no_overwrite,
    )
    summary = {
        "admitted_beta": calibration.admitted_beta,
        "calibration_id": calibration.calibration_id,
        "confirmed": calibration.confirmed,
        "output": str(Path(args.output_calibration_json).resolve()),
        "reasons": list(calibration.reasons),
        "selected_beta": calibration.selected_beta,
        "selection_relative_improvement": calibration.selection[
            "relative_improvement"
        ],
        "confirmation_relative_improvement": calibration.confirmation[
            "relative_improvement"
        ],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 3 if args.require_admission and not calibration.confirmed else 0


if __name__ == "__main__":
    raise SystemExit(main())
