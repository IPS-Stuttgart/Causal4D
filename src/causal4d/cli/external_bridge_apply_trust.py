"""Apply a frozen external-bridge trust calibration to one target case."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global apply_external_bridge_trust
    global load_external_bridge_trust_calibration
    global load_external_forecast
    global load_external_reference
    global load_external_rollout_bank
    global publish_external_bridge_run

    from causal4d.external_bridge_run import publish_external_bridge_run
    from causal4d.external_bridge_trust import (
        apply_external_bridge_trust,
        load_external_bridge_trust_calibration,
    )
    from causal4d.external_forecast import load_external_forecast
    from causal4d.external_reference import load_external_reference
    from causal4d.external_rollout import load_external_rollout_bank


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a source-selected, independently confirmed beta using only "
            "label-free target OOD diagnostics; rejection preserves beta=0 exactly."
        )
    )
    parser.add_argument("external_forecast_npz")
    parser.add_argument("external_rollout_bank_npz")
    parser.add_argument("forecast_id")
    parser.add_argument("trust_calibration_json")
    parser.add_argument("output_dir")
    parser.add_argument(
        "--reference",
        help="optional evaluation-only target reference; never affects admission",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--require-acceptance",
        action="store_true",
        help="return exit status 3 after publication when the gate falls back",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    forecast = load_external_forecast(args.external_forecast_npz)
    rollouts = load_external_rollout_bank(args.external_rollout_bank_npz)
    calibration = load_external_bridge_trust_calibration(args.trust_calibration_json)
    reference = load_external_reference(args.reference) if args.reference else None
    report, arrays, decision = apply_external_bridge_trust(
        forecast,
        args.forecast_id,
        rollouts,
        calibration,
        reference=reference,
    )
    manifest = publish_external_bridge_run(
        args.output_dir,
        report,
        arrays,
        overwrite=args.overwrite,
    )
    summary = {
        "accepted": decision.accepted,
        "admitted_beta": decision.admitted_beta,
        "applied_beta": decision.applied_beta,
        "calibration_id": decision.calibration_id,
        "decision_id": decision.decision_id,
        "evaluation_only_best_beta": report["evaluation_only_best_beta"],
        "manifest_id": manifest["manifest_id"],
        "output_dir": str(Path(args.output_dir).resolve()),
        "reasons": list(decision.reasons),
        "reference_evaluated": reference is not None,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 3 if args.require_acceptance and not decision.accepted else 0


if __name__ == "__main__":
    raise SystemExit(main())
