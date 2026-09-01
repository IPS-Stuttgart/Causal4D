#!/usr/bin/env python3
"""Run the frozen PokeFlex source gate from an exact verified cache."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from causal4d_public.pokeflex_cached_source_gate import (  # noqa: E402
    run_cached_pokeflex_source_gate,
    validate_cached_source_gate_decision,
)
from causal4d_public.pokeflex_realized_load import (  # noqa: E402
    load_realized_load_policy,
)


DEFAULT_DISCOVERY = ROOT / (
    "public-realworld-probe/pokeflex-development-replica.json"
)
DEFAULT_SOURCE_QA = ROOT / (
    "milestones/pokeflex-001-source-warp-gate-v1/artifacts/"
    "3dPrintedBunny_source_qa_v1.json"
)
DEFAULT_POLICY = ROOT / (
    "configs/causal4d_public/pokeflex_realized_load_source_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--discovery", type=Path, default=DEFAULT_DISCOVERY)
    parser.add_argument("--source-qa", type=Path, default=DEFAULT_SOURCE_QA)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        discovery = json.loads(args.discovery.read_text(encoding="utf-8"))
        source_qa = json.loads(args.source_qa.read_text(encoding="utf-8"))
        config = load_realized_load_policy(args.policy)
        decision = run_cached_pokeflex_source_gate(
            discovery=discovery,
            source_qa=source_qa,
            output_dir=args.output_dir,
            config=config,
        )
        validation = validate_cached_source_gate_decision(decision)
    except (OSError, KeyError, TypeError, ValueError) as error:
        failure = {
            "passed": False,
            "technical_status": "cached-source-gate-failed-closed",
            "error": f"{type(error).__name__}: {error}",
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        }
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "cached_source_gate_failure.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
