#!/usr/bin/env python3
"""Run the source-only Deform360 logged cross-intervention abduction study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.deform360_logged_counterfactual import (
    LoggedCounterfactualConfig,
    build_logged_counterfactual_source_artifact,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("observation_json", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--protocol-id",
        default="causal4d-deform360-logged-counterfactual-source-v1",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.observation_json
    ]
    result = build_logged_counterfactual_source_artifact(
        payloads,
        protocol_id=args.protocol_id,
        config=LoggedCounterfactualConfig(),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["source_gate"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
