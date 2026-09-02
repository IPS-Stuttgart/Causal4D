#!/usr/bin/env python3
"""Run the source-frozen PokeFlex initial-state matching audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.pokeflex_initial_state_matching import (
    build_initial_state_matching_audit,
    validate_initial_state_matching_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--salt", required=True)
    args = parser.parse_args()

    result = build_initial_state_matching_audit(
        root=args.dataset_root,
        salt=args.salt,
    )
    validate_initial_state_matching_audit(result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["gates"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
