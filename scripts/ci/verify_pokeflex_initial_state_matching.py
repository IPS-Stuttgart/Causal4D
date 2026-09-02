#!/usr/bin/env python3
"""Verify one retained PokeFlex initial-state matching result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.pokeflex_initial_state_matching import (
    validate_initial_state_matching_audit,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text(encoding="utf-8"))
    validate_initial_state_matching_audit(result)
    print(json.dumps(result["decision"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
