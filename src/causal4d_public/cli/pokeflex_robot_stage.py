"""Stage verified PokeFlex development robot records read-only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d_public.pokeflex_realized_load import load_realized_load_policy
from causal4d_public.pokeflex_robot_stage import (
    stage_pokeflex_development_robot_records,
    validate_pokeflex_robot_stage,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("source_qa_json")
    parser.add_argument("destination_root")
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    try:
        source_qa = json.loads(
            Path(args.source_qa_json).read_text(encoding="utf-8")
        )
        config = load_realized_load_policy(args.policy)
        result = stage_pokeflex_development_robot_records(
            args.dataset_root,
            source_qa,
            args.destination_root,
            config,
        )
        validation = validate_pokeflex_robot_stage(result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
