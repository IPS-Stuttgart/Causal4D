"""Run the frozen development-only PokeFlex realized-load source gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from causal4d_public.pokeflex_realized_load import (
    load_realized_load_policy,
    run_pokeflex_realized_load_source_gate,
    validate_realized_load_artifact,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root")
    parser.add_argument("source_qa_json")
    parser.add_argument("output_dir")
    parser.add_argument("--policy", required=True)
    args = parser.parse_args()
    try:
        source_qa = json.loads(Path(args.source_qa_json).read_text(encoding="utf-8"))
        config = load_realized_load_policy(args.policy)
        result = run_pokeflex_realized_load_source_gate(
            args.dataset_root,
            source_qa,
            args.output_dir,
            config,
        )
        validation = validate_realized_load_artifact(result)
    except (OSError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
