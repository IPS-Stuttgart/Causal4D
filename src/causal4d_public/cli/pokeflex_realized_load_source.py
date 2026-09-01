"""Run the frozen development-only PokeFlex realized-load source gate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np

from causal4d_public.pokeflex_realized_load import (
    load_realized_load_policy,
    run_pokeflex_realized_load_source_gate,
    validate_realized_load_artifact,
)
from causal4d_public.pokeflex_robot_stage import (
    stage_pokeflex_development_robot_records,
    validate_pokeflex_robot_stage,
)


def _official_public_archive_root(dataset_root: str | Path) -> Path:
    """Return the frozen public poking-archive root below the mounted dataset."""

    return Path(dataset_root).resolve() / "poking"


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
        output = Path(args.output_dir).resolve()
        output.mkdir(parents=True, exist_ok=True)
        archive_root = _official_public_archive_root(args.dataset_root)
        with tempfile.TemporaryDirectory(
            prefix="causal4d-pokeflex-robot-stage-"
        ) as temporary:
            stage_root = Path(temporary) / "dataset"
            stage = stage_pokeflex_development_robot_records(
                archive_root,
                source_qa,
                stage_root,
                config,
            )
            stage_validation = validate_pokeflex_robot_stage(stage)
            (output / "input_stage_manifest.json").write_text(
                json.dumps(stage, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            result = run_pokeflex_realized_load_source_gate(
                stage_root,
                source_qa,
                output,
                config,
            )
        validation = validate_realized_load_artifact(result)
        validation["input_stage_manifest_sha256"] = stage_validation[
            "stage_manifest_sha256"
        ]
    except (OSError, KeyError, TypeError, ValueError, np.linalg.LinAlgError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
