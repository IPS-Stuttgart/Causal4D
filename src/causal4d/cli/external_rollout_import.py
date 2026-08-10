"""Import a portable external physical rollout bank."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global import_external_rollouts
    global save_external_rollout_bank

    from causal4d.external_rollout import (
        import_external_rollouts,
        save_external_rollout_bank,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize a flat external simulator rollout support into the "
            "content-addressed Causal4D JointRolloutBank contract."
        )
    )
    parser.add_argument("source_npz")
    parser.add_argument("import_manifest_json")
    parser.add_argument("output_rollout_bank_npz")
    parser.add_argument("--no-overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    bundle = import_external_rollouts(args.source_npz, args.import_manifest_json)
    save_external_rollout_bank(
        args.output_rollout_bank_npz,
        bundle,
        overwrite=not args.no_overwrite,
    )
    summary = bundle.summary()
    summary["output"] = str(Path(args.output_rollout_bank_npz).resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
