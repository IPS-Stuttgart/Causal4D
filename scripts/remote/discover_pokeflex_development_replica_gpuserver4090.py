#!/usr/bin/env python3
"""Discover a verified PokeFlex development replica on gpuserver4090."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from causal4d_public.pokeflex_replica_discovery import (  # noqa: E402
    discover_pokeflex_development_replica,
    validate_pokeflex_replica_discovery,
)


DEFAULT_SEARCH_ROOTS = (
    Path("/home/github-runner/.cache"),
    Path("/home/github-runner/actions-runner/_work"),
    Path("/home/github-runner/_work"),
    Path("/mnt/seagate10tb/florianpfaff"),
    Path("/mnt/lexar4tb"),
    Path("/home/florianpfaff"),
    Path("/tmp"),
)
DEFAULT_CACHE_ROOT = Path(
    "/home/github-runner/.cache/datasets/pokeflex-causal4d-realized-load-v1"
)
DEFAULT_SOURCE_QA = ROOT / (
    "milestones/pokeflex-001-source-warp-gate-v1/artifacts/"
    "3dPrintedBunny_source_qa_v1.json"
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-qa", type=Path, default=DEFAULT_SOURCE_QA)
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        dest="search_roots",
        help="Override the bounded default search roots; may be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        source_qa = json.loads(args.source_qa.read_text(encoding="utf-8"))
        result = discover_pokeflex_development_replica(
            source_qa=source_qa,
            search_roots=tuple(args.search_roots or DEFAULT_SEARCH_ROOTS),
            cache_root=args.cache_root,
        )
        validation = validate_pokeflex_replica_discovery(result)
        _write_json(args.output, result)
    except (OSError, KeyError, TypeError, ValueError) as error:
        failure = {
            "passed": False,
            "technical_status": "replica-discovery-failed-closed",
            "error": f"{type(error).__name__}: {error}",
            "calibration_take_data_read": False,
            "target_take_data_read": False,
        }
        _write_json(args.output, failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 2
    print(json.dumps(validation, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
