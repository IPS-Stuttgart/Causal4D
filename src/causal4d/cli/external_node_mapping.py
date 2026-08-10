"""Map visual query anchors to persistent simulator node identities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def _load_runtime_dependencies() -> None:
    global _load_array
    global _text_ids
    global build_external_node_mapping
    global save_external_node_mapping

    from causal4d.external_node_mapping import (
        _load_array,
        _text_ids,
        build_external_node_mapping,
        save_external_node_mapping,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an audited one-to-one minimum-distance assignment from query "
            "anchors to simulator nodes. Geometric proximity is not material proof."
        )
    )
    parser.add_argument("query_npz")
    parser.add_argument("simulator_nodes_npz")
    parser.add_argument("output_json")
    parser.add_argument("--output-npz")
    parser.add_argument("--output-svg")
    parser.add_argument("--query-position-key", default="anchor_positions_world_m")
    parser.add_argument("--query-id-key")
    parser.add_argument("--node-position-key", default="node_positions_world_m")
    parser.add_argument("--node-id-key", default="node_ids")
    parser.add_argument("--maximum-distance-m", type=float, default=0.005)
    parser.add_argument("--projection", choices=("xy", "xz", "yz"), default="xy")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _load_runtime_dependencies()
    query_positions, query_hash = _load_array(
        args.query_npz,
        args.query_position_key,
        name="query",
    )
    node_positions, node_hash = _load_array(
        args.simulator_nodes_npz,
        args.node_position_key,
        name="simulator nodes",
    )
    node_ids, second_node_hash = _load_array(
        args.simulator_nodes_npz,
        args.node_id_key,
        name="simulator nodes",
    )
    if node_hash != second_node_hash:
        raise RuntimeError("simulator node NPZ changed while reading arrays")
    query_ids = None
    if args.query_id_key:
        raw_query_ids, second_query_hash = _load_array(
            args.query_npz,
            args.query_id_key,
            name="query",
        )
        if query_hash != second_query_hash:
            raise RuntimeError("query NPZ changed while reading arrays")
        query_ids = _text_ids(raw_query_ids, name="query_ids")
    report = build_external_node_mapping(
        query_positions,
        node_positions,
        node_ids,
        query_ids=query_ids,
        maximum_distance_m=args.maximum_distance_m,
        query_source_sha256=query_hash,
        node_source_sha256=node_hash,
    )
    save_external_node_mapping(
        report,
        args.output_json,
        output_npz=args.output_npz,
        output_svg=args.output_svg,
        projection=args.projection,
        overwrite=args.overwrite,
    )
    summary = {
        "accepted": report["accepted"],
        "mapping_id": report["mapping_id"],
        "maximum_assigned_distance_m": report["maximum_assigned_distance_m"],
        "output_json": str(Path(args.output_json).resolve()),
        "query_count": report["query_count"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if report["accepted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
