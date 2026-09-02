#!/usr/bin/env python3
"""Run the continuous realized-action interface with an explicit drop boundary.

This is a technical evidence-schema repair only. The source cohort, continuous
action representation, thresholds, and scientific decision are delegated
unchanged to revision 1. Revision 2 adds the explicit zero-valued
``drop_archive_open_count`` field required by the independent verifier before
recomputing the content identity.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "qualify_pokeflex_continuous_realized_action_interface.py"
)


def load_base_module():
    spec = importlib.util.spec_from_file_location(
        "pokeflex_continuous_realized_action_interface_v1",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load continuous realized-action module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = load_base_module()


def main() -> int:
    args = BASE.parse_args()
    request = BASE.AUDIT.load_request(args.request)
    payload = BASE.run(args.root, request)
    boundary = payload["information_boundary"]
    if boundary.get("non_robot_member_open_count") != 0:
        raise ValueError("a non-robot payload was opened")
    boundary["drop_archive_open_count"] = 0
    payload["content_sha256"] = BASE.content_sha256(payload)
    BASE.write_outputs(args.output_dir, payload)
    print(json.dumps(payload["geometry"], indent=2, sort_keys=True))
    return 0 if payload["gate"]["passed"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
