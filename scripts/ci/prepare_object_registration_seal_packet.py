#!/usr/bin/env python3
"""Validate approved anatomy and prepare a non-mutating registration packet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from causal4d.atomic_io import atomic_write_json
from causal4d.object_registration_anatomy import (
    build_object_registration_seal_packet,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--object-instance-serial")
    parser.add_argument("--phystwin-model-id", required=True)
    parser.add_argument("--phystwin-model-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    arguments = _parser().parse_args()
    protocol = json.loads(arguments.protocol.read_text(encoding="utf-8"))
    packet = build_object_registration_seal_packet(
        protocol,
        arguments.evidence_root,
        object_instance_serial=arguments.object_instance_serial,
        phystwin_model_id=arguments.phystwin_model_id,
        phystwin_model_sha256=arguments.phystwin_model_sha256,
    )
    atomic_write_json(arguments.output, packet, overwrite=True)
    print(json.dumps(packet, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
