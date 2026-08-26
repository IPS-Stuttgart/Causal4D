"""Evaluate the preregistered fresh-reset graph-mode-zero scale cross-check."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from causal4d.acquisition_flight_common import _assert_no_symlink_components
from causal4d.preacquisition_readiness_contracts import (
    load_registered_preacquisition_chain,
)
from causal4d.reset_mode0_crosscheck import (
    RESET_MODE0_ARTIFACT_PATH,
    RESET_MODE0_INPUT_PATH,
    evaluate_reset_mode0_npz,
    load_reset_registration_binding,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository_root")
    parser.add_argument("dataset_root")
    parser.add_argument("input_npz")
    parser.add_argument("output_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        dataset_root = Path(args.dataset_root).resolve()
        _assert_no_symlink_components(Path(args.dataset_root), name="dataset root")
        _assert_no_symlink_components(Path(args.input_npz), name="reset pilot NPZ")
        _assert_no_symlink_components(Path(args.output_json), name="mode-0 output")
        if not dataset_root.is_dir():
            raise ValueError("dataset root is invalid")
        input_path = Path(args.input_npz).resolve()
        output_path = Path(args.output_json).resolve()
        if input_path != (dataset_root / RESET_MODE0_INPUT_PATH).resolve():
            raise ValueError("input NPZ differs from the registered reset-pilot path")
        if output_path != (dataset_root / RESET_MODE0_ARTIFACT_PATH).resolve():
            raise ValueError("output JSON differs from the registered mode-0 path")
        protocol, _, _, preacquisition_v5 = load_registered_preacquisition_chain(
            args.repository_root
        )
        registration_binding = load_reset_registration_binding(
            protocol,
            preacquisition_v5,
            args.dataset_root,
        )
        result = evaluate_reset_mode0_npz(
            protocol,
            preacquisition_v5,
            args.input_npz,
            args.output_json,
            registration_binding=registration_binding,
            source_path_label=RESET_MODE0_INPUT_PATH,
        )
    except (FileExistsError, OSError, KeyError, TypeError, ValueError) as error:
        print(json.dumps({"passed": False, "error": str(error)}, indent=2))
        return 2
    print(json.dumps({"passed": True, **result}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
