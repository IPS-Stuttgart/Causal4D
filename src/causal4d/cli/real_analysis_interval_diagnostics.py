"""CLI for source-verified registered real-analysis intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from causal4d.real_analysis_interval_diagnostics import (
    build_real_analysis_interval_diagnostics,
    write_real_analysis_interval_diagnostics,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the registered bootstrap-t primary interval, Student-t "
            "robustness gate, and historical percentile sensitivity."
        )
    )
    parser.add_argument("effect_table", type=Path)
    parser.add_argument("protocol", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--method-freeze", type=Path, required=True)
    parser.add_argument("--analysis-manifest", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build and publish the companion interval artifact."""

    arguments = _parser().parse_args(argv)
    payload = build_real_analysis_interval_diagnostics(
        arguments.effect_table,
        arguments.protocol,
        method_freeze_path=arguments.method_freeze,
        analysis_manifest_path=arguments.analysis_manifest,
    )
    write_real_analysis_interval_diagnostics(
        arguments.output,
        payload,
        overwrite=arguments.overwrite,
    )
    print(
        json.dumps(
            {
                "artifact_kind": payload["artifact_kind"],
                "diagnostic_id": payload["diagnostic_id"],
                "output": str(arguments.output),
                "primary_interval_method": "target_session_bootstrap_t",
                "student_t_may_veto_positive_claim": True,
                "student_t_may_rescue_primary_failure": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
