#!/usr/bin/env python3
"""Technical launcher for the unchanged PokeFlex source-panel v3 science."""

from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent
    / "extract_pokeflex_source_active_probe_panel_v3.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "pokeflex_source_active_probe_panel_v3_launcher",
        MODULE_PATH,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load PokeFlex source-panel v3")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


if __name__ == "__main__":
    raise SystemExit(load_module().main())
