"""Single module entry point for the external forecast/physics bridge."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from importlib import import_module

_COMMANDS = {
    "import-forecast": "causal4d.cli.external_forecast_import:main",
    "import-rollouts": "causal4d.cli.external_rollout_import:main",
    "doctor": "causal4d.cli.external_bridge_doctor:main",
    "map-nodes": "causal4d.cli.external_node_mapping:main",
    "run": "causal4d.cli.external_bridge_run:main",
}


def _help() -> str:
    return "\n".join(
        (
            "usage: python -m causal4d.cli.external_bridge_workflow ",
            "       {import-forecast,import-rollouts,doctor,map-nodes,run} ...",
            "",
            "Portable MolmoMotion/external-simulator bridge commands:",
            "  import-forecast  Normalize a sparse external forecast.",
            "  import-rollouts  Normalize a finite external rollout bank.",
            "  doctor           Validate node, time, anchor, and scale alignment.",
            "  map-nodes        Audit a one-to-one geometric node assignment.",
            "  run              Sweep semantic weights and publish a report bundle.",
            "",
            "Append --help after a subcommand for its detailed arguments.",
        )
    )


def _target(specification: str) -> Callable[[Sequence[str] | None], int]:
    module_name, function_name = specification.split(":", 1)
    return getattr(import_module(module_name), function_name)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(argv or ())
    if not arguments or arguments[0] in {"-h", "--help"}:
        print(_help())
        return 0
    command = arguments.pop(0)
    if command not in _COMMANDS:
        available = ", ".join(sorted(_COMMANDS))
        raise SystemExit(f"unknown bridge command {command!r}; choose from {available}")
    return int(_target(_COMMANDS[command])(arguments) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
