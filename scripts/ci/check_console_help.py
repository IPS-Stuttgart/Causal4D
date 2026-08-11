#!/usr/bin/env python3
"""Verify the single installed executable and every grouped route's ``--help``."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import tomllib


def _declared_scripts(pyproject: Path) -> dict[str, str]:
    with pyproject.open("rb") as handle:
        project = tomllib.load(handle)["project"]
    scripts = project.get("scripts", {})
    if not isinstance(scripts, dict):
        raise ValueError("project.scripts must be a table")
    return {str(name): str(target) for name, target in scripts.items()}


def _installed_scripts(distribution: str) -> dict[str, str]:
    installed = importlib.metadata.distribution(distribution)
    return {
        entry_point.name: entry_point.value
        for entry_point in installed.entry_points
        if entry_point.group == "console_scripts"
        and (entry_point.name == "causal4d" or entry_point.name.startswith("causal4d-"))
    }


def _require_installed_file(distribution: str, relative_path: str) -> Path:
    installed = importlib.metadata.distribution(distribution)
    files = installed.files
    if files is None:
        raise RuntimeError(
            f"installed distribution {distribution!r} exposes no file inventory"
        )
    normalized = {
        str(package_path).replace("\\", "/"): package_path for package_path in files
    }
    package_path = normalized.get(relative_path)
    if package_path is None:
        raise RuntimeError(
            f"installed distribution {distribution!r} does not contain {relative_path}"
        )
    resolved = Path(str(installed.locate_file(package_path)))
    if not resolved.is_file():
        raise RuntimeError(
            f"installed distribution records {relative_path}, but {resolved} is absent"
        )
    return resolved


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def _run(
    arguments: list[str],
    *,
    cwd: str,
    environment: dict[str, str],
    timeout_seconds: float,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            arguments,
            cwd=cwd,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            f"{' '.join(arguments)} timed out after {timeout_seconds:g} s"
        ) from error


def verify_console_help(
    pyproject: Path,
    *,
    distribution: str = "causal4d",
    timeout_seconds: float = 30.0,
) -> None:
    typing_marker = _require_installed_file(distribution, "causal4d/py.typed")
    print(f"verified installed PEP 561 marker: {typing_marker}")
    typing_stub = _require_installed_file(distribution, "causal4d/__init__.pyi")
    print(f"verified installed package-root typing stub: {typing_stub}")

    environment = _clean_environment()
    root_surface_script = Path(__file__).with_name("check_installed_root_surface.py")
    root_surface = _run(
        [
            sys.executable,
            str(root_surface_script),
            "--repository-root",
            str(pyproject.resolve().parent),
        ],
        cwd=str(pyproject.resolve().parent),
        environment=environment,
        timeout_seconds=timeout_seconds,
    )
    if root_surface.returncode != 0:
        raise RuntimeError(root_surface.stdout + root_surface.stderr)
    print(root_surface.stdout.strip())

    declared = _declared_scripts(pyproject)
    expected = {"causal4d": "causal4d.cli.root:main"}
    if declared != expected:
        raise RuntimeError(
            "pyproject.toml must declare exactly the single causal4d executable: "
            f"declared={declared}"
        )
    installed = _installed_scripts(distribution)
    if installed != expected:
        raise RuntimeError(
            "installed console-script metadata must contain only causal4d: "
            f"installed={installed}"
        )

    executable = shutil.which("causal4d")
    if executable is None:
        raise RuntimeError("causal4d executable is not on PATH")
    failures: list[str] = []
    with tempfile.TemporaryDirectory(prefix="causal4d-cli-help-") as directory:
        validation = _run(
            [
                executable,
                "commands",
                "validate",
                "--json",
                "--require-installed",
            ],
            cwd=directory,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        if validation.returncode != 0:
            raise RuntimeError(validation.stdout + validation.stderr)
        report = json.loads(validation.stdout)
        if report.get("valid") is not True:
            raise RuntimeError(f"installed command inventory is invalid: {report}")
        if report.get("removed_historical_executables_installed"):
            raise RuntimeError("removed historical executables remain installed")

        listing = _run(
            [executable, "commands", "list", "--json"],
            cwd=directory,
            environment=environment,
            timeout_seconds=timeout_seconds,
        )
        if listing.returncode != 0:
            raise RuntimeError(listing.stdout + listing.stderr)
        inventory = json.loads(listing.stdout)
        routes = [tuple(item["route"]) for item in inventory]
        if not routes or len(routes) != len(set(routes)):
            raise RuntimeError(
                "installed grouped route inventory is empty or duplicated"
            )

        for route in routes:
            completed = _run(
                [executable, *route, "--help"],
                cwd=directory,
                environment=environment,
                timeout_seconds=timeout_seconds,
            )
            output = completed.stdout + completed.stderr
            invocation = "causal4d " + " ".join(route)
            if completed.returncode != 0:
                failures.append(
                    f"{invocation}: exited with {completed.returncode}\n"
                    f"{output[-4000:]}"
                )
            elif "usage:" not in output.lower():
                failures.append(
                    f"{invocation}: successful help output did not contain 'usage:'"
                )
            else:
                print(f"ok {invocation}")

    if failures:
        joined = "\n\n".join(failures)
        raise RuntimeError(
            f"{len(failures)} of {len(routes)} grouped routes failed --help:\n{joined}"
        )
    print(f"verified --help for all {len(routes)} grouped routes")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pyproject", type=Path, default=Path("pyproject.toml"))
    parser.add_argument("--distribution", default="causal4d")
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    arguments = parser.parse_args(argv)
    verify_console_help(
        arguments.pyproject,
        distribution=arguments.distribution,
        timeout_seconds=arguments.timeout_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
