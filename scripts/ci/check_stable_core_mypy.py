#!/usr/bin/env python3
"""Run the stable-core MyPy ratchet with an exact, fail-closed debt manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import subprocess
import sys
from typing import Final, Sequence


STABLE_CORE_TARGETS: Final = (
    "scripts/ci",
    "src/causal4d/atomic_io.py",
    "src/causal4d/result_bundle_publication.py",
    "src/causal4d/trusted_pickle.py",
    "src/causal4d/immutable_array.py",
    "src/causal4d/immutable_json.py",
    "src/causal4d/low_rank_numerics.py",
    "src/causal4d/contracts.py",
    "src/causal4d/counterfactual.py",
    "src/causal4d/grouped_likelihood.py",
    "src/causal4d/intervention_abduction.py",
    "src/causal4d/observation_evidence.py",
    "src/causal4d/api/v1.py",
    "src/causal4d/provider_contract.py",
    "src/causal4d/replay_provider_contract.py",
)


@dataclass(frozen=True, order=True)
class DiagnosticKey:
    """Stable location and MyPy error-code identity for one diagnostic."""

    path: str
    line: int
    code: str


@dataclass(frozen=True)
class MypyDiagnostic:
    """One parsed MyPy error diagnostic."""

    key: DiagnosticKey
    message: str


EXPECTED_DEBT: Final = {
    DiagnosticKey("src/causal4d/contracts.py", 1081, "arg-type"): (
        'Argument 1 to "savez_compressed" has incompatible type'
    ),
    DiagnosticKey(
        "src/causal4d/intervention_abduction.py", 253, "var-annotated"
    ): 'Need type annotation for "flat_indices"',
    DiagnosticKey("src/causal4d/intervention_abduction.py", 608, "no-redef"): (
        'Name "metadata" already defined on line 587'
    ),
    DiagnosticKey(
        "src/causal4d/intervention_abduction.py", 664, "var-annotated"
    ): 'Need type annotation for "result"',
}

_ERROR_PATTERN: Final = re.compile(
    r"^(?P<path>.+?):(?P<line>[0-9]+): error: "
    r"(?P<message>.*?)  \[(?P<code>[^]]+)]$"
)


class StableCoreMypyError(RuntimeError):
    """The current diagnostics differ from the exact registered debt set."""


def parse_mypy_diagnostics(output: str) -> tuple[MypyDiagnostic, ...]:
    """Extract deterministic error diagnostics from non-pretty MyPy output."""

    diagnostics: list[MypyDiagnostic] = []
    for raw_line in output.splitlines():
        match = _ERROR_PATTERN.fullmatch(raw_line.strip())
        if match is None:
            continue
        diagnostics.append(
            MypyDiagnostic(
                key=DiagnosticKey(
                    path=match.group("path").replace("\\", "/"),
                    line=int(match.group("line")),
                    code=match.group("code"),
                ),
                message=match.group("message"),
            )
        )
    return tuple(diagnostics)


def validate_exact_debt(
    diagnostics: Sequence[MypyDiagnostic],
) -> tuple[MypyDiagnostic, ...]:
    """Accept only the complete exact debt set and reject every drift."""

    by_key: dict[DiagnosticKey, MypyDiagnostic] = {}
    duplicate_keys: list[DiagnosticKey] = []
    for diagnostic in diagnostics:
        if diagnostic.key in by_key:
            duplicate_keys.append(diagnostic.key)
        by_key[diagnostic.key] = diagnostic

    expected_keys = set(EXPECTED_DEBT)
    actual_keys = set(by_key)
    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(actual_keys - expected_keys)
    wrong_messages = sorted(
        key
        for key in expected_keys & actual_keys
        if EXPECTED_DEBT[key] not in by_key[key].message
    )
    if duplicate_keys or missing or unexpected or wrong_messages:
        raise StableCoreMypyError(
            "stable-core MyPy debt changed; "
            f"duplicate={sorted(duplicate_keys)}, missing={missing}, "
            f"unexpected={unexpected}, wrong_message={wrong_messages}"
        )
    return tuple(by_key[key] for key in sorted(expected_keys))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    return parser


def run_stable_core_mypy(
    *,
    repository_root: Path,
    python_version: str,
) -> tuple[MypyDiagnostic, ...]:
    """Run MyPy and return the exact accepted debt diagnostics."""

    command = [
        sys.executable,
        "-m",
        "mypy",
        "--python-version",
        python_version,
        "--show-error-codes",
        "--no-color-output",
        "--no-pretty",
        *STABLE_CORE_TARGETS,
    ]
    completed = subprocess.run(
        command,
        cwd=repository_root,
        check=False,
        capture_output=True,
        text=True,
    )
    output = completed.stdout + completed.stderr
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    diagnostics = parse_mypy_diagnostics(output)
    accepted = validate_exact_debt(diagnostics)
    if completed.returncode != 1:
        raise StableCoreMypyError(
            "MyPy must return one for the exact registered debt set; "
            f"returncode={completed.returncode}"
        )
    print(
        "accepted exact stable-core MyPy debt: "
        + ", ".join(
            f"{value.key.path}:{value.key.line}[{value.key.code}]"
            for value in accepted
        )
    )
    return accepted


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        run_stable_core_mypy(
            repository_root=arguments.repository_root.resolve(),
            python_version=arguments.python_version,
        )
    except StableCoreMypyError as error:
        print(f"stable-core MyPy gate failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
