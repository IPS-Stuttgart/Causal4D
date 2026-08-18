"""Run the required strict MyPy policy over the stable causal core."""

from __future__ import annotations

import subprocess
import sys
from typing import Final


STRICT_MYPY_OPTIONS: Final = (
    "--python-version",
    "3.12",
    "--disallow-untyped-defs",
    "--warn-return-any",
    "--no-implicit-reexport",
)

STABLE_CORE_TARGETS: Final = (
    "src/causal4d/api/v1.py",
    "src/causal4d/artifacts/v1.py",
    "src/causal4d/inference/v1.py",
    "src/causal4d/contracts.py",
    "src/causal4d/counterfactual.py",
    "src/causal4d/grouped_likelihood.py",
    "src/causal4d/intervention_abduction.py",
    "src/causal4d/observation_evidence.py",
    "src/causal4d/atomic_io.py",
    "src/causal4d/result_bundle_verification.py",
    "src/causal4d/result_bundle_publication.py",
    "src/causal4d/provider_contract.py",
    "src/causal4d/replay_provider_contract.py",
)


def build_command(
    *,
    python_executable: str = sys.executable,
) -> tuple[str, ...]:
    """Return the exact stable-core MyPy command."""

    return (
        python_executable,
        "-m",
        "mypy",
        *STRICT_MYPY_OPTIONS,
        *STABLE_CORE_TARGETS,
    )


def main() -> int:
    """Execute the stable-core MyPy policy and preserve its exit status."""

    return subprocess.run(build_command(), check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
