from pathlib import Path

from scripts.ci.run_stable_core_mypy import (
    STABLE_CORE_TARGETS,
    STRICT_MYPY_OPTIONS,
    build_command,
)


EXPECTED_TARGETS = (
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

EXPECTED_OPTIONS = (
    "--python-version",
    "3.12",
    "--disallow-untyped-defs",
    "--warn-return-any",
    "--no-implicit-reexport",
)


def test_stable_core_inventory_and_options_are_exact() -> None:
    assert STABLE_CORE_TARGETS == EXPECTED_TARGETS
    assert STRICT_MYPY_OPTIONS == EXPECTED_OPTIONS


def test_command_is_constructed_from_the_locked_policy() -> None:
    assert build_command(python_executable="python-test") == (
        "python-test",
        "-m",
        "mypy",
        *EXPECTED_OPTIONS,
        *EXPECTED_TARGETS,
    )


def test_required_workflows_invoke_the_single_policy_runner() -> None:
    invocation = "python scripts/ci/run_stable_core_mypy.py"
    for path in (
        Path(".github/workflows/ci.yml"),
        Path(".github/workflows/merge-gate.yml"),
    ):
        assert path.read_text(encoding="utf-8").count(invocation) == 1
