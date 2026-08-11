from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType

import pytest


_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT_PATH = _REPOSITORY_ROOT / "scripts" / "ci" / "check_stable_core_mypy.py"


def _load_gate_module() -> ModuleType:
    module_name = "_causal4d_stable_core_mypy_gate_test"
    spec = importlib.util.spec_from_file_location(module_name, _SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load stable-core MyPy gate")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_GATE = _load_gate_module()
DiagnosticKey = _GATE.DiagnosticKey
MypyDiagnostic = _GATE.MypyDiagnostic
StableCoreMypyError = _GATE.StableCoreMypyError


_SAMPLE_OUTPUT = """\
src/causal4d/contracts.py:1081: error: Argument 1 to "savez_compressed" has incompatible type "**dict[str, ndarray[Any, Any]]"; expected "bool"  [arg-type]
src/causal4d/intervention_abduction.py:253: error: Need type annotation for "flat_indices"  [var-annotated]
src/causal4d/intervention_abduction.py:608: error: Name "metadata" already defined on line 587  [no-redef]
src/causal4d/intervention_abduction.py:664: error: Need type annotation for "result"  [var-annotated]
Found 4 errors in 2 files (checked 45 source files)
"""


def _diagnostic(
    path: str,
    line: int,
    code: str,
    message: str,
) -> object:
    return MypyDiagnostic(
        key=DiagnosticKey(path=path, line=line, code=code),
        message=message,
    )


def test_parser_extracts_only_error_diagnostics() -> None:
    diagnostics = _GATE.parse_mypy_diagnostics(_SAMPLE_OUTPUT)
    assert tuple(value.key for value in diagnostics) == tuple(
        sorted(_GATE.EXPECTED_DEBT)
    )
    assert all("Found 4 errors" not in value.message for value in diagnostics)


def test_exact_current_debt_is_accepted() -> None:
    diagnostics = _GATE.parse_mypy_diagnostics(_SAMPLE_OUTPUT)
    accepted = _GATE.validate_exact_debt(diagnostics)
    assert tuple(value.key for value in accepted) == tuple(
        sorted(_GATE.EXPECTED_DEBT)
    )


@pytest.mark.parametrize("mode", ["missing", "unexpected", "wrong_message", "duplicate"])
def test_debt_drift_fails_closed(mode: str) -> None:
    diagnostics = list(_GATE.parse_mypy_diagnostics(_SAMPLE_OUTPUT))
    if mode == "missing":
        diagnostics.pop()
    elif mode == "unexpected":
        diagnostics.append(
            _diagnostic(
                "src/causal4d/counterfactual.py",
                1,
                "assignment",
                "Incompatible assignment",
            )
        )
    elif mode == "wrong_message":
        first = diagnostics[0]
        diagnostics[0] = _diagnostic(
            first.key.path,
            first.key.line,
            first.key.code,
            "different diagnostic text",
        )
    else:
        diagnostics.append(diagnostics[0])

    with pytest.raises(StableCoreMypyError, match="stable-core MyPy debt changed"):
        _GATE.validate_exact_debt(diagnostics)


def test_registered_debt_is_anchored_to_the_current_source_lines() -> None:
    expected_lines = {
        DiagnosticKey("src/causal4d/contracts.py", 1081, "arg-type"): "**arrays,",
        DiagnosticKey(
            "src/causal4d/intervention_abduction.py", 253, "var-annotated"
        ): "flat_indices = np.arange(start, stop, dtype=np.int64)",
        DiagnosticKey(
            "src/causal4d/intervention_abduction.py", 608, "no-redef"
        ): "metadata: dict[str, Any] = {",
        DiagnosticKey(
            "src/causal4d/intervention_abduction.py", 664, "var-annotated"
        ): "result = np.zeros((hypothesis_count, particle_count), dtype=float)",
    }
    assert set(expected_lines) == set(_GATE.EXPECTED_DEBT)
    for key, expected in expected_lines.items():
        lines = (_REPOSITORY_ROOT / key.path).read_text(encoding="utf-8").splitlines()
        assert lines[key.line - 1].strip() == expected


def test_ratchet_covers_the_stable_counterfactual_core() -> None:
    expected_targets = {
        "src/causal4d/contracts.py",
        "src/causal4d/counterfactual.py",
        "src/causal4d/grouped_likelihood.py",
        "src/causal4d/intervention_abduction.py",
        "src/causal4d/observation_evidence.py",
        "src/causal4d/api/v1.py",
    }
    assert expected_targets.issubset(set(_GATE.STABLE_CORE_TARGETS))
    assert len(_GATE.STABLE_CORE_TARGETS) == len(set(_GATE.STABLE_CORE_TARGETS))
