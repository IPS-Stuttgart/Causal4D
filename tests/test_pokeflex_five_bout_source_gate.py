from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/remote/run_pokeflex_five_bout_source_gate.py"
SPEC = importlib.util.spec_from_file_location("pokeflex_five_bout_source_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_symmetric_chamfer_is_zero_for_identity_and_symmetric() -> None:
    first = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    second = first + np.array([0.0, 0.0, 0.25])
    assert MODULE.symmetric_chamfer(first, first) == pytest.approx(0.0)
    assert MODULE.symmetric_chamfer(first, second) == pytest.approx(
        MODULE.symmetric_chamfer(second, first)
    )
    assert MODULE.symmetric_chamfer(first, second) > 0.0


def test_fit_policy_separates_task_value_from_marginal_variance() -> None:
    # Candidate 1 tracks the registered query. Candidate 2 has much larger
    # marginal variance but is deliberately nuisance-only.
    task_signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    nuisance = np.array([-30.0, 30.0, -25.0, 25.0, -20.0, 20.0])
    matrix = np.column_stack(
        (
            task_signal,
            nuisance,
            np.zeros(6),
            np.ones(6),
            task_signal + 0.05 * np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
        )
    )
    fit = MODULE.fit_policy(matrix)
    assert fit.selected_task == 0
    assert fit.selected_generic == 1
    assert fit.task_values[0] > fit.task_values[1]
    assert fit.variances[1] > fit.variances[0]


def test_dependence_destruction_can_remove_task_value() -> None:
    signal = np.array([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])
    matrix = np.column_stack(
        (
            signal,
            np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
            np.zeros(6),
            np.ones(6),
            signal,
        )
    )
    matched = MODULE.fit_policy(matrix)
    destroyed_query = np.array([0.0, 3.0, -2.0, 2.0, -1.0, 1.0])
    destroyed = MODULE.fit_policy(matrix, destroyed_query)
    assert max(matched.task_values) > max(destroyed.task_values)


def test_source_evaluation_records_all_policies() -> None:
    names = [f"object-{index}" for index in range(6)]
    responses = {
        name: [
            value,
            (-1.0) ** index * 10.0,
            0.1 * index,
            -0.2 * index,
            value + 0.01 * ((-1.0) ** index),
        ]
        for index, (name, value) in enumerate(zip(names, [-2, -1, 0, 1, 2, 3]))
    }
    result = MODULE.source_evaluation(names, responses)
    assert len(result["folds"]) == len(names)
    assert result["full_source_fit"]["selected_task_bout"] == 1
    assert result["full_source_fit"]["selected_generic_bout"] == 2
    assert result["aggregates"]["task_conditioned_mean_squared_log_error"] < result[
        "aggregates"
    ]["generic_information_mean_squared_log_error"]
