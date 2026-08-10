import numpy as np
import pytest

from causal4d.causal_sufficiency import (
    _group_balanced_rmse,
    _ridge_predict,
    assess_command_residual_sufficiency,
)


def _paired_command_effect() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    list[str],
    list[str],
]:
    commands = np.asarray(["a", "b"] * 6)
    blocks = [f"pair-{index}" for index in range(6) for _ in range(2)]
    realization = np.zeros((len(commands), 1), dtype=float)
    target = 3.0 * (commands == "b")[:, None]
    return target, realization, commands, blocks, blocks


def test_exact_block_randomization_detects_paired_command_effect() -> None:
    target, realization, commands, groups, blocks = _paired_command_effect()
    result = assess_command_residual_sufficiency(
        target,
        realization,
        commands,
        group_ids=groups,
        permutation_block_ids=blocks,
        permutation_count=1,
        maximum_exact_assignments=64,
    )

    assert result.command_effect_detected
    assert result.relative_rmse_reduction > 0.99
    assert result.permutation_p_value == pytest.approx(2.0 / 64.0)
    assert result.metadata["permutation_scheme"] == "within_registered_block"
    assert result.metadata["permutation_mode"] == "exact"
    assert result.metadata["distinct_assignment_count"] == 64
    assert result.metadata["evaluated_assignment_count"] == 64
    assert result.metadata["requested_monte_carlo_permutation_count"] == 1


def test_block_randomization_prevents_block_composition_false_positive() -> None:
    commands: list[str] = []
    groups: list[str] = []
    target_values: list[float] = []
    for index in range(6):
        high_response_block = index < 3
        commands.extend(
            ["b", "b", "b", "a"] if high_response_block else ["b", "a", "a", "a"]
        )
        groups.extend([f"session-{index}"] * 4)
        target_values.extend([3.0 if high_response_block else 0.0] * 4)

    target = np.asarray(target_values, dtype=float)[:, None]
    realization = np.zeros((len(commands), 1), dtype=float)
    global_result = assess_command_residual_sufficiency(
        target,
        realization,
        commands,
        group_ids=groups,
        permutation_count=199,
        maximum_exact_assignments=32,
        random_seed=0,
    )
    blocked_result = assess_command_residual_sufficiency(
        target,
        realization,
        commands,
        group_ids=groups,
        permutation_block_ids=groups,
        permutation_count=199,
        maximum_exact_assignments=32,
        random_seed=0,
    )

    assert global_result.command_effect_detected
    assert global_result.permutation_p_value <= 0.05
    assert not blocked_result.command_effect_detected
    assert blocked_result.permutation_p_value == pytest.approx(1.0)
    assert blocked_result.relative_rmse_reduction == pytest.approx(
        global_result.relative_rmse_reduction
    )


def test_block_randomization_fails_closed_without_exchangeable_labels() -> None:
    commands = ["a"] * 4 + ["b"] * 4
    with pytest.raises(ValueError, match="does not permit any command reassignment"):
        assess_command_residual_sufficiency(
            np.zeros((8, 1), dtype=float),
            np.zeros((8, 1), dtype=float),
            commands,
            group_ids=[f"execution-{index}" for index in range(8)],
            permutation_block_ids=["a-only"] * 4 + ["b-only"] * 4,
        )


def test_exact_block_result_is_invariant_to_execution_order() -> None:
    target, realization, commands, groups, blocks = _paired_command_effect()
    reference = assess_command_residual_sufficiency(
        target,
        realization,
        commands,
        group_ids=groups,
        permutation_block_ids=blocks,
        maximum_exact_assignments=64,
    )
    order = np.asarray([7, 2, 11, 0, 9, 4, 1, 10, 5, 8, 3, 6])
    reordered = assess_command_residual_sufficiency(
        target[order],
        realization[order],
        commands[order],
        group_ids=np.asarray(groups)[order],
        permutation_block_ids=np.asarray(blocks)[order],
        maximum_exact_assignments=64,
    )

    assert reordered.baseline_rmse == pytest.approx(reference.baseline_rmse)
    assert reordered.command_augmented_rmse == pytest.approx(
        reference.command_augmented_rmse
    )
    assert reordered.relative_rmse_reduction == pytest.approx(
        reference.relative_rmse_reduction
    )
    assert reordered.permutation_p_value == pytest.approx(reference.permutation_p_value)


def test_group_balanced_rmse_gives_each_session_equal_weight() -> None:
    targets = np.zeros((10, 1), dtype=float)
    prediction = np.zeros_like(targets)
    prediction[0, 0] = 10.0
    group_indices = ((0,), tuple(range(1, 10)))

    assert _group_balanced_rmse(targets, prediction, group_indices) == pytest.approx(
        np.sqrt(50.0)
    )


def test_ridge_solver_remains_stable_for_nearly_collinear_features() -> None:
    rng = np.random.default_rng(2872)
    base = rng.normal(size=20)
    features = np.column_stack(
        [base] + [base + index * 1.0e-7 * rng.normal(size=20) for index in range(1, 8)]
    )
    targets = (2.0 * base + 1.0e-8 * rng.normal(size=20))[:, None]

    prediction = _ridge_predict(
        features,
        targets,
        features,
        ridge=0.0,
    )

    rmse = float(np.sqrt(np.mean(np.square(prediction - targets))))
    assert rmse < 5.0e-8


def test_positive_ridge_is_solved_as_an_augmented_least_squares_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_shapes: list[tuple[int, int]] = []
    original_lstsq = np.linalg.lstsq

    def recording_lstsq(
        design: np.ndarray,
        targets: np.ndarray,
        rcond: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
        observed_shapes.append(design.shape)
        return original_lstsq(design, targets, rcond=rcond)

    monkeypatch.setattr(np.linalg, "lstsq", recording_lstsq)
    train_features = np.arange(12, dtype=float).reshape(6, 2)
    train_targets = np.arange(6, dtype=float)[:, None]
    _ridge_predict(
        train_features,
        train_targets,
        train_features[:2],
        ridge=0.25,
    )

    parameter_count = train_features.shape[1] + 1
    assert observed_shapes == [(len(train_features) + parameter_count, parameter_count)]
