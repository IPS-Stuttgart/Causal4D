"""Cross-fitted falsification test for realized-intervention sufficiency."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass, field
from itertools import product
from math import comb
from typing import Any, Mapping, Sequence

import numpy as np

from causal4d.immutable_json import plain_json, validated_json_mapping


@dataclass(frozen=True)
class CausalSufficiencyResult:
    """Cross-fitted incremental predictive value of commanded-action identity."""

    baseline_rmse: float
    command_augmented_rmse: float
    relative_rmse_reduction: float
    permutation_p_value: float
    command_effect_detected: bool
    execution_count: int
    group_count: int
    command_count: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        numeric = (
            self.baseline_rmse,
            self.command_augmented_rmse,
            self.relative_rmse_reduction,
            self.permutation_p_value,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("sufficiency statistics must be finite")
        if self.baseline_rmse < 0.0 or self.command_augmented_rmse < 0.0:
            raise ValueError("RMSE values must be nonnegative")
        if not 0.0 <= self.permutation_p_value <= 1.0:
            raise ValueError("permutation_p_value must lie in [0, 1]")
        if min(self.execution_count, self.group_count, self.command_count) < 1:
            raise ValueError("counts must be positive")
        object.__setattr__(
            self,
            "metadata",
            validated_json_mapping(
                self.metadata,
                error_message="metadata must contain finite JSON values",
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "baseline_rmse": self.baseline_rmse,
            "command_augmented_rmse": self.command_augmented_rmse,
            "relative_rmse_reduction": self.relative_rmse_reduction,
            "permutation_p_value": self.permutation_p_value,
            "command_effect_detected": self.command_effect_detected,
            "execution_count": self.execution_count,
            "group_count": self.group_count,
            "command_count": self.command_count,
            "metadata": plain_json(self.metadata),
        }


def _matrix(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)
    if array.ndim != 2 or array.shape[0] < 4 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite execution-by-feature matrix")
    return array


def _identifiers(values: Sequence[str], count: int, name: str) -> tuple[str, ...]:
    identifiers = tuple(map(str, values))
    if len(identifiers) != count or any(not value for value in identifiers):
        raise ValueError(f"{name} must contain one nonempty value per execution")
    return identifiers


def _partition_indices(identifiers: tuple[str, ...]) -> tuple[tuple[int, ...], ...]:
    positions: dict[str, list[int]] = {}
    for index, identifier in enumerate(identifiers):
        positions.setdefault(identifier, []).append(index)
    return tuple(tuple(indices) for indices in positions.values())


def _command_features(
    command_ids: tuple[str, ...],
    categories: tuple[str, ...],
) -> np.ndarray:
    if len(categories) <= 1:
        return np.empty((len(command_ids), 0), dtype=float)
    return np.asarray(
        [
            [float(value == category) for category in categories[1:]]
            for value in command_ids
        ],
        dtype=float,
    )


def _ridge_predict(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    test_features: np.ndarray,
    *,
    ridge: float,
) -> np.ndarray:
    mean = np.mean(train_features, axis=0)
    scale = np.std(train_features, axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    train = (train_features - mean) / scale
    test = (test_features - mean) / scale
    train_design = np.column_stack((np.ones(len(train)), train))
    test_design = np.column_stack((np.ones(len(test)), test))

    if ridge > 0.0:
        penalty = np.eye(train_design.shape[1], dtype=float) * np.sqrt(ridge)
        penalty[0, 0] = 0.0
        solve_design = np.vstack((train_design, penalty))
        solve_targets = np.vstack(
            (
                train_targets,
                np.zeros(
                    (train_design.shape[1], train_targets.shape[1]),
                    dtype=float,
                ),
            )
        )
    else:
        solve_design = train_design
        solve_targets = train_targets

    coefficients = np.linalg.lstsq(solve_design, solve_targets, rcond=None)[0]
    return test_design @ coefficients


def _cross_fitted_predictions(
    features: np.ndarray,
    targets: np.ndarray,
    group_ids: tuple[str, ...],
    *,
    ridge: float,
) -> np.ndarray:
    predictions = np.empty_like(targets)
    groups = tuple(dict.fromkeys(group_ids))
    for group in groups:
        test = np.asarray([value == group for value in group_ids], dtype=bool)
        train = ~test
        if int(np.sum(train)) < 2 or int(np.sum(test)) < 1:
            raise ValueError("every cross-fit fold needs training and test executions")
        predictions[test] = _ridge_predict(
            features[train],
            targets[train],
            features[test],
            ridge=ridge,
        )
    return predictions


def _group_balanced_rmse(
    targets: np.ndarray,
    prediction: np.ndarray,
    group_indices: tuple[tuple[int, ...], ...],
) -> float:
    execution_mse = np.mean(np.square(targets - prediction), axis=1)
    group_mse = np.asarray(
        [np.mean(execution_mse[list(indices)]) for indices in group_indices],
        dtype=float,
    )
    return float(np.sqrt(np.mean(group_mse)))


def _relative_improvement(
    targets: np.ndarray,
    baseline_prediction: np.ndarray,
    augmented_prediction: np.ndarray,
    group_indices: tuple[tuple[int, ...], ...],
) -> tuple[float, float, float]:
    baseline_rmse = _group_balanced_rmse(
        targets,
        baseline_prediction,
        group_indices,
    )
    augmented_rmse = _group_balanced_rmse(
        targets,
        augmented_prediction,
        group_indices,
    )
    if baseline_rmse <= np.finfo(float).eps:
        return baseline_rmse, augmented_rmse, 0.0
    return (
        baseline_rmse,
        augmented_rmse,
        float(1.0 - augmented_rmse / baseline_rmse),
    )


def _bounded_multiset_permutation_count(
    values: tuple[str, ...],
    *,
    limit: int,
) -> int:
    total = 1
    placed = 0
    for count in Counter(values).values():
        total *= comb(placed + count, count)
        if total > limit:
            return limit + 1
        placed += count
    return total


def _bounded_assignment_count(
    commands: tuple[str, ...],
    blocks: tuple[tuple[int, ...], ...],
    *,
    limit: int,
) -> int:
    total = 1
    for indices in blocks:
        block_commands = tuple(commands[index] for index in indices)
        total *= _bounded_multiset_permutation_count(
            block_commands,
            limit=limit,
        )
        if total > limit:
            return limit + 1
    return total


def _distinct_permutations(values: tuple[str, ...]) -> Iterator[tuple[str, ...]]:
    counts = Counter(values)
    categories = tuple(sorted(counts))
    current: list[str] = []

    def generate() -> Iterator[tuple[str, ...]]:
        if len(current) == len(values):
            yield tuple(current)
            return
        for category in categories:
            if counts[category] < 1:
                continue
            counts[category] -= 1
            current.append(category)
            yield from generate()
            current.pop()
            counts[category] += 1

    yield from generate()


def _exact_command_assignments(
    commands: tuple[str, ...],
    blocks: tuple[tuple[int, ...], ...],
) -> Iterator[tuple[str, ...]]:
    block_assignments = tuple(
        tuple(
            _distinct_permutations(
                tuple(commands[index] for index in block_indices)
            )
        )
        for block_indices in blocks
    )
    for assignment_by_block in product(*block_assignments):
        assignment = list(commands)
        for block_indices, block_values in zip(blocks, assignment_by_block):
            for index, value in zip(block_indices, block_values):
                assignment[index] = value
        yield tuple(assignment)


def _random_command_assignment(
    commands: np.ndarray,
    blocks: tuple[tuple[int, ...], ...],
    rng: np.random.Generator,
) -> tuple[str, ...]:
    assignment = commands.copy()
    for block_indices in blocks:
        indices = np.asarray(block_indices, dtype=int)
        assignment[indices] = rng.permutation(commands[indices])
    return tuple(map(str, assignment))


def _tail_count(
    null_improvements: np.ndarray,
    observed_improvement: float,
) -> tuple[int, float]:
    comparison_scale = max(
        1.0,
        abs(observed_improvement),
        float(np.max(np.abs(null_improvements))),
    )
    tolerance = 64.0 * np.finfo(float).eps * comparison_scale
    count = int(
        np.sum(null_improvements >= observed_improvement - tolerance)
    )
    return count, tolerance


def assess_command_residual_sufficiency(
    future_residual_targets: np.ndarray,
    realization_features: np.ndarray,
    command_ids: Sequence[str],
    *,
    group_ids: Sequence[str] | None = None,
    permutation_block_ids: Sequence[str] | None = None,
    ridge: float = 1.0e-6,
    permutation_count: int = 199,
    maximum_exact_assignments: int = 10_000,
    random_seed: int = 0,
    significance_level: float = 0.05,
    minimum_relative_improvement: float = 0.01,
) -> CausalSufficiencyResult:
    """Test whether command identity predicts residuals after conditioning on ``z``.

    A significant held-out gain from command identity is evidence that the
    supplied realization features are causally incomplete. Cross-fitting is by
    independent group, normally execution session, and every group contributes
    equal weight to the RMSE statistic.

    ``permutation_block_ids`` restricts command-label reassignment to the
    registered randomization blocks. The command multiset within every block is
    preserved exactly. The test enumerates every distinct allowed assignment
    when their count does not exceed ``maximum_exact_assignments``; otherwise it
    uses a plus-one Monte Carlo p-value. Omitting the block identifiers retains
    the historical global-exchangeability assumption. The permutation test is a
    finite-sample diagnostic, not a proof of conditional independence.
    """

    targets = _matrix(future_residual_targets, "future_residual_targets")
    features = _matrix(realization_features, "realization_features")
    if len(features) != len(targets):
        raise ValueError("targets and realization features must align")
    commands = _identifiers(command_ids, len(targets), "command_ids")
    groups = (
        tuple(f"execution-{index}" for index in range(len(targets)))
        if group_ids is None
        else _identifiers(group_ids, len(targets), "group_ids")
    )
    permutation_blocks = (
        None
        if permutation_block_ids is None
        else _identifiers(
            permutation_block_ids,
            len(targets),
            "permutation_block_ids",
        )
    )
    categories = tuple(sorted(set(commands)))
    unique_groups = tuple(dict.fromkeys(groups))
    group_indices = _partition_indices(groups)
    block_indices = (
        (tuple(range(len(commands))),)
        if permutation_blocks is None
        else _partition_indices(permutation_blocks)
    )
    if len(categories) < 2:
        raise ValueError("at least two commanded actions are required")
    if len(unique_groups) < 3:
        raise ValueError("at least three independent cross-fit groups are required")
    if not np.isfinite(ridge) or ridge < 0.0:
        raise ValueError("ridge must be finite and nonnegative")
    if type(permutation_count) is not int or permutation_count < 1:
        raise ValueError("permutation_count must be a positive integer")
    if type(maximum_exact_assignments) is not int or maximum_exact_assignments < 1:
        raise ValueError("maximum_exact_assignments must be a positive integer")
    if not 0.0 < significance_level < 1.0:
        raise ValueError("significance_level must lie in (0, 1)")
    if not np.isfinite(minimum_relative_improvement):
        raise ValueError("minimum_relative_improvement must be finite")

    bounded_assignment_count = _bounded_assignment_count(
        commands,
        block_indices,
        limit=maximum_exact_assignments,
    )
    if bounded_assignment_count == 1:
        raise ValueError(
            "the permutation design does not permit any command reassignment"
        )
    use_exact_permutations = bounded_assignment_count <= maximum_exact_assignments

    command_features = _command_features(commands, categories)
    augmented_features = np.column_stack((features, command_features))
    baseline_prediction = _cross_fitted_predictions(
        features,
        targets,
        groups,
        ridge=ridge,
    )
    augmented_prediction = _cross_fitted_predictions(
        augmented_features,
        targets,
        groups,
        ridge=ridge,
    )
    baseline_rmse, augmented_rmse, observed_improvement = _relative_improvement(
        targets,
        baseline_prediction,
        augmented_prediction,
        group_indices,
    )

    if use_exact_permutations:
        assignments: Iterator[tuple[str, ...]] = _exact_command_assignments(
            commands,
            block_indices,
        )
        evaluated_assignment_count = bounded_assignment_count
    else:
        rng = np.random.default_rng(random_seed)
        command_array = np.asarray(commands, dtype=object)
        assignments = (
            _random_command_assignment(command_array, block_indices, rng)
            for _ in range(permutation_count)
        )
        evaluated_assignment_count = permutation_count

    null_improvements = np.empty(evaluated_assignment_count, dtype=float)
    for index, permuted in enumerate(assignments):
        permuted_features = np.column_stack(
            (features, _command_features(permuted, categories))
        )
        permuted_prediction = _cross_fitted_predictions(
            permuted_features,
            targets,
            groups,
            ridge=ridge,
        )
        _, _, null_improvements[index] = _relative_improvement(
            targets,
            baseline_prediction,
            permuted_prediction,
            group_indices,
        )

    extreme_count, comparison_tolerance = _tail_count(
        null_improvements,
        observed_improvement,
    )
    if use_exact_permutations:
        p_value = float(extreme_count / evaluated_assignment_count)
        p_value_estimator = "exact_tail_fraction"
    else:
        p_value = float(
            (1 + extreme_count) / (evaluated_assignment_count + 1)
        )
        p_value_estimator = "plus_one_monte_carlo"
    detected = bool(
        observed_improvement >= minimum_relative_improvement
        and p_value <= significance_level
    )
    return CausalSufficiencyResult(
        baseline_rmse=baseline_rmse,
        command_augmented_rmse=augmented_rmse,
        relative_rmse_reduction=observed_improvement,
        permutation_p_value=p_value,
        command_effect_detected=detected,
        execution_count=len(targets),
        group_count=len(unique_groups),
        command_count=len(categories),
        metadata={
            "cross_fit_unit": "group",
            "score_unit": "equal_weight_cross_fit_group",
            "ridge": float(ridge),
            "ridge_solver": "augmented_least_squares",
            "permutation_scheme": (
                "global"
                if permutation_blocks is None
                else "within_registered_block"
            ),
            "permutation_mode": (
                "exact" if use_exact_permutations else "monte_carlo"
            ),
            "permutation_block_count": len(block_indices),
            "permutation_block_sizes": [len(indices) for indices in block_indices],
            "fixed_permutation_block_count": sum(
                len(set(commands[index] for index in indices)) == 1
                for indices in block_indices
            ),
            "maximum_exact_assignments": int(maximum_exact_assignments),
            "distinct_assignment_count": (
                int(bounded_assignment_count)
                if use_exact_permutations
                else None
            ),
            "distinct_assignment_count_exceeds_exact_limit": bool(
                not use_exact_permutations
            ),
            "evaluated_assignment_count": int(evaluated_assignment_count),
            "requested_monte_carlo_permutation_count": int(permutation_count),
            "p_value_estimator": p_value_estimator,
            "tail_comparison_tolerance": float(comparison_tolerance),
            "random_seed": int(random_seed),
            "significance_level": float(significance_level),
            "minimum_relative_improvement": float(minimum_relative_improvement),
            "interpretation": (
                "detected command value indicates incomplete realized-intervention "
                "features; non-detection is not proof of sufficiency"
            ),
        },
    )
