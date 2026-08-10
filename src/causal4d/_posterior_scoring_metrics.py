"""Proper scores and variance decomposition for trajectory posteriors."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from causal4d._posterior_scoring_contracts import (
    _as_component_matrix,
    _require_finite_nonnegative,
    _require_nonempty_string,
    _validated_weights,
)
from causal4d._posterior_scoring_results import (
    GaussianQueryScoreV1,
    OrderedVarianceAttributionV1,
    VarianceContributionV1,
)


def weighted_energy_score(
    samples: Any,
    weights: Any,
    truth: Any,
    *,
    sample_chunk_size: int = 16,
    coordinate_chunk_size: int = 4096,
) -> float:
    """Return the exact weighted energy score under an RMS Euclidean norm.

    Dividing the Euclidean norm by ``sqrt(D)`` preserves propriety while keeping
    the score in metres and comparable across registered query dimensions.
    """

    sample_matrix = _as_component_matrix(samples, name="samples")
    weight_vector = _validated_weights(weights, expected_count=len(sample_matrix))
    truth_vector = np.asarray(truth, dtype=float).reshape(-1)
    if sample_matrix.shape[1] != len(truth_vector):
        raise ValueError("truth dimension does not match samples")
    if not np.all(np.isfinite(truth_vector)):
        raise ValueError("truth must be finite")
    if type(sample_chunk_size) is not int or sample_chunk_size < 1:
        raise ValueError("sample_chunk_size must be a positive integer")
    if type(coordinate_chunk_size) is not int or coordinate_chunk_size < 1:
        raise ValueError("coordinate_chunk_size must be a positive integer")

    dimension = sample_matrix.shape[1]
    first_squared = np.zeros(len(sample_matrix), dtype=float)
    for start in range(0, dimension, coordinate_chunk_size):
        stop = min(start + coordinate_chunk_size, dimension)
        residual = sample_matrix[:, start:stop] - truth_vector[start:stop]
        first_squared += np.einsum("kd,kd->k", residual, residual)
    first_term = float(
        np.dot(weight_vector, np.sqrt(np.maximum(first_squared, 0.0) / dimension))
    )

    pair_term = 0.0
    all_count = len(sample_matrix)
    for row_start in range(0, all_count, sample_chunk_size):
        row_stop = min(row_start + sample_chunk_size, all_count)
        pair_squared = np.zeros((row_stop - row_start, all_count), dtype=float)
        for coordinate_start in range(0, dimension, coordinate_chunk_size):
            coordinate_stop = min(
                coordinate_start + coordinate_chunk_size,
                dimension,
            )
            left = sample_matrix[
                row_start:row_stop,
                coordinate_start:coordinate_stop,
            ]
            right = sample_matrix[:, coordinate_start:coordinate_stop]
            pair_squared += (
                np.sum(np.square(left), axis=1)[:, None]
                + np.sum(np.square(right), axis=1)[None, :]
                - 2.0 * left @ right.T
            )
        distances = np.sqrt(np.maximum(pair_squared, 0.0) / dimension)
        pair_term += float(
            np.sum(
                weight_vector[row_start:row_stop, None]
                * weight_vector[None, :]
                * distances
            )
        )
    score = first_term - 0.5 * pair_term
    if score < -1.0e-12:
        raise RuntimeError("energy score became negative beyond numerical tolerance")
    return max(float(score), 0.0)


def weighted_variogram_score(
    samples: Any,
    weights: Any,
    truth: Any,
    pairs: Any,
    pair_weights: Any,
    *,
    order: float = 0.5,
) -> float:
    """Return a dependence-sensitive weighted variogram score."""

    sample_matrix = _as_component_matrix(samples, name="samples")
    weight_vector = _validated_weights(weights, expected_count=len(sample_matrix))
    truth_vector = np.asarray(truth, dtype=float).reshape(-1)
    if sample_matrix.shape[1] != len(truth_vector):
        raise ValueError("truth dimension does not match samples")
    raw_pairs = np.asarray(pairs)
    if raw_pairs.dtype.kind not in {"i", "u"}:
        raise ValueError("pairs must contain exact integer indices")
    pair_array = np.asarray(raw_pairs, dtype=np.int64)
    if pair_array.ndim != 2 or pair_array.shape[1] != 2 or len(pair_array) == 0:
        raise ValueError("pairs must have nonempty shape (P, 2)")
    if np.any(pair_array < 0) or np.any(pair_array >= len(truth_vector)):
        raise ValueError("variogram pair index lies outside the score vector")
    if np.any(pair_array[:, 0] >= pair_array[:, 1]):
        raise ValueError("variogram pairs must be canonical with left < right")
    if len({tuple(row) for row in pair_array.tolist()}) != len(pair_array):
        raise ValueError("variogram pairs must be unique")
    weights_array = np.asarray(pair_weights, dtype=float)
    if weights_array.shape != (len(pair_array),):
        raise ValueError("pair_weights must identify every variogram pair")
    if not np.all(np.isfinite(weights_array)) or np.any(weights_array < 0.0):
        raise ValueError("pair_weights must be finite and nonnegative")
    if not np.isclose(np.sum(weights_array), 1.0, atol=1.0e-10, rtol=1.0e-10):
        raise ValueError("pair_weights must sum to one")
    order_value = float(order)
    if not np.isfinite(order_value) or not 0.0 < order_value <= 2.0:
        raise ValueError("order must lie in (0, 2]")

    left = pair_array[:, 0]
    right = pair_array[:, 1]
    observed = np.abs(truth_vector[left] - truth_vector[right]) ** order_value
    predicted = np.einsum(
        "k,kp->p",
        weight_vector,
        np.abs(sample_matrix[:, left] - sample_matrix[:, right]) ** order_value,
    )
    return float(np.dot(weights_array, np.square(observed - predicted)))


def gaussian_log_score(
    posterior_mean: Any,
    posterior_covariance: Any,
    truth: Any,
    *,
    labels: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
    covariance_floor_m2: float = 0.0,
) -> GaussianQueryScoreV1:
    """Score a registered multivariate query with its complete covariance."""

    mean = np.asarray(posterior_mean, dtype=float).reshape(-1)
    covariance = np.asarray(posterior_covariance, dtype=float)
    truth_vector = np.asarray(truth, dtype=float).reshape(-1)
    if covariance.shape != (len(mean), len(mean)) or truth_vector.shape != mean.shape:
        raise ValueError("query mean, covariance, and truth dimensions disagree")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
        raise ValueError("query moments must be finite")
    if not np.all(np.isfinite(truth_vector)):
        raise ValueError("query truth must be finite")
    covariance = 0.5 * (covariance + covariance.T)
    floor = _require_finite_nonnegative(
        covariance_floor_m2,
        name="covariance_floor_m2",
    )
    covariance = covariance + floor * np.eye(len(mean), dtype=float)
    sign, log_determinant = np.linalg.slogdet(covariance)
    if sign <= 0.0 or not np.isfinite(log_determinant):
        raise ValueError("query covariance must be positive definite after flooring")
    residual = truth_vector - mean
    try:
        solved = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError as error:
        raise ValueError("query covariance could not be solved") from error
    mahalanobis = float(residual @ solved)
    if mahalanobis < -1.0e-10:
        raise RuntimeError("query Mahalanobis distance became negative")
    mahalanobis = max(mahalanobis, 0.0)
    log_score = 0.5 * (len(mean) * np.log(2.0 * np.pi) + log_determinant + mahalanobis)
    if labels is None:
        label_tuple = tuple(f"query[{index}]" for index in range(len(mean)))
    else:
        if isinstance(labels, (str, bytes)):
            raise ValueError("labels must be a sequence of labels")
        label_tuple = tuple(labels)
    if units is None:
        unit_tuple = ("m",) * len(mean)
    else:
        if isinstance(units, (str, bytes)):
            raise ValueError("units must be a sequence of units")
        unit_tuple = tuple(units)
    return GaussianQueryScoreV1(
        labels=label_tuple,
        units=unit_tuple,
        posterior_mean=mean,
        posterior_covariance=covariance,
        truth_query=truth_vector,
        covariance_floor_m2=floor,
        mahalanobis_squared=mahalanobis,
        log_determinant=float(log_determinant),
        log_score=float(log_score),
    )


def multivariate_gaussian_query_score(
    component_means: Any,
    component_variances: Any,
    weights: Any,
    truth: Any,
    query_matrix: Any,
    *,
    labels: Sequence[str] | None = None,
    units: Sequence[str] | None = None,
    covariance_floor_m2: float = 0.0,
) -> GaussianQueryScoreV1:
    """Moment-match a weighted mixture in one registered linear query."""

    means = _as_component_matrix(component_means, name="component_means")
    variances = _as_component_matrix(
        component_variances,
        name="component_variances",
    )
    if variances.shape != means.shape:
        raise ValueError("component variances must match component means")
    if np.any(variances < 0.0):
        raise ValueError("component variances must be nonnegative")
    weight_vector = _validated_weights(weights, expected_count=len(means))
    truth_vector = np.asarray(truth, dtype=float).reshape(-1)
    if len(truth_vector) != means.shape[1] or not np.all(np.isfinite(truth_vector)):
        raise ValueError("truth must be finite and match the component dimension")
    query = np.asarray(query_matrix, dtype=float)
    if query.ndim != 2 or query.shape[1] != means.shape[1] or query.shape[0] == 0:
        raise ValueError("query_matrix must have shape (Q, D)")
    if not np.all(np.isfinite(query)):
        raise ValueError("query_matrix must be finite")

    component_queries = means @ query.T
    query_mean = np.einsum("k,kq->q", weight_vector, component_queries)
    centered = component_queries - query_mean[None]
    between = (centered.T * weight_vector) @ centered
    average_diagonal_variance = np.einsum("k,kd->d", weight_vector, variances)
    within = (query * average_diagonal_variance[None]) @ query.T
    covariance = 0.5 * (between + within + (between + within).T)
    query_truth = query @ truth_vector
    return gaussian_log_score(
        query_mean,
        covariance,
        query_truth,
        labels=labels,
        units=units,
        covariance_floor_m2=covariance_floor_m2,
    )


def _group_rows(values: Any, *, count: int, name: str) -> tuple[tuple[Any, ...], ...]:
    array = np.asarray(values)
    if array.ndim == 1:
        if len(array) != count:
            raise ValueError(f"{name} grouping does not match the component count")
        array = array[:, None]
    elif array.ndim == 2:
        if array.shape[0] != count:
            raise ValueError(f"{name} grouping does not match the component count")
    else:
        raise ValueError(f"{name} grouping must have shape (K,) or (K, P)")
    rows = []
    for row in array:
        converted = []
        for value in row.tolist():
            if isinstance(value, float):
                if not np.isfinite(value):
                    raise ValueError(f"{name} grouping must be finite")
                converted.append(float(value))
            elif isinstance(value, (int, np.integer)):
                converted.append(int(value))
            elif isinstance(value, str):
                converted.append(value)
            else:
                raise ValueError(
                    f"{name} grouping values must be finite numbers or strings"
                )
        rows.append(tuple(converted))
    return tuple(rows)


def ordered_variance_attribution(
    component_means: Any,
    component_variances: Any,
    weights: Any,
    groupings: Sequence[tuple[str, Any]],
) -> OrderedVarianceAttributionV1:
    """Apply a registered nested law-of-total-variance conditioning order."""

    means = _as_component_matrix(component_means, name="component_means")
    variances = _as_component_matrix(
        component_variances,
        name="component_variances",
    )
    if variances.shape != means.shape or np.any(variances < 0.0):
        raise ValueError("component variances must be matching and nonnegative")
    weight_vector = _validated_weights(weights, expected_count=len(means))
    if not groupings:
        raise ValueError("ordered variance attribution requires groupings")

    global_mean = np.einsum("k,kd->d", weight_vector, means)
    centered = means - global_mean[None]
    between_total = float(np.dot(weight_vector, np.mean(np.square(centered), axis=1)))
    conditional = float(np.dot(weight_vector, np.mean(variances, axis=1)))

    prefixes: list[tuple[Any, ...]] = [tuple() for _ in range(len(means))]
    previous_explained = 0.0
    raw_contributions: list[tuple[str, float]] = []
    names: set[str] = set()
    tolerance = 1.0e-11 * max(1.0, between_total)
    for name, values in groupings:
        group_name = _require_nonempty_string(name, name="grouping name")
        if group_name in names:
            raise ValueError("grouping names must be unique")
        names.add(group_name)
        rows = _group_rows(values, count=len(means), name=group_name)
        prefixes = [prefix + row for prefix, row in zip(prefixes, rows, strict=True)]
        groups: dict[tuple[Any, ...], list[int]] = {}
        for index, key in enumerate(prefixes):
            groups.setdefault(key, []).append(index)
        explained = 0.0
        for indices in groups.values():
            selected = np.asarray(indices, dtype=np.int64)
            group_weight = float(np.sum(weight_vector[selected]))
            if group_weight <= 0.0:
                continue
            group_mean = np.einsum(
                "k,kd->d",
                weight_vector[selected] / group_weight,
                means[selected],
            )
            explained += group_weight * float(
                np.mean(np.square(group_mean - global_mean))
            )
        if explained + tolerance < previous_explained:
            raise RuntimeError("nested explained variance decreased")
        explained = min(max(explained, previous_explained), between_total)
        increment = max(explained - previous_explained, 0.0)
        raw_contributions.append((group_name, increment))
        previous_explained = explained

    residual = max(between_total - previous_explained, 0.0)
    raw_contributions.append(("residual_component_support", residual))
    raw_contributions.append(("conditional_readout_discrepancy", conditional))
    total = between_total + conditional
    if total > 0.0:
        contributions = tuple(
            VarianceContributionV1(
                name=name,
                mean_coordinate_variance_m2=value,
                fraction_of_total=value / total,
            )
            for name, value in raw_contributions
        )
    else:
        contributions = tuple(
            VarianceContributionV1(
                name=name,
                mean_coordinate_variance_m2=0.0,
                fraction_of_total=0.0,
            )
            for name, _ in raw_contributions
        )
    return OrderedVarianceAttributionV1(
        contributions=contributions,
        between_component_variance_m2=between_total,
        conditional_readout_variance_m2=conditional,
        total_mean_coordinate_variance_m2=total,
    )
