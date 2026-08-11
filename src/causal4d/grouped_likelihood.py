"""Correlation-aware robust likelihoods over grouped observation evidence."""

from __future__ import annotations

from dataclasses import dataclass
from math import lgamma
from typing import Literal, Mapping

import numpy as np

from causal4d.low_rank_numerics import nonnegative_woodbury_quadratic
from causal4d.observation_evidence import (
    GroupedObservationEvidence,
    ObservationGroup,
)
from causal4d.weighting import log_weights_from_probabilities


GroupedScoreSemantics = Literal[
    "legacy_sum_v1",
    "normalized_coordinate_mean_v3",
]


@dataclass(frozen=True)
class GroupLikelihoodDiagnostics:
    """Nominal responsibilities and effective powers for one grouped update."""

    group_ids: tuple[str, ...]
    effective_group_weights: tuple[float, ...]
    nominal_responsibilities: np.ndarray
    full_covariance_group_ids: tuple[str, ...] = ()
    low_rank_covariance_group_ids: tuple[str, ...] = ()
    score_semantics: GroupedScoreSemantics = "legacy_sum_v1"
    likelihood_power: float = 1.0
    contributor_power_caps: tuple[float, ...] = ()
    group_coordinate_counts: tuple[int, ...] = ()
    normalization_coordinate_mass: float | None = None
    source_covariance_condition_numbers: tuple[float, ...] = ()
    normalization_coordinate_fractions: tuple[float, ...] = ()


def _student_t_log_density_from_terms(
    *,
    dimension: int,
    degrees_of_freedom: float,
    log_determinant: np.ndarray,
    mahalanobis: np.ndarray,
) -> np.ndarray:
    normalization = (
        lgamma(0.5 * (degrees_of_freedom + dimension))
        - lgamma(0.5 * degrees_of_freedom)
        - 0.5 * (dimension * np.log(degrees_of_freedom * np.pi) + log_determinant)
    )
    return normalization - 0.5 * (degrees_of_freedom + dimension) * np.log1p(
        mahalanobis / degrees_of_freedom
    )


def _multivariate_student_t_log_density(
    residual: np.ndarray,
    covariance_m2: np.ndarray,
    *,
    degrees_of_freedom: float,
    covariance_multiplier: float = 1.0,
) -> np.ndarray:
    """Evaluate a conventional multivariate Student-t with declared covariance."""

    values = np.asarray(residual, dtype=float)
    covariance = np.asarray(covariance_m2, dtype=float) * covariance_multiplier
    dimension = values.shape[-1]
    if covariance.shape[-2:] != (dimension, dimension):
        raise ValueError("covariance_m2 must end in (coordinate, coordinate)")
    scale = ((degrees_of_freedom - 2.0) / degrees_of_freedom) * covariance
    sign, log_determinant = np.linalg.slogdet(scale)
    if np.any(sign <= 0.0):
        raise ValueError("Student-t scale matrix must be positive definite")
    solved = np.linalg.solve(scale, values[..., None])[..., 0]
    mahalanobis = np.einsum("...i,...i->...", values, solved)
    return _student_t_log_density_from_terms(
        dimension=dimension,
        degrees_of_freedom=degrees_of_freedom,
        log_determinant=log_determinant,
        mahalanobis=mahalanobis,
    )


def _broadcast_additive_covariance(
    values: np.ndarray,
    *,
    leading_shape: tuple[int, ...],
    dimension: int,
) -> np.ndarray:
    covariance = np.asarray(values, dtype=float)
    try:
        covariance = np.broadcast_to(
            covariance,
            (*leading_shape, dimension, dimension),
        )
    except ValueError as error:
        raise ValueError(
            "additive_covariance_m2 must broadcast to component leading dimensions "
            "and end in (coordinate, coordinate)"
        ) from error
    if not np.all(np.isfinite(covariance)):
        raise ValueError("additive covariance must be finite")
    if not np.allclose(
        covariance,
        covariance.swapaxes(-1, -2),
        atol=1e-12,
        rtol=1e-10,
    ):
        raise ValueError("additive covariance must be symmetric")
    if float(np.min(np.linalg.eigvalsh(covariance), initial=0.0)) < -1e-10:
        raise ValueError("additive covariance must be positive semidefinite")
    return covariance


def _broadcast_additive_covariance_factor(
    values: np.ndarray,
    *,
    leading_shape: tuple[int, ...],
    dimension: int,
) -> np.ndarray:
    factor = np.asarray(values, dtype=float)
    if factor.ndim < 2 or factor.shape[-2] != dimension or factor.shape[-1] < 1:
        raise ValueError(
            "additive_covariance_factor_m must end in (coordinate, positive_rank)"
        )
    rank = factor.shape[-1]
    try:
        factor = np.broadcast_to(
            factor,
            (*leading_shape, dimension, rank),
        )
    except ValueError as error:
        raise ValueError(
            "additive_covariance_factor_m must broadcast to component leading "
            "dimensions and end in (coordinate, rank)"
        ) from error
    if not np.all(np.isfinite(factor)):
        raise ValueError("additive covariance factor must be finite")
    return factor


def _multivariate_student_t_log_density_low_rank(
    residual: np.ndarray,
    base_covariance_m2: np.ndarray,
    covariance_factor_m: np.ndarray,
    *,
    degrees_of_freedom: float,
    covariance_multiplier: float = 1.0,
) -> np.ndarray:
    """Evaluate Student-t density for ``base + factor @ factor.T``.

    The implementation uses Cholesky whitening plus the matrix determinant lemma
    and Woodbury identity. It never materializes the low-rank covariance update.
    """

    values = np.asarray(residual, dtype=float)
    dimension = values.shape[-1]
    leading_shape = values.shape[:-1]
    base = np.asarray(base_covariance_m2, dtype=float)
    if base.shape[-2:] != (dimension, dimension):
        raise ValueError("base_covariance_m2 must end in (coordinate, coordinate)")
    try:
        base = np.broadcast_to(base, (*leading_shape, dimension, dimension))
    except ValueError as error:
        raise ValueError(
            "base_covariance_m2 must broadcast to the residual leading dimensions"
        ) from error
    if not np.all(np.isfinite(base)):
        raise ValueError("base covariance must be finite")
    factor = _broadcast_additive_covariance_factor(
        covariance_factor_m,
        leading_shape=leading_shape,
        dimension=dimension,
    )
    try:
        base_cholesky = np.linalg.cholesky(base)
    except np.linalg.LinAlgError as error:
        raise ValueError("base covariance must be positive definite") from error

    whitened_residual = np.linalg.solve(
        base_cholesky,
        values[..., None],
    )[..., 0]
    whitened_factor = np.linalg.solve(base_cholesky, factor)
    rank = factor.shape[-1]
    low_rank_system = np.eye(rank) + np.einsum(
        "...ir,...is->...rs",
        whitened_factor,
        whitened_factor,
    )
    try:
        low_rank_cholesky = np.linalg.cholesky(low_rank_system)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "low-rank covariance system must be positive definite"
        ) from error

    projected_residual = np.einsum(
        "...ir,...i->...r",
        whitened_factor,
        whitened_residual,
    )
    whitened_projection = np.linalg.solve(
        low_rank_cholesky,
        projected_residual[..., None],
    )[..., 0]
    base_quadratic = np.einsum(
        "...i,...i->...",
        whitened_residual,
        whitened_residual,
    )
    correction_quadratic = np.einsum(
        "...r,...r->...",
        whitened_projection,
        whitened_projection,
    )
    covariance_quadratic = nonnegative_woodbury_quadratic(
        base_quadratic,
        correction_quadratic,
        dimension=dimension,
        name="grouped Student-t Woodbury quadratic",
    )
    base_log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(base_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    low_rank_log_determinant = 2.0 * np.sum(
        np.log(np.diagonal(low_rank_cholesky, axis1=-2, axis2=-1)),
        axis=-1,
    )
    scale_multiplier = (
        (degrees_of_freedom - 2.0) / degrees_of_freedom * covariance_multiplier
    )
    if not np.isfinite(scale_multiplier) or scale_multiplier <= 0.0:
        raise ValueError("Student-t covariance multiplier must be positive")
    log_determinant = (
        base_log_determinant
        + low_rank_log_determinant
        + dimension * np.log(scale_multiplier)
    )
    mahalanobis = covariance_quadratic / scale_multiplier
    return _student_t_log_density_from_terms(
        dimension=dimension,
        degrees_of_freedom=degrees_of_freedom,
        log_determinant=log_determinant,
        mahalanobis=mahalanobis,
    )


def group_log_likelihood(
    predicted_values_m: np.ndarray,
    group: ObservationGroup,
    *,
    additive_variance_m2: np.ndarray | None = None,
    additive_covariance_m2: np.ndarray | None = None,
    additive_covariance_factor_m: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return robust mixture log likelihood and posterior nominal responsibility.

    ``additive_covariance_m2`` preserves component-specific dense correlation.
    ``additive_covariance_factor_m`` represents a positive-semidefinite update as
    ``factor @ factor.T`` and is evaluated without forming that dense matrix. The
    legacy ``additive_variance_m2`` argument remains a diagonal convenience; all
    three forms can be combined without double counting when they represent
    distinct uncertainty sources.
    """

    predictions = np.asarray(predicted_values_m, dtype=float)
    if predictions.shape[-1] != group.coordinate_count:
        raise ValueError(
            "predicted group coordinates do not match the observation group"
        )
    residual = predictions - group.values_m
    covariance = group.covariance_m2
    if additive_variance_m2 is not None:
        additive = np.asarray(additive_variance_m2, dtype=float)
        if additive.shape != predictions.shape:
            raise ValueError("additive_variance_m2 must match predicted group values")
        if np.any(~np.isfinite(additive)) or np.any(additive < 0.0):
            raise ValueError("additive variances must be finite and nonnegative")
        covariance = covariance + additive[..., :, None] * np.eye(
            group.coordinate_count
        )
    if additive_covariance_m2 is not None:
        covariance = covariance + _broadcast_additive_covariance(
            additive_covariance_m2,
            leading_shape=predictions.shape[:-1],
            dimension=group.coordinate_count,
        )
    if additive_covariance_factor_m is None:
        nominal = _multivariate_student_t_log_density(
            residual,
            covariance,
            degrees_of_freedom=group.degrees_of_freedom,
        )
        outlier = _multivariate_student_t_log_density(
            residual,
            covariance,
            degrees_of_freedom=group.degrees_of_freedom,
            covariance_multiplier=group.outlier_scale_multiplier,
        )
    else:
        factor = _broadcast_additive_covariance_factor(
            additive_covariance_factor_m,
            leading_shape=predictions.shape[:-1],
            dimension=group.coordinate_count,
        )
        nominal = _multivariate_student_t_log_density_low_rank(
            residual,
            covariance,
            factor,
            degrees_of_freedom=group.degrees_of_freedom,
        )
        outlier = _multivariate_student_t_log_density_low_rank(
            residual,
            covariance,
            factor,
            degrees_of_freedom=group.degrees_of_freedom,
            covariance_multiplier=group.outlier_scale_multiplier,
        )
    log_nominal_component = np.log(group.prior_nominal_probability) + nominal
    log_outlier_component = np.log1p(-group.prior_nominal_probability) + outlier
    log_mixture = np.logaddexp(log_nominal_component, log_outlier_component)
    responsibility = np.exp(log_nominal_component - log_mixture)
    return log_mixture, responsibility


def grouped_component_log_likelihoods(
    predicted_components_m: np.ndarray,
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
    component_variance_m2: np.ndarray | None = None,
    component_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
    component_group_covariance_factor_m: Mapping[str, np.ndarray] | None = None,
    score_semantics: GroupedScoreSemantics = "legacy_sum_v1",
    likelihood_power: float = 1.0,
    max_source_covariance_condition_number: float = 1.0e12,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics]:
    """Score arbitrary leading component dimensions against grouped O-plus evidence.

    ``component_group_covariance_m2`` maps group IDs to dense covariance updates.
    ``component_group_covariance_factor_m`` maps group IDs to low-rank factors in
    meters. Each factor must broadcast to
    ``component_shape + (group_coordinate, rank)`` and contributes
    ``factor @ factor.T`` without dense materialization.

    ``legacy_sum_v1`` preserves the original robust composite likelihood. The
    experimental ``normalized_coordinate_mean_v3`` divides the contributor-capped
    score by contributor-capped coordinate mass and then applies the explicit
    likelihood power. Source ``composite_weight`` values remain multiplicative
    reliability temperatures and therefore do not cancel in the normalization.
    """

    if score_semantics not in {
        "legacy_sum_v1",
        "normalized_coordinate_mean_v3",
    }:
        raise ValueError("unsupported grouped score semantics")
    if not np.isfinite(likelihood_power) or likelihood_power <= 0.0:
        raise ValueError("likelihood_power must be finite and positive")
    if (
        not np.isfinite(max_source_covariance_condition_number)
        or max_source_covariance_condition_number < 1.0
    ):
        raise ValueError(
            "max_source_covariance_condition_number must be finite and at least one"
        )

    components = np.asarray(predicted_components_m, dtype=float)
    if components.ndim < 4:
        raise ValueError("predicted_components_m must end in (frame, node, coordinate)")
    if not np.all(np.isfinite(components)):
        raise ValueError("predicted components must be finite")
    evidence.validate_prefix(
        prefix_frame_count=prefix_frame_count,
        rollout_shape=components.shape[-3:],
    )
    leading_shape = components.shape[:-3]
    variance = None
    if component_variance_m2 is not None:
        variance = np.broadcast_to(
            np.asarray(component_variance_m2, dtype=float), components.shape
        )
        if np.any(~np.isfinite(variance)) or np.any(variance < 0.0):
            raise ValueError("component variances must be finite and nonnegative")
    covariance_by_group = dict(component_group_covariance_m2 or {})
    factor_by_group = dict(component_group_covariance_factor_m or {})
    known_group_ids = {group.group_id for group in evidence.groups}
    unknown_covariance = set(covariance_by_group) - known_group_ids
    if unknown_covariance:
        raise ValueError(
            "component covariance references unknown groups: "
            f"{sorted(unknown_covariance)}"
        )
    unknown_factor = set(factor_by_group) - known_group_ids
    if unknown_factor:
        raise ValueError(
            "component covariance factor references unknown groups: "
            f"{sorted(unknown_factor)}"
        )
    total = np.zeros(leading_shape, dtype=float)
    responsibilities = []
    contributor_caps = evidence.contributor_power_caps
    effective_weights = evidence.effective_group_weights
    coordinate_counts = tuple(group.coordinate_count for group in evidence.groups)
    condition_numbers = []
    full_covariance_groups = []
    low_rank_covariance_groups = []
    for group, weight in zip(evidence.groups, effective_weights, strict=True):
        if score_semantics == "normalized_coordinate_mean_v3":
            eigenvalues = np.linalg.eigvalsh(group.covariance_m2)
            condition_number = float(eigenvalues[-1] / eigenvalues[0])
            condition_numbers.append(condition_number)
            if condition_number > max_source_covariance_condition_number:
                raise ValueError(
                    f"group {group.group_id!r} source covariance condition number "
                    "exceeds the normalized-v3 limit"
                )
        selected = group.selected_predictions(components)
        selected_variance = (
            None if variance is None else group.selected_predictions(variance)
        )
        selected_covariance = covariance_by_group.get(group.group_id)
        selected_factor = factor_by_group.get(group.group_id)
        if selected_covariance is not None:
            full_covariance_groups.append(group.group_id)
        if selected_factor is not None:
            low_rank_covariance_groups.append(group.group_id)
        log_likelihood, responsibility = group_log_likelihood(
            selected,
            group,
            additive_variance_m2=selected_variance,
            additive_covariance_m2=selected_covariance,
            additive_covariance_factor_m=selected_factor,
        )
        total += weight * log_likelihood
        responsibilities.append(responsibility)

    normalization_mass: float | None = None
    normalization_fractions: tuple[float, ...] = ()
    if score_semantics == "normalized_coordinate_mean_v3":
        normalization_mass = float(
            sum(
                cap * coordinate_count
                for cap, coordinate_count in zip(
                    contributor_caps,
                    coordinate_counts,
                    strict=True,
                )
            )
        )
        if not np.isfinite(normalization_mass) or normalization_mass <= 0.0:
            raise RuntimeError("normalized grouped coordinate mass is invalid")
        total = likelihood_power * total / normalization_mass
        normalization_fractions = tuple(
            cap * coordinate_count / normalization_mass
            for cap, coordinate_count in zip(
                contributor_caps,
                coordinate_counts,
                strict=True,
            )
        )
    else:
        total = likelihood_power * total

    diagnostics = GroupLikelihoodDiagnostics(
        group_ids=tuple(group.group_id for group in evidence.groups),
        effective_group_weights=effective_weights,
        nominal_responsibilities=np.stack(responsibilities, axis=-1),
        full_covariance_group_ids=tuple(full_covariance_groups),
        low_rank_covariance_group_ids=tuple(low_rank_covariance_groups),
        score_semantics=score_semantics,
        likelihood_power=float(likelihood_power),
        contributor_power_caps=contributor_caps,
        group_coordinate_counts=coordinate_counts,
        normalization_coordinate_mass=normalization_mass,
        source_covariance_condition_numbers=tuple(condition_numbers),
        normalization_coordinate_fractions=normalization_fractions,
    )
    return total, diagnostics


def posterior_weights_from_grouped_evidence(
    prior_weights: np.ndarray,
    predicted_components_m: np.ndarray,
    evidence: GroupedObservationEvidence,
    *,
    prefix_frame_count: int,
    component_variance_m2: np.ndarray | None = None,
    component_group_covariance_m2: Mapping[str, np.ndarray] | None = None,
    component_group_covariance_factor_m: Mapping[str, np.ndarray] | None = None,
    score_semantics: GroupedScoreSemantics = "legacy_sum_v1",
    likelihood_power: float = 1.0,
    max_source_covariance_condition_number: float = 1.0e12,
) -> tuple[np.ndarray, GroupLikelihoodDiagnostics]:
    """Apply grouped evidence to finite component support in log space."""

    prior = np.asarray(prior_weights, dtype=float)
    if prior.shape != np.asarray(predicted_components_m).shape[:-3]:
        raise ValueError("prior_weights must match the component leading dimensions")
    if np.any(prior < 0.0) or not np.isclose(np.sum(prior), 1.0):
        raise ValueError("prior_weights must be nonnegative and sum to one")
    score, diagnostics = grouped_component_log_likelihoods(
        predicted_components_m,
        evidence,
        prefix_frame_count=prefix_frame_count,
        component_variance_m2=component_variance_m2,
        component_group_covariance_m2=component_group_covariance_m2,
        component_group_covariance_factor_m=(component_group_covariance_factor_m),
        score_semantics=score_semantics,
        likelihood_power=likelihood_power,
        max_source_covariance_condition_number=(
            max_source_covariance_condition_number
        ),
    )
    log_posterior = log_weights_from_probabilities(prior, name="prior_weights") + score
    maximum = float(np.max(log_posterior))
    posterior = np.exp(log_posterior - maximum)
    posterior /= np.sum(posterior)
    return posterior, diagnostics
