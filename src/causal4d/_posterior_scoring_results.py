"""Immutable result artifacts for dependence-aware posterior scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from causal4d._posterior_scoring_contracts import (
    POSTERIOR_SCORE_CLAIM_BOUNDARY,
    POSTERIOR_SCORE_SCHEMA_VERSION,
    _canonical_id,
    _readonly_array,
    _require_finite_nonnegative,
    _require_nonempty_string,
    _require_sha256,
)


@dataclass(frozen=True)
class VarianceContributionV1:
    """One nonnegative contribution in an ordered total-variance expansion."""

    name: str
    mean_coordinate_variance_m2: float
    fraction_of_total: float

    def __post_init__(self) -> None:
        name = _require_nonempty_string(self.name, name="contribution name")
        variance = _require_finite_nonnegative(
            self.mean_coordinate_variance_m2,
            name="mean_coordinate_variance_m2",
        )
        fraction = float(self.fraction_of_total)
        if not np.isfinite(fraction) or not -1.0e-12 <= fraction <= 1.0 + 1.0e-12:
            raise ValueError("fraction_of_total must lie in [0, 1]")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "mean_coordinate_variance_m2", variance)
        object.__setattr__(self, "fraction_of_total", min(max(fraction, 0.0), 1.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mean_coordinate_variance_m2": self.mean_coordinate_variance_m2,
            "fraction_of_total": self.fraction_of_total,
        }


@dataclass(frozen=True)
class OrderedVarianceAttributionV1:
    """Order-explicit nested law-of-total-variance attribution."""

    contributions: tuple[VarianceContributionV1, ...]
    between_component_variance_m2: float
    conditional_readout_variance_m2: float
    total_mean_coordinate_variance_m2: float
    order_dependent: bool = True
    warning: str = (
        "Nested variance increments are nonnegative but depend on the registered "
        "conditioning order; they are not causal effects of uncertainty sources."
    )

    def __post_init__(self) -> None:
        contributions = tuple(self.contributions)
        if not contributions:
            raise ValueError("variance attribution requires contributions")
        if len({value.name for value in contributions}) != len(contributions):
            raise ValueError("variance contribution names must be unique")
        between = _require_finite_nonnegative(
            self.between_component_variance_m2,
            name="between_component_variance_m2",
        )
        conditional = _require_finite_nonnegative(
            self.conditional_readout_variance_m2,
            name="conditional_readout_variance_m2",
        )
        total = _require_finite_nonnegative(
            self.total_mean_coordinate_variance_m2,
            name="total_mean_coordinate_variance_m2",
        )
        if type(self.order_dependent) is not bool or not self.order_dependent:
            raise ValueError("ordered attribution must declare order dependence")
        warning = _require_nonempty_string(self.warning, name="warning")
        contribution_sum = sum(
            value.mean_coordinate_variance_m2 for value in contributions
        )
        tolerance = 1.0e-10 * max(1.0, total)
        if abs(contribution_sum - total) > tolerance:
            raise ValueError("variance contributions do not sum to total variance")
        if abs((between + conditional) - total) > tolerance:
            raise ValueError("between and conditional variance do not sum to total")
        object.__setattr__(self, "contributions", contributions)
        object.__setattr__(self, "between_component_variance_m2", between)
        object.__setattr__(self, "conditional_readout_variance_m2", conditional)
        object.__setattr__(self, "total_mean_coordinate_variance_m2", total)
        object.__setattr__(self, "warning", warning)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POSTERIOR_SCORE_SCHEMA_VERSION,
            "artifact_kind": "OrderedVarianceAttribution",
            "contributions": [value.as_dict() for value in self.contributions],
            "between_component_variance_m2": self.between_component_variance_m2,
            "conditional_readout_variance_m2": (self.conditional_readout_variance_m2),
            "total_mean_coordinate_variance_m2": (
                self.total_mean_coordinate_variance_m2
            ),
            "order_dependent": self.order_dependent,
            "warning": self.warning,
        }

    @property
    def attribution_id(self) -> str:
        return _canonical_id(self.as_dict())


@dataclass(frozen=True)
class GaussianQueryScoreV1:
    """Multivariate Gaussian log score with the complete query covariance."""

    labels: tuple[str, ...]
    units: tuple[str, ...]
    posterior_mean: np.ndarray
    posterior_covariance: np.ndarray
    truth_query: np.ndarray
    covariance_floor_m2: float
    mahalanobis_squared: float
    log_determinant: float
    log_score: float

    def __post_init__(self) -> None:
        labels = tuple(self.labels)
        units = tuple(self.units)
        mean = _readonly_array(self.posterior_mean, dtype=float)
        covariance = _readonly_array(self.posterior_covariance, dtype=float)
        truth = _readonly_array(self.truth_query, dtype=float)
        if mean.ndim != 1 or truth.shape != mean.shape:
            raise ValueError("query mean and truth must be matching vectors")
        if covariance.shape != (len(mean), len(mean)):
            raise ValueError("query covariance must be square")
        if len(labels) != len(mean) or len(set(labels)) != len(labels):
            raise ValueError("query labels must uniquely identify every row")
        if len(units) != len(mean):
            raise ValueError("query units must identify every row")
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(covariance)):
            raise ValueError("query moments must be finite")
        if not np.all(np.isfinite(truth)):
            raise ValueError("query truth must be finite")
        if not np.allclose(covariance, covariance.T, atol=1.0e-12, rtol=1.0e-12):
            raise ValueError("query covariance must be symmetric")
        for index, label in enumerate(labels):
            _require_nonempty_string(label, name=f"labels[{index}]")
        for index, unit in enumerate(units):
            if _require_nonempty_string(unit, name=f"units[{index}]") != "m":
                raise ValueError("Gaussian query score V1 requires metre outputs")
        floor = _require_finite_nonnegative(
            self.covariance_floor_m2,
            name="covariance_floor_m2",
        )
        sign, expected_log_determinant = np.linalg.slogdet(covariance)
        if sign <= 0.0 or not np.isfinite(expected_log_determinant):
            raise ValueError("query covariance must be positive definite")
        residual = truth - mean
        try:
            solved = np.linalg.solve(covariance, residual)
        except np.linalg.LinAlgError as error:
            raise ValueError("query covariance could not be solved") from error
        expected_mahalanobis = max(float(residual @ solved), 0.0)
        expected_log_score = 0.5 * (
            len(mean) * np.log(2.0 * np.pi)
            + expected_log_determinant
            + expected_mahalanobis
        )
        supplied = (
            self.mahalanobis_squared,
            self.log_determinant,
            self.log_score,
        )
        if not all(np.isfinite(value) for value in supplied):
            raise ValueError("query score statistics must be finite")
        if self.mahalanobis_squared < 0.0:
            raise ValueError("mahalanobis_squared must be nonnegative")
        expected = (
            expected_mahalanobis,
            float(expected_log_determinant),
            float(expected_log_score),
        )
        if not all(
            np.isclose(actual, target, atol=1.0e-10, rtol=1.0e-10)
            for actual, target in zip(supplied, expected, strict=True)
        ):
            raise ValueError("query score statistics disagree with stored moments")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "units", units)
        object.__setattr__(self, "posterior_mean", mean)
        object.__setattr__(self, "posterior_covariance", covariance)
        object.__setattr__(self, "truth_query", truth)
        object.__setattr__(self, "covariance_floor_m2", floor)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": POSTERIOR_SCORE_SCHEMA_VERSION,
            "artifact_kind": "GaussianQueryScore",
            "labels": list(self.labels),
            "units": list(self.units),
            "posterior_mean": self.posterior_mean.tolist(),
            "posterior_covariance": self.posterior_covariance.tolist(),
            "truth_query": self.truth_query.tolist(),
            "covariance_floor_m2": self.covariance_floor_m2,
            "mahalanobis_squared": self.mahalanobis_squared,
            "log_determinant": self.log_determinant,
            "log_score": self.log_score,
        }

    @property
    def query_score_id(self) -> str:
        return _canonical_id(self.as_dict())


@dataclass(frozen=True)
class TrajectoryPosteriorScoreV1:
    """Content-addressed joint score of one frozen trajectory posterior."""

    source_posterior_id: str
    source_weight_kind: str
    source_weight_id: str
    specification_id: str
    truth_sha256: str
    exact_component_energy_score_m: float
    sampled_mixture_energy_score_m: float
    exact_component_variogram_score: float | None
    sampled_mixture_variogram_score: float | None
    variogram_order: float
    component_count: int
    selected_coordinate_count: int
    variogram_pair_count: int
    conditional_draws_per_component: int
    random_seed: int
    effective_component_count: float
    variance_attribution: OrderedVarianceAttributionV1
    query_score: GaussianQueryScoreV1 | None
    claim_boundary: str = POSTERIOR_SCORE_CLAIM_BOUNDARY

    def __post_init__(self) -> None:
        for name in (
            "source_posterior_id",
            "source_weight_id",
            "specification_id",
            "truth_sha256",
        ):
            _require_sha256(getattr(self, name), name=name)
        if self.source_weight_kind not in {"physical", "task"}:
            raise ValueError("source_weight_kind must be 'physical' or 'task'")
        for value, name in (
            (self.exact_component_energy_score_m, "exact_component_energy_score_m"),
            (self.sampled_mixture_energy_score_m, "sampled_mixture_energy_score_m"),
        ):
            _require_finite_nonnegative(value, name=name)
        for value, name in (
            (self.exact_component_variogram_score, "exact component variogram"),
            (self.sampled_mixture_variogram_score, "sampled mixture variogram"),
        ):
            if value is not None:
                _require_finite_nonnegative(value, name=name)
        if (self.exact_component_variogram_score is None) != (
            self.sampled_mixture_variogram_score is None
        ):
            raise ValueError("component and mixture variogram scores must agree on use")
        if (
            not np.isfinite(self.variogram_order)
            or not 0.0 < self.variogram_order <= 2.0
        ):
            raise ValueError("variogram_order must lie in (0, 2]")
        for value, name in (
            (self.component_count, "component_count"),
            (self.selected_coordinate_count, "selected_coordinate_count"),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.variogram_pair_count) is not int or self.variogram_pair_count < 0:
            raise ValueError("variogram_pair_count must be a nonnegative integer")
        if (self.variogram_pair_count == 0) != (
            self.exact_component_variogram_score is None
        ):
            raise ValueError("variogram_pair_count disagrees with variogram scores")
        if type(self.conditional_draws_per_component) is not int:
            raise ValueError("conditional_draws_per_component must be an integer")
        if self.conditional_draws_per_component < 1:
            raise ValueError("conditional_draws_per_component must be positive")
        if type(self.random_seed) is not int or self.random_seed < 0:
            raise ValueError("random_seed must be a nonnegative integer")
        if not np.isfinite(self.effective_component_count) or not (
            1.0 - 1.0e-12
            <= self.effective_component_count
            <= self.component_count + 1.0e-12
        ):
            raise ValueError(
                "effective_component_count must lie between one and component_count"
            )
        if not isinstance(self.variance_attribution, OrderedVarianceAttributionV1):
            raise TypeError("variance_attribution has the wrong type")
        if self.query_score is not None and not isinstance(
            self.query_score,
            GaussianQueryScoreV1,
        ):
            raise TypeError("query_score has the wrong type")
        _require_nonempty_string(self.claim_boundary, name="claim_boundary")

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "schema_version": POSTERIOR_SCORE_SCHEMA_VERSION,
            "artifact_kind": "TrajectoryPosteriorScore",
            "source_posterior_id": self.source_posterior_id,
            "source_weight_kind": self.source_weight_kind,
            "source_weight_id": self.source_weight_id,
            "specification_id": self.specification_id,
            "truth_sha256": self.truth_sha256,
            "exact_component_energy_score_m": self.exact_component_energy_score_m,
            "sampled_mixture_energy_score_m": self.sampled_mixture_energy_score_m,
            "exact_component_variogram_score": (self.exact_component_variogram_score),
            "sampled_mixture_variogram_score": (self.sampled_mixture_variogram_score),
            "variogram_order": self.variogram_order,
            "variogram_score_unit_power_m": 2.0 * self.variogram_order,
            "component_count": self.component_count,
            "selected_coordinate_count": self.selected_coordinate_count,
            "variogram_pair_count": self.variogram_pair_count,
            "conditional_draws_per_component": (self.conditional_draws_per_component),
            "random_seed": self.random_seed,
            "effective_component_count": self.effective_component_count,
            "variance_attribution": self.variance_attribution.as_dict(),
            "variance_attribution_id": (self.variance_attribution.attribution_id),
            "query_score": (
                None if self.query_score is None else self.query_score.as_dict()
            ),
            "query_score_id": (
                None if self.query_score is None else self.query_score.query_score_id
            ),
            "claim_boundary": self.claim_boundary,
        }
        return payload

    @property
    def score_id(self) -> str:
        return _canonical_id(self.as_dict())
