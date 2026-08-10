"""Intervention-versus-nuisance identifiability diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from causal4d.immutable_array import readonly_array


@dataclass(frozen=True)
class IdentifiabilityConfig:
    """Frozen thresholds for conditional intervention information."""

    relative_rank_tolerance: float = 1e-6
    minimum_information_eigenvalue: float = 1e-6
    maximum_condition_number: float = 1e8
    minimum_residualized_response_fraction: float = 0.10
    maximum_subspace_cosine: float = 0.995
    maximum_query_null_response_fraction: float = 0.10

    def __post_init__(self) -> None:
        if not 0.0 < self.relative_rank_tolerance < 1.0:
            raise ValueError("relative_rank_tolerance must lie in (0, 1)")
        if self.minimum_information_eigenvalue <= 0.0:
            raise ValueError("minimum_information_eigenvalue must be positive")
        if self.maximum_condition_number <= 1.0:
            raise ValueError("maximum_condition_number must exceed one")
        if not 0.0 <= self.minimum_residualized_response_fraction <= 1.0:
            raise ValueError(
                "minimum_residualized_response_fraction must lie in [0, 1]"
            )
        if not 0.0 <= self.maximum_subspace_cosine <= 1.0:
            raise ValueError("maximum_subspace_cosine must lie in [0, 1]")
        if not 0.0 <= self.maximum_query_null_response_fraction <= 1.0:
            raise ValueError("maximum_query_null_response_fraction must lie in [0, 1]")


@dataclass(frozen=True)
class InterventionIdentifiabilityResult:
    """Conditional information remaining after projecting out nuisance response.

    ``identified_basis`` and ``null_basis`` are expressed in standardized
    intervention coordinates. ``parameter_scales`` maps those coordinates back
    to the physical parameter units supplied to the sensitivity calculation.
    """

    conditional_information: np.ndarray
    eigenvalues: np.ndarray
    effective_rank: int
    parameter_count: int
    minimum_eigenvalue: float
    condition_number: float
    residualized_response_fraction: float
    maximum_subspace_cosine: float
    parameter_scales: np.ndarray
    identified_basis: np.ndarray
    null_basis: np.ndarray
    query_null_response_fraction: float | None
    query_identifiable: bool | None
    identifiable: bool
    failure_reasons: tuple[str, ...]
    query_failure_reasons: tuple[str, ...] = ()
    extended_diagnostics: bool = False

    def __post_init__(self) -> None:
        information = np.asarray(self.conditional_information, dtype=float).copy()
        eigenvalues = np.asarray(self.eigenvalues, dtype=float).copy()
        scales = np.asarray(self.parameter_scales, dtype=float).copy()
        identified = np.asarray(self.identified_basis, dtype=float).copy()
        null = np.asarray(self.null_basis, dtype=float).copy()
        if information.shape != (self.parameter_count, self.parameter_count):
            raise ValueError("conditional_information must match parameter_count")
        if eigenvalues.shape != (self.parameter_count,):
            raise ValueError("eigenvalues must match parameter_count")
        if scales.shape != (self.parameter_count,):
            raise ValueError("parameter_scales must match parameter_count")
        if identified.shape != (self.parameter_count, self.effective_rank):
            raise ValueError("identified_basis must match effective_rank")
        if null.shape != (
            self.parameter_count,
            self.parameter_count - self.effective_rank,
        ):
            raise ValueError("null_basis must span the unresolved complement")
        if not all(
            np.all(np.isfinite(value))
            for value in (information, eigenvalues, scales, identified, null)
        ):
            raise ValueError("identifiability arrays must be finite")
        if np.any(scales <= 0.0):
            raise ValueError("parameter_scales must be positive")
        for basis, name in ((identified, "identified_basis"), (null, "null_basis")):
            if basis.shape[1] and not np.allclose(
                basis.T @ basis,
                np.eye(basis.shape[1]),
                atol=1e-10,
                rtol=1e-10,
            ):
                raise ValueError(f"{name} must have orthonormal columns")
        if (
            identified.shape[1]
            and null.shape[1]
            and not np.allclose(
                identified.T @ null,
                0.0,
                atol=1e-10,
                rtol=1e-10,
            )
        ):
            raise ValueError("identified and null bases must be orthogonal")
        if self.query_null_response_fraction is not None and not (
            np.isfinite(self.query_null_response_fraction)
            and 0.0 <= self.query_null_response_fraction <= 1.0 + 1e-12
        ):
            raise ValueError("query_null_response_fraction must lie in [0, 1]")
        information = readonly_array(information)
        eigenvalues = readonly_array(eigenvalues)
        scales = readonly_array(scales)
        identified = readonly_array(identified)
        null = readonly_array(null)
        object.__setattr__(self, "conditional_information", information)
        object.__setattr__(self, "eigenvalues", eigenvalues)
        object.__setattr__(self, "parameter_scales", scales)
        object.__setattr__(self, "identified_basis", identified)
        object.__setattr__(self, "null_basis", null)
        object.__setattr__(self, "failure_reasons", tuple(self.failure_reasons))
        object.__setattr__(
            self,
            "query_failure_reasons",
            tuple(self.query_failure_reasons),
        )

    @property
    def identified_projection(self) -> np.ndarray:
        """Projection onto identified standardized intervention directions."""

        return self.identified_basis @ self.identified_basis.T

    def as_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "effective_rank": self.effective_rank,
            "parameter_count": self.parameter_count,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "condition_number": (
                self.condition_number if np.isfinite(self.condition_number) else None
            ),
            "residualized_response_fraction": self.residualized_response_fraction,
            "maximum_subspace_cosine": self.maximum_subspace_cosine,
            "identifiable": self.identifiable,
            "failure_reasons": list(self.failure_reasons),
            "eigenvalues": self.eigenvalues.tolist(),
        }
        if self.extended_diagnostics:
            result.update(
                {
                    "parameter_scales": self.parameter_scales.tolist(),
                    "identified_basis": self.identified_basis.tolist(),
                    "null_basis": self.null_basis.tolist(),
                    "query_null_response_fraction": self.query_null_response_fraction,
                    "query_identifiable": self.query_identifiable,
                    "query_failure_reasons": list(self.query_failure_reasons),
                }
            )
        return result


def finite_response_sensitivity(
    reference_response: np.ndarray,
    perturbed_responses: np.ndarray,
    perturbation_steps: Sequence[float],
    *,
    valid: np.ndarray | None = None,
) -> np.ndarray:
    """Build a flattened secant-sensitivity matrix from finite perturbations.

    ``perturbed_responses`` has shape ``(parameter, ...)`` and each remaining
    dimension must match ``reference_response``. A boolean ``valid`` mask may
    select response coordinates before flattening.
    """

    reference = np.asarray(reference_response, dtype=float)
    perturbed = np.asarray(perturbed_responses, dtype=float)
    steps = np.asarray(tuple(perturbation_steps), dtype=float)
    if perturbed.ndim != reference.ndim + 1 or perturbed.shape[1:] != reference.shape:
        raise ValueError(
            "perturbed_responses must have shape (parameter, *reference.shape)"
        )
    if (
        steps.shape != (len(perturbed),)
        or np.any(~np.isfinite(steps))
        or np.any(steps == 0.0)
    ):
        raise ValueError(
            "perturbation_steps must be finite, nonzero, and match parameters"
        )
    responses = (perturbed - reference[None]) / steps.reshape(
        (-1,) + (1,) * reference.ndim
    )
    if valid is None:
        selected = np.ones(reference.shape, dtype=bool)
    else:
        selected = np.asarray(valid, dtype=bool)
        if selected.shape != reference.shape:
            raise ValueError("valid must match reference_response")
    if not np.any(selected):
        raise ValueError("finite-response sensitivity has no valid coordinates")
    matrix = responses[:, selected].T
    if not np.all(np.isfinite(matrix)):
        raise ValueError("finite-response sensitivities must be finite")
    return matrix


def _whiten_sensitivities(
    matrices: tuple[np.ndarray, ...],
    covariance: np.ndarray | None,
    covariance_factor: np.ndarray | None,
) -> tuple[np.ndarray, ...]:
    """Whiten aligned sensitivities under ``B + U U.T`` response covariance."""

    if not matrices:
        raise ValueError("at least one sensitivity matrix is required")
    values = tuple(np.asarray(matrix, dtype=float) for matrix in matrices)
    response_count = values[0].shape[0] if values[0].ndim == 2 else 0
    for matrix in values:
        if matrix.ndim != 2 or matrix.shape[0] != response_count:
            raise ValueError("sensitivity matrices must share a nonempty row dimension")
        if not np.all(np.isfinite(matrix)):
            raise ValueError("sensitivity matrices must be finite")
    if response_count < 1 or values[0].shape[1] < 1:
        raise ValueError("the primary sensitivity matrix must be nonempty")

    shared_factor: np.ndarray | None = None
    if covariance_factor is not None:
        if covariance is None:
            raise ValueError("covariance_factor requires a base covariance")
        shared_factor = np.asarray(covariance_factor, dtype=float)
        if (
            shared_factor.ndim != 2
            or shared_factor.shape[0] != response_count
            or shared_factor.shape[1] < 1
        ):
            raise ValueError(
                "covariance_factor must have shape (response, positive rank)"
            )
        if not np.all(np.isfinite(shared_factor)):
            raise ValueError("covariance_factor must be finite")

    if covariance is None:
        return values

    parts = list(values)
    if shared_factor is not None:
        parts.append(shared_factor)
    widths = [part.shape[1] for part in parts]
    combined = np.column_stack(parts)
    noise = np.asarray(covariance, dtype=float)
    if noise.ndim == 1:
        if noise.shape != (response_count,):
            raise ValueError("diagonal covariance must match the response dimension")
        if not np.all(np.isfinite(noise)) or np.any(noise <= 0.0):
            raise ValueError("diagonal covariance must be finite and positive")
        whitened = combined / np.sqrt(noise)[:, None]
    elif noise.ndim == 2:
        if noise.shape != (response_count, response_count):
            raise ValueError("covariance must match the response dimension")
        if not np.all(np.isfinite(noise)) or not np.allclose(
            noise,
            noise.T,
            atol=1e-12,
        ):
            raise ValueError("covariance must be finite and symmetric")
        try:
            base_factor = np.linalg.cholesky(noise)
        except np.linalg.LinAlgError as error:
            raise ValueError("covariance must be positive definite") from error
        whitened = np.linalg.solve(base_factor, combined)
    else:
        raise ValueError("covariance must be a diagonal vector or square matrix")

    split_points = np.cumsum(widths[:-1])
    whitened_parts = tuple(np.split(whitened, split_points, axis=1))
    whitened_values = whitened_parts[: len(values)]
    if shared_factor is None:
        return whitened_values

    whitened_factor = whitened_parts[-1]
    left, singular_values, _ = np.linalg.svd(
        whitened_factor,
        full_matrices=False,
    )
    normalizer = np.hypot(1.0, singular_values)
    shrinkage = (singular_values / normalizer) * (
        singular_values / (1.0 + normalizer)
    )
    return tuple(
        matrix - left @ (shrinkage[:, None] * (left.T @ matrix))
        for matrix in whitened_values
    )


def _orthonormal_basis(matrix: np.ndarray, tolerance: float) -> np.ndarray:
    if matrix.shape[1] == 0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    left, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if len(singular_values) == 0 or singular_values[0] == 0.0:
        return np.zeros((matrix.shape[0], 0), dtype=float)
    rank = int(np.sum(singular_values > tolerance * singular_values[0]))
    return left[:, :rank]


def _parameter_scales(values: np.ndarray | None, parameter_count: int) -> np.ndarray:
    if values is None:
        return np.ones(parameter_count, dtype=float)
    scales = np.asarray(values, dtype=float)
    if scales.shape != (parameter_count,):
        raise ValueError("parameter_scales must match the intervention parameter count")
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter_scales must be finite and positive")
    return scales


def project_identifiable_intervention_update(
    update: np.ndarray,
    result: InterventionIdentifiabilityResult,
) -> np.ndarray:
    """Project a physical-unit update onto locally identified directions.

    Scaling occurs before projection, so the result is invariant to equivalent
    unit changes when ``parameter_scales`` are transformed consistently.
    """

    values = np.asarray(update, dtype=float)
    if values.shape[-1] != result.parameter_count or not np.all(np.isfinite(values)):
        raise ValueError("update must be finite and end in parameter_count")
    standardized = values / result.parameter_scales
    projected = standardized @ result.identified_projection
    return projected * result.parameter_scales


def assess_intervention_identifiability(
    intervention_sensitivity: np.ndarray,
    nuisance_sensitivity: np.ndarray | None = None,
    *,
    covariance: np.ndarray | None = None,
    covariance_factor: np.ndarray | None = None,
    parameter_scales: np.ndarray | None = None,
    query_sensitivity: np.ndarray | None = None,
    config: IdentifiabilityConfig | None = None,
) -> InterventionIdentifiabilityResult:
    """Assess intervention information conditional on nuisance response.

    Intervention columns are first converted to standardized parameter
    coordinates using ``parameter_scales`` and whitened by ``covariance``. A
    positive diagonal vector may replace a dense base covariance, and
    ``covariance_factor`` adds an exact positive-semidefinite ``U U.T`` term
    without materializing it. Sensitivities are then projected onto the
    orthogonal complement of nuisance response. If a future
    ``query_sensitivity`` is supplied, the result also reports how much of that
    query response lies in the unresolved intervention subspace. Thus a
    parameter vector can be only partially identified while a particular future
    prediction remains locally identifiable.
    """

    settings = config or IdentifiabilityConfig()
    raw_intervention = np.asarray(intervention_sensitivity, dtype=float)
    if (
        raw_intervention.ndim != 2
        or raw_intervention.shape[0] == 0
        or raw_intervention.shape[1] == 0
    ):
        raise ValueError("intervention_sensitivity must be a nonempty matrix")
    scales = _parameter_scales(parameter_scales, raw_intervention.shape[1])
    if nuisance_sensitivity is None:
        nuisance_raw = np.zeros((raw_intervention.shape[0], 0), dtype=float)
    else:
        nuisance_raw = np.asarray(nuisance_sensitivity, dtype=float)
        if nuisance_raw.ndim != 2 or nuisance_raw.shape[0] != len(raw_intervention):
            raise ValueError("nuisance_sensitivity must share the response dimension")
    intervention, nuisance = _whiten_sensitivities(
        (raw_intervention * scales[None], nuisance_raw),
        covariance,
        covariance_factor,
    )
    parameter_count = intervention.shape[1]

    nuisance_basis = _orthonormal_basis(nuisance, settings.relative_rank_tolerance)
    intervention_basis = _orthonormal_basis(
        intervention, settings.relative_rank_tolerance
    )
    residualized = intervention - nuisance_basis @ (nuisance_basis.T @ intervention)
    information = residualized.T @ residualized
    information = 0.5 * (information + information.T)
    eigenvalues, eigenvectors = np.linalg.eigh(information)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    largest = float(eigenvalues[-1])
    tolerance = settings.relative_rank_tolerance * max(largest, 1.0)
    identified_mask = eigenvalues > tolerance
    effective_rank = int(np.sum(identified_mask))
    identified_basis = eigenvectors[:, identified_mask]
    null_basis = eigenvectors[:, ~identified_mask]
    minimum = float(eigenvalues[0])
    positive = eigenvalues[identified_mask]
    condition_number = (
        float(np.max(positive) / np.min(positive)) if len(positive) else float("inf")
    )
    original_energy = float(np.sum(np.square(intervention)))
    residual_energy = float(np.sum(np.square(residualized)))
    residual_fraction = (
        residual_energy / original_energy if original_energy > 0.0 else 0.0
    )
    if nuisance_basis.shape[1] and intervention_basis.shape[1]:
        maximum_cosine = float(
            np.max(
                np.linalg.svd(nuisance_basis.T @ intervention_basis, compute_uv=False)
            )
        )
    else:
        maximum_cosine = 0.0

    reasons = []
    if effective_rank < parameter_count:
        reasons.append("rank_deficient_after_nuisance_projection")
    if minimum < settings.minimum_information_eigenvalue:
        reasons.append("conditional_information_below_threshold")
    if condition_number > settings.maximum_condition_number:
        reasons.append("conditional_information_ill_conditioned")
    if residual_fraction < settings.minimum_residualized_response_fraction:
        reasons.append("intervention_response_absorbed_by_nuisance")
    if maximum_cosine > settings.maximum_subspace_cosine:
        reasons.append("intervention_and_nuisance_subspaces_nearly_collinear")

    query_fraction: float | None = None
    query_identifiable: bool | None = None
    query_reasons: list[str] = []
    if query_sensitivity is not None:
        query = np.asarray(query_sensitivity, dtype=float)
        if query.ndim != 2 or query.shape[1] != parameter_count:
            raise ValueError("query_sensitivity must have shape (query, parameter)")
        if not np.all(np.isfinite(query)):
            raise ValueError("query_sensitivity must be finite")
        standardized_query = query * scales[None]
        total_query_energy = float(np.sum(np.square(standardized_query)))
        null_query = standardized_query @ null_basis
        null_query_energy = float(np.sum(np.square(null_query)))
        query_fraction = (
            null_query_energy / total_query_energy if total_query_energy > 0.0 else 0.0
        )
        query_fraction = min(max(query_fraction, 0.0), 1.0)
        query_identifiable = (
            query_fraction <= settings.maximum_query_null_response_fraction
        )
        if not query_identifiable:
            query_reasons.append(
                "query_depends_on_unidentified_intervention_directions"
            )

    return InterventionIdentifiabilityResult(
        conditional_information=information,
        eigenvalues=eigenvalues,
        effective_rank=effective_rank,
        parameter_count=parameter_count,
        minimum_eigenvalue=minimum,
        condition_number=condition_number,
        residualized_response_fraction=float(residual_fraction),
        maximum_subspace_cosine=maximum_cosine,
        parameter_scales=scales,
        identified_basis=identified_basis,
        null_basis=null_basis,
        query_null_response_fraction=query_fraction,
        query_identifiable=query_identifiable,
        identifiable=not reasons,
        failure_reasons=tuple(reasons),
        query_failure_reasons=tuple(query_reasons),
        extended_diagnostics=(
            parameter_scales is not None or query_sensitivity is not None
        ),
    )
